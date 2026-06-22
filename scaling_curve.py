"""
maxinfo/scaling_curve.py

데이터 규모(n_train)를 단계적으로 늘리며 각 projector 변종의 Action L1을 측정해
"데이터가 많아질수록 honeybee/attn이 사전학습 MLP를 따라잡는가"를 검증한다.

- 모델 1회 로드, (데이터크기 × 변종) 격자를 순회
- projector-only 학습, LLM frozen, lr 고정
- 결과: maxinfo/scaling_result.json  (이어달리기: 이미 끝난 셀은 건너뜀)

실행:
  python maxinfo/scaling_curve.py --sizes 500 3000 10000 30000 \
      --variants baseline_mlp_frozen mlp_scratch self_attn honeybee \
      --steps_per_epoch 2 --lr 2e-5 --val_n 256
"""
import os, sys, json, time, argparse
os.environ["BNB_CUDA_VERSION"] = "130"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from data import load_jaco_subset
from projectors_zoo import build_projector
from train_eval import (MODEL_ID, CONFIG_PATH, load_norm, encode_action,
                        evaluate, train)
from PIL import Image
import torch as _torch

OUT = os.path.join(os.path.dirname(__file__), "scaling_result.json")


class LazyExamples:
    """raw (img_uint8, instr, action)만 RAM에 들고, 인덱싱 시점에 전처리.

    train/evaluate/make_batch가 기대하는 (full, pixel_values, atoks) 튜플을
    __getitem__에서 즉석 생성 → 3만 개 pixel_values를 한꺼번에 안 올려 RAM OOM 방지.
    """
    def __init__(self, data, processor, vocab_size):
        self.data = data
        self.processor = processor
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):                       # prefix view 반환 (RAM 복제 없음)
            return LazyExamples(self.data[idx], self.processor, self.vocab_size)
        img_np, instr, act = self.data[int(idx)]
        image = Image.fromarray(img_np).convert("RGB")
        instr = (instr or "").strip().rstrip(".")
        prompt = f"In: What action should the robot take to {instr}?\nOut:"
        enc = self.processor(prompt, image, return_tensors="pt")
        ids = enc["input_ids"][0]
        if int(ids[-1]) != 29871:
            ids = _torch.cat([ids, _torch.tensor([29871])])
        atoks = _torch.tensor(encode_action(act, self.vocab_size), dtype=_torch.long)
        full = _torch.cat([ids, atoks])
        return (full, enc["pixel_values"][0].to(_torch.float16), atoks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[500, 3000, 10000, 30000])
    ap.add_argument("--variants", nargs="+",
                    default=["baseline_mlp_frozen", "mlp_scratch", "self_attn", "honeybee"])
    ap.add_argument("--steps_per_epoch", type=float, default=2.0,
                    help="step = round(n_train * 이 값). 데이터에 비례해 학습량도 늘림")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--val_n", type=int, default=256)
    ap.add_argument("--n_train_shards", type=int, default=128)
    args = ap.parse_args()

    results = json.load(open(OUT)) if os.path.exists(OUT) else {}

    print("로드: openvla-7b (4bit)…", flush=True)
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                            llm_int8_skip_modules=["projector"])
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID, quantization_config=qc, device_map={"": 0},
        torch_dtype=torch.float16, attn_implementation="sdpa", trust_remote_code=True)
    model.gradient_checkpointing_enable()
    orig_base = model.projector
    vocab_size = model.vocab_size
    norm = load_norm()

    # 최대 크기 데이터 1회 로드 후, 각 size는 prefix로 사용 (동일 부분집합 보장)
    max_n = max(args.sizes)
    print(f"데이터 로드: train≤{max_n} val={args.val_n}", flush=True)
    train_data, val_data = load_jaco_subset(max_n, args.val_n, CONFIG_PATH,
                                            n_train_shards=args.n_train_shards)
    avail = len(train_data)
    print(f"가용 train={avail} (요청 {max_n})", flush=True)
    print("lazy examples 준비(raw 이미지만 RAM 적재)…", flush=True)
    full_train_ex = LazyExamples(train_data, processor, vocab_size)
    val_ex = LazyExamples(val_data, processor, vocab_size)

    for size in args.sizes:
        n = min(size, avail)
        steps = max(1, int(round(n * args.steps_per_epoch)))
        sub = full_train_ex[:n]
        for name in args.variants:
            key = f"{name}@{size}"
            if key in results:
                print(f"[skip] {key} (이미 완료)", flush=True)
                continue
            print(f"\n===== {key}  (n={n}, steps={steps}) =====", flush=True)
            model.projector = orig_base
            model.requires_grad_(False)
            proj, params, meta = build_projector(name, model)
            model.projector = proj
            for p in params:
                p.requires_grad = True
            nparam = sum(p.numel() for p in params)
            torch.cuda.reset_peak_memory_stats()

            tr = dict(final_loss=float("nan"), train_time=0.0)
            if meta["trainable"] and steps > 0:
                tr = train(model, params, sub, steps, args.lr)
            m = evaluate(model, val_ex, vocab_size, norm)
            peak = torch.cuda.max_memory_allocated() / 1e9
            gamma = float(proj.gamma.detach().cpu()) if hasattr(proj, "gamma") else None
            results[key] = dict(variant=name, n_train=n, steps=steps, lr=args.lr,
                                token_acc=m["token_acc"], action_l1=m["action_l1"],
                                action_mse=m["action_mse"], **tr,
                                tokens=meta["tokens"], trainable_params=nparam,
                                peak_vram_gb=peak, gamma=gamma)
            print(f"  n={n} acc={m['token_acc']:.3f} L1={m['action_l1']:.4f} "
                  f"loss={tr['final_loss']:.3f} VRAM={peak:.1f}GB", flush=True)
            json.dump(results, open(OUT, "w"), indent=2)   # 셀 단위 저장(이어달리기)

            if name != "baseline_mlp_frozen":
                del proj
            model.projector = orig_base
            torch.cuda.empty_cache()

    print(f"\n저장: {OUT}", flush=True)
    # 요약표
    print("\n===== Scaling 요약 (Action L1) =====")
    print(f"{'variant':<22}" + "".join(f"{s:>9}" for s in args.sizes))
    for name in args.variants:
        row = f"{name:<22}"
        for s in args.sizes:
            r = results.get(f"{name}@{s}")
            row += f"{r['action_l1']:>9.4f}" if r else f"{'-':>9}"
        print(row)


if __name__ == "__main__":
    main()
