"""
maxinfo/spatial_action.py

비평 정면 대응: proxy(정보전달)뿐 아니라 downstream(액션 L1)도 보되,
**공간추론이 실제로 필요한 상태(=GT xyz 이동량이 큰 타임스텝)** 에서만 측정한다.
jaco action7 = [world_vector(xyz,3), zeros(3), gripper(1)] → 공간성분 = dims 0:3.

가설: 전체 평균 L1은 자명한(정지) 상태에 희석되어 +1%지만, **공간-hard 부분집합**에서는
multiscale_attn 이 frozen 보다 xyz 액션 오차를 더 줄인다.

실행: python maxinfo/spatial_action.py --train_n 1500 --steps 800 --seeds 0 1 2 3 4
출력: maxinfo/spatial_action_result.json
"""
import os, sys, json, argparse
os.environ["BNB_CUDA_VERSION"] = "130"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from data import load_jaco_subset
from projectors_zoo import build_projector
from train_eval import (MODEL_ID, DEVICE, CONFIG_PATH, load_norm, decode_tokens,
                        build_examples, train, make_batch)

OUT = os.path.join(os.path.dirname(__file__), "spatial_action_result.json")
XYZ = slice(0, 3)


@torch.no_grad()
def per_example_xyz_l1(model, val_ex, vocab_size, norm):
    """각 val 예제의 xyz(공간) 액션 L1 배열."""
    q01, q99, mask = norm
    out = []
    model.eval()
    for ex in val_ex:
        input_ids, pix, _, atoks = make_batch(ex)
        o = model(input_ids=input_ids, pixel_values=pix)
        pred = o.logits[0, -8:-1, :].argmax(-1).cpu().numpy()
        pa = decode_tokens(pred, vocab_size, q01, q99, mask)
        ga = decode_tokens(atoks.numpy(), vocab_size, q01, q99, mask)
        out.append(np.abs(pa - ga)[XYZ].mean())
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_n", type=int, default=1500)
    ap.add_argument("--val_n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--variant", default="multiscale_attn")
    ap.add_argument("--hard_frac", type=float, default=0.33)
    args = ap.parse_args()

    res = json.load(open(OUT)) if os.path.exists(OUT) else {}

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

    train_data, val_data = load_jaco_subset(args.train_n, args.val_n, CONFIG_PATH,
                                            n_train_shards=4, n_test_shards=2)
    train_ex = build_examples(train_data, processor, vocab_size)
    val_ex = build_examples(val_data, processor, vocab_size)

    # 공간-hard 부분집합: GT xyz 이동 크기 상위 hard_frac
    gt_mag = np.array([np.abs(decode_tokens(ex[2].numpy(), vocab_size, *norm)[XYZ]).mean() for ex in val_ex])
    thr = np.quantile(gt_mag, 1 - args.hard_frac)
    hard = gt_mag >= thr
    easy = ~hard
    print(f"val={len(val_ex)} | hard(공간 큰 이동)={hard.sum()} thr={thr:.4f} | easy={easy.sum()}", flush=True)

    def subsets(arr):
        return dict(all=float(arr.mean()), hard=float(arr[hard].mean()), easy=float(arr[easy].mean()))

    if "frozen" not in res:
        model.projector = orig_base; model.requires_grad_(False)
        fl = per_example_xyz_l1(model, val_ex, vocab_size, norm)
        res["frozen"] = subsets(fl)
        json.dump(res, open(OUT, "w"), indent=2)
    fr = res["frozen"]
    print(f"[frozen] xyz-L1  all={fr['all']:.4f}  hard={fr['hard']:.4f}  easy={fr['easy']:.4f}", flush=True)

    runs = res.setdefault("runs", {})
    for seed in args.seeds:
        key = f"{args.variant}_seed{seed}"
        if key in runs:
            print(f"[skip] {key}", flush=True); continue
        print(f"===== {key} =====", flush=True)
        model.projector = orig_base; model.requires_grad_(False)
        torch.manual_seed(seed)
        proj, params, _ = build_projector(args.variant, model)
        model.projector = proj
        for p in params: p.requires_grad = True
        train(model, params, train_ex, args.steps, args.lr, seed=seed)
        al = per_example_xyz_l1(model, val_ex, vocab_size, norm)
        s = subsets(al)
        runs[key] = dict(seed=seed, **s)
        print(f"  xyz-L1  all={s['all']:.4f}  hard={s['hard']:.4f} (frozen {fr['hard']:.4f}, "
              f"Δ={(s['hard']-fr['hard'])/fr['hard']*100:+.1f}%)  easy={s['easy']:.4f}", flush=True)
        json.dump(res, open(OUT, "w"), indent=2)
        del proj; model.projector = orig_base; torch.cuda.empty_cache()

    # 요약
    from scipy.stats import ttest_1samp
    for sub in ["all", "hard", "easy"]:
        ds = np.array([(runs[k][sub] - fr[sub]) / fr[sub] * 100 for k in runs])  # %개선(음수=좋음)
        p = ttest_1samp(ds, 0)[1] if len(ds) > 1 else float("nan")
        print(f">> {sub:5s}: attn xyz-L1 Δ vs frozen = {ds.mean():+.1f}% ± {ds.std():.1f}  (p={p:.4f}, n={len(ds)})  "
              f"[음수=개선]", flush=True)


if __name__ == "__main__":
    main()
