"""
maxinfo/scale_spatial.py

자율 스케일링 실험: multiscale projector의 공간정보 전달 이득이 데이터·스텝 규모와 함께
  (1) 커지는가  (2) seed간 분산이 줄어 재현성이 생기는가  (3) 액션 L1으로 전환되는가
를 추적한다. frozen MLP 대비 paired(같은 쌍) 비교.

모델 1회 로드 → config(데이터크기·스텝·샤드)별로 데이터 적재 → frozen 측정 1회 →
multiscale을 seed별 학습/측정. 증분 저장(중단돼도 부분결과 보존).

실행:
  python maxinfo/scale_spatial.py
출력: maxinfo/scale_spatial_result.json (config별 frozen/multiscale 통계)
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
from train_eval import (MODEL_ID, CONFIG_PATH, load_norm, build_examples, train, evaluate)
from vision_dep_spatial import (select_spatial_pairs, spatial_dependency,
                                spatial_dependency_cont, paired_stats)

OUT = os.path.join(os.path.dirname(__file__), "scale_spatial_result.json")

# (tag, train_n, steps, n_train_shards) — 점진적 스케일업
CONFIGS = [
    ("d6k_s3000",  6000,  3000, 16),
    ("d15k_s6000", 15000, 6000, 40),
]
SEEDS = [0, 1, 2, 3, 4]
N_VAL = 256
K_PAIRS = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="multiscale")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default=None, help="단일 config 직접 지정")
    ap.add_argument("--train_n", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--shards", type=int)
    args = ap.parse_args()

    global CONFIGS
    if args.smoke:
        CONFIGS = [("smoke", 300, 6, 4)]
        args.seeds = [0]
    elif args.tag:
        CONFIGS = [(args.tag, args.train_n, args.steps, args.shards)]

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

    for tag, train_n, steps, n_shards in CONFIGS:
        print(f"\n######## CONFIG {tag}: train_n={train_n} steps={steps} shards={n_shards} ########", flush=True)
        cfg = results.setdefault(tag, dict(train_n=train_n, steps=steps))

        print("데이터 로드…", flush=True)
        train_data, val_data = load_jaco_subset(train_n, N_VAL, CONFIG_PATH,
                                                n_train_shards=n_shards, n_test_shards=2)
        train_ex = build_examples(train_data, processor, vocab_size)
        val_ex = build_examples(val_data, processor, vocab_size)
        pairs = select_spatial_pairs(val_ex, val_data, vocab_size, norm, k=K_PAIRS)
        print(f"  train={len(train_ex)} val={len(val_ex)} pairs={len(pairs)}", flush=True)

        # --- frozen 기준선 (학습 없음): 이 config의 pairs/val 기준 ---
        if "frozen" not in cfg:
            model.projector = orig_base; model.requires_grad_(False); model.eval()
            f_sp, _ = spatial_dependency(model, pairs, vocab_size, norm)
            f_spc = spatial_dependency_cont(model, pairs, vocab_size, norm)
            f_m = evaluate(model, val_ex, vocab_size, norm)
            cfg["frozen"] = dict(spatial_shift=float(f_sp.mean()), per_pair=f_sp.tolist(),
                                 spatial_shift_cont=float(f_spc.mean()), per_pair_cont=f_spc.tolist(),
                                 action_l1=f_m["action_l1"], action_mse=f_m["action_mse"])
            json.dump(results, open(OUT, "w"), indent=2)
            print(f"  [frozen] sp_shift={f_sp.mean():.4f} L1={f_m['action_l1']:.4f}", flush=True)
        base_sp = np.array(cfg["frozen"]["per_pair"])
        base_spc = np.array(cfg["frozen"]["per_pair_cont"])
        base_l1 = cfg["frozen"]["action_l1"]

        # --- multiscale seed별 ---
        runs = cfg.setdefault("runs", {})
        for seed in args.seeds:
            key = f"{args.variant}_seed{seed}"
            if key in runs:
                print(f"  [skip] {key}", flush=True); continue
            print(f"  ===== {key} =====", flush=True)
            model.projector = orig_base; model.requires_grad_(False)
            torch.manual_seed(seed)
            proj, params, meta = build_projector(args.variant, model)
            model.projector = proj
            for p in params: p.requires_grad = True
            train(model, params, train_ex, steps, 2e-4, seed=seed)
            model.eval()
            sp, _ = spatial_dependency(model, pairs, vocab_size, norm)
            spc = spatial_dependency_cont(model, pairs, vocab_size, norm)
            m = evaluate(model, val_ex, vocab_size, norm)
            gamma = float(proj.gamma.detach().cpu()) if hasattr(proj, "gamma") else None
            st = paired_stats(base_sp, sp)
            stc = paired_stats(base_spc, spc)
            runs[key] = dict(spatial_shift=float(sp.mean()), per_pair=sp.tolist(),
                             spatial_shift_cont=float(spc.mean()), per_pair_cont=spc.tolist(),
                             rel_gain_cont=stc["rel_gain"], sign_test_p_cont=stc["sign_test_p"],
                             action_l1=m["action_l1"], action_mse=m["action_mse"],
                             gamma=gamma, seed=seed, **st)
            print(f"    [argmax] gain={st['rel_gain']*100:+.1f}% p={st['sign_test_p']:.3f} | "
                  f"[cont] gain={stc['rel_gain']*100:+.1f}% p={stc['sign_test_p']:.3f} | "
                  f"L1={m['action_l1']:.4f}(fr {base_l1:.4f}) g={gamma:+.4f}", flush=True)
            json.dump(results, open(OUT, "w"), indent=2)
            del proj; model.projector = orig_base; torch.cuda.empty_cache()

        # --- config 요약 ---
        present = [f"{args.variant}_seed{s}" for s in args.seeds if f"{args.variant}_seed{s}" in runs]
        gains = np.array([runs[k]["rel_gain"] for k in present]) * 100
        gainsc = np.array([runs[k]["rel_gain_cont"] for k in present]) * 100
        l1s = np.array([runs[k]["action_l1"] for k in present])
        try:
            from scipy.stats import ttest_1samp
            p = float(ttest_1samp(gains, 0.0)[1]) if len(gains) > 1 else float("nan")
            pc = float(ttest_1samp(gainsc, 0.0)[1]) if len(gainsc) > 1 else float("nan")
        except Exception:
            p = pc = float("nan")
        print(f"  >>> {tag} (n={len(present)}): "
              f"argmax {gains.mean():+.1f}±{gains.std():.1f}% p={p:.4f} | "
              f"cont {gainsc.mean():+.1f}±{gainsc.std():.1f}% p={pc:.4f} | "
              f"L1 {l1s.mean():.4f} vs frozen {base_l1:.4f}", flush=True)

    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
