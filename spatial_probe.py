"""
maxinfo/spatial_probe.py

Exp 3 (약식) — projector 출력의 공간정보 보유량 probing.

라벨 한계: jaco_play엔 물체 절대위치 라벨이 없다. 대신 약식 proxy로,
projector 출력(LLM이 받는 시각 토큰)에서 '공간적으로 분포된 토큰 표현'이
얼마나 풍부한지를 두 가지로 본다.
  (P1) token-variance: 256(또는 64) 토큰 표현의 토큰 간 분산 평균.
       공간 mixing이 일어나면 토큰들이 위치별로 차별화 → 분산↑(공간 구조 보존).
  (P2) action-readout: projector 출력(mean-pool)에서 선형 probe로 GT action(xyz 3D)을
       회귀 → R^2. 공간 의도를 더 잘 담을수록 R^2↑. (frozen base는 학습 안 함)

학습 가중치 저장이 없으므로, 학습된 변종은 vision_dep_per_variant 단계에서
이미 in-memory로 만들었던 것과 동일 구조를 재학습해 측정한다.

실행:
  python maxinfo/spatial_probe.py --variants baseline_mlp_frozen mlp_scratch mlp_scratch_ln self_attn honeybee maxinfo
출력: maxinfo/spatial_probe_result.json
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
from train_eval import MODEL_ID, DEVICE, CONFIG_PATH, load_norm, build_examples, train

OUT = os.path.join(os.path.dirname(__file__), "spatial_probe_result.json")


@torch.no_grad()
def vision_features(model, pix):
    return model.vision_backbone(pix.to(DEVICE, dtype=torch.float16))


@torch.no_grad()
def collect_proj_feats(model, proj, exs, n):
    """각 example의 projector 출력 -> (mean-pool 벡터, token-variance) 수집."""
    pooled, tvar = [], []
    for ex in exs[:n]:
        _, pix, _ = ex
        vf = vision_features(model, pix.unsqueeze(0))
        out = proj(vf).float()                      # [1, T, 4096]
        pooled.append(out.mean(1)[0].cpu().numpy())
        tvar.append(float(out[0].var(dim=0).mean().cpu()))   # 토큰 간 분산
    return np.array(pooled), float(np.mean(tvar))


def ridge_r2(X, y, lam=1.0):
    """간단 선형 probe(ridge) 5-fold R^2 평균."""
    from numpy.linalg import lstsq
    n = len(X); idx = np.arange(n); rng = np.random.default_rng(0); rng.shuffle(idx)
    folds = np.array_split(idx, 5); r2s = []
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-6)
    for f in folds:
        te = f; tr = np.setdiff1d(idx, te)
        A = Xc[tr]; b = y[tr]
        W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ b)
        pred = Xc[te] @ W
        ss_res = ((y[te] - pred) ** 2).sum(); ss_tot = ((y[te] - y[te].mean(0)) ** 2).sum() + 1e-9
        r2s.append(1 - ss_res / ss_tot)
    return float(np.mean(r2s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_n", type=int, default=1500)
    ap.add_argument("--val_n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--probe_n", type=int, default=200)
    ap.add_argument("--variants", nargs="+",
                    default=["baseline_mlp_frozen", "mlp_scratch", "mlp_scratch_ln",
                             "self_attn", "honeybee", "maxinfo"])
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

    train_data, val_data = load_jaco_subset(args.train_n, args.val_n, CONFIG_PATH)
    train_ex = build_examples(train_data, processor, vocab_size)
    val_ex = build_examples(val_data, processor, vocab_size)
    y = np.array([d[2][:3] for d in val_data[:args.probe_n]], dtype=np.float32)  # GT xyz

    for name in args.variants:
        if name in results:
            print(f"[skip] {name}", flush=True); continue
        print(f"\n===== {name} =====", flush=True)
        model.projector = orig_base
        model.requires_grad_(False)
        proj, params, meta = build_projector(name, model)
        model.projector = proj
        for p in params:
            p.requires_grad = True
        if meta["trainable"] and args.steps > 0:
            train(model, params, train_ex, args.steps, args.lr)
        proj.eval()
        X, tvar = collect_proj_feats(model, proj, val_ex, args.probe_n)
        r2 = ridge_r2(X, y)
        results[name] = dict(token_variance=tvar, action_readout_r2=r2, tokens=meta["tokens"])
        print(f"  token_variance={tvar:.4f} (공간 차별화↑)  action_readout_R2={r2:.4f} (공간의도 보유↑)",
              flush=True)
        json.dump(results, open(OUT, "w"), indent=2)
        if name != "baseline_mlp_frozen":
            del proj
        model.projector = orig_base
        torch.cuda.empty_cache()

    print(f"\n저장: {OUT}")
    print(f"\n{'variant':<22}{'token_var':>12}{'action_R2':>12}")
    for k, v in results.items():
        print(f"{k:<22}{v['token_variance']:>12.4f}{v['action_readout_r2']:>12.4f}")


if __name__ == "__main__":
    main()
