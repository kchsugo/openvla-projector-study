"""
maxinfo/vision_dep_spatial.py

Exp2-공간한정 — "공간정보가 필요한 이미지"에서 variant별로 LLM에 전달되는 공간정보량 비교.

기존 vision_dep_per_variant.py 와 차이:
  1) 쌍 선별: GT 액션의 **pose 차원(xyz+회전, dim 0..5)** 거리가 큰 쌍만 고른다.
     (그리퍼 open/close 차이가 지배적인 쌍을 배제 → 순수 '공간' 변별.)
  2) 측정: 이미지를 swap했을 때 **pose 차원 예측 변화량(spatial_shift)** 만 본다.
     gripper(dim 6)는 제외. spatial_shift 큼 = 공간정보를 실제로 전달/사용.

가설 검증:
  - 기존 MLP(frozen)는 ViT 끝-2 레이어만 받음 → fine 공간 디테일 미전달.
  - multiscale 은 중간 레이어 추가 전달 → spatial_shift(frozen) < spatial_shift(multiscale) 면
    "MLP가 공간정보를 덜 전달한다 / multiscale이 더 전달한다"는 직접 증거.

실행:
  python maxinfo/vision_dep_spatial.py --train_n 1500 --steps 800 --lr 2e-4 \
      --variants baseline_mlp_frozen multiscale
출력: maxinfo/vision_dep_spatial_result.json
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
                        build_examples, train, BIN_CENTERS)
from vision_dep_per_variant import predict


@torch.no_grad()
def predict_continuous(model, full, pix, vocab_size, norm):
    """argmax 대신 action-bin softmax 기대값으로 연속 7D 액션 예측(양자화 노이즈 제거)."""
    q01, q99, mask = norm
    out = model(input_ids=full.unsqueeze(0).to(DEVICE), pixel_values=pix.unsqueeze(0).to(DEVICE))
    logits = out.logits[0, -8:-1, :].float()                  # [7, vocab]
    # action bin token ids: vocab-1 .. vocab-255  →  disc 0..254  →  BIN_CENTERS[0..254]
    bin_logits = logits[:, vocab_size - 255:vocab_size].flip(-1)   # [7,255], col j = disc j
    prob = torch.softmax(bin_logits, dim=-1).cpu().numpy()    # [7,255]
    norm_pred = prob @ BIN_CENTERS                            # [7] 기대 normalized action
    return np.where(mask, 0.5 * (norm_pred + 1) * (q99 - q01) + q01, norm_pred)

OUT = os.path.join(os.path.dirname(__file__), "vision_dep_spatial_result.json")
POSE = slice(0, 6)   # xyz(0,1,2) + rotation(3,4,5);  gripper = dim 6 (제외)


def select_spatial_pairs(val_ex, val_data, vocab_size, norm, k=16, gripper_tol=0.3):
    """pose 거리가 크고 gripper 차이는 작은 쌍을 골라 '순수 공간' 변별력 확보."""
    acts = np.array([d[2] for d in val_data])              # normalized 7D
    n = len(val_ex)
    cand = []
    for i in range(n):
        for j in range(i + 1, n):
            pose_d = float(np.abs(acts[i, POSE] - acts[j, POSE]).mean())
            grip_d = float(abs(acts[i, 6] - acts[j, 6]))
            if grip_d <= gripper_tol:                      # 그리퍼는 비슷한 쌍만
                cand.append((pose_d, i, j))
    cand.sort(reverse=True)                                # pose 차이 큰 순
    return [(val_ex[i], val_ex[j]) for _, i, j in cand[:k]]


@torch.no_grad()
def spatial_dependency(model, pairs, vocab_size, norm):
    """이미지만 A→B로 swap했을 때 pose 차원 예측 변화(spatial_shift)와 gripper 변화(대조).
    per-pair 배열을 반환 → frozen vs multiscale 같은 쌍 paired 비교 가능."""
    sp_shifts, grip_shifts = [], []
    for (fa, pa, ta), (fb, pb, tb) in pairs:
        pred_A = predict(model, fa, pa, vocab_size, norm)         # A프롬프트+A이미지
        pred_Aimg_B = predict(model, fa, pb, vocab_size, norm)    # 텍스트 A, 이미지만 B
        d = np.abs(pred_A - pred_Aimg_B)
        sp_shifts.append(float(d[POSE].mean()))                   # pose 차원만
        grip_shifts.append(float(d[6]))                           # gripper(대조)
    return np.array(sp_shifts), np.array(grip_shifts)


@torch.no_grad()
def spatial_dependency_cont(model, pairs, vocab_size, norm):
    """연속(softmax 기대값) 기반 pose-차원 spatial_shift per-pair. 양자화 노이즈 없음."""
    sp = []
    for (fa, pa, ta), (fb, pb, tb) in pairs:
        a = predict_continuous(model, fa, pa, vocab_size, norm)
        b = predict_continuous(model, fa, pb, vocab_size, norm)
        sp.append(float(np.abs(a - b)[POSE].mean()))
    return np.array(sp)


def paired_stats(base_arr, var_arr):
    """frozen(base) vs variant 같은 쌍 paired 검정. 부호검정 + (가능시) Wilcoxon."""
    diff = var_arr - base_arr
    n = len(diff)
    wins = int((diff > 0).sum())                                  # variant가 더 많이 움직인 쌍 수
    out = dict(mean_base=float(base_arr.mean()), mean_var=float(var_arr.mean()),
               mean_diff=float(diff.mean()), rel_gain=float(diff.mean() / (base_arr.mean() + 1e-9)),
               wins=wins, n=n)
    try:
        from scipy.stats import wilcoxon
        nz = diff[diff != 0]
        if len(nz) > 0:
            out["wilcoxon_p"] = float(wilcoxon(nz)[1])
    except Exception:
        pass
    # 부호검정(정규근사) p값
    from math import erf, sqrt
    k = wins
    z = (k - n / 2) / (sqrt(n) / 2 + 1e-9)
    out["sign_test_p"] = float(2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2)))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_n", type=int, default=1500)
    ap.add_argument("--val_n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+", default=["multiscale", "multiscale3"])
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

    train_data, val_data = load_jaco_subset(args.train_n, args.val_n, CONFIG_PATH)
    train_ex = build_examples(train_data, processor, vocab_size)
    val_ex = build_examples(val_data, processor, vocab_size)
    pairs = select_spatial_pairs(val_ex, val_data, vocab_size, norm, k=args.k)
    print(f"공간(pose) 변별쌍 {len(pairs)}개 선별 (gripper 유사, pose 거리 큰 순)", flush=True)

    # --- 기준선: frozen MLP (학습 없음, seed 무관) per-pair ---
    model.projector = orig_base
    model.requires_grad_(False)
    model.eval()
    base_sp, base_grip = spatial_dependency(model, pairs, vocab_size, norm)
    results["baseline_mlp_frozen"] = dict(
        spatial_shift=float(base_sp.mean()), gripper_shift=float(base_grip.mean()),
        per_pair=base_sp.tolist(), tokens=256, gamma=None)
    print(f"[frozen] spatial_shift={base_sp.mean():.4f}", flush=True)
    json.dump(results, open(OUT, "w"), indent=2)

    # --- 학습 변종: seed별로 학습→측정→frozen과 paired 비교 ---
    for name in args.variants:
        for seed in args.seeds:
            key = f"{name}_seed{seed}"
            if key in results:
                print(f"[skip] {key}", flush=True); continue
            print(f"\n===== {key} =====", flush=True)
            model.projector = orig_base
            model.requires_grad_(False)
            torch.manual_seed(seed)                           # enhance 랜덤 init도 seed별 분리
            proj, params, meta = build_projector(name, model)
            model.projector = proj
            for p in params:
                p.requires_grad = True
            train(model, params, train_ex, args.steps, args.lr, seed=seed)
            model.eval()
            sp, grip = spatial_dependency(model, pairs, vocab_size, norm)
            gamma = float(proj.gamma.detach().cpu()) if hasattr(proj, "gamma") else None
            st = paired_stats(base_sp, sp)
            results[key] = dict(spatial_shift=float(sp.mean()), gripper_shift=float(grip.mean()),
                                per_pair=sp.tolist(), gamma=gamma, seed=seed, **st)
            print(f"  spatial_shift={sp.mean():.4f}  rel_gain={st['rel_gain']*100:+.1f}%  "
                  f"wins={st['wins']}/{st['n']}  sign_p={st['sign_test_p']:.4f}  "
                  f"wilcoxon_p={st.get('wilcoxon_p', float('nan')):.4f}  gamma={gamma:+.4f}", flush=True)
            json.dump(results, open(OUT, "w"), indent=2)
            del proj
            model.projector = orig_base
            torch.cuda.empty_cache()

    # --- 요약: 변종별 seed 평균 ± 표준편차 ---
    print(f"\n저장: {OUT}")
    print(f"\nfrozen spatial_shift = {base_sp.mean():.4f}")
    print(f"\n{'variant':<14}{'sp_shift(mean±std)':>22}{'rel_gain%':>11}{'gamma(mean)':>13}{'min_sign_p':>12}")
    for name in args.variants:
        vals = [results[f"{name}_seed{s}"] for s in args.seeds if f"{name}_seed{s}" in results]
        if not vals:
            continue
        sps = np.array([v["spatial_shift"] for v in vals])
        gains = np.array([v["rel_gain"] for v in vals]) * 100
        gms = np.array([v["gamma"] for v in vals])
        minp = max(v["sign_test_p"] for v in vals)
        print(f"{name:<14}{sps.mean():>12.4f} ± {sps.std():<6.4f}{gains.mean():>11.1f}"
              f"{gms.mean():>13.4f}{minp:>12.4f}")


if __name__ == "__main__":
    main()
