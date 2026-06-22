"""
maxinfo/train_eval.py

jaco_play 실데이터로 projector 변종들을 동일 조건에서 학습/평가하고 비교한다.
- base openvla-7b(4bit) 1회 로드, projector만 학습(나머지 frozen)
- 평가: teacher-forced 단일 forward로 액션 토큰 accuracy + (역정규화) 액션 L1/MSE

실행:
  python maxinfo/train_eval.py --smoke
  python maxinfo/train_eval.py --train_n 1500 --val_n 256 --steps 1200
"""
import os, sys, time, json, argparse
os.environ["BNB_CUDA_VERSION"] = "130"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from data import load_jaco_subset
from projectors_zoo import build_projector

MODEL_ID = "openvla/openvla-7b"
DEVICE = "cuda:0"
CONFIG_PATH = "my_openvla_honeybee/config.json"
DS_KEY = "jaco_play"
VARIANTS = ["baseline_mlp_frozen", "baseline_mlp_trained", "mlp_scratch", "honeybee",
            "self_attn", "cross_attn", "maxinfo", "maxinfo_scratch"]

BINS = np.linspace(-1, 1, 256)
BIN_CENTERS = (BINS[:-1] + BINS[1:]) / 2.0


def load_norm():
    a = json.load(open(CONFIG_PATH))["norm_stats"][DS_KEY]["action"]
    return (np.array(a["q01"], np.float32), np.array(a["q99"], np.float32),
            np.array(a["mask"], bool))


def encode_action(norm_action, vocab_size):
    a = np.clip(norm_action, -1, 1)
    disc = np.clip(np.digitize(a, BINS), 1, 255)
    return (vocab_size - disc).astype(np.int64)           # [7] token ids


def decode_tokens(token_ids, vocab_size, q01, q99, mask):
    disc = np.clip(vocab_size - token_ids - 1, 0, 254)
    norm = BIN_CENTERS[disc]
    return np.where(mask, 0.5 * (norm + 1) * (q99 - q01) + q01, norm)


def build_examples(data, processor, vocab_size):
    """각 (image, instr, norm_action) -> (input_ids, pixel_values, gt_action_tokens)."""
    examples = []
    for img_np, instr, act in data:
        image = Image.fromarray(img_np).convert("RGB")
        instr = (instr or "").strip().rstrip(".")
        prompt = f"In: What action should the robot take to {instr}?\nOut:"
        enc = processor(prompt, image, return_tensors="pt")
        ids = enc["input_ids"][0]
        if int(ids[-1]) != 29871:                          # OpenVLA: ':' 뒤 공백 토큰
            ids = torch.cat([ids, torch.tensor([29871])])
        atoks = torch.tensor(encode_action(act, vocab_size), dtype=torch.long)
        full = torch.cat([ids, atoks])
        examples.append((full, enc["pixel_values"][0].to(torch.float16), atoks))
    return examples


def make_batch(ex):
    full, pix, atoks = ex
    input_ids = full.unsqueeze(0).to(DEVICE)
    labels = full.clone(); labels[:-7] = -100
    return input_ids, pix.unsqueeze(0).to(DEVICE), labels.unsqueeze(0).to(DEVICE), atoks


@torch.no_grad()
def evaluate(model, examples, vocab_size, norm, return_samples=False):
    q01, q99, mask = norm
    model.eval()
    tok_correct = tok_total = 0
    l1s, mses = [], []
    samples_out = []

    for idx, ex in enumerate(examples):
        input_ids, pix, labels, atoks = make_batch(ex)
        out = model(input_ids=input_ids, pixel_values=pix)
        logits = out.logits if hasattr(out, "logits") else out[0]
        pred = logits[0, -8:-1, :].argmax(-1).cpu().numpy()   # 7개 액션 토큰 예측
        gt = atoks.numpy()
        tok_correct += int((pred == gt).sum()); tok_total += 7
        pa = decode_tokens(pred, vocab_size, q01, q99, mask)
        ga = decode_tokens(gt, vocab_size, q01, q99, mask)
        l1 = np.abs(pa - ga).mean()
        mse = ((pa - ga) ** 2).mean()
        l1s.append(l1); mses.append(mse)

        # 샘플 저장 (처음 3개만)
        if return_samples and idx < 3:
            samples_out.append({
                "sample_idx": idx,
                "predicted_action": pa.tolist(),
                "ground_truth_action": ga.tolist(),
                "l1_error": float(l1),
                "token_pred": pred.tolist(),
                "token_gt": gt.tolist(),
                "token_acc": int((pred == gt).sum()) / 7
            })

    result = dict(token_acc=tok_correct / tok_total,
                  action_l1=float(np.mean(l1s)), action_mse=float(np.mean(mses)))
    if return_samples:
        result["samples"] = samples_out
    return result


def train(model, params, examples, steps, lr, seed=0):
    import bitsandbytes as bnb
    model.train()
    opt = bnb.optim.PagedAdamW8bit(params, lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    order = rng.integers(0, len(examples), size=steps)
    t0 = time.time(); losses = []
    for i, idx in enumerate(order):
        input_ids, pix, labels, _ = make_batch(examples[idx])
        out = model(input_ids=input_ids, pixel_values=pix, labels=labels)
        loss = out.loss
        if torch.isnan(loss):
            continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        losses.append(loss.item())
        if (i + 1) % max(1, steps // 6) == 0:
            print(f"    step {i+1}/{steps} loss={np.mean(losses[-50:]):.4f}", flush=True)
    return dict(final_loss=float(np.mean(losses[-50:])) if losses else float("nan"),
                train_time=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train_n", type=int, default=1500)
    ap.add_argument("--val_n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    ap.add_argument("--out", type=str, default="maxinfo/compare_real_result.json")
    args = ap.parse_args()
    if args.smoke:
        args.train_n, args.val_n, args.steps = 40, 16, 12

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
    print(f"vocab_size={vocab_size}", flush=True)

    print(f"데이터 로드: train={args.train_n} val={args.val_n}", flush=True)
    train_data, val_data = load_jaco_subset(args.train_n, args.val_n, CONFIG_PATH)
    print("examples 전처리 중…", flush=True)
    train_ex = build_examples(train_data, processor, vocab_size)
    val_ex = build_examples(val_data, processor, vocab_size)

    results = {}
    for name in args.variants:
        print(f"\n===== {name} =====", flush=True)
        model.projector = orig_base                       # build 전 원복
        model.requires_grad_(False)
        proj, params, meta = build_projector(name, model)
        model.projector = proj
        for p in params:
            p.requires_grad = True
        nparam = sum(p.numel() for p in params)
        torch.cuda.reset_peak_memory_stats()

        tr = dict(final_loss=float("nan"), train_time=0.0)
        if meta["trainable"] and args.steps > 0:
            tr = train(model, params, train_ex, args.steps, args.lr)
        metrics = evaluate(model, val_ex, vocab_size, norm, return_samples=(name == args.variants[0]))
        peak = torch.cuda.max_memory_allocated() / 1e9
        gamma = float(proj.gamma.detach().cpu()) if hasattr(proj, "gamma") else None
        results[name] = dict(**metrics, **tr, tokens=meta["tokens"],
                             trainable_params=nparam, peak_vram_gb=peak, gamma=gamma)
        print(f"  acc={metrics['token_acc']:.3f} L1={metrics['action_l1']:.4f} "
              f"MSE={metrics['action_mse']:.4f} loss={tr['final_loss']:.4f} "
              f"VRAM={peak:.2f}GB gamma={gamma}", flush=True)

        if name not in ("baseline_mlp_frozen",):
            del proj
        model.projector = orig_base
        torch.cuda.empty_cache()

    print("\n\n============ 실데이터(jaco_play) 비교표 ============")
    h = f"{'variant':<22}{'tokens':>7}{'params(M)':>11}{'tok_acc':>9}{'act_L1':>9}{'act_MSE':>9}{'VRAM':>7}"
    print(h); print("-" * len(h))
    for n, r in results.items():
        print(f"{n:<22}{r['tokens']:>7}{r['trainable_params']/1e6:>11.2f}"
              f"{r['token_acc']:>9.3f}{r['action_l1']:>9.4f}{r['action_mse']:>9.4f}{r['peak_vram_gb']:>7.2f}")
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
