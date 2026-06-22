"""
maxinfo/llm_attn_ablation.py

핵심 가설 직접검증: "LLM의 self-attention이 projector의 공간 mixing 역할을 이미 대신한다."

방법: 수동 forward로 LLM에 4D attention mask를 주입해 **비전토큰끼리(patch i ↔ patch j)**
attention을 차단한다(BOS·causal·text는 유지). 라이브러리 미수정.
시퀀스 = [BOS(1) | vision(256) | text(T)]  → 비전 위치 = [1, 257).

판정:
  (1) frozen에서 비전-비전 차단 시 액션 L1이 크게 악화 → LLM이 실제로 공간 mixing 중이었다(가설 지지).
  (2) 그 악화를 projector mixing(self_attn/multiscale_attn)이 복구/완화 → projector mixing이 LLM mixing을 대체.

실행: python maxinfo/llm_attn_ablation.py --train_n 1500 --steps 800
출력: maxinfo/llm_attn_ablation_result.json
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

OUT = os.path.join(os.path.dirname(__file__), "llm_attn_ablation_result.json")
XYZ = slice(0, 3)
N_VIS = 256


@torch.no_grad()
def manual_logits(model, input_ids, pixel_values, block_vis_vis=False):
    """PrismaticForConditionalGeneration.forward를 수동 재현하되,
    block_vis_vis=True면 비전-비전 attention을 차단한 4D mask를 LLM에 준다."""
    lm = model.language_model
    patch = model.vision_backbone(pixel_values)
    proj = model.projector(patch)                                   # [B,256,d]
    emb = model.get_input_embeddings()(input_ids)                   # [B,T0,d]
    mm = torch.cat([emb[:, :1], proj, emb[:, 1:]], dim=1)           # [B, L, d]
    B, L, _ = mm.shape
    dtype = mm.dtype
    neg = torch.finfo(dtype).min
    # causal additive mask
    m = torch.full((L, L), neg, dtype=dtype, device=mm.device)
    m = torch.triu(m, diagonal=1)                                  # 0 on/below diag, neg above
    if block_vis_vis:
        v0, v1 = 1, 1 + N_VIS                                       # vision positions [1,257)
        block = torch.zeros((L, L), dtype=torch.bool, device=mm.device)
        block[v0:v1, v0:v1] = True
        idx = torch.arange(v0, v1, device=mm.device)
        block[idx, idx] = False                                     # keep self
        m = m.masked_fill(block, neg)
    m = m[None, None]                                              # [1,1,L,L]
    out = lm(inputs_embeds=mm, attention_mask=m, use_cache=False)
    return out.logits


@torch.no_grad()
def xyz_l1(model, val_ex, vocab_size, norm, block_vis_vis):
    q01, q99, mask = norm
    model.eval()
    errs = []
    for ex in val_ex:
        input_ids, pix, _, atoks = make_batch(ex)
        logits = manual_logits(model, input_ids, pix, block_vis_vis)
        pred = logits[0, -8:-1, :].argmax(-1).cpu().numpy()
        pa = decode_tokens(pred, vocab_size, q01, q99, mask)
        ga = decode_tokens(atoks.numpy(), vocab_size, q01, q99, mask)
        errs.append(np.abs(pa - ga)[XYZ].mean())
    return np.array(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_n", type=int, default=1500)
    ap.add_argument("--val_n", type=int, default=128)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--variants", nargs="+", default=["baseline_mlp_frozen", "self_attn"])
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

    train_data, val_data = load_jaco_subset(args.train_n, args.val_n, CONFIG_PATH, n_train_shards=4, n_test_shards=2)
    train_ex = build_examples(train_data, processor, vocab_size)
    val_ex = build_examples(val_data, processor, vocab_size)
    gt_mag = np.array([np.abs(decode_tokens(ex[2].numpy(), vocab_size, *norm)[XYZ]).mean() for ex in val_ex])
    hard = gt_mag >= np.quantile(gt_mag, 0.67)

    # --- self-test: 마스크 없는 수동 forward ≈ 표준 forward ---
    model.projector = orig_base; model.requires_grad_(False); model.eval()
    ex = val_ex[0]; input_ids, pix, _, _ = make_batch(ex)
    std = model(input_ids=input_ids, pixel_values=pix).logits[0, -8:-1].argmax(-1)
    man = manual_logits(model, input_ids, pix, block_vis_vis=False)[0, -8:-1].argmax(-1)
    agree = (std == man).float().mean().item()
    print(f"[self-test] manual vs standard forward token agreement = {agree:.2f} (1.0이면 수동 forward 정확)", flush=True)

    def report(name, l1n, l1b):
        d = dict(normal_all=float(l1n.mean()), block_all=float(l1b.mean()),
                 normal_hard=float(l1n[hard].mean()), block_hard=float(l1b[hard].mean()))
        d["degrade_all_%"] = (d["block_all"] - d["normal_all"]) / d["normal_all"] * 100
        d["degrade_hard_%"] = (d["block_hard"] - d["normal_hard"]) / d["normal_hard"] * 100
        res[name] = d
        json.dump(res, open(OUT, "w"), indent=2)
        print(f"  {name}: xyz-L1 normal={d['normal_all']:.4f} → block={d['block_all']:.4f} "
              f"(all {d['degrade_all_%']:+.1f}%, hard {d['degrade_hard_%']:+.1f}%)", flush=True)

    for name in args.variants:
        if name in res:
            print(f"[skip] {name}", flush=True); continue
        print(f"===== {name} =====", flush=True)
        model.projector = orig_base; model.requires_grad_(False)
        proj, params, meta = build_projector(name, model)
        model.projector = proj
        if meta["trainable"]:
            for p in params: p.requires_grad = True
            train(model, params, train_ex, args.steps, args.lr, seed=0)
        l1n = xyz_l1(model, val_ex, vocab_size, norm, block_vis_vis=False)
        l1b = xyz_l1(model, val_ex, vocab_size, norm, block_vis_vis=True)
        report(name, l1n, l1b)
        if name != "baseline_mlp_frozen": del proj
        model.projector = orig_base; torch.cuda.empty_cache()

    print("\n=== 해석 ===")
    print("frozen이 block 시 크게 악화 → LLM이 공간 mixing을 실제로 수행 중(가설 지지).")
    print("projector mixing 변종의 degrade가 frozen보다 작으면 → projector가 LLM mixing을 대체.")


if __name__ == "__main__":
    main()
