"""
maxinfo/attn_map_analysis.py

핵심 가설의 관찰적 증거: "LLM 자기어텐션이 256개 비전토큰의 공간 통합을 실제로 수행하는가?"
eager attention으로 attention map을 뽑아, 비전토큰 query가 **다른 비전토큰**(자기 제외)에
두는 attention 질량을 층별로 측정한다. 질량이 크면 LLM이 패치 간 공간 통합을 하고 있는 것.

시퀀스 = [BOS(1) | vision(256) | text(T)]. 비전 위치 = [1,257).
지표:
  vis->vis  : 비전 query가 다른 비전 key에 두는 평균 attention 비중(자기 제외)
  vis->bos  : 비전 query가 BOS에 두는 비중
  vis->self : 자기 자신
  txt->vis  : (액션 직전) 텍스트 query가 비전 전체에 두는 비중
출력: maxinfo/attn_map_result.json

실행: python maxinfo/attn_map_analysis.py --n 16
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
from train_eval import MODEL_ID, CONFIG_PATH, load_norm, build_examples, make_batch

OUT = os.path.join(os.path.dirname(__file__), "attn_map_result.json")
N_VIS = 256


@torch.no_grad()
def analyze_example(model, input_ids, pix):
    """수동 merge 후 eager LLM forward로 attentions 추출 → 층별 비전토큰 attention 분해."""
    lm = model.language_model
    patch = model.vision_backbone(pix)
    proj = model.projector(patch)
    emb = model.get_input_embeddings()(input_ids)
    mm = torch.cat([emb[:, :1], proj, emb[:, 1:]], dim=1)          # [1,L,d]
    out = lm(inputs_embeds=mm, use_cache=False, output_attentions=True)
    v0, v1 = 1, 1 + N_VIS
    L = mm.shape[1]
    per_layer = []
    for att in out.attentions:                                     # att: [1,heads,L,L]
        a = att[0].float().mean(0)                                 # heads 평균 → [L,L]
        vis_q = a[v0:v1]                                           # [256, L] 비전 query 행
        self_mass = vis_q[torch.arange(N_VIS), torch.arange(v0, v1)].mean().item()
        bos_mass = vis_q[:, 0].mean().item()
        vis_block = vis_q[:, v0:v1]                                # [256,256]
        vv_total = vis_block.sum(-1).mean().item()
        vv_excl_self = vv_total - self_mass                        # 다른 비전토큰에 둔 질량
        # 액션 직전 텍스트 query(마지막 행)가 비전에 두는 질량
        txt_to_vis = a[-1, v0:v1].sum().item()
        per_layer.append(dict(vv_excl_self=vv_excl_self, vv_self=self_mass,
                              vis_to_bos=bos_mass, txt_to_vis=txt_to_vis))
    return per_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()

    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                            llm_int8_skip_modules=["projector"])
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    # attention map을 받으려면 eager 필요
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID, quantization_config=qc, device_map={"": 0},
        torch_dtype=torch.float16, attn_implementation="eager", trust_remote_code=True)
    model.eval()
    vocab_size = model.vocab_size

    _, val_data = load_jaco_subset(10, args.n, CONFIG_PATH, n_train_shards=1, n_test_shards=2)
    val_ex = build_examples(val_data, processor, vocab_size)

    acc = None
    for k, ex in enumerate(val_ex[:args.n]):
        input_ids, pix, _, _ = make_batch(ex)
        pl = analyze_example(model, input_ids, pix)
        if acc is None:
            acc = [{kk: [] for kk in pl[0]} for _ in pl]
        for li, d in enumerate(pl):
            for kk, vv in d.items():
                acc[li][kk].append(vv)
        if (k + 1) % 4 == 0:
            print(f"  {k+1}/{args.n}", flush=True)

    layers = [{kk: float(np.mean(vv)) for kk, vv in d.items()} for d in acc]
    nL = len(layers)
    summ = {}
    for kk in layers[0]:
        vals = np.array([layers[li][kk] for li in range(nL)])
        summ[kk] = dict(mean=float(vals.mean()), max=float(vals.max()),
                        early=float(vals[:nL//3].mean()), late=float(vals[2*nL//3:].mean()))
    res = dict(n_examples=args.n, n_layers=nL, per_layer=layers, summary=summ)
    json.dump(res, open(OUT, "w"), indent=2)

    print(f"\n=== 비전토큰 attention 분해 (층 {nL}개, n={args.n} 평균) ===")
    print(f"vis->vis(자기 제외, 다른 패치로): mean={summ['vv_excl_self']['mean']:.3f}  "
          f"max={summ['vv_excl_self']['max']:.3f}  (early {summ['vv_excl_self']['early']:.3f} → late {summ['vv_excl_self']['late']:.3f})")
    print(f"vis->self                       : mean={summ['vv_self']['mean']:.3f}")
    print(f"vis->bos                        : mean={summ['vis_to_bos']['mean']:.3f}")
    print(f"txt->vis(액션 직전 텍스트→비전)  : mean={summ['txt_to_vis']['mean']:.3f}")
    print("\n해석: vis->vis(자기 제외)가 크면 = LLM이 패치 간 공간정보를 실제로 통합 중 → 핵심 가설 지지.")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
