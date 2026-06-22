"""
maxinfo/report.py

모든 실험 결과를 cat 가능한 텍스트로 한 번에 출력 + maxinfo/REPORT.txt 저장.
  - 메인 8종 (compare_real_result.json, lr 2e-4, 3000 step)
  - lr 진단 (compare_lowlr_diag.json)
  - scaling curve (scaling_result.json)
  - 비전전달 (vision_transfer_result.json)

실행: python maxinfo/report.py
"""
import os, json

HERE = os.path.dirname(__file__)
def load(name):
    p = os.path.join(HERE, name)
    return json.load(open(p)) if os.path.exists(p) else None


def scaling_trend(sc, nm, sizes):
    """변종의 데이터별 L1 추세를 한 줄 요약."""
    vals = [(s, sc[f"{nm}@{s}"]["action_l1"]) for s in sizes if f"{nm}@{s}" in sc]
    if len(vals) < 2:
        return ""
    first, last = vals[0][1], vals[-1][1]
    if nm == "baseline_mlp_frozen":
        return "고정 기준선"
    d = last - first
    arrow = "개선(↓)" if d < -0.003 else ("악화(↑)" if d > 0.003 else "평탄")
    return f"{first:.4f}→{last:.4f} {arrow}"

lines = []
def P(s=""):
    lines.append(s)

P("=" * 78)
P("OpenVLA Projector 실험 — 전체 결과 리포트")
P("데이터: jaco_play(Open-X) · val 256 · 4bit · RTX 5060 8GB")
P("=" * 78)

# 1) 메인 8종
main = load("compare_real_result.json")
if main:
    P("\n[1] 메인 8종 비교 (lr 2e-4, 3000 step)")
    P("-" * 78)
    P(f"{'variant':<22}{'tokens':>7}{'params(M)':>11}{'tok_acc':>9}{'L1':>9}{'MSE':>9}{'gamma':>7}")
    for k, v in main.items():
        g = "" if v.get("gamma") is None else f"{v['gamma']:.2f}"
        P(f"{k:<22}{v['tokens']:>7}{v['trainable_params']/1e6:>11.1f}"
          f"{v['token_acc']:>9.3f}{v['action_l1']:>9.4f}{v['action_mse']:>9.4f}{g:>7}")

# 2) lr 진단
diag = load("compare_lowlr_diag.json")
if diag:
    P("\n[2] lr 진단: scratch 변종 lr 낮춤 (붕괴가 lr 탓인지 검증)")
    P("-" * 78)
    P(f"{'variant':<22}{'lr':>9}{'L1':>9}{'tok_acc':>9}   (메인 lr2e-4 → 진단 lr2e-5)")
    for k, v in diag.items():
        main_l1 = main[k]["action_l1"] if main and k in main else float("nan")
        P(f"{k:<22}{v['lr']:>9.0e}{v['action_l1']:>9.4f}{v['token_acc']:>9.3f}"
          f"   (메인 {main_l1:.4f} → {v['action_l1']:.4f})")

# 3) scaling curve
sc = load("scaling_result.json")
if sc:
    P("\n[3] Scaling Curve — 데이터 크기별 Action L1 (lr 2e-5)")
    P("-" * 78)
    sizes = sorted({v["n_train"] for v in sc.values()})
    req_sizes = sorted({int(k.split("@")[1]) for k in sc})
    variants = []
    for k in sc:
        nm = k.split("@")[0]
        if nm not in variants:
            variants.append(nm)
    P(f"{'variant':<22}" + "".join(f"{s:>9}" for s in req_sizes))
    for nm in variants:
        row = f"{nm:<22}"
        for s in req_sizes:
            k = f"{nm}@{s}"
            row += f"{sc[k]['action_l1']:>9.4f}" if k in sc else f"{'-':>9}"
        row += "  ← " + scaling_trend(sc, nm, req_sizes)
        P(row)
    P("\n  (frozen=고정 기준선. 다른 변종이 데이터↑ 따라 frozen에 수렴/추월하는지 관찰)")

# 4) 비전전달
vt = load("vision_transfer_result.json")
if vt:
    P("\n[4] 비전정보 전달 검증 (원본 projector)")
    P("-" * 78)
    P(f"  실제 이미지 L1 = {vt['mean_l1_real']:.4f}")
    P(f"  이미지 가림 L1 = {vt['mean_l1_blank']:.4f}  (가리면 악화 = 비전정보 실제 전달 증거)")
    P(f"  악화폭 = {vt['degradation_when_blanked']:+.4f}")

P("\n" + "=" * 78)
out = "\n".join(lines)
print(out)
open(os.path.join(HERE, "REPORT.txt"), "w").write(out)
print(f"\n저장: maxinfo/REPORT.txt")
