"""
maxinfo/projectors_zoo.py

공정 비교용 projector 5(+1)종 빌더. 모두 frozen base 모델 위에 projector만 학습한다.

주의(공정성/해석):
- baseline_mlp_frozen / baseline_mlp_trained / maxinfo 는 **사전학습 MLP 정렬을 계승**한다
  (maxinfo는 residual base로, baseline_*는 MLP 가중치 자체로).
- honeybee / self_attn / cross_attn 은 **랜덤 초기화 + 토큰 구조 변경**이라 사전학습 정렬을
  계승하지 못한다. 따라서 projector-only·frozen-LLM 조건에선 불리하다. 이 차이 자체가
  "정보 보존(=정렬 계승) vs 교체"의 핵심 비교 포인트다.
"""
import copy
import torch
import torch.nn as nn

from projector import MaxInfoProjector  # same folder
from multiscale import build_multiscale_projector, build_multiscale_attn_projector  # same folder


# ---------- 기존 3종 (저장 가중치와 동일 구조, 단 여기선 랜덤 초기화 후 재학습) ----------
class HoneybeeProjector(nn.Module):
    def __init__(self, vision_dim, llm_dim, bottleneck_dim=1024):
        super().__init__()
        self.conv_compress = nn.Conv2d(vision_dim, bottleneck_dim, 3, 2, 1)
        self.ln = nn.LayerNorm(bottleneck_dim)
        self.act = nn.GELU()
        self.bottleneck_to_llm = nn.Linear(bottleneck_dim, llm_dim)

    def forward(self, vf):
        od = vf.dtype
        B, N, C = vf.shape
        x = vf.to(torch.float32).transpose(1, 2).view(B, C, int(N**0.5), int(N**0.5))
        x = self.conv_compress(x).flatten(2).transpose(1, 2)
        x = self.act(self.ln(x))
        return self.bottleneck_to_llm(x).to(od)


class SelfAttnProjector(nn.Module):
    def __init__(self, vision_dim, llm_dim, bottleneck_dim=1024, num_heads=8):
        super().__init__()
        self.vision_to_bottleneck = nn.Linear(vision_dim, bottleneck_dim)
        self.self_attn = nn.MultiheadAttention(bottleneck_dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(bottleneck_dim)
        self.ffn = nn.Sequential(nn.Linear(bottleneck_dim, bottleneck_dim*4), nn.GELU(),
                                 nn.Linear(bottleneck_dim*4, bottleneck_dim))
        self.post_ln = nn.LayerNorm(bottleneck_dim)
        self.bottleneck_to_llm = nn.Linear(bottleneck_dim, llm_dim)

    def forward(self, vf):
        od = vf.dtype
        x = self.vision_to_bottleneck(vf.to(torch.float32))
        a, _ = self.self_attn(x, x, x)
        x = self.ln(x + a)
        x = self.post_ln(x + self.ffn(x))
        return self.bottleneck_to_llm(x).to(od)


class MLPTrainable(nn.Module):
    """사전학습 MLP 가중치를 계승해 추가학습. fp32 연산 후 입력 dtype으로 복귀."""
    def __init__(self, base):
        super().__init__()
        self.proj = copy.deepcopy(base).float()

    def forward(self, vf):
        od = vf.dtype
        return self.proj(vf.to(torch.float32)).to(od)


class MLPScratch(nn.Module):
    """원본 MLP와 **구조는 완전히 동일**하되 가중치만 랜덤(사전학습 미계승).

    대조군 핵심: baseline_mlp_trained(같은 구조, 계승O)와 비교하면
    '구조'를 고정한 채 '사전학습 정렬 계승'의 순수 효과만 분리해 측정할 수 있다.
    """
    def __init__(self, base):
        super().__init__()
        self.proj = copy.deepcopy(base).float()      # 구조 복제
        for m in self.proj.modules():                # 가중치는 전부 랜덤 재초기화
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def forward(self, vf):
        od = vf.dtype
        return self.proj(vf.to(torch.float32)).to(od)


class MLPScratchLN(nn.Module):
    """원본 MLP 구조 + 각 Linear 뒤 LayerNorm. 랜덤 초기화(계승X), token-wise(공간 mixing 없음).

    교란 분리 대조군: self_attn(LN O, mixing O)과 비교하면
      - mlp_scratch_ln ≈ self_attn  → 이득은 'LayerNorm(안정화)' 덕
      - self_attn > mlp_scratch_ln  → 이득은 'token mixing(공간 attention)' 덕
    게다가 mlp_scratch_ln(71.4M) > self_attn(19M)이라, self_attn이 이기면 capacity 교란도 배제.
    """
    def __init__(self, base):
        super().__init__()
        vd = base.fc1.in_features      # 2176
        hid = base.fc1.out_features    # 8704
        ld = base.fc3.out_features     # 4096
        self.net = nn.Sequential(
            nn.Linear(vd, hid), nn.LayerNorm(hid), nn.GELU(),
            nn.Linear(hid, ld), nn.LayerNorm(ld), nn.GELU(),
            nn.Linear(ld, ld),
        ).float()

    def forward(self, vf):
        od = vf.dtype
        return self.net(vf.to(torch.float32)).to(od)


class CrossAttnProjector(nn.Module):
    def __init__(self, vision_dim, llm_dim, bottleneck_dim=1024, num_queries=64, num_heads=8):
        super().__init__()
        self.vision_to_bottleneck = nn.Linear(vision_dim, bottleneck_dim)
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, bottleneck_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(bottleneck_dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(bottleneck_dim)
        self.ffn = nn.Sequential(nn.Linear(bottleneck_dim, bottleneck_dim*4), nn.GELU(),
                                 nn.Linear(bottleneck_dim*4, bottleneck_dim))
        self.post_ln = nn.LayerNorm(bottleneck_dim)
        self.bottleneck_to_llm = nn.Linear(bottleneck_dim, llm_dim)

    def forward(self, vf):
        od = vf.dtype
        x = vf.to(torch.float32)
        kv = self.vision_to_bottleneck(x)
        q = self.query_tokens.expand(x.size(0), -1, -1).to(torch.float32)
        a, _ = self.cross_attn(q, kv, kv)
        x = self.ln(q + a)
        x = self.post_ln(x + self.ffn(x))
        return self.bottleneck_to_llm(x).to(od)


def build_projector(name, model, device="cuda:0"):
    """이름으로 projector를 만들어 (모듈, 학습대상 파라미터 리스트, 메타) 반환."""
    base = model.projector
    vision_dim = base.fc1.in_features
    llm_dim = base.fc3.out_features

    if name == "baseline_mlp_frozen":
        proj = base  # 사전학습 MLP 그대로, 학습 안 함
        proj.to(device)
        return proj, [], dict(trainable=False, tokens=256)

    if name == "baseline_mlp_trained":
        proj = MLPTrainable(base).to(device)                  # 사전학습 가중치 계승 후 추가학습(fp32)
        return proj, list(proj.parameters()), dict(trainable=True, tokens=256)

    if name == "mlp_scratch":
        proj = MLPScratch(base).to(device)                    # 구조=원본 MLP, 가중치=랜덤(계승X)
        return proj, list(proj.parameters()), dict(trainable=True, tokens=256)

    if name == "mlp_scratch_ln":
        proj = MLPScratchLN(base).to(device)                  # 원본 MLP + LayerNorm, 랜덤, mixing 없음
        return proj, list(proj.parameters()), dict(trainable=True, tokens=256)

    if name == "honeybee":
        proj = HoneybeeProjector(vision_dim, llm_dim).to(device, dtype=torch.float32)
        return proj, list(proj.parameters()), dict(trainable=True, tokens=64)

    if name == "self_attn":
        proj = SelfAttnProjector(vision_dim, llm_dim).to(device, dtype=torch.float32)
        return proj, list(proj.parameters()), dict(trainable=True, tokens=256)

    if name == "cross_attn":
        proj = CrossAttnProjector(vision_dim, llm_dim).to(device, dtype=torch.float32)
        return proj, list(proj.parameters()), dict(trainable=True, tokens=64)

    if name == "maxinfo_scratch":
        # maxinfo 구조 그대로지만 base를 랜덤 재초기화(계승X)하고 base까지 학습.
        # 대조: maxinfo(계승O,base freeze)와 비교 → "구조 자체의 힘 vs 사전학습 base 덕분"을 분리.
        base_rand = MLPScratch(base)                          # 구조=원본 MLP, 가중치=랜덤, fp32
        proj = MaxInfoProjector(base_rand, vision_dim, llm_dim, inner_dim=2048, depth=2).to(device)
        for _, p in proj.named_parameters():                 # base 포함 전부 fp32·학습
            p.data = p.data.float()
        return proj, list(proj.parameters()), dict(trainable=True, tokens=256)

    if name == "maxinfo":
        proj = MaxInfoProjector(base, vision_dim, llm_dim, inner_dim=2048, depth=2).to(device)
        # base는 fp16 유지(사전학습), enhancement는 fp32
        proj.freeze_base()
        for n, p in proj.named_parameters():
            if not n.startswith("base."):
                p.data = p.data.float()
        return proj, proj.trainable_parameters(), dict(trainable=True, tokens=256)

    if name == "maxinfo_fixed":
        # maxinfo와 동일하되 proj_out zero-init 제거 → 데드락 해소(gamma가 실제로 학습됨).
        # 기존 "maxinfo γ→0 무용" 결론이 데드락 아티팩트였는지 검증용.
        proj = MaxInfoProjector(base, vision_dim, llm_dim, inner_dim=2048, depth=2,
                                zero_init_out=False).to(device)
        proj.freeze_base()
        for n, p in proj.named_parameters():
            if not n.startswith("base."):
                p.data = p.data.float()
        return proj, proj.trainable_parameters(), dict(trainable=True, tokens=256)

    if name == "multiscale":
        # frozen base MLP + zero-init gate * token-wise Enhance(멀티스케일 ViT feature).
        # 끝-2 외 중간 레이어를 추가로 LLM에 전달 → 기존 파이프라인에 없던 fine 공간정보 주입.
        # gamma가 0에서 벗어나면 "멀티스케일 정보가 유용"하다는 직접 증거.
        proj = build_multiscale_projector(model, fracs=(0.5,), inner_dim=2048).to(device)
        return proj, proj.trainable_parameters(), dict(trainable=True, tokens=256)

    if name == "multiscale3":
        # 멀티스케일 3개 레이어(0.25/0.5/0.75 + 끝-2 포함) → 더 많은 fine 공간정보 주입.
        proj = build_multiscale_projector(model, fracs=(0.25, 0.5, 0.75), inner_dim=2048).to(device)
        return proj, proj.trainable_parameters(), dict(trainable=True, tokens=256)

    if name == "multiscale_attn":
        # 스케일 간 cross-attention 융합: 패치 mixing 없이 각 패치에서 k개 ViT 층을 융합.
        # LLM이 못 보는 '층 선택'을 projector가 대신 → mixing을 옳은 차원(스케일)에 사용.
        proj = build_multiscale_attn_projector(model, fracs=(0.25, 0.5, 0.75), d=1024).to(device)
        return proj, proj.trainable_parameters(), dict(trainable=True, tokens=256)

    raise ValueError(name)
