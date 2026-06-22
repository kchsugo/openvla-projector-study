"""
maxinfo/projector.py

비전 정보를 "최대로 보존"하는 것을 목표로 설계한 OpenVLA projector.

설계 4원칙:
  1. 토큰 수 유지        : 256 -> 256 (압축 없음)
  2. 채널 병목 금지      : 출력은 항상 llm_dim(4096), enhancement 내부도 vision_dim 이상 권장
  3. cross-token mixing : self-attention 블록으로 패치 간 관계/공간 정보 보강
  4. 사전학습 정렬 보존  : 원본 사전학습 MLP를 그대로 residual base로 두고,
                          enhancement 브랜치는 gamma=0 (zero-init gate)로 시작
                          -> 학습 시작 시점에 모델 == 원본 OpenVLA

  out = MLP_orig(x)  +  gamma * Enhance(x)        (gamma 초기값 0)

이 구조는 "정보 보존"을 구조적으로 보장한다: enhancement가 도움이 안 되면
gamma가 0 근처로 남아 원본 MLP 출력으로 폴백할 수 있다(하한선 = 원본 OpenVLA).
"""
import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    """Pre-norm Transformer 인코더 블록 (self-attention + FFN)."""
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x


class MaxInfoProjector(nn.Module):
    """
    원본 projector(base_projector)를 residual base로 감싸고,
    zero-init gate로 self-attention enhancement를 더한다.

    Args:
        base_projector : 사전학습된 OpenVLA projector 모듈(그대로 재사용; freeze 권장)
        vision_dim     : 비전 feature 차원 (fused = 2176)
        llm_dim        : LLM hidden 차원 (4096)
        inner_dim      : enhancement 내부 연산 차원 (병목 방지 위해 크게; 기본 2048)
        depth          : enhancement self-attention 블록 수
        num_heads      : attention head 수
        grid           : 패치 그리드 한 변 (16 -> 256 토큰); 위치 임베딩용
    """
    def __init__(self, base_projector: nn.Module, vision_dim: int, llm_dim: int,
                 inner_dim: int = 2048, depth: int = 2, num_heads: int = 8, grid: int = 16,
                 zero_init_out: bool = True):
        super().__init__()
        # zero_init_out=True(원본): proj_out=0 + gamma=0 이중 zero-init → ∂L/∂gamma=0,
        #   ∂L/∂enhance=0 데드락(enhancement가 영원히 학습 안 됨).
        # zero_init_out=False(수정): gamma=0만 → init에서 out=base(no-regret)이되
        #   ∂L/∂gamma=Σ(enh·grad)≠0 이라 gamma가 학습됨.
        self.zero_init_out = zero_init_out
        self.base = base_projector
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim
        self.inner_dim = inner_dim
        self.grid = grid

        self.lift = nn.Linear(vision_dim, inner_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, grid * grid, inner_dim))
        self.blocks = nn.ModuleList([EncoderBlock(inner_dim, num_heads) for _ in range(depth)])
        self.out_ln = nn.LayerNorm(inner_dim)
        self.proj_out = nn.Linear(inner_dim, llm_dim)
        # zero-init gate: 학습 시작 == 원본 OpenVLA
        self.gamma = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.lift.weight); nn.init.zeros_(self.lift.bias)
        if self.zero_init_out:
            # 원본: proj_out도 0 → gamma와 이중 zero-init(데드락). 논문 재현용으로만 유지.
            nn.init.zeros_(self.proj_out.weight); nn.init.zeros_(self.proj_out.bias)

    def freeze_base(self):
        for p in self.base.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("base.")]

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        # Path A: 원본 사전학습 MLP (정보/정렬 하한선)
        base = self.base(vision_features)  # [B, 256, llm_dim]

        # Path B: enhancement (fp32 안정 연산, 병목 없음)
        od = base.dtype
        h = self.lift(vision_features.to(torch.float32))      # [B, 256, inner]
        h = h + self.pos_embed.to(torch.float32)
        for blk in self.blocks:
            h = blk(h)
        enh = self.proj_out(self.out_ln(h))                   # [B, 256, llm_dim] (fp32)

        return base + (self.gamma * enh).to(od)               # base(od) + gated enhancement(od)


def build_maxinfo_projector(model, **kwargs) -> MaxInfoProjector:
    """로드된 OpenVLA model의 projector를 MaxInfoProjector로 감싸 교체한다."""
    base = model.projector
    vision_dim = base.fc1.in_features
    llm_dim = base.fc3.out_features
    proj = MaxInfoProjector(base, vision_dim, llm_dim, **kwargs)
    proj.freeze_base()
    return proj
