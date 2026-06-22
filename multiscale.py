"""
maxinfo/multiscale.py

가설: 기존 OpenVLA projector(2-layer MLP)가 "공간정보를 전달 안 한다"는 직관은
     projector 내부(token-wise냐 mixing이냐)가 아니라 **입력**에서 비롯된다.

근거(코드): PrismaticVisionBackbone은 DINO/SigLIP 각각 **끝에서 2번째 블록**의 패치
     feature만 뽑아 LLM에 넘긴다 (get_intermediate_layers, n={len(blocks)-2}).
     ViT 초기/중간 레이어의 fine-scale 공간 디테일은 LLM에 **도달조차 안 한다.**
     → mixing(LLM이 대신 함)·압축(손실)과 달리, 멀티스케일 feature는 파이프라인에
       구조적으로 없던 정보를 추가하므로 LLM이 복원할 수 없다 = 진짜 새 정보.

설계:
  1. enable_multiscale(model, fracs): vision_backbone.forward를 패치.
     - 반환값(main) = 기존 끝-2 레이어 [B,256,2176] 그대로 → base MLP 경로 100% 보존.
     - 추가 레이어(early/mid)들을 vb._ms_cache 에 [B,256, 2176*k] 로 캐시(no_grad).
  2. MultiScaleProjector: out = MLP_orig(x) + gamma * Enhance(ms_cache)
     - base = 사전학습 MLP (freeze) → 학습 시작 == 원본 OpenVLA(gamma=0).
     - Enhance = token-wise(공간 mixing 없음) MLP. "입력(멀티스케일) 효과"만 격리.
     - gamma zero-init: 멀티스케일 정보가 유용하면 gamma가 0에서 벗어난다(=증거).
"""
import types
import torch
import torch.nn as nn


def enable_multiscale(model, fracs=(0.5,)):
    """vision_backbone을 멀티스케일로 패치. fracs = 추가로 뽑을 레이어의 (블록수 대비) 위치.
    예: fracs=(0.5,) → 중간 레이어 1개 추가(끝-2 포함 총 2스케일).
    반환: ms_dim (멀티스케일 캐시의 채널 수, = 2176 * (len(fracs)+1)).
    """
    vb = model.vision_backbone
    # fracs가 같으면 재패치 skip; 다르면 다시 패치(한 프로세스서 multiscale·multiscale3 둘 다 돌릴 때 필요)
    if getattr(vb, "_ms_enabled", False) and getattr(vb, "_ms_fracs", None) == tuple(fracs):
        return vb._ms_dim

    fused = vb.use_fused_vision_backbone
    f0 = vb.featurizer
    f1 = vb.fused_featurizer if fused else None

    def layer_set(feat):
        n = len(feat.blocks)
        penult = n - 2
        idxs = sorted({max(0, min(n - 1, int(round(f * n)))) for f in fracs} | {penult})
        return idxs, penult

    idx0, pen0 = layer_set(f0)
    idx1, pen1 = (layer_set(f1) if fused else (None, None))

    per_scale_dim = vb.embed_dim                       # 2176 (dino+siglip penult 합)
    ms_dim = per_scale_dim * len(idx0)

    @torch.no_grad()
    def ms_forward(self, pixel_values):
        if not fused:
            outs = self.featurizer.get_intermediate_layers(pixel_values, n=set(idx0))
            scales = list(outs)                        # [ [B,N,C], ... ] in idx0 순서
            main = scales[idx0.index(pen0)]
            self._ms_cache = torch.cat(scales, dim=2)
            return main

        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        d_outs = self.featurizer.get_intermediate_layers(img, n=set(idx0))
        s_outs = self.fused_featurizer.get_intermediate_layers(img_fused, n=set(idx1))
        # 각 스케일별로 dino,siglip concat → [B,N,2176]; 그다음 스케일끼리 concat → [B,N,2176*k]
        scales = [torch.cat([d, s], dim=2) for d, s in zip(d_outs, s_outs)]
        main = scales[idx0.index(pen0)]                # 끝-2 스케일 = 원본과 동일
        self._ms_cache = torch.cat(scales, dim=2)      # [B,N, ms_dim]
        return main

    vb.forward = types.MethodType(ms_forward, vb)
    vb._ms_enabled = True
    vb._ms_dim = ms_dim
    vb._ms_fracs = tuple(fracs)
    return ms_dim


class MultiScaleProjector(nn.Module):
    """base 사전학습 MLP(freeze) + zero-init gate * token-wise Enhance(멀티스케일 feature).

    base는 vision_backbone이 반환하는 끝-2 feature(vf)를 그대로 받고,
    Enhance는 vb._ms_cache(멀티스케일, ms_dim)를 읽어 처리한다.
    """
    def __init__(self, base_projector, vision_backbone, ms_dim, llm_dim,
                 inner_dim=2048):
        super().__init__()
        self.base = base_projector
        self.vb = vision_backbone           # _ms_cache 읽기용(파라미터로 등록 안 함)
        self.ms_dim = ms_dim
        self.llm_dim = llm_dim

        self.enhance = nn.Sequential(
            nn.Linear(ms_dim, inner_dim), nn.LayerNorm(inner_dim), nn.GELU(),
            nn.Linear(inner_dim, inner_dim), nn.LayerNorm(inner_dim), nn.GELU(),
            nn.Linear(inner_dim, llm_dim),
        )
        # gate: gamma만 zero-init → init에서 out=base(no-regret) 이지만
        # ∂L/∂gamma = Σ(enh·grad) ≠ 0 이라 gamma가 학습될 수 있다.
        # (enhance 출력까지 zero-init하면 ∂L/∂gamma=0, ∂L/∂enhance=0 데드락 → 절대 학습 안 됨)
        self.gamma = nn.Parameter(torch.zeros(1))

    def __setattr__(self, name, value):
        # self.vb 를 nn.Module 서브모듈로 등록하지 않도록(파라미터 중복/freeze 꼬임 방지)
        if name == "vb":
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def freeze_base(self):
        for p in self.base.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("base.")]

    def forward(self, vision_features):
        base = self.base(vision_features)                       # [B,256,llm] 원본 정렬 하한선
        ms = self.vb._ms_cache.to(torch.float32)                # [B,256,ms_dim]
        enh = self.enhance(ms)                                  # [B,256,llm] (fp32)
        return base + (self.gamma * enh).to(base.dtype)


class MultiScaleAttnProjector(nn.Module):
    """스케일 간(cross-scale) attention 융합 projector.

    패치는 섞지 않는다(LLM이 이미 함). 대신 각 패치 위치에서 k개 스케일(ViT 층) feature를
    attention으로 융합한다 — 어느 층이 그 패치에 중요한지 고르는 일은 LLM이 다른 층을
    아예 못 보므로 대체 불가능하다. mixing을 '옳은 차원(스케일)'에 쓰는 설계.

      per-patch:  query  --attend-->  {scale_0, scale_1, ..., scale_{k-1}}  -> fused
      out = frozen_MLP(x_penult) + gamma * proj_out(fused)   (gamma zero-init, deadlock-fix)
    """
    def __init__(self, base_projector, vision_backbone, ms_dim, per_scale_dim, llm_dim,
                 d=1024, num_heads=8):
        super().__init__()
        self.base = base_projector
        self.vb = vision_backbone
        self.per_scale_dim = per_scale_dim
        self.k = ms_dim // per_scale_dim
        self.d = d
        self.llm_dim = llm_dim

        self.lift = nn.Linear(per_scale_dim, d)
        self.scale_embed = nn.Parameter(torch.zeros(1, self.k, d))   # 어느 스케일인지 구분
        self.query = nn.Parameter(torch.zeros(1, 1, d))
        self.attn = nn.MultiheadAttention(d, num_heads, batch_first=True)
        self.out_ln = nn.LayerNorm(d)
        self.proj_out = nn.Linear(d, llm_dim)
        self.gamma = nn.Parameter(torch.zeros(1))                    # gamma만 zero(데드락 회피)
        nn.init.trunc_normal_(self.scale_embed, std=0.02)
        nn.init.trunc_normal_(self.query, std=0.02)

    def __setattr__(self, name, value):
        if name == "vb":
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def freeze_base(self):
        for p in self.base.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("base.")]

    def forward(self, vision_features):
        base = self.base(vision_features)                           # [B,N,llm]
        ms = self.vb._ms_cache.to(torch.float32)                    # [B,N,k*C]
        B, N, _ = ms.shape
        x = ms.view(B, N, self.k, self.per_scale_dim)               # [B,N,k,C]
        x = self.lift(x) + self.scale_embed.unsqueeze(1)            # [B,N,k,d]
        xr = x.reshape(B * N, self.k, self.d)                       # 패치를 배치로(패치 mixing 없음)
        q = self.query.expand(B * N, 1, self.d)
        fused, _ = self.attn(q, xr, xr)                             # [B*N,1,d] 스케일 융합
        f = fused.squeeze(1).view(B, N, self.d)
        enh = self.proj_out(self.out_ln(f))                        # [B,N,llm]
        return base + (self.gamma * enh).to(base.dtype)


def build_multiscale_attn_projector(model, fracs=(0.5,), d=1024, num_heads=8):
    ms_dim = enable_multiscale(model, fracs=fracs)
    base = model.projector
    llm_dim = base.fc3.out_features
    per_scale_dim = model.vision_backbone.embed_dim
    proj = MultiScaleAttnProjector(base, model.vision_backbone, ms_dim, per_scale_dim,
                                   llm_dim, d=d, num_heads=num_heads)
    proj.freeze_base()
    for n, p in proj.named_parameters():
        if not n.startswith("base."):
            p.data = p.data.float()
    return proj


def build_multiscale_projector(model, fracs=(0.5,), inner_dim=2048):
    ms_dim = enable_multiscale(model, fracs=fracs)
    base = model.projector
    llm_dim = base.fc3.out_features
    proj = MultiScaleProjector(base, model.vision_backbone, ms_dim, llm_dim, inner_dim=inner_dim)
    proj.freeze_base()
    for n, p in proj.named_parameters():
        if not n.startswith("base."):
            p.data = p.data.float()
    return proj
