from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utilities
# ============================================================


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm2D(nn.Module):
    """LayerNorm over channels for BCHW tensors."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BCHW -> normalize over C
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class LowRankMLP(nn.Module):
    def __init__(self, dim: int, hidden_rank: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(dim, hidden_rank, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Conv2d(hidden_rank, dim, kernel_size=1, bias=True)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# ============================================================
# Harmonic tokenization
# ============================================================


def build_dct_basis_2d(patch_size: int, keep_coeffs: int, channels: int = 3) -> torch.Tensor:
    """
    Build a fixed low-frequency 2D-DCT basis for patch tokenization.
    Returns a tensor of shape [C*P*P, K].
    """
    p = patch_size
    coords = torch.arange(p, dtype=torch.float32)
    basis_1d = []
    for u in range(p):
        alpha = math.sqrt(1.0 / p) if u == 0 else math.sqrt(2.0 / p)
        basis_1d.append(alpha * torch.cos((math.pi * (2 * coords + 1) * u) / (2 * p)))
    basis_1d = torch.stack(basis_1d, dim=0)  # [P, P]

    coeffs = []
    # zig-zag-ish by low total frequency first
    freq_pairs = [(u, v) for u in range(p) for v in range(p)]
    freq_pairs = sorted(freq_pairs, key=lambda t: (t[0] + t[1], t[0], t[1]))[:keep_coeffs]
    for u, v in freq_pairs:
        patch_basis = torch.outer(basis_1d[u], basis_1d[v]).reshape(-1)
        for c in range(channels):
            vec = torch.zeros(channels * p * p, dtype=torch.float32)
            start = c * p * p
            vec[start:start + p * p] = patch_basis
            coeffs.append(vec)
    B = torch.stack(coeffs, dim=1)  # [C*P*P, K*C]
    return B


class HarmonicTokenizer(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        patch_size: int = 8,
        keep_coeffs: int = 16,
        embed_dim: int = 192,
        rank: int = 64,
        norm: bool = True,
    ) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.keep_coeffs = keep_coeffs
        basis = build_dct_basis_2d(patch_size, keep_coeffs, channels=in_chans)
        self.register_buffer("basis", basis, persistent=False)
        basis_dim = basis.shape[1]
        self.proj1 = nn.Linear(basis_dim, rank)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(rank, embed_dim)
        self.norm = nn.LayerNorm(embed_dim) if norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        if H % p != 0 or W % p != 0:
            raise ValueError(f"Input spatial size {(H, W)} must be divisible by patch_size={p}.")

        patches = F.unfold(x, kernel_size=p, stride=p).transpose(1, 2)  # [B, N, C*P*P]
        z = patches @ self.basis  # [B, N, basis_dim]
        z = self.proj2(self.act(self.proj1(z)))
        z = self.norm(z)
        Hp, Wp = H // p, W // p
        z = z.transpose(1, 2).reshape(B, -1, Hp, Wp)
        return z


# ============================================================
# 1D state-space scan used in 2D directions
# ============================================================


class SelectiveSSM1D(nn.Module):
    """
    Lightweight linear-time SSM scan over a sequence.
    Input: [B, L, D]
    Output: [B, L, D]
    """

    def __init__(self, dim: int, state_dim: int = 16, rank: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.rank = rank

        # Diagonal dynamics + low-rank correction
        self.a_logit = nn.Parameter(torch.zeros(state_dim))
        self.U = nn.Parameter(torch.randn(state_dim, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(state_dim, rank) * 0.02)

        self.in_proj = nn.Linear(dim, state_dim, bias=True)
        self.gate_proj = nn.Linear(dim, state_dim, bias=True)
        self.out_proj = nn.Linear(state_dim, dim, bias=True)
        self.skip = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        if reverse:
            x = torch.flip(x, dims=[1])
        B, L, D = x.shape
        u = self.in_proj(x)  # [B, L, S]
        g = torch.sigmoid(self.gate_proj(x))
        # Stable diagonal term in (0,1)
        a = torch.sigmoid(self.a_logit).view(1, 1, self.state_dim)
        # Low-rank state coupling
        lowrank = self.U @ self.V.t()  # [S, S]

        h = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            ut = u[:, t, :]
            # cheap low-rank correction on current state
            h = a.squeeze(1) * h + g[:, t, :] * ut + (h @ lowrank.t())
            ys.append(self.out_proj(h) + self.skip(x[:, t, :]))
        y = torch.stack(ys, dim=1)
        if reverse:
            y = torch.flip(y, dims=[1])
        return y


class HSSM2D(nn.Module):
    def __init__(self, dim: int, state_dim: int = 16, rank: int = 8, gate_rank: int = 64) -> None:
        super().__init__()
        self.row_ssm = SelectiveSSM1D(dim, state_dim=state_dim, rank=rank)
        self.col_ssm = SelectiveSSM1D(dim, state_dim=state_dim, rank=rank)
        self.gate = nn.Sequential(
            nn.Conv2d(dim, gate_rank, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(gate_rank, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.out_norm = LayerNorm2D(dim)

    def _scan_rows(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        B, C, H, W = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
        y = self.row_ssm(seq, reverse=reverse)
        y = y.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return y

    def _scan_cols(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        B, C, H, W = x.shape
        seq = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
        y = self.col_ssm(seq, reverse=reverse)
        y = y.reshape(B, W, H, C).permute(0, 3, 2, 1)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = (
            self._scan_rows(x, reverse=False)
            + self._scan_rows(x, reverse=True)
            + self._scan_cols(x, reverse=False)
            + self._scan_cols(x, reverse=True)
        ) * 0.25
        g = self.gate(self.out_norm(x))
        out = g * y + (1.0 - g) * x
        return out


class HSSMBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        state_dim: int = 16,
        ssm_rank: int = 8,
        mlp_rank: int = 64,
        drop_path_prob: float = 0.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm2D(dim)
        self.mixer = HSSM2D(dim, state_dim=state_dim, rank=ssm_rank, gate_rank=max(dim // 4, 32))
        self.drop_path1 = DropPath(drop_path_prob)
        self.norm2 = LayerNorm2D(dim)
        self.mlp = LowRankMLP(dim, hidden_rank=mlp_rank, dropout=dropout)
        self.drop_path2 = DropPath(drop_path_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.mixer(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class TokenDownsample(nn.Module):
    """Convolution-free stage transition by 2x2 average pooling + channel projection."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=True)
        self.norm = LayerNorm2D(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.proj(x)
        x = self.norm(x)
        return x


class LinearFPNFusion(nn.Module):
    def __init__(self, dims: Sequence[int], out_dim: int = 256) -> None:
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(d, out_dim, kernel_size=1) for d in dims])
        self.refine = nn.ModuleList([HSSMBlock(out_dim, state_dim=16, ssm_rank=8, mlp_rank=64) for _ in dims])

    def forward(self, feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        xs = [lat(f) for lat, f in zip(self.lateral, feats)]
        out = [None] * len(xs)
        out[-1] = self.refine[-1](xs[-1])
        for i in range(len(xs) - 2, -1, -1):
            up = F.interpolate(out[i + 1], size=xs[i].shape[-2:], mode="bilinear", align_corners=False)
            out[i] = self.refine[i](xs[i] + up)
        return out


@dataclass
class HSSMVisionConfig:
    image_size: int = 224
    in_chans: int = 3
    num_classes: int = 1000
    patch_size: int = 8
    keep_coeffs: int = 16
    dims: Tuple[int, int, int, int] = (192, 384, 576, 768)
    depths: Tuple[int, int, int, int] = (2, 2, 6, 2)
    state_dims: Tuple[int, int, int, int] = (16, 16, 24, 24)
    ssm_ranks: Tuple[int, int, int, int] = (8, 8, 8, 8)
    embed_ranks: Tuple[int, int, int, int] = (64, 96, 144, 192)
    mlp_ranks: Tuple[int, int, int, int] = (64, 96, 144, 192)
    drop_path_rate: float = 0.1
    dropout: float = 0.0


class HSSMVision(nn.Module):
    def __init__(self, cfg: HSSMVisionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = HarmonicTokenizer(
            in_chans=cfg.in_chans,
            patch_size=cfg.patch_size,
            keep_coeffs=cfg.keep_coeffs,
            embed_dim=cfg.dims[0],
            rank=cfg.embed_ranks[0],
        )

        total_blocks = sum(cfg.depths)
        dp_rates = torch.linspace(0, cfg.drop_path_rate, total_blocks).tolist()
        dp_idx = 0

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i, (dim, depth, sdim, srank, mrank) in enumerate(
            zip(cfg.dims, cfg.depths, cfg.state_dims, cfg.ssm_ranks, cfg.mlp_ranks)
        ):
            blocks = []
            for _ in range(depth):
                blocks.append(
                    HSSMBlock(
                        dim=dim,
                        state_dim=sdim,
                        ssm_rank=srank,
                        mlp_rank=mrank,
                        drop_path_prob=dp_rates[dp_idx],
                        dropout=cfg.dropout,
                    )
                )
                dp_idx += 1
            self.stages.append(nn.Sequential(*blocks))
            if i < len(cfg.dims) - 1:
                self.downsamples.append(TokenDownsample(cfg.dims[i], cfg.dims[i + 1]))

        self.final_norm = LayerNorm2D(cfg.dims[-1])
        self.head = nn.Linear(cfg.dims[-1], cfg.num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, LayerNorm2D)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor, return_pyramid: bool = False):
        feats = []
        x = self.tokenizer(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            feats.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        x = self.final_norm(x)
        if return_pyramid:
            feats[-1] = x
            return feats
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = x.mean(dim=(2, 3))
        x = self.head(x)
        return x


class HSSMDetector(nn.Module):
    """Simple anchor-free detection-ready scaffold using HSSM backbone features.
    This is a minimal research scaffold, not a full production detector.
    """

    def __init__(self, backbone: HSSMVision, num_classes: int = 80, fpn_dim: int = 256) -> None:
        super().__init__()
        self.backbone = backbone
        self.fpn = LinearFPNFusion(backbone.cfg.dims, out_dim=fpn_dim)
        self.cls_head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(fpn_dim, num_classes, kernel_size=1),
        )
        self.box_head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(fpn_dim, 4, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        feats = self.backbone.forward_features(x, return_pyramid=True)
        pyramid = self.fpn(feats)
        cls_logits = [self.cls_head(f) for f in pyramid]
        box_regs = [self.box_head(f) for f in pyramid]
        return {"cls_logits": cls_logits, "box_regs": box_regs, "features": pyramid}


def hssm_vision_tiny(num_classes: int = 1000) -> HSSMVision:
    cfg = HSSMVisionConfig(
        num_classes=num_classes,
        dims=(160, 320, 480, 640),
        depths=(2, 2, 5, 2),
        state_dims=(12, 16, 20, 24),
        embed_ranks=(48, 80, 128, 160),
        mlp_ranks=(48, 80, 128, 160),
        drop_path_rate=0.08,
    )
    return HSSMVision(cfg)


def hssm_vision_small(num_classes: int = 1000) -> HSSMVision:
    cfg = HSSMVisionConfig(num_classes=num_classes)
    return HSSMVision(cfg)


def hssm_vision_base(num_classes: int = 1000) -> HSSMVision:
    cfg = HSSMVisionConfig(
        num_classes=num_classes,
        dims=(224, 448, 672, 896),
        depths=(3, 3, 9, 3),
        state_dims=(16, 20, 24, 32),
        embed_ranks=(64, 112, 160, 224),
        mlp_ranks=(64, 112, 160, 224),
        drop_path_rate=0.2,
    )
    return HSSMVision(cfg)


if __name__ == "__main__":
    model = hssm_vision_small(num_classes=1000)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("logits:", y.shape)
    det = HSSMDetector(model, num_classes=80)
    out = det(x)
    print([t.shape for t in out["cls_logits"]])
