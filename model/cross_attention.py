# models/cross_attention.py
import torch
import torch.nn as nn
from typing import Optional, Tuple


def _to_last_channel(x: torch.Tensor, c_expected: int):
    """
    把输入规范为 (B, N, C)，返回 (x_std, transposed)
    - 若 x 是 (B, N, C) 且 C==c_expected：直接返回，transposed=False
    - 若 x 是 (B, C, N) 且 C==c_expected：转置到 (B, N, C)，transposed=True
    - 否则报错
    """
    if x.dim() != 3:
        raise RuntimeError(f"Expect 3D tensor, got {x.shape}")
    B, A, B_or_C = x.shape
    # 情况1：(..., C) 在末维
    if x.size(-1) == c_expected:
        return x, False
    # 情况2：(..., C) 在中间维 (B, C, N)
    if x.size(1) == c_expected:
        return x.transpose(1, 2).contiguous(), True

    raise RuntimeError(
        f"Given normalized_shape=[{c_expected}], but got input shape {tuple(x.shape)}; "
        f"expect (B, N, {c_expected}) or (B, {c_expected}, N)."
    )


class SimpleCrossAttention(nn.Module):
    """
    跨模态注意力（健壮版）
    - 支持 Q 与 K/V 的通道维既可能在末维 (B,N,C)，也可能在中维 (B,C,N)
    - 内部自动转成 (B,N,C) 做注意力，最后把输出还原成和 Q 一致的布局
    """

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        d_model: Optional[int] = None,
        num_heads: int = 8,
        drop: float = 0.0,
    ):
        super().__init__()
        self.q_dim = int(q_dim)
        self.kv_dim = int(kv_dim)
        self.d = int(d_model) if d_model is not None else max(self.q_dim, self.kv_dim)
        self.h = int(num_heads)
        assert self.d % self.h == 0, f"d_model={self.d} must be divisible by num_heads={self.h}"
        self.dh = self.d // self.h

        self.q_proj = nn.Linear(self.q_dim, self.d, bias=False)
        self.k_proj = nn.Linear(self.kv_dim, self.d, bias=False)
        self.v_proj = nn.Linear(self.kv_dim, self.d, bias=False)
        self.out = nn.Linear(self.d, self.q_dim, bias=False)

        self.drop = nn.Dropout(drop)
        self.ln_q = nn.LayerNorm(self.q_dim)
        self.ln_kv = nn.LayerNorm(self.kv_dim)

    def forward(self, Q: torch.Tensor, KV: torch.Tensor) -> torch.Tensor:
        """
        Q:  (B, Nq, q_dim) 或 (B, q_dim, Nq)
        KV: (B, Nk, kv_dim) 或 (B, kv_dim, Nk)
        return: 形状与 Q 保持一致（即与输入 Q 的布局一致）
        """
        # 统一到 (B, N, C)
        Q_std, q_trans = _to_last_channel(Q, self.q_dim)
        KV_std, _ = _to_last_channel(KV, self.kv_dim)

        B, Nq, _ = Q_std.shape
        Nk = KV_std.shape[1]
        scale = self.dh ** -0.5

        # LN + 线性投影
        q = self.q_proj(self.ln_q(Q_std)).view(B, Nq, self.h, self.dh).transpose(1, 2)  # (B,h,Nq,dh)
        k = self.k_proj(self.ln_kv(KV_std)).view(B, Nk, self.h, self.dh).transpose(1, 2)  # (B,h,Nk,dh)
        v = self.v_proj(self.ln_kv(KV_std)).view(B, Nk, self.h, self.dh).transpose(1, 2)  # (B,h,Nk,dh)

        attn = (q @ k.transpose(-2, -1)) * scale  # (B,h,Nq,Nk)
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        z = attn @ v  # (B,h,Nq,dh)
        z = z.transpose(1, 2).contiguous().view(B, Nq, self.d)  # (B,Nq,d)
        out = self.out(z)  # (B,Nq,q_dim)

        # 若 Q 原本是 (B, q_dim, Nq)，则把输出还原为同布局
        if q_trans:
            out = out.transpose(1, 2).contiguous()  # (B,q_dim,Nq)
        return out


class BiDirectionalCrossAttention(nn.Module):
    """
    双向交叉注意力：
      - 2D <- 3D：用 3D 更新 2D
      - 3D <- 2D：用 2D 更新 3D
    内部自动处理通道轴位置差异（(B,N,C) / (B,C,N) 都可）
    """

    def __init__(
        self,
        dim_2d: int,
        dim_3d: int,
        num_heads: int = 8,
        drop: float = 0.0,
        order: str = "2d_first",
        d_model: Optional[int] = None,
    ):
        super().__init__()
        d_attn = d_model if d_model is not None else max(int(dim_2d), int(dim_3d))
        self.ca_2d_from_3d = SimpleCrossAttention(
            q_dim=dim_2d, kv_dim=dim_3d, d_model=d_attn, num_heads=num_heads, drop=drop
        )
        self.ca_3d_from_2d = SimpleCrossAttention(
            q_dim=dim_3d, kv_dim=dim_2d, d_model=d_attn, num_heads=num_heads, drop=drop
        )
        self.order = order

    def forward(self, t2: torch.Tensor, t3: torch.Tensor):
        """
        t2: (B,HW,C2) 或 (B,C2,HW)
        t3: (B,N,C3)  或 (B,C3,N)
        """
        if self.order == "2d_first":
            t2 = t2 + self.ca_2d_from_3d(t2, t3)
            t3 = t3 + self.ca_3d_from_2d(t3, t2)
        else:
            t3 = t3 + self.ca_3d_from_2d(t3, t2)
            t2 = t2 + self.ca_2d_from_3d(t2, t3)
        return t2, t3
