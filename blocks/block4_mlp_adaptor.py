from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class Block4Output:
    """
    Output of Block 4.

    audio_tokens:
        [B, T, 896] if input was batched [B, T, 1024]
        [T, 896]    if input was unbatched [T, 1024]

    residual:
        The projected skip branch Linear(1024 -> 896) on the original input.

    mlp_out:
        The main MLP branch output before residual addition:
        Linear(1024->896) -> GELU -> Dropout -> Linear(896->896)

    num_tokens:
        Sequence length T

    num_chunks:
        Optional chunk count propagated from caller if available
    """
    audio_tokens: torch.Tensor
    residual: torch.Tensor
    mlp_out: torch.Tensor
    num_tokens: int
    num_chunks: Optional[int]


class RMSNorm(nn.Module):
    """
    RMSNorm over the last dimension.
    """
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class Block4MLPAdaptor(nn.Module):
    """
    Block 4 · MLP Adaptor

    Purpose:
        Maps Block 3 fused audio tokens from 1024-dim into the LLM hidden space
        for Qwen-2.5-0.5B-Instruct (896-dim).

    Architecture:
        main branch:
            Linear(1024 -> 896)
            GELU
            Dropout(p)
            Linear(896 -> 896)

        residual branch:
            Linear(1024 -> 896)

        output:
            RMSNorm(main + residual)

    Input:
        h_full  [B, T, 1024] or [T, 1024]

    Output:
        audio_tokens [B, T, 896] or [T, 896]

    Stage usage:
        Stage 1: only this block trains; earlier blocks remain frozen
    """

    def __init__(
        self,
        in_dim: int = 1024,
        hidden_dim: int = 896,
        dropout: float = 0.1,
        rms_eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout
        self.rms_eps = rms_eps

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.residual_proj = nn.Linear(in_dim, hidden_dim)
        self.norm = RMSNorm(hidden_dim, eps=rms_eps)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Stable default init for adaptor training.
        """
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        nn.init.xavier_uniform_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)

    def forward(
        self,
        h_full: torch.Tensor,
        num_chunks: Optional[int] = None,
    ) -> Block4Output:
        """
        Args:
            h_full:
                [B, T, 1024] or [T, 1024]
            num_chunks:
                Optional metadata from Block 3

        Returns:
            Block4Output with audio_tokens in 896-dim LLM space
        """
        if h_full.ndim not in (2, 3):
            raise ValueError(
                f"Expected h_full with shape [T, {self.in_dim}] or [B, T, {self.in_dim}], "
                f"got {tuple(h_full.shape)}"
            )
        if h_full.size(-1) != self.in_dim:
            raise ValueError(
                f"Expected last dim = {self.in_dim}, got {tuple(h_full.shape)}"
            )

        squeeze = (h_full.ndim == 2)
        if squeeze:
            h_full = h_full.unsqueeze(0)  # [1, T, 1024]

        residual = self.residual_proj(h_full)               # [B, T, 896]
        mlp_out = self.fc2(self.dropout(self.act(self.fc1(h_full))))  # [B, T, 896]
        audio_tokens = self.norm(mlp_out + residual)        # [B, T, 896]

        if squeeze:
            residual = residual.squeeze(0)
            mlp_out = mlp_out.squeeze(0)
            audio_tokens = audio_tokens.squeeze(0)

        return Block4Output(
            audio_tokens=audio_tokens,
            residual=residual,
            mlp_out=mlp_out,
            num_tokens=int(audio_tokens.size(-2)),
            num_chunks=num_chunks,
        )

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    # Keep naming consistent with earlier blocks
    def freeze_base_weights(self) -> None:
        self.freeze()

    def unfreeze_all(self) -> None:
        self.unfreeze()

    def enable_training_dropouts(self, dropout: float = 0.1) -> None:
        self.dropout.p = dropout
        self.dropout_p = dropout

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"dropout={self.dropout_p}, "
            f"total_params={self.total_params:,}"
        )


if __name__ == "__main__":
    B, T = 2, 2250
    x = torch.randn(B, T, 1024)

    block4 = Block4MLPAdaptor()
    out = block4(x, num_chunks=3)

    print(block4)
    print("audio_tokens:", out.audio_tokens.shape)  # [2, 2250, 896]
    print("residual    :", out.residual.shape)
    print("mlp_out     :", out.mlp_out.shape)
    print("num_tokens  :", out.num_tokens)
    print("num_chunks  :", out.num_chunks)
    print("total params:", block4.total_params)
    print("trainable   :", block4.trainable_params)