"""
blocks/block3_perceiver_fusion.py
─────────────────────────────────────────────────────────────────────────────
Block 3: Perceiver Resampler + Cross-Attention Fusion

Sub-blocks:
  [3a] PerceiverResampler   — compress h_ob: 1496 tokens → 64 tokens
  [3b] CrossAttentionFusion — fuse h_w (Whisper) with h_resampled (OpenBEATs)
  [3c] ChunkConcat          — concat multi-chunk with learnable chunk-pos-enc

No dimension projection inside Block 3.
Both encoders stay in 1024-dim throughout.

Inputs (per chunk):
  h_ob : [B, 1496, 1024]   OpenBEATs tokens   ← Block 2b output
  h_w  : [B,  750, 1024]   Whisper tokens     ← Block 2a output

Output:
  h_full: [B, 750*N_chunks, 1024]

For Qwen-2.5-0.5B-Instruct (hidden_dim=896):
  Add nn.Linear(1024, 896) AFTER this block to project into LLM space.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Block3Output:
    """
    h_full          : [B, T, 1024]  full fused sequence (T = 750 * N_chunks)
    h_fused_chunks  : list of N tensors, each [B, 750, 1024]
    h_resampled     : [B, 64, 1024]  last Perceiver output (for debugging)
    n_chunks        : int
    seq_len         : int  T = 750 * n_chunks
    """
    h_full:          torch.Tensor
    h_fused_chunks:  List[torch.Tensor]
    h_resampled:     torch.Tensor
    n_chunks:        int
    seq_len:         int


# ─────────────────────────────────────────────────────────────────────────────
# Shared building block: Pre-Norm Cross-Attention + FFN
# ─────────────────────────────────────────────────────────────────────────────
class _CrossAttentionBlock(nn.Module):
    """
    Pre-norm cross-attention with residual + FFN.

    y = x + CrossAttn(LayerNorm(x), LayerNorm(kv))
    y = y + FFN(LayerNorm(y))
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query:            torch.Tensor,
        kv:               torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : [B, N_q, D]
            kv    : [B, N_kv, D]
            key_padding_mask: [B, N_kv] bool, True = padded position
        Returns:
            out   : [B, N_q, D]
        """
        q_n  = self.norm_q(query)
        kv_n = self.norm_kv(kv)

        attn_out, _ = self.cross_attn(
            query=q_n,
            key=kv_n,
            value=kv_n,
            key_padding_mask=key_padding_mask,
        )
        query = query + attn_out

        query = query + self.ff(self.norm_ff(query))
        return query


# ─────────────────────────────────────────────────────────────────────────────
# [3a] Perceiver Resampler
# ─────────────────────────────────────────────────────────────────────────────
class PerceiverResampler(nn.Module):
    """
    Compresses h_ob [B, 1496, 1024] → h_resampled [B, 64, 1024]

    2-layer cross-attention: queries attend to h_ob twice,
    refining the compressed representation in each pass.

    Q = learned queries   [B, 64,   1024]
    K = h_ob patches      [B, 1496, 1024]
    V = h_ob patches      [B, 1496, 1024]
    out: h_resampled      [B, 64,   1024]
    """

    def __init__(
        self,
        d_model:   int = 1024,
        n_queries: int = 64,
        n_heads:   int = 16,
        n_layers:  int = 2,
        dropout:   float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0

        self.n_queries = n_queries

        # Learned query tokens — the core of the Perceiver
        self.queries = nn.Parameter(
            torch.empty(1, n_queries, d_model).normal_(std=d_model ** -0.5)
        )

        self.layers = nn.ModuleList([
            _CrossAttentionBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model)

    def forward(
        self,
        h_ob:             torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            h_ob             : [B, N_ob, 1024]  OpenBEATs patches
            key_padding_mask : [B, N_ob] bool, True = padded (optional)
        Returns:
            h_resampled      : [B, 64, 1024]
        """
        B = h_ob.size(0)
        q = self.queries.expand(B, -1, -1)   # [B, 64, 1024]

        for layer in self.layers:
            q = layer(query=q, kv=h_ob, key_padding_mask=key_padding_mask)

        return self.norm_out(q)   # [B, 64, 1024]


# ─────────────────────────────────────────────────────────────────────────────
# [3b] Cross-Attention Fusion
# ─────────────────────────────────────────────────────────────────────────────
class CrossAttentionFusion(nn.Module):
    """
    Fuses Whisper tokens with OpenBEATs summary.

    Q = h_w           [B, 750, 1024]  ← Whisper tokens (queries)
    K = h_resampled   [B,  64, 1024]  ← OpenBEATs summary (keys)
    V = h_resampled   [B,  64, 1024]  ← OpenBEATs summary (values)
    16 attention heads · 1 cross-attn layer
    out + residual: h_fused [B, 750, 1024]
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn_block = _CrossAttentionBlock(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )
        self.norm_out = nn.LayerNorm(d_model)

    def forward(
        self,
        h_w:          torch.Tensor,
        h_resampled:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h_w          : [B, 750, 1024]  Whisper tokens (queries)
            h_resampled  : [B,  64, 1024]  OpenBEATs summary (keys + values)
        Returns:
            h_fused      : [B, 750, 1024]
        """
        h_fused = self.attn_block(query=h_w, kv=h_resampled)
        return self.norm_out(h_fused)   # [B, 750, 1024]


# ─────────────────────────────────────────────────────────────────────────────
# [3c] Long-Audio Chunk Concat
# ─────────────────────────────────────────────────────────────────────────────
class ChunkConcat(nn.Module):
    """
    Concatenates per-chunk h_fused tensors along the sequence dim,
    adding learnable chunk-index positional encodings.

    For N chunks:
      h_full = cat([h_fused_1 + pos[0], ..., h_fused_N + pos[N-1]])
             → [B, 750*N, 1024]
    """

    def __init__(
        self,
        d_model:    int = 1024,
        max_chunks: int = 64,
    ) -> None:
        super().__init__()
        self.max_chunks = max_chunks

        # chunk_pos_enc[i] is added to every token in chunk i
        # shape: [max_chunks, 1, d_model] — broadcasts over [B, N_w, D]
        self.chunk_pos_enc = nn.Parameter(
            torch.zeros(max_chunks, 1, d_model)
        )
        nn.init.normal_(self.chunk_pos_enc, std=0.02)

    def forward(self, h_fused_chunks: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            h_fused_chunks : list of N tensors, each [B, N_w, 1024]
        Returns:
            h_full         : [B, N_w*N, 1024]
        """
        N = len(h_fused_chunks)
        assert N <= self.max_chunks, (
            f"Got {N} chunks, max_chunks={self.max_chunks}. "
            "Increase max_chunks at init time."
        )

        encoded = []
        for i, chunk in enumerate(h_fused_chunks):
            # chunk_pos_enc[i]: [1, D] → broadcasts to [B, N_w, D]
            enc = chunk + self.chunk_pos_enc[i]   # [B, N_w, 1024]
            encoded.append(enc)

        return torch.cat(encoded, dim=1)   # [B, N_w*N, 1024]


# ─────────────────────────────────────────────────────────────────────────────
# Block3PerceiverFusion — top-level module
# ─────────────────────────────────────────────────────────────────────────────
class Block3PerceiverFusion(nn.Module):
    """
    Block 3: Perceiver Resampler + Cross-Attention Fusion

    Combines Block 2a (Whisper) and Block 2b (OpenBEATs) outputs into a
    single fused audio token sequence ready for LLM conditioning.

    ─── Single chunk ───────────────────────────────────────────────────
    out = block3(h_ob, h_w)
    out.h_full  →  [B, 750, 1024]

    ─── Multi-chunk long audio ─────────────────────────────────────────
    chunks = [(h_ob_1, h_w_1), (h_ob_2, h_w_2), ...]
    out = block3.forward_multi_chunk(chunks)
    out.h_full  →  [B, 750*N, 1024]

    ─── Projecting to Qwen-2.5-0.5B (hidden_dim=896) ──────────────────
    qwen_proj = nn.Linear(1024, 896)       # lives outside Block 3
    h_qwen = qwen_proj(out.h_full)         # [B, T, 896]
    """

    def __init__(
        self,
        d_model:     int = 1024,
        n_queries:   int = 64,       # Perceiver output tokens
        n_heads_3a:  int = 16,       # Perceiver cross-attn heads
        n_heads_3b:  int = 16,       # Fusion cross-attn heads
        n_layers_3a: int = 2,        # Perceiver depth
        max_chunks:  int = 64,       # Max audio chunks for long-form
        dropout:     float = 0.0,
    ) -> None:
        super().__init__()

        self.d_model   = d_model
        self.n_queries = n_queries

        # [3a]
        self.perceiver = PerceiverResampler(
            d_model=d_model,
            n_queries=n_queries,
            n_heads=n_heads_3a,
            n_layers=n_layers_3a,
            dropout=dropout,
        )

        # [3b]
        self.cross_fuse = CrossAttentionFusion(
            d_model=d_model,
            n_heads=n_heads_3b,
            dropout=dropout,
        )

        # [3c]
        self.chunk_concat = ChunkConcat(
            d_model=d_model,
            max_chunks=max_chunks,
        )

    # ─── Internal helpers ────────────────────────────────────────────
    def _process_chunk(
        self,
        h_ob:            torch.Tensor,
        h_w:             torch.Tensor,
        ob_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process one (h_ob, h_w) pair.

        Returns:
            h_fused     : [B, N_w, 1024]
            h_resampled : [B, 64,  1024]
        """
        # [3a] Compress 1496 → 64 OpenBEATs tokens
        h_resampled = self.perceiver(h_ob, key_padding_mask=ob_padding_mask)

        # [3b] Whisper tokens attend to OpenBEATs summary
        h_fused = self.cross_fuse(h_w=h_w, h_resampled=h_resampled)

        return h_fused, h_resampled

    # ─── Forward: single chunk ───────────────────────────────────────
    def forward(
        self,
        h_ob:            torch.Tensor,
        h_w:             torch.Tensor,
        ob_padding_mask: Optional[torch.Tensor] = None,
    ) -> Block3Output:
        """
        Single-chunk forward.

        Args:
            h_ob : [B, N_ob, 1024]  OpenBEATs tokens (from Block 2b)
            h_w  : [B, N_w,  1024]  Whisper tokens   (from Block 2a)
            ob_padding_mask : [B, N_ob] optional bool mask
        Returns:
            Block3Output  — h_full: [B, N_w, 1024]
        """
        h_fused, h_resampled = self._process_chunk(h_ob, h_w, ob_padding_mask)

        # [3c] Single chunk still gets pos-enc at index 0
        h_full = self.chunk_concat([h_fused])

        return Block3Output(
            h_full=h_full,
            h_fused_chunks=[h_fused],
            h_resampled=h_resampled,
            n_chunks=1,
            seq_len=h_full.size(1),
        )

    # ─── Forward: multi-chunk long audio ─────────────────────────────
    def forward_multi_chunk(
        self,
        chunks: List[Tuple],
    ) -> Block3Output:
        """
        Multi-chunk forward for long-form audio (N > 1).

        Args:
            chunks : list of (h_ob, h_w) or (h_ob, h_w, ob_mask) tuples
                     each h_ob : [B, N_ob, 1024]
                     each h_w  : [B, N_w,  1024]
        Returns:
            Block3Output  — h_full: [B, N_w*N, 1024]
        """
        h_fused_list   = []
        h_resampled_last = None

        for chunk in chunks:
            if len(chunk) == 3:
                h_ob, h_w, ob_mask = chunk
            else:
                h_ob, h_w = chunk
                ob_mask = None

            h_fused, h_resampled = self._process_chunk(h_ob, h_w, ob_mask)
            h_fused_list.append(h_fused)
            h_resampled_last = h_resampled

        # [3c] Concat all chunks with learnable positional encodings
        h_full = self.chunk_concat(h_fused_list)

        return Block3Output(
            h_full=h_full,
            h_fused_chunks=h_fused_list,
            h_resampled=h_resampled_last,
            n_chunks=len(chunks),
            seq_len=h_full.size(1),
        )

    # ─── Training stage control ──────────────────────────────────────
    def freeze(self) -> None:
        """Freeze all Block 3 parameters."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all Block 3 parameters."""
        for p in self.parameters():
            p.requires_grad = True

    def enable_training_dropouts(self, dropout: float = 0.1) -> None:
        """Enable dropout for fine-tuning stages."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = dropout

    # ─── Introspection ───────────────────────────────────────────────
    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, "
            f"n_queries={self.n_queries}, "
            f"total_params={self.total_params:,}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch

    block3 = Block3PerceiverFusion()
    print(block3)
    print()

    B = 2

    # ── Single chunk ──────────────────────────────────────────────────
    h_ob = torch.randn(B, 1496, 1024)   # from Block 2b (OpenBEATs)
    h_w  = torch.randn(B,  750, 1024)   # from Block 2a (Whisper)

    out = block3(h_ob=h_ob, h_w=h_w)
    print(f"Single chunk:")
    print(f"  h_resampled : {out.h_resampled.shape}")   # [2, 64, 1024]
    print(f"  h_full      : {out.h_full.shape}")         # [2, 750, 1024]
    print()

    # ── Multi-chunk (3 × 30s = 90s audio) ────────────────────────────
    N_CHUNKS = 3
    chunks = [
        (torch.randn(B, 1496, 1024), torch.randn(B, 750, 1024))
        for _ in range(N_CHUNKS)
    ]
    out_multi = block3.forward_multi_chunk(chunks)
    print(f"Multi-chunk ({N_CHUNKS} chunks):")
    print(f"  h_full : {out_multi.h_full.shape}")        # [2, 2250, 1024]
    print()

    # ── Qwen-2.5-0.5B projection (outside Block 3) ───────────────────
    QWEN_DIM = 896
    qwen_proj = nn.Linear(1024, QWEN_DIM)
    h_qwen = qwen_proj(out.h_full)
    print(f"After Qwen projection (hidden_dim={QWEN_DIM}):")
    print(f"  h_qwen : {h_qwen.shape}")                  # [2, 750, 896]

    # ── Param count ───────────────────────────────────────────────────
    print(f"\nTotal params    : {block3.total_params:,}")
    print(f"Trainable params: {block3.trainable_params:,}")