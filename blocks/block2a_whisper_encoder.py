# blocks/block2a_whisper_encoder.py
# ─────────────────────────────────────────────────────────────────────────────
# Block 2a — Whisper-Medium Encoder (speech specialist)
#
# Input:  mel chunks  [B, 128, 3000]  (from Block 1 .stack())
#         or single chunk  [128, 3000]
# Output: h_w         [B, 750, 1024]
#         or            [750, 1024]
#
# Pipeline inside:
#   Conv1d(128→1024, k=3, s=1) → Conv1d(1024→1024, k=3, s=2)
#   → Sinusoidal pos-emb
#   → 24× Transformer encoder blocks (16-head MHSA, FFN 1024→4096→1024)
#   → [B, 1500, 1024] @ 50 Hz
#   → AvgPool1d(k=2, s=2)
#   → h_w [B, 750, 1024] @ 25 Hz
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import WhisperModel

# ─────────────────────────────────────────────────────────────────────────────
# Output contract
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WhisperEncoderOutput:
    """
    Output of Block 2a.

    h_w        : Tensor [B, 750, 1024] or [750, 1024] for single-chunk input.
                 Represents Whisper speech features @ 25 Hz, 1024-dim.
    num_tokens : Always 750 (750 time-steps per 30s chunk at 25 Hz).
    num_chunks : Number of chunks processed in this forward pass (= B).
    """
    h_w:        torch.Tensor
    num_tokens: int
    num_chunks: int


# ─────────────────────────────────────────────────────────────────────────────
# Block
# ─────────────────────────────────────────────────────────────────────────────

class Block2aWhisperEncoder(nn.Module):
    """
    Whisper-Medium encoder adapted for 128-bin mel input.

    Stage 1 (adaptor-only): all encoder weights are frozen.
    Stage 2+: LoRA r=16 is injected via .enable_lora(), base weights stay frozen.

    Usage:
        enc = Block2aWhisperEncoder()
        enc.freeze_base_weights()    # Stage 1 — nothing trains here
        # --- later for Stage 2 ---
        enc.enable_lora()            # injects LoRA into Q/K/V/O
    """

    HIDDEN_DIM = 1024
    OUT_TOKENS = 750   # 1500 encoder tokens → 750 after stride-2 pool

    def __init__(
        self,
        model_name: str = "openai/whisper-medium",
        n_mels_in:  int = 128,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.n_mels_in     = n_mels_in
        self.device        = device or torch.device("cpu")
        self._lora_enabled = False

        # ── Load Whisper, keep encoder only ──────────────────────────────────
        whisper       = WhisperModel.from_pretrained(model_name)
        self.encoder  = whisper.encoder
        del whisper   # free decoder weights immediately

        # ── Replace conv1: Conv1d(80→1024) → Conv1d(128→1024) ────────────────
        self._replace_conv1(n_mels_in)

        # ── Stride-2 temporal pool: [B, 1024, 1500] → [B, 1024, 750] ─────────
        self.temporal_pool = nn.AvgPool1d(kernel_size=2, stride=2)

    # ─────────────────────────────────────────────────────────────────────────
    # Architecture setup
    # ─────────────────────────────────────────────────────────────────────────

    def _replace_conv1(self, n_mels_in: int) -> None:
        """
        Swap conv1 to accept n_mels_in input channels instead of 80.

        Weights for channels 0:80 are copied from the pretrained checkpoint.
        Weights for channels 80:128 are Xavier-initialized.
        Bias is copied verbatim from the pretrained checkpoint.
        """
        old = self.encoder.conv1  # Conv1d(80, 1024, k=3, p=1)

        new = nn.Conv1d(
            in_channels  = n_mels_in,
            out_channels = old.out_channels,   # 1024
            kernel_size  = old.kernel_size[0], # 3
            padding      = old.padding[0],     # 1
            bias         = old.bias is not None,
        )

        with torch.no_grad():
            nn.init.xavier_uniform_(new.weight)
            if new.bias is not None:
                nn.init.zeros_(new.bias)
                new.bias.data.copy_(old.bias.data)  # preserve pretrained bias

            # Copy pretrained weights for the original 80 input channels
            n_copy = min(n_mels_in, old.in_channels)
            new.weight.data[:, :n_copy, :].copy_(
                old.weight.data[:, :n_copy, :]
            )

        self.encoder.conv1 = new

    # ─────────────────────────────────────────────────────────────────────────
    # LoRA / freezing
    # ─────────────────────────────────────────────────────────────────────────

    def freeze_base_weights(self) -> None:
        """
        Freeze all non-LoRA parameters.
        Call this in Stage 1 (and keep frozen in Stages 2–4).
        """
        for name, param in self.named_parameters():
            if "lora_" not in name:
                param.requires_grad_(False)

    def enable_lora(
        self,
        r:            int   = 16,
        lora_alpha:   int   = 32,
        lora_dropout: float = 0.1,
    ) -> None:
        """
        Inject LoRA adapters into Q / K / V / O projections.
        Call once at the start of Stage 2. Idempotent.

        After this call, only lora_A and lora_B parameters have
        requires_grad=True — base weights remain frozen.
        """
        from peft import LoraConfig, get_peft_model

        if self._lora_enabled:
            return

        config = LoraConfig(
            r              = r,
            lora_alpha     = lora_alpha,
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout   = lora_dropout,
            bias           = "none",
        )
        self.encoder       = get_peft_model(self.encoder, config)
        self._lora_enabled = True

    # ─────────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, chunks: torch.Tensor) -> WhisperEncoderOutput:
        """
        Args:
            chunks: [B, N_MELS, CHUNK_FRAMES]  (batched)
                    or  [N_MELS, CHUNK_FRAMES]  (single chunk — auto-batched)

        Returns:
            WhisperEncoderOutput
                .h_w  → [B, 750, 1024]  or  [750, 1024]  float32
        """
        squeeze = (chunks.ndim == 2)
        if squeeze:
            chunks = chunks.unsqueeze(0)                      # [1, 128, 3000]

        B = chunks.shape[0]
        chunks = chunks.to(dtype=torch.float32, device=self.device)

        # HuggingFace Whisper encoder:
        # in  → [B, 128, 3000]
        # out → last_hidden_state [B, 1500, 1024]
        enc_out = self.encoder(input_features=chunks)
        hidden  = enc_out.last_hidden_state                   # [B, 1500, 1024]

        # Stride-2 temporal pooling:
        # AvgPool1d expects [B, C, L]
        hidden = hidden.transpose(1, 2)                       # [B, 1024, 1500]
        hidden = self.temporal_pool(hidden)                   # [B, 1024, 750]
        hidden = hidden.transpose(1, 2)                       # [B, 750, 1024]

        if squeeze:
            hidden = hidden.squeeze(0)                        # [750, 1024]

        return WhisperEncoderOutput(
            h_w        = hidden,
            num_tokens = self.OUT_TOKENS,
            num_chunks = B,
        )