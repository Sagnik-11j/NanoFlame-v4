"""
blocks/block6_sequence_packer.py
══════════════════════════════════════════════════════════════════════
Block 6 · Sequence Packer

Receives:
  Block 4 → audio_tokens  List[ Tensor[T_i, 896] ]   (one per audio file)
  Block 5 → ChatPrompt    (input_ids, audio_placeholders, ...)
            TrainingTarget (input_ids, loss_mask, stage)  ← training only

Produces:
  PackedSequence
    .inputs_embeds  [1, total_seq_len, 896]  ← fed directly to Block 7
    .attention_mask [1, total_seq_len]
    .loss_mask      [1, total_seq_len]       ← 1 only on answer/CoT span
    .prompt_len, .target_len, .total_len
    .audio_spans    List[(start, end)]       ← injected audio positions

How injection works
───────────────────
The prompt text contains audio boundary tokens:
    ... <audio1> </audio1> ...  (from Block 5)

Block 6 splits the prompt_ids at those boundary tokens, embeds each
text segment with embed_tokens(), then splices in the real float audio
tensors between the open/close boundary embeddings:

    embed(<audio1>)  →  audio_tokens_1 [T, 896]  →  embed(</audio1>)

Text tokens  → embed_tokens() → 896-dim vectors
Audio tokens → injected directly (already 896-dim from Block 4)
Loss         → computed ONLY on assistant span (target_len tokens)
══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from blocks.block5_text_tokenizer import (
        Block5TextTokenizer, ChatPrompt, TrainingTarget
    )

HIDDEN_DIM = 896


# ─────────────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PackedSequence:
    """
    Ready-to-feed sequence for Block 7 (Qwen-2.5 decoder).

    inputs_embeds  [1, total_len, 896]  — text via embed_tokens,
                                          audio injected directly
    attention_mask [1, total_len]       — all 1s (no padding)
    loss_mask      [1, total_len]       — 1 on answer/CoT span, 0 elsewhere
    prompt_len     int                  — tokens before assistant turn
    target_len     int                  — tokens in answer/CoT span
    total_len      int                  — prompt_len + target_len
    audio_spans    List[(start, end)]   — inclusive positions of audio blobs
    n_audios       int
    stage          int | None           — training stage or None (inference)
    """
    inputs_embeds:  torch.Tensor
    attention_mask: torch.Tensor
    loss_mask:      torch.Tensor
    prompt_len:     int
    target_len:     int
    total_len:      int
    audio_spans:    List[Tuple[int, int]]
    n_audios:       int
    stage:          Optional[int]

    def __repr__(self) -> str:
        return (
            f"PackedSequence("
            f"shape={tuple(self.inputs_embeds.shape)}, "
            f"prompt={self.prompt_len}, "
            f"target={self.target_len}, "
            f"audios={self.n_audios}, "
            f"stage={self.stage})"
        )


# ─────────────────────────────────────────────────────────────────────
# Main block
# ─────────────────────────────────────────────────────────────────────

class Block6SequencePacker:
    """
    Block 6 — Sequence Packer for NanoFlame v4.

    Parameters
    ----------
    tokenizer : Block5TextTokenizer
        Used to look up audio boundary token IDs.
    embed_tokens : nn.Module
        Qwen-2.5-0.5B embed_tokens layer (vocab_size × 896).
        Must already be resized to include Block 5's audio boundary tokens.

    Usage
    -----
    packer = Block6SequencePacker(tokenizer, model.model.embed_tokens)

    # Inference
    packed = packer.pack_for_inference(prompt, audio_tokens_list)

    # Training
    packed = packer.pack_for_training(prompt, audio_tokens_list, target)
    """

    def __init__(
        self,
        tokenizer: "Block5TextTokenizer",
        embed_tokens: nn.Module,
    ) -> None:
        self.tokenizer   = tokenizer
        self.embed_tokens = embed_tokens

    # ── Core injection helper ─────────────────────────────────────────

    def _inject(
        self,
        prompt_ids: List[int],
        audio_tokens_list: List[torch.Tensor],
        device: str,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Build inputs_embeds for the prompt portion by:
          1. Finding <audioN> / </audioN> boundary pairs in prompt_ids
          2. Embedding text segments with embed_tokens
          3. Splicing real audio tensors between boundary embeddings

        Returns
        -------
        prompt_embeds : Tensor [prompt_seq_len, 896]
        audio_spans   : List[(start_pos, end_pos)]  in the final sequence
        """
        # Locate all audio spans in prompt_ids
        spans: List[Tuple[int, int, int]] = []   # (open_i, close_i, audio_idx)
        for audio_idx in range(1, len(audio_tokens_list) + 1):
            open_id  = self.tokenizer.audio_open_id(audio_idx)
            close_id = self.tokenizer.audio_close_id(audio_idx)
            try:
                open_i  = prompt_ids.index(open_id)
                close_i = prompt_ids.index(close_id)
                spans.append((open_i, close_i, audio_idx))
            except ValueError:
                pass   # audio boundary tokens not found — skip

        spans.sort(key=lambda x: x[0])

        segments: List[torch.Tensor] = []
        audio_spans_out: List[Tuple[int, int]] = []
        pos   = 0
        prev  = 0   # where we left off in prompt_ids

        for (open_i, close_i, audio_idx) in spans:
            # ── embed text before this audio span ──────────────────
            if open_i > prev:
                ids  = torch.tensor(prompt_ids[prev:open_i],
                                    dtype=torch.long, device=device)
                emb  = self.embed_tokens(ids)             # [n, 896]
                segments.append(emb)
                pos += emb.size(0)

            # ── embed open boundary token ──────────────────────────
            open_emb = self.embed_tokens(
                torch.tensor([prompt_ids[open_i]],
                             dtype=torch.long, device=device)
            )                                              # [1, 896]
            segments.append(open_emb)
            pos += 1

            # ── inject audio tensor ────────────────────────────────
            audio_t = audio_tokens_list[audio_idx - 1]   # [T, 896]
            if audio_t.dim() == 3:                        # [1, T, 896] → squeeze
                audio_t = audio_t.squeeze(0)
            audio_t = audio_t.to(device=device,
                                  dtype=self.embed_tokens.weight.dtype)
            span_start = pos
            segments.append(audio_t)
            pos += audio_t.size(0)
            span_end = pos
            audio_spans_out.append((span_start, span_end))

            # ── embed close boundary token ─────────────────────────
            close_emb = self.embed_tokens(
                torch.tensor([prompt_ids[close_i]],
                             dtype=torch.long, device=device)
            )                                              # [1, 896]
            segments.append(close_emb)
            pos += 1

            prev = close_i + 1

        # ── embed remaining text tokens after last audio span ───────
        if prev < len(prompt_ids):
            ids  = torch.tensor(prompt_ids[prev:],
                                dtype=torch.long, device=device)
            emb  = self.embed_tokens(ids)
            segments.append(emb)
            pos += emb.size(0)

        prompt_embeds = torch.cat(segments, dim=0)        # [prompt_seq_len, 896]
        return prompt_embeds, audio_spans_out

    # ── Public API ────────────────────────────────────────────────────

    def pack_for_inference(
        self,
        prompt: "ChatPrompt",
        audio_tokens_list: List[torch.Tensor],
        device: str = "cpu",
    ) -> PackedSequence:
        """
        Build the packed sequence for inference (no loss mask needed).

        The model will generate tokens starting after the last prompt token.
        """
        if not audio_tokens_list:
            raise ValueError("audio_tokens_list must contain at least one tensor")

        prompt_embeds, audio_spans = self._inject(
            prompt.input_ids, audio_tokens_list, device
        )
        prompt_len = prompt_embeds.size(0)

        inputs_embeds  = prompt_embeds.unsqueeze(0)         # [1, L, 896]
        attention_mask = torch.ones(1, prompt_len,
                                    dtype=torch.long, device=device)
        loss_mask      = torch.zeros(1, prompt_len,
                                     dtype=torch.long, device=device)

        return PackedSequence(
            inputs_embeds  = inputs_embeds,
            attention_mask = attention_mask,
            loss_mask      = loss_mask,
            prompt_len     = prompt_len,
            target_len     = 0,
            total_len      = prompt_len,
            audio_spans    = audio_spans,
            n_audios       = len(audio_tokens_list),
            stage          = None,
        )

    def pack_for_training(
        self,
        prompt: "ChatPrompt",
        audio_tokens_list: List[torch.Tensor],
        training_target: "TrainingTarget",
        device: str = "cpu",
    ) -> PackedSequence:
        """
        Build the packed sequence for training.

        The training_target (from Block 5) carries:
          - input_ids  = prompt_ids + answer_ids
          - loss_mask  = [0]*prompt_len + [1]*answer_len

        Block 6 re-builds inputs_embeds with audio injection and re-aligns
        the loss mask to match the expanded sequence length.

        Loss is computed ONLY on the answer/CoT span (loss_mask == 1).
        Audio tokens and all prompt tokens have loss_mask == 0.
        """
        if not audio_tokens_list:
            raise ValueError("audio_tokens_list must contain at least one tensor")

        # ── Determine prompt / target split from training_target ────
        prompt_id_len = len(prompt.input_ids)
        target_ids    = training_target.input_ids[prompt_id_len:]
        target_len    = len(target_ids)

        # ── Build prompt embeds with audio injection ────────────────
        prompt_embeds, audio_spans = self._inject(
            prompt.input_ids, audio_tokens_list, device
        )
        actual_prompt_len = prompt_embeds.size(0)

        # ── Embed target tokens (text only — no audio here) ─────────
        if target_len > 0:
            target_id_tensor = torch.tensor(
                target_ids, dtype=torch.long, device=device
            )
            target_embeds = self.embed_tokens(target_id_tensor)  # [target_len, 896]
        else:
            target_embeds = torch.zeros(
                0, HIDDEN_DIM,
                dtype=prompt_embeds.dtype, device=device
            )

        # ── Concatenate ─────────────────────────────────────────────
        full_embeds = torch.cat([prompt_embeds, target_embeds], dim=0)  # [total, 896]
        total_len   = full_embeds.size(0)

        # ── Build loss mask ─────────────────────────────────────────
        # Prompt portion (including audio tokens): loss = 0
        # Target portion (answer / CoT): loss = 1
        loss_mask_1d = torch.cat([
            torch.zeros(actual_prompt_len, dtype=torch.long, device=device),
            torch.ones(target_len,         dtype=torch.long, device=device),
        ])                                                   # [total_len]

        inputs_embeds  = full_embeds.unsqueeze(0)            # [1, total, 896]
        attention_mask = torch.ones(1, total_len,
                                    dtype=torch.long, device=device)
        loss_mask_2d   = loss_mask_1d.unsqueeze(0)           # [1, total_len]

        return PackedSequence(
            inputs_embeds  = inputs_embeds,
            attention_mask = attention_mask,
            loss_mask      = loss_mask_2d,
            prompt_len     = actual_prompt_len,
            target_len     = target_len,
            total_len      = total_len,
            audio_spans    = audio_spans,
            n_audios       = len(audio_tokens_list),
            stage          = training_target.stage,
        )

    # ── Convenience wrapper ───────────────────────────────────────────

    def pack(
        self,
        prompt: "ChatPrompt",
        audio_tokens_list: List[torch.Tensor],
        training_target: Optional["TrainingTarget"] = None,
        device: str = "cpu",
    ) -> PackedSequence:
        """
        Dispatch to pack_for_inference or pack_for_training based on
        whether training_target is provided.
        """
        if training_target is not None:
            return self.pack_for_training(
                prompt, audio_tokens_list, training_target, device
            )
        return self.pack_for_inference(prompt, audio_tokens_list, device)


if __name__ == "__main__":
    import torch.nn as nn
    from blocks.block5_text_tokenizer import Block5TextTokenizer

    tok  = Block5TextTokenizer()
    emb  = nn.Embedding(tok.vocab_size, HIDDEN_DIM)      # mock embed_tokens
    packer = Block6SequencePacker(tok, emb)

    T = 750
    audio_tokens = [torch.randn(T, HIDDEN_DIM)]           # mock Block 4 output

    prompt = tok.build_chat_prompt(
        "What emotion does the speaker convey?", n_audios=1
    )

    # Inference
    packed_inf = packer.pack_for_inference(prompt, audio_tokens)
    print("INFERENCE:", packed_inf)

    # Training Stage 1
    target = tok.build_training_target(
        prompt.input_ids, "The speaker conveys excitement.", stage=1
    )
    packed_tr = packer.pack_for_training(prompt, audio_tokens, target)
    print("TRAINING :", packed_tr)
    print(f"  loss tokens : {packed_tr.loss_mask.sum().item()}")
    print(f"  audio span  : {packed_tr.audio_spans}")