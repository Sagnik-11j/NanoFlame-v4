"""
blocks/block7_qwen_decoder.py
══════════════════════════════════════════════════════════════════════
Block 7 · Qwen-2.5-0.5B-Instruct  (4-bit NF4 + QLoRA)
══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

MODEL_ID       = "Qwen/Qwen2.5-0.5B-Instruct"
HIDDEN_DIM     = 896
NUM_LAYERS     = 24
NUM_Q_HEADS    = 14
NUM_KV_HEADS   = 2
FFN_INNER      = 4864
BASE_VOCAB     = 151_936
AUDIO_TOKENS   = 8
EXTENDED_VOCAB = BASE_VOCAB + AUDIO_TOKENS   # 151,944

LORA_R         = 16
LORA_ALPHA     = 32
LORA_TARGETS   = ["q_proj", "k_proj", "v_proj", "o_proj"]


@dataclass
class Block7Output:
    logits: torch.Tensor
    loss:   Optional[torch.Tensor]
    hidden: torch.Tensor


class Block7QwenDecoder(nn.Module):
    """
    Block 7 — Qwen-2.5-0.5B-Instruct (4-bit NF4 + QLoRA).

    Accepts inputs_embeds directly from Block 6 (audio already injected).
    Base weights frozen in NF4; only QLoRA adapters train.
    embed_tokens resized +8 rows for audio boundary tokens from Block 5.
    lm_head weight is EXPLICITLY UNTIED from embed_tokens after resize so
    Stage 1 can train both independently via named_parameters().
    """

    def __init__(
        self,
        model_id:            str   = MODEL_ID,
        extended_vocab_size: int   = EXTENDED_VOCAB,
        lora_r:              int   = LORA_R,
        lora_alpha:          int   = LORA_ALPHA,
        lora_dropout:        float = 0.05,
        use_4bit:            bool  = True,
        use_flash_attn:      bool  = True,
        local_dir:           Optional[str] = None,
    ) -> None:
        super().__init__()

        self.model_id            = model_id
        self.extended_vocab_size = extended_vocab_size
        self.lora_r              = lora_r
        self.lora_alpha          = lora_alpha
        self.lora_dropout        = lora_dropout
        self.use_4bit            = use_4bit
        self._lora_applied       = False
        self._peft_wrapped       = False

        src = local_dir if local_dir else model_id
        self._model = self._load(src, use_4bit, use_flash_attn)
        self._resize_embedding()   # resize → init new rows → untie lm_head
        self._inject_lora()

    # ── Load ─────────────────────────────────────────────────────────

    def _load(self, src: str, use_4bit: bool, use_flash_attn: bool):
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        attn_impl = None
        if use_flash_attn:
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
                logger.info("Flash Attention 2 enabled")
            except ImportError:
                logger.warning("Flash Attention 2 not installed — using SDPA")

        bnb_cfg = None
        if use_4bit:
            try:
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                logger.info("4-bit NF4 enabled (~0.25 GB VRAM)")
            except Exception:
                logger.warning("bitsandbytes not available — loading fp32")
                bnb_cfg = None

        kwargs: dict = dict(trust_remote_code=True, dtype=torch.float16)
        if bnb_cfg is not None:
            kwargs["quantization_config"] = bnb_cfg
            kwargs["device_map"]          = "auto"
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl

        model = AutoModelForCausalLM.from_pretrained(src, **kwargs)
        model.config.use_cache = False
        return model

    # ── Resize embedding ─────────────────────────────────────────────

    def _resize_embedding(self) -> None:
        emb     = self._get_embed_tokens()
        current = emb.weight.shape[0]

        if current >= self.extended_vocab_size:
            logger.info(f"Embedding already {current} rows — skipping resize")
            # Still untie in case the base checkpoint has tied weights
            self._untie_lm_head()
            return

        n_new = self.extended_vocab_size - current
        old_w = emb.weight.data.float()
        init  = old_w.mean(dim=0, keepdim=True).repeat(n_new, 1)

        self._model.resize_token_embeddings(self.extended_vocab_size)

        with torch.no_grad():
            new_emb = self._get_embed_tokens()
            new_emb.weight.data[current:] = init.to(new_emb.weight.dtype)

        logger.info(
            f"Embedding resized: {current} → {self.extended_vocab_size} "
            f"(+{n_new} audio boundary tokens)"
        )

        # Qwen2 uses tie_word_embeddings=True, so after resize HuggingFace
        # re-ties lm_head.weight to embed_tokens.weight (same tensor).
        # Untie explicitly so Stage 1 can train them independently and
        # "lm_head" appears as a distinct entry in named_parameters().
        self._untie_lm_head()

    def _untie_lm_head(self) -> None:
        """
        Give lm_head its own independent nn.Parameter, breaking the weight
        tie with embed_tokens.  This is a no-op if weights are already
        independent (data_ptr differs).
        """
        try:
            emb = self._get_embed_tokens()
            lm  = self._get_raw_lm_head()   # must be called before PEFT wrap

            if lm is None:
                logger.warning("_untie_lm_head: lm_head not found, skipping")
                return

            emb_ptr = emb.weight.data_ptr()
            lm_ptr  = lm.weight.data_ptr()

            if emb_ptr != lm_ptr:
                logger.info("lm_head weight already independent — no untie needed")
                return

            # Clone into a fresh Parameter with the same data & dtype/device
            lm.weight = nn.Parameter(
                lm.weight.detach().clone(),
                requires_grad=False,   # frozen by default; stage control unfreezes
            )
            logger.info(
                "lm_head weight untied from embed_tokens "
                "(Qwen2 tie_word_embeddings=True workaround)"
            )
        except Exception as exc:
            logger.warning(f"_untie_lm_head: {exc} — lm_head may remain tied")

    # ── Inject LoRA ───────────────────────────────────────────────────

    def _inject_lora(self) -> None:
        try:
            from peft import (
                LoraConfig,
                TaskType,
                get_peft_model,
                prepare_model_for_kbit_training,
            )

            if self.use_4bit:
                self._model = prepare_model_for_kbit_training(
                    self._model,
                    use_gradient_checkpointing=True,
                )

            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=LORA_TARGETS,
                bias="none",
            )
            self._model = get_peft_model(self._model, lora_cfg)
            self._lora_applied = True
            self._peft_wrapped = True
            logger.info(f"QLoRA injected: r={self.lora_r}, α={self.lora_alpha}")
            self._model.print_trainable_parameters()

        except ImportError:
            logger.warning("PEFT not installed — QLoRA skipped (pip install peft)")

    # ── Helpers to navigate PEFT wrapper ─────────────────────────────

    def _get_embed_tokens(self) -> nn.Embedding:
        """Return embed_tokens regardless of PEFT wrapping depth."""
        for path in [
            lambda m: m.model.embed_tokens,
            lambda m: m.base_model.model.model.embed_tokens,
            lambda m: m.base_model.model.embed_tokens,
        ]:
            try:
                emb = path(self._model)
                if isinstance(emb, nn.Embedding):
                    return emb
            except AttributeError:
                continue
        raise AttributeError("Could not locate embed_tokens in model hierarchy")

    def _get_raw_lm_head(self) -> Optional[nn.Linear]:
        """
        Return the raw lm_head nn.Linear (or None if not found / not Linear).
        Used BEFORE and AFTER PEFT wrapping — works via attribute proxy.
        Does NOT raise; returns None on failure so callers can handle gracefully.
        """
        for path in [
            lambda m: m.lm_head,
            lambda m: m.base_model.model.lm_head,
            lambda m: m.model.lm_head,
        ]:
            try:
                lm = path(self._model)
                if isinstance(lm, nn.Linear):
                    return lm
            except AttributeError:
                continue
        return None

    def _get_lm_head(self) -> nn.Linear:
        """Like _get_raw_lm_head but raises AttributeError on failure."""
        lm = self._get_raw_lm_head()
        if lm is None:
            raise AttributeError("Could not locate lm_head in model hierarchy")
        return lm

    # ── Stage-aware parameter control ────────────────────────────────

    def configure_for_stage(self, stage: int) -> None:
        """
        Stage 1 : LM head + new audio embed rows only
        Stage 2 : QLoRA adapters + audio embed rows
        Stage 3+: QLoRA adapters only
        """
        for p in self._model.parameters():
            p.requires_grad = False

        if stage == 1:
            self._unfreeze_lm_head()
            self._unfreeze_audio_embed_rows()
            logger.info("Stage 1: LM head + audio embed rows trainable")
        elif stage in (2, 3, 4):
            self._unfreeze_lora()
            if stage == 2:
                self._unfreeze_audio_embed_rows()
            logger.info(f"Stage {stage}: QLoRA adapters trainable")

    def _unfreeze_lora(self) -> None:
        for name, p in self._model.named_parameters():
            if "lora_" in name:
                p.requires_grad = True

    def _unfreeze_lm_head(self) -> None:
        """
        Unfreeze the lm_head weight.
        After _untie_lm_head() this is always an independent Parameter, so
        it appears under a "lm_head" key in named_parameters().
        Primary path: iterate named_parameters (catches all PEFT prefix forms).
        Fallback: direct object traversal.
        """
        found = False

        # Primary: iterate named_parameters so we handle any PEFT prefix
        for name, p in self._model.named_parameters():
            if "lm_head" in name:
                p.requires_grad = True
                found = True

        if not found:
            # Fallback: get the module object directly and unfreeze its params
            lm = self._get_raw_lm_head()
            if lm is not None:
                for p in lm.parameters():
                    p.requires_grad = True
                found = True
                logger.warning(
                    "_unfreeze_lm_head: used object fallback — "
                    "'lm_head' may not appear in named_parameters()"
                )

        if not found:
            logger.warning(
                "_unfreeze_lm_head: lm_head not found at all — "
                "ensure _untie_lm_head() ran before PEFT wrapping"
            )

    def _unfreeze_audio_embed_rows(self) -> None:
        """
        Make only the new audio-boundary token rows trainable.
        A gradient hook zeroes gradients for all base-vocab rows so
        pre-trained token embeddings are not disturbed.
        """
        emb = self._get_embed_tokens()
        if emb.weight.requires_grad:
            return

        def _zero_base_grads(grad: torch.Tensor) -> torch.Tensor:
            grad[:BASE_VOCAB].zero_()
            return grad

        emb.weight.requires_grad = True
        emb.weight.register_hook(_zero_base_grads)

    def freeze_all(self) -> None:
        for p in self._model.parameters():
            p.requires_grad = False

    def unfreeze_lora(self) -> None:
        self._unfreeze_lora()

    # ── Properties ───────────────────────────────────────────────────

    @property
    def embed_tokens(self) -> nn.Embedding:
        """Exposed for Block 6 to use without loading Block 7 twice."""
        return self._get_embed_tokens()

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self._model.parameters())

    # ── Forward ──────────────────────────────────────────────────────

    def forward(
        self,
        packed_sequence=None,
        inputs_embeds:  Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        loss_mask:      Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache:      bool = False,
    ) -> Block7Output:
        """
        Forward pass.  Accepts either a Block6 PackedSequence or raw tensors.
        Loss is computed only on positions where loss_mask == 1.
        """
        if packed_sequence is not None:
            inputs_embeds  = packed_sequence.inputs_embeds
            attention_mask = packed_sequence.attention_mask
            loss_mask      = getattr(packed_sequence, "loss_mask", None)
            if loss_mask is not None and loss_mask.sum().item() == 0:
                loss_mask = None

        if inputs_embeds is None:
            raise ValueError(
                "Either packed_sequence or inputs_embeds must be provided"
            )

        self._model.config.use_cache = use_cache

        outputs = self._model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
            output_hidden_states=True,
        )

        logits = outputs.logits           # [B, T, V]
        hidden = outputs.hidden_states[-1]  # [B, T, 896]

        loss = None
        if loss_mask is not None and loss_mask.sum() > 0:
            if packed_sequence is not None and hasattr(
                packed_sequence, "_target_ids"
            ):
                target_ids   = packed_sequence._target_ids
                shift_logits = logits[:, :-1, :].contiguous()
                shift_mask   = loss_mask[:, 1:].contiguous()
                loss = self._masked_ce(shift_logits, target_ids, shift_mask)
            else:
                logger.warning(
                    "loss_mask provided but packed_sequence._target_ids "
                    "missing — set it to enable training loss."
                )

        return Block7Output(logits=logits, loss=loss, hidden=hidden)

    @staticmethod
    def _masked_ce(
        logits:  torch.Tensor,  # [B, T, V]
        targets: torch.Tensor,  # [B, T] int64
        mask:    torch.Tensor,  # [B, T]
    ) -> torch.Tensor:
        B, T, V      = logits.shape
        flat_logits  = flat_targets = flat_mask = None
        flat_logits  = logits.reshape(-1, V)
        flat_targets = targets.reshape(-1)
        flat_mask    = mask.reshape(-1).bool()
        return F.cross_entropy(
            flat_logits[flat_mask],
            flat_targets[flat_mask],
            reduction="mean",
        )

    # ── Inference helper ──────────────────────────────────────────────

    @torch.no_grad()
    def generate_step(
        self,
        inputs_embeds:  torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values=None,
        temperature:    float = 0.7,
        top_p:          float = 0.9,
    ) -> Tuple[int, object]:
        self._model.config.use_cache = True

        out = self._model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

        logits = out.logits[:, -1, :].float()

        if temperature > 0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)

        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum_probs                = torch.cumsum(sorted_probs, dim=-1)
        cutoff                   = (cum_probs - sorted_probs) > top_p
        sorted_probs[cutoff]     = 0.0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

        next_token    = torch.multinomial(sorted_probs, num_samples=1)
        next_token_id = sorted_idx.gather(-1, next_token).item()

        return int(next_token_id), out.past_key_values

    def __repr__(self) -> str:
        lora_status = f"QLoRA r={self.lora_r}" if self._lora_applied else "no LoRA"
        return (
            f"Block7QwenDecoder("
            f"model={self.model_id}, "
            f"vocab={self.extended_vocab_size}, "
            f"4bit={self.use_4bit}, "
            f"{lora_status}, "
            f"trainable={self.trainable_params:,}/{self.total_params:,})"
        )