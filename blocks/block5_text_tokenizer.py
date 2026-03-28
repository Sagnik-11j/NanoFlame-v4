"""
blocks/block5_text_tokenizer.py
══════════════════════════════════════════════════════════════════════
Block 5 · Text Tokenizer

Thin wrapper around the Qwen-2.5-0.5B-Instruct BPE tokenizer that:
  - Loads the tokenizer from HuggingFace (or a local cache)
  - Adds custom audio boundary tokens: <audio1>..</audio4> + </audio1>..</audio4>
  - Provides training-stage-aware target builders
  - Builds Qwen-2.5 chat-format prompts for Block 6

Vocab after extension: 151,665 + 8 audio tokens = 151,673

No neural network. No GPU required. Pure HuggingFace tokenizers.
══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from transformers import AutoTokenizer


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

MODEL_ID        = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_VOCAB_SIZE = 151_665
MAX_AUDIOS      = 4          # pipeline supports 1–4 audio files

# Custom audio boundary tokens added on top of Qwen-2.5 vocabulary
AUDIO_TOKENS: List[str] = [
    f"<audio{i}>"  for i in range(1, MAX_AUDIOS + 1)
] + [
    f"</audio{i}>" for i in range(1, MAX_AUDIOS + 1)
]

# Qwen-2.5 chat special tokens
SYSTEM_START = "<|im_start|>system"
USER_START   = "<|im_start|>user"
ASST_START   = "<|im_start|>assistant"
TURN_END     = "<|im_end|>"

# Default system prompt
SYSTEM_PROMPT = (
    "You are an audio AI that analyzes sound carefully before answering."
)

# CoT trigger suffix for Stage 4
COT_TRIGGER = "Think step by step before answering."

# Think block delimiters (Stage 4)
THINK_OPEN  = "<think>"
THINK_CLOSE = "</think>"


# ─────────────────────────────────────────────────────────────────────
# Output dataclasses
# ─────────────────────────────────────────────────────────────────────

@dataclass
class TokenizerOutput:
    """Result of Block5TextTokenizer.encode()."""
    input_ids:      List[int]
    attention_mask: List[int]
    num_tokens:     int
    text:           str         # original text before tokenisation


@dataclass
class TrainingTarget:
    """
    Full sequence + loss mask for one training example.

    loss_mask: 1 where loss is computed (answer span or <think>+answer span),
               0 everywhere else (prompt, padding).
    """
    input_ids:  List[int]
    loss_mask:  List[int]
    stage:      int
    num_tokens: int


@dataclass
class ChatPrompt:
    """
    Qwen-2.5 chat-format prompt (text only, no audio tensors).

    audio_placeholders: dict mapping audio_idx → (open_token_id, close_token_id)
    Block 6 replaces the placeholder span with real audio_tokens from Block 4.
    """
    text:               str
    input_ids:          List[int]
    attention_mask:     List[int]
    num_tokens:         int
    audio_placeholders: Dict[int, tuple]   # {1: (open_id, close_id), ...}
    has_cot_trigger:    bool


# ─────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────

class Block5TextTokenizer:
    """
    Block 5 — Text Tokenizer for NanoFlame v4.

    Usage:
        tok = Block5TextTokenizer()

        # Encode a plain question
        out = tok.encode("What emotion does the speaker convey?")
        # out.input_ids → [2515, 14230, 1341, ...]

        # Build a full chat prompt with audio placeholders
        prompt = tok.build_chat_prompt(
            question="What emotion does the speaker convey?",
            n_audios=2,
        )

        # Build a training target (Stage 1-3: answer tokens only)
        target = tok.build_training_target(
            prompt_ids=prompt.input_ids,
            answer="The speaker conveys excitement.",
            stage=1,
        )

        # Build a training target (Stage 4: <think>...</think> + answer)
        target4 = tok.build_training_target(
            prompt_ids=prompt.input_ids,
            answer="The speaker conveys excitement.",
            stage=4,
            think_text="Rising F0 and rapid speech rate suggest excitement.",
        )
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        local_dir: Optional[str] = None,
        max_length: int = 4096,
    ) -> None:
        self.max_length = max_length
        src = local_dir if local_dir else model_id

        self._tok = AutoTokenizer.from_pretrained(
            src,
            trust_remote_code=True,
            padding_side="right",
        )
        self._add_audio_tokens()

    # ── Private helpers ───────────────────────────────────────────────

    def _add_audio_tokens(self) -> None:
        """Add <audioN> / </audioN> tokens to the vocabulary."""
        new = [t for t in AUDIO_TOKENS if t not in self._tok.get_vocab()]
        if new:
            self._tok.add_special_tokens({"additional_special_tokens": new})

    def _ids(self, text: str) -> List[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def _token_id(self, token: str) -> int:
        return self._tok.convert_tokens_to_ids(token)

    # ── Public properties ─────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self._tok)

    @property
    def base_vocab_size(self) -> int:
        return BASE_VOCAB_SIZE

    @property
    def pad_id(self) -> int:
        pid = self._tok.pad_token_id
        return pid if pid is not None else self._tok.eos_token_id

    @property
    def eos_id(self) -> int:
        return self._tok.eos_token_id

    @property
    def bos_id(self) -> int:
        return self._tok.bos_token_id

    def audio_open_id(self, audio_idx: int) -> int:
        """Token ID for <audioN>. audio_idx is 1-based."""
        if not isinstance(audio_idx, int) or not (1 <= audio_idx <= MAX_AUDIOS):
            raise ValueError(f"audio_idx must be between 1 and {MAX_AUDIOS}")
        return self._token_id(f"<audio{audio_idx}>")

    def audio_close_id(self, audio_idx: int) -> int:
        """Token ID for </audioN>. audio_idx is 1-based."""
        if not isinstance(audio_idx, int) or not (1 <= audio_idx <= MAX_AUDIOS):
            raise ValueError(f"audio_idx must be between 1 and {MAX_AUDIOS}")
        return self._token_id(f"</audio{audio_idx}>")

    # ── Core encode / decode ──────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        truncate: bool = True,
    ) -> TokenizerOutput:
        """
        Tokenize a raw text string.
        Returns TokenizerOutput with input_ids and attention_mask.
        """
        enc = self._tok(
            text,
            add_special_tokens=add_special_tokens,
            truncation=truncate,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        ids = enc["input_ids"]
        return TokenizerOutput(
            input_ids=ids,
            attention_mask=enc["attention_mask"],
            num_tokens=len(ids),
            text=text,
        )

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=skip_special_tokens)

    # ── Chat prompt builder ───────────────────────────────────────────

    def build_chat_prompt(
        self,
        question: str,
        n_audios: int = 1,
        system_prompt: str = SYSTEM_PROMPT,
        add_cot_trigger: bool = False,
    ) -> ChatPrompt:
        """
        Build a Qwen-2.5 chat-format prompt with audio boundary tokens.

        The returned text looks like:
            <|im_start|>system
            You are an audio AI ...
            <|im_end|>
            <|im_start|>user
            <audio1> </audio1>
            <audio2> </audio2>   ← if n_audios >= 2
            What emotion does the speaker convey?
            Think step by step before answering.   ← if add_cot_trigger
            <|im_end|>
            <|im_start|>assistant

        Block 6 will insert real audio_tokens [T×896] in place of the
        audio placeholder spans at inference/training time.
        """
        if not 1 <= n_audios <= MAX_AUDIOS:
            raise ValueError(f"n_audios must be 1–{MAX_AUDIOS}, got {n_audios}")

        lines = []
        lines.append(f"{SYSTEM_START}\n{system_prompt}\n{TURN_END}")

        user_content = []
        for i in range(1, n_audios + 1):
            user_content.append(f"<audio{i}> </audio{i}>")
        user_content.append(question)
        if add_cot_trigger:
            user_content.append(COT_TRIGGER)

        lines.append(f"{USER_START}\n" + "\n".join(user_content) + f"\n{TURN_END}")
        lines.append(f"{ASST_START}\n")

        text = "\n".join(lines)

        enc = self._tok(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        ids = enc["input_ids"]

        placeholders = {
            i: (self.audio_open_id(i), self.audio_close_id(i))
            for i in range(1, n_audios + 1)
        }

        return ChatPrompt(
            text=text,
            input_ids=ids,
            attention_mask=enc["attention_mask"],
            num_tokens=len(ids),
            audio_placeholders=placeholders,
            has_cot_trigger=add_cot_trigger,
        )

    # ── Training target builder ───────────────────────────────────────

    def build_training_target(
        self,
        prompt_ids: List[int],
        answer: str,
        stage: int,
        think_text: Optional[str] = None,
    ) -> TrainingTarget:
        """
        Construct the full token sequence and binary loss mask.

        Stage 1–3: loss computed only on answer tokens.
        Stage 4:   loss computed on <think>...</think> block AND answer tokens.

        The prompt tokens always have loss_mask = 0.
        The final <|im_end|> token is included in the answer span (loss = 1).

        Args:
            prompt_ids: token IDs from build_chat_prompt()
            answer:     ground-truth answer string (plain text)
            stage:      training stage 1–4
            think_text: required for stage 4, ignored otherwise

        Returns:
            TrainingTarget with input_ids [prompt + target] and loss_mask
        """
        # Validate stage parameter early
        if not isinstance(stage, int) or not (1 <= stage <= 4):
            raise ValueError("stage must be 1..4")

        if stage == 4:
            if not think_text:
                raise ValueError("stage=4 requires think_text")
            target_text = (
                f"{THINK_OPEN}{think_text}{THINK_CLOSE}"
                f"{answer}"
                f"{TURN_END}"
            )
        else:
            target_text = f"{answer}{TURN_END}"

        target_ids = self._ids(target_text)
        full_ids   = list(prompt_ids) + target_ids

        # Loss mask: 0 for prompt, 1 for target span
        loss_mask = [0] * len(prompt_ids) + [1] * len(target_ids)

        return TrainingTarget(
            input_ids=full_ids,
            loss_mask=loss_mask,
            stage=stage,
            num_tokens=len(full_ids),
        )

    # ── Utility ───────────────────────────────────────────────────────

    def count_answer_tokens(self, answer: str) -> int:
        return len(self._ids(answer))

    def count_think_tokens(self, think_text: str) -> int:
        wrapped = f"{THINK_OPEN}{think_text}{THINK_CLOSE}"
        return len(self._ids(wrapped))

    def audio_token_ids(self, audio_idx: int) -> tuple:
        """Return (open_id, close_id) for audio_idx (1-based)."""
        return self.audio_open_id(audio_idx), self.audio_close_id(audio_idx)

    def __repr__(self) -> str:
        return (
            f"Block5TextTokenizer("
            f"vocab_size={self.vocab_size}, "
            f"base={self.base_vocab_size}, "
            f"audio_tokens={len(AUDIO_TOKENS)}, "
            f"max_length={self.max_length})"
        )


if __name__ == "__main__":
    tok = Block5TextTokenizer()
    print(tok)

    prompt = tok.build_chat_prompt(
        question="What emotion does the speaker convey? Compare with audio 2.",
        n_audios=2,
    )
    print(f"Prompt tokens : {prompt.num_tokens}")
    print(f"Audio placeholders: {prompt.audio_placeholders}")

    t = tok.build_training_target(
        prompt_ids=prompt.input_ids,
        answer="Audio 1 conveys excitement. Audio 2 conveys sadness.",
        stage=1,
    )
    print(f"Stage-1 target: {t.num_tokens} tokens, "
          f"loss on {sum(t.loss_mask)} tokens")

    t4 = tok.build_training_target(
        prompt_ids=prompt.input_ids,
        answer="Audio 1 conveys excitement. Audio 2 conveys sadness.",
        stage=4,
        think_text="Rising F0 and rapid speech in audio 1 → excitement. "
                   "Flat pitch, slow pace in audio 2 → sadness.",
    )
    print(f"Stage-4 target: {t4.num_tokens} tokens, "
          f"loss on {sum(t4.loss_mask)} tokens")