"""
tests/test_block5.py
══════════════════════════════════════════════════════════════════════
NanoFlame v4 — Block 5 unit tests

Block 5 is a pure text processing block (no GPU needed).
All tests run in a few seconds on CPU.

Run:
    python tests/test_block5.py
    python tests/test_block5.py --model Qwen/Qwen2.5-0.5B-Instruct
    python tests/test_block5.py --local-dir ./qwen_local
"""

__test__ = False

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

G  = "\033[92m"; Y = "\033[93m"
R  = "\033[91m"; BD = "\033[1m"; E = "\033[0m"
ok   = lambda m: print(f"  {G}✔{E}  {m}")
warn = lambda m: print(f"  {Y}⚠{E}  {m}")
fail = lambda m: print(f"  {R}✘{E}  {m}")
info = lambda m: print(f"     {m}")
hdr  = lambda t: print(f"\n{BD}── {t} {'─'*(54-len(t))}{E}")

PASS = FAIL = 0


def check(name: str, ok_: bool, detail: str = "") -> bool:
    global PASS, FAIL
    if ok_:
        ok(f"{name}  {detail}")
        PASS += 1
    else:
        fail(f"{name}  {detail}")
        FAIL += 1
    return ok_


def get_tok(args):
    from blocks.block5_text_tokenizer import Block5TextTokenizer
    return Block5TextTokenizer(
        model_id=args.model,
        local_dir=args.local_dir if args.local_dir else None,
    )


# ─────────────────────────────────────────────────────────────────────
# T1 · Load + vocab
# ─────────────────────────────────────────────────────────────────────

def t1_load_and_vocab(args):
    hdr("T1 · Load tokenizer + vocab size")
    from blocks.block5_text_tokenizer import BASE_VOCAB_SIZE, AUDIO_TOKENS

    tok = get_tok(args)
    info(repr(tok))

    check("vocab_size > base",
          tok.vocab_size > tok.base_vocab_size,
          f"{tok.vocab_size} > {tok.base_vocab_size}")
    check("audio tokens added",
          tok.vocab_size == BASE_VOCAB_SIZE + len(AUDIO_TOKENS),
          f"+{tok.vocab_size - BASE_VOCAB_SIZE} tokens")
    check("eos_id is int", isinstance(tok.eos_id, int))
    check("pad_id is int", isinstance(tok.pad_id, int))

    return tok


# ─────────────────────────────────────────────────────────────────────
# T2 · Audio boundary tokens
# ─────────────────────────────────────────────────────────────────────

def t2_audio_tokens(tok):
    hdr("T2 · Audio boundary token IDs")

    for i in range(1, 5):
        open_id  = tok.audio_open_id(i)
        close_id = tok.audio_close_id(i)
        check(f"<audio{i}> id > base vocab",
              open_id > tok.base_vocab_size - 1,
              str(open_id))
        check(f"</audio{i}> id > base vocab",
              close_id > tok.base_vocab_size - 1,
              str(close_id))
        check(f"<audio{i}> ≠ </audio{i}>", open_id != close_id)

    check("audio_token_ids() returns tuple",
          isinstance(tok.audio_token_ids(1), tuple))


# ─────────────────────────────────────────────────────────────────────
# T3 · Encode / decode round-trip
# ─────────────────────────────────────────────────────────────────────

def t3_encode_decode(tok):
    hdr("T3 · Encode / decode round-trip")

    text = "What emotion does the speaker convey?"
    out  = tok.encode(text)

    check("returns TokenizerOutput", hasattr(out, "input_ids"))
    check("num_tokens > 0", out.num_tokens > 0, str(out.num_tokens))
    check("input_ids is list", isinstance(out.input_ids, list))
    check("attention_mask same length",
          len(out.attention_mask) == len(out.input_ids))
    check("all attention_mask = 1",
          all(v == 1 for v in out.attention_mask))

    decoded = tok.decode(out.input_ids)
    check("decode recovers original text", text in decoded, repr(decoded[:60]))


# ─────────────────────────────────────────────────────────────────────
# T4 · Chat prompt builder
# ─────────────────────────────────────────────────────────────────────

def t4_chat_prompt(tok):
    hdr("T4 · build_chat_prompt()")
    # FIX: removed unused THINK_OPEN import
    from blocks.block5_text_tokenizer import (
        SYSTEM_PROMPT, USER_START, ASST_START, COT_TRIGGER
    )

    q = "What emotion does the speaker convey?"

    # Single audio
    p1 = tok.build_chat_prompt(q, n_audios=1)
    check("prompt is ChatPrompt",    hasattr(p1, "audio_placeholders"))
    check("audio_placeholders has 1 entry", len(p1.audio_placeholders) == 1)
    check("num_tokens > 0",          p1.num_tokens > 0, str(p1.num_tokens))
    check("input_ids is list",       isinstance(p1.input_ids, list))
    check("system prompt in text",   SYSTEM_PROMPT[:20] in p1.text)
    check("assistant turn in text",  ASST_START in p1.text)
    check("audio1 open in text",     "<audio1>" in p1.text)
    check("has_cot_trigger = False", not p1.has_cot_trigger)

    # Multi audio
    p2 = tok.build_chat_prompt(q, n_audios=2)
    check("two audio placeholders",  len(p2.audio_placeholders) == 2)
    check("<audio2> in text",        "<audio2>" in p2.text)
    check("</audio2> in text",       "</audio2>" in p2.text)

    # CoT trigger
    p4 = tok.build_chat_prompt(q, n_audios=1, add_cot_trigger=True)
    check("cot trigger in text",     COT_TRIGGER in p4.text)
    check("has_cot_trigger = True",  p4.has_cot_trigger)

    # Bad n_audios
    try:
        tok.build_chat_prompt(q, n_audios=5)
        check("raises on n_audios=5", False)
    except ValueError:
        check("raises on n_audios=5", True)

    return p1, p2


# ─────────────────────────────────────────────────────────────────────
# T5 · Training targets
# ─────────────────────────────────────────────────────────────────────

def t5_training_targets(tok, p1, p2):
    hdr("T5 · build_training_target()")

    answer     = "The speaker conveys excitement."
    prompt_ids = p1.input_ids

    # Stage 1 target
    t1 = tok.build_training_target(prompt_ids, answer, stage=1)
    check("stage=1 returns TrainingTarget", hasattr(t1, "loss_mask"))
    check("stage=1 total = prompt + answer",
          t1.num_tokens > len(prompt_ids))
    check("stage=1 prompt has zero loss",
          all(v == 0 for v in t1.loss_mask[:len(prompt_ids)]))
    check("stage=1 answer span has all-1 loss",
          all(v == 1 for v in t1.loss_mask[len(prompt_ids):]))
    check("stage=1 loss_mask sums > 0", sum(t1.loss_mask) > 0,
          str(sum(t1.loss_mask)))

    # Stage 3 — same structure as stage 1
    t3 = tok.build_training_target(prompt_ids, answer, stage=3)
    check("stage=3 same shape as stage=1", t3.num_tokens == t1.num_tokens)

    # Stage 4 target
    think = "Rising F0 and rapid speech rate suggest excitement."
    t4 = tok.build_training_target(
        prompt_ids, answer, stage=4, think_text=think
    )
    check("stage=4 longer than stage=1",
          t4.num_tokens > t1.num_tokens,
          f"{t4.num_tokens} > {t1.num_tokens}")

    # FIX: BPE tokenization is not perfectly additive at chunk boundaries.
    # Allow ±2 token tolerance instead of requiring exact equality.
    diff     = t4.num_tokens - t1.num_tokens
    expected = tok.count_think_tokens(think)
    check("stage=4 extra tokens ≈ think block tokens (±2)",
          abs(diff - expected) <= 2,
          f"diff={diff}, expected≈{expected}")

    check("stage=4 prompt still zero loss",
          all(v == 0 for v in t4.loss_mask[:len(prompt_ids)]))

    # Stage 4 without think_text raises
    try:
        tok.build_training_target(prompt_ids, answer, stage=4)
        check("stage=4 raises without think_text", False)
    except ValueError:
        check("stage=4 raises without think_text", True)


# ─────────────────────────────────────────────────────────────────────
# T6 · Token count helpers
# ─────────────────────────────────────────────────────────────────────

def t6_token_count_helpers(tok):
    hdr("T6 · count_answer_tokens / count_think_tokens")

    text = "Rising F0 and rapid speech rate suggest excitement."

    n_raw     = tok.count_answer_tokens(text)
    n_wrapped = tok.count_think_tokens(text)

    # FIX: compare the SAME text with and without think tags
    # (original compared two different strings of unequal length, which
    # is not guaranteed to hold for any BPE tokenizer)
    check("count_answer_tokens > 0", n_raw > 0, str(n_raw))
    check("count_think_tokens > 0",  n_wrapped > 0, str(n_wrapped))
    check("think wrapping adds tokens (same text)",
          n_wrapped > n_raw,
          f"{n_wrapped} > {n_raw}  (<think>...</think> overhead)")


# ─────────────────────────────────────────────────────────────────────
# T7 · Audio placeholder IDs are in the actual vocab
# ─────────────────────────────────────────────────────────────────────

def t7_placeholder_ids_in_vocab(tok):
    hdr("T7 · Audio placeholder IDs in vocab range")

    for i in range(1, 5):
        oid = tok.audio_open_id(i)
        cid = tok.audio_close_id(i)
        check(f"<audio{i}> id < vocab_size", oid < tok.vocab_size, str(oid))
        check(f"</audio{i}> id < vocab_size", cid < tok.vocab_size, str(cid))


# ─────────────────────────────────────────────────────────────────────
# T8 · Consistent tokenization (same text → same IDs)
# ─────────────────────────────────────────────────────────────────────

def t8_deterministic(tok):
    hdr("T8 · Deterministic tokenization")

    text = "What is the dominant instrument in the recording?"
    ids1 = tok.encode(text).input_ids
    ids2 = tok.encode(text).input_ids
    check("same text → same IDs", ids1 == ids2)


# ─────────────────────────────────────────────────────────────────────
# T9 · Max-length truncation
# ─────────────────────────────────────────────────────────────────────

def t9_truncation(tok):
    hdr("T9 · Max-length truncation")

    long_text = "audio " * 10_000
    out = tok.encode(long_text, truncate=True)
    check("truncated to max_length",
          out.num_tokens <= tok.max_length,
          f"{out.num_tokens} ≤ {tok.max_length}")


# ─────────────────────────────────────────────────────────────────────
# T10 · Qwen chat format structure
# ─────────────────────────────────────────────────────────────────────

def t10_chat_format_structure(tok):
    hdr("T10 · Qwen-2.5 chat format structure")
    from blocks.block5_text_tokenizer import (
        SYSTEM_START, USER_START, ASST_START, TURN_END
    )

    q = "Describe the acoustic environment."
    p = tok.build_chat_prompt(q, n_audios=1)

    check("has im_start system",    SYSTEM_START in p.text)
    check("has im_start user",      USER_START   in p.text)
    check("has im_start assistant", ASST_START   in p.text)
    check("has im_end tokens",      p.text.count(TURN_END) >= 2)
    check("question in text",       q in p.text)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="NanoFlame v4 — Block5 tests")
    p.add_argument("--model",     default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--local-dir", default=None,
                   help="Path to local tokenizer dir (skips HF download)")
    return p.parse_args()


def main():
    args = get_args()

    print(f"\n{BD}{'═'*62}{E}")
    print(f"{BD}  NanoFlame v4 — Block 5 Tests{E}")
    print(f"{BD}{'═'*62}{E}")
    print(f"  model : {args.model}")

    tok = t1_load_and_vocab(args)
    t2_audio_tokens(tok)
    t3_encode_decode(tok)
    p1, p2 = t4_chat_prompt(tok)
    t5_training_targets(tok, p1, p2)
    t6_token_count_helpers(tok)
    t7_placeholder_ids_in_vocab(tok)
    t8_deterministic(tok)
    t9_truncation(tok)
    t10_chat_format_structure(tok)

    total = PASS + FAIL
    print(f"\n{BD}{'─'*62}{E}")
    print(f"  {G}{PASS} passed{E}  "
          f"{(R+str(FAIL)+E) if FAIL else str(FAIL)} failed  /  {total} run")

    if FAIL == 0:
        print(f"\n{G}{BD}✔  All Block5 tests passed.{E}\n")
    else:
        print(f"\n{R}{BD}✘  {FAIL} test(s) failed.{E}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()