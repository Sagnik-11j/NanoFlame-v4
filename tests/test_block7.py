"""
tests/test_block7.py
══════════════════════════════════════════════════════════════════════
NanoFlame v4 — Block 7 unit + integration tests

Run:
    python tests/test_block7.py                     # unit only (CPU, fast)
    python tests/test_block7.py --full              # + integration (HF download)
    python tests/test_block7.py --full --device cuda
    python tests/test_block7.py --full --local ./qwen_dir
"""

__test__ = False

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

G  = "\033[92m"; Y = "\033[93m"
R  = "\033[91m"; BD = "\033[1m"; E = "\033[0m"
ok   = lambda m: print(f"  {G}✔{E}  {m}")
warn = lambda m: print(f"  {Y}⚠{E}  {m}")
fail = lambda m: print(f"  {R}✘{E}  {m}")
info = lambda m: print(f"     {m}")
hdr  = lambda t: print(f"\n{BD}── {t} {'─'*(54-len(t))}{E}")

PASS = FAIL = 0
HIDDEN  = 896
B_VOCAB = 151_936
EXT_VOC = 151_944

def check(name, ok_, detail=""):
    global PASS, FAIL
    if ok_:
        ok(f"{name}  {detail}"); PASS += 1
    else:
        fail(f"{name}  {detail}"); FAIL += 1
    return ok_

def nonan(t):
    return not (torch.isnan(t).any() or torch.isinf(t).any())


# ─────────────────────────────────────────────────────────────────────
# Minimal Qwen mock — mirrors real Qwen attribute access patterns
# ─────────────────────────────────────────────────────────────────────

class _MockConfig:
    use_cache         = False
    hidden_size       = HIDDEN
    num_hidden_layers = 2
    vocab_size        = EXT_VOC


class _MockOutput:
    def __init__(self, logits, hidden_states, past_kv=None):
        self.logits        = logits
        self.hidden_states = hidden_states
        self.past_key_values = past_kv


class _MockInnerModel(nn.Module):
    """Simulates model.model (the transformer backbone)."""
    def __init__(self, vocab=EXT_VOC, hidden=HIDDEN):
        super().__init__()
        # Use attribute access matching the real model interface
        self.embed_tokens = nn.Embedding(vocab, hidden)

    def resize_embed(self, new_size):
        old = self.embed_tokens
        new = nn.Embedding(new_size, old.embedding_dim)
        with torch.no_grad():
            rows = min(old.num_embeddings, new_size)
            new.weight.data[:rows] = old.weight.data[:rows]
        self.embed_tokens = new
        return new


class _MockQwenModel(nn.Module):
    """
    Tiny Qwen-like model for CPU unit tests.
    Matches the attribute paths Block7QwenDecoder expects:
      - self._model.model.embed_tokens  → nn.Embedding
      - self._model.lm_head             → nn.Linear
      - self._model.config.use_cache    → bool
    """
    def __init__(self, vocab=EXT_VOC, hidden=HIDDEN):
        super().__init__()
        self.config  = _MockConfig()
        self.model   = _MockInnerModel(vocab, hidden)   # .model.embed_tokens
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, inputs_embeds, attention_mask=None,
                past_key_values=None, use_cache=False,
                return_dict=True, output_hidden_states=True, **kw):
        h      = inputs_embeds
        logits = self.lm_head(h)
        return _MockOutput(
            logits=logits,
            hidden_states=[h, h],
        )

    def resize_token_embeddings(self, new_size):
        new_emb = self.model.resize_embed(new_size)
        self.config.vocab_size = new_size
        return new_emb

    def print_trainable_parameters(self):
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        info(f"(mock) trainable params: {n:,}")


def make_mock_block7():
    """
    Returns a Block7QwenDecoder instance with a tiny CPU mock model.
    No HuggingFace download, no PEFT, no bitsandbytes required.
    """
    from blocks.block7_qwen_decoder import Block7QwenDecoder

    b7 = object.__new__(Block7QwenDecoder)
    # Set required instance attributes manually
    nn.Module.__init__(b7)
    b7.model_id            = "mock"
    b7.extended_vocab_size = EXT_VOC
    b7.lora_r              = 16
    b7.lora_alpha          = 32
    b7.lora_dropout        = 0.05
    b7.use_4bit            = False
    b7._lora_applied       = False
    b7._peft_wrapped       = False
    b7._model              = _MockQwenModel(vocab=EXT_VOC, hidden=HIDDEN)
    return b7


# ─────────────────────────────────────────────────────────────────────
# Unit tests (mock model, CPU, no HF download)
# ─────────────────────────────────────────────────────────────────────

def t1_vocab_size(b7):
    hdr("T1 · Extended vocab = BASE(151936) + 8 audio tokens = 151944")
    emb = b7._get_embed_tokens()
    check("embedding rows == 151,944", emb.num_embeddings == EXT_VOC,
          str(emb.num_embeddings))
    check("embedding dim == 896", emb.embedding_dim == HIDDEN)


def t2_forward_inference(b7):
    hdr("T2 · Inference forward — inputs_embeds [1, 800, 896]")
    from blocks.block7_qwen_decoder import Block7Output

    T  = 800
    x  = torch.randn(1, T, HIDDEN)
    am = torch.ones(1, T, dtype=torch.long)

    with torch.no_grad():
        out = b7.forward(inputs_embeds=x, attention_mask=am)

    check("returns Block7Output",    isinstance(out, Block7Output))
    check("logits shape [1, T, V]",
          out.logits.shape == (1, T, EXT_VOC), str(tuple(out.logits.shape)))
    check("hidden shape [1, T, 896]",
          out.hidden.shape == (1, T, HIDDEN))
    check("loss = None",             out.loss is None)
    check("no NaN in logits",        nonan(out.logits))


def t3_embed_tokens_accessor(b7):
    hdr("T3 · embed_tokens property for Block 6")
    emb = b7.embed_tokens
    check("returns nn.Embedding",   isinstance(emb, nn.Embedding))
    check("rows == EXT_VOC",
          emb.weight.shape[0] == EXT_VOC, str(emb.weight.shape[0]))
    check("dim == 896",             emb.weight.shape[1] == HIDDEN)


def t4_freeze_and_lm_head(b7):
    hdr("T4 · freeze_all() and _unfreeze_lm_head()")
    b7._model.requires_grad_(False)
    frozen = sum(1 for p in b7._model.parameters() if not p.requires_grad)
    total  = sum(1 for p in b7._model.parameters())
    check("freeze_all works",  frozen == total, f"({frozen}/{total})")

    b7._unfreeze_lm_head()
    lm_trainable = any(
        p.requires_grad
        for n, p in b7._model.named_parameters()
        if "lm_head" in n
    )
    check("lm_head is trainable after unfreeze", lm_trainable)


def t5_lm_head_vocab(b7):
    hdr("T5 · LM head output dim = 151,944")
    lm = b7._get_lm_head()
    check("lm_head out_features == 151,944",
          lm.out_features == EXT_VOC, str(lm.out_features))


def t6_use_cache_disabled(b7):
    hdr("T6 · use_cache disabled during training forward")
    b7._model.config.use_cache = False
    T  = 50
    x  = torch.randn(1, T, HIDDEN)
    am = torch.ones(1, T, dtype=torch.long)
    with torch.no_grad():
        _ = b7.forward(inputs_embeds=x, attention_mask=am, use_cache=False)
    check("config.use_cache = False after forward",
          not b7._model.config.use_cache)


def t7_packed_sequence_dispatch(b7):
    hdr("T7 · PackedSequence dispatch")

    class FakePacked:
        inputs_embeds  = torch.randn(1, 100, HIDDEN)
        attention_mask = torch.ones(1, 100, dtype=torch.long)
        target_len     = 0
        loss_mask      = torch.zeros(1, 100, dtype=torch.long)

    with torch.no_grad():
        out = b7.forward(FakePacked())

    check("dispatches from packed_seq", out.logits.shape[1] == 100)
    check("no NaN", nonan(out.logits))


def t8_generate_step(b7):
    hdr("T8 · generate_step() returns int token ID")
    T  = 20
    x  = torch.randn(1, T, HIDDEN)
    am = torch.ones(1, T, dtype=torch.long)

    token_id, _ = b7.generate_step(x, am, temperature=0.7, top_p=0.9)
    check("token_id is int", isinstance(token_id, int))
    check("in vocab range", 0 <= token_id < EXT_VOC, str(token_id))


def t9_repr(b7):
    hdr("T9 · __repr__")
    r = repr(b7)
    check("is string",        isinstance(r, str))
    check("contains vocab",   str(EXT_VOC) in r or "vocab" in r.lower())
    check("contains 4bit",    "4bit" in r or "False" in r)


def t10_trainable_params_count(b7):
    hdr("T10 · trainable_params <= total_params")
    check("trainable ≤ total",
          b7.trainable_params <= b7.total_params,
          f"{b7.trainable_params:,}/{b7.total_params:,}")


# ─────────────────────────────────────────────────────────────────────
# Integration tests (real Qwen weights)
# ─────────────────────────────────────────────────────────────────────

def ti1_load_real(args):
    hdr("TI1 · Load Qwen-2.5-0.5B-Instruct (4-bit NF4)")
    from blocks.block7_qwen_decoder import Block7QwenDecoder

    local = args.local if hasattr(args, "local") and args.local else None
    b7 = Block7QwenDecoder(
        model_id=args.model,
        use_4bit=True,
        use_flash_attn=False,
        local_dir=local,
    )
    info(repr(b7))

    emb = b7.embed_tokens
    check("vocab resized to 151,944", emb.num_embeddings == EXT_VOC,
          str(emb.num_embeddings))
    check("lora applied",      b7._lora_applied)
    check("trainable < total", b7.trainable_params < b7.total_params,
          f"{b7.trainable_params:,}/{b7.total_params:,}")
    return b7


def ti2_stage1_trainable(b7):
    hdr("TI2 · Stage 1: LM head + audio embed rows")
    b7.configure_for_stage(1)
    trainable = {n for n, p in b7._model.named_parameters() if p.requires_grad}
    check("some params trainable", len(trainable) > 0,
          f"{len(trainable)} tensors")
    lm_trainable = any("lm_head" in n for n in trainable)
    check("lm_head trainable", lm_trainable)


def ti3_stage2_lora_trainable(b7):
    hdr("TI3 · Stage 2: QLoRA adapters trainable")
    b7.configure_for_stage(2)
    lora_params = [n for n, p in b7._model.named_parameters()
                   if p.requires_grad and "lora_" in n]
    check("lora params trainable", len(lora_params) > 0,
          f"{len(lora_params)} tensors")


def ti4_inference_forward(b7, device):
    hdr("TI4 · Inference forward on real weights")
    T  = 50
    x  = torch.randn(1, T, HIDDEN, device=device, dtype=torch.float16)
    am = torch.ones(1, T, dtype=torch.long, device=device)

    with torch.no_grad():
        out = b7.forward(inputs_embeds=x, attention_mask=am)

    check("logits [1, T, 151944]",
          out.logits.shape == (1, T, EXT_VOC), str(tuple(out.logits.shape)))
    check("hidden [1, T, 896]",    out.hidden.shape == (1, T, HIDDEN))
    check("no NaN",                nonan(out.logits.float()))


def ti5_generate_step(b7, device):
    hdr("TI5 · generate_step() on real weights")
    T  = 10
    x  = torch.randn(1, T, HIDDEN, device=device, dtype=torch.float16)
    am = torch.ones(1, T, dtype=torch.long, device=device)

    tid, kv = b7.generate_step(x, am, temperature=0.7, top_p=0.9)
    check("token_id is int",    isinstance(tid, int))
    check("0 ≤ id < 151944",   0 <= tid < EXT_VOC, str(tid))


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--local",  default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--full",   action="store_true",
                   help="Run integration tests (downloads Qwen weights)")
    return p.parse_args()


def main():
    args = get_args()

    print(f"\n{BD}{'═'*62}{E}")
    print(f"{BD}  NanoFlame v4 — Block 7 Tests{E}")
    print(f"{BD}{'═'*62}{E}")
    print(f"  device : {args.device}  |  full : {args.full}")

    print(f"\n{BD}{'═'*62}{E}")
    print(f"{BD}  UNIT TESTS — mock model (CPU, no HF download){E}")
    print(f"{BD}{'═'*62}{E}")

    b7 = make_mock_block7()
    t1_vocab_size(b7)
    t2_forward_inference(b7)
    t3_embed_tokens_accessor(b7)
    t4_freeze_and_lm_head(b7)
    t5_lm_head_vocab(b7)
    t6_use_cache_disabled(b7)
    t7_packed_sequence_dispatch(b7)
    t8_generate_step(b7)
    t9_repr(b7)
    t10_trainable_params_count(b7)

    if args.full:
        print(f"\n{BD}{'═'*62}{E}")
        print(f"{BD}  INTEGRATION TESTS — real Qwen-2.5-0.5B-Instruct{E}")
        print(f"{BD}{'═'*62}{E}")
        try:
            real_b7 = ti1_load_real(args)
            ti2_stage1_trainable(real_b7)
            ti3_stage2_lora_trainable(real_b7)
            ti4_inference_forward(real_b7, args.device)
            ti5_generate_step(real_b7, args.device)
        except Exception as e:
            fail(f"Integration test failed: {e}")
    else:
        warn("Integration tests skipped — run with --full to enable")

    total = PASS + FAIL
    print(f"\n{BD}{'─'*62}{E}")
    print(f"  {G}{PASS} passed{E}  "
          f"{(R+str(FAIL)+E) if FAIL else str(FAIL)} failed  /  {total} run")

    if FAIL == 0:
        print(f"\n{G}{BD}✔  All Block7 tests passed.{E}\n")
    else:
        print(f"\n{R}{BD}✘  {FAIL} test(s) failed.{E}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()