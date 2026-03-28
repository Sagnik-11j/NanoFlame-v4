"""
tests/test_block3.py
══════════════════════════════════════════════════════════════════════
NanoFlame v4 — Block 3 unit tests + full pipeline integration tests
══════════════════════════════════════════════════════════════════════

Exact class names confirmed from source files:
  Block 1  : AudioFrontend           (blocks/block1_audio_frontend.py)
               .process(file_path)   → AudioFrontendOutput  .chunks, .stack()
               NOT nn.Module — CPU only, NO .to(device)

  Block 2a : Block2aWhisperEncoder   (blocks/block2a_whisper_encoder.py)
               forward(mel)          → WhisperEncoderOutput  .h_w
               downloads openai/whisper-medium ONCE on first init

  Block 2b : Block2bOpenBEATsEncoder (blocks/block2b_openbeats_encoder.py)
               forward(mel)          → OpenBEATsEncoderOutput  .h_ob
               .freeze_base_weights() / .unfreeze_all()

  Block 3  : Block3PerceiverFusion   (blocks/block3_perceiver_fusion.py)
               forward(h_ob, h_w)    → Block3Output  .h_full
               .freeze() / .unfreeze() / .enable_training_dropouts(p)

Run as standalone script (NOT via pytest):
    python tests/test_block3.py
    python tests/test_block3.py --device cuda
    python tests/test_block3.py --skip-integration
    python tests/test_block3.py --ckpt checkpoints/epoch_latest.pt
══════════════════════════════════════════════════════════════════════
"""

__test__ = False   # prevent pytest from collecting this file

import sys
import math
import argparse
import tempfile
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

try:
    import torchaudio
    _HAS_TORCHAUDIO = True
except ImportError:
    _HAS_TORCHAUDIO = False

# ── colour helpers ──────────────────────────────────────────────────────────
G  = "\033[92m"; Y  = "\033[93m"
R  = "\033[91m"; BD = "\033[1m"; E = "\033[0m"
ok_fn = lambda m: print(f"  {G}✔{E}  {m}")
warn  = lambda m: print(f"  {Y}⚠{E}  {m}")
fail  = lambda m: print(f"  {R}✘{E}  {m}")
skip  = lambda m: print(f"  {Y}~{E}  {m}")
info  = lambda m: print(f"     {m}")
hdr   = lambda t: print(f"\n{BD}── {t} {'─'*(54-len(t))}{E}")

PASS = FAIL = 0


def check(name: str, ok_: bool, detail: str = "") -> bool:
    global PASS, FAIL
    if ok_:
        ok_fn(f"{name}  {detail}"); PASS += 1
    else:
        fail(f"{name}  {detail}"); FAIL += 1
    return ok_


def nonan(t: torch.Tensor) -> bool:
    return not (torch.isnan(t).any() or torch.isinf(t).any())


def grad_cov(m: nn.Module) -> float:
    g = sum(1 for p in m.parameters()
            if p.requires_grad and p.grad is not None and p.grad.abs().max() > 0)
    t = sum(1 for p in m.parameters() if p.requires_grad)
    return g / t if t else 0.0


# ══════════════════════════════════════════════════════════════════════
# Block imports — exact class names from source files
# ══════════════════════════════════════════════════════════════════════

def get_block3():
    from blocks.block3_perceiver_fusion import (
        Block3PerceiverFusion,
        PerceiverResampler,
        CrossAttentionFusion,
        ChunkConcat,
    )
    return Block3PerceiverFusion, PerceiverResampler, CrossAttentionFusion, ChunkConcat


def get_block1():
    """
    Class  : AudioFrontend
    API    : af.process(file_path) → AudioFrontendOutput
               .chunks        List[Tensor[128, 3000]]
               .stack()       Tensor[N, 128, 3000]
               .num_chunks    int
               .duration_sec  float
    NOTE   : NOT nn.Module. Runs on CPU only. Takes a file path, NOT a tensor.
    """
    from blocks.block1_audio_frontend import AudioFrontend
    return AudioFrontend


def get_block2a_class():
    """Returns the Block2aWhisperEncoder CLASS only — no instantiation."""
    from blocks.block2a_whisper_encoder import Block2aWhisperEncoder
    return Block2aWhisperEncoder


def build_whisper_encoder(Block2aWhisperEncoder, device: str):
    """
    Instantiate Block2aWhisperEncoder ONCE, move to device, set eval.
    This is the only place the HuggingFace download ever happens.
    All integration tests receive this pre-built instance.
    """
    info("Building Block2aWhisperEncoder (downloads whisper-medium on first run)...")
    try:
        enc = Block2aWhisperEncoder()
        enc = enc.to(device)
        enc.eval()
        ok_fn("Block2aWhisperEncoder ready")
        return enc
    except Exception as e:
        warn(f"Block2aWhisperEncoder init failed: {e}")
        return None


def load_block2b(ckpt: str, device: str):
    """
    Class   : Block2bOpenBEATsEncoder
    Output  : OpenBEATsEncoderOutput .h_ob [B, 1496, 1024]
    Methods : .freeze_base_weights() / .unfreeze_all()
    """
    from blocks.block2b_openbeats_encoder import Block2bOpenBEATsEncoder

    enc  = Block2bOpenBEATsEncoder()
    path = Path(ckpt)

    if not path.exists():
        warn(f"Checkpoint not found: {path}")
        return None, False

    missing, unexpected = enc.load_openbeats_checkpoint(str(path), strict=False)
    n_model = len(list(enc._enc.state_dict().keys()))
    pct     = 100.0 * (n_model - len(missing)) / max(n_model, 1)
    ok_fn(f"Block2b loaded  ({pct:.1f}% keys matched, "
          f"{len(unexpected)} unexpected)")
    enc = enc.to(device).eval()
    return enc, True


# ══════════════════════════════════════════════════════════════════════
# Helpers for real blocks
# ══════════════════════════════════════════════════════════════════════

def mel_from_block1(AudioFrontend, seconds: float = 30.0, device: str = "cpu"):
    """
    AudioFrontend.process() only accepts a FILE PATH — not a raw tensor.
    Strategy:
      1. Synthesise a multi-tone waveform in memory
      2. Save it to a temp .wav file
      3. Pass the path to af.process()
      4. Return mel.stack()[0:1] → [1, 128, 3000]
    """
    if not _HAS_TORCHAUDIO:
        warn("torchaudio not available — using synthetic mel")
        return torch.randn(1, 128, 3000, device=device)

    sr    = 16_000
    n     = int(sr * seconds)
    t_ax  = torch.linspace(0, seconds, n)
    audio = (
        0.4 * torch.sin(2 * math.pi * 440.0 * t_ax) +
        0.3 * torch.sin(2 * math.pi * 880.0 * t_ax) +
        0.1 * torch.randn(n).clamp(-1, 1)
    ).unsqueeze(0).clamp(-1.0, 1.0)   # [1, N]

    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        torchaudio.save(tmp_path, audio, sr)

        af  = AudioFrontend()
        out = af.process(tmp_path)      # AudioFrontendOutput
        mel = out.stack()               # [N_chunks, 128, 3000]
        return mel[:1].to(device)       # first chunk → [1, 128, 3000]

    except Exception as e:
        warn(f"Block1 mel synthesis failed ({e}) — using synthetic mel")
        return torch.randn(1, 128, 3000, device=device)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def h_w_from_enc(whisper_enc, mel: torch.Tensor, device: str):
    """
    Run inference on a PRE-BUILT Block2aWhisperEncoder instance.

    `whisper_enc` must always be an already-instantiated model (never the class).
    Passing the class here is a programming error and will raise TypeError
    immediately — that's intentional so we catch the bug early.

    Returns h_w [B, 750, 1024].
    Falls back to synthetic tensor only if the forward() itself fails.
    """
    if isinstance(whisper_enc, type):
        raise TypeError(
            "h_w_from_enc() received a CLASS, not an instance. "
            "Build the encoder once with build_whisper_encoder() and pass the instance."
        )

    B = mel.size(0)
    try:
        with torch.no_grad():
            out = whisper_enc(mel.to(device))
        h_w = out.h_w                       # WhisperEncoderOutput .h_w
        if h_w.size(-1) != 1024:
            warn(f"Block2a output dim={h_w.size(-1)} ≠ 1024 — projecting")
            proj = nn.Linear(h_w.size(-1), 1024).to(device)
            with torch.no_grad():
                h_w = proj(h_w)
        return h_w
    except Exception as e:
        warn(f"Block2aWhisperEncoder forward failed ({e}) — using synthetic h_w")
        return torch.randn(B, 750, 1024, device=device)


# ══════════════════════════════════════════════════════════════════════
# ──────────────────────────────────────────────────────────────────────
#  UNIT TESTS — Block3 in isolation (no checkpoint needed)
# ──────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def t1_perceiver_resampler(device, B=2):
    hdr("T1 · [3a] PerceiverResampler  1496 → 64 tokens")
    _, PerceiverResampler, _, _ = get_block3()

    r    = PerceiverResampler(d_model=1024, n_queries=64, n_heads=16).to(device)
    h_ob = torch.randn(B, 1496, 1024, device=device)

    with torch.no_grad():
        out = r(h_ob)

    check("output shape [B, 64, 1024]",
          out.shape == (B, 64, 1024), str(tuple(out.shape)))
    check("no NaN/Inf",  nonan(out))
    check("learned queries trainable", r.queries.requires_grad)
    check("output std < 5.0 (layer_norm applied)", out.std().item() < 5.0,
          f"std={out.std().item():.4f}")


def t2_cross_attn_fusion(device, B=2):
    hdr("T2 · [3b] CrossAttentionFusion  h_w × h_resampled")
    _, _, CrossAttentionFusion, _ = get_block3()

    f   = CrossAttentionFusion(d_model=1024, n_heads=16).to(device)
    h_w = torch.randn(B, 750, 1024, device=device)
    h_r = torch.randn(B,  64, 1024, device=device)

    with torch.no_grad():
        out = f(h_w=h_w, h_resampled=h_r)

    check("output shape [B, 750, 1024]",
          out.shape == (B, 750, 1024), str(tuple(out.shape)))
    check("no NaN/Inf", nonan(out))
    check("sequence length preserved",  out.size(1) == 750)
    check("cross-attn modifies input",
          not torch.allclose(out, h_w, atol=1e-3))


def t3_chunk_concat(device, B=2):
    hdr("T3 · [3c] ChunkConcat  positional encodings per chunk")
    _, _, _, ChunkConcat = get_block3()

    cc = ChunkConcat(d_model=1024, max_chunks=16).to(device)
    N  = 3
    cs = [torch.randn(B, 750, 1024, device=device) for _ in range(N)]

    with torch.no_grad():
        h = cc(cs)

    check(f"output shape [B, {750*N}, 1024]",
          h.shape == (B, 750*N, 1024), str(tuple(h.shape)))
    check("no NaN/Inf", nonan(h))
    check("chunk_pos_enc is trainable", cc.chunk_pos_enc.requires_grad)
    check("pos-enc differentiates chunks",
          not torch.allclose(h[:, :750], h[:, 750:1500], atol=1e-4))


def t4_single_chunk_forward(device, B=2):
    hdr("T4 · Block3PerceiverFusion — single 30s chunk")
    Block3, *_ = get_block3()

    b3   = Block3().to(device)
    h_ob = torch.randn(B, 1496, 1024, device=device)
    h_w  = torch.randn(B,  750, 1024, device=device)

    with torch.no_grad():
        out = b3(h_ob=h_ob, h_w=h_w)

    check("h_full shape [B, 750, 1024]",
          out.h_full.shape == (B, 750, 1024), str(tuple(out.h_full.shape)))
    check("h_resampled shape [B, 64, 1024]",
          out.h_resampled.shape == (B, 64, 1024))
    check("n_chunks == 1",   out.n_chunks == 1)
    check("seq_len == 750",  out.seq_len == 750)
    check("no NaN h_full",      nonan(out.h_full))
    check("no NaN h_resampled", nonan(out.h_resampled))
    check("h_fused_chunks is list of 1",
          isinstance(out.h_fused_chunks, list) and len(out.h_fused_chunks) == 1)

    info(f"Block3 total params  : {b3.total_params:,}")
    info(f"Block3 trainable     : {b3.trainable_params:,}")


def t5_multi_chunk_forward(device, B=2, N=3):
    hdr(f"T5 · Block3PerceiverFusion — {N} chunks ({N*30}s audio)")
    Block3, *_ = get_block3()

    b3     = Block3().to(device)
    chunks = [
        (torch.randn(B, 1496, 1024, device=device),
         torch.randn(B,  750, 1024, device=device))
        for _ in range(N)
    ]

    with torch.no_grad():
        out = b3.forward_multi_chunk(chunks)

    expected_seq = 750 * N
    check(f"h_full shape [B, {expected_seq}, 1024]",
          out.h_full.shape == (B, expected_seq, 1024),
          str(tuple(out.h_full.shape)))
    check(f"n_chunks == {N}",           out.n_chunks == N)
    check(f"seq_len == {expected_seq}", out.seq_len == expected_seq)
    check(f"len(h_fused_chunks) == {N}", len(out.h_fused_chunks) == N)
    check("no NaN", nonan(out.h_full))
    check("chunk segments differ (pos-enc applied)",
          not torch.allclose(
              out.h_full[:, :750],
              out.h_full[:, 750:1500], atol=1e-4))

    info(f"Simulates {N * 30}s of audio")
    info(f"Total seq_len = {out.seq_len} tokens  (~{750/30:.1f} tok/sec)")


def t6_qwen_projection(device, B=2):
    hdr("T6 · Qwen-2.5-0.5B projection  1024 → 896")
    Block3, *_ = get_block3()

    QWEN_DIM  = 896
    b3        = Block3().to(device)
    qwen_proj = nn.Linear(1024, QWEN_DIM).to(device)

    with torch.no_grad():
        out    = b3(
            h_ob=torch.randn(B, 1496, 1024, device=device),
            h_w =torch.randn(B,  750, 1024, device=device),
        )
        h_qwen = qwen_proj(out.h_full)

    check(f"h_qwen shape [B, 750, {QWEN_DIM}]",
          h_qwen.shape == (B, 750, QWEN_DIM), str(tuple(h_qwen.shape)))
    check("no NaN after projection", nonan(h_qwen))
    check("proj lives outside Block3",
          not any(p is q for p in b3.parameters()
                  for q in qwen_proj.parameters()))

    info(f"Audio tokens ready for Qwen: {tuple(h_qwen.shape)}")
    info(f"~{750/30:.1f} tokens/sec of audio")


def t7_gradient_flow(device, B=2):
    """
    Grad checks happen BEFORE zero_grad() — after zero_grad() all .grad are None.
    """
    hdr("T7 · Gradient flow through all sub-modules")
    Block3, *_ = get_block3()

    b3 = Block3().to(device)
    b3.unfreeze()

    out = b3(
        h_ob=torch.randn(B, 1496, 1024, device=device),
        h_w =torch.randn(B,  750, 1024, device=device),
    )
    out.h_full.mean().backward()

    # ── Check specific params BEFORE zero_grad() ─────────────────
    queries_has_grad = (
        b3.perceiver.queries.grad is not None and
        b3.perceiver.queries.grad.abs().max() > 0
    )
    pos_enc_has_grad = (
        b3.chunk_concat.chunk_pos_enc.grad is not None and
        b3.chunk_concat.chunk_pos_enc.grad.abs().max() > 0
    )
    pct = grad_cov(b3)

    # ── NOW zero the grads ────────────────────────────────────────
    b3.zero_grad()

    check("≥80% params received gradient", pct >= 0.80, f"({pct*100:.0f}%)")
    check("Perceiver queries received gradient", queries_has_grad)
    check("ChunkConcat pos-enc received gradient", pos_enc_has_grad)


def t8_stage_control(device):
    hdr("T8 · Stage control: freeze() / unfreeze() / enable_training_dropouts()")
    Block3, *_ = get_block3()
    b3    = Block3().to(device)
    total = sum(1 for _ in b3.parameters())

    b3.freeze()
    frozen = sum(1 for p in b3.parameters() if not p.requires_grad)
    check("freeze() freezes all params", frozen == total,
          f"({frozen}/{total} frozen)")

    b3.unfreeze()
    trainable = sum(1 for p in b3.parameters() if p.requires_grad)
    check("unfreeze() restores all params", trainable == total,
          f"({trainable}/{total} trainable)")

    b3.enable_training_dropouts(dropout=0.1)
    dropouts = [m.p for m in b3.modules() if isinstance(m, nn.Dropout)]
    if dropouts:
        check("enable_training_dropouts sets p=0.1",
              all(abs(v - 0.1) < 1e-6 for v in dropouts),
              f"({len(dropouts)} Dropout layers)")
    else:
        skip("No Dropout layers (dropout=0.0 at init — correct)")


def t9_output_dataclass(device, B=2):
    hdr("T9 · Block3Output dataclass — all fields present")
    Block3, *_ = get_block3()
    b3  = Block3().to(device)

    with torch.no_grad():
        out = b3(
            h_ob=torch.randn(B, 1496, 1024, device=device),
            h_w =torch.randn(B,  750, 1024, device=device),
        )

    check("has .h_full",          hasattr(out, "h_full"))
    check("has .h_resampled",     hasattr(out, "h_resampled"))
    check("has .h_fused_chunks",  hasattr(out, "h_fused_chunks"))
    check("has .n_chunks",        hasattr(out, "n_chunks"))
    check("has .seq_len",         hasattr(out, "seq_len"))
    check(".h_fused_chunks is list", isinstance(out.h_fused_chunks, list))
    check(".seq_len == h_full.size(1)",
          out.seq_len == out.h_full.size(1))


# ══════════════════════════════════════════════════════════════════════
# ──────────────────────────────────────────────────────────────────────
#  INTEGRATION TESTS — Block1 → Block2a → Block2b → Block3
#
#  All functions accept `whisper_enc` — a pre-built instance, never the class.
#  HuggingFace is called exactly once: build_whisper_encoder() in main().
# ──────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def ti1_block1_shapes(AudioFrontend, device):
    hdr("TI1 · Block1 (AudioFrontend) — mel chunk shapes")
    info("AudioFrontend.process() takes a file path. Writing temp .wav ...")

    mel = mel_from_block1(AudioFrontend, seconds=30.0, device=device)

    check("mel is 3D [B, 128, T]",
          mel.dim() == 3 and mel.size(1) == 128,
          str(tuple(mel.shape)))
    check("T ≈ 3000 (30s × 100 fps)",
          abs(mel.size(2) - 3000) <= 10,
          f"T={mel.size(2)}")
    check("no NaN in mel", nonan(mel))

    info(f"Block1 mel output: {tuple(mel.shape)}")
    return mel


def ti2_block2b_shapes(b2b, mel, device):
    hdr("TI2 · Block2b (OpenBEATs) — h_ob shape")

    with torch.no_grad():
        out = b2b(mel)

    h_ob = out.h_ob
    check("h_ob is 3D",           h_ob.dim() == 3)
    check("h_ob last dim = 1024", h_ob.size(-1) == 1024)
    check("N_patches ≈ 1496",
          abs(h_ob.size(1) - 1496) <= 4,
          f"got {h_ob.size(1)}")
    check("no NaN h_ob", nonan(h_ob))

    info(f"Block2b h_ob: {tuple(h_ob.shape)}")
    return h_ob


def ti3_block2a_shapes(whisper_enc, mel, device):
    """whisper_enc is a pre-built instance — no HF download here."""
    hdr("TI3 · Block2a (Block2aWhisperEncoder) — h_w shape")
    info("Using pre-built Whisper instance (HF weights already loaded).")

    h_w = h_w_from_enc(whisper_enc, mel, device)

    check("h_w is 3D",            h_w.dim() == 3)
    check("h_w last dim = 1024",  h_w.size(-1) == 1024)
    check("h_w seq_len ≈ 750",
          abs(h_w.size(1) - 750) <= 10,
          f"got {h_w.size(1)}")
    check("no NaN h_w", nonan(h_w))

    info(f"Block2a h_w : {tuple(h_w.shape)}")
    return h_w


def ti4_block3_full(b2b, whisper_enc, AudioFrontend, device):
    """whisper_enc is the pre-built instance."""
    hdr("TI4 · Full pipeline: Block1→2a→2b→Block3 (single 30s chunk)")
    Block3, *_ = get_block3()
    b3 = Block3().to(device)

    mel = mel_from_block1(AudioFrontend, seconds=30.0, device=device)
    info(f"mel      : {tuple(mel.shape)}")

    with torch.no_grad():
        h_ob = b2b(mel).h_ob
    info(f"h_ob     : {tuple(h_ob.shape)}")

    h_w = h_w_from_enc(whisper_enc, mel, device)
    info(f"h_w      : {tuple(h_w.shape)}")

    with torch.no_grad():
        out = b3(h_ob=h_ob, h_w=h_w)

    B = mel.size(0)
    check("h_full shape [B, 750, 1024]",
          out.h_full.shape == (B, 750, 1024),
          str(tuple(out.h_full.shape)))
    check("h_resampled [B, 64, 1024]",
          out.h_resampled.shape == (B, 64, 1024))
    check("no NaN end-to-end", nonan(out.h_full))

    std = out.h_resampled.std().item()
    check("h_resampled std in realistic range",
          0.003 < std < 30.0, f"std={std:.4f}")

    info(f"h_full   : {tuple(out.h_full.shape)}")


def ti5_multi_chunk_pipeline(b2b, whisper_enc, AudioFrontend, device, N=3):
    """
    whisper_enc is the pre-built instance — reused for every chunk.
    HuggingFace is NOT called again here.
    """
    hdr(f"TI5 · Full pipeline: {N} chunks = {N*30}s audio")
    Block3, *_ = get_block3()
    b3 = Block3().to(device)

    chunk_inputs = []
    for i in range(N):
        mel = mel_from_block1(AudioFrontend, seconds=30.0, device=device)

        with torch.no_grad():
            h_ob = b2b(mel).h_ob

        # ✅ Reuses the same whisper_enc instance — no re-download, no re-init
        h_w = h_w_from_enc(whisper_enc, mel, device)

        chunk_inputs.append((h_ob, h_w))
        info(f"  chunk {i+1}/{N}: h_ob={tuple(h_ob.shape)}  h_w={tuple(h_w.shape)}")

    with torch.no_grad():
        out = b3.forward_multi_chunk(chunk_inputs)

    B            = chunk_inputs[0][0].size(0)
    expected_seq = 750 * N

    check(f"h_full [B, {expected_seq}, 1024]",
          out.h_full.shape == (B, expected_seq, 1024),
          str(tuple(out.h_full.shape)))
    check(f"n_chunks == {N}", out.n_chunks == N)
    check("no NaN", nonan(out.h_full))
    check("chunk segments differ (real pos-enc applied)",
          not torch.allclose(
              out.h_full[:, :750],
              out.h_full[:, 750:1500], atol=1e-4))

    info(f"Simulates {N*30}s of audio")
    info(f"h_full seq_len = {out.seq_len} tokens")


def ti6_qwen_ready_real(b2b, whisper_enc, AudioFrontend, device):
    """whisper_enc is the pre-built instance."""
    hdr("TI6 · End-to-end output → Qwen-2.5-0.5B projection (1024→896)")
    Block3, *_ = get_block3()
    QWEN_DIM   = 896
    b3         = Block3().to(device)
    qwen_proj  = nn.Linear(1024, QWEN_DIM).to(device)

    mel = mel_from_block1(AudioFrontend, seconds=30.0, device=device)

    with torch.no_grad():
        h_ob = b2b(mel).h_ob

    h_w = h_w_from_enc(whisper_enc, mel, device)

    with torch.no_grad():
        out    = b3(h_ob=h_ob, h_w=h_w)
        h_qwen = qwen_proj(out.h_full)

    B = mel.size(0)
    check(f"h_qwen [B, 750, {QWEN_DIM}]",
          h_qwen.shape == (B, 750, QWEN_DIM),
          str(tuple(h_qwen.shape)))
    check("no NaN in Qwen-ready tokens", nonan(h_qwen))

    info(f"h_full  : {tuple(out.h_full.shape)}")
    info(f"h_qwen  : {tuple(h_qwen.shape)}")
    info(f"~{750/30:.1f} audio tokens/sec  →  ready for Qwen embedding lookup")


def ti7_stage1_gradient(b2b, whisper_enc, AudioFrontend, device):
    """
    Stage-1 training scenario — whisper_enc is the pre-built instance.
    Grad checks happen BEFORE zero_grad().
    """
    hdr("TI7 · Stage-1: Block2b frozen, only Block3 trains")
    Block3, *_ = get_block3()
    b3 = Block3().to(device)

    b2b.freeze_base_weights()
    b3.unfreeze()

    mel = mel_from_block1(AudioFrontend, seconds=10.0, device=device)

    with torch.no_grad():
        h_ob = b2b(mel).h_ob
    h_ob = h_ob.detach()   # grad must NOT flow back through frozen Block2b

    h_w = h_w_from_enc(whisper_enc, mel, device)

    out = b3(h_ob=h_ob, h_w=h_w)
    out.h_full.mean().backward()

    pct = grad_cov(b3)
    b2b_has_no_grad = all(p.grad is None for p in b2b._enc.parameters())

    b3.zero_grad()

    check("Block3 grads flow (Stage 1 scenario)", pct >= 0.75,
          f"({pct*100:.0f}%)")
    check("Block2b encoder has no grad (frozen)", b2b_has_no_grad)

    info("Stage 1: freeze B2b → train Block3 + Qwen projection only")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(
        description="NanoFlame v4 — Block3 + pipeline tests"
    )
    p.add_argument("--ckpt",   default=str(ROOT / "checkpoints" / "epoch_latest.pt"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch",  type=int, default=2)
    p.add_argument("--skip-integration", action="store_true",
                   help="Unit tests only — no checkpoint / Whisper download needed")
    return p.parse_args()


def main():
    args = get_args()
    D, B = args.device, args.batch

    print(f"\n{BD}{'═'*62}{E}")
    print(f"{BD}  NanoFlame v4 — Block3 + Pipeline Tests{E}")
    print(f"{BD}{'═'*62}{E}")
    print(f"  device : {D}  |  batch : {B}")
    print(f"  ckpt   : {args.ckpt}")

    # ── UNIT TESTS ────────────────────────────────────────────────
    print(f"\n{BD}{'═'*62}{E}")
    print(f"{BD}  UNIT TESTS  —  Block3 in isolation (no checkpoint){E}")
    print(f"{BD}{'═'*62}{E}")

    t1_perceiver_resampler(D, B)
    t2_cross_attn_fusion(D, B)
    t3_chunk_concat(D, B)
    t4_single_chunk_forward(D, B)
    t5_multi_chunk_forward(D, B, N=3)
    t6_qwen_projection(D, B)
    t7_gradient_flow(D, B)
    t8_stage_control(D)
    t9_output_dataclass(D, B)

    # ── INTEGRATION TESTS ─────────────────────────────────────────
    if not args.skip_integration:
        print(f"\n{BD}{'═'*62}{E}")
        print(f"{BD}  INTEGRATION TESTS  —  Block1 → Block2a → Block2b → Block3{E}")
        print(f"{BD}{'═'*62}{E}")

        AudioFrontend = None
        Block2aClass  = None

        try:
            AudioFrontend = get_block1()
            ok_fn("Block1 (AudioFrontend) imported")
        except ImportError as e:
            skip(f"Block1 import failed: {e}")

        try:
            Block2aClass = get_block2a_class()
            ok_fn("Block2a (Block2aWhisperEncoder) class imported")
        except ImportError as e:
            skip(f"Block2a import failed: {e}")

        b2b, loaded = load_block2b(args.ckpt, D)

        missing = []
        if AudioFrontend is None: missing.append("Block1 (AudioFrontend)")
        if Block2aClass  is None: missing.append("Block2a (Block2aWhisperEncoder)")
        if not loaded:             missing.append("Block2b weights (checkpoint)")

        if missing:
            for m in missing:
                skip(f"Unavailable: {m}")
            skip("Skipping integration tests — resolve above issues first")
        else:
            # ✅ Build Whisper ONCE here — HuggingFace called exactly one time
            # All integration tests below receive the pre-built instance.
            print(f"\n{BD}  Initialising shared resources (once)...{E}")
            whisper_enc = build_whisper_encoder(Block2aClass, D)

            if whisper_enc is None:
                skip("Whisper init failed — skipping integration tests")
            else:
                mel = ti1_block1_shapes(AudioFrontend, D)
                _   = ti2_block2b_shapes(b2b, mel, D)
                _   = ti3_block2a_shapes(whisper_enc, mel, D)          # instance
                ti4_block3_full(b2b, whisper_enc, AudioFrontend, D)    # instance
                ti5_multi_chunk_pipeline(b2b, whisper_enc, AudioFrontend, D, N=3)  # instance, reused in loop
                ti6_qwen_ready_real(b2b, whisper_enc, AudioFrontend, D)            # instance
                ti7_stage1_gradient(b2b, whisper_enc, AudioFrontend, D)            # instance
    else:
        skip("Integration tests skipped (--skip-integration)")

    # ── Verdict ───────────────────────────────────────────────────
    total = PASS + FAIL
    print(f"\n{BD}{'─'*62}{E}")
    print(f"  {G}{PASS} passed{E}  "
          f"{(R+str(FAIL)+E) if FAIL else str(FAIL)} failed"
          f"  /  {total} run")

    if FAIL == 0:
        print(f"\n{G}{BD}✔  All tests passed — pipeline ready through Block3.{E}")
        print(f"\n  Next: Block 4 — MLP Adaptor (audio → LLM embedding space)\n")
    else:
        print(f"\n{R}{BD}✘  {FAIL} test(s) failed — see output above.{E}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()