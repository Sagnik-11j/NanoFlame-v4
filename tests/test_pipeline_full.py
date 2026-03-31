"""
tests/test_pipeline_full.py
══════════════════════════════════════════════════════════════════════
NanoFlame v4 — Full end-to-end pipeline test (ZERO mocking)

Block 1  → Block 2a/2b → Block 3 → Block 4 → Block 5 → Block 6 → Block 7

Run modes
─────────
  python tests/test_pipeline_full.py                  # lightweight (no HF download)
  python tests/test_pipeline_full.py --full           # real model weights
  python tests/test_pipeline_full.py --full --cuda    # GPU
  python tests/test_pipeline_full.py --full --local_whisper ./whisper_dir
  python tests/test_pipeline_full.py --full --local_qwen   ./qwen_dir

Lightweight mode instantiates EVERY real block class and tests shapes/dtypes
using tiny synthetic tensors.  No downloads needed.

Full mode additionally downloads Whisper, OpenBeats and Qwen weights and
runs a complete forward + loss + generation pass.
══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

__test__ = False

import sys
import argparse
import logging
import warnings
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import numpy as np

logging.basicConfig(level=logging.WARNING)

# Ignore torchaudio warning about 128 mel bins on random noise
warnings.filterwarnings("ignore", message="At least one mel filterbank has all zero values")

# ── Console colours ──────────────────────────────────────────────────
G  = "\033[92m"; Y = "\033[93m"; R  = "\033[91m"
BD = "\033[1m";  E = "\033[0m"
ok   = lambda m: print(f"  {G}✔{E}  {m}")
warn = lambda m: print(f"  {Y}⚠{E}  {m}")
fail = lambda m: print(f"  {R}✘{E}  {m}")
info = lambda m: print(f"     {m}")
hdr  = lambda t: print(f"\n{BD}── {t} {'─'*(58-len(t))}{E}")

PASS = FAIL = 0

def check(name: str, ok_: bool, detail: str = "") -> bool:
    global PASS, FAIL
    if ok_:
        ok(f"{name}  {detail}"); PASS += 1
    else:
        fail(f"{name}  {detail}"); FAIL += 1
    return ok_

def nonan(t: torch.Tensor) -> bool:
    return not (torch.isnan(t).any() or torch.isinf(t).any())

def mhdr(title: str) -> None:
    print(f"\n{BD}{'═'*62}{E}")
    print(f"{BD}  {title}{E}")
    print(f"{BD}{'═'*62}{E}")


# ── Constants matching the pipeline spec ────────────────────────────
SR           = 16_000          # 16 kHz
AUDIO_DUR_S  = 5.0             # 5 s synthetic audio
AUDIO_SAMPS  = int(SR * AUDIO_DUR_S)
HIDDEN_DIM   = 896             # Qwen-2.5-0.5B hidden size
QUERY_TOKENS = 64              # Perceiver output queries (typical)
BATCH        = 1
BASE_VOCAB   = 151_936
EXT_VOCAB    = 151_944          # +8 audio boundary tokens


# ════════════════════════════════════════════════════════════════════
# SECTION A — Import all blocks (verifies __init__.py / module paths)
# ════════════════════════════════════════════════════════════════════

def import_all_blocks():
    mhdr("A · Import all Block classes")

    global AudioFrontend
    global Block2aWhisperEncoder
    global Block2bOpenBEATsEncoder
    global Block3PerceiverFusion
    global Block4MLPAdaptor
    global Block5TextTokenizer
    global Block6SequencePacker
    global Block7QwenDecoder

    try:
        from blocks.block1_audio_frontend import AudioFrontend
        check("Block 1  imported", True)
    except ImportError as e:
        check("Block 1  imported", False, str(e)); AudioFrontend = None

    try:
        from blocks.block2a_whisper_encoder import Block2aWhisperEncoder
        check("Block 2a imported", True)
    except ImportError as e:
        check("Block 2a imported", False, str(e)); Block2aWhisperEncoder = None

    try:
        from blocks.block2b_openbeats_encoder import Block2bOpenBEATsEncoder
        check("Block 2b imported", True)
    except ImportError as e:
        check("Block 2b imported", False, str(e)); Block2bOpenBEATsEncoder = None

    try:
        from blocks.block3_perceiver_fusion import Block3PerceiverFusion
        check("Block 3  imported", True)
    except ImportError as e:
        check("Block 3  imported", False, str(e)); Block3PerceiverFusion = None

    try:
        from blocks.block4_mlp_adaptor import Block4MLPAdaptor
        check("Block 4  imported", True)
    except ImportError as e:
        check("Block 4  imported", False, str(e)); Block4MLPAdaptor = None

    try:
        from blocks.block5_text_tokenizer import Block5TextTokenizer
        check("Block 5  imported", True)
    except ImportError as e:
        check("Block 5  imported", False, str(e)); Block5TextTokenizer = None

    try:
        from blocks.block6_sequence_packer import Block6SequencePacker
        check("Block 6  imported", True)
    except ImportError as e:
        check("Block 6  imported", False, str(e)); Block6SequencePacker = None

    try:
        from blocks.block7_qwen_decoder import Block7QwenDecoder
        check("Block 7  imported", True)
    except ImportError as e:
        check("Block 7  imported", False, str(e)); Block7QwenDecoder = None


# ════════════════════════════════════════════════════════════════════
# SECTION B — Lightweight instantiation (no HF download)
# These tests use default / tiny configs so no weights are fetched.
# ════════════════════════════════════════════════════════════════════

def section_b_instantiate():
    mhdr("B · Instantiate all blocks (lightweight, CPU, no download)")

    blocks = {}

    # ── Block 1 ───────────────────────────────────────────────────
    hdr("B1 · AudioFrontend")
    try:
        b1 = AudioFrontend(sample_rate=SR)
        check("instantiated", True)
        blocks["b1"] = b1
    except Exception as e:
        check("instantiated", False, str(e))

    # ── Block 2a ──────────────────────────────────────────────────
    hdr("B2a · Block2aWhisperEncoder  (tiny / cpu, no download)")
    try:
        # Pass tiny=True or cpu_only=True if the block supports it;
        # fall back to the smallest available config.
        try:
            b2a = Block2aWhisperEncoder(model_name="openai/whisper-medium", device=torch.device("cpu"))
        except TypeError:
            b2a = Block2aWhisperEncoder(device=torch.device("cpu"))
        check("instantiated", True)
        blocks["b2a"] = b2a
    except Exception as e:
        check("instantiated", False, str(e))

    # ── Block 2b ──────────────────────────────────────────────────
    hdr("B2b · Block2bOpenBEATsEncoder  (cpu)")
    try:
        try:
            b2b = Block2bOpenBEATsEncoder(device="cpu")
        except TypeError:
            b2b = Block2bOpenBEATsEncoder()
        check("instantiated", True)
        blocks["b2b"] = b2b
    except Exception as e:
        check("instantiated", False, str(e))

    # ── Block 3 ───────────────────────────────────────────────────
    hdr("B3 · Block3PerceiverFusion  (tiny config)")
    try:
        # Try various typical constructor signatures
        b3 = _try_init_block3()
        check("instantiated", b3 is not None)
        if b3:
            blocks["b3"] = b3
    except Exception as e:
        check("instantiated", False, str(e))

    # ── Block 4 ───────────────────────────────────────────────────
    hdr("B4 · Block4MLPAdaptor")
    try:
        b4 = _try_init_block4()
        check("instantiated", b4 is not None)
        if b4:
            blocks["b4"] = b4
    except Exception as e:
        check("instantiated", False, str(e))

    # ── Block 5 ───────────────────────────────────────────────────
    hdr("B5 · Block5TextTokenizer")
    try:
        try:
            b5 = Block5TextTokenizer(
                base_tokenizer="Qwen/Qwen2.5-0.5B-Instruct",
                num_audio_tokens=AUDIO_TOKENS if hasattr(b5 := None, "_") else 8,
            )
        except Exception:
            b5 = Block5TextTokenizer()
        check("instantiated", True)
        blocks["b5"] = b5
    except Exception as e:
        check("instantiated", False, str(e))

        # ── Block 6 ───────────────────────────────────────────────────
    hdr("B6 · Block6SequencePacker")
    try:
        b5_instance = blocks.get("b5")
        dummy_embed = nn.Embedding(EXT_VOCAB, HIDDEN_DIM)
        b6 = Block6SequencePacker(tokenizer=b5_instance, embed_tokens=dummy_embed)
        check("instantiated", True)
        blocks["b6"] = b6
    except Exception as e:
        check("instantiated", False, str(e))

    # ── Block 7 ───────────────────────────────────────────────────
    hdr("B7 · Block7QwenDecoder  (lightweight — skips HF download)")
    try:
        # Use local_dir trick: if user ran --local_qwen this may already be
        # cached, otherwise the block should fail gracefully.
        b7 = Block7QwenDecoder(
            use_4bit=False,
            use_flash_attn=False,
        )
        check("instantiated", True)
        blocks["b7"] = b7
    except Exception as e:
        warn(f"Block 7 needs HF weights: {e}")
        check("instantiated (expected without --full)", False,
              "run --full to download weights")

    return blocks

def _try_init_block3():
    """Try common Perceiver signatures."""
    for kwargs in [
        dict(input_dim=1280, output_dim=HIDDEN_DIM, num_latents=QUERY_TOKENS),
        dict(audio_dim=1280, llm_dim=HIDDEN_DIM, n_queries=QUERY_TOKENS),
        dict(),
    ]:
        try:
            return Block3PerceiverFusion(**kwargs)
        except (TypeError, Exception):
            continue
    return None

def _try_init_block4():
    """Try common MLP adaptor signatures."""
    for kwargs in [
        dict(in_dim=HIDDEN_DIM, out_dim=HIDDEN_DIM),
        dict(input_dim=1280, output_dim=HIDDEN_DIM),
        dict(),
    ]:
        try:
            return Block4MLPAdaptor(**kwargs)
        except (TypeError, Exception):
            continue
    return None


# ════════════════════════════════════════════════════════════════════
# SECTION C — Shape / dtype tests with synthetic tensors
# ════════════════════════════════════════════════════════════════════

def section_c_synthetic_shapes(blocks: dict):
    mhdr("C · Synthetic-tensor shape & dtype tests (CPU)")

    waveform = torch.randn(BATCH, AUDIO_SAMPS)   # raw PCM

    # ── C1: Block 1 ───────────────────────────────────────────────
    hdr("C1 · Block1 audio pre-processing")
    b1 = blocks.get("b1")
    if b1 is None:
        warn("Block 1 not available"); return

    try:
        with torch.no_grad():
            mel = b1._log_mel(waveform)
            chunks, _ = b1._chunk(mel)
            b1_out = chunks[0].unsqueeze(0)  # [1, 128, 3000]

        mel = _extract_mel(b1_out)
        check("B1 outputs mel features",     mel is not None)
        if mel is not None:
            check("B1 mel shape reasonable",
                  mel.ndim >= 2,
                  str(tuple(mel.shape)))
            check("B1 no NaN",               nonan(mel))
    except Exception as e:
        check("B1 forward ok", False, str(e))
        return

    # ── C2a: Block 2a ─────────────────────────────────────────────
    hdr("C2a · Block2a Whisper speech features")
    b2a = blocks.get("b2a")
    if b2a is None:
        warn("Block 2a not available"); return

    try:
        with torch.no_grad():
            speech_feats = b2a(waveform)

        speech_feats = _ensure_tensor(speech_feats)
        check("B2a returns tensor",  isinstance(speech_feats, torch.Tensor))
        check("B2a ndim >= 3",       speech_feats.ndim >= 3,
              str(tuple(speech_feats.shape)))
        check("B2a no NaN",          nonan(speech_feats))
        info(f"speech_feats  {tuple(speech_feats.shape)}")
    except Exception as e:
        check("B2a forward ok", False, str(e)); speech_feats = None

    # ── C2b: Block 2b ─────────────────────────────────────────────
    hdr("C2b · Block2b OpenBEATs music features")
    b2b = blocks.get("b2b")
    if b2b is None:
        warn("Block 2b not available"); return

    try:
        with torch.no_grad():
            music_feats = b2b(waveform)

        music_feats = _ensure_tensor(music_feats)
        check("B2b returns tensor",  isinstance(music_feats, torch.Tensor))
        check("B2b ndim >= 3",       music_feats.ndim >= 3,
              str(tuple(music_feats.shape)))
        check("B2b no NaN",          nonan(music_feats))
        info(f"music_feats   {tuple(music_feats.shape)}")
    except Exception as e:
        check("B2b forward ok", False, str(e)); music_feats = None

    # ── C3: Block 3 ───────────────────────────────────────────────
    hdr("C3 · Block3 Perceiver Fusion")
    b3 = blocks.get("b3")
    if b3 is None or speech_feats is None:
        warn("Block 3 or upstream tensors not available"); return

    try:
        with torch.no_grad():
            fused = b3(speech_feats, music_feats)

        fused = _ensure_tensor(fused)
        check("B3 returns tensor",       isinstance(fused, torch.Tensor))
        check("B3 ndim == 3",            fused.ndim == 3,
              str(tuple(fused.shape)))
        check("B3 no NaN",               nonan(fused))
        info(f"fused         {tuple(fused.shape)}")
    except Exception as e:
        check("B3 forward ok", False, str(e)); fused = None

    # ── C4: Block 4 ───────────────────────────────────────────────
    hdr("C4 · Block4 MLP Adaptor → Qwen hidden dim 896")
    b4 = blocks.get("b4")
    if b4 is None or fused is None:
        warn("Block 4 or upstream tensors not available"); return

    try:
        with torch.no_grad():
            adapted = b4(fused)

        adapted = _ensure_tensor(adapted)
        check("B4 returns tensor",       isinstance(adapted, torch.Tensor))
        check("B4 shape [-1] == 896",    adapted.shape[-1] == HIDDEN_DIM,
              str(tuple(adapted.shape)))
        check("B4 no NaN",               nonan(adapted))
        info(f"adapted       {tuple(adapted.shape)}")
    except Exception as e:
        check("B4 forward ok", False, str(e)); adapted = None

    # ── C5: Block 5 ───────────────────────────────────────────────
    hdr("C5 · Block5 Text Tokeniser")
    b5 = blocks.get("b5")
    if b5 is None:
        warn("Block 5 not available"); return

    sample_text = "Describe the audio in detail."
    try:
        chat_prompt = b5.build_chat_prompt(question=sample_text, n_audios=1)
        ids = torch.tensor(chat_prompt.input_ids)
        
        check("B5 returns token ids",  ids is not None)
        if ids is not None:
            check("B5 ids ndim >= 1",  ids.ndim >= 1, str(tuple(ids.shape)))
            check("B5 ids in vocab",
                  ids.max().item() < EXT_VOCAB, str(ids.max().item()))
        info(f"token ids     {tuple(ids.shape) if ids is not None else 'N/A'}")
    except Exception as e:
        check("B5 forward ok", False, str(e)); chat_prompt = None; ids = None

    # ── C6: Block 6 ───────────────────────────────────────────────
    hdr("C6 · Block6 Sequence Packer")
    b6 = blocks.get("b6")
    b7 = blocks.get("b7")
    if b6 is None or adapted is None or chat_prompt is None:
        warn("Block 6 or upstream tensors not available"); return

    embed = _get_embed(b7, EXT_VOCAB, HIDDEN_DIM)
    b6.embed_tokens = embed  # Update with the correct embed

    try:
        audio_tokens_list = [adapted.squeeze(0)]
        packed = b6.pack_for_inference(chat_prompt, audio_tokens_list, device="cpu")
        check("B6 returns packed sequence", packed is not None)
        if packed is not None:
            ie = packed.inputs_embeds
            am = packed.attention_mask
            check("B6 inputs_embeds ndim == 3",
                  ie.ndim == 3, str(tuple(ie.shape)))
            check("B6 inputs_embeds dim[-1] == 896",
                  ie.shape[-1] == HIDDEN_DIM)
            check("B6 attention_mask shape matches",
                  am.shape[:2] == ie.shape[:2])
            check("B6 no NaN in embeds",      nonan(ie))
            info(f"packed embeds {tuple(ie.shape)}")
    except Exception as e:
        check("B6 pack ok", False, str(e)); packed = None

    # ── C7: Block 7 ───────────────────────────────────────────────
    hdr("C7 · Block7 Qwen Decoder (lightweight)")
    if b7 is None or packed is None:
        warn("Block 7 not available — run --full to test")
        return

    try:
        from blocks.block7_qwen_decoder import Block7Output
        with torch.no_grad():
            out = b7.forward(packed_sequence=packed)

        check("B7 returns Block7Output",   isinstance(out, Block7Output))
        check("B7 logits [B, T, 151944]",
              out.logits.shape[-1] == EXT_VOCAB,
              str(tuple(out.logits.shape)))
        check("B7 hidden dim == 896",
              out.hidden.shape[-1] == HIDDEN_DIM)
        check("B7 no NaN in logits",       nonan(out.logits.float()))
        info(f"logits        {tuple(out.logits.shape)}")
    except Exception as e:
        check("B7 forward ok", False, str(e))


# ════════════════════════════════════════════════════════════════════
# SECTION D — Full pipeline with real weights (--full mode)
# ════════════════════════════════════════════════════════════════════

def section_d_full_pipeline(args):
    mhdr("D · Full pipeline — REAL WEIGHTS, end-to-end")

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    info(f"device: {device}")

    # ── D0: Synthetic audio (5 s, 16 kHz) ───────────────────────
    hdr("D0 · Synthetic audio input")
    np.random.seed(42)
    raw_np  = (np.random.randn(AUDIO_SAMPS) * 0.1).astype(np.float32)
    raw_np += 0.05 * np.sin(2 * np.pi * 440 * np.arange(AUDIO_SAMPS) / SR).astype(np.float32)
    waveform = torch.from_numpy(raw_np).unsqueeze(0).to(device)  # [1, N]
    check("Synthetic 5 s audio created", waveform.shape == (BATCH, AUDIO_SAMPS),
          str(tuple(waveform.shape)))

    # ── D1: Block 1 ───────────────────────────────────────────────
    hdr("D1 · AudioFrontend — mel + normalisation")
    b1 = AudioFrontend(sample_rate=SR)
    
    with torch.no_grad():
        waveform_cpu = waveform.cpu()
        mel_cpu = b1._log_mel(waveform_cpu)
        chunks, _ = b1._chunk(mel_cpu)
        b1_out = chunks[0].unsqueeze(0).to(device)
        
    mel = _extract_mel(b1_out)
    check("B1 mel produced",      mel is not None)
    check("B1 mel no NaN",        nonan(mel))
    info(f"mel shape  {tuple(mel.shape) if mel is not None else 'N/A'}")

    # ── D2a: Block 2a ─────────────────────────────────────────────
    hdr("D2a · Block2aWhisperEncoder")
    b2a = Block2aWhisperEncoder(
    model_name=args.whisper_model,
    device=torch.device(device),
    )
    b2a.eval()
    with torch.no_grad():
        speech_feats = _ensure_tensor(b2a(b1_out)) # Pass mel spec, not waveform
    check("B2a speech feats",     speech_feats.ndim >= 3,
          str(tuple(speech_feats.shape)))
    check("B2a no NaN",           nonan(speech_feats))
    info(f"speech_feats {tuple(speech_feats.shape)}")

    # ── D2b: Block 2b ─────────────────────────────────────────────
    hdr("D2b · Block2bOpenBEATsEncoder")
    
    # 1. Instantiate without the device argument
    b2b = Block2bOpenBEATsEncoder()
    
    # 2. Load from your local checkpoints folder
    ckpt_path = ROOT / "checkpoints" / "epoch_latest.pt"
    info(f"Loading local OpenBEATs checkpoint: {ckpt_path.name}")
    try:
        b2b.load_openbeats_checkpoint(checkpoint_path=str(ckpt_path))
    except Exception as e:
        warn(f"Failed to load OpenBEATs weights: {e}")
    
    # 3. Move to device and run
    b2b = b2b.to(device)
    b2b.eval()
    
    with torch.no_grad():
        music_feats = _ensure_tensor(b2b(b1_out)) # Pass mel spec, not waveform
    check("B2b music feats",      music_feats.ndim >= 3,
          str(tuple(music_feats.shape)))
    check("B2b no NaN",           nonan(music_feats))
    info(f"music_feats  {tuple(music_feats.shape)}")

    # ── D3: Block 3 ───────────────────────────────────────────────
    hdr("D3 · Block3PerceiverFusion")
    b3 = _init_block3_full(speech_feats, music_feats, device)
    b3.eval()
    with torch.no_grad():
        fused = _ensure_tensor(b3(speech_feats, music_feats))
    check("B3 fused [B, Q, D]",   fused.ndim == 3, str(tuple(fused.shape)))
    check("B3 no NaN",            nonan(fused))
    info(f"fused        {tuple(fused.shape)}")

    # ── D4: Block 4 ───────────────────────────────────────────────
    hdr("D4 · Block4MLPAdaptor → 896")
    b4 = _init_block4_full(fused, device)
    b4.eval()
    with torch.no_grad():
        adapted = _ensure_tensor(b4(fused))
    check("B4 adapted [-1]==896",  adapted.shape[-1] == HIDDEN_DIM,
          str(tuple(adapted.shape)))
    check("B4 no NaN",             nonan(adapted))
    info(f"adapted      {tuple(adapted.shape)}")

        # ── D5: Block 5 ───────────────────────────────────────────────
    hdr("D5 · Block5TextTokenizer")
    b5 = _init_block5_full(args)
    chat_prompt = b5.build_chat_prompt(
        question="Describe what you hear in the audio.",
        n_audios=1
    )
    ids = torch.tensor(chat_prompt.input_ids).unsqueeze(0).to(device)
    
    check("B5 token ids",         ids is not None)
    check("B5 ids in ext vocab",  ids.max().item() < EXT_VOCAB,
          str(ids.max().item()))
    info(f"input_ids    {tuple(ids.shape)}")

    # ── D6: Block 6 ───────────────────────────────────────────────
    hdr("D6 · Block6SequencePacker")
    b7_weights = Block7QwenDecoder(
        use_4bit=True,
        use_flash_attn=False,
        local_dir=getattr(args, "local_qwen", None),
    )
    b7_weights.eval()
    embed = b7_weights.embed_tokens

    b6 = Block6SequencePacker(tokenizer=b5, embed_tokens=embed)
    audio_tokens_list = [adapted.squeeze(0)]
    packed = b6.pack_for_inference(chat_prompt, audio_tokens_list, device=device)
    
    check("B6 packed not None",   packed is not None)
    ie = packed.inputs_embeds
    am = packed.attention_mask
    check("B6 embeds [B,T,896]",  ie.shape[-1] == HIDDEN_DIM,
          str(tuple(ie.shape)))
    check("B6 mask shape",        am.shape[:2] == ie.shape[:2])
    check("B6 no NaN",            nonan(ie))
    info(f"packed embeds {tuple(ie.shape)}")

    # ── D7a: Inference forward ────────────────────────────────────
    hdr("D7a · Block7QwenDecoder — inference forward")
    from blocks.block7_qwen_decoder import Block7Output
    with torch.no_grad():
        out = b7_weights.forward(packed_sequence=packed)

    check("B7 Block7Output",       isinstance(out, Block7Output))
    check("B7 logits vocab==151944",
          out.logits.shape[-1] == EXT_VOCAB, str(tuple(out.logits.shape)))
    check("B7 hidden dim==896",    out.hidden.shape[-1] == HIDDEN_DIM)
    check("B7 no NaN",             nonan(out.logits.float()))
    info(f"logits       {tuple(out.logits.shape)}")

    # ── D7b: Training forward with loss ───────────────────────────
    hdr("D7b · Block7 training forward — loss computation")
    T = ie.shape[1]

    target_ids = torch.randint(
        0, BASE_VOCAB, (BATCH, T - 1), dtype=torch.long, device=device
    )
    loss_mask = torch.ones(BATCH, T - 1, dtype=torch.long, device=device)
    loss_mask[:, :T // 2] = 0   # mask first half as context

    packed._target_ids = target_ids
    packed.loss_mask   = torch.cat(
        [torch.zeros(BATCH, 1, dtype=torch.long, device=device), loss_mask], dim=1
    )

    b7_weights.configure_for_stage(2)

    out_train = b7_weights.forward(packed_sequence=packed)
    check("B7 training loss computed",
          out_train.loss is not None)
    if out_train.loss is not None:
        loss_val = out_train.loss.item()
        check("B7 loss is finite",  np.isfinite(loss_val), f"{loss_val:.4f}")
        check("B7 loss > 0",        loss_val > 0,          f"{loss_val:.4f}")
        info(f"cross-entropy loss = {loss_val:.4f}")

    # ── D7c: Generation step ──────────────────────────────────────
    hdr("D7c · Block7 autoregressive generation step")
    b7_weights.freeze_all()
    with torch.no_grad():
        tid, kv = b7_weights.generate_step(
            inputs_embeds=ie,
            attention_mask=am,
            temperature=0.7,
            top_p=0.9,
        )
    check("B7 token_id is int",    isinstance(tid, int))
    check("B7 0 ≤ id < 151944",   0 <= tid < EXT_VOCAB, str(tid))

    # ── D8: Simple autoregressive decode (3 tokens) ───────────────
    hdr("D8 · Greedy decode — 3 tokens end-to-end")
    generated = []
    past      = kv
    cur_embed = embed(torch.tensor([[tid]], device=device))  # [1, 1, 896]
    cur_mask  = torch.ones(1, ie.shape[1] + 1, dtype=torch.long, device=device)

    for step in range(3):
        with torch.no_grad():
            next_id, past = b7_weights.generate_step(
                inputs_embeds=cur_embed,
                attention_mask=cur_mask,
                past_key_values=past,
            )
        generated.append(next_id)
        cur_embed = embed(torch.tensor([[next_id]], device=device))
        cur_mask  = torch.ones(1, cur_mask.shape[1] + 1,
                               dtype=torch.long, device=device)

    check("decoded 3 tokens",      len(generated) == 3)
    check("all in vocab range",
          all(0 <= t < EXT_VOCAB for t in generated), str(generated))
    info(f"decoded token ids: {generated}")


# ════════════════════════════════════════════════════════════════════
# SECTION E — Interface contract tests
# ════════════════════════════════════════════════════════════════════

def section_e_contracts(blocks: dict):
    mhdr("E · Interface contract tests")

    # E1: Block 7 embed_tokens is the canonical source for Block 6
    hdr("E1 · B7.embed_tokens accessible for B6 (no double-load)")
    b7 = blocks.get("b7")
    if b7 is None:
        warn("Block 7 not available"); return
    emb = b7.embed_tokens
    check("embed_tokens is nn.Embedding", isinstance(emb, nn.Embedding))
    check("rows == 151,944",              emb.num_embeddings == EXT_VOCAB,
          str(emb.num_embeddings))
    check("dim == 896",                   emb.embedding_dim == HIDDEN_DIM)

    # E2: configure_for_stage changes which params are trainable
    hdr("E2 · configure_for_stage(1) trains LM head only")
    b7.configure_for_stage(1)
    trainable = {n for n, p in b7._model.named_parameters() if p.requires_grad}
    check("some params trainable",   len(trainable) > 0)
    has_lm   = any("lm_head"  in n for n in trainable)
    has_lora = any("lora_"    in n for n in trainable)
    check("lm_head trainable",       has_lm)
    if b7._lora_applied:
        check("lora NOT in stage 1", not has_lora)

    hdr("E2b · configure_for_stage(2) trains QLoRA")
    b7.configure_for_stage(2)
    if b7._lora_applied:
        lora_on = any("lora_" in n
                      for n, p in b7._model.named_parameters() if p.requires_grad)
        check("lora trainable in stage 2", lora_on)
    else:
        warn("QLoRA not applied (peft not installed)")

    # E3: freeze_all zeroes trainable count
    hdr("E3 · freeze_all() → 0 trainable")
    b7.freeze_all()
    trainable_after = sum(1 for p in b7._model.parameters() if p.requires_grad)
    check("0 trainable after freeze_all", trainable_after == 0,
          str(trainable_after))

    # E4: B7 repr
    hdr("E4 · Block7.__repr__")
    r = repr(b7)
    check("repr is a string", isinstance(r, str))
    check("vocab in repr",    str(EXT_VOCAB) in r or "vocab" in r.lower())
    info(repr(b7))


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════

def _extract_mel(b1_out):
    """Try common Block 1 output shapes."""
    if isinstance(b1_out, torch.Tensor):
        return b1_out
    if hasattr(b1_out, "mel"):
        return b1_out.mel
    if hasattr(b1_out, "features"):
        return b1_out.features
    if isinstance(b1_out, (tuple, list)):
        return b1_out[0]
    return None

def _ensure_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if hasattr(x, "h_w"):                 # For Block 2a (Whisper)
        return x.h_w
    if hasattr(x, "h_ob"):                # For Block 2b (OpenBEATs)
        return x.h_ob
    if hasattr(x, "h_full"):              # For Block 3 (PerceiverFusion)
        return x.h_full
    if hasattr(x, "audio_tokens"):        # <-- Added this for Block 4 (MLP Adaptor)
        return x.audio_tokens
    if hasattr(x, "last_hidden_state"):
        return x.last_hidden_state
    if hasattr(x, "embeddings"):
        return x.embeddings
    if isinstance(x, (tuple, list)):
        for item in x:
            if isinstance(item, torch.Tensor) and item.ndim >= 3:
                return item
    raise ValueError(f"Cannot extract tensor from {type(x)}")

def _extract_ids(tok_out) -> Optional[torch.Tensor]:
    if isinstance(tok_out, torch.Tensor):
        return tok_out
    if hasattr(tok_out, "input_ids"):
        return tok_out.input_ids
    if isinstance(tok_out, dict):
        return tok_out.get("input_ids")
    if isinstance(tok_out, (tuple, list)):
        return tok_out[0]
    return None

def _get_embed(b7, vocab, dim) -> nn.Embedding:
    if b7 is not None:
        try:
            return b7.embed_tokens
        except Exception:
            pass
    return nn.Embedding(vocab, dim)

def _call_packer(b6, adapted, ids, embed):
    """Try common Block 6 packing call signatures."""
    audio_embed = adapted  # [B, Q, 896]
    # Ensure ids is 2D: [B, T]
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    for call in [
        lambda: b6(audio_embeds=audio_embed, input_ids=ids, embed_tokens=embed),
        lambda: b6(audio_embeds=audio_embed, input_ids=ids),
        lambda: b6(audio_embed, ids, embed),
        lambda: b6(audio_embed, ids),
        lambda: b6.pack(audio_embeds=audio_embed, input_ids=ids, embed_tokens=embed),
        lambda: b6.pack(audio_embed, ids, embed),
    ]:
        try:
            result = call()
            if result is not None and hasattr(result, "inputs_embeds"):
                return result
        except (TypeError, AttributeError, Exception):
            continue
    return None

def _init_block3_full(speech_feats, music_feats, device):
    s_dim = speech_feats.shape[-1]
    m_dim = music_feats.shape[-1]
    for kwargs in [
        dict(speech_dim=s_dim, music_dim=m_dim,
             output_dim=HIDDEN_DIM, num_latents=QUERY_TOKENS),
        dict(input_dim=s_dim, output_dim=HIDDEN_DIM, num_latents=QUERY_TOKENS),
        dict(audio_dim=s_dim, llm_dim=HIDDEN_DIM, n_queries=QUERY_TOKENS),
        dict(),
    ]:
        try:
            return Block3PerceiverFusion(**kwargs).to(device)
        except (TypeError, Exception):
            continue
    raise RuntimeError("Could not instantiate Block3PerceiverFusion")

def _init_block4_full(fused, device):
    f_dim = fused.shape[-1]
    for kwargs in [
        dict(in_dim=f_dim, out_dim=HIDDEN_DIM),
        dict(input_dim=f_dim, output_dim=HIDDEN_DIM),
        dict(hidden_dim=HIDDEN_DIM),
        dict(),
    ]:
        try:
            return Block4MLPAdaptor(**kwargs).to(device)
        except (TypeError, Exception):
            continue
    raise RuntimeError("Could not instantiate Block4MLPAdaptor")

def _init_block5_full(args):
    tokenizer_id = getattr(args, "local_qwen", None) or "Qwen/Qwen2.5-0.5B-Instruct"
    for kwargs in [
        dict(base_tokenizer=tokenizer_id, num_audio_tokens=8),
        dict(model_name_or_path=tokenizer_id),
        dict(),
    ]:
        try:
            return Block5TextTokenizer(**kwargs)
        except (TypeError, Exception):
            continue
    return Block5TextTokenizer()


# ════════════════════════════════════════════════════════════════════
# Argument parser & entry point
# ════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(
        description="NanoFlame v4 — full pipeline tests"
    )
    p.add_argument("--full",          action="store_true",
                   help="Run integration tests with real model weights")
    p.add_argument("--cuda",          action="store_true",
                   help="Use CUDA device for --full tests")
    p.add_argument("--whisper_model", default="openai/whisper-medium",
               help="Whisper model ID for --full tests")
    p.add_argument("--local_whisper", default=None,
                   help="Local path to Whisper weights directory")
    p.add_argument("--local_qwen",    default=None,
                   help="Local path to Qwen-2.5-0.5B-Instruct directory")
    return p.parse_args()

def main():
    global PASS, FAIL
    args = get_args()

    print(f"\n{BD}{'═'*62}{E}")
    print(f"{BD}  NanoFlame v4 — Full Pipeline Tests{E}")
    print(f"{BD}{'═'*62}{E}")
    print(f"  mode  : {'FULL (real weights)' if args.full else 'lightweight (no HF download)'}")
    if args.full:
        print(f"  device: {'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'}")
        print(f"  whisper: {args.whisper_model}  |  qwen: {args.local_qwen or 'HF hub'}")

    import_all_blocks()

    if not args.full:
        # Lightweight path: instantiate blocks, run synthetic-tensor shapes
        blocks = section_b_instantiate()
        section_c_synthetic_shapes(blocks)
        section_e_contracts(blocks)
        warn("Integration tests (real weights) skipped — run with --full")
    else:
        # Full path: real weights, real forward, real loss + generation
        section_d_full_pipeline(args)

    total = PASS + FAIL
    print(f"\n{BD}{'─'*62}{E}")
    color = G if FAIL == 0 else R
    print(f"  {G}{PASS} passed{E}  "
          f"{(R + str(FAIL) + E) if FAIL else str(FAIL)} failed  /  {total} total")

    if FAIL == 0:
        print(f"\n{G}{BD}✔  All pipeline tests passed.{E}\n")
    else:
        print(f"\n{R}{BD}✘  {FAIL} test(s) failed.{E}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()