"""
tests/test_block6.py
══════════════════════════════════════════════════════════════════════
NanoFlame v4 — Block 6 unit + integration + full pipeline tests

Run:
    python tests/test_block6.py
    python tests/test_block6.py --skip-integration
    python tests/test_block6.py --skip-pipeline
    python tests/test_block6.py --no-real-embed     # mock embed (fast unit tests)
    python tests/test_block6.py --ckpt checkpoints/epoch_latest.pt
    python tests/test_block6.py --device cuda
"""

__test__ = False

import sys
import gc
import math
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
HIDDEN = 896


def check(name: str, ok_: bool, detail: str = "") -> bool:
    global PASS, FAIL
    if ok_:
        ok(f"{name}  {detail}"); PASS += 1
    else:
        fail(f"{name}  {detail}"); FAIL += 1
    return ok_


def nonan(t: torch.Tensor) -> bool:
    return not (torch.isnan(t).any() or torch.isinf(t).any())


# ─────────────────────────────────────────────────────────────────────
# Fixtures — basic
# ─────────────────────────────────────────────────────────────────────

def make_tok():
    from blocks.block5_text_tokenizer import Block5TextTokenizer
    return Block5TextTokenizer()


def load_real_embed_tokens(tok, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """
    Load Qwen-2.5-0.5B embed_tokens, resize for Block5 audio tokens,
    then free the rest of the model to save RAM.
    """
    from transformers import AutoModelForCausalLM

    info(f"Loading real embed_tokens from {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.resize_token_embeddings(tok.vocab_size)

    embed_state = {k: v.clone() for k, v in
                   model.model.embed_tokens.state_dict().items()}
    del model
    gc.collect()

    real_embed = nn.Embedding(tok.vocab_size, HIDDEN)
    real_embed.load_state_dict(embed_state)
    real_embed.eval()
    ok(f"Real embed_tokens loaded  (vocab={tok.vocab_size}, dim={HIDDEN})")
    return real_embed


def make_mock_embed(tok):
    emb = nn.Embedding(tok.vocab_size, HIDDEN)
    nn.init.normal_(emb.weight, std=0.02)
    return emb


def make_packer(tok, embed):
    from blocks.block6_sequence_packer import Block6SequencePacker
    return Block6SequencePacker(tok, embed)


def make_audio(T=750, n=1):
    return [torch.randn(T, HIDDEN) for _ in range(n)]


# ─────────────────────────────────────────────────────────────────────
# Fixtures — full pipeline helpers
# ─────────────────────────────────────────────────────────────────────

def make_fake_audio_file(path: Path, seconds: float = 30.0, sr: int = 16000):
    """Synthesise a 440 Hz sine wave and save as WAV."""
    import torchaudio
    n   = int(sr * seconds)
    t   = torch.linspace(0, seconds, n)
    wav = (0.4 * torch.sin(2 * math.pi * 440 * t)).unsqueeze(0)   # [1, N]
    torchaudio.save(str(path), wav, sr)


def get_block1_class():
    from blocks.block1_audio_frontend import AudioFrontend
    return AudioFrontend


def get_block2a_class():
    """Dynamic import — handles any of the exported class names."""
    import importlib
    mod = importlib.import_module("blocks.block2a_whisper_encoder")
    for name in ["Block2aWhisperEncoder", "WhisperEncoder",
                 "WhisperMediumEncoder", "Block2aEncoder"]:
        if hasattr(mod, name):
            return getattr(mod, name)
    public = [x for x in dir(mod) if not x.startswith("_") and
              isinstance(getattr(mod, x), type)]
    raise ImportError(
        f"No usable WhisperEncoder class found in block2a_whisper_encoder. "
        f"Available: {public}"
    )


def load_block2b(ckpt: str, device: str):
    from blocks.block2b_openbeats_encoder import Block2bOpenBEATsEncoder
    enc = Block2bOpenBEATsEncoder()
    path = Path(ckpt)
    if not path.exists():
        warn(f"Checkpoint not found: {path}")
        return None, False
    try:
        if hasattr(enc, "load_openbeats_checkpoint"):
            missing, unexpected = enc.load_openbeats_checkpoint(str(path), strict=False)
            enc = enc.to(device).eval()
            ok(f"Block2b loaded  missing={len(missing)} unexpected={len(unexpected)}")
            return enc, True
        # Fallback raw load
        raw = torch.load(str(path), map_location="cpu", weights_only=False)
        sd  = raw.get("model", raw) if isinstance(raw, dict) else raw
        model_keys = set(enc.state_dict().keys()) if hasattr(enc, "state_dict") else set()
        best, best_n = sd, len(set(sd.keys()) & model_keys) if isinstance(sd, dict) else 0
        enc.load_state_dict(best, strict=False)
        enc = enc.to(device).eval()
        ok("Block2b loaded (fallback)")
        return enc, True
    except Exception as ex:
        warn(f"Block2b load failed: {ex}")
        return None, False


def run_block1(AudioFrontend, audio_path: Path, device: str):
    """
    AudioFrontend is NOT nn.Module — runs on CPU, no .to(device).
    Returns (mel [1, 128, 3000], AudioFrontendOutput).
    """
    af  = AudioFrontend()
    out = af.process(audio_path)
    mel = out.chunks[0].unsqueeze(0).to(device)    # [1, 128, 3000]
    return mel, out


def run_block2a(enc_instance, mel: torch.Tensor, device: str) -> torch.Tensor:
    """
    Runs a pre-built (already instantiated) WhisperEncoder.
    Returns h_w [1, 750, 1024].
    """
    with torch.no_grad():
        out = enc_instance(mel.to(device))
    h_w = out.h_w if hasattr(out, "h_w") else out
    if isinstance(h_w, (list, tuple)):
        h_w = h_w[0]
    if h_w.size(-1) != 1024:
        proj = nn.Linear(h_w.size(-1), 1024).to(device)
        with torch.no_grad():
            h_w = proj(h_w)
    return h_w


# ─────────────────────────────────────────────────────────────────────
# T1 · Inference — single audio
# ─────────────────────────────────────────────────────────────────────

def t1_inference_single(tok, packer):
    hdr("T1 · Inference: single audio [T=750, 896]")
    T      = 750
    prompt = tok.build_chat_prompt("What sound is this?", n_audios=1)
    audio  = make_audio(T, 1)

    packed = packer.pack_for_inference(prompt, audio)

    check("inputs_embeds is 3D",
          packed.inputs_embeds.dim() == 3,
          str(tuple(packed.inputs_embeds.shape)))
    check("last dim=896",  packed.inputs_embeds.size(-1) == HIDDEN)
    check("batch dim=1",   packed.inputs_embeds.size(0) == 1)
    check("seq_len > raw prompt_ids",
          packed.prompt_len > len(prompt.input_ids),
          f"injected={packed.prompt_len}, original={len(prompt.input_ids)}")
    check("total_len = prompt_len",  packed.total_len == packed.prompt_len)
    check("target_len = 0",          packed.target_len == 0)
    check("stage = None",            packed.stage is None)
    check("1 audio span",            len(packed.audio_spans) == 1)
    check("audio span size = T",
          packed.audio_spans[0][1] - packed.audio_spans[0][0] == T,
          str(packed.audio_spans[0]))
    check("attention_mask all 1",
          packed.attention_mask.sum().item() == packed.total_len)
    check("loss_mask all 0",         packed.loss_mask.sum().item() == 0)
    check("no NaN",                  nonan(packed.inputs_embeds))
    info(str(packed))
    return packed


# ─────────────────────────────────────────────────────────────────────
# T2 · Inference — two audios
# ─────────────────────────────────────────────────────────────────────

def t2_inference_two_audios(tok, packer):
    hdr("T2 · Inference: two audios [T=750 each]")
    T      = 750
    prompt = tok.build_chat_prompt("Compare the two sounds.", n_audios=2)
    audio  = make_audio(T, 2)

    packed = packer.pack_for_inference(prompt, audio)

    check("2 audio spans",    len(packed.audio_spans) == 2)
    check("n_audios = 2",     packed.n_audios == 2)
    check("seq_len > original + T",
          packed.prompt_len > len(prompt.input_ids) + T)
    check("no NaN",           nonan(packed.inputs_embeds))
    check("spans in order",
          packed.audio_spans[0][1] < packed.audio_spans[1][0])
    info(f"spans: {packed.audio_spans}")


# ─────────────────────────────────────────────────────────────────────
# T3 · Training Stage 1 — answer tokens only
# ─────────────────────────────────────────────────────────────────────

def t3_training_stage1(tok, packer):
    hdr("T3 · Training Stage 1: answer tokens only")
    T      = 750
    prompt = tok.build_chat_prompt("What emotion?", n_audios=1)
    audio  = make_audio(T, 1)
    answer = "The speaker conveys excitement."

    target = tok.build_training_target(prompt.input_ids, answer, stage=1)
    packed = packer.pack_for_training(prompt, audio, target)

    check("stage = 1",        packed.stage == 1)
    check("target_len > 0",   packed.target_len > 0, str(packed.target_len))
    check("total = prompt + target",
          packed.total_len == packed.prompt_len + packed.target_len)
    check("loss_mask sum = target_len",
          packed.loss_mask.sum().item() == packed.target_len,
          f"{int(packed.loss_mask.sum())} == {packed.target_len}")
    check("loss_mask 0 on prompt",
          packed.loss_mask[0, :packed.prompt_len].sum().item() == 0)
    check("loss_mask 1 on target",
          packed.loss_mask[0, packed.prompt_len:].sum().item() == packed.target_len)
    check("no NaN", nonan(packed.inputs_embeds))
    info(str(packed))


# ─────────────────────────────────────────────────────────────────────
# T4 · Training Stage 4 — CoT + answer
# ─────────────────────────────────────────────────────────────────────

def t4_training_stage4(tok, packer):
    hdr("T4 · Training Stage 4: <think>...</think> + answer")
    T      = 750
    prompt = tok.build_chat_prompt("What emotion?", n_audios=1,
                                   add_cot_trigger=True)
    audio  = make_audio(T, 1)
    answer = "The speaker conveys excitement."
    think  = "Rising F0 and rapid speech → excitement."

    target_s1 = tok.build_training_target(prompt.input_ids, answer, stage=1)
    target_s4 = tok.build_training_target(
        prompt.input_ids, answer, stage=4, think_text=think
    )

    packed_s1 = packer.pack_for_training(prompt, audio, target_s1)
    packed_s4 = packer.pack_for_training(prompt, audio, target_s4)

    actual_delta   = packed_s4.target_len - packed_s1.target_len
    expected_delta = target_s4.num_tokens  - target_s1.num_tokens

    check("stage=4 longer than stage=1 by think tokens",
          actual_delta == expected_delta,
          f"Δ={actual_delta}  expected={expected_delta}")
    check("stage=4 prompt unchanged",
          packed_s4.prompt_len == packed_s1.prompt_len)
    check("loss on full target (think + answer)",
          packed_s4.loss_mask[0, packed_s4.prompt_len:].sum().item() == packed_s4.target_len)
    check("no NaN", nonan(packed_s4.inputs_embeds))
    info(f"s1 target={packed_s1.target_len}  "
         f"s4 target={packed_s4.target_len}  "
         f"Δ={actual_delta}")


# ─────────────────────────────────────────────────────────────────────
# T5 · Audio span content matches injected tensor
# ─────────────────────────────────────────────────────────────────────

def t5_audio_span_content(tok, packer):
    hdr("T5 · Audio span content matches injected tensor")
    T     = 200
    audio = make_audio(T, 1)

    prompt = tok.build_chat_prompt("Test.", n_audios=1)
    packed = packer.pack_for_inference(prompt, audio)

    start, end = packed.audio_spans[0]
    injected   = packed.inputs_embeds[0, start:end]

    check("span length = T",  injected.size(0) == T, str(injected.shape))
    check("injected matches source",
          torch.allclose(injected.float(), audio[0].float(), atol=1e-5))


# ─────────────────────────────────────────────────────────────────────
# T6 · Variable audio lengths (multi-chunk)
# ─────────────────────────────────────────────────────────────────────

def t6_variable_length_audios(tok, packer):
    hdr("T6 · Variable audio lengths (multi-chunk 2250 tokens)")
    T1, T2 = 750, 1500
    prompt  = tok.build_chat_prompt("Compare.", n_audios=2)
    audio   = [torch.randn(T1, HIDDEN), torch.randn(T2, HIDDEN)]

    packed = packer.pack_for_inference(prompt, audio)

    s0, e0 = packed.audio_spans[0]
    s1, e1 = packed.audio_spans[1]
    check("audio1 span size = T1", e0 - s0 == T1, f"{e0-s0}={T1}")
    check("audio2 span size = T2", e1 - s1 == T2, f"{e1-s1}={T2}")
    check("no NaN", nonan(packed.inputs_embeds))


# ─────────────────────────────────────────────────────────────────────
# T7 · pack() dispatch
# ─────────────────────────────────────────────────────────────────────

def t7_dispatch_pack(tok, packer):
    hdr("T7 · pack() dispatches inference vs training")
    prompt = tok.build_chat_prompt("Test.", n_audios=1)
    audio  = make_audio(750, 1)
    target = tok.build_training_target(prompt.input_ids, "Answer.", stage=2)

    inf = packer.pack(prompt, audio)
    tr  = packer.pack(prompt, audio, training_target=target)

    check("no target → inference", inf.stage is None)
    check("with target → training", tr.stage == 2)
    check("training has nonzero loss", tr.loss_mask.sum().item() > 0)


# ─────────────────────────────────────────────────────────────────────
# T8 · Attention mask shape
# ─────────────────────────────────────────────────────────────────────

def t8_attention_mask(tok, packer):
    hdr("T8 · Attention mask shape + all-ones")
    prompt = tok.build_chat_prompt("Test.", n_audios=1)
    audio  = make_audio(750, 1)
    packed = packer.pack_for_inference(prompt, audio)

    check("attention_mask shape [1, total_len]",
          tuple(packed.attention_mask.shape) == (1, packed.total_len),
          str(tuple(packed.attention_mask.shape)))
    check("all ones (no padding)",
          packed.attention_mask.sum().item() == packed.total_len)


# ─────────────────────────────────────────────────────────────────────
# T9 · Empty audio list raises
# ─────────────────────────────────────────────────────────────────────

def t9_empty_audio_raises(tok, packer):
    hdr("T9 · Empty audio list raises ValueError")
    prompt = tok.build_chat_prompt("Test.", n_audios=1)

    try:
        packer.pack_for_inference(prompt, [])
        check("raises on empty audio", False)
    except ValueError:
        check("raises on empty audio", True)


# ─────────────────────────────────────────────────────────────────────
# T10 · 3D audio tensor handled correctly
# ─────────────────────────────────────────────────────────────────────

def t10_batched_audio_tensor(tok, packer):
    hdr("T10 · [1, T, 896] audio (batched Block4 output) handled")
    T     = 750
    audio = [torch.randn(1, T, HIDDEN)]
    prompt = tok.build_chat_prompt("Test.", n_audios=1)

    packed = packer.pack_for_inference(prompt, audio)

    check("span size = T after squeeze",
          packed.audio_spans[0][1] - packed.audio_spans[0][0] == T)
    check("no NaN", nonan(packed.inputs_embeds))


# ─────────────────────────────────────────────────────────────────────
# Integration tests — Block4 → Block5 → Block6 (real embed_tokens)
# ─────────────────────────────────────────────────────────────────────

def ti1_full_chain_single(tok, embed):
    hdr("TI1 · Block4 + Block5 + Block6: single 30s audio")
    from blocks.block4_mlp_adaptor import Block4MLPAdaptor
    from blocks.block6_sequence_packer import Block6SequencePacker

    b4 = Block4MLPAdaptor()
    with torch.no_grad():
        out4 = b4(torch.randn(1, 750, 1024), num_chunks=1)
    audio_tokens = [out4.audio_tokens.squeeze(0)]

    prompt = tok.build_chat_prompt(
        "What emotion does the speaker convey?", n_audios=1
    )
    target = tok.build_training_target(
        prompt.input_ids,
        "The speaker conveys excitement.",
        stage=1,
    )

    packer = Block6SequencePacker(tok, embed)
    packed = packer.pack_for_training(prompt, audio_tokens, target)

    check("inputs_embeds [1, L, 896]",    packed.inputs_embeds.size(-1) == HIDDEN)
    check("audio injected correctly",
          packed.audio_spans[0][1] - packed.audio_spans[0][0] == 750)
    check("loss on answer only",
          packed.loss_mask[0, :packed.prompt_len].sum().item() == 0)
    check("no NaN",                       nonan(packed.inputs_embeds))
    info(str(packed))


def ti2_full_chain_multi_audio(tok, embed):
    hdr("TI2 · Block4 + Block5 + Block6: 2 audios")
    from blocks.block4_mlp_adaptor import Block4MLPAdaptor
    from blocks.block6_sequence_packer import Block6SequencePacker

    b4 = Block4MLPAdaptor()
    audio_tokens = []
    for _ in range(2):
        with torch.no_grad():
            out4 = b4(torch.randn(1, 750, 1024))
        audio_tokens.append(out4.audio_tokens.squeeze(0))

    prompt = tok.build_chat_prompt("Compare the two recordings.", n_audios=2)
    target = tok.build_training_target(
        prompt.input_ids,
        "Audio 1 is speech. Audio 2 is music.",
        stage=2,
    )

    packer = Block6SequencePacker(tok, embed)
    packed = packer.pack_for_training(prompt, audio_tokens, target)

    check("2 audio spans",  len(packed.audio_spans) == 2)
    check("each span = 750",
          all((e - s) == 750 for s, e in packed.audio_spans))
    check("loss on answer", packed.loss_mask.sum().item() == packed.target_len)
    check("no NaN",         nonan(packed.inputs_embeds))
    info(str(packed))


def ti3_stage4_cot_chain(tok, embed):
    hdr("TI3 · Stage 4 CoT: <think> + answer loss")
    from blocks.block4_mlp_adaptor import Block4MLPAdaptor
    from blocks.block6_sequence_packer import Block6SequencePacker

    b4 = Block4MLPAdaptor()
    with torch.no_grad():
        out4 = b4(torch.randn(1, 750, 1024))
    audio_tokens = [out4.audio_tokens.squeeze(0)]

    prompt = tok.build_chat_prompt(
        "What emotion? Think step by step before answering.",
        n_audios=1, add_cot_trigger=True
    )
    think  = "Rising F0 and rapid speech rate suggest excitement."
    answer = "The speaker conveys excitement."

    target_s1 = tok.build_training_target(prompt.input_ids, answer, stage=1)
    target_s4 = tok.build_training_target(
        prompt.input_ids, answer, stage=4, think_text=think
    )

    packer    = Block6SequencePacker(tok, embed)
    packed_s1 = packer.pack_for_training(prompt, audio_tokens, target_s1)
    packed_s4 = packer.pack_for_training(prompt, audio_tokens, target_s4)

    expected_delta = target_s4.num_tokens - target_s1.num_tokens
    actual_delta   = packed_s4.target_len - packed_s1.target_len

    check("same prompt_len both stages",
          packed_s1.prompt_len == packed_s4.prompt_len)
    check("stage=4 more loss tokens (Δ correct)",
          actual_delta == expected_delta,
          f"Δ={actual_delta} expected={expected_delta}")
    check("no NaN (stage 4)", nonan(packed_s4.inputs_embeds))
    info(f"s1 target={packed_s1.target_len}  "
         f"s4 target={packed_s4.target_len}  "
         f"Δ={actual_delta}")


# ═════════════════════════════════════════════════════════════════════
# FULL PIPELINE TESTS — Block1 → Block2a → Block2b → Block3 → Block4
#                       → Block5 → Block6   (NO mocking)
# ═════════════════════════════════════════════════════════════════════

def _build_pipeline_blocks(ckpt: str, device: str, tok, embed):
    """
    Instantiate all blocks.  WhisperEncoder is built ONCE here so HF
    weights are not re-loaded inside any loop.

    Returns dict with all live block instances.
    """
    from blocks.block1_audio_frontend  import AudioFrontend
    from blocks.block3_perceiver_fusion import Block3PerceiverFusion
    from blocks.block4_mlp_adaptor      import Block4MLPAdaptor
    from blocks.block6_sequence_packer  import Block6SequencePacker

    Block2a = get_block2a_class()

    # Build WhisperEncoder ONCE — avoid HF re-load in loops
    enc2a = Block2a()
    if isinstance(enc2a, nn.Module):
        enc2a = enc2a.to(device).eval()
    ok("Block2a (WhisperEncoder) ready")

    b2b, b2b_ok = load_block2b(ckpt, device)

    b3 = Block3PerceiverFusion().to(device).eval()
    b4 = Block4MLPAdaptor().to(device).eval()

    packer = Block6SequencePacker(tok, embed)

    return dict(
        AudioFrontend=AudioFrontend,
        enc2a=enc2a,
        b2b=b2b, b2b_ok=b2b_ok,
        b3=b3, b4=b4,
        packer=packer,
    )


def _process_one_audio(wav_path: Path, blocks: dict, device: str):
    """
    Run one WAV file through Block 1 → 2a → 2b → 3 → 4.
    Returns (audio_tokens [T, 896], block outputs for assertions).
    """
    AudioFrontend = blocks["AudioFrontend"]
    enc2a         = blocks["enc2a"]
    b2b           = blocks["b2b"]
    b3            = blocks["b3"]
    b4            = blocks["b4"]

    # ── Block 1 ─────────────────────────────────────────────────────
    mel, b1_out = run_block1(AudioFrontend, wav_path, device)
    # mel: [1, 128, 3000]

    # ── Block 2a ────────────────────────────────────────────────────
    h_w = run_block2a(enc2a, mel, device)
    # h_w: [1, 750, 1024]

    # ── Block 2b ────────────────────────────────────────────────────
    with torch.no_grad():
        out2b = b2b(mel)
    h_ob = out2b.h_ob
    # h_ob: [1, 1496, 1024]

    # ── Block 3 ─────────────────────────────────────────────────────
    with torch.no_grad():
        out3 = b3(h_ob=h_ob, h_w=h_w)
    # out3.h_full: [1, 750, 1024]

    # ── Block 4 ─────────────────────────────────────────────────────
    with torch.no_grad():
        out4 = b4(out3.h_full, num_chunks=out3.n_chunks)
    # out4.audio_tokens: [1, 750, 896]

    audio_tokens = out4.audio_tokens.squeeze(0)   # [750, 896]

    return audio_tokens, dict(
        mel=mel, h_w=h_w, h_ob=h_ob,
        h_full=out3.h_full, n_chunks=out3.n_chunks,
        b1_num_chunks=b1_out.num_chunks,
    )


def _process_multi_chunk_audio(wav_path: Path, blocks: dict, device: str):
    """
    Process a long audio file (>30s) through Block1→2a→2b, then use
    Block3.forward_multi_chunk to merge all chunks, then Block4.
    Returns (audio_tokens [T_total, 896], metadata).
    """
    AudioFrontend = blocks["AudioFrontend"]
    enc2a         = blocks["enc2a"]
    b2b           = blocks["b2b"]
    b3            = blocks["b3"]
    b4            = blocks["b4"]

    # Block 1 — returns multiple chunks
    af     = AudioFrontend()
    b1_out = af.process(wav_path)
    n_chunks = b1_out.num_chunks
    info(f"Block1: {n_chunks} chunks from {wav_path.name}")

    chunk_inputs = []
    for i in range(n_chunks):
        mel = b1_out.chunks[i].unsqueeze(0).to(device)   # [1, 128, 3000]

        h_w = run_block2a(enc2a, mel, device)

        with torch.no_grad():
            out2b = b2b(mel)
        h_ob = out2b.h_ob

        chunk_inputs.append((h_ob, h_w))
        info(f"  chunk {i+1}/{n_chunks}: h_ob={tuple(h_ob.shape)} h_w={tuple(h_w.shape)}")

    # Block 3 multi-chunk
    with torch.no_grad():
        out3 = b3.forward_multi_chunk(chunk_inputs)
    # out3.h_full: [1, 750*n_chunks, 1024]

    # Block 4
    with torch.no_grad():
        out4 = b4(out3.h_full, num_chunks=out3.n_chunks)
    audio_tokens = out4.audio_tokens.squeeze(0)   # [750*n_chunks, 896]

    return audio_tokens, dict(
        n_chunks=n_chunks,
        seq_len=out3.seq_len,
        h_full=out3.h_full,
    )


# ─────────────────────────────────────────────────────────────────────
# TP1 · Full pipeline — single audio, inference
# ─────────────────────────────────────────────────────────────────────

def tp1_full_pipeline_inference(blocks, tok, device, tmp_dir):
    hdr("TP1 · Block1→2a→2b→3→4→5→6: single 30s audio — inference")

    wav = tmp_dir / "tp1_30s.wav"
    make_fake_audio_file(wav, seconds=30.0)

    audio_tokens, meta = _process_one_audio(wav, blocks, device)

    # Block 5
    prompt = tok.build_chat_prompt(
        "What sound is this?", n_audios=1
    )

    # Block 6
    packed = blocks["packer"].pack_for_inference(prompt, [audio_tokens])

    # ── Shape checks ─────────────────────────────────────────────────
    check("B1: 1 chunk produced", meta["b1_num_chunks"] == 1, "1")
    check("B2b: h_ob [1, 1496, 1024]",
          meta["h_ob"].shape == (1, 1496, 1024),
          str(tuple(meta["h_ob"].shape)))
    check("B2a: h_w [1, 750, 1024]",
          meta["h_w"].shape == (1, 750, 1024),
          str(tuple(meta["h_w"].shape)))
    check("B3: h_full [1, 750, 1024]",
          meta["h_full"].shape == (1, 750, 1024),
          str(tuple(meta["h_full"].shape)))
    check("B4: audio_tokens [750, 896]",
          audio_tokens.shape == (750, 896),
          str(tuple(audio_tokens.shape)))
    check("B6: inputs_embeds [1, L, 896]",
          packed.inputs_embeds.size(-1) == HIDDEN)
    check("B6: audio span = 750",
          packed.audio_spans[0][1] - packed.audio_spans[0][0] == 750)
    check("B6: no NaN", nonan(packed.inputs_embeds))
    check("B6: loss_mask all 0 (inference)", packed.loss_mask.sum().item() == 0)

    info(str(packed))


# ─────────────────────────────────────────────────────────────────────
# TP2 · Full pipeline — single audio, training Stage 1
# ─────────────────────────────────────────────────────────────────────

def tp2_full_pipeline_training_s1(blocks, tok, device, tmp_dir):
    hdr("TP2 · Block1→2a→2b→3→4→5→6: training Stage 1")

    wav = tmp_dir / "tp2_30s.wav"
    make_fake_audio_file(wav, seconds=30.0)

    audio_tokens, meta = _process_one_audio(wav, blocks, device)

    prompt = tok.build_chat_prompt(
        "What emotion does the speaker convey?", n_audios=1
    )
    answer = "The speaker conveys a neutral tone."
    target = tok.build_training_target(prompt.input_ids, answer, stage=1)

    packed = blocks["packer"].pack_for_training(prompt, [audio_tokens], target)

    check("stage = 1",        packed.stage == 1)
    check("target_len > 0",   packed.target_len > 0, str(packed.target_len))
    check("prompt has zero loss",
          packed.loss_mask[0, :packed.prompt_len].sum().item() == 0)
    check("target has full loss",
          packed.loss_mask[0, packed.prompt_len:].sum().item() == packed.target_len)
    check("audio span in prompt region",
          packed.audio_spans[0][0] < packed.prompt_len)
    check("no NaN", nonan(packed.inputs_embeds))

    info(f"prompt_len={packed.prompt_len}  target_len={packed.target_len}")
    info(f"answer token count={tok.count_answer_tokens(answer)}")


# ─────────────────────────────────────────────────────────────────────
# TP3 · Full pipeline — two audio files, training Stage 2
# ─────────────────────────────────────────────────────────────────────

def tp3_full_pipeline_two_audios(blocks, tok, device, tmp_dir):
    hdr("TP3 · Block1→2a→2b→3→4→5→6: two audio files — Stage 2")

    wav1 = tmp_dir / "tp3_audio1.wav"
    wav2 = tmp_dir / "tp3_audio2.wav"
    make_fake_audio_file(wav1, seconds=30.0)
    make_fake_audio_file(wav2, seconds=30.0)   # different content, same length

    tokens1, _ = _process_one_audio(wav1, blocks, device)
    tokens2, _ = _process_one_audio(wav2, blocks, device)

    prompt = tok.build_chat_prompt(
        "Compare the two audio recordings.", n_audios=2
    )
    answer = "Audio 1 is speech. Audio 2 is ambient noise."
    target = tok.build_training_target(prompt.input_ids, answer, stage=2)

    packed = blocks["packer"].pack_for_training(
        prompt, [tokens1, tokens2], target
    )

    check("2 audio spans",   len(packed.audio_spans) == 2)
    check("each span = 750",
          all((e - s) == 750 for s, e in packed.audio_spans))
    check("spans in order",  packed.audio_spans[0][1] < packed.audio_spans[1][0])
    check("stage = 2",       packed.stage == 2)
    check("loss only on answer",
          packed.loss_mask.sum().item() == packed.target_len)
    check("no NaN",          nonan(packed.inputs_embeds))

    info(f"total_len={packed.total_len}  "
         f"spans={packed.audio_spans}")


# ─────────────────────────────────────────────────────────────────────
# TP4 · Full pipeline — 90s audio (3 chunks), training Stage 3
# ─────────────────────────────────────────────────────────────────────

def tp4_full_pipeline_multi_chunk(blocks, tok, device, tmp_dir):
    hdr("TP4 · Block1→2a→2b→3→4→5→6: 90s audio (3 chunks) — Stage 3")

    wav = tmp_dir / "tp4_90s.wav"
    make_fake_audio_file(wav, seconds=90.0)

    audio_tokens, meta = _process_multi_chunk_audio(wav, blocks, device)

    N_CHUNKS  = meta["n_chunks"]
    SEQ_TOTAL = 750 * N_CHUNKS

    check("Block1 produced 3 chunks", N_CHUNKS == 3, str(N_CHUNKS))
    check("Block3 seq_len = 2250", meta["seq_len"] == SEQ_TOTAL,
          str(meta["seq_len"]))
    check("audio_tokens shape [2250, 896]",
          audio_tokens.shape == (SEQ_TOTAL, HIDDEN),
          str(tuple(audio_tokens.shape)))

    # Block 5 + 6: treat as a single (long) audio file
    prompt = tok.build_chat_prompt(
        "Describe this 90-second audio clip.", n_audios=1
    )
    answer = "The clip contains a sustained 440 Hz tone throughout."
    target = tok.build_training_target(prompt.input_ids, answer, stage=3)

    packed = blocks["packer"].pack_for_training(prompt, [audio_tokens], target)

    check("stage = 3",       packed.stage == 3)
    check("audio span = 2250",
          packed.audio_spans[0][1] - packed.audio_spans[0][0] == SEQ_TOTAL,
          str(packed.audio_spans[0]))
    check("loss on answer only",
          packed.loss_mask[0, :packed.prompt_len].sum().item() == 0)
    check("no NaN",          nonan(packed.inputs_embeds))

    info(f"total_len={packed.total_len}  "
         f"prompt={packed.prompt_len}  target={packed.target_len}")


# ─────────────────────────────────────────────────────────────────────
# TP5 · Full pipeline — Stage 4 CoT with real audio
# ─────────────────────────────────────────────────────────────────────

def tp5_full_pipeline_stage4_cot(blocks, tok, device, tmp_dir):
    hdr("TP5 · Block1→2a→2b→3→4→5→6: Stage 4 CoT — real audio")

    wav = tmp_dir / "tp5_30s.wav"
    make_fake_audio_file(wav, seconds=30.0)

    audio_tokens, _ = _process_one_audio(wav, blocks, device)

    prompt_s4 = tok.build_chat_prompt(
        "What emotion does the speaker convey? Think step by step before answering.",
        n_audios=1, add_cot_trigger=True
    )
    think  = "The audio has a steady frequency with no pitch variation, suggesting neutrality."
    answer = "The speaker conveys a neutral, calm tone."

    target_s1 = tok.build_training_target(prompt_s4.input_ids, answer, stage=1)
    target_s4 = tok.build_training_target(
        prompt_s4.input_ids, answer, stage=4, think_text=think
    )

    packed_s1 = blocks["packer"].pack_for_training(
        prompt_s4, [audio_tokens], target_s1
    )
    packed_s4 = blocks["packer"].pack_for_training(
        prompt_s4, [audio_tokens], target_s4
    )

    expected_delta = target_s4.num_tokens - target_s1.num_tokens
    actual_delta   = packed_s4.target_len  - packed_s1.target_len

    check("Stage 1 and 4 same prompt_len",
          packed_s1.prompt_len == packed_s4.prompt_len)
    check("Stage 4 target_len > Stage 1",
          packed_s4.target_len > packed_s1.target_len,
          f"{packed_s4.target_len} > {packed_s1.target_len}")
    check("Δ matches think token count",
          actual_delta == expected_delta,
          f"Δ={actual_delta} expected={expected_delta}")
    check("Stage 4 no NaN", nonan(packed_s4.inputs_embeds))
    check("Stage 4 full loss on think+answer",
          packed_s4.loss_mask[0, packed_s4.prompt_len:].sum().item() == packed_s4.target_len)

    info(f"s1_target={packed_s1.target_len}  "
         f"s4_target={packed_s4.target_len}  "
         f"think_Δ={actual_delta}")


# ─────────────────────────────────────────────────────────────────────
# TP6 · Sanity — audio token values come from real audio (not random)
# ─────────────────────────────────────────────────────────────────────

def tp6_different_audios_produce_different_tokens(blocks, tok, device, tmp_dir):
    hdr("TP6 · Different audio files → different audio_tokens")

    wav_a = tmp_dir / "tp6_a.wav"
    wav_b = tmp_dir / "tp6_b.wav"
    make_fake_audio_file(wav_a, seconds=30.0)

    # Different frequency → different audio content
    import torchaudio
    sr  = 16000
    n   = sr * 30
    t   = torch.linspace(0, 30.0, n)
    wav = (0.4 * torch.sin(2 * math.pi * 880 * t)).unsqueeze(0)   # 880 Hz
    torchaudio.save(str(wav_b), wav, sr)

    tokens_a, _ = _process_one_audio(wav_a, blocks, device)
    tokens_b, _ = _process_one_audio(wav_b, blocks, device)

    check("shapes equal",      tokens_a.shape == tokens_b.shape)
    check("content different", not torch.allclose(tokens_a, tokens_b, atol=1e-3))
    check("no NaN a",          nonan(tokens_a))
    check("no NaN b",          nonan(tokens_b))

    diff = (tokens_a - tokens_b).abs().mean().item()
    info(f"mean |Δ| between 440Hz and 880Hz audio: {diff:.4f}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="NanoFlame v4 — Block6 tests")
    p.add_argument("--skip-integration", action="store_true")
    p.add_argument("--skip-pipeline",    action="store_true",
                   help="Skip full Block1→6 pipeline tests")
    p.add_argument("--no-real-embed",    action="store_true",
                   help="Use mock embed_tokens (faster unit tests)")
    p.add_argument("--model",
                   default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="HF model ID for real embed_tokens")
    p.add_argument("--ckpt",
                   default=str(ROOT / "checkpoints" / "epoch_latest.pt"),
                   help="Path to OpenBEATs checkpoint")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = get_args()

    print(f"\n{BD}{'═'*62}{E}")
    print(f"{BD}  NanoFlame v4 — Block 6 Tests{E}")
    print(f"{BD}{'═'*62}{E}")
    print(f"  device  : {args.device}")
    print(f"  ckpt    : {args.ckpt}")
    print(f"  model   : {args.model}")

    tok = make_tok()

    # ── Load embed_tokens ─────────────────────────────────────────────
    if args.no_real_embed:
        warn("Using mock embed_tokens (--no-real-embed)")
        embed = make_mock_embed(tok)
    else:
        embed = load_real_embed_tokens(tok, model_id=args.model)

    packer = make_packer(tok, embed)

    # ── Unit tests ────────────────────────────────────────────────────
    print(f"\n{BD}{'═'*62}{E}")
    print(f"{BD}  UNIT TESTS — Block6{E}")
    print(f"{BD}{'═'*62}{E}")

    t1_inference_single(tok, packer)
    t2_inference_two_audios(tok, packer)
    t3_training_stage1(tok, packer)
    t4_training_stage4(tok, packer)
    t5_audio_span_content(tok, packer)
    t6_variable_length_audios(tok, packer)
    t7_dispatch_pack(tok, packer)
    t8_attention_mask(tok, packer)
    t9_empty_audio_raises(tok, packer)
    t10_batched_audio_tensor(tok, packer)

    # ── Integration tests (Block4+5+6) ────────────────────────────────
    if not args.skip_integration:
        print(f"\n{BD}{'═'*62}{E}")
        print(f"{BD}  INTEGRATION TESTS — Block4 → Block5 → Block6{E}")
        print(f"{BD}{'═'*62}{E}")
        ti1_full_chain_single(tok, embed)
        ti2_full_chain_multi_audio(tok, embed)
        ti3_stage4_cot_chain(tok, embed)
    else:
        warn("Integration tests skipped (--skip-integration)")

    # ── Full pipeline tests (Block1→2a→2b→3→4→5→6) ───────────────────
    if not args.skip_pipeline:
        print(f"\n{BD}{'═'*62}{E}")
        print(f"{BD}  FULL PIPELINE TESTS — Block1 → Block2a → Block2b{E}")
        print(f"{BD}              → Block3 → Block4 → Block5 → Block6{E}")
        print(f"{BD}{'═'*62}{E}")

        # Check Block2b checkpoint
        _, b2b_ok = load_block2b(args.ckpt, args.device)
        if not b2b_ok:
            warn("Block2b checkpoint unavailable — skipping pipeline tests")
            warn(f"Place epoch_latest.pt in {Path(args.ckpt).parent}")
        else:
            # Build all blocks ONCE (WhisperEncoder loaded once here)
            blocks = _build_pipeline_blocks(args.ckpt, args.device, tok, embed)

            tmp_dir = ROOT / "tests" / "_tmp_pipeline_audio"
            tmp_dir.mkdir(parents=True, exist_ok=True)

            tp1_full_pipeline_inference(blocks, tok, args.device, tmp_dir)
            tp2_full_pipeline_training_s1(blocks, tok, args.device, tmp_dir)
            tp3_full_pipeline_two_audios(blocks, tok, args.device, tmp_dir)
            tp4_full_pipeline_multi_chunk(blocks, tok, args.device, tmp_dir)
            tp5_full_pipeline_stage4_cot(blocks, tok, args.device, tmp_dir)
            tp6_different_audios_produce_different_tokens(blocks, tok, args.device, tmp_dir)
    else:
        warn("Full pipeline tests skipped (--skip-pipeline)")

    # ── Verdict ───────────────────────────────────────────────────────
    total = PASS + FAIL
    print(f"\n{BD}{'─'*62}{E}")
    print(f"  {G}{PASS} passed{E}  "
          f"{(R+str(FAIL)+E) if FAIL else str(FAIL)} failed  /  {total} run")

    if FAIL == 0:
        print(f"\n{G}{BD}✔  All Block6 tests passed.{E}\n")
    else:
        print(f"\n{R}{BD}✘  {FAIL} test(s) failed.{E}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()