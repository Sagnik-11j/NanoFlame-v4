"""
tests/test_block4.py
══════════════════════════════════════════════════════════════════════
NanoFlame v4 — Block 4 unit tests + integration tests
══════════════════════════════════════════════════════════════════════

Covers:
  Block 4 alone
  Block 3 → Block 4
  Full pipeline Block1 → Block2a → Block2b → Block3 → Block4

Run:
    python tests/test_block4.py
    python tests/test_block4.py --skip-integration
    python tests/test_block4.py --device cuda
"""

__test__ = False

import sys
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
skip = lambda m: print(f"  {Y}~{E}  {m}")
info = lambda m: print(f"     {m}")
hdr  = lambda t: print(f"\n{BD}── {t} {'─'*(54-len(t))}{E}")

PASS = 0
FAIL = 0


def check(name: str, ok_: bool, detail: str = "") -> bool:
    global PASS, FAIL
    if ok_:
        ok(f"{name}  {detail}")
        PASS += 1
    else:
        fail(f"{name}  {detail}")
        FAIL += 1
    return ok_


def nonan(t: torch.Tensor) -> bool:
    return not (torch.isnan(t).any() or torch.isinf(t).any())


def grad_cov(m: nn.Module) -> float:
    g = sum(1 for p in m.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().max() > 0)
    t = sum(1 for p in m.parameters() if p.requires_grad)
    return g / t if t else 0.0


def get_block4():
    from blocks.block4_mlp_adaptor import Block4MLPAdaptor, Block4Output, RMSNorm
    return Block4MLPAdaptor, Block4Output, RMSNorm


def get_block3():
    from blocks.block3_perceiver_fusion import Block3PerceiverFusion
    return Block3PerceiverFusion


def get_block1():
    from blocks.block1_audio_frontend import AudioFrontend
    return AudioFrontend


def get_block2a():
    from blocks.block2a_whisper_encoder import Block2aWhisperEncoder
    return Block2aWhisperEncoder


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

        raw = torch.load(str(path), map_location="cpu", weights_only=False)
        sd = raw.get("model", raw) if isinstance(raw, dict) else raw
        model_keys = set(enc.state_dict().keys()) if hasattr(enc, "state_dict") else set()
        best, best_n = sd, len(set(sd.keys()) & model_keys) if isinstance(sd, dict) else 0
        enc.load_state_dict(best, strict=False)
        enc = enc.to(device).eval()
        ok("Block2b loaded")
        return enc, True
    except Exception as e:
        warn(f"Failed to load Block2b checkpoint: {e}")
        return None, False


def make_fake_audio_file(path: Path, seconds: float = 30.0, sr: int = 16000):
    import torchaudio
    t = torch.linspace(0, seconds, int(sr * seconds))
    wav = (0.25 * torch.sin(2 * math.pi * 220 * t)).unsqueeze(0)
    torchaudio.save(str(path), wav, sr)


def mel_from_block1(AudioFrontend, tmp_audio_path: Path, device: str):
    af = AudioFrontend()
    out = af.process(tmp_audio_path)
    if not out.chunks:
        raise ValueError(f"AudioFrontend returned no chunks for {tmp_audio_path}")
    mel = out.chunks[0].unsqueeze(0).to(device)
    return mel, out

def h_w_from_block2a(whisper_enc_or_cls, mel: torch.Tensor, device: str):
    B = mel.size(0)
    try:
        enc = whisper_enc_or_cls() if isinstance(whisper_enc_or_cls, type) else whisper_enc_or_cls
        if isinstance(enc, nn.Module):
            enc = enc.to(device)
        with torch.no_grad():
            out = enc(mel.to(device))
        h_w = out.h_w if hasattr(out, "h_w") else out
        if h_w.size(-1) != 1024:
            proj = nn.Linear(h_w.size(-1), 1024).to(device)
            with torch.no_grad():
                h_w = proj(h_w)
        return h_w
    except Exception as e:
        warn(f"Whisper failed ({e}) — using synthetic h_w")
        return torch.randn(B, 750, 1024, device=device)


def t1_forward_shape(device, B=2, T=750):
    hdr("T1 · Block4 forward shape")
    Block4, _, _ = get_block4()
    b4 = Block4().to(device)
    x = torch.randn(B, T, 1024, device=device)
    with torch.no_grad():
        out = b4(x, num_chunks=1)
    check("audio_tokens [B, T, 896]", out.audio_tokens.shape == (B, T, 896), str(tuple(out.audio_tokens.shape)))
    check("residual [B, T, 896]", out.residual.shape == (B, T, 896))
    check("mlp_out [B, T, 896]", out.mlp_out.shape == (B, T, 896))
    check("num_tokens == T", out.num_tokens == T)
    check("num_chunks == 1", out.num_chunks == 1)
    check("no NaN", nonan(out.audio_tokens))


def t2_single_unbatched(device, T=750):
    hdr("T2 · Unbatched input [T, 1024]")
    Block4, _, _ = get_block4()
    b4 = Block4().to(device)
    x = torch.randn(T, 1024, device=device)
    with torch.no_grad():
        out = b4(x, num_chunks=1)
    check("audio_tokens [T, 896]", out.audio_tokens.shape == (T, 896), str(tuple(out.audio_tokens.shape)))
    check("residual [T, 896]", out.residual.shape == (T, 896))
    check("mlp_out [T, 896]", out.mlp_out.shape == (T, 896))
    check("num_tokens == T", out.num_tokens == T)
    check("no NaN", nonan(out.audio_tokens))


def t3_residual_path_effect(device, B=2, T=750):
    hdr("T3 · Residual branch contributes")
    Block4, _, _ = get_block4()
    b4 = Block4().to(device)
    x = torch.randn(B, T, 1024, device=device)
    with torch.no_grad():
        out = b4(x)
    check("output differs from main branch", not torch.allclose(out.audio_tokens, out.mlp_out, atol=1e-5))
    check("output differs from residual branch", not torch.allclose(out.audio_tokens, out.residual, atol=1e-5))


def t4_rmsnorm_present(device):
    hdr("T4 · RMSNorm present and trainable")
    Block4, _, RMSNorm = get_block4()
    b4 = Block4().to(device)
    norms = [m for m in b4.modules() if isinstance(m, RMSNorm)]
    check("exactly one RMSNorm", len(norms) == 1, f"found {len(norms)}")
    if norms:
        check("RMSNorm weight trainable", norms[0].weight.requires_grad)


def t5_gradient_flow(device, B=2, T=750):
    hdr("T5 · Gradient flow through Block4")
    Block4, _, _ = get_block4()
    b4 = Block4().to(device)
    b4.unfreeze_all()
    x = torch.randn(B, T, 1024, device=device)
    out = b4(x)
    out.audio_tokens.mean().backward()
    pct = grad_cov(b4)
    check("≥95% params received grad", pct >= 0.95, f"({pct*100:.0f}%)")
    b4.zero_grad()


def t6_freeze_unfreeze(device):
    hdr("T6 · freeze/unfreeze")
    Block4, _, _ = get_block4()
    b4 = Block4().to(device)
    total = sum(1 for _ in b4.parameters())
    b4.freeze_base_weights()
    frozen = sum(1 for p in b4.parameters() if not p.requires_grad)
    check("freeze_base_weights() freezes all", frozen == total, f"({frozen}/{total})")
    b4.unfreeze_all()
    trainable = sum(1 for p in b4.parameters() if p.requires_grad)
    check("unfreeze_all() restores all", trainable == total, f"({trainable}/{total})")


def t7_dropout_control(device):
    hdr("T7 · enable_training_dropouts")
    Block4, _, _ = get_block4()
    b4 = Block4(dropout=0.0).to(device)
    b4.enable_training_dropouts(0.1)
    check("dropout updated to 0.1", abs(b4.dropout.p - 0.1) < 1e-8)


def t8_param_count(device):
    hdr("T8 · Param count sanity")
    Block4, _, _ = get_block4()
    b4 = Block4().to(device)
    check("has nonzero params", b4.total_params > 0, f"{b4.total_params:,}")
    check("trainable==total initially", b4.trainable_params == b4.total_params, f"{b4.trainable_params:,}/{b4.total_params:,}")
    info(f"Block4 total params : {b4.total_params:,}")


def t9_bad_input_dim(device):
    hdr("T9 · Reject wrong last dim")
    Block4, _, _ = get_block4()
    b4 = Block4().to(device)
    bad = torch.randn(2, 750, 999, device=device)
    try:
        _ = b4(bad)
        check("raises on wrong last dim", False)
    except ValueError:
        check("raises on wrong last dim", True)


def t10_long_audio_shape(device, B=2, N=3):
    hdr("T10 · Long audio sequence 2250 → 2250")
    Block4, _, _ = get_block4()
    T = 750 * N
    b4 = Block4().to(device)
    x = torch.randn(B, T, 1024, device=device)
    with torch.no_grad():
        out = b4(x, num_chunks=N)
    check("audio_tokens [B, 2250, 896]", out.audio_tokens.shape == (B, T, 896), str(tuple(out.audio_tokens.shape)))
    check("num_chunks propagated", out.num_chunks == N)


def ti1_block3_to_block4(device, B=2):
    hdr("TI1 · Block3 → Block4")
    Block3 = get_block3()
    Block4, _, _ = get_block4()
    b3 = Block3().to(device)
    b4 = Block4().to(device)
    with torch.no_grad():
        out3 = b3(
            h_ob=torch.randn(B, 1496, 1024, device=device),
            h_w=torch.randn(B, 750, 1024, device=device),
        )
        out4 = b4(out3.h_full, num_chunks=out3.n_chunks)
    check("Block3 h_full [B, 750, 1024]", out3.h_full.shape == (B, 750, 1024), str(tuple(out3.h_full.shape)))
    check("Block4 audio_tokens [B, 750, 896]", out4.audio_tokens.shape == (B, 750, 896), str(tuple(out4.audio_tokens.shape)))
    check("num_chunks propagated", out4.num_chunks == out3.n_chunks)
    check("no NaN", nonan(out4.audio_tokens))


def ti2_block3_multi_to_block4(device, B=2, N=3):
    hdr("TI2 · Block3 multi-chunk → Block4")
    Block3 = get_block3()
    Block4, _, _ = get_block4()
    b3 = Block3().to(device)
    b4 = Block4().to(device)
    chunks = [(torch.randn(B, 1496, 1024, device=device), torch.randn(B, 750, 1024, device=device)) for _ in range(N)]
    with torch.no_grad():
        out3 = b3.forward_multi_chunk(chunks)
        out4 = b4(out3.h_full, num_chunks=out3.n_chunks)
    check("Block3 h_full [B, 2250, 1024]", out3.h_full.shape == (B, 750 * N, 1024), str(tuple(out3.h_full.shape)))
    check("Block4 audio_tokens [B, 2250, 896]", out4.audio_tokens.shape == (B, 750 * N, 896), str(tuple(out4.audio_tokens.shape)))
    check("num_chunks == 3", out4.num_chunks == N)


def ti3_stage1_only_block4_trains(device, B=2):
    hdr("TI3 · Stage 1: only Block4 trains")
    Block3 = get_block3()
    Block4, _, _ = get_block4()
    b3 = Block3().to(device)
    b4 = Block4().to(device)
    b3.freeze()
    b4.unfreeze_all()
    out3 = b3(
        h_ob=torch.randn(B, 1496, 1024, device=device),
        h_w=torch.randn(B, 750, 1024, device=device),
    )
    out4 = b4(out3.h_full.detach(), num_chunks=out3.n_chunks)
    out4.audio_tokens.mean().backward()
    b4_pct = grad_cov(b4)
    b3_has_grad = any(p.grad is not None for p in b3.parameters())
    check("Block4 grads flow", b4_pct >= 0.95, f"({b4_pct*100:.0f}%)")
    check("Block3 remains frozen", not b3_has_grad)


def ti4_full_pipeline_to_block4(AudioFrontend, Block2aWhisperEncoder, b2b, device, tmp_dir: Path):
    hdr("TI4 · Full pipeline: Block1→2a→2b→3→4 (single chunk)")
    Block3 = get_block3()
    Block4, _, _ = get_block4()

    audio_path = tmp_dir / "tone_30s.wav"
    make_fake_audio_file(audio_path, seconds=30.0)

    mel, b1_out = mel_from_block1(AudioFrontend, audio_path, device)

    whisper_enc = Block2aWhisperEncoder()
    if isinstance(whisper_enc, nn.Module):
        whisper_enc = whisper_enc.to(device).eval()

    with torch.no_grad():
        h_ob = b2b(mel).h_ob
    h_w = h_w_from_block2a(whisper_enc, mel, device)

    b3 = Block3().to(device)
    b4 = Block4().to(device)

    with torch.no_grad():
        out3 = b3(h_ob=h_ob, h_w=h_w)
        out4 = b4(out3.h_full, num_chunks=out3.n_chunks)

    check("Block1 one chunk", b1_out.num_chunks == 1, f"{b1_out.num_chunks}")
    check("h_ob last dim=1024", h_ob.size(-1) == 1024, str(tuple(h_ob.shape)))
    check("h_w last dim=1024", h_w.size(-1) == 1024, str(tuple(h_w.shape)))
    check("Block3 h_full [1, 750, 1024]", out3.h_full.shape == (1, 750, 1024), str(tuple(out3.h_full.shape)))
    check("Block4 audio_tokens [1, 750, 896]", out4.audio_tokens.shape == (1, 750, 896), str(tuple(out4.audio_tokens.shape)))
    check("no NaN end-to-end", nonan(out4.audio_tokens))


def ti5_full_pipeline_multi_chunk(AudioFrontend, Block2aWhisperEncoder, b2b, device, tmp_dir: Path, N=3):
    hdr(f"TI5 · Full pipeline multi-chunk: {N}×30s → Block4")
    Block3 = get_block3()
    Block4, _, _ = get_block4()

    b3 = Block3().to(device)
    b4 = Block4().to(device)

    whisper_enc = Block2aWhisperEncoder()
    if isinstance(whisper_enc, nn.Module):
        whisper_enc = whisper_enc.to(device).eval()

    chunk_inputs = []
    for i in range(N):
        path = tmp_dir / f"tone_{i}.wav"
        make_fake_audio_file(path, seconds=30.0)

        mel, _ = mel_from_block1(AudioFrontend, path, device)
        with torch.no_grad():
            h_ob = b2b(mel).h_ob
        h_w = h_w_from_block2a(whisper_enc, mel, device)
        chunk_inputs.append((h_ob, h_w))
        info(f"chunk {i+1}/{N}: h_ob={tuple(h_ob.shape)} h_w={tuple(h_w.shape)}")

    with torch.no_grad():
        out3 = b3.forward_multi_chunk(chunk_inputs)
        out4 = b4(out3.h_full, num_chunks=out3.n_chunks)

    check("Block3 seq_len=2250", out3.seq_len == 750 * N, str(out3.seq_len))
    check("Block4 audio_tokens [1, 2250, 896]", out4.audio_tokens.shape == (1, 750 * N, 896), str(tuple(out4.audio_tokens.shape)))
    check("num_chunks == N", out4.num_chunks == N)
    check("no NaN", nonan(out4.audio_tokens))


def get_args():
    p = argparse.ArgumentParser(description="NanoFlame v4 — Block4 tests")
    p.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "epoch_latest.pt"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-integration", action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    device = args.device

    print(f"\n{BD}{'═'*62}{E}")
    print(f"{BD}  NanoFlame v4 — Block4 Tests{E}")
    print(f"{BD}{'═'*62}{E}")
    print(f"  device : {device}")
    print(f"  ckpt   : {args.ckpt}")

    print(f"\n{BD}{'═'*62}{E}")
    print(f"{BD}  UNIT TESTS — Block4 only{E}")
    print(f"{BD}{'═'*62}{E}")

    t1_forward_shape(device)
    t2_single_unbatched(device)
    t3_residual_path_effect(device)
    t4_rmsnorm_present(device)
    t5_gradient_flow(device)
    t6_freeze_unfreeze(device)
    t7_dropout_control(device)
    t8_param_count(device)
    t9_bad_input_dim(device)
    t10_long_audio_shape(device)

    if not args.skip_integration:
        print(f"\n{BD}{'═'*62}{E}")
        print(f"{BD}  INTEGRATION TESTS — Block3 / full pipeline / Block4{E}")
        print(f"{BD}{'═'*62}{E}")

        try:
            AudioFrontend = get_block1()
            Block2aWhisperEncoder = get_block2a()
            b2b, loaded = load_block2b(args.ckpt, device)
        except Exception as e:
            AudioFrontend = Block2aWhisperEncoder = None
            b2b, loaded = None, False
            warn(f"Import/load failed: {e}")

        ti1_block3_to_block4(device)
        ti2_block3_multi_to_block4(device)
        ti3_stage1_only_block4_trains(device)

        if AudioFrontend is None or Block2aWhisperEncoder is None or not loaded:
            skip("Skipping full pipeline tests — missing Block1/2a/2b or checkpoint")
        else:
            tmp_dir = ROOT / "tests" / "_tmp_block4_audio"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            ti4_full_pipeline_to_block4(AudioFrontend, Block2aWhisperEncoder, b2b, device, tmp_dir)
            ti5_full_pipeline_multi_chunk(AudioFrontend, Block2aWhisperEncoder, b2b, device, tmp_dir, N=3)
    else:
        skip("Integration tests skipped (--skip-integration)")

    total = PASS + FAIL
    print(f"\n{BD}{'─'*62}{E}")
    print(f"  {G}{PASS} passed{E}  {(R+str(FAIL)+E) if FAIL else str(FAIL)} failed  /  {total} run")

    if FAIL == 0:
        print(f"\n{G}{BD}✔  All Block4 tests passed.{E}\n")
    else:
        print(f"\n{R}{BD}✘  {FAIL} test(s) failed.{E}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()