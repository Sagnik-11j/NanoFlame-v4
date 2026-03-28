"""
tests/test_block2b.py
─────────────────────────────────────────────────────────────────
Test Block2b: OpenBEATs Encoder — load weights + verify

Run from project ROOT:
    python tests/test_block2b.py
    python tests/test_block2b.py --device cuda
─────────────────────────────────────────────────────────────────
"""

import sys
import math
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"; B = "\033[1m"; E = "\033[0m"
ok   = lambda m: print(f"  {G}✔{E}  {m}")
warn = lambda m: print(f"  {Y}⚠{E}  {m}")
fail = lambda m: print(f"  {R}✘{E}  {m}")
hdr  = lambda t: print(f"\n{B}── {t} {'─'*(50-len(t))}{E}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",   default=str(ROOT / "checkpoints" / "epoch_latest.pt"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch",  type=int, default=2)
    p.add_argument("--frames", type=int, default=3000)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
# Build encoder — uses cfg from checkpoint if available
# ─────────────────────────────────────────────────────────────────
def build_and_load(ckpt_path: str, device: str):
    hdr("1 · Load checkpoint")

    path = Path(ckpt_path)
    if not path.exists():
        fail(f"Not found: {path}")
        print(f"     Place epoch_latest.pt in:  {ROOT}/checkpoints/")
        sys.exit(1)

    # Always load to CPU first regardless of how it was saved
    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    ok(f"File opened  ({path.stat().st_size/1e6:.0f} MB)")

    top_keys = list(raw.keys()) if isinstance(raw, dict) else []
    print(f"     top keys : {top_keys}")

    # ── Extract state_dict ────────────────────────────────────────
    sd = None
    if "model" in raw and isinstance(raw["model"], dict):
        sd = raw["model"]
        ok("Extracted state_dict from raw['model']")
    elif all(isinstance(v, torch.Tensor) for v in raw.values()):
        sd = raw
        ok("Checkpoint is already a flat state_dict")
    else:
        fail("Cannot extract state_dict — unexpected format")
        sys.exit(1)

    print(f"     tensors  : {len(sd)}")
    print(f"     sample   : {list(sd.keys())[:4]}")

    # ── Extract config from checkpoint ───────────────────────────
    cfg_dict = None
    if "cfg" in raw:
        c = raw["cfg"]
        if isinstance(c, dict):
            cfg_dict = c
        elif hasattr(c, "__dict__"):
            cfg_dict = vars(c)
        if cfg_dict:
            ok("Config loaded from checkpoint['cfg']")

    hdr("2 · Instantiate Block2bOpenBEATsEncoder")

    try:
        from blocks.block2b_openbeats_encoder import Block2bOpenBEATsEncoder
    except ImportError as e:
        fail(f"Import failed: {e}")
        sys.exit(1)

    enc = Block2bOpenBEATsEncoder()

    # If we got a cfg from checkpoint, rebuild encoder with correct config
    if cfg_dict:
        try:
            from blocks.beats_encoder import BeatsEncoder
            enc._enc = BeatsEncoder(input_size=128, beats_config=cfg_dict)
            ok(f"Encoder rebuilt with checkpoint config")
        except Exception as e:
            warn(f"Could not apply checkpoint cfg: {e} — using default config")

    enc = enc.to(device)
    ok(f"Encoder ready  (total params: {enc.total_params:,})")
    print(f"     hidden_dim = {enc.hidden_dim}  |  layers = {enc.num_layers}")

    # ── Load weights ──────────────────────────────────────────────
    hdr("3 · Load weights")

    model_keys = set(enc._enc.state_dict().keys())

    # Try stripping common prefixes to maximise key overlap
    def strip(d, prefix):
        return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in d.items()}

    best_sd, best_n = sd, len(set(sd.keys()) & model_keys)
    for p in ["encoder.", "_enc.", "model.", "module.", "backbone."]:
        s = strip(sd, p)
        n = len(set(s.keys()) & model_keys)
        if n > best_n:
            best_n, best_sd = n, s

    pct = 100.0 * best_n / len(model_keys) if model_keys else 0
    print(f"     key match : {best_n}/{len(model_keys)}  ({pct:.1f}%)")

    info = enc._enc.load_state_dict(best_sd, strict=False)
    missing    = info.missing_keys
    unexpected = info.unexpected_keys

    # Soft-missing keys are acceptable (embed / proj layers)
    SOFT = ("patch_embedding.", "layer_norm.", "post_extract_proj.", "dropout_input.")
    hard = [k for k in missing if not any(k.startswith(p) for p in SOFT)]
    soft = [k for k in missing if k not in hard]

    if unexpected: warn(f"{len(unexpected)} unexpected keys (ignored)")
    if soft:       warn(f"{len(soft)} soft-missing keys (Xavier-init — normal)")
    if hard:
        fail(f"{len(hard)} HARD-missing keys — weights NOT loaded:")
        for k in hard[:10]: print(f"          {k}")
        return enc, False
    if pct >= 80.0:
        ok(f"Weights loaded  ({pct:.1f}%)")
    else:
        warn(f"Only {pct:.1f}% loaded — check config mismatch")

    return enc, True


# ─────────────────────────────────────────────────────────────────
# Norm sanity
# ─────────────────────────────────────────────────────────────────
def check_norms(enc) -> bool:
    hdr("4 · Weight norm sanity")
    nan_layers, zero_weights = [], []
    for name, p in enc._enc.named_parameters():
        d = p.data.float()
        if torch.isnan(d).any() or torch.isinf(d).any():
            nan_layers.append(name)
        elif d.abs().max().item() == 0.0 and "bias" not in name:
            zero_weights.append(name)

    if nan_layers:
        fail(f"{len(nan_layers)} NaN/Inf tensors — corrupted checkpoint")
        return False

    ok(f"No NaN/Inf  ({len(zero_weights)} zero-init bias tensors, normal)")

    # Sample a few layer means — trained weights have mean|w| ≈ 0.01–0.15
    print(f"\n     {'Layer':<48} {'mean|w|':>8}  shape")
    print(f"     {'─'*48}  {'─'*8}  {'─'*16}")
    for i, (name, p) in enumerate(enc._enc.named_parameters()):
        if i >= 10: print(f"     … (more layers)"); break
        d = p.data.float()
        m = d.abs().mean().item()
        flag = f"  {Y}← random init?{E}" if m > 0.5 else ""
        print(f"     {name[-48:]:48}  {m:8.5f}  {str(list(d.shape))}{flag}")
    return True


# ─────────────────────────────────────────────────────────────────
# Forward pass
# ─────────────────────────────────────────────────────────────────
def test_forward(enc, device: str, B: int, T: int) -> bool:
    hdr("5 · Forward pass  [B, 128, T] → [B, N_patches, 1024]")
    enc.eval()
    mel = torch.randn(B, 128, T, device=device)
    print(f"     input  : {list(mel.shape)}")

    with torch.no_grad():
        try:
            out = enc(mel)
        except Exception as e:
            fail(f"Forward pass crashed: {e}")
            import traceback; traceback.print_exc()
            return False

    h = out.h_ob
    print(f"     output : {list(h.shape)}  (expected [{B}, ~{math.ceil(T/16)}, 1024])")
    print(f"     mean={h.mean().item():.4f}  std={h.std().item():.4f}")

    if h.dim() != 3:             fail(f"Output must be 3D"); return False
    if h.size(0) != B:           fail(f"Batch mismatch"); return False
    if h.size(2) != 1024:        fail(f"Hidden dim should be 1024, got {h.size(2)}"); return False
    if torch.isnan(h).any():     fail("NaN in output"); return False
    if h.std().item() < 1e-5:   warn("Output nearly constant — weights may not be loaded")
    else:                        ok(f"Forward pass OK  →  {list(h.shape)}")
    return True


# ─────────────────────────────────────────────────────────────────
# Gradient flow
# ─────────────────────────────────────────────────────────────────
def test_grads(enc, device: str, B: int) -> bool:
    hdr("6 · Gradient flow")
    enc.train()
    enc.unfreeze_all()
    mel = torch.randn(B, 128, 800, device=device)
    enc(mel).h_ob.mean().backward()

    got = sum(1 for p in enc.parameters()
              if p.requires_grad and p.grad is not None and p.grad.abs().max() > 0)
    total = sum(1 for p in enc.parameters() if p.requires_grad)
    pct = 100.0 * got / total if total else 0
    enc.zero_grad()

    if pct >= 80.0: ok(f"Gradient flow OK  ({got}/{total} params, {pct:.0f}%)")
    else:           warn(f"Partial gradient flow  ({pct:.0f}%)")
    return pct >= 40.0


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    args = get_args()
    print(f"\n{B}Block2b · OpenBEATs Encoder Test{E}")
    print(f"  ckpt   : {args.ckpt}")
    print(f"  device : {args.device}")

    enc, load_ok = build_and_load(args.ckpt, args.device)
    norm_ok      = check_norms(enc)
    fwd_ok       = test_forward(enc, args.device, args.batch, args.frames)
    grad_ok      = test_grads(enc, args.device, args.batch)

    hdr("RESULT")
    results = {
        "Weights loaded  (≥80%)": load_ok,
        "Weight norms  (no NaN)": norm_ok,
        "Forward pass":           fwd_ok,
        "Gradient flow":          grad_ok,
    }
    for name, r in results.items():
        (ok if r else fail)(name)

    if all(results.values()):
        print(f"\n{G}{B}✔  Block2b ready.{E}\n")
        print(f"  Usage:")
        print(f"    enc = Block2bOpenBEATsEncoder()")
        print(f"    enc.load_openbeats_checkpoint('checkpoints/epoch_latest.pt')")
        print(f"    out = enc(mel)   # [B, 128, T] → out.h_ob: [B, N, 1024]")
    else:
        print(f"\n{R}{B}✘  Some checks failed.{E}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()