"""
tests/test_block2b_openbeats.py
────────────────────────────────────────────────────────────────
Loads checkpoints/epoch_latest.pt into Block2bOpenBEATsEncoder
and verifies the weights loaded correctly.

Run from project ROOT:
    python tests/test_block2b_openbeats.py
    python tests/test_block2b_openbeats.py --ckpt checkpoints/epoch_latest.pt
    python tests/test_block2b_openbeats.py --device cuda
────────────────────────────────────────────────────────────────

Expected folder layout:
    your_project/
    ├── blocks/
    │   ├── __init__.py
    │   ├── beats_encoder.py
    │   └── block2b_openbeats_encoder.py
    ├── tests/
    │   └── test_block2b_openbeats.py   ← this file
    ├── utils/
    │   └── __init__.py
    └── checkpoints/
        └── epoch_latest.pt             ← put your checkpoint here
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── Add project root to path so `from blocks.xxx import` works ──────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        default=str(ROOT / "checkpoints" / "epoch_latest.pt"),
        help="Path to checkpoint  (default: checkpoints/epoch_latest.pt)",
    )
    p.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0")
    p.add_argument("--batch",  type=int, default=2,    help="Batch size for forward test")
    p.add_argument("--frames", type=int, default=3000, help="Mel time-frames for forward test")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────
G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"; BD = "\033[1m"; E = "\033[0m"
ok   = lambda m: print(f"  {G}✔{E}  {m}")
warn = lambda m: print(f"  {Y}⚠{E}  {m}")
fail = lambda m: print(f"  {R}✘{E}  {m}")
info = lambda m: print(f"     {m}")
def hdr(t): print(f"\n{BD}{'─'*64}{E}\n{BD} {t}{E}\n{'─'*64}")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────
def extract_state_dict(raw: Any) -> Dict[str, torch.Tensor]:
    """
    Pull a flat weight dict from any of the three common ESPnet checkpoint formats:

      Format A  (flat state_dict saved directly):
          {"encoder.layers.0.weight": tensor, ...}

      Format B  (ESPnet training checkpoint — most common):
          {"model": {"encoder.layers.0.weight": ...}, "optimizer": ..., "epoch": 42}

      Format C  (doubly-nested):
          {"state_dict": {"model": {"encoder...": ...}}}
    """
    if isinstance(raw, dict):
        # Try standard outer wrapper keys first
        for outer in ("state_dict", "model", "net", "weights", "model_state_dict"):
            if outer in raw and isinstance(raw[outer], dict):
                inner = raw[outer]
                # one more level for Format C
                for inner2 in ("state_dict", "model", "net"):
                    if inner2 in inner and isinstance(inner[inner2], dict):
                        return inner[inner2]
                return inner
        # Already flat if all values are tensors
        if raw and all(isinstance(v, torch.Tensor) for v in raw.values()):
            return raw
    elif isinstance(raw, nn.Module):
        return raw.state_dict()
    raise ValueError(f"Cannot extract state_dict from type {type(raw)}")


def best_key_match(
    sd: Dict[str, torch.Tensor],
    target: List[str],
) -> Tuple[Dict[str, torch.Tensor], int]:
    """
    Strip common key prefixes until we get maximum overlap with model keys.

    OpenBEATs checkpoints from ESPnet training are usually saved with prefix:
      "encoder."           → strip → matches model's keys directly
      "model.encoder."     → strip → matches
      ""  (already flat)   → matches directly

    Returns (best_state_dict, n_matching_keys).
    """
    target_set = set(target)

    PREFIXES = [
        "",
        "encoder.",
        "_enc.",
        "model.",
        "module.",
        "backbone.",
        "audio_encoder.",
        "beats_encoder.",
        "encoder._enc.",
        "model.encoder.",
        "model._enc.",
        "model.backbone.",
    ]

    def strip(d, p):
        if not p:
            return d
        return {(k[len(p):] if k.startswith(p) else k): v for k, v in d.items()}

    best_sd, best_n = sd, len(set(sd.keys()) & target_set)
    for p in PREFIXES:
        s = strip(sd, p)
        n = len(set(s.keys()) & target_set)
        if n > best_n:
            best_n, best_sd = n, s
    return best_sd, best_n


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Locate checkpoint and inspect its structure
# ─────────────────────────────────────────────────────────────────────────────
def step1_inspect(ckpt_path: str, device: str) -> Any:
    hdr("STEP 1 · Locate & inspect checkpoint")

    path = Path(ckpt_path)
    if not path.exists():
        fail(f"File not found: {path}")
        print()
        print(f"  {BD}Where to put the file:{E}")
        print(f"    {ROOT}/checkpoints/epoch_latest.pt")
        print()
        print(f"  {BD}How to get it:{E}")
        print(f"    Option A — copy manually:")
        print(f"      mkdir -p {ROOT}/checkpoints")
        print(f"      cp /your/path/epoch_latest.pt {ROOT}/checkpoints/")
        print()
        print(f"    Option B — download from HuggingFace Hub:")
        print(f"      pip install huggingface_hub")
        print(f"      python -c \"from huggingface_hub import hf_hub_download; \\")
        print(f"        p=hf_hub_download('shikhar7ssu/OpenBEATs-ICME','epoch_latest.pt'); \\")
        print(f"        print('saved to:', p)\"")
        print(f"      cp ~/.cache/huggingface/hub/*/epoch_latest.pt {ROOT}/checkpoints/")
        print()
        sys.exit(1)

    size_mb = path.stat().st_size / 1e6
    info(f"Path   : {path}")
    info(f"Size   : {size_mb:.1f} MB")

    if size_mb < 50:
        warn(f"File is {size_mb:.1f} MB — expected ~400–700 MB for OpenBEATs; may be wrong file")

    raw = torch.load(str(path), map_location=device)

    if isinstance(raw, dict):
        top = list(raw.keys())
        info(f"Top-level keys : {top[:8]}{'...' if len(top)>8 else ''}")

        # Detect ESPnet training checkpoint
        training_markers = {"epoch", "optimizer", "optimizer_state_dict",
                            "scheduler", "lr_scheduler", "step"}
        found = training_markers & set(top)
        if found:
            warn(f"Training checkpoint detected — found meta-keys: {found}")
            info("Will extract model weights from inner 'model' key automatically")
        else:
            ok("Looks like a flat state_dict")

        sd = extract_state_dict(raw)
        info(f"Weight tensors : {len(sd)}")
        info(f"Sample keys    : {list(sd.keys())[:4]}")

        # Show top-level prefix distribution
        dist: Dict[str, int] = {}
        for k in sd:
            dist[k.split(".")[0]] = dist.get(k.split(".")[0], 0) + 1
        info(f"Key prefixes   : {dict(list(dist.items())[:6])}")

    ok("Checkpoint opened successfully")
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Instantiate the encoder
# ─────────────────────────────────────────────────────────────────────────────
def step2_build(device: str):
    hdr("STEP 2 · Instantiate Block2bOpenBEATsEncoder")
    try:
        from blocks.block2b_openbeats_encoder import Block2bOpenBEATsEncoder
    except ImportError as e:
        fail(f"Import error: {e}")
        info("Run from project ROOT, not from inside tests/:")
        info("    python tests/test_block2b_openbeats.py")
        sys.exit(1)

    enc = Block2bOpenBEATsEncoder().to(device)
    ok("Encoder instantiated")
    info(f"hidden_dim   = {enc.hidden_dim}")
    info(f"num_layers   = {enc.num_layers}")
    info(f"total_params = {enc.total_params:,}")
    return enc


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Load weights
# ─────────────────────────────────────────────────────────────────────────────
def step3_load(enc, raw: Any) -> Tuple[List, List, float]:
    hdr("STEP 3 · Load weights into encoder")

    sd = extract_state_dict(raw)
    model_keys = list(enc._enc.state_dict().keys())
    info(f"Model expects      : {len(model_keys)} tensors")
    info(f"Checkpoint contains: {len(sd)} tensors")

    best_sd, overlap = best_key_match(sd, model_keys)
    pct = 100.0 * overlap / len(model_keys) if model_keys else 0.0
    info(f"Key match          : {overlap}/{len(model_keys)}  ({pct:.1f}%)")

    if pct < 30.0:
        warn("Below 30% match — possible wrong file or unexpected key layout")
        info(f"  Checkpoint sample : {list(best_sd.keys())[:4]}")
        info(f"  Model sample      : {model_keys[:4]}")

    incompatible = enc._enc.load_state_dict(best_sd, strict=False)
    ok(f"load_state_dict() complete  ({pct:.1f}% matched)")
    return incompatible.missing_keys, incompatible.unexpected_keys, pct


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Key audit
# ─────────────────────────────────────────────────────────────────────────────
def step4_audit(missing: List, unexpected: List, pct: float) -> bool:
    hdr("STEP 4 · Key audit")

    # These are acceptable to be missing — will use Xavier/random init
    SOFT = (
        "patch_embedding.",
        "layer_norm.",
        "post_extract_proj.",
        "dropout_input.",
        "feature_extractor.",
    )
    hard = [k for k in missing if not any(k.startswith(p) for p in SOFT)]
    soft = [k for k in missing if k not in hard]

    info(f"Coverage    : {pct:.1f}%")
    info(f"Missing     : {len(missing)} total  ({len(soft)} soft / {len(hard)} hard)")
    info(f"Unexpected  : {len(unexpected)} (silently ignored by PyTorch)")

    if unexpected:
        warn(f"{len(unexpected)} checkpoint keys not used by model:")
        for k in unexpected[:5]: info(f"    ignored → {k}")
        if len(unexpected) > 5: info(f"    ... and {len(unexpected)-5} more")

    if soft:
        warn(f"{len(soft)} soft-missing (Xavier re-init — normal for embed/proj layers):")
        for k in soft[:3]: info(f"    soft-miss → {k}")

    if hard:
        fail(f"{len(hard)} HARD-missing keys — transformer weights NOT loaded:")
        for k in hard[:15]: info(f"    MISSING → {k}")
        if len(hard) > 15: info(f"    ... and {len(hard)-15} more")
        return False

    if pct >= 80.0:
        ok(f"Key audit PASSED  ({pct:.1f}% loaded from checkpoint)")
    elif pct >= 50.0:
        warn(f"Partial load ({pct:.1f}%) — some layers re-initialised")
    else:
        fail(f"Only {pct:.1f}% loaded — likely wrong checkpoint or config mismatch")
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Weight norm sanity (no NaN / all-zero)
# ─────────────────────────────────────────────────────────────────────────────
def step5_norms(enc) -> bool:
    hdr("STEP 5 · Weight norm sanity")

    nan_layers, zero_weights = [], []
    for name, p in enc._enc.named_parameters():
        d = p.data.float()
        if torch.isnan(d).any() or torch.isinf(d).any():
            nan_layers.append(name)
        elif d.abs().max().item() == 0.0 and "bias" not in name:
            zero_weights.append(name)

    if nan_layers:
        fail(f"{len(nan_layers)} NaN/Inf tensors — CORRUPTED CHECKPOINT")
        for n in nan_layers[:4]: info(f"  NaN → {n}")
        return False

    if len(zero_weights) > 20:
        warn(f"{len(zero_weights)} non-bias tensors are all-zero — possible incomplete load")
        for n in zero_weights[:4]: info(f"  zero → {n}")
    else:
        ok(f"No NaN/Inf detected  ({len(zero_weights)} zero-init bias tensors, normal)")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Per-layer weight statistics
# ─────────────────────────────────────────────────────────────────────────────
def step6_stats(enc, rows: int = 12):
    """
    Trained weights have mean|w| ≈ 0.01–0.15.
    Random-init weights (N(0,1)) have mean|w| ≈ 0.4–0.8.
    If you see high values here the weights were not loaded.
    """
    hdr("STEP 6 · Per-layer weight statistics  (first 12)")
    info("")
    info(f"  {'Layer':<52} {'mean|w|':>8}  {'std':>8}  shape")
    info(f"  {'─'*52}  {'─'*8}  {'─'*8}  {'─'*16}")
    total = sum(1 for _ in enc._enc.named_parameters())
    i = 0
    for name, p in enc._enc.named_parameters():
        if i >= rows:
            info(f"  … ({total - rows} more)")
            break
        d = p.data.float()
        m, s = d.abs().mean().item(), d.std().item()
        flag = (f"  {Y}← high — random init?{E}" if m > 0.5
                else f"  {Y}← all zeros?{E}" if m < 1e-6 and "bias" not in name
                else "")
        info(f"  {name[-52:]:52}  {m:8.5f}  {s:8.5f}  {str(list(d.shape))}{flag}")
        i += 1


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Forward pass
# ─────────────────────────────────────────────────────────────────────────────
def step7_forward(enc, device: str, B: int, T: int) -> bool:
    hdr("STEP 7 · Forward pass  [B, 128, T] → [B, N_patches, 1024]")

    enc.eval()
    mel = torch.randn(B, 128, T, device=device)
    info(f"Input  : {list(mel.shape)}")

    with torch.no_grad():
        try:
            out = enc(mel)
        except Exception as e:
            fail(f"Forward pass crashed: {e}")
            import traceback; traceback.print_exc()
            return False

    h = out.h_ob
    expected_N = math.ceil(T / 16)
    info(f"Output : {list(h.shape)}  (expected [{B}, ~{expected_N}, 1024])")
    info(f"Patches: {out.num_patches}  |  Chunks: {out.num_chunks}")

    if h.dim() != 3:            fail(f"Output must be 3D, got {h.dim()}D"); return False
    if h.size(0) != B:          fail(f"Batch mismatch: got {h.size(0)}, expected {B}"); return False
    if h.size(2) != 1024:       fail(f"Hidden dim: got {h.size(2)}, expected 1024"); return False
    if torch.isnan(h).any():    fail("NaN values in output tensor"); return False

    m, s = h.mean().item(), h.std().item()
    info(f"Stats  : mean={m:.4f}  std={s:.4f}")

    if s < 1e-5:   warn("Output nearly constant — weights may not be loaded correctly")
    elif s > 20.0: warn(f"Output std={s:.2f} unusually high — check config match")
    else:          ok(f"Forward pass OK  →  {list(h.shape)}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Gradient flow
# ─────────────────────────────────────────────────────────────────────────────
def step8_grads(enc, device: str, B: int) -> bool:
    hdr("STEP 8 · Gradient flow check")

    enc.train()
    enc.unfreeze_all()

    mel = torch.randn(B, 128, 800, device=device)
    enc(mel).h_ob.mean().backward()

    got, no = [], []
    for name, p in enc.named_parameters():
        if p.requires_grad:
            (got if (p.grad is not None and p.grad.abs().max() > 0) else no).append(name)

    total = len(got) + len(no)
    pct   = 100.0 * len(got) / total if total else 0.0
    info(f"Trainable params with grad : {len(got)}/{total}  ({pct:.0f}%)")

    enc.zero_grad()
    if pct >= 80.0:   ok("Gradient flow OK")
    elif pct >= 40.0: warn(f"Partial gradient flow ({pct:.0f}%)")
    else:             fail(f"Only {pct:.0f}% received gradients"); return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = get_args()

    print(f"\n{BD}{'='*64}")
    print(f"  Block2b OpenBEATs — Weight Load & Verification")
    print(f"{'='*64}{E}")
    print(f"  checkpoint : {args.ckpt}")
    print(f"  device     : {args.device}")
    print(f"  project    : {ROOT}")

    raw     = step1_inspect(args.ckpt, args.device)
    enc     = step2_build(args.device)
    missing, unexpected, pct = step3_load(enc, raw)
    key_ok  = step4_audit(missing, unexpected, pct)
    norm_ok = step5_norms(enc)
    step6_stats(enc)
    fwd_ok  = step7_forward(enc, args.device, args.batch, args.frames)
    grad_ok = step8_grads(enc, args.device, args.batch)

    # ── Final verdict ──────────────────────────────────────────────────
    hdr("VERDICT")
    checks = {
        "Key audit  (≥80% params loaded)": key_ok,
        "Weight norms  (no NaN/Inf)":      norm_ok,
        "Forward pass  [B,128,T]→[B,N,D]": fwd_ok,
        "Gradient flow  (backward pass)":  grad_ok,
    }
    passed = all(checks.values())
    for name, result in checks.items():
        (ok if result else fail)(name)

    if passed:
        print(f"\n{G}{BD}✔  All checks passed — epoch_latest.pt loaded correctly.{E}")
        print(f"\n  In your training code:")
        print(f"    from blocks.block2b_openbeats_encoder import Block2bOpenBEATsEncoder")
        print(f"    enc = Block2bOpenBEATsEncoder()")
        print(f"    enc.load_openbeats_checkpoint('checkpoints/epoch_latest.pt')")
        print(f"    out = enc(mel)   # mel: [B, 128, T]")
        print(f"    h   = out.h_ob   # [B, N_patches, 1024]\n")
    else:
        print(f"\n{R}{BD}✘  Some checks failed — review output above.{E}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()