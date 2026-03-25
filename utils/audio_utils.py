# nanoflame/utils/audio_utils.py
# ─────────────────────────────────────────────────────────────────────────────
# Utility functions shared across blocks.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
from pathlib import Path
from typing import List
import torch


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".aiff"}


def validate_audio_path(path: str | Path) -> Path:
    """Check that the file exists and has a supported extension."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{p.suffix}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return p


def chunks_to_device(
    chunks : List[torch.Tensor],
    device : str | torch.device,
) -> List[torch.Tensor]:
    """Move a list of chunk tensors to the specified device."""
    return [c.to(device) for c in chunks]


def describe_chunks(chunks: List[torch.Tensor]) -> str:
    """Human-readable summary of a chunk list."""
    if not chunks:
        return "No chunks."
    n   = len(chunks)
    shp = chunks[0].shape
    dur = n * 30
    return (
        f"{n} chunk(s) × {shp}  "
        f"(~{dur}s of audio, "
        f"min={chunks[0].min():.2f}, max={chunks[0].max():.2f})"
    )
