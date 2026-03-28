from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# ESPnet2 stubs
# ─────────────────────────────────────────────────────────────────────────────
def _inject_espnet2_stubs() -> None:
    """
    Inject minimal stubs so beats_encoder.py can be imported without ESPnet2.

    Covers:
      from espnet2.asr.encoder.abs_encoder import AbsEncoder
      from espnet2.asr.specaug.specaug import SpecAug
      from espnet2.legacy.nets.pytorch_backend.nets_utils import make_pad_mask, roll_tensor
    """
    if "espnet2" in sys.modules:
        return

    def _mod(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    _mod("espnet2")
    _mod("espnet2.asr")
    _mod("espnet2.asr.encoder")
    abs_enc_mod = _mod("espnet2.asr.encoder.abs_encoder")
    _mod("espnet2.asr.specaug")
    specaug_mod = _mod("espnet2.asr.specaug.specaug")
    _mod("espnet2.legacy")
    _mod("espnet2.legacy.nets")
    _mod("espnet2.legacy.nets.pytorch_backend")
    nets_utils = _mod("espnet2.legacy.nets.pytorch_backend.nets_utils")

    class AbsEncoder(nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()

    class SpecAug(nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()
        def forward(self, x, x_lengths=None):
            return x, x_lengths

    def make_pad_mask(
        lengths: torch.Tensor,
        maxlen: Optional[int] = None,
        dim: int = 1,
    ) -> torch.Tensor:
        if lengths.dim() != 1:
            raise ValueError(f"lengths must be 1D, got {tuple(lengths.shape)}")
        B = lengths.size(0)
        L = maxlen if maxlen is not None else int(lengths.max().item())
        idx = torch.arange(L, device=lengths.device).unsqueeze(0).expand(B, L)
        return idx >= lengths.unsqueeze(1)

    def roll_tensor(x: torch.Tensor, shift: int = 0, dim: int = 0) -> torch.Tensor:
        return torch.roll(x, shifts=shift, dims=dim)

    abs_enc_mod.AbsEncoder = AbsEncoder
    specaug_mod.SpecAug = SpecAug
    nets_utils.make_pad_mask = make_pad_mask
    nets_utils.roll_tensor = roll_tensor


_inject_espnet2_stubs()


# ─────────────────────────────────────────────────────────────────────────────
# Import beats_encoder (supports both package and standalone usage)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from .beats_encoder import BeatsConfig, BeatsEncoder
except Exception:
    from beats_encoder import BeatsConfig, BeatsEncoder


# ─────────────────────────────────────────────────────────────────────────────
# OpenBEATs config overrides (24-layer / 1024-dim model)
#
# IMPORTANT: This must stay as a plain dict.
# BeatsEncoder.__init__ calls config.update(beats_config) which calls
# self.__dict__.update(cfg) — this requires a dict, NOT a BeatsConfig object.
# ─────────────────────────────────────────────────────────────────────────────
_OPENBEATS_DEFAULT_CFG: Dict[str, Any] = {
    "input_patch_size": 16,
    "embed_dim": 512,
    "conv_bias": False,
    "encoder_layers": 24,
    "encoder_embed_dim": 1024,
    "encoder_ffn_embed_dim": 4096,
    "encoder_attention_heads": 16,
    "activation_fn": "gelu",
    "layer_norm_first": False,
    "deep_norm": False,
    "dropout": 0.0,
    "attention_dropout": 0.0,
    "activation_dropout": 0.0,
    "encoder_layerdrop": 0.0,
    "dropout_input": 0.0,
    "conv_pos": 128,
    "conv_pos_groups": 16,
    "relative_position_embedding": True,
    "num_buckets": 320,
    "max_distance": 1280,
    "gru_rel_pos": True,
    "finetuned_model": False,
    "predictor_dropout": 0.0,
    "predictor_class": 527,
}


@dataclass
class OpenBEATsEncoderOutput:
    h_ob: torch.Tensor        # [B, N_patches, 1024]
    num_patches: int
    num_chunks: int


# ─────────────────────────────────────────────────────────────────────────────
# Block2bOpenBEATsEncoder
# ─────────────────────────────────────────────────────────────────────────────
class Block2bOpenBEATsEncoder(nn.Module):
    """
    Block 2b: wraps BeatsEncoder from beats_encoder.py for use in the
    ALM pipeline where Block 1 already produces mel [B, 128, T].

    Does NOT call BeatsEncoder.preprocess() — routes directly into
    patch_embedding → transformer stack.
    """

    def __init__(
        self,
        hf_repo_id: str = "shikhar7ssu/OpenBEATs-ICME",
        checkpoint_filename: Optional[str] = None,
        local_checkpoint: Optional[str] = None,
        enable_init: bool = True,
    ) -> None:
        super().__init__()

        self.hf_repo_id = hf_repo_id
        self.checkpoint_filename = checkpoint_filename
        self.local_checkpoint = local_checkpoint

        # Keep a BeatsConfig for introspection (hidden_dim, num_layers etc.)
        # but pass the raw DICT to BeatsEncoder — it calls __dict__.update(dict)
        self.cfg = BeatsConfig()
        for k, v in _OPENBEATS_DEFAULT_CFG.items():
            setattr(self.cfg, k, v)

        # ← FIX: pass dict, not BeatsConfig object
        self._enc = BeatsEncoder(input_size=128, beats_config=_OPENBEATS_DEFAULT_CFG)

        if enable_init and hasattr(self._enc, "reload_pretrained_parameters"):
            try:
                self._enc.reload_pretrained_parameters()
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────
    # Checkpoint loading
    # ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
        """
        Pull a flat weight dict from any common checkpoint format.

        The shikhar7ssu checkpoint has top-level keys ['model', 'cfg'].
        We extract checkpoint['model'] as the state_dict.
        """
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model", "net", "weights", "model_state_dict"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    return checkpoint[key]
            # Already flat if all values are tensors
            if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                return checkpoint
        elif isinstance(checkpoint, nn.Module):
            return checkpoint.state_dict()
        raise ValueError(f"Cannot extract state_dict from checkpoint type {type(checkpoint)}")

    @staticmethod
    def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}

    def _best_state_dict_variant(
        self,
        state_dict: Dict[str, torch.Tensor],
        target_keys: Iterable[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Try common key prefixes and return the variant with max overlap
        against the model's own keys.
        """
        target = set(target_keys)
        prefixes = [
            "",
            "module.",
            "_enc.",
            "model.",
            "backbone.",
            "network.",
            "wrapper.",
            "beats_encoder.",
            "audio_encoder.",
            "encoder.",
        ]

        best_sd  = state_dict
        best_n   = len(set(state_dict.keys()) & target)

        candidates = [state_dict]
        for p in prefixes:
            if p:
                candidates.append(self._strip_prefix(state_dict, p))

        for sd in candidates:
            n = len(set(sd.keys()) & target)
            if n > best_n:
                best_n, best_sd = n, sd

        return best_sd

    def _download_from_hf(self) -> str:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required to auto-download. "
                "Run: pip install huggingface_hub"
            ) from e

        candidates = []
        if self.checkpoint_filename:
            candidates.append(self.checkpoint_filename)
        candidates += [
            "epoch_latest.pt",
            "OpenBEATs_iter3_plus_AS2M.pt",
            "OpenBEATs_AS2M.pt",
            "openbeats_encoder.pt",
            "pytorch_model.bin",
        ]

        last_err = None
        for fname in candidates:
            try:
                return hf_hub_download(repo_id=self.hf_repo_id, filename=fname)
            except Exception as e:
                last_err = e

        raise FileNotFoundError(
            f"No checkpoint found in '{self.hf_repo_id}'. "
            f"Tried: {candidates}. Last error: {last_err}"
        )

    def load_openbeats_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        strict: bool = False,
        map_location: str = "cpu",    # always load to CPU first, then .to(device)
    ) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        """
        Load weights from a local .pt file or auto-download from HF Hub.

        Always loads to CPU first regardless of map_location arg to avoid
        CUDA deserialisation errors on CPU-only machines.
        """
        path = checkpoint_path or self.local_checkpoint
        if path is None:
            path = self._download_from_hf()

        # Always load to CPU to avoid CUDA deserialisation crash.
        # weights_only=False needed for ESPnet training checkpoints.
        raw = torch.load(path, map_location="cpu", weights_only=False)

        # If checkpoint has a saved cfg, re-init encoder with exact training config
        if isinstance(raw, dict) and "cfg" in raw:
            saved_cfg = raw["cfg"]
            if isinstance(saved_cfg, dict):
                self._enc = BeatsEncoder(input_size=128, beats_config=saved_cfg)
                print("  ✔  Re-initialised encoder from checkpoint['cfg']")
            elif hasattr(saved_cfg, "__dict__"):
                self._enc = BeatsEncoder(input_size=128, beats_config=vars(saved_cfg))
                print("  ✔  Re-initialised encoder from checkpoint['cfg'] (BeatsConfig)")

        sd = self._extract_state_dict(raw)
        sd = self._best_state_dict_variant(sd, self._enc.state_dict().keys())

        incompatible = self._enc.load_state_dict(sd, strict=strict)

        missing    = tuple(getattr(incompatible, "missing_keys",    []))
        unexpected = tuple(getattr(incompatible, "unexpected_keys", []))
        return missing, unexpected

    # ─────────────────────────────────────────────────────────────────
    # Training stage control
    # ─────────────────────────────────────────────────────────────────
    def freeze_base_weights(self) -> None:
        """Stage 1: freeze all encoder weights."""
        for p in self._enc.parameters():
            p.requires_grad = False

    def unfreeze_all(self) -> None:
        """Stage 3+: unfreeze everything."""
        for p in self.parameters():
            p.requires_grad = True

    def enable_training_dropouts(
        self,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        dropout_input: float = 0.0,
    ) -> None:
        """Re-enable dropout after loading checkpoint weights."""
        if hasattr(self._enc, "dropout_input") and hasattr(self._enc.dropout_input, "p"):
            self._enc.dropout_input.p = dropout_input

        enc = getattr(self._enc, "encoder", None)
        if enc is None:
            return

        dm = getattr(enc, "dropout_module", None)
        if dm is not None and hasattr(dm, "p"):
            dm.p = dropout

        for layer in getattr(enc, "layers", []):
            if hasattr(getattr(layer, "dropout_module", None), "p"):
                layer.dropout_module.p = dropout
            if hasattr(getattr(getattr(layer, "self_attn", None), "dropout_module", None), "p"):
                layer.self_attn.dropout_module.p = attention_dropout
            if hasattr(getattr(layer, "activation_dropout_module", None), "p"):
                layer.activation_dropout_module.p = activation_dropout

    def enable_lora(self, r: int = 16, alpha: int = 32, dropout: float = 0.05) -> None:
        """Stage 2: inject LoRA adapters. Requires: pip install peft"""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError("Run: pip install peft") from e

        self._enc.encoder = get_peft_model(
            self._enc.encoder,
            LoraConfig(
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                task_type="FEATURE_EXTRACTION",
            ),
        )

    # ─────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _downsample_padding_mask(
        padding_mask: Optional[torch.Tensor],
        target_len: int,
        patch_size: int,
    ) -> Optional[torch.Tensor]:
        if padding_mask is None:
            return None
        if padding_mask.dtype != torch.bool:
            padding_mask = padding_mask.bool()
        bsz, src_len = padding_mask.shape
        needed = target_len * patch_size
        if src_len < needed:
            padding_mask = F.pad(padding_mask, (0, needed - src_len), value=True)
        elif src_len > needed:
            padding_mask = padding_mask[:, :needed]
        return padding_mask.view(bsz, target_len, patch_size).all(dim=-1)

    @staticmethod
    def _resolve_patch_size(module: nn.Module, default: int = 16) -> int:
        ks = getattr(module, "kernel_size", None)
        if isinstance(ks, tuple) and len(ks) >= 1:
            return int(ks[0])
        if isinstance(ks, int):
            return int(ks)
        return default

    # ─────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────
    def forward(
        self,
        mel: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> OpenBEATsEncoderOutput:
        """
        Args:
            mel           [B, 128, T]  — mel chunks from Block 1
            padding_mask  [B, T]       — True where frame is padded (optional)

        Returns:
            OpenBEATsEncoderOutput
                .h_ob         [B, N_patches, 1024]
                .num_patches  int
                .num_chunks   int
        """
        if mel.dim() != 3:
            raise ValueError(f"Expected mel [B, 128, T], got {tuple(mel.shape)}")
        if mel.size(1) != 128:
            raise ValueError(f"Expected 128 mel bins on dim=1, got {tuple(mel.shape)}")

        # Block 1 → [B, 128, T]
        # Conv2d patch_embedding expects [B, 1, T, 128] (time-major, channel-last)
        x = mel.transpose(1, 2).unsqueeze(1)           # [B, 1, T, 128]
        x = self._enc.patch_embedding(x)               # [B, embed_dim, T', F']
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B, N_patches, 512]

        if hasattr(self._enc, "layer_norm") and self._enc.layer_norm is not None:
            x = self._enc.layer_norm(x)

        if hasattr(self._enc, "post_extract_proj") and self._enc.post_extract_proj is not None:
            x = self._enc.post_extract_proj(x)         # [B, N_patches, 1024]

        if hasattr(self._enc, "dropout_input") and self._enc.dropout_input is not None:
            x = self._enc.dropout_input(x)

        patch_size = self._resolve_patch_size(self._enc.patch_embedding, default=16)
        patch_padding_mask = self._downsample_padding_mask(
            padding_mask=padding_mask,
            target_len=x.size(1),
            patch_size=patch_size,
        )

        enc_out = self._enc.encoder(x, padding_mask=patch_padding_mask)
        h_ob = enc_out[0] if isinstance(enc_out, tuple) else enc_out

        return OpenBEATsEncoderOutput(
            h_ob=h_ob,
            num_patches=int(h_ob.size(1)),
            num_chunks=int(h_ob.size(0)),
        )

    # ─────────────────────────────────────────────────────────────────
    # Introspection
    # ─────────────────────────────────────────────────────────────────
    @property
    def hidden_dim(self) -> int:
        return int(getattr(self.cfg, "encoder_embed_dim", 1024))

    @property
    def num_layers(self) -> int:
        return int(getattr(self.cfg, "encoder_layers", 24))

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"layers={self.num_layers}, "
            f"trainable_params={self.trainable_params}, "
            f"total_params={self.total_params}"
        )


if __name__ == "__main__":
    enc = Block2bOpenBEATsEncoder()
    x = torch.randn(2, 128, 3000)
    out = enc(x)
    print(enc)
    print("h_ob:", out.h_ob.shape)
    print("num_patches:", out.num_patches)
    print("num_chunks:", out.num_chunks)