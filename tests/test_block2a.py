# tests/test_block2a.py
# ─────────────────────────────────────────────────────────────────────────────
# Tests for Block 2a — Whisper-Small Encoder
# Run with:  pytest tests/test_block2a.py -v
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import wave
import warnings
import tempfile

import numpy as np
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

from blocks.block1_audio_frontend  import AudioFrontend
from blocks.block2a_whisper_encoder import Block2aWhisperEncoder, WhisperEncoderOutput
from config import N_MELS, CHUNK_FRAMES


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_wav(duration_sec: float, sr: int = 16000) -> str:
    n_samples = int(sr * duration_sec)
    rng       = np.random.default_rng(seed=42)
    mono_f32  = rng.standard_normal(n_samples).astype(np.float32) * 0.3
    mono_i16  = (mono_f32 * 32767).clip(-32768, 32767).astype(np.int16)
    tmp       = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(mono_i16.tobytes())
    return tmp.name


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def frontend():
    return AudioFrontend()


@pytest.fixture(scope="module")
def encoder():
    """Single encoder instance shared across the whole module (model load is slow)."""
    return Block2aWhisperEncoder()


@pytest.fixture
def wav_30s():
    path = make_wav(30.0)
    yield path
    os.unlink(path)


@pytest.fixture
def wav_90s():
    path = make_wav(90.0)
    yield path
    os.unlink(path)


@pytest.fixture
def single_chunk():
    """Synthetic mel chunk  [128, 3000]  float32."""
    rng = np.random.default_rng(seed=0)
    return torch.from_numpy(
        rng.standard_normal((N_MELS, CHUNK_FRAMES)).astype(np.float32)
    )


@pytest.fixture
def batch_chunks():
    """Three stacked mel chunks  [3, 128, 3000]  float32."""
    rng = np.random.default_rng(seed=1)
    return torch.from_numpy(
        rng.standard_normal((3, N_MELS, CHUNK_FRAMES)).astype(np.float32)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputType:
    def test_returns_dataclass(self, encoder, single_chunk):
        out = encoder(single_chunk)
        assert isinstance(out, WhisperEncoderOutput)

    def test_h_w_is_tensor(self, encoder, single_chunk):
        out = encoder(single_chunk)
        assert isinstance(out.h_w, torch.Tensor)

    def test_dtype_float32(self, encoder, single_chunk):
        out = encoder(single_chunk)
        assert out.h_w.dtype == torch.float32


class TestOutputShape:
    def test_single_chunk_shape(self, encoder, single_chunk):
        out = encoder(single_chunk)
        assert out.h_w.shape == (750, 768), (
            f"Expected (750, 768), got {out.h_w.shape}"
        )

    def test_batch_shape(self, encoder, batch_chunks):
        out = encoder(batch_chunks)
        assert out.h_w.shape == (3, 750, 768), (
            f"Expected (3, 750, 768), got {out.h_w.shape}"
        )

    def test_hidden_dim_is_768(self, encoder, single_chunk):
        out = encoder(single_chunk)
        assert out.h_w.shape[-1] == 768

    def test_num_tokens_attribute(self, encoder, single_chunk):
        out = encoder(single_chunk)
        assert out.num_tokens == 750

    def test_num_chunks_single(self, encoder, single_chunk):
        out = encoder(single_chunk)
        assert out.num_chunks == 1

    def test_num_chunks_batch(self, encoder, batch_chunks):
        out = encoder(batch_chunks)
        assert out.num_chunks == 3


class TestNumerics:
    def test_no_nan(self, encoder, single_chunk):
        out = encoder(single_chunk)
        assert not torch.isnan(out.h_w).any(), "NaN in h_w"

    def test_no_inf(self, encoder, single_chunk):
        out = encoder(single_chunk)
        assert not torch.isinf(out.h_w).any(), "Inf in h_w"

    def test_output_not_all_zeros(self, encoder, single_chunk):
        out = encoder(single_chunk)
        assert out.h_w.abs().sum().item() > 0, "h_w is all zeros"

    def test_different_inputs_give_different_outputs(self, encoder, single_chunk, batch_chunks):
        out1 = encoder(single_chunk)
        out2 = encoder(batch_chunks[0])           # different seed
        assert not torch.allclose(out1.h_w, out2.h_w), (
            "Different inputs produced identical outputs"
        )


class TestArchitecture:
    def test_conv1_in_channels_is_128(self, encoder):
        conv1 = encoder.encoder.conv1
        assert conv1.in_channels == 128, (
            f"conv1 expects {conv1.in_channels} input channels — should be 128"
        )

    def test_conv1_out_channels_is_768(self, encoder):
        conv1 = encoder.encoder.conv1
        assert conv1.out_channels == 768

    def test_temporal_pooling_halves_to_750(self, encoder, single_chunk):
        out = encoder(single_chunk)
        assert out.h_w.shape[0] == 750, (
            f"Temporal pooling should give 750 tokens, got {out.h_w.shape[0]}"
        )

    def test_decoder_is_absent(self, encoder):
        assert not hasattr(encoder, "decoder"), (
            "Decoder should have been discarded"
        )


class TestLoRA:
    def test_lora_disabled_by_default(self):
        enc = Block2aWhisperEncoder()
        assert enc._lora_enabled is False

    def test_freeze_leaves_zero_trainable_params(self):
        enc = Block2aWhisperEncoder()
        enc.freeze_base_weights()
        trainable = [n for n, p in enc.named_parameters() if p.requires_grad]
        assert len(trainable) == 0, (
            f"Expected 0 trainable params after freeze, got {len(trainable)}"
        )

    def test_enable_lora_adds_trainable_params(self):
        enc = Block2aWhisperEncoder()
        enc.freeze_base_weights()
        enc.enable_lora()
        trainable = [n for n, p in enc.named_parameters() if p.requires_grad]
        assert len(trainable) > 0, "No trainable params after enable_lora()"

    def test_only_lora_params_are_trainable(self):
        enc = Block2aWhisperEncoder()
        enc.freeze_base_weights()
        enc.enable_lora()
        non_lora_trainable = [
            n for n, p in enc.named_parameters()
            if p.requires_grad and "lora_" not in n
        ]
        assert len(non_lora_trainable) == 0, (
            f"Non-LoRA params are trainable: {non_lora_trainable[:3]}"
        )

    def test_lora_output_shape_unchanged(self):
        enc   = Block2aWhisperEncoder()
        enc.enable_lora()
        rng   = np.random.default_rng(seed=0)
        chunk = torch.from_numpy(
            rng.standard_normal((N_MELS, CHUNK_FRAMES)).astype(np.float32)
        )
        out = enc(chunk)
        assert out.h_w.shape == (750, 768)

    def test_enable_lora_is_idempotent(self):
        enc = Block2aWhisperEncoder()
        enc.enable_lora()
        enc.enable_lora()                          # second call should be a no-op
        assert enc._lora_enabled is True


class TestIntegration:
    def test_block1_to_block2a_30s(self, frontend, encoder, wav_30s):
        b1  = frontend.process(wav_30s)
        b2a = encoder(b1.stack())                  # [1, 128, 3000] → [1, 750, 768]
        assert b2a.h_w.shape == (1, 750, 768)

    def test_block1_to_block2a_90s(self, frontend, encoder, wav_90s):
        b1  = frontend.process(wav_90s)
        b2a = encoder(b1.stack())                  # [3, 128, 3000] → [3, 750, 768]
        assert b2a.h_w.shape == (3, 750, 768)

    def test_pipeline_no_nan_end_to_end(self, frontend, encoder, wav_30s):
        b1  = frontend.process(wav_30s)
        b2a = encoder(b1.stack())
        assert not torch.isnan(b2a.h_w).any(), "NaN in end-to-end pipeline"
