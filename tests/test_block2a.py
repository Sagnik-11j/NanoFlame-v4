# tests/test_block2a.py
# ─────────────────────────────────────────────────────────────────────────────
# Tests for Block 2a — Whisper-Medium Encoder
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
warnings.filterwarnings("ignore", category=UserWarning)

from blocks.block1_audio_frontend    import AudioFrontend
from blocks.block2a_whisper_encoder  import Block2aWhisperEncoder, WhisperEncoderOutput
from config import N_MELS, CHUNK_FRAMES


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_wav(duration_sec: float, sr: int = 16000) -> str:
    n_samples = int(sr * duration_sec)
    rng       = np.random.default_rng(seed=42)
    mono_f32  = rng.standard_normal(n_samples).astype(np.float32) * 0.3
    mono_i16  = (mono_f32 * 32767).clip(-32768, 32767).astype(np.int16)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
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
    rng = np.random.default_rng(seed=0)
    return torch.from_numpy(
        rng.standard_normal((N_MELS, CHUNK_FRAMES)).astype(np.float32)
    )


@pytest.fixture
def batch_chunks():
    rng = np.random.default_rng(seed=1)
    return torch.from_numpy(
        rng.standard_normal((3, N_MELS, CHUNK_FRAMES)).astype(np.float32)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputType:
    def test_returns_dataclass(self, encoder, single_chunk):
        assert isinstance(encoder(single_chunk), WhisperEncoderOutput)

    def test_h_w_is_tensor(self, encoder, single_chunk):
        assert isinstance(encoder(single_chunk).h_w, torch.Tensor)

    def test_dtype_float32(self, encoder, single_chunk):
        assert encoder(single_chunk).h_w.dtype == torch.float32


class TestOutputShape:
    def test_single_chunk_shape(self, encoder, single_chunk):
        out = encoder(single_chunk)
        assert out.h_w.shape == (750, 1024), (
            f"Expected (750, 1024), got {out.h_w.shape}"
        )

    def test_batch_shape(self, encoder, batch_chunks):
        out = encoder(batch_chunks)
        assert out.h_w.shape == (3, 750, 1024), (
            f"Expected (3, 750, 1024), got {out.h_w.shape}"
        )

    def test_hidden_dim_is_1024(self, encoder, single_chunk):
        assert encoder(single_chunk).h_w.shape[-1] == 1024

    def test_time_steps_is_750(self, encoder, single_chunk):
        assert encoder(single_chunk).h_w.shape[-2] == 750

    def test_num_tokens_attribute(self, encoder, single_chunk):
        assert encoder(single_chunk).num_tokens == 750

    def test_num_chunks_single(self, encoder, single_chunk):
        assert encoder(single_chunk).num_chunks == 1

    def test_num_chunks_batch(self, encoder, batch_chunks):
        assert encoder(batch_chunks).num_chunks == 3


class TestNumerics:
    def test_no_nan(self, encoder, single_chunk):
        assert not torch.isnan(encoder(single_chunk).h_w).any()

    def test_no_inf(self, encoder, single_chunk):
        assert not torch.isinf(encoder(single_chunk).h_w).any()

    def test_output_not_all_zeros(self, encoder, single_chunk):
        assert encoder(single_chunk).h_w.abs().sum().item() > 0

    def test_different_inputs_give_different_outputs(self, encoder, single_chunk, batch_chunks):
        out1 = encoder(single_chunk)
        out2 = encoder(batch_chunks[0])
        assert not torch.allclose(out1.h_w, out2.h_w)


class TestArchitecture:
    def test_hidden_dim_constant(self, encoder):
        assert encoder.HIDDEN_DIM == 1024

    def test_out_tokens_constant(self, encoder):
        assert encoder.OUT_TOKENS == 750

    def test_conv1_in_channels(self, encoder):
        conv1 = encoder.encoder.conv1
        assert conv1.in_channels == 128, (
            f"conv1 should accept 128 mel bins, got {conv1.in_channels}"
        )

    def test_conv1_out_channels(self, encoder):
        conv1 = encoder.encoder.conv1
        assert conv1.out_channels == 1024, (
            f"conv1 out_channels should be 1024, got {conv1.out_channels}"
        )

    def test_temporal_pool_exists(self, encoder):
        assert hasattr(encoder, "temporal_pool")

    def test_temporal_pool_kernel(self, encoder):
        assert encoder.temporal_pool.kernel_size == (2,), (
            f"Expected kernel_size (2,), got {encoder.temporal_pool.kernel_size}"
        )

    def test_temporal_pool_stride(self, encoder):
        assert encoder.temporal_pool.stride == (2,), (
            f"Expected stride (2,), got {encoder.temporal_pool.stride}"
        )

    def test_model_name_is_medium(self):
        import inspect
        sig = inspect.signature(Block2aWhisperEncoder.__init__)
        default = sig.parameters["model_name"].default
        assert "medium" in default, (
            f"Default model_name should be whisper-medium, got '{default}'"
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
            f"Expected 0 trainable params after freeze, got {len(trainable)}: {trainable[:3]}"
        )

    def test_enable_lora_adds_trainable_params(self):
        enc = Block2aWhisperEncoder()
        enc.freeze_base_weights()
        enc.enable_lora()
        trainable = [n for n, p in enc.named_parameters() if p.requires_grad]
        assert len(trainable) > 0

    def test_only_lora_params_trainable_after_freeze_and_enable(self):
        enc = Block2aWhisperEncoder()
        enc.freeze_base_weights()
        enc.enable_lora()
        non_lora = [
            n for n, p in enc.named_parameters()
            if p.requires_grad and "lora_" not in n
        ]
        assert len(non_lora) == 0, f"Non-LoRA params are trainable: {non_lora[:3]}"

    def test_lora_output_shape_unchanged(self):
        enc = Block2aWhisperEncoder()
        enc.enable_lora()
        rng   = np.random.default_rng(seed=0)
        chunk = torch.from_numpy(
            rng.standard_normal((N_MELS, CHUNK_FRAMES)).astype(np.float32)
        )
        assert enc(chunk).h_w.shape == (750, 1024)

    def test_enable_lora_is_idempotent(self):
        enc = Block2aWhisperEncoder()
        enc.enable_lora()
        enc.enable_lora()
        assert enc._lora_enabled is True


class TestIntegration:
    def test_block1_to_block2a_30s(self, frontend, encoder, wav_30s):
        b1  = frontend.process(wav_30s)
        b2a = encoder(b1.stack())
        assert b2a.h_w.shape == (1, 750, 1024)

    def test_block1_to_block2a_90s(self, frontend, encoder, wav_90s):
        b1  = frontend.process(wav_90s)
        b2a = encoder(b1.stack())
        assert b2a.h_w.shape == (3, 750, 1024)

    def test_pipeline_no_nan_end_to_end(self, frontend, encoder, wav_30s):
        b1  = frontend.process(wav_30s)
        b2a = encoder(b1.stack())
        assert not torch.isnan(b2a.h_w).any()

    def test_pipeline_no_inf_end_to_end(self, frontend, encoder, wav_30s):
        b1  = frontend.process(wav_30s)
        b2a = encoder(b1.stack())
        assert not torch.isinf(b2a.h_w).any()

    def test_block2a_and_block2b_parallel_shapes(self, frontend, wav_30s):
        from blocks.block2b_openbeats_encoder import Block2bOpenBEATsEncoder
        enc_a  = Block2aWhisperEncoder()
        enc_b  = Block2bOpenBEATsEncoder()
        chunks = frontend.process(wav_30s).stack()
        out_a  = enc_a(chunks)
        out_b  = enc_b(chunks)
        assert out_a.h_w.shape  == (1, 750,  1024)
        assert out_b.h_ob.shape == (1, 1496, 1024)