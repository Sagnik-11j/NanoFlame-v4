# tests/test_block1.py
# ─────────────────────────────────────────────────────────────────────────────
# Tests for Block 1 — Audio Frontend
# Run with:  pytest tests/test_block1.py -v
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

from blocks.block1_audio_frontend import AudioFrontend, AudioFrontendOutput
from config import N_MELS, CHUNK_FRAMES, SAMPLE_RATE, MAX_CHUNKS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_wav(duration_sec: float, sr: int = 16000, channels: int = 1) -> str:
    n_samples = int(sr * duration_sec)
    rng       = np.random.default_rng(seed=42)
    mono_f32  = rng.standard_normal(n_samples).astype(np.float32) * 0.3
    mono_i16  = (mono_f32 * 32767).clip(-32768, 32767).astype(np.int16)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        if channels == 2:
            stereo = np.stack([mono_i16, mono_i16], axis=1).flatten()
            wf.writeframes(stereo.tobytes())
        else:
            wf.writeframes(mono_i16.tobytes())
    return tmp.name


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def frontend():
    return AudioFrontend()


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
def wav_10s():
    path = make_wav(10.0)
    yield path
    os.unlink(path)


@pytest.fixture
def wav_stereo():
    path = make_wav(30.0, channels=2)
    yield path
    os.unlink(path)


@pytest.fixture
def wav_44khz():
    path = make_wav(30.0, sr=44100)
    yield path
    os.unlink(path)


@pytest.fixture
def wav_long():
    path = make_wav(360.0)   # 6 minutes → should be capped at 5 min
    yield path
    os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputType:
    def test_returns_dataclass(self, frontend, wav_30s):
        assert isinstance(frontend.process(wav_30s), AudioFrontendOutput)

    def test_chunks_is_list(self, frontend, wav_30s):
        assert isinstance(frontend.process(wav_30s).chunks, list)

    def test_each_chunk_is_tensor(self, frontend, wav_30s):
        out = frontend.process(wav_30s)
        assert all(isinstance(c, torch.Tensor) for c in out.chunks)

    def test_dtype_float32(self, frontend, wav_30s):
        out = frontend.process(wav_30s)
        assert all(c.dtype == torch.float32 for c in out.chunks)


class TestChunkShape:
    def test_chunk_shape_30s(self, frontend, wav_30s):
        out = frontend.process(wav_30s)
        for chunk in out.chunks:
            assert chunk.shape == (N_MELS, CHUNK_FRAMES), (
                f"Expected ({N_MELS}, {CHUNK_FRAMES}), got {chunk.shape}"
            )

    def test_chunk_shape_10s(self, frontend, wav_10s):
        out = frontend.process(wav_10s)
        for chunk in out.chunks:
            assert chunk.shape == (N_MELS, CHUNK_FRAMES)

    def test_n_mels_is_128(self, frontend, wav_30s):
        out = frontend.process(wav_30s)
        assert out.chunks[0].shape[0] == 128

    def test_chunk_frames_is_3000(self, frontend, wav_30s):
        out = frontend.process(wav_30s)
        assert out.chunks[0].shape[1] == 3000

    def test_stack_shape_single_chunk(self, frontend, wav_30s):
        out = frontend.process(wav_30s)
        stacked = out.stack()
        assert stacked.shape == (1, N_MELS, CHUNK_FRAMES)

    def test_stack_shape_multi_chunk(self, frontend, wav_90s):
        out = frontend.process(wav_90s)
        stacked = out.stack()
        assert stacked.shape[1] == N_MELS
        assert stacked.shape[2] == CHUNK_FRAMES


class TestChunkCount:
    def test_30s_gives_1_chunk(self, frontend, wav_30s):
        assert frontend.process(wav_30s).num_chunks == 1

    def test_90s_gives_3_chunks(self, frontend, wav_90s):
        assert frontend.process(wav_90s).num_chunks == 3

    def test_10s_gives_1_chunk(self, frontend, wav_10s):
        assert frontend.process(wav_10s).num_chunks == 1

    def test_long_audio_capped_at_max_chunks(self, frontend, wav_long):
        out = frontend.process(wav_long)
        assert out.num_chunks == MAX_CHUNKS

    def test_long_audio_sets_capped_flag(self, frontend, wav_long):
        assert frontend.process(wav_long).capped is True

    def test_short_audio_capped_flag_false(self, frontend, wav_30s):
        assert frontend.process(wav_30s).capped is False


class TestNumerics:
    def test_no_nan(self, frontend, wav_30s):
        out = frontend.process(wav_30s)
        assert not any(torch.isnan(c).any() for c in out.chunks)

    def test_no_inf(self, frontend, wav_30s):
        out = frontend.process(wav_30s)
        assert not any(torch.isinf(c).any() for c in out.chunks)

    def test_values_not_all_zero(self, frontend, wav_30s):
        out = frontend.process(wav_30s)
        assert out.chunks[0].abs().sum().item() > 0

    def test_normalised_mean_near_zero(self, frontend, wav_30s):
        out = frontend.process(wav_30s)
        mean = out.chunks[0].mean().item()
        assert abs(mean) < 0.1, f"Mean {mean:.4f} too far from 0"

    def test_normalised_std_near_one(self, frontend, wav_30s):
        out = frontend.process(wav_30s)
        std = out.chunks[0].std().item()
        assert 0.8 < std < 1.2, f"Std {std:.4f} too far from 1"


class TestResampling:
    def test_44khz_input_processed(self, frontend, wav_44khz):
        out = frontend.process(wav_44khz)
        assert out.num_chunks >= 1

    def test_44khz_chunk_shape_correct(self, frontend, wav_44khz):
        out = frontend.process(wav_44khz)
        assert out.chunks[0].shape == (N_MELS, CHUNK_FRAMES)

    def test_stereo_input_processed(self, frontend, wav_stereo):
        out = frontend.process(wav_stereo)
        assert out.num_chunks >= 1

    def test_stereo_chunk_shape_correct(self, frontend, wav_stereo):
        out = frontend.process(wav_stereo)
        assert out.chunks[0].shape == (N_MELS, CHUNK_FRAMES)


class TestMetadata:
    def test_num_chunks_matches_len(self, frontend, wav_30s):
        out = frontend.process(wav_30s)
        assert out.num_chunks == len(out.chunks)

    def test_duration_sec_approx(self, frontend, wav_30s):
        out = frontend.process(wav_30s)
        assert 29.0 <= out.duration_sec <= 31.0

    def test_file_path_stored(self, frontend, wav_30s):
        out = frontend.process(wav_30s)
        assert wav_30s in out.file_path

    def test_original_sr_stored(self, frontend, wav_30s):
        out = frontend.process(wav_30s)
        assert out.sample_rate_original == SAMPLE_RATE


class TestEdgeCases:
    def test_missing_file_raises(self, frontend):
        with pytest.raises(FileNotFoundError):
            frontend.process("/nonexistent/file.wav")

    def test_to_device_cpu(self, frontend, wav_30s):
        out = frontend.process(wav_30s).to("cpu")
        assert all(c.device.type == "cpu" for c in out.chunks)

    def test_process_batch_returns_list(self, frontend, wav_30s, wav_10s):
        results = frontend.process_batch([wav_30s, wav_10s])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_process_batch_shapes_correct(self, frontend, wav_30s, wav_10s):
        results = frontend.process_batch([wav_30s, wav_10s])
        for out in results:
            for chunk in out.chunks:
                assert chunk.shape == (N_MELS, CHUNK_FRAMES)