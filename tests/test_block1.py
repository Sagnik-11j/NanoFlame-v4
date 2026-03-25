# nanoflame/tests/test_block1.py
# ─────────────────────────────────────────────────────────────────────────────
# Tests for Block 1 — Audio Frontend.
# Run with:  pytest tests/test_block1.py -v
# ─────────────────────────────────────────────────────────────────────────────

import math
import pytest
import torch
import torchaudio
import tempfile
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from blocks.block1_audio_frontend import AudioFrontend, AudioFrontendOutput
from config import N_MELS, CHUNK_FRAMES, SAMPLE_RATE, MAX_CHUNKS


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — generate synthetic .wav files in a temp directory
# ─────────────────────────────────────────────────────────────────────────────

def make_wav(duration_sec: float, sr: int = 16000, channels: int = 1) -> str:
    """Create a temporary sine-wave .wav file. Returns its path."""
    n_samples  = int(sr * duration_sec)
    t          = torch.linspace(0, duration_sec, n_samples)
    waveform   = torch.sin(2 * math.pi * 440 * t).unsqueeze(0)   # [1, N]
    if channels == 2:
        waveform = waveform.repeat(2, 1)                          # [2, N]

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    torchaudio.save(tmp.name, waveform, sr)
    return tmp.name


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def frontend():
    return AudioFrontend()


@pytest.fixture
def wav_10s():
    path = make_wav(10.0)
    yield path
    os.unlink(path)

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
def wav_400s():
    path = make_wav(400.0)
    yield path
    os.unlink(path)

@pytest.fixture
def wav_stereo():
    path = make_wav(15.0, channels=2)
    yield path
    os.unlink(path)

@pytest.fixture
def wav_44khz():
    path = make_wav(10.0, sr=44100)
    yield path
    os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputType:
    def test_returns_dataclass(self, frontend, wav_10s):
        out = frontend.process(wav_10s)
        assert isinstance(out, AudioFrontendOutput)

    def test_chunks_is_list(self, frontend, wav_10s):
        out = frontend.process(wav_10s)
        assert isinstance(out.chunks, list)


class TestChunkShape:
    def test_each_chunk_shape(self, frontend, wav_10s):
        out = frontend.process(wav_10s)
        for chunk in out.chunks:
            assert chunk.shape == (N_MELS, CHUNK_FRAMES), (
                f"Expected ({N_MELS}, {CHUNK_FRAMES}), got {chunk.shape}"
            )

    def test_chunk_dtype(self, frontend, wav_10s):
        out = frontend.process(wav_10s)
        for chunk in out.chunks:
            assert chunk.dtype == torch.float32


class TestChunkCount:
    def test_10s_gives_1_chunk(self, frontend, wav_10s):
        out = frontend.process(wav_10s)
        assert out.num_chunks == 1

    def test_30s_gives_1_chunk(self, frontend, wav_30s):
        out = frontend.process(wav_30s)
        assert out.num_chunks == 1

    def test_90s_gives_3_chunks(self, frontend, wav_90s):
        out = frontend.process(wav_90s)
        assert out.num_chunks == 3

    def test_capping_at_max_chunks(self, frontend, wav_400s):
        out = frontend.process(wav_400s)
        assert out.num_chunks == MAX_CHUNKS
        assert out.capped is True

    def test_short_audio_not_capped(self, frontend, wav_10s):
        out = frontend.process(wav_10s)
        assert out.capped is False


class TestMonoConversion:
    def test_stereo_input_ok(self, frontend, wav_stereo):
        out = frontend.process(wav_stereo)
        assert out.num_chunks >= 1
        for chunk in out.chunks:
            assert chunk.shape == (N_MELS, CHUNK_FRAMES)


class TestResampling:
    def test_44khz_input_resampled(self, frontend, wav_44khz):
        out = frontend.process(wav_44khz)
        assert out.sample_rate_original == 44100
        for chunk in out.chunks:
            assert chunk.shape == (N_MELS, CHUNK_FRAMES)


class TestNumerics:
    def test_no_nan(self, frontend, wav_10s):
        out = frontend.process(wav_10s)
        for chunk in out.chunks:
            assert not torch.isnan(chunk).any(), "NaN detected in chunk"

    def test_no_inf(self, frontend, wav_10s):
        out = frontend.process(wav_10s)
        for chunk in out.chunks:
            assert not torch.isinf(chunk).any(), "Inf detected in chunk"

    def test_normalised_roughly_unit(self, frontend, wav_30s):
        # After per-spectrogram normalisation, std should be ~1
        out = frontend.process(wav_30s)
        for chunk in out.chunks:
            std = chunk.std().item()
            assert 0.5 < std < 2.0, f"Unexpected std after normalisation: {std}"


class TestStackHelper:
    def test_stack_shape(self, frontend, wav_90s):
        out = frontend.process(wav_90s)
        stacked = out.stack()
        # [N, 128, 1500]
        assert stacked.shape == (out.num_chunks, N_MELS, CHUNK_FRAMES)


class TestErrors:
    def test_missing_file_raises(self, frontend):
        with pytest.raises(FileNotFoundError):
            frontend.process("/nonexistent/path/audio.wav")
