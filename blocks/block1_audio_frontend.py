# nanoflame/blocks/block1_audio_frontend.py
# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 1 — AUDIO FRONTEND
#
# Converts a raw audio file into a list of normalised log-mel spectrogram
# chunks, each of shape [N_MELS, CHUNK_FRAMES] = [128, 1500].
#
# Pipeline:
#   1a  Load      : read file from disk → float32 waveform tensor
#   1b  Resample  : convert to 16 kHz mono
#   1c  Log-Mel   : STFT → mel filterbank → log compression
#   1d  Chunk     : split into non-overlapping 30s windows (max 10)
#
# This block is STATELESS — no learned parameters, no GPU required.
# It runs on CPU and can be used inside a DataLoader worker (num_workers > 0).
#
# Usage:
#   frontend = AudioFrontend()
#   out = frontend.process("audio.wav")
#   chunks = out.chunks          # List[Tensor[128, 1500]]
#   print(out.num_chunks)        # int
#   print(out.duration_sec)      # float
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

# import global constants — never hardcode values inside this file
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    CHUNK_FRAMES,
    MAX_CHUNKS,
    CHUNK_DURATION_SEC,
    FRAMES_PER_SEC,
)


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AudioFrontendOutput:
    """
    Holds the result of processing one audio file through Block 1.

    Attributes
    ----------
    chunks : List[Tensor]
        List of N mel-spectrogram chunks.
        Each tensor has shape [N_MELS, CHUNK_FRAMES] = [128, 1500].
        dtype = float32, values are log-mel power in range roughly [-10, 5].

    num_chunks : int
        Number of 30-second chunks produced. Equal to len(chunks).

    duration_sec : float
        True duration of the original audio in seconds (before any capping).

    capped : bool
        True if the audio was longer than MAX_CHUNKS × 30s (5 minutes)
        and was silently truncated.

    sample_rate_original : int
        The sample rate of the source file before resampling.

    file_path : str
        Absolute path to the source audio file.
    """

    chunks             : List[torch.Tensor]
    num_chunks         : int
    duration_sec       : float
    capped             : bool
    sample_rate_original : int
    file_path          : str = ""

    def to(self, device: Union[str, torch.device]) -> "AudioFrontendOutput":
        """Move all chunk tensors to the given device."""
        self.chunks = [c.to(device) for c in self.chunks]
        return self

    def stack(self) -> torch.Tensor:
        """
        Stack all chunks into a single tensor of shape [N, 128, 1500].
        Useful when you want to batch-encode all chunks through the encoders.
        """
        return torch.stack(self.chunks, dim=0)  # [N, 128, 1500]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class AudioFrontend:
    """
    Block 1 — Audio Frontend.

    Stateless preprocessing pipeline. No nn.Module, no parameters.
    Thread-safe; safe to instantiate once and share across workers.

    Parameters
    ----------
    sample_rate : int
        Target sample rate. Default 16000 (required by Whisper).
    n_fft : int
        FFT window size in samples. Default 400 (25ms @ 16kHz).
    hop_length : int
        STFT hop size in samples. Default 160 (10ms @ 16kHz).
    n_mels : int
        Number of mel filterbank bands. Default 128.
    chunk_frames : int
        Time-frames per chunk. Default 1500 (= 30s × 100 fps).
    max_chunks : int
        Maximum number of chunks before truncation. Default 10 (5 min).
    """

    def __init__(
        self,
        sample_rate  : int = SAMPLE_RATE,
        n_fft        : int = N_FFT,
        hop_length   : int = HOP_LENGTH,
        n_mels       : int = N_MELS,
        chunk_frames : int = CHUNK_FRAMES,
        max_chunks   : int = MAX_CHUNKS,
    ) -> None:
        self.sample_rate  = sample_rate
        self.n_fft        = n_fft
        self.hop_length   = hop_length
        self.n_mels       = n_mels
        self.chunk_frames = chunk_frames
        self.max_chunks   = max_chunks

        # Build the mel transform once — reused for every file.
        # Using torchaudio's MelSpectrogram which internally runs STFT
        # followed by the mel filterbank projection.
        self._mel_transform = T.MelSpectrogram(
            sample_rate = self.sample_rate,
            n_fft       = self.n_fft,
            hop_length  = self.hop_length,
            n_mels      = self.n_mels,
            window_fn   = torch.hann_window,   # Hann window prevents spectral leakage
            power       = 2.0,                 # output is power spectrum (not amplitude)
            center      = False,                # pad edges so frame 0 is centred on t=0
            pad_mode    = "reflect",
            norm        = "slaney",            # area-normalise mel filterbank triangles
            mel_scale   = "htk",               # HTK mel scale (same as Whisper)
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ─────────────────────────────────────────────────────────────────────────

    def process(
        self,
        audio_path: Union[str, Path],
        verbose   : bool = False,
    ) -> AudioFrontendOutput:
        """
        Full Block 1 pipeline for one audio file.

        Parameters
        ----------
        audio_path : str or Path
            Path to audio file. Supports .wav .mp3 .flac .ogg .m4a
            (anything torchaudio can decode).
        verbose : bool
            If True, print shape information at each sub-step.

        Returns
        -------
        AudioFrontendOutput
            Dataclass with .chunks (List[Tensor[128,1500]]) and metadata.
        """
        audio_path = Path(audio_path).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # ── 1a  LOAD ──────────────────────────────────────────────────────────
        waveform, original_sr = self._load(audio_path)
        if verbose:
            print(f"[1a] Loaded  : {waveform.shape}  sr={original_sr}")

        # ── 1b  RESAMPLE + MONO ───────────────────────────────────────────────
        waveform = self._resample_mono(waveform, original_sr)
        if verbose:
            print(f"[1b] Resampled: {waveform.shape}  sr={self.sample_rate}")

        duration_sec = waveform.shape[-1] / self.sample_rate

        # ── 1c  LOG-MEL SPECTROGRAM ───────────────────────────────────────────
        mel = self._log_mel(waveform)
        if verbose:
            print(f"[1c] Log-Mel : {mel.shape}  (freqs × frames)")

        # ── 1d  CHUNK ─────────────────────────────────────────────────────────
        chunks, capped = self._chunk(mel)
        if verbose:
            print(f"[1d] Chunks  : {len(chunks)} × {chunks[0].shape}"
                  + ("  [CAPPED]" if capped else ""))

        return AudioFrontendOutput(
            chunks              = chunks,
            num_chunks          = len(chunks),
            duration_sec        = duration_sec,
            capped              = capped,
            sample_rate_original= original_sr,
            file_path           = str(audio_path),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1a — LOAD
    # ─────────────────────────────────────────────────────────────────────────

    def _load(self, path: Path):
        """
        Read audio file from disk into a float32 tensor.

        torchaudio.load returns:
          waveform  : Tensor[channels, num_samples]  dtype=float32
          sample_rate : int

        Values are in [-1.0, +1.0] (normalised by torchaudio).
        Supports: .wav, .mp3, .flac, .ogg, .m4a, .opus, etc.
        """
        try:
            waveform, sr = torchaudio.load(str(path))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load audio file '{path}'.\n"
                f"Ensure torchaudio has the right backend installed "
                f"(pip install soundfile for .wav/.flac, "
                f"ffmpeg for .mp3/.m4a).\nOriginal error: {e}"
            )

        # Sanity check: reject completely silent files early
        if waveform.abs().max() < 1e-9:
            warnings.warn(
                f"Audio file '{path.name}' appears to be completely silent. "
                "Processing will continue but results may be meaningless."
            )

        return waveform, sr   # Tensor[C, N], int

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1b — RESAMPLE + MONO
    # ─────────────────────────────────────────────────────────────────────────

    def _resample_mono(
        self,
        waveform   : torch.Tensor,
        original_sr: int,
    ) -> torch.Tensor:
        """
        1. Convert to mono by averaging all channels.
        2. Resample to self.sample_rate (16000 Hz) using polyphase resampling.

        Polyphase resampling (torchaudio's default) uses an anti-aliasing
        low-pass filter before downsampling — far cleaner than
        nearest-neighbour interpolation.

        Input  : Tensor[C, N]
        Output : Tensor[1, N']   where N' = N × (target_sr / original_sr)
        """

        # ── Mono conversion ──────────────────────────────────────────────────
        # Mean across channel dim → [1, N]
        # torch.mean preserves the batch dim (dim=0) as size 1
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)   # [1, N]
        # If already mono (shape [1, N]), this is a no-op.

        # ── Resampling ───────────────────────────────────────────────────────
        if original_sr != self.sample_rate:
            resampler = T.Resample(
                orig_freq    = original_sr,
                new_freq     = self.sample_rate,
                resampling_method = "sinc_interp_hann",  # polyphase + Hann window
            )
            waveform = resampler(waveform)   # [1, N']

        return waveform   # Tensor[1, num_samples]

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1c — LOG-MEL SPECTROGRAM
    # ─────────────────────────────────────────────────────────────────────────

    def _log_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert mono waveform → log-mel spectrogram.

        Steps performed internally by torchaudio.MelSpectrogram:
          1. Apply Hann window of size n_fft=400 (25ms)
          2. STFT with hop=160 (10ms) → complex spectrogram [201, T]
          3. Magnitude² → power spectrum
          4. Project onto 128 mel triangular filters → [128, T]

        Then we apply log compression here:
          5. Clamp to minimum value (avoids log(0) = -inf)
          6. 10 × log10(mel_power) → dB scale

        Input  : Tensor[1, N]
        Output : Tensor[128, T]   T = ceil(N / hop_length)
        """

        # mel_transform expects [batch, time] or [time]
        # Input is [1, N]; squeeze to [N], transform, result is [n_mels, T]
        mel_power = self._mel_transform(waveform.squeeze(0))   # [128, T]

        # Log compression: clamp then convert to dB
        # 1e-10 is the floor — corresponds to -100 dB, well below any real signal
        mel_power   = mel_power.clamp(min=1e-10)
        log_mel     = 10.0 * torch.log10(mel_power)           # [128, T]

        # Normalise: subtract per-spectrogram mean, divide by std
        # This stabilises training — values end up roughly N(0,1)
        mean = log_mel.mean()
        std  = log_mel.std().clamp(min=1e-6)
        log_mel = (log_mel - mean) / std                       # [128, T]

        return log_mel   # Tensor[128, T_total]

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1d — 30s SLIDING WINDOW CHUNKS
    # ─────────────────────────────────────────────────────────────────────────

    def _chunk(
        self,
        mel: torch.Tensor,
    ) -> tuple[List[torch.Tensor], bool]:
        """
        Split log-mel spectrogram into N non-overlapping 30-second chunks.

        - Each chunk is exactly [N_MELS, CHUNK_FRAMES] = [128, 1500]
        - The last chunk is zero-padded on the right if it is shorter
          than 1500 frames (i.e., audio is not a multiple of 30s)
        - If N > MAX_CHUNKS (10), only the first MAX_CHUNKS are kept
          (audio is truncated to 5 minutes)

        Input  : Tensor[128, T_total]
        Output : List of N tensors, each [128, 1500]
                 bool flag indicating whether truncation occurred
        """

        n_mels, T_total = mel.shape
        capped = False

        # Number of full or partial 30s windows
        n_chunks = math.ceil(T_total / self.chunk_frames)

        # Apply 5-minute cap
        if n_chunks > self.max_chunks:
            warnings.warn(
                f"Audio is longer than {self.max_chunks * CHUNK_DURATION_SEC:.0f}s "
                f"({n_chunks} chunks). Truncating to first {self.max_chunks} chunks."
            )
            n_chunks = self.max_chunks
            capped   = True

        chunks: List[torch.Tensor] = []

        for i in range(n_chunks):
            start = i * self.chunk_frames
            end   = start + self.chunk_frames

            if end <= T_total:
                # Full chunk — no padding needed
                chunk = mel[:, start:end]                          # [128, 1500]
            else:
                # Last partial chunk — zero-pad on the right
                partial      = mel[:, start:]                      # [128, T_partial]
                pad_frames   = self.chunk_frames - partial.shape[1]
                chunk        = F.pad(partial, (0, pad_frames))     # [128, 1500]
                # (0, pad_frames) pads the last dimension on the right only

            assert chunk.shape == (n_mels, self.chunk_frames), (
                f"Chunk {i} has unexpected shape {chunk.shape}. "
                f"Expected ({n_mels}, {self.chunk_frames})."
            )

            chunks.append(chunk)

        return chunks, capped

    # ─────────────────────────────────────────────────────────────────────────
    # CONVENIENCE: process multiple files at once
    # ─────────────────────────────────────────────────────────────────────────

    def process_batch(
        self,
        audio_paths: List[Union[str, Path]],
        verbose    : bool = False,
    ) -> List[AudioFrontendOutput]:
        """
        Process a list of audio files sequentially.
        Returns a list of AudioFrontendOutput, one per file.

        For parallel processing across DataLoader workers,
        use this inside a Dataset.__getitem__ instead.
        """
        return [self.process(p, verbose=verbose) for p in audio_paths]
