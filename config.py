# ─────────────────────────────────────────────────────────────
# Global constants for the entire NanoFlame v4 pipeline.
# Every block imports from here — never hardcode these values.
# ─────────────────────────────────────────────────────────────

# ── Audio Frontend (Block 1) ──────────────────────────────────
SAMPLE_RATE         = 16_000          # Hz  — Whisper's required input rate
N_FFT               = 400             # samples  — 25ms window at 16kHz
HOP_LENGTH          = 160             # samples  — 10ms hop at 16kHz
N_MELS              = 128             # mel filterbank bands
CHUNK_FRAMES        = 3000            # time-frames per 30s chunk (30s × 100 fps)
MAX_CHUNKS          = 10              # hard cap = 5 minutes of audio
CHUNK_DURATION_SEC  = 30.0            # seconds per chunk
FRAMES_PER_SEC      = 100             # = SAMPLE_RATE / HOP_LENGTH

# ── Whisper Encoder (Block 2a) ────────────────────────────────
WHISPER_HIDDEN      = 512
WHISPER_HEADS       = 8
WHISPER_LAYERS      = 6
WHISPER_MODEL_ID    = "openai/whisper-small"

# ── OpenBEATs Encoder (Block 2b) ─────────────────────────────
OPENBEATS_HIDDEN    = 768
OPENBEATS_HEADS     = 12
OPENBEATS_LAYERS    = 12
OPENBEATS_PATCHES   = 752             # 8 freq × 94 time patches per chunk
OPENBEATS_MODEL_ID  = "shikhar7ssu/OpenBEATs-ICME"
PATCH_SIZE          = 16              # 16×16 mel patch

# ── Perceiver Resampler (Block 3) ────────────────────────────
NUM_QUERIES         = 64
RESAMPLER_LAYERS    = 2

# ── MLP Adaptor (Block 4) ────────────────────────────────────
ADAPTOR_INPUT_DIM   = 512             # from fused Whisper output
ADAPTOR_HIDDEN_DIM  = 896             # matches LLM hidden
ADAPTOR_OUTPUT_DIM  = 896             # Qwen-2.5-0.5B hidden dim
ADAPTOR_DROPOUT     = 0.1

# ── LLM (Block 7) ────────────────────────────────────────────
LLM_MODEL_ID        = "Qwen/Qwen2.5-0.5B-Instruct"
LLM_HIDDEN          = 896
LLM_LAYERS          = 24
LLM_Q_HEADS         = 14
LLM_KV_HEADS        = 2
LLM_FFN_INNER       = 4864
LLM_VOCAB_SIZE      = 151_936

# ── Decoding (Block 8) ────────────────────────────────────────
DECODE_TEMPERATURE  = 0.7
DECODE_TOP_P        = 0.9
DECODE_MAX_TOKENS   = 512

# ── LoRA / QLoRA ─────────────────────────────────────────────
LORA_R              = 16
LORA_ALPHA          = 32
LORA_DROPOUT        = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
