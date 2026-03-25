# NanoFlame v4 — Laptop-Scale Hybrid Audio LM

**Whisper-Small + OpenBEATs-Base + Qwen-2.5-0.5B-Instruct (4-bit)**  
**6 GB VRAM · 16 GB RAM · Trained CoT · Multi-Audio Support**

---

## Overview

NanoFlame v4 is a hybrid multimodal pipeline that combines two specialist audio encoders with a quantized language model to perform audio understanding, reasoning, and question answering — entirely on a laptop GPU.

The system accepts 1–4 audio files (speech, sound, music, or bioacoustics, up to 5 minutes each) alongside a text question, and produces either a direct answer or a chain-of-thought reasoned response.

```
Audio → [Chunk] → [Whisper ║ OpenBEATs] → [Fuse] → [Adapt]
                                                          ↓
                   Answer ← [Decode] ← [LLM] ← [Pack + Text]
```

---

## Architecture at a Glance

| Component | Model | Params | VRAM |
|---|---|---|---|
| Speech Encoder | Whisper-Small | 244M | ~0.24 GB |
| Sound/Music Encoder | OpenBEATs-Base | 90M | ~0.18 GB |
| Language Model | Qwen-2.5-0.5B-Instruct (4-bit NF4) | 0.5B | ~0.25 GB |
| Perceiver Resampler | Learned queries | — | ~0.02 GB |
| Cross-Attn Fusion + MLP Adaptor | Bridge layers | ~8M | ~0.06 GB |
| QLoRA Adapters (r=16) | fp16 | ~20M | ~0.02 GB |
| **Total** | | | **~1.15 GB ✅** |

> ~4.85 GB of VRAM headroom on a 6 GB card.

---

## Key Differences vs v3 (R1-Distill 1.5B)

| Parameter | v3 (R1-Distill 1.5B) | v4 (Qwen 2.5 0.5B) |
|---|---|---|
| LLM hidden dim | 1536 | **896** |
| LLM layers | 28 | **24** |
| LLM Q-heads | 12 | **14** |
| LLM KV-heads | 2 | 2 |
| FFN inner dim | 8960 | **4864** |
| MLP Adaptor output | 1536 | **896** |
| Native CoT | ✅ Always | ❌ Must be trained |
| Stage 4 | Multi-audio chat | **CoT training + chat** |
| VRAM (LLM) | ~0.75 GB | **~0.25 GB** |
| Decoding temp | 0.6 | **0.7** |

---

## Pipeline Blocks

### Block 1 — Audio Frontend *(pure signal processing, no neural network)*

Converts raw audio files into log-mel spectrograms.

| Step | Operation | Output |
|---|---|---|
| 1a | Load audio from disk | `float32 [channels, num_samples]`, normalized to `[-1.0, +1.0]` |
| 1b | Resample to 16 kHz mono | `[1, 16000 × duration_secs]` |
| 1c | Log-Mel Spectrogram (128 mel bins, 25ms window, 10ms hop) | `[128, T]` where T = 1500 for 30s |
| 1d | 30s sliding window chunks (non-overlapping, max 10 chunks) | N × `[128, 1500]` |

The same mel chunk is sent to **both** Block 2a and Block 2b simultaneously and independently.

---

### Block 2a — Whisper-Small Encoder *(speech specialist)*

244M parameters. Trained on 680,000 hours of audio across 100+ languages. Only the encoder is used; the decoder is discarded.

| Step | Operation | Output |
|---|---|---|
| 2a-1 | 2× strided Conv1D stem along time axis | `[1500, 512]` |
| 2a-2 | Sinusoidal positional embeddings (1D, fixed) | `[1500, 512]` |
| 2a-3 | 6× Transformer encoder blocks (8-head MHSA, bidirectional, FFN 512→2048→512) | `[1500, 512]` |
| 2a-4 | Output at 50 Hz | `[1500, 512]` |
| 2a-5 | Stride-2 temporal pooling | **`h_w [750, 512]`** @ 25 Hz |

LoRA r=16, α=32 applied from Stage 2 onward.

---

### Block 2b — OpenBEATs-Base Encoder *(sound / music / bioacoustics SSL)*

90M parameters. Trained via iterative masked token prediction on 20,000 hours of non-speech audio. Never exposed to speech — fully non-redundant with Whisper.

| Step | Operation | Output |
|---|---|---|
| 2b-1 | 16×16 non-overlapping patches (8 freq × 94 time = 752 patches) | 752 patches |
| 2b-2 | Patch linear embedding: `flatten(16×16) → Linear(256 → 768)` | `[752, 768]` |
| 2b-3 | 2D positional embeddings (freq-pos + time-pos, learned) | `[752, 768]` |
| 2b-4 | 12× Transformer blocks (12-head MHSA, full global attention, SwiGLU FFN inner=3072) | `[752, 768]` |
| 2b-5 | Full global context captured | **`h_ob [752, 768]`** |

LoRA r=16, α=32 applied from Stage 2 onward.

---

### Block 3 — Perceiver Resampler + Cross-Attention Fusion

Merges the two encoder outputs into a single unified representation.

```
h_w  [750 × 512]  (Whisper)
h_ob [752 × 768]  (OpenBEATs)
```

| Step | Operation | Output |
|---|---|---|
| 3a | Perceiver Resampler: 64 learnable queries × 2-layer cross-attention compress h_ob | `h_resampled [64, 768]` |
| 3b | Dimension projection: `Linear(768 → 512)` | `h_proj [64, 512]` |
| 3c | Cross-attention fusion: Q=h_w, K=h_proj, V=h_proj (2 heads) + residual | `h_fused [750, 512]` |
| 3d | Long-audio chunk concat (if N > 1) + learnable chunk-index positional encodings | **`h_full [T × 512]`** |

Each Whisper token at time t now carries both fine phonetic detail (Whisper) and sound/music awareness (OpenBEATs) within the same 512-dim vector.

---

### Block 4 — MLP Adaptor *(changed for 0.5B: output dim 1536 → 896)*

The only bridge between the audio world and the language model's embedding space. **Only this block trains in Stage 1.**

```
in:  h_full [T × 512]
  ├─ Linear(512 → 896)       expand by 1.75× into LLM dim space
  ├─ GELU                    smooth nonlinearity
  ├─ Dropout(p=0.1)          regularisation
  ├─ Linear(896 → 896)       refine within LLM dim space
  ├─ RMSNorm                 normalise for LLM compatibility
  └─ + Residual shortcut     Linear(512 → 896) on original input

out: audio_tokens [T × 896]
```

---

### Block 5 — Text Tokenizer *(runs in parallel with Blocks 1–4)*

Qwen-2.5 BPE tokenizer, vocabulary = 151,936.

- Text question → integer token IDs → fed into Block 6
- Same tokenizer across all Qwen-2.5 variants (0.5B / 1.5B / 7B)
- Custom audio boundary tokens added: `<audio1>`, `</audio1>`, `<audio2>`, `</audio2>`, etc.
- Training Stages 1–3: ground-truth answer tokens only
- Training Stage 4: `<think>...</think>` + answer tokens

---

### Block 6 — Sequence Packing

Everything assembled into a single flat sequence in Qwen-2.5 chat format:

```
[BOS]
<|im_start|>system
You are an audio AI that analyzes sound carefully before answering.
<|im_end|>
<|im_start|>user
[<audio1>] audio_tokens_1 [T × 896] [</audio1>]
[<audio2>] audio_tokens_2 [T × 896] [</audio2>]
What emotion does the speaker convey? Compare with audio 2.
<|im_end|>
<|im_start|>assistant
← model generates from here
```

Audio tokens (already 896-dim) bypass the embedding table and are injected directly. Text tokens are looked up in the embedding table → 896-dim vectors. Both live in the same 896-dim space. Loss is computed **only on the assistant response span**.

> **Key difference from v3:** Qwen-2.5-0.5B has no native CoT. `<think>` blocks are NOT automatic. In Stages 1–3 the model trains without CoT targets. In Stage 4, synthetic CoT traces are explicitly injected as training targets, triggered by the prompt suffix: *"Think step by step before answering."*

---

### Block 7 — Qwen-2.5-0.5B-Instruct (4-bit NF4 + QLoRA)

*Main change from v3: 1.5B R1-Distill → 0.5B Qwen standard (no native reasoning)*

| Parameter | Value |
|---|---|
| Base model | Qwen-2.5-0.5B-Instruct |
| Quantization | 4-bit NF4 via bitsandbytes → **~0.25 GB VRAM** |
| Hidden dim | 896 |
| Transformer layers | 24 |
| Q-heads | 14 (head_dim = 896 ÷ 14 = **64**) |
| KV-heads | 2 (GQA ratio 7:1 — KV cache 7× smaller than MHA) |
| FFN inner dim | 4864 (5.43× expansion) |
| Vocabulary | 151,936 |
| Context window | 32,768 tokens (native) |
| CoT | None native — must be trained in Stage 4 |

**Internals per decoder block:**

```
RMSNorm
  └─ Grouped-Query Attention (GQA)
       14 Q-heads · 2 KV-heads
       every 7 Q-heads share 1 K-head and 1 V-head
       QLoRA on W_q, W_k, W_v, W_o (r=16, α=32)
+ Residual
RMSNorm
  └─ SwiGLU FFN
       gate_proj: Linear(896 → 4864)
       up_proj:   Linear(896 → 4864)
       out = SiLU(gate_proj(x)) ⊙ up_proj(x)
       down_proj: Linear(4864 → 896)
+ Residual
```

RoPE (Rotary Positional Encoding) applied per-layer. 4-bit NF4 base weights are **never updated** — only QLoRA adapters (fp16, r=16) are trained.

---

### Block 8 — Autoregressive Decoding *(changed: no native CoT, new sampling params)*

```
1. LM head produces logits at last position
2. temperature = 0.7  (standard Qwen-2.5; was 0.6 for R1-Distill)
3. top_p = 0.9        (sample from top 90% probability mass)
4. sample token → append to sequence
5. KV cache reused — audio tokens never re-encoded per step
6. repeat until <|im_end|> or max_new_tokens
7. decode token IDs → final answer string
```

**KV cache size** (24 layers, 2 KV-heads, 800 tokens): `2 × 24 × 800 × 2 × 64 × 2 bytes ≈ 75 MB`

**Output modes:**

| Mode | Trigger | Example |
|---|---|---|
| Direct answer (Stages 1–3) | Default | `Audio 1 conveys excitement. Audio 2 conveys sadness.` |
| CoT answer (Stage 4+) | Prompt ends with *"Think step by step before answering."* | `<think>…reasoning…</think>` followed by answer |

---

## 4-Stage Curriculum Training

> **Stage 4 is reintroduced for v4** — Qwen-2.5-0.5B has no native chain-of-thought, unlike R1-Distill-1.5B which inherited CoT via knowledge distillation from a 671B teacher.

| | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---|---|---|---|---|
| **Name** | Adaptor Alignment | Encoder Adaptation | Reasoning QLoRA | CoT Training + Multi-Audio |
| **Trains** | MLP Adaptor, Perceiver Resampler, 64 queries | Whisper LoRA r=16, OpenBEATs LoRA r=16, Adaptor, Perceiver | QLoRA on LLM r=16, Adaptor, all LoRAs, Perceiver | QLoRA r=16, Adaptor, all LoRAs, Perceiver |
| **Frozen** | Whisper, OpenBEATs, LLM (4-bit NF4) | LLM (4-bit NF4) | LLM base (4-bit) | LLM base (4-bit) |
| **Data** | AudioCaps, ClothoV2, MusicCaps, ESC-50 QA | AudioCaps, MusicCaps, LibriSpeech, VoxCeleb, FSD-50K QA | AudioSkills, WavText5K, Custom MCQs | Synthetic CoT traces (~50k), multi-audio dialogue, AF-Chat pairs |
| **Loss** | CE on answer tokens only | CE on answer tokens only | CE on answer tokens only (no `<think>`) | CE on `<think>` span + answer span |
| **Time** | ~2h | ~6h | ~10h | ~6h |
| **Goal** | Bridge audio → 896-dim LLM space | 3-domain unified encoder repr. | Audio-grounded QA reasoning | On-demand CoT + multi-audio chat |

All stages use: gradient checkpointing · bf16 mixed precision · AdamW · batch=16, grad_accum=4 (effective batch=64) · cosine LR decay.

### How CoT Traces Are Generated (Stage 4)

1. Take the Stage 3 training set (correct Q+A pairs)
2. Send each pair to GPT-4o-mini / Gemini Flash with the prompt:
   > *"Given this audio question and correct answer, write a short `<think>` block (40–60 words) that shows reasoning grounded in what you would observe in the audio spectrogram."*
3. Use the generated `<think>` blocks as Stage 4 training targets
4. Produces ~50k CoT-augmented examples cheaply

---

## VRAM Budget

| Component | VRAM | Notes |
|---|---|---|
| Qwen-2.5-0.5B-Instruct (4-bit NF4) | ~0.25 GB | ↓ was 0.75 GB in v3 |
| Whisper-Small (fp16, 244M) | ~0.24 GB | unchanged |
| OpenBEATs-Base (fp16, 90M) | ~0.18 GB | unchanged |
| Perceiver Resampler (fp16, 64 queries) | ~0.02 GB | unchanged |
| Cross-Attn Fusion + MLP Adaptor (fp16) | ~0.06 GB | smaller now |
| QLoRA adapters (fp16, r=16, 0.5B) | ~0.02 GB | ↓ was 0.05 GB |
| KV cache (24L, 2 KV-heads, ~800 tokens) | ~0.08 GB | ↓ was 0.15 GB |
| Activations + gradient buffers | ~0.30 GB | ↓ was 0.50 GB |
| **Total** | **~1.15 GB ✅** | **~4.85 GB free** |

### What You Can Do With the Headroom

- Upgrade to **OpenBEATs-Large (300M)** → +0.42 GB, total ~1.57 GB
- Run **two full pipelines in parallel** → 2× training throughput
- Use **batch size 32+** during training → faster convergence
- Extend **max audio to 10 minutes** (20 chunks) → still only ~1.5 GB total
- Keep everything in **fp16**, skip int8 → simpler code
- Add a **second Whisper-Medium** → richer speech features

---

## Training Speed

With only 0.5B parameters, every forward/backward pass is ~3× faster than the 1.5B model:

| Stage | Time |
|---|---|
| Stage 1 | ~2 hours |
| Stage 2 | ~6 hours |
| Stage 3 | ~10 hours |
| Stage 4 | ~6 hours |
| **Total** | **< 24 hours** |

The massive VRAM headroom also allows batch sizes of 16–32, further accelerating convergence — faster than any prior NanoFlame version.

---

## Block Connection Diagram

```
╔══════════════════════════════════════════════════════════════════════════════╗
║             NanoFlame v4 — Laptop-Scale Hybrid Audio LM                      ║
║       Whisper-Small + OpenBEATs-Base + Qwen-2.5-0.5B-Instruct (4-bit)        ║
║            6 GB VRAM · 16 GB RAM · Trained CoT · Multi-Audio Support         ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────┐
│  INPUTS                                                                  │
│                                                                          │
│  audio1.wav ─┐                                                           │
│  audio2.wav ─┤  1–4 audio files  (speech / sound / music / bioac.)       │
│  audio3.wav ─┘  up to 5 minutes each                                     │
│                                                                          │
│  text_question: "What emotion is in audio 1? Compare with audio 2."      │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  BLOCK 1 · AUDIO FRONTEND  [UNCHANGED from v3]                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  [1a] Load audio file                                                    │
│       read from disk → float32 tensor [num_samples]                      │
│       │                                                                  │
│       ▼                                                                  │
│  [1b] Resample + Mono                                                    │
│       resample to exactly 16,000 samples/sec                             │
│       stereo → mono: avg(left_channel, right_channel)                    │
│       output: [16000 × duration_secs]                                    │
│       │                                                                  │
│       ▼                                                                  │
│  [1c] Log-Mel Spectrogram                                                │
│       STFT window = 25ms (400 samples)                                   │
│       STFT hop   = 10ms (160 samples) → 100 frames per second            │
│       Mel filterbank: 128 triangular filters on log-frequency axis       │
│       take log of mel power values                                       │
│       output: [128 freq_bins × T time_frames]                            │
│       T = 1500 for 30 seconds                                            │
│       │                                                                  │
│       ▼                                                                  │
│  [1d] 30s Sliding Window Chunks                                          │
│       split [128 × T_full] into N non-overlapping chunks                 │
│       each chunk: [128 × 1500]                                           │
│       N = ceil(duration ÷ 30s),  max N = 10  (5 min cap)                 │
│                                                                          │
│  OUTPUT: chunk_1[128×1500], chunk_2[128×1500], ..., chunk_N[128×1500]    │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │  SAME mel chunk sent to     │
              │  BOTH paths simultaneously  │
              ▼                             ▼
┌─────────────────────────┐   ┌───────────────────────────────────────┐
│  BLOCK 2a               │   │  BLOCK 2b                             │
│  WHISPER-SMALL ENCODER  │   │  OPENBEATS-BASE ENCODER               │
│  Speech Specialist      │   │  Sound · Music · Bioacoustics SSL     │
│  244M params            │   │  90M params                           │
│  hidden dim = 512       │   │  hidden dim = 768                     │
│  [UNCHANGED from v3]    │   │  [UNCHANGED from v3]                  │
├─────────────────────────┤   ├───────────────────────────────────────┤
│                         │   │                                       │
│  in: [128 × 1500]       │   │  in: [128 × 1500]                     │
│       │                 │   │       │                               │
│       ▼                 │   │       ▼                               │
│  [2a-1] Conv1D Stem     │   │  [2b-1] Patchify                      │
│  2× strided Conv1D      │   │  16×16 non-overlapping patches        │
│  along time axis        │   │  freq: 128 ÷ 16 = 8 patches           │
│  out: [1500 × 512]      │   │  time: 1500 ÷ 16 ≈ 94 patches         │
│       │                 │   │  total: 752 patches per chunk         │
│       ▼                 │   │       │                               │
│  [2a-2] Sinusoidal      │   │       ▼                               │
│  Positional Embeddings  │   │  [2b-2] Patch Linear Embedding        │
│  1D time-axis only      │   │  flatten 16×16 → 256 values           │
│  added to each token    │   │  Linear(256 → 768)                    │
│       │                 │   │  out: [752 × 768]                     │
│       ▼                 │   │       │                               │
│  [2a-3] 6× Transformer  │   │       ▼                               │
│  Encoder Blocks         │   │  [2b-3] 2D Positional Embeddings      │
│  ┌─────────────────┐    │   │  freq-pos emb + time-pos emb          │
│  │ LayerNorm (pre) │    │   │  summed per patch token               │
│  │ MHSA  8 heads   │    │   │  out: [752 × 768]                     │
│  │ bidirectional   │    │   │       │                               │
│  │ + Residual      │    │   │       ▼                               │
│  │ LayerNorm (pre) │    │   │  [2b-4] 12× Transformer Blocks        │
│  │ FFN: Lin→GELU   │    │   │  ┌─────────────────────────────────┐  │
│  │      →Lin       │    │   │  │ LayerNorm (pre-norm)            │  │
│  │ + Residual      │    │   │  │ MHSA 12 heads                   │  │
│  └─────────────────┘    │   │  │ FULL GLOBAL ATTENTION           │  │
│       │                 │   │  │ all 752 patches attend all 752  │  │
│       ▼                 │   │  │ + Residual                      │  │
│  [2a-4]                 │   │  │ LayerNorm (pre-norm)            │  │
│  out: [1500 × 512]      │   │  │ SwiGLU FFN (3072 inner dim)     │  │
│  @ 50 Hz                │   │  │ + Residual                      │  │
│       │                 │   │  └─────────────────────────────────┘  │
│       ▼                 │   │       │                               │
│  [2a-5] Stride-2 Pool   │   │       ▼                               │
│  [1500×512]→[750×512]   │   │  [2b-5]                               │
│  @ 25 Hz                │   │  out: [752 × 768] per chunk           │
│                         │   │  full global context captured         │
│  LoRA r=16, α=32        │   │  LoRA r=16, α=32                      │
│  from Stage 2 onward    │   │  from Stage 2 onward                  │
└───────────┬─────────────┘   └────────────────────┬──────────────────┘
            │                                       │
     h_w: [750 × 512]                      h_ob: [752 × 768]
            │                                       │
            └──────────────────┬────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  BLOCK 3 · PERCEIVER RESAMPLER + CROSS-ATTENTION FUSION  [UNCHANGED]     │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  [3a] Perceiver Resampler  (OpenBEATs: 752 tokens → 64 tokens)           │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │    Q = learned queries        [64  × 768]                       │     │
│  │    K = OpenBEATs patches      [752 × 768]  ← h_ob               │     │
│  │    V = OpenBEATs patches      [752 × 768]                       │     │
│  │    out: h_resampled [64 × 768]                                  │     │
│  └──────────────────────────────────┬──────────────────────────────┘     │
│                                     │  h_resampled: [64 × 768]           │
│                                     ▼                                    │
│  [3b] Dimension Projection                                               │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │  Linear(768 → 512)                                              │     │
│  │  out: h_proj [64 × 512]  ← same dim as Whisper                  │     │
│  └──────────────────────────────────┬──────────────────────────────┘     │
│                                     │                                    │
│  h_w   [750 × 512]  ────────────────┘                                    │
│  h_proj [ 64 × 512]                                                      │
│                                     ▼                                    │
│  [3c] Cross-Attention Fusion                                             │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │  Q = h_w    [750 × 512]  ← Whisper tokens (to be enriched)      │     │
│  │  K = h_proj [ 64 × 512]  ← OpenBEATs summary                    │     │
│  │  V = h_proj [ 64 × 512]                                         │     │
│  │  2 attention heads · 1 cross-attn layer                         │     │
│  │  out + residual: h_fused [750 × 512]                            │     │
│  └──────────────────────────────────┬──────────────────────────────┘     │
│                                     │                                    │
│  [3d] Long-Audio Chunk Concat  (when N > 1)                              │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │  h_full = cat([h_fused_1, ..., h_fused_N])  →  [750N × 512]     │     │
│  │  + learnable chunk-index positional encodings per chunk         │     │
│  │  out: h_full [T × 512]                                          │     │
│  └──────────────────────────────────┬──────────────────────────────┘     │
└─────────────────────────────────────┼────────────────────────────────────┘
                                      │  h_full: [T × 512]
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  BLOCK 4 · MLP ADAPTOR                                                   │
│  *** CHANGED FOR 0.5B: output dim 1536 → 896 ***                         │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  in: h_full [T × 512]                                                    │
│       │                                                                  │
│       ├─ Linear(512 → 896)     expand to LLM hidden dim  (ratio 1.75×)   │
│       │                                                                  │
│       ├─ GELU activation       smooth nonlinearity                       │
│       │                                                                  │
│       ├─ Dropout(p=0.1)        regularisation                            │
│       │                                                                  │
│       ├─ Linear(896 → 896)     refine within LLM dim space               │
│       │                        (was 768→1536 for 1.5B)                   │
│       │                                                                  │
│       ├─ RMSNorm               normalise values                          │
│       │                                                                  │
│       └─ + Residual shortcut   Linear(512 → 896) on original input       │
│                                added after norm                          │
│                                                                          │
│  out: audio_tokens [T × 896]                                             │
│       ↑ was [T × 1536] for 1.5B,  [T × 3584] for 7B                      │
│                                                                          │
│  ← ONLY this block trains in Stage 1. All else frozen. ←                 │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                        audio_tokens [T × 896]
                                   │
                   ┌───────────────┘
                   │                         (parallel path — text side)
                   │
                   │              ┌──────────────────────────────────────┐
                   │              │  BLOCK 5 · TEXT TOKENIZER            │
                   │              ├──────────────────────────────────────┤
                   │              │  Qwen-2.5 tokenizer (vocab=151,936)  │
                   │              │  "What emotion..." →                 │
                   │              │  [2515, 14230, 1341, ...]  (int IDs) │
                   │              │  out: [tok_1 ... tok_M]              │
                   │              │                                      │
                   │              │  SAME tokenizer as 1.5B / 7B         │
                   │              │  Qwen 0.5B shares vocab + format     │
                   │              │                                      │
                   │              │  Training only:                      │
                   │              │    Stage 1–3: answer tokens only     │
                   │              │    Stage 4: <think>...</think>+ans   │
                   │              └──────────────────┬───────────────────┘
                   │                                 │
                   └─────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  BLOCK 6 · SEQUENCE PACKING                                              │
│  Qwen-2.5 chat format  [MOSTLY UNCHANGED]                                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  [BOS]                                                                   │
│  <|im_start|>system                                                      │
│  You are an audio AI that analyzes sound carefully before answering.     │
│  <|im_end|>                                                              │
│  <|im_start|>user                                                        │
│  [<audio1>]  audio_tokens_1  [T × 896]  [</audio1>]                      │
│  [<audio2>]  audio_tokens_2  [T × 896]  [</audio2>]  ← if present        │
│  What emotion does the speaker convey? Compare with audio 2.             │
│  <|im_end|>                                                              │
│  <|im_start|>assistant                                                   │
│    ← INFERENCE: model generates from here                                │
│    ← TRAINING Stages 1–3: plain answer target appended                   │
│    ← TRAINING Stage 4:    <think>...</think> + answer appended           │
│                                                                          │
│  KEY DIFFERENCE FROM v3:                                                 │
│  Qwen-2.5-0.5B has NO native CoT. <think> blocks are NOT automatic.      │
│  Stage 4 explicitly injects synthetic CoT traces as training targets     │
│  and teaches the model to trigger CoT when prompted with                 │
│  "Think step by step before answering."                                  │
│                                                                          │
│  audio_tokens [T×896] injected directly, bypass embedding table          │
│  text tokens looked up in embedding table → [896-dim] vectors            │
│  both live in same 896-dim space for the LLM                             │
│  causal mask: answer tokens attend freely back to all audio tokens       │
│  loss: ONLY on assistant span                                            │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  BLOCK 7 · Qwen-2.5-0.5B-Instruct  (4-bit NF4 + QLoRA)                   │
│  *** MAIN CHANGE FROM v3: 1.5B R1-Distill → 0.5B Qwen standard ***       │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Base:      Qwen-2.5-0.5B-Instruct  (standard instruction fine-tuned)    │
│  Params:    0.5B                                                         │
│  Quant:     4-bit NF4 via bitsandbytes  →  ~0.25 GB VRAM                 │
│  Hidden:    896     (was 1536 in 1.5B,  3584 in 7B)                      │
│  Layers:    24      (was 28  in 1.5B and 7B)                             │
│  FFN inner: 4864    (was 8960 in 1.5B, 18944 in 7B)                      │
│  Q heads:   14      (was 12  in 1.5B,  28 in 7B)                         │
│  KV heads:  2       (same as 1.5B)                                       │
│  head_dim:  64      (896 ÷ 14 = 64)                                      │
│  Vocab:     151,936 (same across all Qwen-2.5 variants)                  │
│  CoT:       NONE native  (must be trained in Stage 4)                    │
│                                                                          │
│  [7a] Token Embedding                                                    │
│  ┌────────────────────────────────────────────────────────────────┐      │
│  │  text token IDs  → 896-dim vectors via embedding table         │      │
│  │  audio tokens already 896-dim  → injected directly             │      │
│  │  RoPE (Rotary Positional Encoding) applied per-layer           │      │
│  └──────────────────────────────────┬─────────────────────────────┘      │
│                                     │                                    │
│                                     ▼                                    │
│  [7b] 24× Transformer Decoder Blocks                                     │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │                                                                 │     │
│  │  ┌── RMSNorm ─────────────────────────────────────────────────┐ │     │
│  │  │  Grouped-Query Attention (GQA)                             │ │     │
│  │  │  14 Q-heads · 2 KV-heads                                   │ │     │
│  │  │  every 7 Q-heads share 1 K-head and 1 V-head               │ │     │
│  │  │  KV cache 7× smaller than standard MHA                     │ │     │
│  │  │  head_dim = 896 ÷ 14 = 64                                  │ │     │
│  │  │  causal mask: pos i attends only to pos ≤ i                │ │     │
│  │  │  audio + text tokens attend freely to each other           │ │     │
│  │  │  QLoRA on W_q, W_k, W_v, W_o  (r=16, α=32)                 │ │     │
│  │  └───────────────────────────────────────────────────────────-┘ │     │
│  │  + Residual                                                     │     │
│  │                                                                 │     │
│  │  ┌── RMSNorm ─────────────────────────────────────────────────┐ │     │
│  │  │  SwiGLU Feed-Forward Network                               │ │     │
│  │  │  gate_proj:  Linear(896 → 4864)                            │ │     │
│  │  │  up_proj:    Linear(896 → 4864)                            │ │     │
│  │  │  out = SiLU(gate_proj(x)) ⊙ up_proj(x)                    │ │     │
│  │  │  down_proj:  Linear(4864 → 896)                            │ │     │
│  │  └────────────────────────────────────────────────────────────┘ │     │
│  │  + Residual                                                     │     │
│  └─────────────────────────────────────────────────────────────────┘     │
│                                     │                                    │
│                                     ▼                                    │
│  [7c] LM Head                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │  RMSNorm on final hidden state                                  │     │
│  │  Linear(896 → 151936)  → logits over full vocabulary            │     │
│  │  Softmax  → probability distribution over next token            │     │
│  └──────────────────────────────────┬─────────────────────────────-┘     │
│                                     │                                    │
│  Flash Attention 2 · KV cache active in inference                        │
│  4-bit NF4 base weights NEVER updated                                    │
│  Only QLoRA adapters (fp16, r=16) trained                                │
└──────────────────────────────────────┬───────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  BLOCK 8 · AUTOREGRESSIVE DECODING                                       │
│  *** CHANGED: no native CoT, different sampling params ***               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. LM head produces logits at last position                             │
│  2. temperature = 0.7   (standard Qwen-2.5 recommendation)               │
│                          (was 0.6 for R1-Distill — lower temp needed     │
│                           there to keep CoT focused)                     │
│  3. top_p = 0.9          (sample from top 90% probability mass)          │
│  4. sample token  →  append to sequence                                  │
│  5. KV cache reused — audio tokens never re-encoded per step             │
│  6. repeat until <|im_end|> or max_new_tokens                            │
│  7. decode token IDs  →  final answer string                             │
│                                                                          │
│  Stages 1–3 model: direct answer, NO <think> block                       │
│  Stage 4+  model:  triggered by prompt suffix                            │
│                    "Think step by step before answering."                │
│                    → model generates <think>...</think> then answer      │
│                                                                          │
└──────────────────────────────────────┬───────────────────────────────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         ▼                           ▼
         ┌───────────────────────┐   ┌──────────────────────────────────┐
         │  STANDARD OUTPUT      │   │  COT OUTPUT  (Stage 4+)          │
         │  (Stages 1–3)         │   │  prompt must include:            │
         │                       │   │  "Think step by step..."         │
         │  Audio 1 conveys      │   │                                  │
         │  excitement.          │   │  <think>                         │
         │  Audio 2 conveys      │   │  Audio 1: rising F0, rapid       │
         │  sadness.             │   │  speech rate → excitement.       │
         │                       │   │  Audio 2: flat pitch, slow       │
         │                       │   │  pace → sadness.                 │
         │                       │   │  </think>                        │
         │                       │   │  Audio 1 conveys excitement.     │
         │                       │   │  Audio 2 conveys sadness.        │
         └───────────────────────┘   └──────────────────────────────────┘
```
