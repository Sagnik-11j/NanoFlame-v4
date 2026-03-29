# NanoFlame v4 — Laptop-Scale Hybrid Audio LM

**Whisper-Medium + OpenBEATs-Large + Qwen-2.5-0.5B-Instruct (4-bit)**  
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
| Speech Encoder | Whisper-Medium (encoder only) | ~300M | ~0.58 GB |
| Sound/Music Encoder | OpenBEATs-Large | ~300M | ~0.58 GB |
| Language Model | Qwen-2.5-0.5B-Instruct (4-bit NF4) | 0.5B | ~0.25 GB |
| Perceiver Resampler | Learned queries | — | ~0.05 GB |
| Cross-Attn Fusion + MLP Adaptor | Bridge layers | ~10M | ~0.08 GB |
| QLoRA Adapters (r=16) | fp16 | ~25M | ~0.04 GB |
| **Total** | | | **~1.58 GB ✅** |

> ~4.42 GB of VRAM headroom on a 6 GB card.

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
| 1c | Log-Mel Spectrogram (128 mel bins, 25ms window / 400 samples, 10ms hop / 160 samples, `center=False`) | `[128, T]` where **T = 3000 for 30s** (100 fps × 30s) |
| 1d | 30s sliding window chunks (non-overlapping, max 10 chunks) | N × `[128, 3000]` |

> **`center=False`** — padding is not added at the edges of the signal. This gives `T = floor((N_samples − n_fft) / hop) + 1 = 2998` frames for a 30s clip, which fits cleanly in one 3000-frame chunk. `center=True` would add `n_fft//2 = 200` samples of padding on each side, producing 3001 frames and spilling into a second chunk.

The same mel chunk is sent to **both** Block 2a and Block 2b simultaneously and independently.

---

### Block 2a — Whisper-Medium Encoder *(speech specialist)*

~300M encoder parameters. Trained on 680,000 hours of audio across 100+ languages. Only the encoder is used; the decoder is discarded.

| Step | Operation | Output |
|---|---|---|
| 2a-1 | 2× strided Conv1D stem: Conv1(128→1024, k=3, s=1) then Conv2(1024→1024, k=3, s=2) | `[1500, 1024]` |
| 2a-2 | Sinusoidal positional embeddings (1D, fixed) | `[1500, 1024]` |
| 2a-3 | 24× Transformer encoder blocks (16-head MHSA, bidirectional, FFN 1024→4096→1024) | `[1500, 1024]` |
| 2a-4 | Output at 50 Hz | `[1500, 1024]` |
| 2a-5 | Stride-2 temporal pooling (AvgPool1d k=2, s=2) | **`h_w [750, 1024]`** @ 25 Hz |

> **whisper-medium hidden dim = 1024**, with 24 encoder layers and 16 attention heads — vs whisper-small's 768-dim / 6 layers. The original `conv1` expects 80 mel bins; we replace it with `Conv1d(128→1024)`, copying pretrained weights for channels 0–79 and Xavier-initialising channels 80–127.

LoRA r=16, α=32 applied to Q/K/V/O projections from Stage 2 onward.

---

### Block 2b — OpenBEATs-Large Encoder *(sound / music / bioacoustics SSL)*

~300M parameters. Trained via iterative masked token prediction on non-speech audio. Never exposed to speech — fully non-redundant with Whisper.

| Step | Operation | Output |
|---|---|---|
| 2b-1 | 16×16 non-overlapping patches (**8 freq × 187 time = 1496 patches**) | 1496 patches |
| 2b-2 | Patch linear embedding: `flatten(16×16) → Linear(256 → 1024)` | `[1496, 1024]` |
| 2b-3 | 2D positional embeddings (freq-pos + time-pos, learned) | `[1496, 1024]` |
| 2b-4 | 24× Transformer blocks (16-head MHSA, full global attention, SwiGLU FFN inner=4096) | `[1496, 1024]` |
| 2b-5 | Full global context captured | **`h_ob [1496, 1024]`** |

> Upgraded from OpenBEATs-Base (768-dim, 12 layers) to **OpenBEATs-Large (1024-dim, 24 layers)**. Patch grid and count remain unchanged at 8×187 = 1496.

> **Implementation note:** `blocks/block2b_openbeats_encoder.py` wraps the ESPnet
> `BeatsEncoder` directly. Block 1 already produces mel `[B, 128, T]`, so
> `BeatsEncoder.preprocess()` (which recomputes mel from waveform) is bypassed.
> The wrapper routes mel directly into `patch_embedding → layer_norm →
> post_extract_proj → encoder`. No ESPnet2 install is required — lightweight
> stubs are injected at import time.

LoRA r=16, α=32 applied from Stage 2 onward.

---

### Block 2b — Setup & Verification

**Files:**
- `blocks/beats_encoder.py` — ESPnet source file (do not modify)
- `blocks/block2b_openbeats_encoder.py` — wrapper, stub injector, weight loader
- `tests/test_block2b.py` — load + verify test
- `tests/conftest.py` — prevents pytest from collecting standalone test scripts

**Checkpoint:** Place `epoch_latest.pt` (~1.2 GB) from `shikhar7ssu/OpenBEATs-ICME` in `checkpoints/`.

**Run test:**
```bash
python tests/test_block2b.py
```

**Known issues fixed:**
- Checkpoint was saved on CUDA → always load with `map_location="cpu"`
- `BeatsConfig` object not iterable → pass `_OPENBEATS_CFG_OVERRIDES` dict directly to `BeatsEncoder`
- ESPnet import path is `espnet2.legacy.nets.pytorch_backend.nets_utils` (not `espnet2.asr.nets`)

---

### Block 3 — Perceiver Resampler + Cross-Attention Fusion

Merges the two encoder outputs into a single unified representation.

```
h_w  [750  × 1024]  (Whisper-Medium)
h_ob [1496 × 1024]  (OpenBEATs-Large)
```


Both encoders output **1024-dim** — no dimension projection is needed.

| Step | Operation | Output |
|---|---|---|
| 3a | Perceiver Resampler: 64 learnable queries × 2-layer cross-attention compress h_ob | `h_resampled [64, 1024]` |
| 3b | Cross-attention fusion: Q=h_w, K=h_resampled, V=h_resampled (16 heads) + residual | `h_fused [750, 1024]` |
| 3c | Long-audio chunk concat (if N > 1) + learnable chunk-index positional encodings | **`h_full [T × 1024]`** |

> Both Whisper-Medium and OpenBEATs-Large share the same 1024-dim output space — no projection step needed. Each Whisper token at time t carries both phonetic detail and sound/music awareness in the same 1024-dim vector.

**Output dataclass:** `Block3Output` with fields `h_full`, `h_resampled`, `h_fused_chunks`, `n_chunks`, `seq_len`.  
**Stage control methods:** `freeze()` · `unfreeze()` · `enable_training_dropouts(p)`.  
**Multi-chunk API:** `b3.forward_multi_chunk(List[Tuple[h_ob, h_w]])` → concatenates N×750 tokens with learnable chunk-index positional encodings.
---

### Block 3 — Setup & Verification

**Files:**
- `blocks/block3_perceiver_fusion.py` — `PerceiverResampler`, `CrossAttentionFusion`, `ChunkConcat`, `Block3PerceiverFusion`, `Block3Output`
- `tests/test_block3.py` — 9 unit tests (synthetic tensors) + 7 integration tests (real pipeline)
- `tests/conftest.py` — tells pytest not to collect the standalone test scripts

**No checkpoint required** for unit tests — Block 3 has no pretrained weights.  
**Checkpoint required** for integration tests — needs `epoch_latest.pt` in `checkpoints/` (same file as Block 2b).

**Run tests:**
```bash
# Unit tests only (fast, no checkpoint):
python tests/test_block3.py --skip-integration

# Full pipeline: Block1 → Block2a → Block2b → Block3:
python tests/test_block3.py

# GPU:
python tests/test_block3.py --device cuda
```

**Unit tests cover:** PerceiverResampler shape + learned queries · CrossAttentionFusion seq-len preservation · ChunkConcat pos-enc differentiation · single chunk forward · 3-chunk forward · Qwen projection 1024→896 · gradient flow ≥80% · freeze/unfreeze/dropout cycle · Block3Output dataclass fields.

**Integration tests cover:** Block1 mel shapes · Block2b h_ob shapes · Block2a h_w shapes · full 4-block chain (30s) · 3-chunk 90s pipeline · Qwen-ready 1024→896 projection · Stage-1 gradient isolation (Block2b frozen, Block3 trains).

---

### Block 4 — MLP Adaptor *(bridges audio → LLM embedding space)*

The only bridge between the audio world and the language model's embedding space. **Only this block trains in Stage 1.**


```
in: h_full [T × 1024]
├─ Linear(1024 → 896) compress into LLM hidden dim (ratio 0.875×)
├─ GELU smooth nonlinearity
├─ Dropout(p=0.1) regularisation
├─ Linear(896 → 896) refine within LLM dim space
├─ RMSNorm normalise for LLM compatibility
└─ + Residual shortcut Linear(1024 → 896) on original input

out: audio_tokens [T × 896]
```


> Input dim updated from 768 → **1024** to match upgraded encoders. The adaptor now **compresses** (1024→896, ratio 0.875×) rather than expanding — this is fine since the richer 1024-dim encoder representations have more redundancy to compress from.

---



### Block 5 — Text Tokenizer *(runs in parallel with Blocks 1–4)*

Qwen-2.5 BPE tokenizer, vocabulary = 151,936.

- Text question → integer token IDs → fed into Block 6
- Same tokenizer across all Qwen-2.5 variants (0.5B / 1.5B / 7B)
- Custom audio boundary tokens added: `<audio1>`, `</audio1>`, `<audio2>`, `</audio2>`, etc.
- Training Stages 1–3: ground-truth answer tokens only
- Training Stage 4: `<think>...</think>` + answer tokens

---

###  Setup & Verification

**Files:**
- `blocks/block5_text_tokenizer.py` — `Block5TextTokenizer`, `TokenizerOutput`, `ChatPrompt`, `TrainingTarget`
- `tests/test_block5.py` — 10 unit tests (no GPU required)
- `tests/conftest.py` — add `test_block5.py` to `collect_ignore`

**No checkpoint required** — downloads Qwen-2.5-0.5B tokenizer from HuggingFace on first run (~1 MB, not the model weights).

**Run tests:**
```bash
python tests/test_block5.py

# Skip HF network call after first run:
python tests/test_block5.py --local-dir ./qwen_local
```

**Tests cover:** tokenizer load + vocab size · audio boundary token IDs (8 tokens added) · encode/decode round-trip · chat prompt structure (system/user/assistant turns) · multi-audio placeholders · CoT trigger insertion · Stage 1–3 training targets (loss on answer span only) · Stage 4 training target (loss on `<think>` span + answer) · token count helpers · max-length truncation · deterministic tokenization.

**Vocab:** base 151,936 + 8 audio tokens (`<audio1>`…`<audio4>`, `</audio1>`…`</audio4>`) = **151,944 total**.

---

> **Implementation:** `build_chat_prompt()` places `<audioN> </audioN>` boundary tokens as placeholders in the user turn. Block 6 replaces the span between these boundary tokens with real `audio_tokens [T × 896]` from Block 4. `build_training_target(stage=N)` produces the full token sequence + binary `loss_mask` in one call — loss is applied only on the assistant span.

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

### Block 6 — Setup & Verification

**Files:**
- `blocks/block6_sequence_packer.py` — `Block6SequencePacker`, `PackedSequence`
- `tests/test_block6.py` — 10 unit tests + 3 integration tests
- `tests/conftest.py` — add `test_block6.py` to `collect_ignore`

**No checkpoint required** for unit tests — uses a mock `nn.Embedding` for `embed_tokens`.  
**HF tokenizer required** for integration tests — same Qwen-2.5-0.5B tokenizer as Block 5.

**Run tests:**
```bash
python tests/test_block6.py                   # unit + integration
python tests/test_block6.py --skip-integration  # unit only (no HF call)
```

**Tests cover:** single-audio inference packing · multi-audio inference packing · Stage 1–3 training loss mask · Stage 4 CoT loss mask · audio tensor injection at correct positions · variable-length audio spans · `pack()` dispatch · attention mask shape · empty audio raises · batched `[1,T,896]` tensor handling · full Block4→Block5→Block6 chain (3 integration tests).

**Output:** `PackedSequence` with `inputs_embeds [1, total_len, 896]`, `attention_mask`, `loss_mask` (1 only on assistant span), `audio_spans`, `prompt_len`, `target_len`.

___

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

### Block 8 — Autoregressive Decoding

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
| **Time** | ~3h | ~10h | ~12h | ~8h |
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
| Qwen-2.5-0.5B-Instruct (4-bit NF4) | ~0.25 GB | unchanged |
| Whisper-Medium encoder (fp16, ~300M) | ~0.58 GB | ↑ was ~0.24 GB (Small) |
| OpenBEATs-Large (fp16, ~300M) | ~0.58 GB | ↑ was ~0.18 GB (Base) |
| Perceiver Resampler (fp16, 64 queries) | ~0.05 GB | 2 cross-attn layers × ~12.6M params |
| Cross-Attn Fusion + MLP Adaptor (fp16) | ~0.08 GB | slightly larger (1024-dim) |
| QLoRA adapters (fp16, r=16) | ~0.04 GB | slightly larger (bigger encoders) |
| KV cache (24L, 2 KV-heads, ~800 tokens) | ~0.08 GB | unchanged |
| Activations + gradient buffers | ~0.45 GB | ↑ larger encoders |
| **Total** | **~1.58 GB ✅** | **~4.42 GB free** |

---

## Training Speed

Larger encoders mean slower forward/backward passes than the Small+Base configuration:

| Stage | Time |
|---|---|
| Stage 1 | ~3 hours |
| Stage 2 | ~10 hours |
| Stage 3 | ~12 hours |
| Stage 4 | ~8 hours |
| **Total** | **~33 hours** |

Still within a single overnight+day run. The ~4.42 GB VRAM headroom allows batch sizes of 16–32 to keep training efficient.

---

## Block Connection Diagram


```
╔══════════════════════════════════════════════════════════════════════════════╗
║             NanoFlame v4 — Laptop-Scale Hybrid Audio LM                      ║
║       Whisper-Medium + OpenBEATs-Large + Qwen-2.5-0.5B-Instruct (4-bit)      ║
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
│  BLOCK 1 · AUDIO FRONTEND                                                │
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
│       center = False  → no edge padding, clean chunk boundaries          │
│       Mel filterbank: 128 triangular filters on log-frequency axis       │
│       take log of mel power values                                       │
│       output: [128 freq_bins × T time_frames]                            │
│       T = 3000 for 30 seconds  (100 fps × 30s)                           │
│       │                                                                  │
│       ▼                                                                  │
│  [1d] 30s Sliding Window Chunks                                          │
│       split [128 × T_full] into N non-overlapping chunks                 │
│       each chunk: [128 × 3000]                                           │
│       N = ceil(duration ÷ 30s),  max N = 10  (5 min cap)                 │
│                                                                          │
│  OUTPUT: chunk_1[128×3000], chunk_2[128×3000], ..., chunk_N[128×3000]    │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │  SAME mel chunk sent to     │
              │  BOTH paths simultaneously  │
              ▼                             ▼
┌─────────────────────────┐   ┌───────────────────────────────────────┐
│  BLOCK 2a               │   │  BLOCK 2b                             │
│  WHISPER-MEDIUM ENCODER │   │  OPENBEATS-LARGE ENCODER              │
│  Speech Specialist      │   │  Sound · Music · Bioacoustics SSL     │
│  ~300M params           │   │  ~300M params                         │
│  hidden dim = 1024      │   │  hidden dim = 1024                    │
├─────────────────────────┤   ├───────────────────────────────────────┤
│                         │   │                                       │
│  in: [128 × 3000]       │   │  in: [128 × 3000]                     │
│       │                 │   │       │                               │
│       ▼                 │   │       ▼                               │
│  [2a-1] Conv1D Stem     │   │  [2b-1] Patchify                      │
│  Conv1: 128→1024 str=1  │   │  16×16 non-overlapping patches        │
│  Conv2: 1024→1024 str=2 │   │  freq: 128 ÷ 16 = 8 patches           │
│  3000 → 1500 time steps │   │  time: 3000 ÷ 16 = 187 patches        │
│  out: [1500 × 1024]     │   │  total: 1496 patches per chunk        │
│       │                 │   │       │                               │
│       ▼                 │   │       ▼                               │
│  [2a-2] Sinusoidal      │   │  [2b-2] Patch Linear Embedding        │
│  Positional Embeddings  │   │  flatten 16×16 → 256 values           │
│  1D time-axis only      │   │  Linear(256 → 1024)                   │
│  added to each token    │   │  out: [1496 × 1024]                   │
│       │                 │   │       │                               │
│       ▼                 │   │       ▼                               │
│  [2a-3] 24× Transformer │   │  [2b-3] 2D Positional Embeddings      │
│  Encoder Blocks         │   │  freq-pos emb + time-pos emb          │
│  ┌─────────────────┐    │   │  summed per patch token               │
│  │ LayerNorm (pre) │    │   │  out: [1496 × 1024]                   │
│  │ MHSA 16 heads   │    │   │       │                               │
│  │ bidirectional   │    │   │       ▼                               │
│  │ + Residual      │    │   │  [2b-4] 24× Transformer Blocks        │
│  │ LayerNorm (pre) │    │   │  ┌─────────────────────────────────┐  │
│  │ FFN 1024→4096   │    │   │  │ LayerNorm (pre-norm)            │  │
│  │      →1024      │    │   │  │ MHSA 16 heads                   │  │
│  │ + Residual      │    │   │  │ FULL GLOBAL ATTENTION           │  │
│  └─────────────────┘    │   │  │ all 1496 patches attend all     │  │
│       │                 │   │  │ + Residual                      │  │
│       ▼                 │   │  │ LayerNorm (pre-norm)            │  │
│  [2a-4]                 │   │  │ SwiGLU FFN (4096 inner dim)     │  │
│  out: [1500 × 1024]     │   │  │ + Residual                      │  │
│  @ 50 Hz                │   │  └─────────────────────────────────┘  │
│       │                 │   │       │                               │
│       ▼                 │   │       ▼                               │
│  [2a-5] Stride-2 Pool   │   │  [2b-5]                               │
│  [1500×1024]→[750×1024] │   │  out: [1496 × 1024] per chunk         │
│  @ 25 Hz                │   │  full global context captured         │
│                         │   │                                       │
│  LoRA r=16, α=32        │   │  LoRA r=16, α=32                      │
│  from Stage 2 onward    │   │  from Stage 2 onward                  │
└───────────┬─────────────┘   └────────────────────┬──────────────────┘
            │                                      │
     h_w: [750 × 1024]                     h_ob: [1496 × 1024]
            │                                      │
            └──────────────────┬───────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  BLOCK 3 · PERCEIVER RESAMPLER + CROSS-ATTENTION FUSION                  │
│  (no dim projection — both encoders output 1024-dim)                     │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  [3a] Perceiver Resampler  (OpenBEATs: 1496 tokens → 64 tokens)          │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │    Q = learned queries        [64   × 1024]                     │     │
│  │    K = OpenBEATs patches      [1496 × 1024]  ← h_ob             │     │
│  │    V = OpenBEATs patches      [1496 × 1024]                     │     │
│  │    out: h_resampled [64 × 1024]                                 │     │
│  └──────────────────────────────────┬──────────────────────────────┘     │
│                                     │  h_resampled: [64 × 1024]          │
│                                     ▼                                    │
│  h_w   [750 × 1024]  ───────────────┘                                    │
│                                     ▼                                    │
│  [3b] Cross-Attention Fusion                                             │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │  Q = h_w         [750 × 1024]  ← Whisper tokens                 │     │
│  │  K = h_resampled [ 64 × 1024]  ← OpenBEATs summary              │     │
│  │  V = h_resampled [ 64 × 1024]                                   │     │
│  │  16 attention heads · 1 cross-attn layer                        │     │
│  │  out + residual: h_fused [750 × 1024]                           │     │
│  └──────────────────────────────────┬──────────────────────────────┘     │
│                                     │                                    │
│  [3c] Long-Audio Chunk Concat  (when N > 1)                              │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │  h_full = cat([h_fused_1, ..., h_fused_N])  →  [750N × 1024]    │     │
│  │  + learnable chunk-index positional encodings per chunk         │     │
│  │  out: h_full [T × 1024]                                         │     │
│  └──────────────────────────────────┬──────────────────────────────┘     │
└─────────────────────────────────────┼────────────────────────────────────┘
                                      │  h_full: [T × 1024]
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  BLOCK 4 · MLP ADAPTOR                                                   │
│  input dim: 1024 (confirmed)  output dim: 896 (LLM hidden)               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  in: h_full [T × 1024]                                                   │
│       │                                                                  │
│       ├─ Linear(1024 → 896)   compress to LLM hidden dim (ratio 0.875×)  │
│       │                                                                  │
│       ├─ GELU activation       smooth nonlinearity                       │
│       │                                                                  │
│       ├─ Dropout(p=0.1)        regularisation                            │
│       │                                                                  │
│       ├─ Linear(896 → 896)     refine within LLM dim space               │
│       │                                                                  │
│       ├─ RMSNorm               normalise values                          │
│       │                                                                  │
│       └─ + Residual shortcut   Linear(1024 → 896) on original input      │
│                                                                          │
│  out: audio_tokens [T × 896]                                             │
│                                                                          │
│  ← This block + Block 3 bridge components train in Stage 1.              |
|     Encoders and LM frozen. ←                                            │
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
                   │              │  "What emotion...?" →                │
                   │              │  [2515, 14230, 1341, ...]  (int IDs) │
                   │              │  out: [tok_1 ... tok_M]              │
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
│  BLOCK 6 · SEQUENCE PACKING  (Qwen-2.5 chat format)                      │
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
│  audio_tokens [T×896] injected directly, bypass embedding table          │
│  text tokens looked up in embedding table → [896-dim] vectors            │
│  loss: ONLY on assistant span                                            │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  BLOCK 7 · Qwen-2.5-0.5B-Instruct  (4-bit NF4 + QLoRA)                   │
│  *** MAIN CHANGE FROM v3: 1.5B R1-Distill → 0.5B Qwen standard ***       │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Base: Qwen-2.5-0.5B-Instruct  ·  4-bit NF4  →  ~0.25 GB VRAM            │
│  hidden=896  ·  24 layers  ·  14 Q-heads  ·  2 KV-heads  ·  FFN=4864     │
│                                                                          │
│  [7a] Token Embedding                                                    │
│       text token IDs → 896-dim via embedding table                       │
│       audio tokens already 896-dim → injected directly                   │
│       RoPE applied per-layer                                             │
│                                                                          │
│  [7b] 24× Transformer Decoder Blocks                                     │
│       RMSNorm → GQA (14Q / 2KV heads, QLoRA r=16) + Residual             │
│       RMSNorm → SwiGLU FFN (896→4864→896) + Residual                     │
│                                                                          │
│  [7c] LM Head                                                            │
│       RMSNorm → Linear(896 → 151936) → logits → softmax                  │
│                                                                          │
│  Flash Attention 2  ·  KV cache active in inference                      │
│  4-bit NF4 base weights NEVER updated                                    │
│  Only QLoRA adapters (fp16, r=16) trained                                │
└──────────────────────────────────────┬───────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  BLOCK 8 · AUTOREGRESSIVE DECODING                                       │
│  temperature=0.7  ·  top_p=0.9  ·  KV cache reused                       │
└──────────────────────────────────────┬───────────────────────────────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         ▼                           ▼
         ┌───────────────────────┐   ┌──────────────────────────────────┐
         │  STANDARD OUTPUT      │   │  COT OUTPUT  (Stage 4+)          │
         │  (Stages 1–3)         │   │  prompt: "Think step by step..." │
         │                       │   │                                  │
         │  Audio 1 conveys      │   │  <think>                         │
         │  excitement.          │   │  Audio 1: rising F0, rapid       │
         │  Audio 2 conveys      │   │  speech rate → excitement.       │
         │  sadness.             │   │  Audio 2: flat pitch, slow       │
         │                       │   │  pace → sadness.                 │
         │                       │   │  </think>                        │
         │                       │   │  Audio 1 conveys excitement.     │
         │                       │   │  Audio 2 conveys sadness.        │
         └───────────────────────┘   └──────────────────────────────────┘
```

---

## Verified Block Status

| Block | Status | Tests |
|---|---|---|
| Block 1 — Audio Frontend | ✅ Complete | 16/16 passing |
| Block 2a — Whisper-Medium Encoder | ✅ Complete | 26/26 passing |
| Block 2b — OpenBEATs-Large Encoder | ✅ Complete | passing |
| Block 3 — Perceiver Resampler + Fusion | ✅ Complete | 16/16 passing |
| Block 4 — MLP Adaptor | ✅ Complete | 43/43 passing |
| Block 5 — Text Tokenizer | ✅ Complete | 65/65 passing |
| Block 6 — Sequence Packing | ✅ Complete | 89/89 passing |
| Block 7 — Qwen-2.5-0.5B LLM | 🔲 Pending | — |
| Block 8 — Autoregressive Decoding | 🔲 Pending | — |
