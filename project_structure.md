nanoflame/
│
├── config.py                      ← global constants (sample rate, FFT params, etc.)
│
├── blocks/
│   ├── __init__.py
│   ├── block1_audio_frontend.py   ← YOU ARE HERE
│   ├── block2a_whisper.py         ← next
│   ├── block2b_openbeats.py
│   ├── block3_fusion.py
│   ├── block4_adaptor.py
│   ├── block5_tokenizer.py
│   ├── block6_sequence.py
│   ├── block7_llm.py
│   └── block8_decode.py
│
├── training/
│   ├── __init__.py
│   ├── dataset.py
│   ├── collator.py
│   └── trainer.py
│
├── utils/
│   ├── __init__.py
│   └── audio_utils.py
│
├── tests/
│   ├── __init__.py
│   └── test_block1.py
│
├── scripts/
│   ├── train.py
│   └── infer.py
│
└── requirements.txt
