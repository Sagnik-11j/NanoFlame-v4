# scripts/test_block1_quick.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from blocks.block1_audio_frontend import AudioFrontend
from utils.audio_utils import describe_chunks

frontend = AudioFrontend()

out = frontend.process("your_audio.wav", verbose=True)

print(describe_chunks(out.chunks))
print(f"Duration : {out.duration_sec:.1f}s")
print(f"Capped   : {out.capped}")
print(f"Stack    : {out.stack().shape}")   # [N, 128, 1500]
