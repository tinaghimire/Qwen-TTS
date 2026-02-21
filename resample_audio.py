#!/usr/bin/env python3
import soundfile as sf
import librosa
import numpy as np

# Load the audio
audio, sr = sf.read('/workspace/Qwen-TTS/voices/english_voice/english_voice.wav')
print(f'Original sample rate: {sr} Hz')

# Resample to 24kHz
audio_24k = librosa.resample(audio, orig_sr=sr, target_sr=24000)

# Save the resampled audio
sf.write('/workspace/Qwen-TTS/voices/english_voice/english_voice_24k.wav', audio_24k, 24000)
print(f'Resampled to 24000 Hz and saved to english_voice_24k.wav')
print(f'Duration: {len(audio_24k)/24000:.2f} seconds')
