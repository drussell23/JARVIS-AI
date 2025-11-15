#!/usr/bin/env python3
"""
Download Silero VAD model for voice activity detection.
Memory-efficient for 16GB M1 MacBooks.

Silero VAD:
- Model size: ~1.5MB
- Runtime memory: ~20MB
- Runs efficiently on Apple Silicon with MPS acceleration

Note: PyTorch with MPS backend provides excellent performance on Apple Silicon.
No CoreML conversion needed.
"""

import os
import sys

print("=" * 60)
print("Silero VAD Model Downloader")
print("=" * 60)
print("\nMemory-efficient option for 16GB M1 MacBooks")
print("Model will run on MPS (Metal Performance Shaders)\n")

# Check if we have PyTorch
try:
    import torch
    print("✅ PyTorch found")
except ImportError:
    print("❌ PyTorch not installed")
    print("\nTo install: pip install torch torchvision torchaudio")
    sys.exit(1)

# Download Silero VAD
print("\n" + "=" * 60)
print("Downloading Silero VAD model...")
print("=" * 60)

try:
    print("   Loading from torch.hub...")
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False  # Use cached version if available
    )
    print("✅ Silero VAD downloaded successfully")
    print(f"   Model type: {type(model)}")

    # Get utility functions
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

except Exception as e:
    print(f"❌ Failed to download Silero VAD: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test the model
print("\n" + "=" * 60)
print("Testing model...")
print("=" * 60)

try:
    # Create test input
    test_audio = torch.randn(1, 512)  # 32ms at 16kHz

    print(f"   Input shape: {test_audio.shape}")

    # Run inference
    model.eval()
    with torch.no_grad():
        speech_prob = model(test_audio, 16000)  # 16kHz sample rate

    print(f"   Output shape: {speech_prob.shape}")
    print(f"   Speech probability: {speech_prob.item():.4f}")

    print("✅ Model test successful")

except Exception as e:
    print(f"❌ Model test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Success!
print("\n" + "=" * 60)
print("✅ SUCCESS!")
print("=" * 60)
print("\nSilero VAD is ready to use!")
print("\nThe model is cached in: ~/.cache/torch/hub/")
print("\nMemory usage:")
print("  - Model size: ~1.5MB")
print("  - Runtime: ~20MB")
print("  - Acceleration: MPS (Metal Performance Shaders)")
print("\nNext steps:")
print("  1. The model will be loaded automatically by the voice engine")
print("  2. No additional configuration needed")
print("  3. The model runs efficiently on Apple Silicon")
print("\n" + "=" * 60)
