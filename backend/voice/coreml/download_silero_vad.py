#!/usr/bin/env python3
"""
Download and convert Silero VAD to CoreML format.
Memory-efficient for 16GB M1 MacBooks.

Silero VAD:
- Model size: ~1.5MB
- Runtime memory: ~20MB
- Optimized for Apple Silicon
"""

import os
import sys
import urllib.request

print("=" * 60)
print("Silero VAD Model Downloader for CoreML")
print("=" * 60)
print("\nMemory-efficient option for 16GB M1 MacBooks")
print("Model will run on Neural Engine with minimal RAM usage\n")

# Check if we have the required packages
try:
    import torch
    print("✅ PyTorch found")
except ImportError:
    print("❌ PyTorch not installed")
    print("\nTo install: pip install torch torchvision torchaudio")
    sys.exit(1)

try:
    import coremltools as ct
    print("✅ CoreML Tools found")
except ImportError:
    print("❌ CoreML Tools not installed")
    print("\nTo install: pip install coremltools")
    sys.exit(1)

# Download Silero VAD
print("\n" + "=" * 60)
print("Step 1: Downloading Silero VAD model...")
print("=" * 60)

try:
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=True,
        onnx=False
    )
    print("✅ Silero VAD downloaded successfully")
    print(f"   Model type: {type(model)}")

    # Get model info
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

except Exception as e:
    print(f"❌ Failed to download Silero VAD: {e}")
    sys.exit(1)

# Convert to CoreML
print("\n" + "=" * 60)
print("Step 2: Converting to CoreML format...")
print("=" * 60)

try:
    # Set model to eval mode
    model.eval()

    # Create example input (Silero VAD expects audio chunks)
    # Input shape: (batch_size, audio_length)
    # Typical: 512 samples at 16kHz = 32ms chunks
    example_input = torch.randn(1, 512)

    print(f"   Input shape: {example_input.shape}")

    # Trace the model
    print("   Tracing model with example input...")
    traced_model = torch.jit.trace(model, example_input)

    # Convert to CoreML
    print("   Converting to CoreML...")
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="audio", shape=(1, 512))],
        outputs=[ct.TensorType(name="output")],
        compute_units=ct.ComputeUnit.ALL,  # Use Neural Engine + GPU + CPU
        minimum_deployment_target=ct.target.macOS12
    )

    print("✅ Conversion successful")

    # Save the model
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(models_dir, exist_ok=True)

    output_path = os.path.join(models_dir, 'vad_model.mlpackage')

    print(f"\n   Saving to: {output_path}")
    coreml_model.save(output_path)

    print("✅ Model saved successfully")

    # Get model size
    import subprocess
    result = subprocess.run(['du', '-sh', output_path], capture_output=True, text=True)
    size = result.stdout.split()[0]
    print(f"   Model size: {size}")

except Exception as e:
    print(f"❌ Conversion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Success!
print("\n" + "=" * 60)
print("✅ SUCCESS!")
print("=" * 60)
print(f"\nVAD Model ready: {output_path}")
print("\nMemory usage:")
print("  - Download: ~500MB (temporary, PyTorch cache)")
print("  - Runtime: ~20MB (loaded in Neural Engine)")
print("\nNext steps:")
print("  1. The model is ready to use with the CoreML Voice Engine")
print("  2. You still need a Speaker Recognition model (optional)")
print("  3. The voice engine will work with just VAD for now")
print("\n" + "=" * 60)
