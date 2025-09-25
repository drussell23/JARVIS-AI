#!/usr/bin/env python3
"""Quick test to verify JARVIS audio is working properly."""

import subprocess
import tempfile
import os

# Test Daniel voice directly
print("Testing Daniel voice directly...")
subprocess.run(["say", "-v", "Daniel", "Hello Sir, this is JARVIS with Daniel's British voice."])

# Test generating audio file
print("\nGenerating audio file with Daniel voice...")
with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
    tmp_path = tmp.name

subprocess.run([
    "say",
    "-v",
    "Daniel",
    "-o",
    tmp_path,
    "Testing audio file generation. Sir, your audio system is working correctly."
])

# Check file size
file_size = os.path.getsize(tmp_path)
print(f"Audio file generated: {tmp_path} ({file_size} bytes)")

# Play it back
print("Playing generated audio...")
subprocess.run(["afplay", tmp_path])

# Clean up
os.unlink(tmp_path)
print("\nAudio test complete!")