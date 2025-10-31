#!/usr/bin/env python3
"""
Fix JARVIS STT by injecting Whisper override
"""

import sys
import os
import time

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), "backend")
sys.path.insert(0, backend_path)

def fix_jarvis_stt():
    """Apply Whisper override to JARVIS"""

    print("🔧 FIXING JARVIS STT WITH WHISPER")
    print("="*50)

    # Import the override
    from voice.whisper_override import patch_jarvis_stt, get_whisper_stt

    # Initialize Whisper
    print("📦 Loading Whisper model...")
    whisper_stt = get_whisper_stt()
    whisper_stt.initialize()

    # Apply patch
    print("🔨 Patching JARVIS STT system...")
    patch_jarvis_stt()

    print("\n✅ JARVIS STT FIXED!")
    print("-"*50)
    print("Whisper is now the default STT engine")
    print("Transcription should work correctly")
    print("\n🎤 Test with: 'Hey JARVIS, unlock my screen'")

if __name__ == "__main__":
    fix_jarvis_stt()

    # Keep running to maintain patch
    print("\nPatch applied. Keep this running while testing JARVIS.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nPatch removed")