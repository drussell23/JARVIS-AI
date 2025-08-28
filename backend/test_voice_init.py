#!/usr/bin/env python3
"""Test voice initialization step by step"""

print("Step 1: Import VoiceConfig")
from engines.voice_engine import VoiceConfig
print("  ✓ VoiceConfig imported")

print("\nStep 2: Import VoiceAssistant")
from engines.voice_engine import VoiceAssistant  
print("  ✓ VoiceAssistant imported")

print("\nStep 3: Create VoiceConfig")
config = VoiceConfig()
print("  ✓ VoiceConfig created")

print("\nStep 4: Create VoiceAssistant")
print("  Creating assistant...")
assistant = VoiceAssistant(config)
print("  ✓ VoiceAssistant created")

print("\nAll steps completed successfully!")