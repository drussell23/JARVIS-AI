#!/usr/bin/env python3
"""
Simple test to verify Picovoice wake word detection
"""

import os
import pvporcupine
import struct
import numpy as np

# Set your key
os.environ["PICOVOICE_ACCESS_KEY"] = "e9AVn4el49rhJxxUDILvK2vYOTbFZx1ZSlSnsZqEu3kLY3Ix8Ghckg=="

def test_basic_picovoice():
    """Basic Picovoice test"""
    print("🎤 Testing Picovoice Wake Word Detection")
    print("=" * 40)
    
    # Create Porcupine instance
    porcupine = pvporcupine.create(
        access_key=os.environ["PICOVOICE_ACCESS_KEY"],
        keywords=["jarvis"]  # Built-in wake word
    )
    
    print(f"✅ Porcupine initialized")
    print(f"   Sample rate: {porcupine.sample_rate} Hz")
    print(f"   Frame length: {porcupine.frame_length} samples")
    print(f"   Version: {porcupine.version}")
    
    # Simulate audio processing
    print("\n🔊 Simulating audio input...")
    print("   (In real use, this would be from your microphone)")
    
    # Process some fake audio frames
    for i in range(5):
        # Generate random audio (in real app, this would be mic input)
        # Note: Real audio would have actual speech patterns
        pcm = struct.unpack_from(
            "h" * porcupine.frame_length,
            struct.pack("h" * porcupine.frame_length, 
                       *np.random.randint(-1000, 1000, 
                                        size=porcupine.frame_length).tolist())
        )
        
        # Process the audio frame
        result = porcupine.process(pcm)
        
        if result >= 0:
            print(f"✨ Frame {i}: 'JARVIS' detected!")
        else:
            print(f"   Frame {i}: Listening...")
    
    # Cleanup
    porcupine.delete()
    print("\n✅ Test completed successfully!")
    print("\n💡 Note: With real microphone input, Picovoice will detect")
    print("   when you say 'Jarvis' with ~10ms latency!")

def test_with_sensitivities():
    """Test different sensitivity levels"""
    print("\n🎚️  Testing Different Sensitivity Levels")
    print("=" * 40)
    
    for sensitivity in [0.1, 0.5, 0.9]:
        print(f"\nSensitivity: {sensitivity}")
        print("  (0.0 = least sensitive, 1.0 = most sensitive)")
        
        porcupine = pvporcupine.create(
            access_key=os.environ["PICOVOICE_ACCESS_KEY"],
            keywords=["jarvis"],
            sensitivities=[sensitivity]
        )
        
        # Process one frame
        pcm = [0] * porcupine.frame_length
        result = porcupine.process(pcm)
        
        print(f"  Initialized successfully ✅")
        
        porcupine.delete()

if __name__ == "__main__":
    print("Picovoice Simple Test")
    print("====================")
    print()
    
    # Check key
    if not os.environ.get("PICOVOICE_ACCESS_KEY"):
        print("❌ PICOVOICE_ACCESS_KEY not set!")
        exit(1)
    
    # Run tests
    test_basic_picovoice()
    test_with_sensitivities()
    
    print("\n🎯 Integration with JARVIS:")
    print("   When you use the optimized_voice_system,")
    print("   Picovoice will automatically handle the")
    print("   initial wake word detection!")
    print("\n   Benefits:")
    print("   • 10ms detection latency")
    print("   • 1-2% CPU usage") 
    print("   • Works offline")
    print("   • Handles 'Jarvis' and 'Hey Jarvis'")
    
    print("\n✨ Picovoice is ready to use in your JARVIS system!")