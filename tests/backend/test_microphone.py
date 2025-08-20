#!/usr/bin/env python3
"""Test microphone and speech recognition setup"""

import speech_recognition as sr
import sys

def test_microphone():
    """Test microphone functionality"""
    print("🎤 Testing Microphone Setup...")
    
    # Initialize recognizer
    recognizer = sr.Recognizer()
    
    # List available microphones
    print("\nAvailable Microphones:")
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"  [{index}] {name}")
    
    # Use default microphone
    try:
        with sr.Microphone() as source:
            print("\n✅ Microphone initialized successfully!")
            
            # Adjust for ambient noise
            print("🔊 Adjusting for ambient noise... (please be quiet)")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print("✅ Noise calibration complete")
            
            # Test recording
            print("\n🎤 Say something (you have 5 seconds)...")
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                print("✅ Audio captured successfully!")
                
                # Try to recognize
                print("\n🔄 Processing speech...")
                try:
                    text = recognizer.recognize_google(audio)
                    print(f"✅ You said: '{text}'")
                    
                    # Check for JARVIS wake word
                    if "jarvis" in text.lower():
                        print("🎯 Wake word detected! JARVIS is listening.")
                    
                except sr.UnknownValueError:
                    print("❌ Could not understand audio")
                except sr.RequestError as e:
                    print(f"❌ Error with speech recognition: {e}")
                    
            except sr.WaitTimeoutError:
                print("⏱️ No speech detected (timeout)")
                
    except Exception as e:
        print(f"❌ Microphone error: {e}")
        print("\nTroubleshooting:")
        print("1. Check microphone permissions")
        print("2. Ensure microphone is connected")
        print("3. Try: pip install pyaudio")
        return False
    
    print("\n✅ Microphone test complete!")
    return True

if __name__ == "__main__":
    print("JARVIS Microphone Test Utility")
    print("=" * 40)
    
    success = test_microphone()
    
    if success:
        print("\n🎉 Your microphone is ready for JARVIS!")
        print("Try saying 'Hey JARVIS' in the web interface.")
    else:
        print("\n❌ Please fix the microphone issues before using JARVIS.")
        sys.exit(1)