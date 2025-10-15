#!/usr/bin/env python3
"""Test JARVIS audio endpoints and diagnose audio playback issues."""

import requests
import subprocess
import os
import tempfile
import time
from datetime import datetime

def test_backend_audio():
    """Test backend audio generation endpoints"""
    print("🎵 Testing JARVIS Audio System")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    # Test 1: Basic GET endpoint
    print("\n1. Testing GET endpoint...")
    test_text = "Hello Sir, this is JARVIS testing the audio system."
    url = f"{base_url}/audio/speak/{requests.utils.quote(test_text)}"
    
    try:
        response = requests.get(url, timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Content-Type: {response.headers.get('Content-Type', 'None')}")
        print(f"   Content-Length: {len(response.content)} bytes")
        
        if response.status_code == 200:
            # Save and play the audio
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name
            
            print(f"   ✅ Audio saved to: {tmp_path}")
            
            # Try to play it
            print("   🔊 Playing audio...")
            subprocess.run(["afplay", tmp_path])
            
            os.unlink(tmp_path)
        else:
            print(f"   ❌ Failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: POST endpoint
    print("\n2. Testing POST endpoint...")
    test_text_long = "Testing POST method. Sir, this is a longer message to ensure the POST endpoint is working correctly with Daniel's British voice."
    
    try:
        response = requests.post(
            f"{base_url}/audio/speak",
            json={"text": test_text_long},
            timeout=10
        )
        print(f"   Status: {response.status_code}")
        print(f"   Content-Type: {response.headers.get('Content-Type', 'None')}")
        
        if response.status_code == 200:
            if response.headers.get('Content-Type', '').startswith('audio/'):
                # Direct audio response
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    tmp.write(response.content)
                    tmp_path = tmp.name
                
                print(f"   ✅ Audio saved to: {tmp_path}")
                print("   🔊 Playing audio...")
                subprocess.run(["afplay", tmp_path])
                os.unlink(tmp_path)
            else:
                # JSON response with base64 audio
                data = response.json()
                if data.get('success') and data.get('audio'):
                    print("   ✅ Received base64 audio data")
                    # Extract base64 data
                    audio_data = data['audio']
                    if audio_data.startswith('data:audio'):
                        audio_data = audio_data.split(',')[1]
                    
                    import base64
                    audio_bytes = base64.b64decode(audio_data)
                    
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                        tmp.write(audio_bytes)
                        tmp_path = tmp.name
                    
                    print(f"   ✅ Audio decoded and saved")
                    print("   🔊 Playing audio...")
                    subprocess.run(["afplay", tmp_path])
                    os.unlink(tmp_path)
        else:
            print(f"   ❌ Failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Test Daniel voice directly
    print("\n3. Testing Daniel voice directly...")
    try:
        result = subprocess.run(
            ["say", "-v", "Daniel", "Testing Daniel's British voice directly"],
            capture_output=True
        )
        if result.returncode == 0:
            print("   ✅ Daniel voice is working")
        else:
            print(f"   ❌ Daniel voice error: {result.stderr.decode()}")
    except Exception as e:
        print(f"   ❌ Error testing Daniel voice: {e}")
    
    # Test 4: List available voices
    print("\n4. Available voices on system:")
    try:
        result = subprocess.run(["say", "-v", "?"], capture_output=True, text=True)
        voices = result.stdout.strip().split('\n')
        british_voices = [v for v in voices if 'en_GB' in v or 'British' in v.lower()]
        print(f"   Found {len(british_voices)} British voices:")
        for voice in british_voices[:5]:  # Show first 5
            print(f"   - {voice.split()[0]}")
    except Exception as e:
        print(f"   ❌ Error listing voices: {e}")
    
    # Test 5: Check CORS headers
    print("\n5. Checking CORS headers...")
    try:
        response = requests.options(f"{base_url}/audio/speak/test", timeout=5)
        cors_headers = {
            'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin', 'Not set'),
            'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods', 'Not set'),
            'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers', 'Not set')
        }
        for header, value in cors_headers.items():
            print(f"   {header}: {value}")
    except Exception as e:
        print(f"   ❌ Error checking CORS: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Audio diagnostics complete")

if __name__ == "__main__":
    test_backend_audio()