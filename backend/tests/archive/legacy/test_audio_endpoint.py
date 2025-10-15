#!/usr/bin/env python3
"""
Test the TTS audio endpoint
"""

import requests
import os

# Test the audio endpoint
url = "http://localhost:8000/audio/speak/Hello from JARVIS"
print(f"Testing audio endpoint: {url}")

try:
    response = requests.get(url)
    print(f"Status: {response.status_code}")
    print(f"Content-Type: {response.headers.get('Content-Type')}")
    
    if response.status_code == 200:
        # Save the audio file
        with open("test_jarvis_audio.mp3", "wb") as f:
            f.write(response.content)
        print("✅ Audio file saved as test_jarvis_audio.mp3")
        
        # Check file size
        file_size = os.path.getsize("test_jarvis_audio.mp3")
        print(f"File size: {file_size} bytes")
        
        # Play the audio (macOS)
        os.system("afplay test_jarvis_audio.mp3")
        
    else:
        print(f"❌ Error: {response.text}")
        
except Exception as e:
    print(f"❌ Connection error: {e}")
    print("Make sure the backend is running on port 8000")