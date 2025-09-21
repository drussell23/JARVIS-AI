#!/usr/bin/env python3
"""
Test the new POST audio endpoint with base64
"""

import requests
import json
import base64
from io import BytesIO

# Test the POST endpoint
url = "http://localhost:8000/audio/speak"
data = {"text": "Hello, this is JARVIS speaking with improved audio."}

print(f"Testing POST audio endpoint: {url}")

try:
    response = requests.post(url, json=data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            # Extract base64 audio
            audio_data = result['audio']
            print("✅ Audio generated successfully")
            print(f"Data URL preview: {audio_data[:50]}...")
            
            # Save and play
            if audio_data.startswith('data:audio/mpeg;base64,'):
                audio_b64 = audio_data.split(',')[1]
                audio_bytes = base64.b64decode(audio_b64)
                
                with open("test_jarvis_post.mp3", "wb") as f:
                    f.write(audio_bytes)
                print("✅ Audio saved as test_jarvis_post.mp3")
                
                # Play it
                import os
                os.system("afplay test_jarvis_post.mp3")
        else:
            print(f"❌ Error: {result.get('error')}")
    else:
        print(f"❌ HTTP Error: {response.text}")
        
except Exception as e:
    print(f"❌ Connection error: {e}")