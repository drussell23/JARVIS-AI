#!/usr/bin/env python3
"""
Test JARVIS monitoring command with purple indicator
"""

import requests
import json
import time

def test_monitoring_command():
    print("\n🟣 Testing JARVIS Monitoring Command")
    print("=" * 60)
    
    # Send monitoring command to JARVIS
    url = "http://localhost:8000/chat"
    
    command = "start monitoring my screen"
    
    print(f"📤 Sending command: '{command}'")
    
    try:
        response = requests.post(
            url,
            json={"message": command}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Response received:")
            print(f"   Message: {result.get('message', 'No message')}")
            
            # Check if it's the correct monitoring response
            if "activated" in result.get('message', '').lower() and "purple" in result.get('message', '').lower():
                print("\n🟣 SUCCESS! Purple indicator should be visible!")
                print("⏳ Check your menu bar for the purple recording indicator")
            else:
                print("\n⚠️ Response doesn't mention purple indicator")
                
        else:
            print(f"❌ Error: Status {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error sending command: {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_monitoring_command()