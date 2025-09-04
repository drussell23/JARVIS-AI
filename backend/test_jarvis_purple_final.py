#!/usr/bin/env python3
"""
Final test for JARVIS purple indicator
"""

import requests
import json
import time

def test_monitoring():
    print("\n🟣 TESTING JARVIS PURPLE INDICATOR FINAL")
    print("=" * 60)
    
    # Test the monitoring command
    url = "http://localhost:8000/chat"
    
    print("1️⃣ Sending 'start monitoring my screen' command...")
    
    response = requests.post(url, json={"message": "start monitoring my screen"})
    
    if response.status_code == 200:
        result = response.json()
        message = result.get('message', '')
        print(f"\n✅ Response: {message}")
        
        if "purple recording indicator" in message.lower() and "successfully" in message.lower():
            print("\n🟣 SUCCESS! Purple indicator should be visible!")
            print("⏳ Waiting 10 seconds...")
            time.sleep(10)
            
            print("\n2️⃣ Sending 'stop monitoring' command...")
            response = requests.post(url, json={"message": "stop monitoring"})
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Stop response: {result.get('message', '')}")
            
        else:
            print(f"\n❌ Unexpected response: {message}")
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(f"Response: {response.text}")
    
    print("\n" + "=" * 60)
    print("Test complete!")

if __name__ == "__main__":
    test_monitoring()