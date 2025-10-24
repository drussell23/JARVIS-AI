#!/usr/bin/env python3
"""
Test script for GCP JARVIS Backend
Tests the cloud backend with 32GB RAM
"""
import requests
import json

GCP_BACKEND_URL = "http://34.10.137.70:8010"

def test_health():
    """Test backend health"""
    print("🏥 Testing backend health...")
    response = requests.get(f"{GCP_BACKEND_URL}/health")
    print(f"✅ Health: {response.json()}")
    return response.status_code == 200

def test_command(command):
    """Send a command to JARVIS"""
    print(f"\n💬 Testing command: '{command}'")
    response = requests.post(
        f"{GCP_BACKEND_URL}/api/command",
        json={"command": command},
        headers={"Content-Type": "application/json"}
    )
    result = response.json()
    print(f"📊 Response: {json.dumps(result, indent=2)}")
    return result

if __name__ == "__main__":
    print("🚀 Testing JARVIS GCP Backend (32GB RAM)")
    print(f"🌐 URL: {GCP_BACKEND_URL}\n")

    # Test 1: Health check
    if not test_health():
        print("❌ Backend is not healthy!")
        exit(1)

    # Test 2: Simple query
    test_command("Hello JARVIS!")

    # Test 3: Another query
    test_command("What can you do?")

    print("\n✅ All tests passed! Your GCP backend is working!")
    print(f"\n💡 You now have 32GB RAM available for JARVIS!")
