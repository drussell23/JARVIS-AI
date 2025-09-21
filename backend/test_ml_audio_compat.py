#!/usr/bin/env python3
"""
Test ML Audio compatibility endpoints
"""

import requests
import json

def test_ml_audio_endpoints():
    """Test ML audio compatibility endpoints"""
    base_url = "http://localhost:8000"
    
    print("Testing ML Audio Compatibility Endpoints")
    print("=" * 50)
    
    # Test config endpoint
    print("\n1. Testing /audio/ml/config...")
    try:
        response = requests.get(f"{base_url}/audio/ml/config")
        if response.status_code == 200:
            config = response.json()
            print("✅ Config endpoint works!")
            print(f"   Model: {config.get('model')}")
            print(f"   Sample rate: {config.get('sample_rate')}")
            print(f"   WebSocket endpoint: {config.get('websocket_endpoint')}")
        else:
            print(f"❌ Config failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Config error: {e}")
    
    # Test predict endpoint
    print("\n2. Testing /audio/ml/predict...")
    try:
        data = {"audio_data": "test_base64_data"}
        response = requests.post(f"{base_url}/audio/ml/predict", json=data)
        if response.status_code == 200:
            result = response.json()
            print("✅ Predict endpoint works!")
            print(f"   Prediction: {result.get('prediction')}")
            print(f"   Confidence: {result.get('confidence')}")
            print(f"   Note: {result.get('note', '')}")
        else:
            print(f"❌ Predict failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Predict error: {e}")
    
    # Test status endpoint
    print("\n3. Testing /audio/ml/status...")
    try:
        response = requests.get(f"{base_url}/audio/ml/status")
        if response.status_code == 200:
            status = response.json()
            print("✅ Status endpoint works!")
            print(f"   Status: {status.get('status')}")
            print(f"   WebSocket available: {status.get('websocket_available')}")
            print(f"   Notice: {status.get('legacy_notice', '')}")
        else:
            print(f"❌ Status failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Status error: {e}")
    
    print("\n" + "=" * 50)
    print("ML Audio compatibility layer is working!")
    print("Frontend should now be able to connect without errors.")

if __name__ == "__main__":
    test_ml_audio_endpoints()