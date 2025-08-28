#!/usr/bin/env python3
"""Quick backend test to verify ML Audio endpoints are working"""

import requests
import time

def test_backend():
    base_url = "http://localhost:8010"
    
    # Wait a bit for backend to start
    print("Waiting for backend to be ready...")
    for i in range(30):
        try:
            resp = requests.get(f"{base_url}/health", timeout=1)
            if resp.status_code == 200:
                print("✅ Backend is healthy!")
                break
        except:
            pass
        time.sleep(1)
        print(f"  Attempt {i+1}/30...")
    else:
        print("❌ Backend didn't start in 30 seconds")
        return
    
    # Test ML Audio endpoints
    ml_endpoints = [
        "/audio/ml/config",
        "/audio/ml/metrics",
        "/audio/ml/models",
        "/audio/ml/status",
    ]
    
    print("\nTesting ML Audio endpoints:")
    for endpoint in ml_endpoints:
        try:
            resp = requests.get(f"{base_url}{endpoint}", timeout=2)
            status = "✅" if resp.status_code == 200 else f"❌ {resp.status_code}"
            print(f"  {endpoint}: {status}")
        except Exception as e:
            print(f"  {endpoint}: ❌ {str(e)}")
    
    # Test model status endpoint
    print("\nTesting Model Status endpoint:")
    try:
        resp = requests.get(f"{base_url}/models/status", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            if 'data' in data:
                print(f"  ✅ Model loading status available")
                print(f"     Total models: {data['data'].get('total', 0)}")
                print(f"     Loaded: {data['data'].get('loaded', 0)}")
                print(f"     Failed: {data['data'].get('failed', 0)}")
        else:
            print(f"  ❌ Status code: {resp.status_code}")
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")

if __name__ == "__main__":
    test_backend()