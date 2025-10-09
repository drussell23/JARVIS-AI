#!/usr/bin/env python3
"""Test CORS headers on audio endpoints."""

import requests

def test_cors_headers():
    """Test CORS headers on audio endpoints"""
    print("🔍 Testing CORS headers on audio endpoints")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    # Test 1: OPTIONS request (preflight)
    print("\n1. Testing OPTIONS request (preflight)...")
    try:
        headers = {
            'Origin': 'http://localhost:3000',
            'Access-Control-Request-Method': 'GET',
            'Access-Control-Request-Headers': 'content-type'
        }
        response = requests.options(f"{base_url}/audio/speak/test", headers=headers, timeout=5)
        print(f"   Status: {response.status_code}")
        print("   CORS Headers:")
        for header in ['Access-Control-Allow-Origin', 'Access-Control-Allow-Methods', 
                      'Access-Control-Allow-Headers', 'Access-Control-Allow-Credentials']:
            value = response.headers.get(header, 'Not set')
            print(f"   - {header}: {value}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: GET request with Origin header
    print("\n2. Testing GET request with Origin header...")
    try:
        headers = {
            'Origin': 'http://localhost:3000'
        }
        response = requests.get(
            f"{base_url}/audio/speak/Test audio", 
            headers=headers,
            timeout=5
        )
        print(f"   Status: {response.status_code}")
        print("   CORS Headers in response:")
        for header in ['Access-Control-Allow-Origin', 'Access-Control-Allow-Credentials']:
            value = response.headers.get(header, 'Not set')
            print(f"   - {header}: {value}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: POST request with Origin header
    print("\n3. Testing POST request with Origin header...")
    try:
        headers = {
            'Origin': 'http://localhost:3000',
            'Content-Type': 'application/json'
        }
        response = requests.post(
            f"{base_url}/audio/speak",
            json={"text": "Test audio"},
            headers=headers,
            timeout=5
        )
        print(f"   Status: {response.status_code}")
        print("   CORS Headers in response:")
        for header in ['Access-Control-Allow-Origin', 'Access-Control-Allow-Credentials']:
            value = response.headers.get(header, 'Not set')
            print(f"   - {header}: {value}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    test_cors_headers()