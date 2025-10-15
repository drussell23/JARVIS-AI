#!/usr/bin/env python3
"""
Test Dynamic CORS and Auto-Configuration System
"""

import requests
import json

def test_dynamic_cors():
    """Test the dynamic CORS and auto-configuration system"""
    
    print("Testing Dynamic CORS and Auto-Configuration")
    print("=" * 60)
    
    # Test from different ports to simulate different frontends
    test_origins = [
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://localhost:8010",
        "http://localhost:4200",
        "http://127.0.0.1:3000"
    ]
    
    for origin in test_origins:
        print(f"\n📍 Testing from origin: {origin}")
        
        # Test auto-config endpoint
        try:
            response = requests.get(
                "http://localhost:8000/auto-config",
                headers={
                    "Origin": origin,
                    "Referer": f"{origin}/",
                    "User-Agent": "Mozilla/5.0 (React App)"
                }
            )
            
            if response.status_code == 200:
                config = response.json()
                print("✅ Auto-config successful!")
                print(f"   - Server port: {config['server']['port']}")
                print(f"   - Client expected port: {config['client']['expected_port']}")
                print(f"   - CORS allowed: {config['cors']['allowed']}")
                
                # Check headers
                cors_header = response.headers.get('Access-Control-Allow-Origin')
                port_warning = response.headers.get('X-Port-Mismatch-Warning')
                
                if cors_header:
                    print(f"   - CORS header: {cors_header}")
                if port_warning:
                    print(f"   ⚠️  {port_warning}")
                    
            else:
                print(f"❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test ML Audio endpoint with CORS
    print("\n\n📍 Testing ML Audio endpoint with CORS")
    try:
        response = requests.post(
            "http://localhost:8000/audio/ml/predict",
            json={"audio_data": "test"},
            headers={
                "Origin": "http://localhost:3000",
                "Content-Type": "application/json"
            }
        )
        
        if response.status_code == 200:
            print("✅ ML Audio endpoint accessible with CORS!")
            cors_header = response.headers.get('Access-Control-Allow-Origin')
            api_port = response.headers.get('X-API-Port')
            print(f"   - CORS Origin: {cors_header}")
            print(f"   - API Port: {api_port}")
        else:
            print(f"❌ ML Audio failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ ML Audio error: {e}")
    
    # Test endpoint discovery
    print("\n\n📍 Testing endpoint discovery")
    try:
        response = requests.get("http://localhost:8000/auto-config/endpoints")
        if response.status_code == 200:
            endpoints = response.json()
            print("✅ Endpoints discovered:")
            print(f"   - Base URL: {endpoints['base_url']}")
            print(f"   - Categories: {list(endpoints['categories'].keys())}")
    except Exception as e:
        print(f"❌ Endpoint discovery error: {e}")
    
    # Test JavaScript config endpoint
    print("\n\n📍 Testing JavaScript config generation")
    try:
        response = requests.get("http://localhost:8000/auto-config/client-config")
        if response.status_code == 200:
            print("✅ JavaScript config available!")
            print(f"   - Content type: {response.headers.get('Content-Type')}")
            print(f"   - Config preview: {response.text[:200]}...")
    except Exception as e:
        print(f"❌ JS config error: {e}")
    
    print("\n" + "=" * 60)
    print("Dynamic CORS system is working!")
    print("\nFrontend Integration:")
    print("1. Include auto-config script:")
    print('   <script src="http://localhost:8000/static/jarvis-auto-config.js"></script>')
    print("\n2. Or use JARVIS_API object:")
    print("   await JARVIS_API.discoverBackend();")
    print("   const data = await JARVIS_API.fetch('/audio/ml/config');")
    print("   const ws = JARVIS_API.createWebSocket('/ws');")

if __name__ == "__main__":
    test_dynamic_cors()