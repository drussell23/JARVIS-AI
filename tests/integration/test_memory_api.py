#!/usr/bin/env python3
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


Test memory optimization API endpoint
"""
import requests
import json
import time

def test_memory_optimization():
    """Test the memory optimization endpoint"""
    base_url = "http://localhost:8000"
    
    print("Testing Memory Optimization API")
    print("=" * 50)
    
    # First check if server is running
    try:
        response = requests.get(f"{base_url}/")
        print(f"✓ Server is running on {base_url}")
    except requests.exceptions.ConnectionError:
        print(f"❌ Server is not running on {base_url}")
        print("Please start the server first: python3 start_system.py")
        return
    
    # Test memory status endpoint
    try:
        print("\n1. Testing memory status...")
        response = requests.get(f"{base_url}/memory/status")
        if response.status_code == 200:
            data = response.json()
            print(f"   Current memory: {data.get('percent_used', 'N/A'):.1f}%")
            print(f"   State: {data.get('state', 'N/A')}")
        else:
            print(f"   Error: {response.status_code}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test optimization endpoint
    try:
        print("\n2. Testing memory optimization...")
        response = requests.post(f"{base_url}/chat/optimize-memory")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Success: {data.get('success', False)}")
            print(f"   Initial memory: {data.get('initial_memory_percent', 'N/A'):.1f}%")
            print(f"   Final memory: {data.get('final_memory_percent', 'N/A'):.1f}%")
            print(f"   Memory freed: {data.get('memory_freed_mb', 0):.0f} MB")
            
            if data.get('actions_taken'):
                print("\n   Actions taken:")
                for action in data['actions_taken']:
                    print(f"   - {action['strategy']}: {action['freed_mb']:.0f} MB")
            
            print(f"\n   Current mode: {data.get('current_mode', 'N/A')}")
            print(f"   Can use LangChain: {data.get('can_use_langchain', False)}")
        else:
            print(f"   Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test chat endpoint to see current mode
    try:
        print("\n3. Testing chat endpoint...")
        payload = {"user_input": "What is 2+2?"}
        response = requests.post(f"{base_url}/chat", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Response: {data.get('response', 'N/A')}")
            print(f"   Mode: {data.get('chatbot_mode', 'N/A')}")
        else:
            print(f"   Error: {response.status_code}")
            
    except Exception as e:
        print(f"   Error: {e}")


if __name__ == "__main__":
    test_memory_optimization()