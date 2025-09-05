#!/usr/bin/env python3
"""Test JARVIS monitoring through the API"""

import asyncio
import os
import httpx
import json

async def test_jarvis_monitoring():
    """Test JARVIS monitoring through the API"""
    
    # First check JARVIS status
    async with httpx.AsyncClient() as client:
        print("🔍 Checking JARVIS status...")
        response = await client.get("http://localhost:8000/voice/jarvis/status")
        status = response.json()
        print(f"✅ JARVIS Status: {json.dumps(status, indent=2)}")
        
        # Process monitoring command
        print("\n🎙️ Sending monitoring command...")
        response = await client.post(
            "http://localhost:8000/voice/jarvis/command",
            json={"text": "start monitoring my screen"}
        )
        
        result = response.json()
        print(f"\n📨 Response: {json.dumps(result, indent=2)}")
        
        # Check if failed
        if "Failed to start" in result.get('response', ''):
            print("\n❌ Monitoring command failed!")
        else:
            print("\n✅ Monitoring command successful!")

if __name__ == "__main__":
    asyncio.run(test_jarvis_monitoring())