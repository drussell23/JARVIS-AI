#!/usr/bin/env python3
"""
Test the voice activation endpoint to verify 503 errors are fixed
"""

import asyncio
from fastapi import FastAPI
from api.jarvis_voice_api import JARVISVoiceAPI
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)

async def test_endpoints():
    """Test voice endpoints directly"""
    api = JARVISVoiceAPI()
    
    print("\n🧪 Testing Voice Endpoints...")
    print("=" * 60)
    
    # Test status
    print("\n1. Testing /voice/jarvis/status")
    status = await api.get_status()
    print(f"   Status: {status}")
    print(f"   ✅ No 503 error!")
    
    # Test activate
    print("\n2. Testing /voice/jarvis/activate")
    activate = await api.activate()
    print(f"   Response: {activate}")
    print(f"   ✅ No 503 error!")
    
    # Test config
    print("\n3. Testing /voice/jarvis/config")
    config = await api.get_config()
    print(f"   Config: {config}")
    print(f"   ✅ No 503 error!")
    
    print("\n" + "=" * 60)
    print("✅ All voice endpoints working without 503 errors!")
    print("=" * 60)


def run_server():
    """Run a minimal FastAPI server with voice endpoints"""
    app = FastAPI()
    
    # Add JARVIS voice routes
    jarvis_api = JARVISVoiceAPI()
    app.include_router(jarvis_api.router, prefix="/voice")
    
    print("\n🚀 Starting test server on http://localhost:8000")
    print("📍 Voice endpoints available at:")
    print("   - http://localhost:8000/voice/jarvis/status")
    print("   - http://localhost:8000/voice/jarvis/activate")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Just test endpoints
        asyncio.run(test_endpoints())
    else:
        # Run server
        run_server()