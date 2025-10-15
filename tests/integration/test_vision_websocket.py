#!/usr/bin/env python3
"""
Test script to verify Vision WebSocket connectivity
"""

import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_vision_websocket():
    """Test connection to vision WebSocket"""
    uri = "ws://localhost:8000/vision/ws/vision"
    
    logger.info(f"🔍 Testing Vision WebSocket at {uri}")
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("✅ Connected to Vision WebSocket!")
            
            # Wait for initial state
            message = await websocket.recv()
            data = json.loads(message)
            logger.info(f"📨 Received initial state: {data['type']}")
            
            if 'workspace' in data:
                logger.info(f"  Window count: {data['workspace'].get('window_count', 0)}")
                logger.info(f"  Focused app: {data['workspace'].get('focused_app', 'None')}")
            
            # Request workspace analysis
            logger.info("\n📡 Requesting workspace analysis...")
            await websocket.send(json.dumps({
                "type": "request_analysis"
            }))
            
            # Wait for response
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(message)
                logger.info(f"📨 Received response: {data['type']}")
                
                if data['type'] == 'workspace_analysis':
                    analysis = data.get('analysis', {})
                    logger.info(f"  Context: {analysis.get('context', 'N/A')}")
                    logger.info(f"  Focused task: {analysis.get('focused_task', 'N/A')}")
                    
            except asyncio.TimeoutError:
                logger.warning("⏱️  Timeout waiting for workspace analysis")
            
            logger.info("\n✅ Vision WebSocket test completed successfully!")
            
    except websockets.exceptions.WebSocketException as e:
        logger.error(f"❌ WebSocket error: {e}")
        logger.info("\nTroubleshooting:")
        logger.info("1. Make sure the backend is running: python backend/main.py")
        logger.info("2. Check that port 8000 is not blocked")
        logger.info("3. Verify ANTHROPIC_API_KEY is set in backend/.env")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")


async def test_backend_health():
    """Test if backend is running"""
    import aiohttp
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/health') as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ Backend is healthy: {data['status']}")
                    return True
                else:
                    logger.error(f"❌ Backend returned status {response.status}")
                    return False
    except Exception as e:
        logger.error(f"❌ Cannot connect to backend: {e}")
        return False


async def main():
    logger.info("🚀 Vision WebSocket Connection Test\n")
    
    # First check if backend is running
    if await test_backend_health():
        # Test vision WebSocket
        await test_vision_websocket()
    else:
        logger.error("\n❌ Backend is not running!")
        logger.info("Please start the backend first:")
        logger.info("  cd backend && python main.py")


if __name__ == "__main__":
    asyncio.run(main())