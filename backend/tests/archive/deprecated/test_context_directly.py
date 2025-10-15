#!/usr/bin/env python3
"""
Direct test of context awareness functionality
"""

import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_jarvis_context():
    """Test JARVIS WebSocket with context awareness"""
    ws_url = "ws://localhost:8000/voice/jarvis/stream"
    
    try:
        logger.info(f"Connecting to {ws_url}")
        async with websockets.connect(ws_url) as websocket:
            logger.info("Connected to JARVIS WebSocket")
            
            # Wait for connection confirmation
            response = await websocket.recv()
            data = json.loads(response)
            logger.info(f"Connection response: {data}")
            
            # Send test command
            test_command = {
                "type": "command",
                "text": "open safari and search for dogs"
            }
            
            logger.info(f"Sending command: {test_command['text']}")
            await websocket.send(json.dumps(test_command))
            
            # Receive responses
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    data = json.loads(response)
                    logger.info(f"Response: {json.dumps(data, indent=2)}")
                    
                    # Check if this is the final response
                    if data.get("type") == "response":
                        break
                        
                except asyncio.TimeoutError:
                    logger.info("Timeout waiting for response")
                    break
                    
    except Exception as e:
        logger.error(f"Error: {e}")
        
if __name__ == "__main__":
    logger.info("Testing JARVIS context awareness...")
    logger.info("Make sure your screen is LOCKED before running this test")
    logger.info("Starting test in 3 seconds...")
    import time
    time.sleep(3)
    
    asyncio.run(test_jarvis_context())