#!/usr/bin/env python3
"""
Test WebSocket Router Integration
Tests the TypeScript WebSocket router with Python backend integration
"""

import asyncio
import json
import websockets
import logging
from datetime import datetime
import sys
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
WS_URL = "ws://localhost:8001"
TEST_ROUTES = [
    "/ws/vision",
    "/ws/voice",
    "/ws/automation",
    "/ws/ml_audio",
    "/ws"
]

class WebSocketTester:
    def __init__(self):
        self.results = []
        self.errors = []
        
    async def test_connection(self, path: str) -> dict:
        """Test WebSocket connection to a specific path"""
        result = {
            "path": path,
            "connected": False,
            "capabilities": None,
            "messages": [],
            "errors": []
        }
        
        try:
            uri = f"{WS_URL}{path}"
            logger.info(f"Testing connection to {uri}")
            
            async with websockets.connect(uri) as websocket:
                # Wait for connection acknowledgment
                ack = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                ack_data = json.loads(ack)
                
                result["connected"] = True
                result["capabilities"] = ack_data.get("capabilities", [])
                logger.info(f"✅ Connected to {path}")
                logger.info(f"   Capabilities: {result['capabilities']}")
                
                # Test ping
                await self.test_ping(websocket, result)
                
                # Test route-specific messages
                if "/vision" in path:
                    await self.test_vision_messages(websocket, result)
                elif "/voice" in path:
                    await self.test_voice_messages(websocket, result)
                elif path == "/ws":
                    await self.test_general_messages(websocket, result)
                
        except asyncio.TimeoutError:
            error = f"Connection timeout for {path}"
            logger.error(f"❌ {error}")
            result["errors"].append(error)
        except Exception as e:
            error = f"Connection failed for {path}: {str(e)}"
            logger.error(f"❌ {error}")
            result["errors"].append(error)
            
        return result
    
    async def test_ping(self, websocket, result: dict):
        """Test ping/pong functionality"""
        try:
            ping_msg = {"type": "ping", "timestamp": datetime.now().isoformat()}
            await websocket.send(json.dumps(ping_msg))
            
            pong = await asyncio.wait_for(websocket.recv(), timeout=2.0)
            pong_data = json.loads(pong)
            
            if pong_data.get("type") == "pong":
                logger.info("   ✅ Ping/pong working")
                result["messages"].append("ping_pong_success")
            else:
                logger.warning(f"   ⚠️  Unexpected ping response: {pong_data}")
                
        except Exception as e:
            error = f"Ping test failed: {str(e)}"
            logger.error(f"   ❌ {error}")
            result["errors"].append(error)
    
    async def test_vision_messages(self, websocket, result: dict):
        """Test vision-specific messages"""
        test_messages = [
            {
                "type": "get_status",
                "expected_response": "system_status"
            },
            {
                "type": "set_monitoring_interval",
                "interval": 3000,
                "expected_response": "config_updated"
            },
            {
                "type": "vision_command",
                "command": "test vision",
                "expected_response": "vision_response"
            }
        ]
        
        for test_msg in test_messages:
            try:
                await websocket.send(json.dumps(test_msg))
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_data = json.loads(response)
                
                if response_data.get("type") == test_msg["expected_response"]:
                    logger.info(f"   ✅ {test_msg['type']} -> {response_data['type']}")
                    result["messages"].append(f"{test_msg['type']}_success")
                else:
                    logger.warning(f"   ⚠️  {test_msg['type']} got {response_data['type']}")
                    
            except Exception as e:
                error = f"{test_msg['type']} failed: {str(e)}"
                logger.error(f"   ❌ {error}")
                result["errors"].append(error)
    
    async def test_voice_messages(self, websocket, result: dict):
        """Test voice-specific messages"""
        test_messages = [
            {
                "type": "get_voices",
                "expected_response": "available_voices"
            },
            {
                "type": "voice_command",
                "command": "test command",
                "expected_response": "command_result"
            }
        ]
        
        for test_msg in test_messages:
            try:
                await websocket.send(json.dumps(test_msg))
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_data = json.loads(response)
                
                if response_data.get("type") == test_msg["expected_response"]:
                    logger.info(f"   ✅ {test_msg['type']} -> {response_data['type']}")
                    result["messages"].append(f"{test_msg['type']}_success")
                else:
                    logger.warning(f"   ⚠️  {test_msg['type']} got {response_data['type']}")
                    
            except Exception as e:
                error = f"{test_msg['type']} failed: {str(e)}"
                logger.error(f"   ❌ {error}")
                result["errors"].append(error)
    
    async def test_general_messages(self, websocket, result: dict):
        """Test general messages"""
        test_messages = [
            {
                "type": "echo",
                "data": "test",
                "expected_response": "echo_response"
            },
            {
                "type": "health_check",
                "expected_response": "health_check_response"
            }
        ]
        
        for test_msg in test_messages:
            try:
                await websocket.send(json.dumps(test_msg))
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_data = json.loads(response)
                
                if response_data.get("type") == test_msg["expected_response"]:
                    logger.info(f"   ✅ {test_msg['type']} -> {response_data['type']}")
                    result["messages"].append(f"{test_msg['type']}_success")
                else:
                    logger.warning(f"   ⚠️  {test_msg['type']} got {response_data['type']}")
                    
            except Exception as e:
                error = f"{test_msg['type']} failed: {str(e)}"
                logger.error(f"   ❌ {error}")
                result["errors"].append(error)
    
    async def run_tests(self):
        """Run all WebSocket tests"""
        logger.info("🚀 Starting WebSocket Router Tests")
        logger.info(f"   Testing URL: {WS_URL}")
        logger.info(f"   Routes: {TEST_ROUTES}")
        logger.info("")
        
        # First check if the server is running
        try:
            async with websockets.connect(f"{WS_URL}/ws") as ws:
                await ws.close()
        except Exception as e:
            logger.error(f"❌ WebSocket server not running at {WS_URL}")
            logger.error(f"   Error: {e}")
            logger.error("   Please start the WebSocket router first:")
            logger.error("   cd backend/websocket && npm start")
            return
        
        # Test each route
        for route in TEST_ROUTES:
            result = await self.test_connection(route)
            self.results.append(result)
            logger.info("")  # Blank line between tests
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        logger.info("=" * 60)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 60)
        
        total_routes = len(self.results)
        connected_routes = sum(1 for r in self.results if r["connected"])
        total_errors = sum(len(r["errors"]) for r in self.results)
        
        logger.info(f"Total routes tested: {total_routes}")
        logger.info(f"Successfully connected: {connected_routes}/{total_routes}")
        logger.info(f"Total errors: {total_errors}")
        logger.info("")
        
        # Details per route
        for result in self.results:
            status = "✅" if result["connected"] else "❌"
            logger.info(f"{status} {result['path']}:")
            
            if result["connected"]:
                logger.info(f"   Capabilities: {', '.join(result['capabilities'])}")
                logger.info(f"   Successful tests: {len(result['messages'])}")
            
            if result["errors"]:
                logger.info(f"   Errors: {len(result['errors'])}")
                for error in result["errors"]:
                    logger.info(f"      - {error}")
        
        # Overall result
        logger.info("")
        if connected_routes == total_routes and total_errors == 0:
            logger.info("✅ All tests passed!")
        else:
            logger.info("❌ Some tests failed. Check the errors above.")


async def main():
    """Main test runner"""
    tester = WebSocketTester()
    await tester.run_tests()


if __name__ == "__main__":
    # Check for test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--wait":
        logger.info("Waiting 5 seconds for server to start...")
        time.sleep(5)
    
    asyncio.run(main())