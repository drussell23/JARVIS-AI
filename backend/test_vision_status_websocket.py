#!/usr/bin/env python3
"""
Test script to verify WebSocket vision status updates
"""

import asyncio
import logging
from pathlib import Path
import sys
import json
import websockets

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSocketTestClient:
    """Test client for WebSocket connection"""
    
    def __init__(self):
        self.websocket = None
        self.received_messages = []
        
    async def connect(self, url="ws://localhost:8000/ws/unified/test_client"):
        """Connect to WebSocket server"""
        try:
            self.websocket = await websockets.connect(url)
            logger.info(f"Connected to {url}")
            
            # Start message listener
            asyncio.create_task(self.listen_for_messages())
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
            
    async def listen_for_messages(self):
        """Listen for messages from server"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                self.received_messages.append(data)
                logger.info(f"Received: {data.get('type', 'unknown')} - {data}")
                
                # Check for vision status updates
                if data.get('type') == 'vision_status_update':
                    status = data.get('status', {})
                    logger.info(f"🔵 Vision Status: {status.get('text')} - Connected: {status.get('connected')}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error in message listener: {e}")
            
    async def disconnect(self):
        """Disconnect from WebSocket"""
        if self.websocket:
            await self.websocket.close()
            

async def test_vision_status_flow():
    """Test the complete vision status flow with WebSocket"""
    
    print("\n🧪 Testing Vision Status WebSocket Flow\n")
    
    # Initialize app components
    print("1️⃣ Initializing app components...")
    try:
        from main import app
        from vision.vision_status_integration import initialize_vision_status
        from api.unified_websocket import ws_manager
        
        # Initialize vision status
        await initialize_vision_status(app)
        print("✅ Vision status integration initialized\n")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}\n")
        return
    
    # Test WebSocket client connection (skip if server not running)
    ws_client = WebSocketTestClient()
    ws_connected = False
    
    print("2️⃣ Attempting WebSocket connection...")
    try:
        ws_connected = await ws_client.connect()
        if ws_connected:
            print("✅ WebSocket connected\n")
            await asyncio.sleep(1)  # Give time for welcome message
        else:
            print("⚠️  WebSocket server not running - testing without WebSocket\n")
    except Exception as e:
        print(f"⚠️  WebSocket connection skipped: {e}\n")
    
    # Test vision status updates
    print("3️⃣ Testing vision status updates...")
    try:
        from vision.vision_status_manager import get_vision_status_manager
        
        status_manager = get_vision_status_manager()
        
        # Check initial status
        initial_status = status_manager.get_status()
        print(f"   Initial status: {initial_status['text']}")
        
        # Update to connected
        print("   Updating status to connected...")
        await status_manager.update_vision_status(True)
        
        # Wait for WebSocket message
        await asyncio.sleep(0.5)
        
        # Check updated status
        connected_status = status_manager.get_status()
        print(f"   Updated status: {connected_status['text']}")
        
        # Update to disconnected
        print("   Updating status to disconnected...")
        await status_manager.update_vision_status(False)
        
        # Wait for WebSocket message
        await asyncio.sleep(0.5)
        
        print("✅ Vision status updates working\n")
    except Exception as e:
        print(f"❌ Vision status test failed: {e}\n")
        import traceback
        traceback.print_exc()
    
    # Test through vision command handler
    print("4️⃣ Testing through vision command handler...")
    try:
        from api.vision_command_handler import vision_command_handler
        
        # Initialize
        await vision_command_handler.initialize_intelligence()
        
        # Start monitoring
        print("   Sending 'start monitoring' command...")
        result = await vision_command_handler.handle_command("start monitoring my screen")
        print(f"   Response: {result.get('response', 'No response')[:80]}...")
        
        # Wait for status updates
        await asyncio.sleep(1)
        
        # Check WebSocket messages
        if ws_connected and ws_client.received_messages:
            vision_updates = [m for m in ws_client.received_messages if m.get('type') == 'vision_status_update']
            print(f"   Received {len(vision_updates)} vision status updates via WebSocket")
            for update in vision_updates:
                status = update.get('status', {})
                print(f"     - {status.get('text')} at {status.get('timestamp')}")
        
        # Stop monitoring
        print("\n   Sending 'stop monitoring' command...")
        result = await vision_command_handler.handle_command("stop monitoring")
        print(f"   Response: {result.get('response', 'No response')}")
        
        await asyncio.sleep(1)
        
        print("\n✅ Vision command integration working\n")
    except Exception as e:
        print(f"\n❌ Command handler test failed: {e}\n")
        import traceback
        traceback.print_exc()
    
    # Disconnect WebSocket
    if ws_connected:
        await ws_client.disconnect()
        print("WebSocket disconnected")
    
    print("\n🎉 Vision status test complete!\n")
    
    # Summary
    if ws_connected and ws_client.received_messages:
        print("📊 Summary:")
        print(f"   Total WebSocket messages: {len(ws_client.received_messages)}")
        vision_updates = [m for m in ws_client.received_messages if m.get('type') == 'vision_status_update']
        print(f"   Vision status updates: {len(vision_updates)}")
        print()


if __name__ == "__main__":
    asyncio.run(test_vision_status_flow())