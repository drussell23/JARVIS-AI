#!/usr/bin/env python3
"""
Test unified WebSocket endpoint with weather functionality
"""

import asyncio
import json
import websocket
import threading
import time

def test_unified_websocket():
    """Test the unified WebSocket endpoint"""
    
    def on_message(ws, message):
        print(f"Received: {message}")
        data = json.loads(message)
        if data.get('type') == 'connection_established':
            print("✅ Connected to unified WebSocket")
            print(f"Available handlers: {data.get('available_handlers', [])}")
            
            # Test weather command
            weather_cmd = {
                "type": "voice_command",
                "text": "what's the weather today"
            }
            print(f"\nSending weather command: {weather_cmd}")
            ws.send(json.dumps(weather_cmd))
        
        elif data.get('type') == 'voice_response':
            print(f"\n🎤 Voice Response:")
            print(f"   Status: {data.get('status')}")
            print(f"   Response: {data.get('response', '')[:200]}")
            if len(data.get('response', '')) > 200:
                print(f"   ... (truncated, total length: {len(data.get('response', ''))} chars)")
            print(f"   Command Type: {data.get('command_type')}")
            ws.close()
    
    def on_error(ws, error):
        print(f"Error: {error}")
    
    def on_close(ws, close_status_code, close_msg):
        print(f"WebSocket closed: {close_status_code} - {close_msg}")
    
    def on_open(ws):
        print("WebSocket connection opened")
    
    # Connect to unified WebSocket
    ws_url = "ws://localhost:8000/ws"
    ws = websocket.WebSocketApp(ws_url,
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)
    
    # Run WebSocket in thread
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()
    
    # Wait for test to complete
    time.sleep(30)  # Give weather command time to process
    
    # Force close if still open
    try:
        ws.close()
    except:
        pass
    
    print("\nTest completed!")

if __name__ == "__main__":
    print("Testing unified WebSocket endpoint...")
    test_unified_websocket()