#!/usr/bin/env python3
"""Test WebSocket to Python integration"""

import asyncio
import json
import websocket
import threading
import time

def on_message(ws, message):
    print(f"Received: {message}")
    data = json.loads(message)
    print(f"Message type: {data.get('type')}")
    
def on_error(ws, error):
    print(f"Error: {error}")
    
def on_close(ws, close_status_code, close_msg):
    print(f"Connection closed: {close_status_code} - {close_msg}")
    
def on_open(ws):
    print("Connection opened")
    
    # Test general WebSocket endpoint
    test_message = {
        "type": "chat",
        "message": "Hello from test client!",
        "timestamp": "2025-08-28T12:00:00Z"
    }
    
    ws.send(json.dumps(test_message))
    print(f"Sent: {test_message}")
    
    # Test echo
    echo_message = {
        "type": "echo",
        "data": "Testing echo functionality"
    }
    ws.send(json.dumps(echo_message))
    print(f"Sent: {echo_message}")
    
    # Test health check
    health_message = {
        "type": "health_check"
    }
    ws.send(json.dumps(health_message))
    print(f"Sent: {health_message}")

def test_general_websocket():
    """Test general WebSocket endpoint"""
    print("\n=== Testing General WebSocket Endpoint ===")
    ws = websocket.WebSocketApp("ws://localhost:8001/ws",
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    
    # Run for 5 seconds
    timer = threading.Timer(5, ws.close)
    timer.start()
    
    ws.run_forever()
    timer.cancel()

def test_vision_websocket():
    """Test vision WebSocket endpoint"""
    print("\n=== Testing Vision WebSocket Endpoint ===")
    
    ws = websocket.WebSocketApp("ws://localhost:8001/ws/vision",
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    
    def on_vision_open(ws):
        print("Vision connection opened")
        
        # Test monitoring interval
        monitor_message = {
            "type": "set_monitoring_interval",
            "interval": 5000
        }
        ws.send(json.dumps(monitor_message))
        print(f"Sent: {monitor_message}")
        
        # Request workspace analysis
        analysis_message = {
            "type": "request_workspace_analysis"
        }
        ws.send(json.dumps(analysis_message))
        print(f"Sent: {analysis_message}")
    
    ws.on_open = on_vision_open
    
    # Run for 5 seconds
    timer = threading.Timer(5, ws.close)
    timer.start()
    
    ws.run_forever()
    timer.cancel()

def test_voice_websocket():
    """Test voice WebSocket endpoint"""
    print("\n=== Testing Voice WebSocket Endpoint ===")
    
    ws = websocket.WebSocketApp("ws://localhost:8001/ws/voice",
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    
    def on_voice_open(ws):
        print("Voice connection opened")
        
        # Test start streaming
        start_message = {
            "type": "start_stream",
            "stream_id": "test-stream-123",
            "config": {
                "sample_rate": 16000,
                "channels": 1
            }
        }
        ws.send(json.dumps(start_message))
        print(f"Sent: {start_message}")
    
    ws.on_open = on_voice_open
    
    # Run for 3 seconds
    timer = threading.Timer(3, ws.close)
    timer.start()
    
    ws.run_forever()
    timer.cancel()

if __name__ == "__main__":
    print("Testing WebSocket to Python Integration")
    print("=" * 50)
    
    # Ensure WebSocket router is running
    try:
        # Test general endpoint
        test_general_websocket()
        time.sleep(1)
        
        # Test vision endpoint
        test_vision_websocket()
        time.sleep(1)
        
        # Test voice endpoint
        test_voice_websocket()
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")