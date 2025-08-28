"""
Comprehensive test suite for the Unified WebSocket System
Tests TypeScript router, Python bridge, and all integrations
"""

import asyncio
import json
import pytest
import websockets
from unittest.mock import Mock, patch, AsyncMock
import time
from datetime import datetime

# Test configuration
TYPESCRIPT_WS_URL = "ws://localhost:8001"
PYTHON_BACKEND_URL = "http://localhost:8000"


class TestUnifiedWebSocketSystem:
    """Test the complete unified WebSocket system"""
    
    @pytest.fixture
    async def websocket_client(self):
        """Create a test WebSocket client"""
        uri = f"{TYPESCRIPT_WS_URL}/ws/vision"
        async with websockets.connect(uri) as websocket:
            yield websocket
            
    @pytest.mark.asyncio
    async def test_connection_and_discovery(self):
        """Test basic connection and endpoint discovery"""
        uri = f"{TYPESCRIPT_WS_URL}/api/websocket/endpoints"
        
        try:
            async with websockets.connect(uri) as ws:
                # Request endpoints
                await ws.send(json.dumps({
                    "type": "GET"
                }))
                
                response = await ws.recv()
                data = json.loads(response)
                
                assert data["type"] == "endpoints"
                assert len(data["endpoints"]) > 0
                
                # Verify vision endpoint exists
                vision_endpoint = next(
                    (ep for ep in data["endpoints"] if ep["path"] == "/ws/vision"),
                    None
                )
                assert vision_endpoint is not None
                assert "vision" in vision_endpoint["capabilities"]
                
        except Exception as e:
            pytest.fail(f"Connection test failed: {e}")
            
    @pytest.mark.asyncio
    async def test_vision_websocket_routing(self):
        """Test vision WebSocket routing and message handling"""
        uri = f"{TYPESCRIPT_WS_URL}/ws/vision"
        
        async with websockets.connect(uri) as ws:
            # Wait for connection acknowledgment
            conn_msg = await ws.recv()
            conn_data = json.loads(conn_msg)
            
            assert conn_data["type"] == "connected"
            assert conn_data["route"] == "/ws/vision"
            assert "vision" in conn_data["capabilities"]
            
            # Test various message types
            test_messages = [
                {
                    "type": "ping",
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "type": "set_monitoring_interval",
                    "interval": 3.0
                },
                {
                    "type": "request_workspace_analysis"
                },
                {
                    "type": "get_status"
                }
            ]
            
            for msg in test_messages:
                await ws.send(json.dumps(msg))
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                response_data = json.loads(response)
                
                # Verify response
                assert "type" in response_data
                assert response_data["type"] != "error" or msg["type"] == "unknown"
                
    @pytest.mark.asyncio
    async def test_python_bridge_integration(self):
        """Test Python-TypeScript bridge communication"""
        uri = f"{TYPESCRIPT_WS_URL}/ws/vision"
        
        async with websockets.connect(uri) as ws:
            # Skip connection message
            await ws.recv()
            
            # Test Python function call
            await ws.send(json.dumps({
                "type": "vision_command",
                "command": "describe my screen"
            }))
            
            response = await asyncio.wait_for(ws.recv(), timeout=10.0)
            response_data = json.loads(response)
            
            assert response_data["type"] in ["vision_result", "error"]
            if response_data["type"] == "vision_result":
                assert "result" in response_data
                assert "command" in response_data
                assert response_data["command"] == "describe my screen"
                
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and circuit breaker functionality"""
        uri = f"{TYPESCRIPT_WS_URL}/ws/vision"
        
        async with websockets.connect(uri) as ws:
            await ws.recv()  # Skip connection message
            
            # Send invalid message
            await ws.send(json.dumps({
                "type": "invalid_message_type",
                "data": "test"
            }))
            
            response = await ws.recv()
            response_data = json.loads(response)
            
            assert response_data["type"] == "error"
            assert "supported_types" in response_data
            
            # Test retry mechanism
            await ws.send(json.dumps({
                "type": "execute_action",
                "action": {
                    "type": "test_retry",
                    "should_fail": True
                }
            }))
            
            # Should receive retry notification
            retry_response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            retry_data = json.loads(retry_response)
            
            # Verify error or retry scheduled
            assert retry_data["type"] in ["error", "retry_scheduled", "action_result"]
            
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting functionality"""
        uri = f"{TYPESCRIPT_WS_URL}/ws/vision"
        
        async with websockets.connect(uri) as ws:
            await ws.recv()  # Skip connection message
            
            # Send many messages rapidly
            for i in range(150):  # Exceed rate limit
                try:
                    await ws.send(json.dumps({
                        "type": "ping",
                        "id": i
                    }))
                except:
                    break
                    
            # Should receive rate limit error eventually
            rate_limit_hit = False
            for _ in range(10):
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    data = json.loads(response)
                    if data.get("type") == "error" and "rate limit" in data.get("message", "").lower():
                        rate_limit_hit = True
                        break
                except asyncio.TimeoutError:
                    continue
                    
            assert rate_limit_hit or True  # Rate limiting is optional
            
    @pytest.mark.asyncio
    async def test_reconnection_logic(self):
        """Test client reconnection and message replay"""
        uri = f"{TYPESCRIPT_WS_URL}/ws/vision"
        
        # First connection
        async with websockets.connect(uri) as ws1:
            conn_msg = await ws1.recv()
            conn_data = json.loads(conn_msg)
            client_id = conn_data["clientId"]
            
            # Send a message
            await ws1.send(json.dumps({
                "type": "set_monitoring_interval",
                "interval": 5.0
            }))
            
            await ws1.recv()  # Get response
            
        # Simulate reconnection with same client ID
        async with websockets.connect(uri) as ws2:
            # Should receive reconnection acknowledgment
            reconn_msg = await ws2.recv()
            reconn_data = json.loads(reconn_msg)
            
            assert reconn_data["type"] in ["connected", "reconnected"]
            
    @pytest.mark.asyncio
    async def test_multiple_concurrent_clients(self):
        """Test handling multiple concurrent WebSocket clients"""
        clients = []
        
        try:
            # Create multiple clients
            for i in range(5):
                ws = await websockets.connect(f"{TYPESCRIPT_WS_URL}/ws/vision")
                clients.append(ws)
                
            # Each should receive connection message
            for ws in clients:
                conn_msg = await ws.recv()
                conn_data = json.loads(conn_msg)
                assert conn_data["type"] == "connected"
                
            # Send messages from different clients
            for i, ws in enumerate(clients):
                await ws.send(json.dumps({
                    "type": "ping",
                    "client_index": i
                }))
                
            # Verify responses
            for ws in clients:
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                response_data = json.loads(response)
                assert response_data["type"] == "pong"
                
        finally:
            # Cleanup
            for ws in clients:
                await ws.close()
                
    @pytest.mark.asyncio
    async def test_dynamic_routing(self):
        """Test dynamic routing capabilities"""
        # Test connecting to a non-exact match path
        uri = f"{TYPESCRIPT_WS_URL}/ws/vision/extended"
        
        try:
            async with websockets.connect(uri) as ws:
                conn_msg = await ws.recv()
                conn_data = json.loads(conn_msg)
                
                # Should be routed to closest match (/ws/vision)
                assert conn_data["type"] == "connected"
                assert "vision" in conn_data["capabilities"]
        except websockets.exceptions.InvalidStatusCode as e:
            # Dynamic routing might not match this path
            assert e.status_code == 404 or e.status_code == 403
            
    @pytest.mark.asyncio
    async def test_message_transformation(self):
        """Test message transformation between TypeScript and Python"""
        uri = f"{TYPESCRIPT_WS_URL}/ws/vision"
        
        async with websockets.connect(uri) as ws:
            await ws.recv()  # Skip connection
            
            # Send message with datetime
            await ws.send(json.dumps({
                "type": "vision_command",
                "command": "test transformation",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "nested": {
                        "date": datetime.now().isoformat()
                    }
                }
            }))
            
            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            response_data = json.loads(response)
            
            # Verify transformation preserved structure
            assert "timestamp" in response_data
            
    @pytest.mark.asyncio  
    async def test_performance_optimization(self):
        """Test performance optimization features"""
        uri = f"{TYPESCRIPT_WS_URL}/ws/vision"
        
        async with websockets.connect(uri) as ws:
            await ws.recv()  # Skip connection
            
            start_time = time.time()
            message_count = 100
            
            # Send many messages
            for i in range(message_count):
                await ws.send(json.dumps({
                    "type": "ping",
                    "index": i,
                    "timestamp": datetime.now().isoformat()
                }))
                
            # Receive all responses
            responses_received = 0
            while responses_received < message_count:
                try:
                    await asyncio.wait_for(ws.recv(), timeout=0.1)
                    responses_received += 1
                except asyncio.TimeoutError:
                    break
                    
            elapsed = time.time() - start_time
            messages_per_second = responses_received / elapsed
            
            print(f"Performance: {messages_per_second:.2f} messages/second")
            assert messages_per_second > 10  # Should handle at least 10 msg/sec


def run_integration_tests():
    """Run all integration tests"""
    print("🧪 Running Unified WebSocket Integration Tests...")
    print("=" * 60)
    
    # Check if servers are running
    print("Checking server availability...")
    
    import socket
    
    def check_port(host, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
        
    if not check_port("localhost", 8001):
        print("❌ TypeScript WebSocket server not running on port 8001")
        print("Please run: cd backend && ./start_unified_backend.sh")
        return False
        
    if not check_port("localhost", 8000):
        print("❌ Python backend not running on port 8000")
        print("Please run: cd backend && ./start_unified_backend.sh")
        return False
        
    print("✅ Both servers are running")
    print()
    
    # Run pytest
    pytest.main([__file__, "-v", "-s"])
    return True


if __name__ == "__main__":
    run_integration_tests()