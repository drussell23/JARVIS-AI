#!/usr/bin/env python3
"""
Comprehensive test case for screen monitoring command verification
Tests the complete flow from command input to video capture activation
"""

import asyncio
import json
import httpx
import websockets
import logging
from datetime import datetime
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MonitoringCommandTest:
    def __init__(self, base_url="http://localhost:8000", ws_url="ws://localhost:8000"):
        self.base_url = base_url
        self.ws_url = ws_url
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {"passed": 0, "failed": 0}
        }
        
    async def run_all_tests(self):
        """Run all monitoring command tests"""
        print("\n🧪 Starting Comprehensive Monitoring Command Tests\n")
        print("=" * 60)
        
        # Test 1: Check backend health
        await self.test_backend_health()
        
        # Test 2: Check JARVIS status
        await self.test_jarvis_status()
        
        # Test 3: Test direct API monitoring command
        await self.test_direct_api_command()
        
        # Test 4: Test WebSocket monitoring command
        await self.test_websocket_command()
        
        # Test 5: Check vision system state
        await self.test_vision_system_state()
        
        # Test 6: Verify command routing
        await self.test_command_routing()
        
        # Test 7: Check if video streaming is available
        await self.test_video_streaming_availability()
        
        # Print summary
        self.print_summary()
        
    async def test_backend_health(self):
        """Test 1: Check if backend is healthy"""
        test_name = "Backend Health Check"
        print(f"\n🔍 Test 1: {test_name}")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health")
                data = response.json()
                
                passed = response.status_code == 200 and data.get("status") == "healthy"
                self.test_results["tests"][test_name] = {
                    "passed": passed,
                    "status_code": response.status_code,
                    "response": data
                }
                
                if passed:
                    print(f"✅ PASSED: Backend is healthy")
                    print(f"   - Model: {data.get('model')}")
                    print(f"   - Vision Components: {data.get('components', {}).get('enhanced_vision')}")
                    self.test_results["summary"]["passed"] += 1
                else:
                    print(f"❌ FAILED: Backend health check failed")
                    self.test_results["summary"]["failed"] += 1
                    
        except Exception as e:
            print(f"❌ FAILED: {e}")
            self.test_results["tests"][test_name] = {"passed": False, "error": str(e)}
            self.test_results["summary"]["failed"] += 1
            
    async def test_jarvis_status(self):
        """Test 2: Check JARVIS status"""
        test_name = "JARVIS Status Check"
        print(f"\n🔍 Test 2: {test_name}")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/voice/jarvis/status")
                data = response.json()
                
                passed = data.get("status") == "online"
                self.test_results["tests"][test_name] = {
                    "passed": passed,
                    "response": data
                }
                
                if passed:
                    print(f"✅ PASSED: JARVIS is online")
                    print(f"   - Features: {', '.join(data.get('features', []))}")
                    self.test_results["summary"]["passed"] += 1
                else:
                    print(f"❌ FAILED: JARVIS is not online (status: {data.get('status')})")
                    self.test_results["summary"]["failed"] += 1
                    
        except Exception as e:
            print(f"❌ FAILED: {e}")
            self.test_results["tests"][test_name] = {"passed": False, "error": str(e)}
            self.test_results["summary"]["failed"] += 1
            
    async def test_direct_api_command(self):
        """Test 3: Send monitoring command via direct API"""
        test_name = "Direct API Monitoring Command"
        print(f"\n🔍 Test 3: {test_name}")
        
        try:
            async with httpx.AsyncClient() as client:
                # Send the monitoring command
                response = await client.post(
                    f"{self.base_url}/voice/jarvis/command",
                    json={"text": "start monitoring my screen"},
                    timeout=30.0
                )
                data = response.json()
                response_text = data.get("response", "")
                
                # Check if response indicates success
                monitoring_success = any([
                    "video capturing" in response_text.lower(),
                    "monitoring your screen" in response_text.lower() and "activated" in response_text.lower(),
                    "purple recording indicator" in response_text.lower(),
                    "native macos video" in response_text.lower()
                ])
                
                # Check for generic response (failure)
                is_generic = "Task completed successfully" in response_text and "Yes sir, I can see your screen" in response_text
                
                passed = monitoring_success and not is_generic
                
                self.test_results["tests"][test_name] = {
                    "passed": passed,
                    "response": response_text[:200] + "..." if len(response_text) > 200 else response_text,
                    "is_monitoring_response": monitoring_success,
                    "is_generic_response": is_generic
                }
                
                if passed:
                    print(f"✅ PASSED: Got proper monitoring response")
                    print(f"   - Response: {response_text[:100]}...")
                    self.test_results["summary"]["passed"] += 1
                else:
                    print(f"❌ FAILED: Got generic response instead of monitoring activation")
                    print(f"   - Response: {response_text[:100]}...")
                    self.test_results["summary"]["failed"] += 1
                    
        except Exception as e:
            print(f"❌ FAILED: {e}")
            self.test_results["tests"][test_name] = {"passed": False, "error": str(e)}
            self.test_results["summary"]["failed"] += 1
            
    async def test_websocket_command(self):
        """Test 4: Send monitoring command via WebSocket"""
        test_name = "WebSocket Monitoring Command"
        print(f"\n🔍 Test 4: {test_name}")
        
        try:
            ws_uri = f"{self.ws_url}/voice/jarvis/stream"
            async with websockets.connect(ws_uri) as websocket:
                # Wait for connection message
                connection_msg = await websocket.recv()
                logger.info(f"WebSocket connected: {connection_msg}")
                
                # Send monitoring command
                command = json.dumps({"text": "start monitoring my screen"})
                await websocket.send(command)
                
                # Wait for response
                response_msg = await websocket.recv()
                response_data = json.loads(response_msg)
                response_text = response_data.get("text", "")
                
                # Check response
                monitoring_success = any([
                    "video capturing" in response_text.lower(),
                    "monitoring your screen" in response_text.lower() and "activated" in response_text.lower(),
                    "purple recording indicator" in response_text.lower()
                ])
                
                is_generic = "Task completed successfully" in response_text
                
                passed = monitoring_success and not is_generic
                
                self.test_results["tests"][test_name] = {
                    "passed": passed,
                    "response": response_text[:200] + "..." if len(response_text) > 200 else response_text,
                    "is_monitoring_response": monitoring_success,
                    "is_generic_response": is_generic
                }
                
                if passed:
                    print(f"✅ PASSED: WebSocket command handled correctly")
                    self.test_results["summary"]["passed"] += 1
                else:
                    print(f"❌ FAILED: WebSocket returned generic response")
                    print(f"   - Response: {response_text[:100]}...")
                    self.test_results["summary"]["failed"] += 1
                    
        except Exception as e:
            print(f"❌ FAILED: {e}")
            self.test_results["tests"][test_name] = {"passed": False, "error": str(e)}
            self.test_results["summary"]["failed"] += 1
            
    async def test_vision_system_state(self):
        """Test 5: Check vision system state"""
        test_name = "Vision System State"
        print(f"\n🔍 Test 5: {test_name}")
        
        try:
            # Check if vision WebSocket monitoring status endpoint exists
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/vision/monitoring_status")
                
                if response.status_code == 200:
                    data = response.json()
                    passed = data.get("vision_available", False)
                    
                    self.test_results["tests"][test_name] = {
                        "passed": passed,
                        "monitoring_active": data.get("monitoring_active", False),
                        "connections": data.get("connections", 0),
                        "vision_available": data.get("vision_available", False)
                    }
                    
                    if passed:
                        print(f"✅ PASSED: Vision system available")
                        print(f"   - Monitoring Active: {data.get('monitoring_active')}")
                        print(f"   - Connections: {data.get('connections')}")
                        self.test_results["summary"]["passed"] += 1
                    else:
                        print(f"❌ FAILED: Vision system not available")
                        self.test_results["summary"]["failed"] += 1
                else:
                    print(f"⚠️  SKIPPED: Vision status endpoint not available")
                    self.test_results["tests"][test_name] = {"passed": False, "skipped": True}
                    
        except Exception as e:
            print(f"⚠️  SKIPPED: {e}")
            self.test_results["tests"][test_name] = {"passed": False, "error": str(e)}
            
    async def test_command_routing(self):
        """Test 6: Verify command routing logic"""
        test_name = "Command Routing Verification"
        print(f"\n🔍 Test 6: {test_name}")
        
        # Test different variations of monitoring commands
        test_commands = [
            "start monitoring my screen",
            "begin monitoring my screen",
            "monitor my screen",
            "watch my screen",
            "can you see my screen"  # This should NOT trigger monitoring
        ]
        
        results = []
        for command in test_commands:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/voice/jarvis/command",
                        json={"text": command},
                        timeout=10.0
                    )
                    data = response.json()
                    response_text = data.get("response", "")
                    
                    # Check if it's a monitoring response
                    is_monitoring = "video capturing" in response_text.lower() or "monitoring your screen" in response_text.lower()
                    
                    results.append({
                        "command": command,
                        "is_monitoring_response": is_monitoring,
                        "response_preview": response_text[:50] + "..."
                    })
                    
            except Exception as e:
                results.append({
                    "command": command,
                    "error": str(e)
                })
        
        # Check if monitoring commands are routed correctly
        monitoring_commands_correct = all(
            r.get("is_monitoring_response", False) 
            for r in results[:4]  # First 4 should be monitoring
        )
        
        # "can you see my screen" should NOT be a monitoring command
        non_monitoring_correct = not results[4].get("is_monitoring_response", True)
        
        passed = monitoring_commands_correct and non_monitoring_correct
        
        self.test_results["tests"][test_name] = {
            "passed": passed,
            "results": results
        }
        
        if passed:
            print(f"✅ PASSED: Command routing working correctly")
            self.test_results["summary"]["passed"] += 1
        else:
            print(f"❌ FAILED: Command routing issues detected")
            for r in results:
                print(f"   - '{r['command']}': {'monitoring' if r.get('is_monitoring_response') else 'regular'}")
            self.test_results["summary"]["failed"] += 1
            
    async def test_video_streaming_availability(self):
        """Test 7: Check if video streaming is available"""
        test_name = "Video Streaming Availability"
        print(f"\n🔍 Test 7: {test_name}")
        
        try:
            # First, send a command to start monitoring
            async with httpx.AsyncClient() as client:
                # Start monitoring
                response = await client.post(
                    f"{self.base_url}/voice/jarvis/command",
                    json={"text": "start monitoring my screen"},
                    timeout=30.0
                )
                
                # Wait a bit for video streaming to initialize
                await asyncio.sleep(2)
                
                # Check monitoring status
                status_response = await client.get(f"{self.base_url}/vision/monitoring_status")
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    monitoring_active = status_data.get("monitoring_active", False)
                    
                    passed = monitoring_active
                    
                    self.test_results["tests"][test_name] = {
                        "passed": passed,
                        "monitoring_active": monitoring_active,
                        "status": status_data
                    }
                    
                    if passed:
                        print(f"✅ PASSED: Video streaming is active")
                        self.test_results["summary"]["passed"] += 1
                    else:
                        print(f"❌ FAILED: Video streaming not active after command")
                        self.test_results["summary"]["failed"] += 1
                        
                    # Stop monitoring for cleanup
                    await client.post(
                        f"{self.base_url}/voice/jarvis/command",
                        json={"text": "stop monitoring my screen"}
                    )
                else:
                    print(f"⚠️  SKIPPED: Cannot verify video streaming status")
                    self.test_results["tests"][test_name] = {"passed": False, "skipped": True}
                    
        except Exception as e:
            print(f"❌ FAILED: {e}")
            self.test_results["tests"][test_name] = {"passed": False, "error": str(e)}
            self.test_results["summary"]["failed"] += 1
            
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total = self.test_results["summary"]["passed"] + self.test_results["summary"]["failed"]
        passed = self.test_results["summary"]["passed"]
        failed = self.test_results["summary"]["failed"]
        
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        
        if failed == 0:
            print("\n🎉 All tests passed! The monitoring command system is working correctly.")
        else:
            print("\n⚠️  Some tests failed. Check the details above for debugging.")
            
        # Save results to file
        with open("monitoring_test_results.json", "w") as f:
            json.dump(self.test_results, f, indent=2)
        print("\n💾 Detailed results saved to: monitoring_test_results.json")

async def main():
    """Run the monitoring command tests"""
    tester = MonitoringCommandTest()
    await tester.run_all_tests()

if __name__ == "__main__":
    # Run the tests
    asyncio.run(main())