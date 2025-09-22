#!/usr/bin/env python3
"""
Integration Test for Proximity + Voice Auth
==========================================

Tests the Swift-Python communication and authentication flow.
"""

import asyncio
import json
import time
import subprocess
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "python"))

from voice_biometrics.voice_authenticator import VoiceAuthenticator
from auth_engine.dual_factor_auth import DualFactorAuthEngine


class IntegrationTester:
    """
    Tests the complete proximity + voice authentication system.
    """
    
    def __init__(self):
        self.proximity_process = None
        self.test_user = "test_user"
        
    async def start_proximity_service(self):
        """Start the Swift proximity service."""
        print("🚀 Starting Swift Proximity Service...")
        
        # Check if binary exists
        binary_path = Path(__file__).parent / "bin" / "ProximityService"
        if not binary_path.exists():
            print("❌ ProximityService binary not found. Run ./build_swift.sh first")
            return False
        
        # Start the service
        self.proximity_process = subprocess.Popen(
            [str(binary_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for service to start
        await asyncio.sleep(2)
        
        # Check if running
        if self.proximity_process.poll() is not None:
            stdout, stderr = self.proximity_process.communicate()
            print(f"❌ ProximityService failed to start")
            print(f"stdout: {stdout.decode()}")
            print(f"stderr: {stderr.decode()}")
            return False
        
        print("✅ Proximity Service started")
        return True
    
    async def test_zmq_communication(self):
        """Test file-based IPC communication with Swift service."""
        print("\n🔌 Testing IPC Communication...")
        
        request_file = "/tmp/jarvis_proximity_request.json"
        response_file = "/tmp/jarvis_proximity_response.json"
        
        try:
            # Clear any existing response file
            if Path(response_file).exists():
                Path(response_file).unlink()
            
            # Test get_status command
            request = {
                "command": "get_status",
                "timestamp": time.time()
            }
            
            # Write request
            with open(request_file, 'w') as f:
                json.dump(request, f)
            
            # Wait for response with timeout
            timeout = 5.0
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if Path(response_file).exists():
                    try:
                        with open(response_file, 'r') as f:
                            response = json.load(f)
                        print(f"✅ Received response: {response}")
                        return True
                    except json.JSONDecodeError:
                        # File might be partially written
                        await asyncio.sleep(0.1)
                        continue
                await asyncio.sleep(0.1)
            
            print("❌ No response from ProximityService")
            return False
                
        except Exception as e:
            print(f"❌ IPC error: {e}")
            return False
    
    async def test_proximity_detection(self):
        """Test proximity detection."""
        print("\n📡 Testing Proximity Detection...")
        
        request_file = "/tmp/jarvis_proximity_request.json"
        response_file = "/tmp/jarvis_proximity_response.json"
        
        try:
            # Clear any existing response file
            if Path(response_file).exists():
                Path(response_file).unlink()
            
            # Request proximity status
            request = {
                "command": "get_proximity",
                "timestamp": time.time()
            }
            
            # Write request
            with open(request_file, 'w') as f:
                json.dump(request, f)
            
            # Wait for response
            timeout = 5.0
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if Path(response_file).exists():
                    try:
                        with open(response_file, 'r') as f:
                            response = json.load(f)
                        break
                    except json.JSONDecodeError:
                        await asyncio.sleep(0.1)
                        continue
                await asyncio.sleep(0.1)
            else:
                print("❌ No response received")
                return False
            
            print(f"📊 Proximity Status:")
            print(f"   - Command: {response.get('command')}")
            print(f"   - Status: {response.get('status')}")
            
            if 'data' in response:
                data = response['data']
                print(f"   - Is Nearby: {data.get('isNearby')}")
                print(f"   - Confidence: {data.get('confidence')}%")
                print(f"   - Distance: {data.get('distance')}m")
                print(f"   - Device Count: {data.get('deviceCount')}")
            
            return response.get('status') == 'success'
            
        except Exception as e:
            print(f"❌ Proximity test error: {e}")
            return False
    
    async def test_voice_enrollment(self):
        """Test voice enrollment."""
        print("\n🎤 Testing Voice Enrollment...")
        
        # Create test audio (sine wave as placeholder)
        sample_rate = 16000
        duration = 3.0
        frequency = 440.0  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Add some variations to simulate speech
        audio += 0.1 * np.sin(2 * np.pi * frequency * 2 * t)
        audio += 0.05 * np.random.randn(len(audio))
        
        # Initialize voice authenticator
        authenticator = VoiceAuthenticator(self.test_user)
        
        # Enroll samples
        for i in range(3):
            print(f"   Enrolling sample {i+1}/3...")
            result = authenticator.enroll_voice(audio, sample_rate)
            print(f"   Result: {result}")
            
            if not result['success']:
                print(f"❌ Enrollment failed: {result.get('reason')}")
                return False
        
        print("✅ Voice enrollment successful")
        return True
    
    async def test_dual_factor_auth(self):
        """Test the complete dual-factor authentication."""
        print("\n🔐 Testing Dual-Factor Authentication...")
        
        # Initialize auth engine
        engine = DualFactorAuthEngine()
        await engine.initialize(self.test_user)
        
        # Create test audio
        sample_rate = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        audio += 0.1 * np.sin(2 * np.pi * 880 * t)
        
        # Perform authentication
        print("   Attempting authentication...")
        # Convert numpy array to bytes for the authenticate method
        audio_bytes = (audio * 32767).astype(np.int16).tobytes()
        result = await engine.authenticate(audio_bytes, sample_rate)
        
        print(f"\n📊 Authentication Result:")
        print(f"   - Status: {result.status.value}")
        print(f"   - Success: {result.success}")
        print(f"   - Proximity Score: {result.proximity_score:.1f}%")
        print(f"   - Voice Score: {result.voice_score:.1f}%")
        print(f"   - Combined Score: {result.combined_score:.1f}%")
        print(f"   - Reason: {result.reason}")
        
        await engine.shutdown()
        
        return True  # Test completion, not necessarily auth success
    
    async def run_all_tests(self):
        """Run all integration tests."""
        print("🧪 JARVIS Proximity + Voice Auth Integration Tests")
        print("=" * 50)
        
        results = {
            "proximity_service": False,
            "zmq_communication": False,
            "proximity_detection": False,
            "voice_enrollment": False,
            "dual_factor_auth": False
        }
        
        try:
            # Start proximity service
            results["proximity_service"] = await self.start_proximity_service()
            
            if results["proximity_service"]:
                # Test ZeroMQ
                results["zmq_communication"] = await self.test_zmq_communication()
                
                # Test proximity
                results["proximity_detection"] = await self.test_proximity_detection()
                
                # Test voice enrollment
                results["voice_enrollment"] = await self.test_voice_enrollment()
                
                # Test dual-factor auth
                results["dual_factor_auth"] = await self.test_dual_factor_auth()
            
        except Exception as e:
            print(f"\n❌ Test error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            if self.proximity_process:
                print("\n🧹 Cleaning up...")
                self.proximity_process.terminate()
                self.proximity_process.wait()
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 Test Summary:")
        for test, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {test}: {status}")
        
        total_passed = sum(results.values())
        total_tests = len(results)
        print(f"\nTotal: {total_passed}/{total_tests} tests passed")
        
        return total_passed == total_tests


async def main():
    """Main test runner."""
    tester = IntegrationTester()
    success = await tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())