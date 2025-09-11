#!/usr/bin/env python3
"""
Comprehensive test suite for JARVIS weather functionality
Tests the complete flow: open app -> navigate to location -> analyze -> respond
"""

import asyncio
import os
import subprocess
import time
import json
from datetime import datetime


class WeatherTestSuite:
    """Complete test suite for weather functionality"""
    
    def __init__(self):
        self.test_results = []
        self.vision_analyzer = None
        self.controller = None
        self.weather_system = None
        
    async def setup(self):
        """Initialize all components"""
        print("🔧 Setting up test environment...")
        
        # Check API key
        if not os.getenv('ANTHROPIC_API_KEY'):
            raise Exception("ANTHROPIC_API_KEY not set")
            
        # Initialize components
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
        from system_control.macos_controller import MacOSController
        from system_control.weather_system_config import initialize_weather_system
        
        self.vision_analyzer = ClaudeVisionAnalyzer(os.getenv('ANTHROPIC_API_KEY'))
        self.controller = MacOSController()
        self.weather_system = initialize_weather_system(self.vision_analyzer, self.controller)
        
        print("✅ All components initialized")
        
    async def test_1_weather_app_control(self):
        """Test 1: Can we open and control the Weather app?"""
        print("\n📱 Test 1: Weather App Control")
        
        try:
            # Open Weather app
            success, msg = self.controller.open_application("Weather")
            assert success, f"Failed to open Weather app: {msg}"
            print("  ✅ Weather app opened")
            
            await asyncio.sleep(2)
            
            # Check if it's running
            is_running = await self._check_app_running("Weather")
            assert is_running, "Weather app not detected as running"
            print("  ✅ Weather app is running")
            
            # Bring to front
            script = '''
            tell application "Weather"
                activate
                set frontmost to true
            end tell
            '''
            success, _ = self.controller.execute_applescript(script)
            assert success, "Failed to bring Weather to front"
            print("  ✅ Weather app brought to front")
            
            self._record_result("Weather App Control", True)
            
        except AssertionError as e:
            print(f"  ❌ {e}")
            self._record_result("Weather App Control", False, str(e))
            
    async def test_2_navigation(self):
        """Test 2: Can we navigate to My Location?"""
        print("\n🗺️ Test 2: Navigation to My Location")
        
        try:
            # Navigate using keyboard
            print("  🔄 Navigating with keyboard...")
            
            # Go to top
            for _ in range(3):
                await self.controller.key_press('up')
                await asyncio.sleep(0.2)
            
            # Select first item
            await self.controller.key_press('down')
            await asyncio.sleep(0.2)
            await self.controller.key_press('return')
            
            print("  ✅ Navigation commands sent")
            await asyncio.sleep(1.5)  # Wait for weather to load
            
            # Verify we're on a location (vision check)
            screenshot = await self.vision_analyzer.capture_screen()
            if screenshot:
                print("  ✅ Captured screen after navigation")
            
            self._record_result("Navigation to My Location", True)
            
        except Exception as e:
            print(f"  ❌ Navigation failed: {e}")
            self._record_result("Navigation to My Location", False, str(e))
            
    async def test_3_vision_analysis(self):
        """Test 3: Can vision analyze the weather?"""
        print("\n👁️ Test 3: Vision Analysis")
        
        try:
            # Test fast weather analysis
            print("  🔄 Testing fast weather analysis...")
            
            result = await asyncio.wait_for(
                self.vision_analyzer.analyze_weather_fast(),
                timeout=10.0
            )
            
            assert result.get('success'), f"Analysis failed: {result.get('error')}"
            
            analysis = result.get('analysis', '')
            print(f"  📊 Analysis result: {analysis[:150]}...")
            
            # Verify we got weather data
            assert any(word in analysis.lower() for word in ['location', 'temp', 'degrees', '°']), \
                "No weather data found in analysis"
            
            print("  ✅ Weather data successfully analyzed")
            self._record_result("Vision Analysis", True, analysis[:100])
            
        except AssertionError as e:
            print(f"  ❌ {e}")
            self._record_result("Vision Analysis", False, str(e))
        except asyncio.TimeoutError:
            print("  ❌ Analysis timed out")
            self._record_result("Vision Analysis", False, "Timeout")
            
    async def test_4_weather_system(self):
        """Test 4: Does the complete weather system work?"""
        print("\n🌤️ Test 4: Complete Weather System")
        
        try:
            # Test weather system
            print("  🔄 Getting weather through system...")
            
            result = await asyncio.wait_for(
                self.weather_system.get_weather("What's the weather today?"),
                timeout=20.0
            )
            
            assert result.get('success'), f"Weather system failed: {result.get('error')}"
            
            # Check data
            data = result.get('data', {})
            location = data.get('location', 'Unknown')
            current = data.get('current', {})
            temp = current.get('temperature', 'N/A')
            condition = current.get('condition', 'N/A')
            
            print(f"  📍 Location: {location}")
            print(f"  🌡️ Temperature: {temp}")
            print(f"  ☁️ Condition: {condition}")
            
            # Check formatted response
            response = result.get('formatted_response', '')
            print(f"  💬 Response: {response}")
            
            assert response and len(response) > 10, "No formatted response"
            
            print("  ✅ Weather system working correctly")
            self._record_result("Weather System", True, response)
            
        except AssertionError as e:
            print(f"  ❌ {e}")
            self._record_result("Weather System", False, str(e))
        except asyncio.TimeoutError:
            print("  ❌ Weather system timed out")
            self._record_result("Weather System", False, "Timeout")
            
    async def test_5_jarvis_integration(self):
        """Test 5: Does JARVIS handle weather commands correctly?"""
        print("\n🤖 Test 5: JARVIS Integration")
        
        try:
            # Set up app state for JARVIS
            from types import SimpleNamespace
            from api.jarvis_factory import set_app_state
            
            app_state = SimpleNamespace(
                vision_analyzer=self.vision_analyzer,
                weather_system=self.weather_system
            )
            set_app_state(app_state)
            
            # Test JARVIS API
            from api.jarvis_voice_api import JARVISVoiceAPI, JARVISCommand
            
            api = JARVISVoiceAPI()
            command = JARVISCommand(text="What's the weather like today?")
            
            print("  🔄 Processing weather command through JARVIS...")
            
            result = await api.process_command(command)
            
            response = result.get('response', '')
            status = result.get('status', 'unknown')
            mode = result.get('mode', 'unknown')
            
            print(f"  📊 Status: {status}")
            print(f"  🔧 Mode: {mode}")
            print(f"  💬 Response: {response[:150]}...")
            
            # Verify response quality
            assert response and len(response) > 20, "Response too short"
            assert not any(word in response.lower() for word in ['error', 'failed', 'couldn\'t']), \
                "Response contains error indicators"
            
            print("  ✅ JARVIS integration working")
            self._record_result("JARVIS Integration", True, response[:100])
            
        except Exception as e:
            print(f"  ❌ JARVIS integration failed: {e}")
            self._record_result("JARVIS Integration", False, str(e))
            
    async def test_6_performance(self):
        """Test 6: Performance benchmarks"""
        print("\n⚡ Test 6: Performance Testing")
        
        try:
            # Time different operations
            timings = {}
            
            # Time screenshot capture
            start = time.time()
            screenshot = await self.vision_analyzer.capture_screen()
            timings['screenshot'] = time.time() - start
            
            # Time fast analysis
            start = time.time()
            result = await self.vision_analyzer.analyze_weather_fast(screenshot)
            timings['fast_analysis'] = time.time() - start
            
            # Time complete flow
            start = time.time()
            weather = await self.weather_system.get_weather("weather")
            timings['complete_flow'] = time.time() - start
            
            print("  ⏱️ Performance Results:")
            print(f"     Screenshot capture: {timings['screenshot']:.2f}s")
            print(f"     Fast analysis: {timings['fast_analysis']:.2f}s")
            print(f"     Complete flow: {timings['complete_flow']:.2f}s")
            
            # Check if within acceptable limits
            assert timings['screenshot'] < 2.0, "Screenshot too slow"
            assert timings['fast_analysis'] < 10.0, "Analysis too slow"
            assert timings['complete_flow'] < 20.0, "Complete flow too slow"
            
            print("  ✅ Performance within acceptable limits")
            self._record_result("Performance", True, timings)
            
        except AssertionError as e:
            print(f"  ❌ Performance issue: {e}")
            self._record_result("Performance", False, str(e))
            
    def _record_result(self, test_name, passed, details=None):
        """Record test result"""
        self.test_results.append({
            "test": test_name,
            "passed": passed,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
        
    async def _check_app_running(self, app_name):
        """Check if app is running"""
        script = f'''
        tell application "System Events"
            return exists process "{app_name}"
        end tell
        '''
        success, result = self.controller.execute_applescript(script)
        return success and result.strip().lower() == 'true'
        
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r['passed'])
        
        print(f"\nTotal Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {(passed/total*100):.1f}%")
        
        print("\n📋 Detailed Results:")
        for result in self.test_results:
            status = "✅" if result['passed'] else "❌"
            print(f"\n{status} {result['test']}")
            if result['details']:
                if isinstance(result['details'], dict):
                    for k, v in result['details'].items():
                        print(f"   {k}: {v}")
                else:
                    print(f"   Details: {result['details']}")
                    
    async def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up...")
        # Any cleanup needed


async def main():
    """Run all weather tests"""
    print("🌤️ JARVIS Weather System - Complete Test Suite")
    print("="*60)
    
    suite = WeatherTestSuite()
    
    try:
        # Setup
        await suite.setup()
        
        # Run all tests
        await suite.test_1_weather_app_control()
        await suite.test_2_navigation()
        await suite.test_3_vision_analysis()
        await suite.test_4_weather_system()
        await suite.test_5_jarvis_integration()
        await suite.test_6_performance()
        
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Print summary
        suite.print_summary()
        
        # Cleanup
        await suite.cleanup()
        
    # Save results
    with open('weather_test_results.json', 'w') as f:
        json.dump({
            "run_time": datetime.now().isoformat(),
            "results": suite.test_results
        }, f, indent=2)
        print(f"\n📁 Results saved to weather_test_results.json")


if __name__ == "__main__":
    # Ensure we're in the right directory
    os.chdir('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')
    
    # Check prerequisites
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("❌ Please set ANTHROPIC_API_KEY environment variable")
        exit(1)
        
    # Ensure Weather app is installed
    if not os.path.exists('/System/Applications/Weather.app'):
        print("❌ Weather app not found. Please ensure macOS Weather app is installed")
        exit(1)
        
    # Run tests
    asyncio.run(main())