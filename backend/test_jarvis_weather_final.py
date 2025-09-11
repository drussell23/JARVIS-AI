#!/usr/bin/env python3
"""
Final test of JARVIS weather integration
Tests the complete flow through JARVIS API
"""

import asyncio
import os
import sys
import json
from datetime import datetime

# Add backend to path
sys.path.insert(0, '/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')

async def test_jarvis_weather():
    """Test JARVIS weather command end-to-end"""
    print("🤖 Testing JARVIS Weather Integration")
    print("="*60)
    
    try:
        # Import and setup JARVIS
        from api.jarvis_voice_api import JARVISVoiceAPI, JARVISCommand
        from api.jarvis_factory import set_app_state
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
        from system_control.macos_controller import MacOSController
        from system_control.weather_system_config import initialize_weather_system
        from types import SimpleNamespace
        
        # Initialize components
        print("1. Initializing components...")
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise Exception("No ANTHROPIC_API_KEY found")
            
        vision = ClaudeVisionAnalyzer(api_key)
        controller = MacOSController()
        weather_system = initialize_weather_system(vision, controller)
        
        # Set up app state
        app_state = SimpleNamespace(
            vision_analyzer=vision,
            weather_system=weather_system
        )
        set_app_state(app_state)
        print("✅ Components initialized")
        
        # Create JARVIS API
        print("\n2. Creating JARVIS API...")
        jarvis_api = JARVISVoiceAPI()
        print("✅ JARVIS API created")
        
        # Test weather command
        print("\n3. Testing weather command...")
        command = JARVISCommand(text="What's the weather for today?")
        
        # Process command
        start_time = datetime.now()
        result = await jarvis_api.process_command(command)
        end_time = datetime.now()
        
        # Analyze results
        print("\n📊 Results:")
        print(f"   Status: {result.get('status', 'unknown')}")
        print(f"   Mode: {result.get('mode', 'unknown')}")
        print(f"   Time taken: {(end_time - start_time).total_seconds():.2f}s")
        
        response = result.get('response', '')
        print(f"\n💬 JARVIS Response:")
        print(f"   {response}")
        
        # Check for success indicators
        success_indicators = [
            "weather app" in response.lower(),
            "°f" in response.lower() or "degrees" in response.lower(),
            len(response) > 50,
            "error" not in response.lower(),
            "failed" not in response.lower()
        ]
        
        passed = sum(success_indicators)
        print(f"\n✅ Passed {passed}/5 quality checks:")
        print(f"   - Mentions weather app: {success_indicators[0]}")
        print(f"   - Contains temperature: {success_indicators[1]}")
        print(f"   - Adequate response length: {success_indicators[2]}")
        print(f"   - No error messages: {success_indicators[3]}")
        print(f"   - No failure indicators: {success_indicators[4]}")
        
        # Overall result
        if passed >= 3:
            print("\n✅ JARVIS Weather Integration: PASSED")
            return True
        else:
            print("\n❌ JARVIS Weather Integration: FAILED")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Check prerequisites
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("❌ Please set ANTHROPIC_API_KEY environment variable")
        exit(1)
        
    # Run test
    success = asyncio.run(test_jarvis_weather())
    exit(0 if success else 1)