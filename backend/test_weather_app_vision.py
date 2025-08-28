#!/usr/bin/env python3
"""
Test the Weather app vision workflow
"""

import asyncio
import os
from pathlib import Path
import sys

backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from system_control import MacOSController
from system_control.vision_action_handler import get_vision_action_handler
from workflows.weather_app_vision import WeatherAppVisionWorkflow

async def test_weather_app_workflow():
    """Test the complete weather app vision workflow"""
    
    print("🌤️  Testing Weather App Vision Workflow")
    print("=" * 50)
    
    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not set - required for vision")
        return
    
    # Initialize components
    controller = MacOSController()
    vision_handler = get_vision_action_handler()
    workflow = WeatherAppVisionWorkflow(controller, vision_handler)
    
    print("\n1️⃣ Testing Weather app opening...")
    success, message = controller.open_application("Weather")
    if success:
        print(f"✅ Weather app opened: {message}")
    else:
        print(f"❌ Failed to open Weather app: {message}")
        print("   Trying alternative approach...")
        success, message = await controller.open_app_intelligently("Weather")
        if success:
            print(f"✅ Weather app opened with intelligent approach")
        else:
            print(f"❌ Still couldn't open Weather app")
    
    print("\n2️⃣ Testing vision analysis...")
    print("   Waiting for app to load...")
    await asyncio.sleep(3)
    
    # Test direct vision
    vision_params = {
        'query': 'Please read and describe the weather information shown in the Weather app. Include the current temperature, conditions, and forecast for today.'
    }
    
    result = await vision_handler.describe_screen(vision_params)
    
    if result.success:
        print(f"✅ Vision analysis successful!")
        print(f"   Response: {result.description[:200]}...")
    else:
        print(f"❌ Vision analysis failed: {result.error}")
    
    print("\n3️⃣ Testing complete workflow...")
    workflow_result = await workflow.check_weather_with_vision()
    
    if workflow_result['success']:
        print(f"✅ Weather workflow successful!")
        print(f"   Weather info: {workflow_result['message']}")
    else:
        print(f"❌ Weather workflow failed: {workflow_result['message']}")
    
    print("\n" + "=" * 50)
    print("Test complete!")

if __name__ == "__main__":
    asyncio.run(test_weather_app_workflow())