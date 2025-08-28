#!/usr/bin/env python3
"""
Integrate macOS Weather app with Claude Vision for weather information
No hardcoding - uses dynamic app control and vision analysis
"""

import os
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def integrate_weather_app_vision():
    """Integrate Weather app opening and vision analysis for weather commands"""
    
    print("🌤️  Integrating macOS Weather App with Vision")
    print("=" * 60)
    
    # First, update the claude_command_interpreter to handle weather as a vision command
    interpreter_path = backend_dir / "system_control" / "claude_command_interpreter.py"
    
    print(f"\n📝 Updating command interpreter for weather app...")
    
    with open(interpreter_path, 'r') as f:
        content = f.read()
    
    # Remove the weather exclusion we added earlier
    weather_exclusion = """        
        # Don't interpret weather queries as system commands
        if any(word in voice_input.lower() for word in ['weather', 'temperature', 'forecast', 'rain', 'snow']):
            logger.info("Weather query detected - should be handled by conversation")
            # Return low confidence to trigger fallback to conversation
            return CommandIntent(
                action="unknown",
                target="",
                parameters={},
                category=CommandCategory.SYSTEM,
                confidence=0.1,  # Low confidence triggers fallback
                raw_command=voice_input,
                interpretation="Weather query - route to conversation"
            )"""
    
    if weather_exclusion in content:
        content = content.replace(weather_exclusion, '')
        print("✅ Removed weather exclusion")
    
    # Update the system prompt to include weather app vision flow
    old_prompt = """        Important: Weather requests (e.g., "what's the weather today") should be categorized as "web" with action="web_search" and the query as the target."""
    
    new_prompt = """        Important: Weather requests (e.g., "what's the weather today") should be handled as a workflow:
        1. Category: "workflow" 
        2. Action: "check_weather_app"
        3. This will open the Weather app and use vision to read the information"""
    
    if old_prompt in content:
        content = content.replace(old_prompt, new_prompt)
    else:
        # Add the new guidance after the common actions
        marker = "- Workflows: morning_routine, development_setup, meeting_prep"
        if marker in content:
            content = content.replace(marker, 
                marker + "\n        " + new_prompt.strip())
    
    print("✅ Updated command interpretation for weather app workflow")
    
    # Write the updated interpreter
    with open(interpreter_path, 'w') as f:
        f.write(content)
    
    # Now create a weather app workflow handler
    workflow_handler = '''#!/usr/bin/env python3
"""
Weather App Vision Workflow
Opens Weather app and uses Claude Vision to read weather information
"""

import asyncio
import logging
from typing import Dict, Any
import time

logger = logging.getLogger(__name__)

class WeatherAppVisionWorkflow:
    """Workflow to check weather using macOS Weather app and vision"""
    
    def __init__(self, controller, vision_handler):
        self.controller = controller
        self.vision_handler = vision_handler
        
    async def check_weather_with_vision(self) -> Dict[str, Any]:
        """
        Open Weather app and read weather information using vision
        
        Returns:
            Dict with weather information from vision analysis
        """
        try:
            # Step 1: Open Weather app
            logger.info("Opening Weather app...")
            success, message = self.controller.open_application("Weather")
            
            if not success:
                # Try alternative names
                success, message = self.controller.open_application("Weather.app")
                if not success:
                    success, message = await self.controller.open_app_intelligently("Weather")
            
            if not success:
                return {
                    'success': False,
                    'message': "I couldn't open the Weather app. Please make sure it's installed."
                }
            
            # Step 2: Wait for app to load
            logger.info("Waiting for Weather app to load...")
            await asyncio.sleep(2.5)  # Give it time to load weather data
            
            # Step 3: Use vision to analyze the weather information
            logger.info("Analyzing weather information with vision...")
            
            # Prepare vision query specifically for weather
            vision_params = {
                'query': 'Please read and describe the weather information shown in the Weather app. Include the current temperature, conditions, and forecast for today. Be specific about the numbers and weather conditions you see.'
            }
            
            # Use vision handler to analyze
            result = await self.vision_handler.describe_screen(vision_params)
            
            if result.success:
                # Extract weather info from vision response
                weather_description = result.description
                
                # If the vision response doesn't seem to contain weather info, try again with more specific prompt
                if 'temperature' not in weather_description.lower() and 'degrees' not in weather_description.lower():
                    vision_params['query'] = 'I see the Weather app is open. Please read the current temperature, weather conditions, and today\\'s forecast that are displayed. What are the specific numbers for temperature and any precipitation chances?'
                    result = await self.vision_handler.describe_screen(vision_params)
                    weather_description = result.description
                
                return {
                    'success': True,
                    'message': weather_description,
                    'source': 'weather_app_vision'
                }
            else:
                return {
                    'success': False,
                    'message': "I could see the Weather app but couldn't read the weather information clearly."
                }
                
        except Exception as e:
            logger.error(f"Weather app vision workflow error: {e}")
            return {
                'success': False,
                'message': f"I encountered an error checking the weather: {str(e)}"
            }

# Integration function
async def execute_weather_app_workflow(controller, vision_handler) -> str:
    """Execute the weather app vision workflow and return formatted response"""
    workflow = WeatherAppVisionWorkflow(controller, vision_handler)
    result = await workflow.check_weather_with_vision()
    
    if result['success']:
        # Format the response nicely
        return f"According to the Weather app, {result['message']}"
    else:
        return result['message']
'''
    
    # Save the workflow handler
    workflow_path = backend_dir / "workflows" / "weather_app_vision.py"
    workflow_path.parent.mkdir(exist_ok=True)
    
    with open(workflow_path, 'w') as f:
        f.write(workflow_handler)
    
    print(f"✅ Created weather app vision workflow: {workflow_path}")
    
    # Update the command interpreter to handle the weather workflow
    print("\n📝 Adding weather workflow execution to command interpreter...")
    
    with open(interpreter_path, 'r') as f:
        content = f.read()
    
    # Add import for the workflow
    import_marker = "from .vision_action_handler import get_vision_action_handler"
    if import_marker in content:
        content = content.replace(import_marker, 
            import_marker + "\nfrom workflows.weather_app_vision import execute_weather_app_workflow")
    
    # Update the workflow execution to handle weather
    old_workflow_method = '''    async def _execute_workflow_command(self, intent: CommandIntent) -> CommandResult:
        """Execute workflow commands"""
        workflow_name = intent.target or intent.action
        success, message = await self.controller.execute_workflow(workflow_name)
        return CommandResult(success=success, message=message)'''
    
    new_workflow_method = '''    async def _execute_workflow_command(self, intent: CommandIntent) -> CommandResult:
        """Execute workflow commands"""
        workflow_name = intent.target or intent.action
        
        # Special handling for weather workflow
        if workflow_name == "check_weather_app" or "weather" in intent.raw_command.lower():
            try:
                # Use the weather app vision workflow
                message = await execute_weather_app_workflow(self.controller, self.vision_handler)
                return CommandResult(success=True, message=message)
            except Exception as e:
                logger.error(f"Weather workflow error: {e}")
                # Fallback to regular workflow
        
        # Regular workflow execution
        success, message = await self.controller.execute_workflow(workflow_name)
        return CommandResult(success=success, message=message)'''
    
    if old_workflow_method in content:
        content = content.replace(old_workflow_method, new_workflow_method)
        print("✅ Updated workflow execution for weather")
    
    with open(interpreter_path, 'w') as f:
        f.write(content)
    
    # Finally, update the intelligent command handler to route weather to system/workflow
    handler_path = backend_dir / "voice" / "intelligent_command_handler.py"
    
    print("\n📝 Updating intelligent handler for weather app routing...")
    
    with open(handler_path, 'r') as f:
        content = f.read()
    
    # Update the weather detection to route to system instead of conversation
    old_weather_check = '''            # Check for weather-related queries - route to conversation
            if any(word in text.lower() for word in ['weather', 'temperature', 'forecast', 'rain', 'snow', 'sunny', 'cloudy', 'hot', 'cold']):
                logger.info(f"Detected weather query, routing to conversation handler")
                return await self._handle_conversation(text, classification)'''
    
    new_weather_check = '''            # Check for weather-related queries - route to system for Weather app workflow
            if any(word in text.lower() for word in ['weather', 'temperature', 'forecast', 'rain', 'snow', 'sunny', 'cloudy', 'hot', 'cold']):
                logger.info(f"Detected weather query, routing to system handler for Weather app workflow")
                # Override classification to ensure it goes to system handler
                classification['type'] = 'system'
                classification['confidence'] = 0.9
                return await self._handle_system_command(text, classification)'''
    
    if old_weather_check in content:
        content = content.replace(old_weather_check, new_weather_check)
        print("✅ Updated weather routing to use system handler")
    
    with open(handler_path, 'w') as f:
        f.write(content)
    
    print("\n✅ Weather app vision integration complete!")
    print("\n🌤️  How it works:")
    print("   1. You ask: 'What's the weather today?'")
    print("   2. JARVIS opens the macOS Weather app")
    print("   3. Claude Vision reads the weather information")
    print("   4. You get real, current weather data!")
    print("\n📱 No API keys needed - uses your Mac's Weather app!")
    print("🤖 No hardcoding - dynamically reads whatever is shown!")

if __name__ == "__main__":
    integrate_weather_app_vision()