#!/usr/bin/env python3
"""
Weather App Vision Workflow
Opens Weather app and uses Claude Vision to read weather information
"""

import asyncio
import logging
from typing import Dict, Any
import time
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.weather_response_parser import WeatherResponseParser

logger = logging.getLogger(__name__)

class WeatherAppVisionWorkflow:
    """Workflow to check weather using macOS Weather app and vision"""
    
    def __init__(self, controller, vision_handler):
        self.controller = controller
        self.vision_handler = vision_handler
        self.continuous_analyzer = None
        
        # Try to use continuous screen analyzer if available
        try:
            from vision.continuous_screen_analyzer import ContinuousScreenAnalyzer
            self.continuous_analyzer = ContinuousScreenAnalyzer(vision_handler)
        except:
            logger.info("Continuous screen analyzer not available")
    
    async def check_weather_with_continuous_vision(self) -> Dict[str, Any]:
        """
        Check weather using continuous screen monitoring
        This is more reliable when user switches windows
        """
        if not self.continuous_analyzer:
            # Fall back to regular method
            return await self.check_weather_with_vision()
        
        try:
            # Use continuous analyzer to query weather
            weather_info = await self.continuous_analyzer.query_screen_for_weather()
            
            if weather_info:
                return {
                    'success': True,
                    'message': weather_info,
                    'source': 'continuous_vision'
                }
            else:
                # Fall back to regular method
                return await self.check_weather_with_vision()
                
        except Exception as e:
            logger.error(f"Error with continuous vision: {e}")
            # Fall back to regular method
            return await self.check_weather_with_vision()
        
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
            
            # Step 2: Wait for app to load and ensure it's in focus
            logger.info("Waiting for Weather app to load...")
            await asyncio.sleep(2.0)  # Initial load time
            
            # Step 3: Try multiple times to ensure Weather app is visible
            max_attempts = 3
            weather_found = False
            weather_description = ""
            
            for attempt in range(max_attempts):
                logger.info(f"Attempt {attempt + 1} to read weather information...")
                
                # Ensure Weather app is in focus before each attempt
                self.controller.switch_to_application("Weather")
                await asyncio.sleep(0.5)  # Brief pause to ensure app is in foreground
                
                # Try to minimize other windows if needed
                if attempt > 0:
                    logger.info("Bringing Weather app to foreground...")
                    # Use AppleScript to ensure Weather is truly in front
                    import subprocess
                    try:
                        subprocess.run([
                            "osascript", "-e", 
                            'tell application "Weather" to activate'
                        ], capture_output=True)
                        await asyncio.sleep(1.0)
                    except:
                        pass
            
            # Prepare vision query specifically for weather
            vision_params = {
                'query': 'Focus ONLY on the Weather app window that should be visible on screen. Read and describe ONLY the weather information: What is the current temperature? What are the current weather conditions (sunny, cloudy, rainy, etc.)? What is the forecast for today including high/low temperatures? Also mention the location if visible. Ignore all other content on the screen.'
            }
            
            # Use vision handler to analyze
            result = await self.vision_handler.describe_screen(vision_params)
            
            if result.success:
                # Extract weather info from vision response
                weather_description = result.description
                
                # If the vision response doesn't seem to contain weather info, try again with more specific prompt
                if 'temperature' not in weather_description.lower() and 'degrees' not in weather_description.lower() and '°' not in weather_description:
                    vision_params['query'] = 'Look at the Weather app window on the screen. What temperature numbers do you see? What weather condition icons or text are shown? Read the specific temperature values and weather conditions displayed in the Weather app ONLY.'
                    result = await self.vision_handler.describe_screen(vision_params)
                    weather_description = result.description
                
                # Parse the weather information
                parser = WeatherResponseParser()
                extracted_weather = parser.extract_weather_info(weather_description)
                
                # Try to find location in the response
                location = None
                import re
                # Look for specific city names in the weather description
                city_patterns = [
                    r'(?:for|in|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*(?:\d+°|weather)',
                    r'location[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                ]
                
                for pattern in city_patterns:
                    location_match = re.search(pattern, weather_description)
                    if location_match:
                        potential_location = location_match.group(1)
                        # Filter out common non-location words
                        if not any(word in potential_location.lower() for word in [
                            'yes', 'sir', 'weather', 'app', 'the', 'currently', 'today'
                        ]):
                            location = potential_location
                            break
                
                # Format the response
                formatted_response = parser.format_weather_response(extracted_weather, location)
                
                return {
                    'success': True,
                    'message': formatted_response,
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
    
    # Try continuous vision first if available
    result = await workflow.check_weather_with_continuous_vision()
    
    if result['success']:
        # The message is already parsed and formatted
        return result['message']
    else:
        return result['message']
