#!/usr/bin/env python3
"""
Quick Weather Reader
Reads whatever is currently showing in Weather app without navigation
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


class WeatherQuickReader:
    """Read weather directly without navigation attempts"""
    
    def __init__(self, vision_handler):
        self.vision_handler = vision_handler
        
    async def read_current_weather(self) -> str:
        """Read whatever weather is currently showing"""
        try:
            if not self.vision_handler or not hasattr(self.vision_handler, 'analyze_weather_fast'):
                return None
                
            # Just read what's on screen
            result = await self.vision_handler.analyze_weather_fast()
            
            if result.get('success') and result.get('analysis'):
                # Format the response
                return self._format_weather_response(result['analysis'])
                
        except Exception as e:
            logger.error(f"Quick weather read error: {e}")
            
        return None
    
    def _format_weather_response(self, raw_analysis: str) -> str:
        """Format raw weather analysis into natural response"""
        if "Location:" in raw_analysis and "Temp:" in raw_analysis:
            # Parse the structured format
            parts = raw_analysis.split('\n')
            location = ""
            temp = ""
            condition = ""
            high_low = ""
            
            for part in parts:
                if "Location:" in part:
                    location = part.split("Location:")[1].strip()
                elif "Temp:" in part:
                    temp = part.split("Temp:")[1].strip()
                elif "Condition:" in part:
                    condition = part.split("Condition:")[1].strip()
                elif "High/Low:" in part:
                    high_low = part.split("High/Low:")[1].strip()
            
            # Build natural response
            response = f"Looking at the weather"
            
            if location:
                response = f"Looking at the weather in {location}"
                
            if temp:
                response += f", it's currently {temp}"
                
            if condition:
                response += f" and {condition.lower()}"
                
            if high_low:
                response += f". Today's high and low are {high_low}"
                
            return response
            
        # Return raw analysis if not in expected format
        return raw_analysis