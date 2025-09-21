"""
Unified Weather Bridge for JARVIS
Single entry point for ALL weather queries using vision
Replaces all previous weather providers with one intelligent system
"""

import logging
from typing import Dict, Optional
from datetime import datetime

from .unified_vision_weather import get_unified_weather_system

logger = logging.getLogger(__name__)


class UnifiedWeatherBridge:
    """
    The ONLY weather bridge JARVIS needs
    Routes all weather queries to vision-based system
    """
    
    def __init__(self, vision_handler=None, controller=None):
        # Get the unified weather system
        self.weather_system = get_unified_weather_system(vision_handler, controller)
        
        logger.info("Unified Weather Bridge initialized - using vision-based weather only")
    
    async def get_weather(self, query: str = "") -> Dict:
        """
        Single method for ALL weather queries
        Handles current weather, forecasts, specific questions
        """
        try:
            # Let the unified system handle everything
            result = await self.weather_system.get_weather(query)
            
            # Log success/failure
            if result.get('success'):
                logger.info(f"Weather query successful: {result.get('location', 'Unknown')}")
            else:
                logger.warning(f"Weather query failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Weather bridge error: {e}")
            return {
                'success': False,
                'error': str(e),
                'formatted_response': "I encountered an error checking the weather.",
                'timestamp': datetime.now().isoformat()
            }
    
    # Convenience methods that all route to the same system
    async def get_current_weather(self) -> Dict:
        """Get current weather - routes to unified system"""
        return await self.get_weather("What's the current weather?")
    
    async def get_weather_by_city(self, city: str) -> Dict:
        """Get weather for a city - routes to unified system"""
        return await self.get_weather(f"What's the weather in {city}?")
    
    async def get_forecast(self, days: int = 7) -> Dict:
        """Get forecast - routes to unified system"""
        return await self.get_weather(f"What's the {days}-day forecast?")
    
    async def check_precipitation(self) -> Dict:
        """Check for rain/snow - routes to unified system"""
        return await self.get_weather("Will it rain or snow today?")
    
    def is_weather_query(self, text: str) -> bool:
        """Check if text is a weather-related query"""
        weather_keywords = [
            'weather', 'temperature', 'forecast', 'rain', 'snow',
            'sunny', 'cloudy', 'humid', 'wind', 'cold', 'hot',
            'warm', 'storm', 'degrees', 'celsius', 'fahrenheit'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in weather_keywords)