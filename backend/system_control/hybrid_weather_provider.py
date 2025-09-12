"""
Hybrid Weather Provider for JARVIS
Uses OpenWeatherMap API as primary with vision fallback
"""

import os
import logging
from typing import Dict, Optional
import asyncio

logger = logging.getLogger(__name__)

class HybridWeatherProvider:
    """Hybrid weather provider that uses OpenWeatherMap API with vision fallback"""
    
    def __init__(self, api_weather_service=None, vision_extractor=None):
        """Initialize hybrid weather provider
        
        Args:
            api_weather_service: The OpenWeatherMap service instance
            vision_extractor: The vision-based weather extractor for fallback
        """
        self.api_weather_service = api_weather_service
        self.vision_extractor = vision_extractor
        
        # Verify API key is available
        self.has_api_key = bool(os.getenv("OPENWEATHER_API_KEY"))
        
        logger.info(f"Hybrid weather provider initialized - API: {self.has_api_key}, Vision: {vision_extractor is not None}")
    
    async def get_weather_data(self, city: Optional[str] = None) -> Dict:
        """Get weather data using hybrid approach
        
        Args:
            city: Optional city name. If None, uses current location
            
        Returns:
            Weather data dictionary
        """
        # FIRST PRIORITY: Try OpenWeatherMap API
        if self.api_weather_service and self.has_api_key:
            try:
                logger.info(f"Attempting OpenWeatherMap API for: {city or 'current location'}")
                
                if city:
                    weather_data = await self.api_weather_service.get_weather_by_city(city)
                else:
                    weather_data = await self.api_weather_service.get_current_weather()
                
                if self._is_valid_api_data(weather_data):
                    logger.info(f"Successfully got weather from API: {weather_data.get('location')} - {weather_data.get('temperature')}°C")
                    
                    # Convert API format to standard format
                    return self._format_api_data(weather_data)
                else:
                    logger.warning("API returned invalid data")
                    
            except Exception as e:
                logger.error(f"OpenWeatherMap API failed: {e}")
        else:
            logger.warning("OpenWeatherMap API not available")
        
        # FALLBACK: Use vision extraction
        if self.vision_extractor:
            try:
                logger.info("Falling back to vision-based weather extraction")
                vision_data = await self.vision_extractor.get_weather_with_cache()
                
                if vision_data and self._is_valid_vision_data(vision_data):
                    logger.info(f"Got weather from vision: {vision_data.get('location')} - {vision_data.get('temperature')}°C")
                    vision_data['source'] = 'vision_fallback'
                    return vision_data
                    
            except Exception as e:
                logger.error(f"Vision weather extraction failed: {e}")
        
        # Last resort - return error response
        return self._get_error_response(city)
    
    def _is_valid_api_data(self, data: Dict) -> bool:
        """Check if API data is valid"""
        if not data or data.get('error'):
            return False
            
        # Must have temperature and location
        return (data.get('temperature') is not None and 
                data.get('location') and 
                data.get('location') != 'Unknown')
    
    def _is_valid_vision_data(self, data: Dict) -> bool:
        """Check if vision data is valid"""
        if not data:
            return False
            
        # Must have temperature data
        return data.get('temperature') is not None
    
    def _format_api_data(self, api_data: Dict) -> Dict:
        """Format OpenWeatherMap data to standard format"""
        # Map API fields to standard format
        formatted = {
            "location": api_data.get("location", "Unknown"),
            "temperature": api_data.get("temperature"),
            "temperature_unit": "°C",
            "feels_like": api_data.get("feels_like", api_data.get("temperature")),
            "description": api_data.get("description", "unknown"),
            "condition": api_data.get("main", api_data.get("description", "unknown")),
            "humidity": api_data.get("humidity", 0),
            "wind_speed": api_data.get("wind_speed", 0),
            "wind_direction": self._get_wind_direction(api_data.get("wind_deg", 0)),
            "pressure": api_data.get("pressure", 0),
            "visibility": api_data.get("visibility", 10),
            "sunrise": api_data.get("sunrise", ""),
            "sunset": api_data.get("sunset", ""),
            "timestamp": api_data.get("timestamp", ""),
            "source": "OpenWeatherMap",
            "country": api_data.get("country", "")
        }
        
        # Add temperature in Fahrenheit
        if formatted["temperature"] is not None:
            formatted["temperature_f"] = round(formatted["temperature"] * 9/5 + 32)
            formatted["feels_like_f"] = round(formatted["feels_like"] * 9/5 + 32)
        
        return formatted
    
    def _get_wind_direction(self, degrees: int) -> str:
        """Convert wind degrees to direction"""
        directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                      "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        index = round(degrees / 22.5) % 16
        return directions[index]
    
    def _get_error_response(self, city: Optional[str] = None) -> Dict:
        """Get error response when all methods fail"""
        location = city or "your location"
        return {
            "location": location,
            "temperature": None,
            "temperature_unit": "°C",
            "feels_like": None,
            "description": "Weather data unavailable",
            "condition": "unavailable",
            "humidity": None,
            "wind_speed": None,
            "wind_direction": None,
            "source": "unavailable",
            "message": "Unable to retrieve weather data. Please try again later.",
            "timestamp": "",
            "error": True
        }
    
    async def get_current_weather(self) -> Dict:
        """Convenience method for current location weather"""
        return await self.get_weather_data()
    
    async def get_weather_by_city(self, city: str) -> Dict:
        """Convenience method for city weather"""
        return await self.get_weather_data(city)