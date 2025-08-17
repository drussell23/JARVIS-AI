"""
Weather Service for JARVIS
Provides real-time weather data using OpenWeatherMap API
"""

import os
import json
import aiohttp
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
import geocoder

logger = logging.getLogger(__name__)


class WeatherService:
    """Service for fetching real-time weather data"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize weather service
        
        Args:
            api_key: OpenWeatherMap API key. If not provided, will check environment
        """
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "https://api.openweathermap.org/data/2.5"
        
        if not self.api_key:
            logger.warning("No OpenWeatherMap API key found. Weather features will be limited.")
    
    async def get_current_location(self) -> Tuple[float, float, str]:
        """Get current location using IP geolocation
        
        Returns:
            Tuple of (latitude, longitude, city_name)
        """
        try:
            g = geocoder.ip('me')
            if g.ok:
                return g.latlng[0], g.latlng[1], g.city
            else:
                # Default to Toronto if location detection fails
                return 43.6532, -79.3832, "Toronto"
        except Exception as e:
            logger.error(f"Error getting location: {e}")
            # Default to Toronto
            return 43.6532, -79.3832, "Toronto"
    
    async def get_weather_by_location(self, lat: float, lon: float) -> Dict:
        """Get weather data for specific coordinates
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Weather data dictionary
        """
        if not self.api_key:
            return self._get_mock_weather()
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/weather"
                params = {
                    'lat': lat,
                    'lon': lon,
                    'appid': self.api_key,
                    'units': 'metric'  # Use Celsius
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._format_weather_data(data)
                    else:
                        logger.error(f"Weather API error: {response.status}")
                        return self._get_mock_weather()
                        
        except Exception as e:
            logger.error(f"Error fetching weather: {e}")
            return self._get_mock_weather()
    
    async def get_weather_by_city(self, city: str) -> Dict:
        """Get weather data for a specific city
        
        Args:
            city: City name
            
        Returns:
            Weather data dictionary
        """
        if not self.api_key:
            return self._get_mock_weather(city)
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/weather"
                params = {
                    'q': city,
                    'appid': self.api_key,
                    'units': 'metric'
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._format_weather_data(data)
                    else:
                        logger.error(f"Weather API error: {response.status}")
                        return self._get_mock_weather(city)
                        
        except Exception as e:
            logger.error(f"Error fetching weather: {e}")
            return self._get_mock_weather(city)
    
    async def get_current_weather(self) -> Dict:
        """Get weather for current location
        
        Returns:
            Weather data dictionary
        """
        lat, lon, city = await self.get_current_location()
        weather_data = await self.get_weather_by_location(lat, lon)
        weather_data['detected_location'] = city
        return weather_data
    
    def _format_weather_data(self, raw_data: Dict) -> Dict:
        """Format raw weather data into a clean structure
        
        Args:
            raw_data: Raw data from OpenWeatherMap API
            
        Returns:
            Formatted weather data
        """
        return {
            'location': raw_data.get('name', 'Unknown'),
            'country': raw_data.get('sys', {}).get('country', ''),
            'temperature': round(raw_data.get('main', {}).get('temp', 0)),
            'feels_like': round(raw_data.get('main', {}).get('feels_like', 0)),
            'description': raw_data.get('weather', [{}])[0].get('description', 'Unknown'),
            'main': raw_data.get('weather', [{}])[0].get('main', 'Unknown'),
            'humidity': raw_data.get('main', {}).get('humidity', 0),
            'wind_speed': round(raw_data.get('wind', {}).get('speed', 0) * 3.6, 1),  # Convert m/s to km/h
            'pressure': raw_data.get('main', {}).get('pressure', 0),
            'visibility': raw_data.get('visibility', 10000) / 1000,  # Convert to km
            'sunrise': datetime.fromtimestamp(raw_data.get('sys', {}).get('sunrise', 0)).strftime('%H:%M'),
            'sunset': datetime.fromtimestamp(raw_data.get('sys', {}).get('sunset', 0)).strftime('%H:%M'),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_mock_weather(self, city: str = "Toronto") -> Dict:
        """Get mock weather data when API is not available
        
        Args:
            city: City name
            
        Returns:
            Mock weather data
        """
        return {
            'location': city,
            'country': 'CA',
            'temperature': 21,
            'feels_like': 19,
            'description': 'partly cloudy',
            'main': 'Clouds',
            'humidity': 65,
            'wind_speed': 15.5,
            'pressure': 1013,
            'visibility': 10,
            'sunrise': '06:45',
            'sunset': '19:30',
            'timestamp': datetime.now().isoformat(),
            'is_mock': True
        }
    
    def format_for_jarvis(self, weather_data: Dict) -> str:
        """Format weather data for JARVIS response
        
        Args:
            weather_data: Weather data dictionary
            
        Returns:
            Formatted string for JARVIS to speak
        """
        location = weather_data.get('location', 'your location')
        temp = weather_data.get('temperature', 0)
        feels_like = weather_data.get('feels_like', temp)
        description = weather_data.get('description', 'unknown conditions')
        wind = weather_data.get('wind_speed', 0)
        
        # Build response
        response = f"The current weather in {location} is {description} "
        response += f"with a temperature of {temp} degrees Celsius"
        
        if abs(feels_like - temp) > 2:
            response += f", though it feels like {feels_like}"
        
        response += f". Wind speed is {wind} kilometers per hour"
        
        # Add weather-appropriate suggestions
        if temp > 25:
            response += ". Quite warm today, sir. Perhaps consider lighter attire"
        elif temp < 10:
            response += ". Rather chilly, sir. I'd recommend a jacket"
        elif 'rain' in description.lower():
            response += ". Don't forget an umbrella if you're heading out"
        elif 'clear' in description.lower() and temp > 18:
            response += ". Beautiful weather for any outdoor activities you might have planned"
        
        return response