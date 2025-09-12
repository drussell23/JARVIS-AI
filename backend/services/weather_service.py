"""
Weather Service for JARVIS
Provides real-time weather data using OpenWeatherMap API
"""

import os
import json
import aiohttp
import asyncio
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import geocoder
import threading
import time
from functools import lru_cache

logger = logging.getLogger(__name__)

class WeatherService:
    """Service for fetching real-time weather data with caching and optimization"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize weather service with caching and pre-loading

        Args:
            api_key: OpenWeatherMap API key. If not provided, will check environment
        """
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
        self.base_url = "https://api.openweathermap.org/data/2.5"
        
        # Cache configuration
        self._weather_cache = {}
        self._location_cache = None
        self._cache_duration = 300  # 5 minutes for weather
        self._location_cache_duration = 3600  # 1 hour for location
        
        # Thread safety
        self._cache_lock = threading.Lock()
        self._location_lock = threading.Lock()
        
        # Connection pooling
        self._session = None
        
        if not self.api_key:
            logger.warning(
                "No OpenWeatherMap API key found. Weather features will be limited."
            )
        else:
            # Pre-load location in background
            threading.Thread(target=self._preload_data, daemon=True).start()

    def _preload_data(self):
        """Pre-load location and weather data in background"""
        try:
            # Pre-load location
            location = self._get_location_sync()
            with self._location_lock:
                self._location_cache = {
                    'data': location,
                    'timestamp': datetime.now()
                }
            logger.info(f"Pre-loaded location: {location[2]}")
            
            # Pre-load weather for current location
            time.sleep(0.5)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            weather = loop.run_until_complete(self._fetch_weather(location[0], location[1]))
            
            cache_key = f"{location[0]},{location[1]}"
            with self._cache_lock:
                self._weather_cache[cache_key] = {
                    'data': weather,
                    'timestamp': datetime.now()
                }
            logger.info("Pre-loaded weather data")
        except Exception as e:
            logger.error(f"Error pre-loading data: {e}")
    
    def _get_location_sync(self) -> Tuple[float, float, str]:
        """Synchronous location getter for threading"""
        try:
            g = geocoder.ip("me")
            if g.ok:
                return g.latlng[0], g.latlng[1], g.city
            else:
                return 43.6532, -79.3832, "Toronto"
        except Exception as e:
            logger.error(f"Error getting location: {e}")
            return 43.6532, -79.3832, "Toronto"
    
    async def get_current_location(self) -> Tuple[float, float, str]:
        """Get current location using IP geolocation with caching

        Returns:
            Tuple of (latitude, longitude, city_name)
        """
        # Check cache first
        with self._location_lock:
            if self._location_cache:
                cache_age = (datetime.now() - self._location_cache['timestamp']).seconds
                if cache_age < self._location_cache_duration:
                    return self._location_cache['data']
        
        # Get fresh location
        location = await asyncio.to_thread(self._get_location_sync)
        
        # Update cache
        with self._location_lock:
            self._location_cache = {
                'data': location,
                'timestamp': datetime.now()
            }
        
        return location

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session for connection pooling"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=3)  # 3 second timeout
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def _fetch_weather(self, lat: float, lon: float) -> Dict:
        """Fetch weather data from API"""
        if not self.api_key:
            return self._get_mock_weather()
        
        try:
            session = await self._get_session()
            url = f"{self.base_url}/weather"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric",
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._format_weather_data(data)
                else:
                    logger.error(f"Weather API error: {response.status}")
                    return self._get_mock_weather()
                    
        except asyncio.TimeoutError:
            logger.error("Weather API timeout")
            return self._get_mock_weather()
        except Exception as e:
            logger.error(f"Error fetching weather: {e}")
            return self._get_mock_weather()
    
    async def get_weather_by_location(self, lat: float, lon: float) -> Dict:
        """Get weather data for specific coordinates with caching

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Weather data dictionary
        """
        cache_key = f"{lat},{lon}"
        
        # Check cache
        with self._cache_lock:
            if cache_key in self._weather_cache:
                cached = self._weather_cache[cache_key]
                cache_age = (datetime.now() - cached['timestamp']).seconds
                if cache_age < self._cache_duration:
                    logger.info("Returning cached weather data")
                    return cached['data']
        
        # Fetch fresh data
        weather_data = await self._fetch_weather(lat, lon)
        
        # Update cache
        with self._cache_lock:
            self._weather_cache[cache_key] = {
                'data': weather_data,
                'timestamp': datetime.now()
            }
        
        return weather_data

    async def get_weather_by_city(self, city: str) -> Dict:
        """Get weather data for a specific city with caching

        Args:
            city: City name

        Returns:
            Weather data dictionary
        """
        cache_key = f"city:{city.lower()}"
        
        # Check cache
        with self._cache_lock:
            if cache_key in self._weather_cache:
                cached = self._weather_cache[cache_key]
                cache_age = (datetime.now() - cached['timestamp']).seconds
                if cache_age < self._cache_duration:
                    return cached['data']
        
        if not self.api_key:
            return self._get_mock_weather(city)

        try:
            session = await self._get_session()
            url = f"{self.base_url}/weather"
            params = {"q": city, "appid": self.api_key, "units": "metric"}
            
            logger.info(f"Fetching weather for: {city}")

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    weather_data = self._format_weather_data(data)
                    logger.info(f"Successfully got weather for: {weather_data.get('location', city)}")
                    
                    # Cache the result
                    with self._cache_lock:
                        self._weather_cache[cache_key] = {
                            'data': weather_data,
                            'timestamp': datetime.now()
                        }
                    
                    return weather_data
                elif response.status == 404:
                    logger.warning(f"Location not found: {city}")
                    # Return a clear error message
                    return {
                        'location': city,
                        'temperature': 0,
                        'feels_like': 0,
                        'description': 'location not found',
                        'humidity': 0,
                        'wind_speed': 0,
                        'error': True
                    }
                else:
                    logger.error(f"Weather API error: {response.status} for location: {city}")
                    return self._get_mock_weather(city)

        except Exception as e:
            logger.error(f"Error fetching weather: {e}")
            return self._get_mock_weather(city)

    async def get_current_weather(self) -> Dict:
        """Get weather for current location - OPTIMIZED

        Returns:
            Weather data dictionary
        """
        # Check if we have pre-loaded data
        if self._location_cache:
            lat, lon, city = self._location_cache['data']
            cache_key = f"{lat},{lon}"
            
            with self._cache_lock:
                if cache_key in self._weather_cache:
                    cached = self._weather_cache[cache_key]
                    cache_age = (datetime.now() - cached['timestamp']).seconds
                    if cache_age < self._cache_duration:
                        logger.info("Using pre-loaded weather data")
                        weather_data = cached['data'].copy()
                        weather_data["detected_location"] = city
                        # Only override location if not set by OpenWeatherMap
                        if not weather_data.get("location") or weather_data.get("location") == "Unknown":
                            weather_data["location"] = city
                        return weather_data
        
        # Fall back to normal flow but run in parallel
        location_task = asyncio.create_task(self.get_current_location())
        
        # If we have cached location, start fetching weather immediately
        if self._location_cache:
            lat, lon, city = self._location_cache['data']
            weather_task = asyncio.create_task(self.get_weather_by_location(lat, lon))
            
            # Wait for both tasks
            lat, lon, city = await location_task
            weather_data = await weather_task
        else:
            # Wait for location first
            lat, lon, city = await location_task
            weather_data = await self.get_weather_by_location(lat, lon)
        
        # Only set detected_location, let OpenWeatherMap's location name be used
        weather_data["detected_location"] = city
        # If OpenWeatherMap didn't provide a location name, use the detected city
        if not weather_data.get("location") or weather_data.get("location") == "Unknown":
            weather_data["location"] = city
        return weather_data

    def _format_weather_data(self, raw_data: Dict) -> Dict:
        """Format raw weather data into a clean structure

        Args:
            raw_data: Raw data from OpenWeatherMap API

        Returns:
            Formatted weather data
        """
        # Get location name and normalize Toronto districts
        location_name = raw_data.get("name", "Unknown")
        
        # Map Toronto districts/neighborhoods to "Toronto"
        toronto_districts = [
            "North York", "Scarborough", "Etobicoke", "East York", 
            "York", "Downtown Toronto", "Willowdale", "Don Mills",
            "Agincourt", "Malvern", "Rouge", "West Hill"
        ]
        
        # Check if this is a Toronto district
        if location_name in toronto_districts:
            location_name = "Toronto"
        
        return {
            "location": location_name,
            "country": raw_data.get("sys", {}).get("country", ""),
            "temperature": round(raw_data.get("main", {}).get("temp", 0)),
            "feels_like": round(raw_data.get("main", {}).get("feels_like", 0)),
            "description": raw_data.get("weather", [{}])[0].get(
                "description", "Unknown"
            ),
            "main": raw_data.get("weather", [{}])[0].get("main", "Unknown"),
            "humidity": raw_data.get("main", {}).get("humidity", 0),
            "wind_speed": round(
                raw_data.get("wind", {}).get("speed", 0) * 3.6, 1
            ),  # Convert m/s to km/h
            "pressure": raw_data.get("main", {}).get("pressure", 0),
            "visibility": raw_data.get("visibility", 10000) / 1000,  # Convert to km
            "sunrise": datetime.fromtimestamp(
                raw_data.get("sys", {}).get("sunrise", 0)
            ).strftime("%H:%M"),
            "sunset": datetime.fromtimestamp(
                raw_data.get("sys", {}).get("sunset", 0)
            ).strftime("%H:%M"),
            "timestamp": datetime.now().isoformat(),
        }

    def _get_mock_weather(self, city: str = "Toronto") -> Dict:
        """Get mock weather data when API is not available

        Args:
            city: City name

        Returns:
            Mock weather data
        """
        return {
            "location": city,
            "country": "CA",
            "temperature": 21,
            "feels_like": 19,
            "description": "partly cloudy",
            "main": "Clouds",
            "humidity": 65,
            "wind_speed": 15.5,
            "pressure": 1013,
            "visibility": 10,
            "sunrise": "06:45",
            "sunset": "19:30",
            "timestamp": datetime.now().isoformat(),
            "is_mock": True,
        }

    def format_for_jarvis(self, weather_data: Dict) -> str:
        """Format weather data for JARVIS response

        Args:
            weather_data: Weather data dictionary

        Returns:
            Formatted string for JARVIS to speak
        """
        location = weather_data.get("location", "your location")
        temp = weather_data.get("temperature", 0)
        feels_like = weather_data.get("feels_like", temp)
        description = weather_data.get("description", "unknown conditions")
        wind = weather_data.get("wind_speed", 0)

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
        elif "rain" in description.lower():
            response += ". Don't forget an umbrella if you're heading out"
        elif "clear" in description.lower() and temp > 18:
            response += (
                ". Beautiful weather for any outdoor activities you might have planned"
            )

        return response
    
    async def close(self):
        """Close the aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def clear_cache(self):
        """Clear all caches"""
        with self._cache_lock:
            self._weather_cache.clear()
        with self._location_lock:
            self._location_cache = None
