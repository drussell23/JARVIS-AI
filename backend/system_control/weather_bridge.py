"""
Weather Bridge for JARVIS
Provides intelligent weather data using macOS WeatherKit or fallback to APIs
"""

import os
import json
import asyncio
import subprocess
import platform
import logging
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
from pathlib import Path
import re
import aiohttp
from functools import lru_cache
import hashlib

logger = logging.getLogger(__name__)

class WeatherBridge:
    """Intelligent weather bridge that prioritizes macOS WeatherKit over external APIs"""
    
    def __init__(self):
        # Cache configuration
        self._weather_cache = {}
        self._location_cache = {}
        self._cache_duration = 300  # 5 minutes
        
        # Pattern recognition for weather queries
        self._weather_patterns = self._compile_weather_patterns()
        
        # Initialize macOS Weather app integration (primary)
        from .macos_weather_app import MacOSWeatherApp
        self.macos_weather_app = MacOSWeatherApp()
        
        # Initialize macOS direct weather
        from .macos_weather_direct import MacOSWeatherDirect
        self.macos_direct = MacOSWeatherDirect()
        
        # Check temperature unit preference
        from .temperature_units import should_use_fahrenheit
        self.use_fahrenheit = should_use_fahrenheit()
        
        # Initialize macOS weather provider
        try:
            from .macos_weather_provider import MacOSWeatherProvider
            self.macos_provider = MacOSWeatherProvider()
        except Exception as e:
            logger.warning(f"Could not initialize macOS weather provider: {e}")
            self.macos_provider = None
        
        # Fallback weather service
        self.api_weather_service = None
        self._init_fallback_service()
    
    def _compile_weather_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for weather query detection"""
        patterns = [
            # Direct weather queries
            r'\b(what|whats|what\'s|how|hows|how\'s)\s*(is|are)?\s*(the)?\s*weather\b',
            r'\bweather\s*(forecast|report|update|conditions?|today|tomorrow|now)\b',
            r'\b(current|today\'s|todays|tomorrow\'s|tomorrows)\s*weather\b',
            
            # Temperature queries
            r'\b(what|whats|what\'s|how)\s*(is|are)?\s*(the)?\s*temperature\b',
            r'\b(how\s*)?(hot|cold|warm|cool)\s*(is\s*it|outside)\b',
            r'\btemperature\s*(outside|now|today)\b',
            
            # Condition queries
            r'\b(is\s*it|will\s*it)\s*(rain|raining|snow|snowing|sunny|cloudy|foggy|windy)\b',
            r'\b(rain|snow|sun|cloud|fog|wind|storm)\s*(today|tomorrow|now)\b',
            
            # Location-specific weather
            r'\bweather\s*(in|at|for)\s*(\w+[\w\s]*)\b',
            r'\b(what|whats|what\'s)\s*(the)?\s*weather\s*(like)?\s*(in|at)\s*(\w+[\w\s]*)\b',
            
            # Forecast queries
            r'\b(weather|temperature)\s*forecast\b',
            r'\b(will|going\s*to)\s*(rain|snow|be\s*sunny|be\s*cloudy)\b',
            
            # Natural language variations
            r'\bdo\s*i\s*need\s*(an?)?\s*(umbrella|jacket|coat|sunscreen)\b',
            r'\bshould\s*i\s*(wear|bring|take)\s*(an?)?\s*(umbrella|jacket|coat)\b',
            r'\bnice\s*(day|weather)\s*(outside|today)\b',
            
            # UV and other conditions
            r'\b(uv|ultraviolet)\s*(index|level)\b',
            r'\b(humidity|wind\s*speed|visibility|pressure)\b',
            r'\b(sunrise|sunset)\s*(time|today)\b',
        ]
        
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def _init_fallback_service(self):
        """Initialize fallback weather service if available"""
        try:
            # Try different import paths
            try:
                from backend.services.weather_service import WeatherService
            except ImportError:
                try:
                    from services.weather_service import WeatherService
                except ImportError:
                    from ..services.weather_service import WeatherService
            
            api_key = os.getenv("OPENWEATHER_API_KEY")
            if api_key:
                self.api_weather_service = WeatherService(api_key)
                logger.info("Initialized fallback weather service")
        except Exception as e:
            logger.debug(f"Fallback weather service not available: {e}")
    
    
    def is_weather_query(self, text: str) -> bool:
        """Check if text is a weather-related query using pattern matching"""
        return any(pattern.search(text) for pattern in self._weather_patterns)
    
    def extract_location_from_query(self, text: str) -> Optional[str]:
        """Extract location from weather query"""
        text_lower = text.lower()
        
        # Check if this is asking about current location (time-based queries)
        current_location_patterns = [
            r'weather\s+(?:for\s+)?today',
            r'weather\s+(?:for\s+)?tomorrow',
            r'weather\s+(?:for\s+)?tonight',
            r'weather\s+(?:for\s+)?now',
            r'weather\s+(?:for\s+)?this\s+(?:week|weekend|morning|afternoon|evening)',
            r'current\s+weather',
            r'weather\s+outside',
            r'weather\s+here',
            r"what's\s+(?:the\s+)?weather\s*$",  # Just "what's the weather" with nothing after
            r"what\s+is\s+(?:the\s+)?weather\s*$",  # Just "what is the weather" with nothing after
        ]
        
        # Check if it matches current location patterns
        for pattern in current_location_patterns:
            if re.search(pattern, text_lower):
                return None  # Current location
        
        # Patterns to extract specific location
        location_patterns = [
            # "weather in Tokyo", "weather at Paris", "weather for London"
            r'weather\s+(?:in|at|for)\s+([A-Za-z]+(?:\s+[A-Za-z]+)*?)(?:\s|$)',
            # "what's the weather like in New York"
            r'weather\s+(?:like\s+)?(?:in|at)\s+([A-Za-z]+(?:\s+[A-Za-z]+)*?)(?:\s|$)',
            # "Tokyo weather"
            r'^([A-Za-z]+(?:\s+[A-Za-z]+)*?)\s+weather',
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                # Filter out common words, time-related words, and query words
                excluded_words = [
                    'the', 'a', 'an', 'it', 'there', 'is', 'are', 'was', 'were',
                    'today', 'tomorrow', 'tonight', 'yesterday', 'now',
                    'current', 'this', 'that', 'here', 'outside'
                ]
                
                # Check if location is meaningful (not just excluded words)
                location_words = location.lower().split()
                meaningful_words = [w for w in location_words if w not in excluded_words]
                
                if meaningful_words and len(location) > 2:
                    return location
        
        return None
    
    
    async def get_current_weather(self, use_cache: bool = True) -> Dict:
        """Get weather for current location using best available method"""
        # Check cache
        cache_key = "current_location"
        if use_cache and cache_key in self._weather_cache:
            cached = self._weather_cache[cache_key]
            if (datetime.now() - cached['timestamp']).seconds < self._cache_duration:
                logger.info("Returning cached weather data")
                return cached['data']
        
        # Use macOS weather provider
        try:
            weather_data = await self.macos_provider.get_weather_data()
            if weather_data and weather_data.get("source") != "fallback":
                # Cache the result
                self._weather_cache[cache_key] = {
                    'data': weather_data,
                    'timestamp': datetime.now()
                }
                return weather_data
        except Exception as e:
            logger.error(f"macOS weather provider failed: {e}")
        
        # Fallback to API service if available
        if self.api_weather_service:
            logger.info("Using fallback weather API")
            try:
                return await self.api_weather_service.get_current_weather()
            except Exception as e:
                logger.error(f"Fallback weather API failed: {e}")
        
        # Last resort - return informative message
        return self._get_fallback_weather_response()
    
    async def get_weather_by_city(self, city: str, use_cache: bool = True) -> Dict:
        """Get weather for specific city"""
        # Normalize city name
        city_normalized = city.strip().title()
        cache_key = f"city:{city_normalized.lower()}"
        
        # Check cache
        if use_cache and cache_key in self._weather_cache:
            cached = self._weather_cache[cache_key]
            if (datetime.now() - cached['timestamp']).seconds < self._cache_duration:
                return cached['data']
        
        # Use macOS weather provider
        try:
            weather_data = await self.macos_provider.get_weather_data(city_normalized)
            if weather_data and weather_data.get("source") != "fallback":
                self._weather_cache[cache_key] = {
                    'data': weather_data,
                    'timestamp': datetime.now()
                }
                return weather_data
        except Exception as e:
            logger.error(f"macOS weather provider failed for {city}: {e}")
        
        # Fallback to API
        if self.api_weather_service:
            try:
                return await self.api_weather_service.get_weather_by_city(city)
            except Exception as e:
                logger.error(f"Fallback weather API failed for {city}: {e}")
        
        return self._get_fallback_weather_response(city)
    
    def _format_weatherkit_data(self, data: Dict, city_override: Optional[str] = None) -> Dict:
        """Format WeatherKit data to match expected structure"""
        formatted = {
            "location": city_override or data.get("location", "Current Location"),
            "temperature": data.get("temperature", 20),
            "temperature_unit": data.get("temperatureUnit", "°C"),
            "feels_like": data.get("feelsLike", data.get("temperature", 20)),
            "description": data.get("description", "unknown"),
            "condition": data.get("condition", "unknown"),
            "humidity": data.get("humidity", 50),
            "wind_speed": data.get("windSpeed", 0),
            "wind_direction": data.get("windDirection", "N"),
            "pressure": data.get("pressure", 1013),
            "visibility": data.get("visibility", 10),
            "uv_index": data.get("uvIndex", 0),
            "cloud_cover": data.get("cloudCover", 0),
            "is_daylight": data.get("isDaylight", True),
            "sunrise": data.get("sunrise", "06:00"),
            "sunset": data.get("sunset", "18:00"),
            "moon_phase": data.get("moonPhase", ""),
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "source": "WeatherKit"
        }
        
        # Add insights if available
        if "insights" in data:
            formatted["insights"] = data["insights"]
        
        # Add hourly forecast if available
        if "hourlyForecast" in data:
            formatted["hourly_forecast"] = data["hourlyForecast"]
        
        # Add alerts if any
        if "alerts" in data:
            formatted["alerts"] = data["alerts"]
        
        return formatted
    
    def _get_fallback_weather_response(self, city: str = "your location") -> Dict:
        """Get fallback weather response when no service is available"""
        return {
            "location": city,
            "temperature": 20,
            "temperature_unit": "°C",
            "feels_like": 18,
            "description": "Weather data temporarily unavailable",
            "condition": "unknown",
            "humidity": 50,
            "wind_speed": 10,
            "wind_direction": "N",
            "source": "unavailable",
            "message": "Weather services are currently unavailable. Please check your internet connection or try again later.",
            "timestamp": datetime.now().isoformat()
        }
    
    def format_for_speech(self, weather_data: Dict, query_type: str = "current") -> str:
        """Format weather data for natural speech output"""
        location = weather_data.get("location", "your location")
        
        # Handle temperature based on system preference
        temp_c = weather_data.get("temperature", 0)
        feels_like_c = weather_data.get("feels_like", temp_c)
        
        # Convert to user's preferred unit
        if self.use_fahrenheit:
            # Convert to Fahrenheit
            temp = weather_data.get("temperature_f", round(temp_c * 9/5 + 32))
            feels_like = weather_data.get("feels_like_f", round(feels_like_c * 9/5 + 32))
            temp_display = f"{temp}°F"
            feels_display = f"{feels_like}°F"
            temp_unit = "°F"
        else:
            # Use Celsius
            temp = temp_c
            feels_like = feels_like_c
            temp_display = f"{temp}°C"
            feels_display = f"{feels_like}°C"
            temp_unit = "°C"
        
        description = weather_data.get("description", "unknown conditions")
        condition = weather_data.get("condition", description)
        wind_speed = weather_data.get("wind_speed", 0)
        humidity = weather_data.get("humidity", 0)
        
        # Clean up location - remove "Lat X and Lon Y" format
        if location.startswith("Lat ") and " and Lon " in location:
            # Try to get actual city name from weather data
            city = weather_data.get("detected_location") or weather_data.get("city", "your area")
            location = city
        
        # Make description more natural
        description = description.lower()
        if description in ["clear", "clear sky", "clear skies"]:
            description = "clear skies"
        elif description in ["partly cloudy", "partially cloudy"]:
            description = "partly cloudy"
        elif description in ["cloudy", "overcast"]:
            description = "cloudy"
        
        # Build conversational response
        hour = datetime.now().hour
        
        # Start with a natural greeting based on time
        if 5 <= hour < 12:
            time_greeting = "this morning"
        elif 12 <= hour < 17:
            time_greeting = "this afternoon"
        elif 17 <= hour < 21:
            time_greeting = "this evening"
        else:
            time_greeting = "tonight"
        
        # Build natural response based on query type
        if query_type == "temperature":
            # Convert temp thresholds based on unit
            if temp_unit == "°F":
                if temp > 86:  # 30°C
                    response = f"It's quite hot in {location} at {temp_display}"
                elif temp > 77:  # 25°C
                    response = f"It's a warm {temp_display} in {location}"
                elif temp > 68:  # 20°C
                    response = f"It's a pleasant {temp_display} in {location}"
                elif temp > 59:  # 15°C
                    response = f"It's {temp_display} in {location} - quite mild"
                elif temp > 50:  # 10°C
                    response = f"It's a cool {temp_display} in {location}"
                elif temp > 41:  # 5°C
                    response = f"It's rather chilly at {temp_display} in {location}"
                else:
                    response = f"It's quite cold in {location} at {temp_display}"
            else:
                # Celsius thresholds
                if temp > 30:
                    response = f"It's quite hot in {location} at {temp_display}"
                elif temp > 25:
                    response = f"It's a warm {temp_display} in {location}"
                elif temp > 20:
                    response = f"It's a pleasant {temp_display} in {location}"
                elif temp > 15:
                    response = f"It's {temp_display} in {location} - quite mild"
                elif temp > 10:
                    response = f"It's a cool {temp_display} in {location}"
                elif temp > 5:
                    response = f"It's rather chilly at {temp_display} in {location}"
                else:
                    response = f"It's quite cold in {location} at {temp_display}"
            
            if abs(feels_like - temp) > 3:
                if feels_like > temp:
                    response += f", but it feels warmer at {feels_display}"
                else:
                    response += f", but it feels colder at {feels_display}"
        
        elif query_type == "condition":
            response = f"We have {description} in {location} {time_greeting}"
            response += f" with temperatures around {temp_display}"
        
        else:  # General weather query for "today"
            # Make it more conversational
            if "rain" in description:
                response = f"Looks like we have {description} in {location} today"
            elif "clear" in description or "sunny" in description:
                response = f"It's a beautiful day in {location} with {description}"
            elif "cloud" in description:
                response = f"We're seeing {description} in {location} today"
            elif "snow" in description:
                response = f"We have {description} in {location}"
            else:
                response = f"The weather in {location} today is {description}"
            
            # Add temperature in a natural way
            response += f", currently {temp_display}"
            
            if abs(feels_like - temp) > 3:
                if feels_like > temp:
                    response += f" but feeling more like {feels_display}"
                else:
                    response += f" but feeling closer to {feels_display}"
        
        # Add contextual advice based on conditions
        advice_added = False
        
        # Use appropriate thresholds based on unit
        if temp_unit == "°F":
            if temp < 41:  # 5°C
                response += ". Bundle up warmly today"
                advice_added = True
            elif temp > 86:  # 30°C
                response += ". Stay cool and hydrated"
                advice_added = True
        else:
            if temp < 5:
                response += ". Bundle up warmly today"
                advice_added = True
            elif temp > 30:
                response += ". Stay cool and hydrated"
                advice_added = True
        
        if "rain" in description.lower() and not advice_added:
            response += ". Don't forget your umbrella"
            advice_added = True
        elif "snow" in description.lower() and not advice_added:
            response += ". Drive carefully if you're heading out"
            advice_added = True
        elif wind_speed > 30 and not advice_added:
            response += f". It's quite windy at {wind_speed} kilometers per hour"
            advice_added = True
        elif humidity > 85 and not advice_added:
            response += ". The humidity is quite high today"
            advice_added = True
        
        # Add a pleasant closing for nice weather
        if not advice_added:
            if temp_unit == "°F":
                # 68-82°F is pleasant
                if temp >= 68 and temp <= 82 and "clear" in description:
                    response += ". Perfect weather to be outside"
            else:
                # 20-28°C is pleasant
                if temp >= 20 and temp <= 28 and "clear" in description:
                    response += ". Perfect weather to be outside"
        
        # Add insights if available and relevant
        insights = weather_data.get("insights", [])
        if insights and len(insights) > 0 and not advice_added:
            # Only add the first insight if it's not redundant
            insight = insights[0]
            if "hydrated" not in response and "hydrated" in insight:
                response += f". {insight}"
            elif "umbrella" not in response and "umbrella" in insight:
                response += f". {insight}"
        
        # Add critical alerts
        alerts = weather_data.get("alerts", [])
        if alerts and alerts[0].get("severity") in ["high", "extreme"]:
            response += f". Weather alert: {alerts[0].get('summary', 'Please check weather warnings')}"
        
        return response
    
    async def process_weather_query(self, query: str) -> str:
        """Process a weather query and return formatted response"""
        try:
            # If user wants to open Weather app
            if "open" in query.lower() and "weather" in query.lower():
                if await self.macos_direct.open_weather_app():
                    return "I've opened the Weather app for you"
                else:
                    return "I couldn't open the Weather app. Please try opening it manually"
            
            # First, try to get weather from macOS Weather app directly
            try:
                weather_data = await self.macos_weather_app.get_weather_with_location()
                if weather_data and weather_data.get("source") != "fallback":
                    logger.info("Got weather from macOS Weather app")
                    # Determine query type and format response
                    query_lower = query.lower()
                    query_type = "current"
                    
                    if "temperature" in query_lower or "hot" in query_lower or "cold" in query_lower:
                        query_type = "temperature"
                    elif any(word in query_lower for word in ["rain", "snow", "sunny", "cloudy", "foggy"]):
                        query_type = "condition"
                    
                    return self.format_for_speech(weather_data, query_type)
            except Exception as e:
                logger.debug(f"Weather app access failed: {e}")
            
            # Fallback to simple direct approach
            response = await self.macos_direct.get_simple_weather_response(query)
            
            # Try to get actual weather data if available
            try:
                # Determine query type
                query_lower = query.lower()
                query_type = "current"
                
                if "temperature" in query_lower or "hot" in query_lower or "cold" in query_lower:
                    query_type = "temperature"
                elif any(word in query_lower for word in ["rain", "snow", "sunny", "cloudy", "foggy"]):
                    query_type = "condition"
                
                # Extract location if specified
                location = self.extract_location_from_query(query)
                
                # Get weather data with timeout
                weather_data = None
                try:
                    if location:
                        logger.info(f"Getting weather for city: {location}")
                        weather_data = await asyncio.wait_for(
                            self.get_weather_by_city(location), 
                            timeout=5.0
                        )
                    else:
                        logger.info("Getting current location weather")
                        weather_data = await asyncio.wait_for(
                            self.get_current_weather(), 
                            timeout=5.0
                        )
                except asyncio.TimeoutError:
                    logger.warning("Weather data fetch timed out")
                    return response  # Return simple response
                
                # If we got weather data, format it properly
                if weather_data and weather_data.get("source") != "fallback":
                    return self.format_for_speech(weather_data, query_type)
                else:
                    return response  # Return simple response
                    
            except Exception as e:
                logger.error(f"Error getting weather data: {e}")
                return response  # Return simple response
                
        except Exception as e:
            logger.error(f"Error processing weather query: {e}")
            return "I'm having trouble accessing weather information right now. You can check the Weather app for current conditions"
    
    def clear_cache(self):
        """Clear all weather caches"""
        self._weather_cache.clear()
        self._location_cache.clear()
        logger.info("Weather cache cleared")
    
    async def close(self):
        """Clean up resources"""
        if self.api_weather_service:
            await self.api_weather_service.close()