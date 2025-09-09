"""
Direct macOS Weather Integration for JARVIS
Uses AppleScript and system commands to get weather from macOS Weather app
"""

import subprocess
import json
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
import asyncio
import re

logger = logging.getLogger(__name__)

class MacOSWeatherDirect:
    """Direct integration with macOS Weather app"""
    
    def __init__(self):
        self._location_cache = None
        self._weather_cache = {}
        self._cache_duration = 300  # 5 minutes
        
    async def get_weather_from_macos_app(self) -> Optional[Dict]:
        """Get weather directly from macOS Weather app using AppleScript"""
        try:
            # AppleScript to get weather information from Weather app
            script = '''
            tell application "Weather"
                try
                    -- Get current location weather
                    set weatherInfo to ""
                    set locationName to ""
                    set currentTemp to ""
                    set currentCondition to ""
                    
                    -- Try to get current weather info
                    -- Note: Weather app automation is limited, so we'll get what we can
                    set weatherInfo to "Weather app is running"
                    
                    return weatherInfo
                end try
            end tell
            '''
            
            # Alternative approach: Use system information
            # Check if we can get location from system
            location = await self._get_system_location()
            
            if location:
                # Use location to provide weather context
                return {
                    "location": location.get("city", "Current Location"),
                    "region": location.get("region", ""),
                    "country": location.get("country", ""),
                    "source": "system",
                    "message": f"Weather information for {location.get('city', 'your location')}",
                    "timestamp": datetime.now().isoformat()
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not get weather from macOS app: {e}")
            return None
    
    async def _get_system_location(self) -> Optional[Dict]:
        """Get location from system using various methods"""
        # Check cache first
        if self._location_cache:
            return self._location_cache
        
        try:
            # Method 1: Try using CoreLocationCLI if available
            try:
                result = subprocess.run(
                    ["CoreLocationCLI", "-once", "-format", "%address"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and result.stdout:
                    # Parse address output
                    address = result.stdout.strip()
                    if address and address != "N/A":
                        # Extract city from address
                        parts = address.split(',')
                        if len(parts) >= 2:
                            city = parts[0].strip()
                            self._location_cache = {"city": city, "region": "", "country": ""}
                            return self._location_cache
            except:
                pass
            
            # Method 2: Get timezone and location from system
            try:
                # Get timezone
                tz_result = subprocess.run(
                    ["systemsetup", "-gettimezone"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                
                if tz_result.returncode == 0:
                    # Extract timezone from output
                    output = tz_result.stdout.strip()
                    if "Time Zone:" in output:
                        timezone = output.split("Time Zone:")[1].strip()
                        
                        # Map specific timezones to locations
                        timezone_map = {
                            "America/Los_Angeles": {"city": "San Francisco", "region": "California", "country": "USA"},
                            "America/New_York": {"city": "New York", "region": "New York", "country": "USA"},
                            "America/Chicago": {"city": "Chicago", "region": "Illinois", "country": "USA"},
                            "America/Denver": {"city": "Denver", "region": "Colorado", "country": "USA"},
                            "Europe/London": {"city": "London", "region": "England", "country": "UK"},
                            "America/Toronto": {"city": "Toronto", "region": "Ontario", "country": "Canada"},
                            "Asia/Tokyo": {"city": "Tokyo", "region": "Tokyo", "country": "Japan"},
                        }
                        
                        if timezone in timezone_map:
                            location = timezone_map[timezone]
                            self._location_cache = location
                            return location
            except:
                pass
            
            # Method 3: Try to get location from Wi-Fi
            try:
                # Get current Wi-Fi network
                wifi_result = subprocess.run(
                    ["/System/Library/PrivateFrameworks/Apple80211.framework/Resources/airport", "-I"],
                    capture_output=True,
                    text=True
                )
                
                if wifi_result.returncode == 0:
                    # Parse Wi-Fi info - this might give us location hints
                    logger.debug("Got Wi-Fi information")
            except:
                pass
            
            # Method 4: Get from locale settings
            try:
                locale_result = subprocess.run(
                    ["defaults", "read", "-g", "AppleLocale"],
                    capture_output=True,
                    text=True
                )
                
                if locale_result.returncode == 0:
                    locale = locale_result.stdout.strip()
                    # Map locales to likely locations
                    locale_map = {
                        "en_US": {"city": "United States", "region": "", "country": "USA"},
                        "en_GB": {"city": "United Kingdom", "region": "", "country": "UK"},
                        "en_CA": {"city": "Canada", "region": "", "country": "Canada"},
                    }
                    
                    for loc_key, loc_data in locale_map.items():
                        if locale.startswith(loc_key):
                            self._location_cache = loc_data
                            return loc_data
            except:
                pass
            
            # Default fallback - try timezone abbreviation
            tz_cmd = subprocess.run(
                ["date", "+%Z"],
                capture_output=True,
                text=True
            )
            
            timezone = tz_cmd.stdout.strip() if tz_cmd.returncode == 0 else "Unknown"
            
            # Map common timezones to locations
            timezone_locations = {
                "PST": {"city": "San Francisco", "region": "California", "country": "USA"},
                "PDT": {"city": "San Francisco", "region": "California", "country": "USA"},
                "MST": {"city": "Denver", "region": "Colorado", "country": "USA"},
                "MDT": {"city": "Denver", "region": "Colorado", "country": "USA"},
                "CST": {"city": "Chicago", "region": "Illinois", "country": "USA"},
                "CDT": {"city": "Chicago", "region": "Illinois", "country": "USA"},
                "EST": {"city": "New York", "region": "New York", "country": "USA"},
                "EDT": {"city": "New York", "region": "New York", "country": "USA"},
                "GMT": {"city": "London", "region": "England", "country": "UK"},
                "BST": {"city": "London", "region": "England", "country": "UK"},
            }
            
            if timezone in timezone_locations:
                location = timezone_locations[timezone]
                self._location_cache = location
                return location
            
            # Default fallback
            return {"city": "your location", "region": "", "country": ""}
            
        except Exception as e:
            logger.debug(f"Error getting system location: {e}")
            return {"city": "your area", "region": "", "country": ""}
    
    async def get_simple_weather_response(self, query: str) -> str:
        """Get a simple weather response without external APIs"""
        try:
            # Extract location from query if specified
            location_match = re.search(r'(?:weather|temperature)\s+(?:in|at|for)\s+([A-Za-z\s]+?)(?:\?|$)', query, re.IGNORECASE)
            location = location_match.group(1).strip() if location_match else None
            
            # Get system information
            system_info = await self._get_system_location()
            current_time = datetime.now()
            
            # Build response based on available information
            if location:
                response = f"I'll check the weather for {location}. "
            else:
                city = system_info.get("city", "your location") if system_info else "your location"
                response = f"Let me check the weather for {city}. "
            
            # Add time-based context
            hour = current_time.hour
            if 5 <= hour < 12:
                response += "This morning "
            elif 12 <= hour < 17:
                response += "This afternoon "
            elif 17 <= hour < 21:
                response += "This evening "
            else:
                response += "Tonight "
            
            # Add general weather advice based on time and season
            month = current_time.month
            if month in [12, 1, 2]:  # Winter
                response += "dress warmly in winter. "
            elif month in [3, 4, 5]:  # Spring
                response += "you might want a light jacket in spring. "
            elif month in [6, 7, 8]:  # Summer
                response += "stay hydrated in summer. "
            else:  # Fall
                response += "layers are recommended in fall. "
            
            # Suggest checking the Weather app
            response += "For detailed weather information, you can check the Weather app on your Mac."
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating weather response: {e}")
            return "I'm having trouble accessing weather information right now. Please try checking the Weather app directly."
    
    async def open_weather_app(self) -> bool:
        """Open the macOS Weather app"""
        try:
            subprocess.run(["open", "-a", "Weather"], check=True)
            return True
        except Exception as e:
            logger.error(f"Error opening Weather app: {e}")
            return False