"""
Unified Vision-Based Weather System for JARVIS
Single source of truth for all weather queries using computer vision
Zero hardcoding - completely dynamic and intelligent
"""

import asyncio
import logging
import subprocess
import tempfile
import re
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class UnifiedVisionWeather:
    """
    The ONLY weather system JARVIS needs
    Uses vision to read Weather app - no GPS/API needed
    """
    
    def __init__(self, vision_handler=None, controller=None):
        self.vision_handler = vision_handler
        self.controller = controller
        self.cache = {}
        self.cache_duration = timedelta(minutes=5)
        
        # Dynamic UI element tracking
        self.ui_patterns = {
            'my_location': [
                r'my\s*location',
                r'current\s*location',
                r'home\s*location',
                r'📍.*home',
                r'•\s*home'
            ],
            'temperature': [
                r'(\d+)°',
                r'(\d+)\s*degrees',
                r'temp.*(\d+)'
            ],
            'conditions': [
                'clear', 'cloudy', 'partly cloudy', 'mostly cloudy',
                'rain', 'drizzle', 'shower', 'storm', 'thunder',
                'snow', 'sleet', 'hail', 'fog', 'mist',
                'sunny', 'overcast', 'windy'
            ]
        }
    
    async def get_weather(self, query: str = "") -> Dict[str, Any]:
        """
        Main entry point - handles ALL weather queries intelligently
        """
        try:
            logger.info(f"[UNIFIED WEATHER] Weather query received: {query}")
            logger.info(f"[UNIFIED WEATHER] Has vision_handler: {self.vision_handler is not None}")
            logger.info(f"[UNIFIED WEATHER] Has controller: {self.controller is not None}")
            
            # Check cache first
            cache_key = self._get_cache_key(query)
            if cached := self._check_cache(cache_key):
                logger.info("Using cached weather data")
                return cached
            
            # Ensure Weather app is ready with timeout
            try:
                app_ready = await asyncio.wait_for(
                    self._ensure_weather_app_ready(),
                    timeout=5.0  # 5 second timeout for app preparation
                )
                if not app_ready:
                    logger.error("[UNIFIED WEATHER] Weather app not ready")
                    return self._fallback_response("Unable to access Weather app")
            except asyncio.TimeoutError:
                logger.error("[UNIFIED WEATHER] Weather app preparation timed out")
                return self._fallback_response("Weather app took too long to open")
            
            # Extract weather based on query type with timeout
            try:
                weather_data = await asyncio.wait_for(
                    self._extract_weather_intelligently(query),
                    timeout=15.0  # 15 second timeout for full extraction
                )
            except asyncio.TimeoutError:
                logger.error("[UNIFIED WEATHER] Weather extraction timed out after 15s")
                return self._fallback_response("Weather analysis took too long")
            
            # Cache and return
            if weather_data.get('success'):
                self._update_cache(cache_key, weather_data)
            
            return weather_data
            
        except Exception as e:
            logger.error(f"[UNIFIED WEATHER] Weather extraction error: {e}", exc_info=True)
            return self._fallback_response(str(e))
    
    async def _ensure_weather_app_ready(self) -> bool:
        """Ensure Weather app is open and ready"""
        try:
            # Check if Weather app is running
            is_running = await self._is_app_running("Weather")
            
            if not is_running:
                # Open Weather app
                logger.info("Opening Weather app...")
                await self._open_weather_app()
                await asyncio.sleep(2)  # Wait for load
            
            # Bring to front
            await self._bring_app_to_front()
            await asyncio.sleep(0.5)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to prepare Weather app: {e}")
            return False
    
    async def _extract_weather_intelligently(self, query: str) -> Dict[str, Any]:
        """
        Extract weather data based on query intent
        Zero hardcoding - adapts to any Weather app layout
        """
        # Analyze query intent
        intent = self._analyze_query_intent(query)
        
        # First, find and select user's location
        await self._select_my_location()
        
        # Extract comprehensive weather data
        weather_data = await self._extract_comprehensive_weather()
        
        # Format response based on intent
        formatted = self._format_response_by_intent(weather_data, intent)
        
        return {
            'success': True,
            'data': weather_data,
            'formatted_response': formatted,
            'source': 'vision',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _select_my_location(self):
        """
        Dynamically find and click on user's location in sidebar
        Works with any location name, any position
        """
        if not self.vision_handler:
            return
        
        try:
            # Ask vision to find user's location with timeout
            try:
                result = await asyncio.wait_for(
                    self.vision_handler.describe_screen({
                        'query': """Look at the Weather app sidebar on the left.
                        Find the location marked as "My Location" or with a HOME indicator.
                        Tell me the exact text and describe where it is in the list."""
                    }),
                    timeout=5.0  # 5 second timeout for location finding
                )
            except asyncio.TimeoutError:
                logger.warning("Location finding timed out")
                return
            
            if result.success:
                # Parse location from response
                location_info = self._parse_location_from_vision(result.description)
                
                if location_info and self.controller:
                    # Click on the location
                    await self._click_location(location_info)
                    await asyncio.sleep(0.5)  # Wait for weather to update
                    
        except Exception as e:
            logger.error(f"Failed to select location: {e}")
    
    async def _extract_comprehensive_weather(self) -> Dict[str, Any]:
        """
        Extract ALL weather data visible on screen
        Completely dynamic - no assumptions about layout
        """
        if not self.vision_handler:
            return {}
        
        # Comprehensive weather extraction prompt
        vision_prompt = """Analyze the Weather app and extract ALL visible information:

CURRENT CONDITIONS:
- Location name (exactly as shown)
- Current temperature (the large number)
- Weather condition (Clear, Cloudy, etc.)
- "Feels like" temperature if shown
- Today's high and low temperatures

DETAILED CONDITIONS (if visible):
- Wind speed and direction
- Humidity percentage
- UV index number
- Air quality index
- Visibility distance
- Pressure
- Dew point
- Sunrise and sunset times

HOURLY FORECAST:
- List each hour shown with temperature and conditions
- Note any precipitation chances

10-DAY FORECAST:
- Each day's high/low temperatures
- Weather conditions for each day
- Any precipitation percentages

ADDITIONAL:
- Any weather alerts or warnings
- Location coordinates if shown
- Last updated time

Return all found information in a structured format."""
        
        # Add timeout for vision analysis to prevent hanging
        try:
            result = await asyncio.wait_for(
                self.vision_handler.describe_screen({'query': vision_prompt}),
                timeout=10.0  # 10 second timeout for vision analysis
            )
        except asyncio.TimeoutError:
            logger.error("Vision analysis timed out after 10 seconds")
            return {}
        
        if result.success:
            # Parse the comprehensive response
            return self._parse_comprehensive_weather(result.description)
        
        return {}
    
    def _parse_comprehensive_weather(self, vision_response: str) -> Dict[str, Any]:
        """
        Parse vision response into structured weather data
        Handles any format dynamically
        """
        data = {
            'location': None,
            'current': {},
            'today': {},
            'hourly': [],
            'daily': [],
            'details': {},
            'alerts': []
        }
        
        # Extract location
        location_match = re.search(
            r'(?:location|city|place)[:\s]+([^,\n]+)', 
            vision_response, 
            re.IGNORECASE
        )
        if location_match:
            data['location'] = location_match.group(1).strip()
        
        # Extract current temperature
        temp_matches = re.findall(r'(\d+)°[CF]?', vision_response)
        if temp_matches:
            # Largest number is usually current temp
            temps = [int(t) for t in temp_matches]
            data['current']['temperature'] = max(temps)
            
            # Look for high/low
            if 'high' in vision_response.lower() and 'low' in vision_response.lower():
                sorted_temps = sorted(temps)
                if len(sorted_temps) >= 2:
                    data['today']['high'] = sorted_temps[-1]
                    data['today']['low'] = sorted_temps[0]
        
        # Extract conditions dynamically
        vision_lower = vision_response.lower()
        for condition in self.ui_patterns['conditions']:
            if condition in vision_lower:
                # Find the context around the condition
                pattern = rf'(?:currently|now|condition)[:\s]*.*?({condition}[\w\s]*)'
                match = re.search(pattern, vision_lower, re.IGNORECASE)
                if match:
                    data['current']['condition'] = match.group(1).strip().title()
                    break
        
        # Extract detailed conditions
        detail_patterns = {
            'wind': r'wind[:\s]+(\d+)\s*(mph|km/h)',
            'humidity': r'humidity[:\s]+(\d+)%',
            'uv_index': r'uv\s*(?:index)?[:\s]+(\d+)',
            'air_quality': r'air\s*quality[:\s]+(\d+|good|moderate|poor)',
            'feels_like': r'feels?\s*like[:\s]+(\d+)°',
            'visibility': r'visibility[:\s]+(\d+(?:\.\d+)?)\s*(mi|km)',
            'pressure': r'pressure[:\s]+(\d+(?:\.\d+)?)',
            'sunrise': r'sunrise[:\s]+(\d{1,2}:\d{2}\s*[ap]m)',
            'sunset': r'sunset[:\s]+(\d{1,2}:\d{2}\s*[ap]m)'
        }
        
        for key, pattern in detail_patterns.items():
            match = re.search(pattern, vision_response, re.IGNORECASE)
            if match:
                data['details'][key] = match.group(1)
        
        # Extract hourly forecast
        hourly_pattern = r'(\d{1,2})\s*([ap]m)[:\s]+(\d+)°\s*([^,\n]+)'
        hourly_matches = re.findall(hourly_pattern, vision_response, re.IGNORECASE)
        for hour, ampm, temp, condition in hourly_matches[:12]:  # Next 12 hours
            data['hourly'].append({
                'time': f"{hour}{ampm}",
                'temperature': int(temp),
                'condition': condition.strip()
            })
        
        # Extract daily forecast
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 
                'today', 'tomorrow']
        daily_pattern = r'({}).*?(\d+)°.*?(\d+)°'.format('|'.join(days))
        daily_matches = re.findall(daily_pattern, vision_response, re.IGNORECASE)
        
        for day, high, low in daily_matches[:10]:  # Next 10 days
            data['daily'].append({
                'day': day.capitalize(),
                'high': int(high),
                'low': int(low)
            })
        
        return data
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Understand what the user is asking for"""
        query_lower = query.lower()
        
        intent = {
            'type': 'current',  # default
            'timeframe': 'now',
            'details_requested': [],
            'location': None
        }
        
        # Timeframe detection
        if any(word in query_lower for word in ['today', "today's", 'now', 'current']):
            intent['timeframe'] = 'today'
        elif 'tomorrow' in query_lower:
            intent['timeframe'] = 'tomorrow'
        elif 'week' in query_lower or 'forecast' in query_lower:
            intent['timeframe'] = 'week'
        elif 'hour' in query_lower:
            intent['timeframe'] = 'hourly'
        
        # Detail requests
        if 'rain' in query_lower:
            intent['details_requested'].append('precipitation')
        if 'wind' in query_lower:
            intent['details_requested'].append('wind')
        if 'humid' in query_lower:
            intent['details_requested'].append('humidity')
        if any(word in query_lower for word in ['hot', 'cold', 'warm', 'temperature']):
            intent['details_requested'].append('temperature')
        
        # Location extraction
        location_match = re.search(r'(?:in|at|for)\s+([A-Z][a-zA-Z\s]+)', query)
        if location_match:
            intent['location'] = location_match.group(1).strip()
        
        return intent
    
    def _format_response_by_intent(self, data: Dict, intent: Dict) -> str:
        """Generate natural response based on intent and data"""
        if not data:
            return "I couldn't read the weather information clearly."
        
        response_parts = []
        
        # Location intro
        location = data.get('location', 'your location')
        response_parts.append(f"Looking at the Weather app for {location}")
        
        # Current conditions
        current = data.get('current', {})
        if current.get('temperature'):
            temp = current['temperature']
            condition = current.get('condition', 'current conditions')
            response_parts.append(f"it's {temp}°F with {condition.lower()}")
        
        # Today's forecast
        if intent['timeframe'] in ['today', 'now'] and data.get('today'):
            today = data['today']
            if 'high' in today and 'low' in today:
                response_parts.append(f"Today's high will be {today['high']}°F with a low of {today['low']}°F")
        
        # Specific details requested
        details = data.get('details', {})
        if 'wind' in intent['details_requested'] and 'wind' in details:
            response_parts.append(f"Winds are {details['wind']} mph")
        
        if 'humidity' in intent['details_requested'] and 'humidity' in details:
            response_parts.append(f"Humidity is at {details['humidity']}%")
        
        # Hourly forecast
        if intent['timeframe'] == 'hourly' and data.get('hourly'):
            hourly_summary = self._summarize_hourly(data['hourly'])
            response_parts.append(hourly_summary)
        
        # Weekly forecast
        if intent['timeframe'] == 'week' and data.get('daily'):
            weekly_summary = self._summarize_weekly(data['daily'])
            response_parts.append(weekly_summary)
        
        # Join with proper punctuation
        response = ". ".join(response_parts) + "."
        
        # Add recommendations if relevant
        recommendations = self._generate_recommendations(data, intent)
        if recommendations:
            response += f" {recommendations}"
        
        return response
    
    def _summarize_hourly(self, hourly_data: List[Dict]) -> str:
        """Create concise hourly summary"""
        if not hourly_data:
            return ""
        
        # Group by significant changes
        summary_parts = []
        
        # Next few hours
        next_hours = hourly_data[:3]
        if next_hours:
            temps = [h['temperature'] for h in next_hours]
            avg_temp = sum(temps) // len(temps)
            summary_parts.append(f"Over the next few hours, temperatures will average {avg_temp}°F")
        
        # Look for precipitation
        rainy_hours = [h for h in hourly_data if any(
            cond in h.get('condition', '').lower() 
            for cond in ['rain', 'shower', 'storm']
        )]
        if rainy_hours:
            summary_parts.append(f"Rain expected around {rainy_hours[0]['time']}")
        
        return ". ".join(summary_parts)
    
    def _summarize_weekly(self, daily_data: List[Dict]) -> str:
        """Create concise weekly summary"""
        if not daily_data:
            return ""
        
        # Temperature trends
        highs = [d['high'] for d in daily_data]
        lows = [d['low'] for d in daily_data]
        
        avg_high = sum(highs) // len(highs)
        avg_low = sum(lows) // len(lows)
        
        # Identify warmest/coolest days
        warmest_idx = highs.index(max(highs))
        coolest_idx = highs.index(min(highs))
        
        summary = f"This week will see highs averaging {avg_high}°F and lows around {avg_low}°F. "
        summary += f"{daily_data[warmest_idx]['day']} will be warmest at {max(highs)}°F, "
        summary += f"while {daily_data[coolest_idx]['day']} will be coolest at {min(highs)}°F"
        
        return summary
    
    def _generate_recommendations(self, data: Dict, intent: Dict) -> str:
        """Generate contextual recommendations"""
        recommendations = []
        
        current = data.get('current', {})
        temp = current.get('temperature', 70)
        
        # Temperature-based recommendations
        if temp > 85:
            recommendations.append("Stay hydrated in this heat")
        elif temp < 40:
            recommendations.append("Bundle up for the cold")
        
        # Condition-based recommendations
        condition = current.get('condition', '').lower()
        if 'rain' in condition:
            recommendations.append("Don't forget an umbrella")
        elif 'snow' in condition:
            recommendations.append("Drive carefully in these conditions")
        
        # UV recommendations
        if data.get('details', {}).get('uv_index'):
            uv = int(data['details']['uv_index'])
            if uv >= 6:
                recommendations.append("High UV levels - consider sunscreen")
        
        return ". ".join(recommendations) if recommendations else ""
    
    # Helper methods
    async def _is_app_running(self, app_name: str) -> bool:
        """Check if app is running"""
        try:
            script = f'''
            tell application "System Events"
                return exists process "{app_name}"
            end tell
            '''
            result = await self._run_applescript(script)
            return result.strip().lower() == 'true'
        except:
            return False
    
    async def _open_weather_app(self):
        """Open Weather app"""
        try:
            await self._run_applescript('tell application "Weather" to activate')
        except Exception as e:
            logger.error(f"Failed to open Weather app: {e}")
    
    async def _bring_app_to_front(self):
        """Bring Weather app to front"""
        try:
            script = '''
            tell application "Weather"
                activate
                set frontmost to true
            end tell
            '''
            await self._run_applescript(script)
        except:
            pass
    
    async def _run_applescript(self, script: str) -> str:
        """Execute AppleScript with timeout"""
        try:
            process = await asyncio.create_subprocess_exec(
                'osascript', '-e', script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            # Add 5 second timeout for AppleScript execution
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=5.0
                )
                if stderr:
                    logger.debug(f"AppleScript stderr: {stderr.decode('utf-8')}")
                return stdout.decode('utf-8').strip()
            except asyncio.TimeoutError:
                logger.warning(f"AppleScript timed out: {script[:50]}...")
                process.terminate()
                await process.wait()
                return ""
        except Exception as e:
            logger.error(f"AppleScript error: {e}")
            return ""
    
    def _parse_location_from_vision(self, vision_response: str) -> Optional[Dict]:
        """Parse location info from vision response"""
        # Look for location indicators
        for pattern in self.ui_patterns['my_location']:
            if re.search(pattern, vision_response, re.IGNORECASE):
                # Extract position info
                return {
                    'found': True,
                    'description': vision_response
                }
        return None
    
    async def _click_location(self, location_info: Dict):
        """Click on location in sidebar"""
        # This would integrate with your controller to click
        # For now, we rely on it being already selected
        pass
    
    def _check_cache(self, key: str) -> Optional[Dict]:
        """Check if we have valid cached data"""
        if key in self.cache:
            cached_time, cached_data = self.cache[key]
            if datetime.now() - cached_time < self.cache_duration:
                return cached_data
        return None
    
    def _update_cache(self, key: str, data: Dict):
        """Update cache with new data"""
        self.cache[key] = (datetime.now(), data)
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key from query"""
        # Simple key generation - could be more sophisticated
        return f"weather_{query.lower().replace(' ', '_')}"
    
    def _fallback_response(self, error_msg: str) -> Dict:
        """Fallback response when vision fails"""
        return {
            'success': False,
            'error': error_msg,
            'formatted_response': "I'm having trouble reading the Weather app right now. Please check that it's open and visible.",
            'source': 'vision_failed',
            'timestamp': datetime.now().isoformat()
        }


# Singleton instance for easy access
_unified_weather = None

def get_unified_weather_system(vision_handler=None, controller=None) -> UnifiedVisionWeather:
    """Get or create the unified weather system"""
    global _unified_weather
    if _unified_weather is None:
        _unified_weather = UnifiedVisionWeather(vision_handler, controller)
    return _unified_weather


# Simple test function
async def test_unified_weather():
    """Test the unified weather system"""
    weather = UnifiedVisionWeather()
    
    # Test various queries
    queries = [
        "What's the weather today?",
        "Will it rain tomorrow?",
        "What's the weekly forecast?",
        "How's the weather this hour?",
        "Is it windy outside?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = await weather.get_weather(query)
        print(f"Response: {result.get('formatted_response', 'No response')}")


if __name__ == "__main__":
    asyncio.run(test_unified_weather())