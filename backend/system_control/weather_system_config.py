"""
Weather System Configuration
Central configuration for the unified weather system
"""

from typing import Optional
from .weather_bridge_unified import UnifiedWeatherBridge
from .unified_vision_weather import get_unified_weather_system

# Global weather system instance
_weather_bridge = None


def initialize_weather_system(vision_handler=None, controller=None):
    """
    Initialize the unified weather system with vision and controller
    This should be called once at startup with the proper handlers
    """
    global _weather_bridge
    
    # Initialize the unified system
    weather_system = get_unified_weather_system(vision_handler, controller)
    
    # Create the bridge
    _weather_bridge = UnifiedWeatherBridge(vision_handler, controller)
    
    return _weather_bridge


def get_weather_system() -> Optional[UnifiedWeatherBridge]:
    """
    Get the initialized weather system
    Returns None if not initialized
    """
    return _weather_bridge


# For backward compatibility
class WeatherBridge:
    """Compatibility wrapper for old code"""
    
    def __init__(self):
        self._bridge = get_weather_system()
        if not self._bridge:
            # Create without handlers if not initialized
            self._bridge = UnifiedWeatherBridge()
    
    async def get_current_weather(self, use_cache=False):
        """Get current weather (compatibility method)"""
        result = await self._bridge.get_weather("What's the current weather?")
        
        # Convert to old format
        if result.get('success') and result.get('data'):
            data = result['data']
            current = data.get('current', {})
            
            return {
                'location': data.get('location', 'Unknown'),
                'temperature': current.get('temperature', 20),
                'temperature_f': current.get('temperature', 20) * 9/5 + 32 if current.get('temperature') else 68,
                'condition': current.get('condition', 'Unknown'),
                'description': current.get('condition', '').lower(),
                'humidity': data.get('details', {}).get('humidity', 50),
                'wind_speed': float(data.get('details', {}).get('wind', 0)),
                'wind_speed_mph': float(data.get('details', {}).get('wind', 0)),
                'source': 'vision',
                'timestamp': result.get('timestamp')
            }
        
        # Return fallback
        return {
            'location': 'your area',
            'temperature': 20,
            'temperature_f': 68,
            'condition': 'unavailable',
            'description': 'weather data temporarily unavailable',
            'source': 'unavailable'
        }
    
    async def get_weather_by_city(self, city: str, use_cache=False):
        """Get weather by city (compatibility method)"""
        result = await self._bridge.get_weather(f"What's the weather in {city}?")
        
        # Use same conversion as get_current_weather
        if result.get('success'):
            weather_dict = await self.get_current_weather()
            weather_dict['location'] = city
            return weather_dict
        
        return None
    
    def is_weather_query(self, text: str) -> bool:
        """Check if text is weather query"""
        return self._bridge.is_weather_query(text)