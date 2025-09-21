"""
JARVIS Location Service Python Wrapper
Provides precise location data using macOS Core Location
"""

import json
import subprocess
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class LocationService:
    """Bridge to Swift Core Location service"""
    
    def __init__(self):
        self.service_dir = Path(__file__).parent
        self.app_bundle = self.service_dir / "JarvisLocationService.app"
        self.cli_binary = self.service_dir / "jarvis-location"
        self.cache_file = self.service_dir / ".location_cache.json"
        self.cache_duration = timedelta(minutes=5)
        
    def get_current_location(self) -> Optional[Dict]:
        """Get current location with caching"""
        # Check cache first
        cached = self._get_cached_location()
        if cached:
            logger.info(f"Using cached location: {cached.get('city', 'Unknown')}")
            return cached
            
        # Try to get fresh location
        location = self._request_location()
        
        if location and location.get('status') == 'success':
            self._save_to_cache(location)
            logger.info(f"Got fresh location: {location.get('city', 'Unknown')}")
            return location
        
        return None
        
    def _request_location(self) -> Optional[Dict]:
        """Request location from Swift service"""
        try:
            # Try app bundle first (has better permissions)
            if self.app_bundle.exists():
                executable = self.app_bundle / "Contents" / "MacOS" / "JarvisLocationService"
                cmd = [str(executable)]
            elif self.cli_binary.exists():
                cmd = [str(self.cli_binary)]
            else:
                logger.error("No location service binary found")
                return None
                
            # Run with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout:
                return json.loads(result.stdout)
            else:
                logger.error(f"Location service failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("Location service timed out")
            return None
        except json.JSONDecodeError:
            logger.error("Invalid JSON from location service")
            return None
        except Exception as e:
            logger.error(f"Location service error: {e}")
            return None
            
    def _get_cached_location(self) -> Optional[Dict]:
        """Get location from cache if valid"""
        try:
            if not self.cache_file.exists():
                return None
                
            with open(self.cache_file, 'r') as f:
                cache = json.load(f)
                
            # Check if cache is still valid
            cached_time = datetime.fromisoformat(cache['timestamp'].replace('Z', '+00:00'))
            if datetime.now().astimezone() - cached_time < self.cache_duration:
                return cache
                
        except Exception:
            pass
            
        return None
        
    def _save_to_cache(self, location: Dict):
        """Save location to cache"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(location, f)
        except Exception as e:
            logger.error(f"Failed to save location cache: {e}")
            
    def get_coordinates(self) -> Optional[Tuple[float, float]]:
        """Get just coordinates (latitude, longitude)"""
        location = self.get_current_location()
        if location and location.get('status') == 'success':
            return (location['latitude'], location['longitude'])
        return None
        
    def get_city_info(self) -> Optional[Dict]:
        """Get city, region, country info"""
        location = self.get_current_location()
        if location and location.get('status') == 'success':
            return {
                'city': location.get('city'),
                'region': location.get('region'),
                'country': location.get('country'),
                'coordinates': (location['latitude'], location['longitude'])
            }
        return None


# Quick test when run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    service = LocationService()
    location = service.get_current_location()
    
    if location:
        print(f"Location: {location.get('city', 'Unknown')}, {location.get('region', 'Unknown')}")
        print(f"Coordinates: {location['latitude']}, {location['longitude']}")
        print(f"Accuracy: {location['accuracy']}m")
    else:
        print("Failed to get location")
