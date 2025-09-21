#!/usr/bin/env python3
"""
Fallback weather provider that uses system commands and APIs
This is used when Swift tool is not available or cannot be built
"""

import json
import subprocess
import sys
import os
import urllib.request
import urllib.parse
from datetime import datetime

def get_location_from_ip():
    """Get location from IP address"""
    try:
        response = urllib.request.urlopen('http://ip-api.com/json/', timeout=3)
        data = json.loads(response.read())
        if data.get('status') == 'success':
            return {
                'latitude': data.get('lat'),
                'longitude': data.get('lon'),
                'city': data.get('city'),
                'state': data.get('regionName'),
                'country': data.get('country'),
                'timezone': data.get('timezone'),
                'source': 'ip-api'
            }
    except:
        pass
    return None

def get_weather_from_wttr(location=None):
    """Get weather from wttr.in (no API key needed)"""
    try:
        if location:
            url = f"https://wttr.in/{urllib.parse.quote(location)}?format=j1"
        else:
            # Get current location first
            loc_data = get_location_from_ip()
            if loc_data:
                url = f"https://wttr.in/{loc_data['latitude']},{loc_data['longitude']}?format=j1"
            else:
                url = "https://wttr.in/?format=j1"
        
        response = urllib.request.urlopen(url, timeout=5)
        data = json.loads(response.read())
        
        current = data.get('current_condition', [{}])[0]
        query_location = data.get('request', [{}])[0].get('query', location or 'Unknown')
        
        # Clean up location format
        if query_location.startswith('Lat ') and ' and Lon ' in query_location:
            nearest_area = data.get('nearest_area', [{}])[0]
            area_name = nearest_area.get('areaName', [{}])[0].get('value', '')
            region = nearest_area.get('region', [{}])[0].get('value', '')
            country = nearest_area.get('country', [{}])[0].get('value', '')
            
            if area_name:
                if region and country == 'United States':
                    query_location = f"{area_name}, {region}"
                elif country:
                    query_location = f"{area_name}, {country}"
                else:
                    query_location = area_name
        
        temp_c = int(current.get('temp_C', 0))
        temp_f = int(current.get('temp_F', temp_c * 9/5 + 32))
        feels_like_c = int(current.get('FeelsLikeC', temp_c))
        feels_like_f = int(current.get('FeelsLikeF', feels_like_c * 9/5 + 32))
        
        return {
            'location': query_location,
            'temperature': temp_c,
            'temperature_f': temp_f,
            'feels_like': feels_like_c,
            'feels_like_f': feels_like_f,
            'condition': current.get('weatherDesc', [{}])[0].get('value', 'Unknown'),
            'description': current.get('weatherDesc', [{}])[0].get('value', 'Unknown').lower(),
            'humidity': int(current.get('humidity', 0)),
            'pressure': int(current.get('pressure', 1013)),
            'wind_speed': float(current.get('windspeedKmph', 0)),
            'wind_speed_mph': float(current.get('windspeedMiles', 0)),
            'wind_direction': current.get('winddir16Point', 'N'),
            'wind_direction_degrees': float(current.get('winddirDegree', 0)),
            'visibility': float(current.get('visibility', 10)),
            'visibility_miles': float(current.get('visibilityMiles', 6)),
            'uv_index': int(current.get('uvIndex', 0)),
            'cloud_cover': int(current.get('cloudcover', 0)),
            'precipitation_chance': 50 if float(current.get('precipMM', 0)) > 0 else 0,
            'dew_point': temp_c - 5,  # Rough estimate
            'dew_point_f': (temp_c - 5) * 9/5 + 32,
            'source': 'wttr.in',
            'timestamp': datetime.now().isoformat(),
            'timezone': 'UTC',
            'alerts': []
        }
        
    except Exception as e:
        return {
            'error': 'weather_failed',
            'message': str(e),
            'code': 'WEATHER_ERROR'
        }

def get_macos_weather():
    """Try to get weather from macOS Weather app via AppleScript"""
    try:
        script = '''
        tell application "Weather" to launch
        delay 1
        tell application "System Events"
            tell process "Weather"
                set frontmost to true
                set allText to value of every static text of window 1
                return allText as string
            end tell
        end tell
        '''
        
        result = subprocess.run(['osascript', '-e', script], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            # Parse the output (this is very basic)
            output = result.stdout
            # Look for temperature patterns
            import re
            temp_match = re.search(r'(\d+)°', output)
            if temp_match:
                temp = int(temp_match.group(1))
                # Assume Fahrenheit if in US
                temp_f = temp
                temp_c = round((temp - 32) * 5/9)
                
                return {
                    'location': 'Current Location',
                    'temperature': temp_c,
                    'temperature_f': temp_f,
                    'source': 'macOS Weather',
                    'timestamp': datetime.now().isoformat()
                }
    except:
        pass
    return None

def main():
    """Main entry point"""
    args = sys.argv[1:]
    
    # Parse arguments
    command = args[0] if args else 'current'
    pretty = '--pretty' in args or '-p' in args
    
    # Handle commands
    if command == '--version' or command == '-v':
        print('jarvis-weather-fallback 1.0.0')
        return
    
    if command == 'city' and len(args) > 1:
        # Get city name (remove flags)
        city_parts = [arg for arg in args[1:] if not arg.startswith('-')]
        city = ' '.join(city_parts)
        weather_data = get_weather_from_wttr(city)
    elif command == 'location':
        location_data = get_location_from_ip()
        if location_data:
            print(json.dumps(location_data, indent=2 if pretty else None))
            return
        else:
            weather_data = {'error': 'location_failed', 'message': 'Could not determine location'}
    elif command == 'temperature':
        weather_data = get_weather_from_wttr()
        if 'error' not in weather_data:
            # Simplify to just temperature
            weather_data = {
                'temperature': weather_data['temperature'],
                'temperature_f': weather_data['temperature_f'],
                'feels_like': weather_data.get('feels_like', weather_data['temperature']),
                'feels_like_f': weather_data.get('feels_like_f', weather_data['temperature_f']),
                'location': weather_data['location']
            }
    else:  # current
        # Try macOS first
        weather_data = get_macos_weather()
        if not weather_data:
            weather_data = get_weather_from_wttr()
    
    # Output JSON
    print(json.dumps(weather_data, indent=2 if pretty else None))

if __name__ == '__main__':
    main()