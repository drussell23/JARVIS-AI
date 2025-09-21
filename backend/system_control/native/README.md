# JARVIS Weather Native Tool

A Swift command-line tool that provides real-time weather data using Apple's native WeatherKit framework.

## Features

- **Native WeatherKit Integration**: Direct access to Apple's weather data
- **Current Location Support**: Automatic location detection using CoreLocation
- **City Weather**: Get weather for any city worldwide
- **Rich Data**: Temperature, conditions, humidity, wind, UV index, alerts, and more
- **Forecast Support**: Optional hourly and daily forecasts
- **Error Handling**: Graceful error responses in JSON format
- **No API Keys**: Uses Apple's native weather service
- **Dynamic & Robust**: No hardcoded values, adapts to location and conditions

## Building

```bash
# Run the build script
./build.sh

# Or compile manually
swiftc jarvis-weather.swift -o jarvis-weather -O -framework CoreLocation -framework WeatherKit -framework ArgumentParser
```

## Usage

### Current Location Weather
```bash
# Basic weather
./jarvis-weather current

# With forecast
./jarvis-weather current --forecast

# Pretty printed
./jarvis-weather current --pretty
```

### City Weather
```bash
# Single word city
./jarvis-weather city Toronto

# Multi-word city
./jarvis-weather city "San Francisco"

# With country
./jarvis-weather city "London, UK"

# With forecast
./jarvis-weather city Tokyo --forecast --pretty
```

### Location Information
```bash
./jarvis-weather location --pretty
```

### Temperature Only
```bash
./jarvis-weather temperature
```

## Output Format

All commands return JSON with comprehensive weather data:

```json
{
  "location": "Toronto, ON",
  "temperature": 19.0,
  "temperature_f": 66.0,
  "feels_like": 17.0,
  "feels_like_f": 63.0,
  "condition": "Partly Cloudy",
  "description": "partly cloudy",
  "humidity": 65.0,
  "wind_speed": 15.0,
  "wind_speed_mph": 9.3,
  "wind_direction": "NW",
  "uv_index": 3,
  "alerts": [],
  ...
}
```

## Error Handling

Errors are returned as JSON:

```json
{
  "error": "location_failed",
  "message": "Location access is denied...",
  "code": "LOCATION_ERROR"
}
```

## Permissions

On first run, macOS will ask for:
- **Location Services**: Required for current location weather
- **Network Access**: Required for weather data

Grant these permissions in System Preferences > Privacy & Security.

## Integration with JARVIS

This tool is designed to be called from Python:

```python
import subprocess
import json

result = subprocess.run(
    ["./jarvis-weather", "current"],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    weather = json.loads(result.stdout)
    print(f"It's {weather['temperature']}°C in {weather['location']}")
```

## Requirements

- macOS 13.0 or later (Ventura+)
- Swift 5.7 or later
- WeatherKit entitlement (automatically available for CLI tools)

## License

Part of the JARVIS AI Agent project.