#!/usr/bin/env python3
"""Simple Toronto Weather Test"""

import asyncio
import os
from dotenv import load_dotenv

async def test():
    load_dotenv()
    
    # Direct weather service test
    from services.weather_service import WeatherService
    weather_service = WeatherService(os.getenv("OPENWEATHER_API_KEY"))
    
    weather = await weather_service.get_current_weather()
    print(f"Location: {weather.get('location')}")
    print(f"Temperature: {weather.get('temperature')}°C")
    
    await weather_service.close()

if __name__ == "__main__":
    os.chdir('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')
    asyncio.run(test())