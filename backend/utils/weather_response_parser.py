#!/usr/bin/env python3
"""
Weather Response Parser - Extracts weather information from vision responses
"""

import re
from typing import Optional, Dict, Any

class WeatherResponseParser:
    """Parse and extract weather information from vision responses"""
    
    @staticmethod
    def extract_weather_info(response: str) -> str:
        """Extract weather-specific information from a vision response"""
        
        # Common weather-related keywords to look for
        weather_keywords = [
            r'\d+°[FCfc]?\b',  # Temperature patterns (77°F, 25°C, etc)
            r'\d+\s*degrees',  # Temperature in words
            r'sunny|cloudy|rainy|partly cloudy|overcast|clear|foggy|snow',
            r'rain|precipitation|humidity|wind',
            r'high.*\d+|low.*\d+',
            r'forecast',
            r'today|tonight|tomorrow',
            r'weather\s+app.*?(?:shows?|displays?|indicates?)',
        ]
        
        # Extract sentences containing weather information
        sentences = response.split('.')
        weather_sentences = []
        
        for sentence in sentences:
            # Check if sentence contains weather keywords
            if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in weather_keywords):
                # Skip sentences about UI elements or development
                skip_patterns = [
                    'code', 'VS Code', 'editor', 'file', 'project', 'development',
                    'screen', 'window', 'interface', 'button', 'menu'
                ]
                if not any(word in sentence.lower() for word in skip_patterns):
                    weather_sentences.append(sentence.strip())
        
        # If we found weather information, join it
        if weather_sentences:
            result = '. '.join(weather_sentences)
            # Clean up the result
            result = re.sub(r'\s+', ' ', result)  # Remove extra spaces
            result = result.strip()
            
            # Ensure proper sentence ending
            if result and not result.endswith('.'):
                result += '.'
                
            return result
        
        # Fallback: Look for any temperature mentions
        temp_match = re.search(r'(\d+)\s*°[FCfc]?\b', response)
        if temp_match:
            return f"The current temperature is {temp_match.group(0)}."
        
        # If no weather info found, return a generic message
        return "I couldn't extract specific weather information from what I see."
    
    @staticmethod
    def format_weather_response(raw_info: str, location: Optional[str] = None) -> str:
        """Format weather information into a natural response"""
        
        # Extract key weather data
        temp_match = re.search(r'(\d+)\s*°([FCfc])?', raw_info)
        condition_match = re.search(
            r'(sunny|cloudy|rainy|partly cloudy|overcast|clear|foggy|snow\w*)', 
            raw_info, re.IGNORECASE
        )
        
        # Build formatted response
        response_parts = []
        
        # Add location if known and valid
        if location and not any(word in location.lower() for word in ['yes', 'sir', 'i', 'can', 'see']):
            response_parts.append(f"In {location}")
        
        # Add current conditions
        if temp_match and condition_match:
            temp = temp_match.group(1)
            unit = temp_match.group(2) or 'F'
            condition = condition_match.group(1).lower()
            response_parts.append(f"it's currently {temp}°{unit.upper()} and {condition}")
        elif temp_match:
            temp = temp_match.group(1)
            unit = temp_match.group(2) or 'F'
            response_parts.append(f"the temperature is {temp}°{unit.upper()}")
        elif condition_match:
            condition = condition_match.group(1).lower()
            response_parts.append(f"it's {condition}")
        
        # Look for high/low temperatures
        high_match = re.search(r'high[:\s]+(\d+)', raw_info, re.IGNORECASE)
        low_match = re.search(r'low[:\s]+(\d+)', raw_info, re.IGNORECASE)
        
        if high_match and low_match:
            response_parts.append(
                f"with a high of {high_match.group(1)}° and low of {low_match.group(1)}°"
            )
        elif high_match:
            response_parts.append(f"with a high of {high_match.group(1)}°")
        elif low_match:
            response_parts.append(f"with a low of {low_match.group(1)}°")
        
        # Join parts into natural sentence
        if response_parts:
            response = ', '.join(response_parts) + '.'
            # Capitalize first letter
            return response[0].upper() + response[1:]
        
        # Fallback to cleaned raw info
        return raw_info