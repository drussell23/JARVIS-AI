#!/usr/bin/env python3
"""Debug location extraction"""

import re

def test_extraction():
    """Test location extraction patterns"""
    
    # Test queries
    test_queries = [
        "What's the weather in Tokyo?",
        "what's the weather in tokyo",
        "Tell me the weather for New York",
        "What is the weather like in London",
        "How's the weather at Paris",
        "Tokyo weather",
        "weather in Sydney"
    ]
    
    # Pattern that should work
    pattern = r'weather\s+(?:in|at|for)\s+([A-Za-z]+(?:\s+[A-Za-z]+)*?)(?:\s|$)'
    
    print("Testing location extraction:")
    print("="*60)
    
    for query in test_queries:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            location = match.group(1).strip()
            print(f"✅ '{query}' -> '{location}'")
        else:
            print(f"❌ '{query}' -> No match")
            
            # Try debugging why
            if "weather" in query.lower():
                # Check what's after weather
                weather_pos = query.lower().find("weather")
                after_weather = query[weather_pos + 7:].strip()
                print(f"   After 'weather': '{after_weather}'")

if __name__ == "__main__":
    test_extraction()