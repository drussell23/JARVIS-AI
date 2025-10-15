#!/usr/bin/env python3
"""
Test response variation - should get different responses each time
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_variation():
    from api.vision_command_handler import vision_command_handler
    
    # Initialize
    await vision_command_handler.initialize_intelligence()
    
    # Test same query multiple times
    query = "can you see my battery percentage?"
    
    print(f"Testing query 3 times: '{query}'\n")
    
    responses = []
    for i in range(3):
        print(f"Attempt {i+1}:")
        result = await vision_command_handler.handle_command(query)
        response = result.get('response')
        responses.append(response)
        print(f"Response: {response}\n")
        await asyncio.sleep(1)  # Brief pause between queries
        
    # Check variation
    unique_responses = set(responses)
    print(f"\n✅ Got {len(unique_responses)} unique responses out of {len(responses)} queries")
    
    if len(unique_responses) == len(responses):
        print("🎉 Perfect variation - every response was unique!")
    else:
        print("⚠️  Some responses were identical")
        
if __name__ == "__main__":
    asyncio.run(test_variation())