#!/usr/bin/env python3
"""
Test the simple display connector
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

async def main():
    from backend.display.simple_display_connector import connect_living_room_tv
    
    print("\n" + "="*80)
    print("Testing Simple Display Connector")
    print("="*80)
    print("\nThis will click in sequence:")
    print("  1. Control Center (1235, 10)")
    print("  2. Screen Mirroring (1396, 177)")  
    print("  3. Living Room TV (1223, 115)")
    print("\nWatch the mouse movement!")
    print("-"*80)
    
    input("\nPress ENTER to start the test...")
    
    result = await connect_living_room_tv()
    
    print("\n" + "="*80)
    print("RESULT:")
    print(f"  Success: {result['success']}")
    print(f"  Duration: {result.get('duration', 0):.2f}s")
    if result['success']:
        print(f"  Method: {result['method']}")
    else:
        print(f"  Error: {result.get('error', 'Unknown')}")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
