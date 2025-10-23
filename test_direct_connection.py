#!/usr/bin/env python3
"""Direct test of display connection to debug coordinates"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

async def test():
    from backend.display.control_center_clicker_factory import get_best_clicker
    
    print("\n" + "="*80)
    print("Getting clicker...")
    clicker = get_best_clicker(force_new=True, enable_verification=False)
    
    print(f"Clicker type: {clicker.__class__.__name__}")
    print("\nAttempting to click Control Center...")
    print("WATCH THE MOUSE CAREFULLY - WHERE DOES IT GO?")
    print("="*80 + "\n")
    
    result = await clicker.click("control_center")
    
    print("\n" + "="*80)
    print(f"Result: {result}")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(test())
