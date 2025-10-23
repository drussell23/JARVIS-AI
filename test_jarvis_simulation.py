#!/usr/bin/env python3
"""
Simulate exactly what happens when you tell JARVIS "living room tv"
This traces the exact code path with debugging
"""
import sys
import os
import asyncio
import logging
import pyautogui

# Setup path
sys.path.insert(0, os.path.dirname(__file__))

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Monkey patch pyautogui to log all movements
original_moveTo = pyautogui.moveTo
original_click = pyautogui.click

def logged_moveTo(x, y, duration=0.0, *args, **kwargs):
    print(f"\n🎯 PyAutoGUI.moveTo({x}, {y}, duration={duration})")
    return original_moveTo(x, y, duration, *args, **kwargs)

def logged_click(x=None, y=None, *args, **kwargs):
    if x is not None and y is not None:
        print(f"🖱️  PyAutoGUI.click({x}, {y})")
    else:
        print(f"🖱️  PyAutoGUI.click(current position)")
    return original_click(x, y, *args, **kwargs)

pyautogui.moveTo = logged_moveTo
pyautogui.click = logged_click

async def simulate_jarvis_command():
    """Simulate the exact flow when user says 'living room tv'"""
    print("\n" + "="*80)
    print("SIMULATING JARVIS COMMAND: 'living room tv'")
    print("="*80 + "\n")

    # Import what JARVIS would use
    from backend.api.unified_command_processor import UnifiedCommandProcessor
    from backend.display.advanced_display_monitor import AdvancedDisplayMonitor

    # Create the processor (what JARVIS uses)
    processor = UnifiedCommandProcessor()

    # The monitor that would be used
    monitor = AdvancedDisplayMonitor()

    # Set the monitor in the processor (simulating what main.py does)
    processor.display_monitor = monitor

    # Simulate the command
    command_text = "living room tv"

    print(f"Processing command: '{command_text}'")
    print("-"*80)

    # Process like JARVIS would
    result = await processor.process_command(command_text)

    print("\n" + "="*80)
    print("RESULT:")
    print(f"Success: {result.get('success')}")
    print(f"Response: {result.get('response')}")
    if 'display_name' in result:
        print(f"Display: {result['display_name']}")
    if 'error' in result:
        print(f"Error: {result['error']}")
    print("="*80 + "\n")

    return result

if __name__ == "__main__":
    try:
        asyncio.run(simulate_jarvis_command())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()