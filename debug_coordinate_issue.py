#!/usr/bin/env python3
"""
Debug script to trace why coordinates are changing
"""

import sys
import asyncio
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Monkey-patch pyautogui to trace all calls
import pyautogui
original_dragTo = pyautogui.dragTo
original_moveTo = pyautogui.moveTo

def traced_dragTo(x, y, *args, **kwargs):
    print(f"\n🔍 pyautogui.dragTo called with: x={x}, y={y}")
    import traceback
    traceback.print_stack(limit=5)
    result = original_dragTo(x, y, *args, **kwargs)
    final = pyautogui.position()
    print(f"📍 Final position after dragTo: {final}")
    return result

def traced_moveTo(x, y, *args, **kwargs):
    print(f"\n🔍 pyautogui.moveTo called with: x={x}, y={y}")
    import traceback
    traceback.print_stack(limit=5)
    result = original_moveTo(x, y, *args, **kwargs)
    final = pyautogui.position()
    print(f"📍 Final position after moveTo: {final}")
    return result

pyautogui.dragTo = traced_dragTo
pyautogui.moveTo = traced_moveTo

async def test_clicker():
    """Test the clicker with tracing"""
    print("\n" + "=" * 80)
    print("TRACING COORDINATE ISSUE")
    print("=" * 80)

    from backend.display.control_center_clicker_factory import get_best_clicker

    clicker = get_best_clicker(force_new=True)
    print(f"\nUsing clicker: {clicker.__class__.__name__}")

    print("\n🎯 Attempting to open Control Center...")
    result = await clicker.open_control_center()

    if result.success:
        print(f"\n✅ Success: {result.success}")
        print(f"Coordinates used: {result.coordinates}")
    else:
        print(f"\n❌ Failed: {result.error}")

if __name__ == "__main__":
    asyncio.run(test_clicker())