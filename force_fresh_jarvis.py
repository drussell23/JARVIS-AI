#!/usr/bin/env python3
"""
Force JARVIS to use fresh instances with the drag fix
Clears all caches and ensures proper initialization
"""

import os
import sys
import time
import subprocess
from pathlib import Path

print("=" * 80)
print("FORCING FRESH JARVIS WITH DRAG FIX")
print("=" * 80)

# Step 1: Kill any existing JARVIS processes
print("\n1. Killing existing JARVIS processes...")
subprocess.run("pkill -f 'python.*main.py'", shell=True)
time.sleep(2)

# Step 2: Clear all caches
print("\n2. Clearing caches...")
cache_files = [
    "/Users/derekjrussell/.jarvis/control_center_cache.json",
    "/Users/derekjrussell/.jarvis/display_cache.json",
    "/tmp/jarvis_cache.json"
]

for cache_file in cache_files:
    if Path(cache_file).exists():
        os.remove(cache_file)
        print(f"   ✅ Removed {cache_file}")
    else:
        print(f"   ⏭️  {cache_file} not found")

# Step 3: Clear Python cache
print("\n3. Clearing Python cache...")
import shutil
backend_dir = Path(__file__).parent / "backend"
for pycache in backend_dir.rglob("__pycache__"):
    shutil.rmtree(pycache, ignore_errors=True)
    print(f"   ✅ Removed {pycache.relative_to(backend_dir.parent)}")

# Step 4: Set environment to force fresh instances
print("\n4. Setting environment for fresh instances...")
os.environ["JARVIS_FORCE_FRESH"] = "1"
os.environ["JARVIS_NO_CACHE"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# Step 5: Import and verify the drag fix is present
print("\n5. Verifying drag fix is in place...")
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.display.adaptive_control_center_clicker import AdaptiveControlCenterClicker
import inspect

# Check if dragTo is in the code
source = inspect.getsource(AdaptiveControlCenterClicker.click)
if "dragTo" in source:
    print("   ✅ Drag fix confirmed in AdaptiveControlCenterClicker")
else:
    print("   ❌ WARNING: Drag fix not found in AdaptiveControlCenterClicker!")

# Check UAE clicker
try:
    from backend.display.uae_enhanced_control_center_clicker import UAEEnhancedControlCenterClicker
    source = inspect.getsource(UAEEnhancedControlCenterClicker._execute_click)
    if "dragTo" in source:
        print("   ✅ Drag fix confirmed in UAEEnhancedControlCenterClicker")
    else:
        print("   ❌ WARNING: Drag fix not found in UAEEnhancedControlCenterClicker!")
except Exception as e:
    print(f"   ⚠️  Could not verify UAE clicker: {e}")

# Step 6: Test the clicker directly
print("\n6. Testing clicker with fresh instance...")
import asyncio

async def test_fresh_clicker():
    from backend.display.control_center_clicker_factory import get_best_clicker

    # Force new instance (no singleton)
    clicker = get_best_clicker(
        vision_analyzer=None,
        cache_ttl=0,  # Disable cache
        enable_verification=False  # Skip verification for speed
    )

    print(f"   Using clicker: {clicker.__class__.__name__}")

    # Test opening Control Center
    result = await clicker.open_control_center()
    if result.success:
        print(f"   ✅ Control Center opened successfully!")
        print(f"      Method: {result.method_used}")
        print(f"      Coordinates: {result.coordinates}")

        # Close it
        import pyautogui
        await asyncio.sleep(1)
        pyautogui.press('escape')
        print("   ✅ Closed Control Center")
        return True
    else:
        print(f"   ❌ Failed to open Control Center: {result.error}")
        return False

success = asyncio.run(test_fresh_clicker())

if success:
    print("\n" + "=" * 80)
    print("✅ FRESH INSTANCE TEST PASSED!")
    print("=" * 80)
    print("\nThe drag fix is working with fresh instances.")
    print("Now starting JARVIS with fresh configuration...")

    # Step 7: Start JARVIS fresh
    print("\n7. Starting fresh JARVIS...")
    subprocess.Popen(
        ["python", "backend/main.py"],
        env={**os.environ, "JARVIS_FRESH_START": "1"}
    )

    print("\n✅ JARVIS started with fresh instances and drag fix!")
    print("Try asking JARVIS to connect to Living Room TV now.")
else:
    print("\n" + "=" * 80)
    print("❌ FRESH INSTANCE TEST FAILED")
    print("=" * 80)
    print("\nThere may be deeper issues. Check the implementation.")