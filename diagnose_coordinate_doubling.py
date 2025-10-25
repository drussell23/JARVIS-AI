#!/usr/bin/env python3
"""
Comprehensive diagnostic for coordinate doubling issue
This will help identify exactly where and why coordinates are being doubled
"""
import sys
import os
import pyautogui
from AppKit import NSScreen

print("\n" + "="*80)
print("COORDINATE DOUBLING DIAGNOSTIC")
print("="*80 + "\n")

# 1. System Information
print("1. SYSTEM INFORMATION:")
print(f"   Platform: {sys.platform}")
print(f"   Python: {sys.version}")
print(f"   PyAutoGUI version: {pyautogui.__version__}")

# 2. Display Information
main_screen = NSScreen.mainScreen()
backing_scale = main_screen.backingScaleFactor()
frame = main_screen.frame()
logical_size = (frame.size.width, frame.size.height)

print(f"\n2. DISPLAY INFORMATION:")
print(f"   Backing Scale Factor: {backing_scale}x (Retina)")
print(f"   Logical Size: {logical_size}")
print(f"   Physical Size: ({logical_size[0] * backing_scale}, {logical_size[1] * backing_scale})")
print(f"   PyAutoGUI Screen Size: {pyautogui.size()}")

# 3. Check for PyAutoGUI modifications
print(f"\n3. PYAUTOGUI FUNCTION CHECK:")
print(f"   moveTo function: {pyautogui.moveTo}")
print(f"   moveTo module: {pyautogui.moveTo.__module__}")
print(f"   click function: {pyautogui.click}")

# Check if functions have been wrapped
if hasattr(pyautogui.moveTo, '__wrapped__'):
    print(f"   ⚠️  moveTo has been WRAPPED")
if hasattr(pyautogui.click, '__wrapped__'):
    print(f"   ⚠️  click has been WRAPPED")

# 4. Test actual mouse movement
print(f"\n4. MOUSE MOVEMENT TEST:")
test_coords = [
    (100, 100, "Simple test"),
    (1236, 12, "Control Center target"),
]

for x, y, desc in test_coords:
    print(f"\n   Test: {desc}")
    print(f"   Sending: ({x}, {y})")

    # Move mouse
    pyautogui.moveTo(x, y, duration=0.3)

    # Get actual position
    actual = pyautogui.position()
    print(f"   Actual: ({actual.x}, {actual.y})")

    # Check if doubled
    if actual.x == x * 2 and actual.y == y * 2:
        print(f"   ❌ DOUBLED! Coordinates were multiplied by 2")
    elif actual.x != x or actual.y != y:
        print(f"   ❌ MISMATCH!")
        if actual.x > 0:
            ratio_x = actual.x / x
            ratio_y = actual.y / y if y > 0 else 0
            print(f"   Ratio: X={ratio_x:.2f}, Y={ratio_y:.2f}")
    else:
        print(f"   ✅ CORRECT")

# 5. Check imports that might affect coordinates
print(f"\n5. LOADED MODULES CHECK:")
suspicious_modules = [
    'coordinate_fix',
    'adaptive_control_center_clicker',
    'direct_vision_clicker',
]

for mod_name in suspicious_modules:
    if mod_name in sys.modules:
        mod = sys.modules[mod_name]
        print(f"   ⚠️  {mod_name} is loaded from: {mod.__file__ if hasattr(mod, '__file__') else 'unknown'}")
    else:
        print(f"   ✅ {mod_name} not loaded")

# 6. Check environment variables
print(f"\n6. ENVIRONMENT VARIABLES:")
env_vars = ['DISPLAY', 'OBJC_DISABLE_INITIALIZE_FORK_SAFETY', 'PYTHONUNBUFFERED']
for var in env_vars:
    value = os.environ.get(var, 'NOT SET')
    print(f"   {var}: {value}")

# 7. Save diagnostic results
print(f"\n7. SAVING DIAGNOSTIC LOG...")
log_path = "/tmp/jarvis_coordinate_diagnostic.log"
with open(log_path, "w") as f:
    f.write("="*80 + "\n")
    f.write("JARVIS COORDINATE DOUBLING DIAGNOSTIC\n")
    f.write("="*80 + "\n\n")
    f.write(f"Platform: {sys.platform}\n")
    f.write(f"Backing Scale: {backing_scale}x\n")
    f.write(f"Logical Size: {logical_size}\n")
    f.write(f"PyAutoGUI Size: {pyautogui.size()}\n")
    f.write(f"\nPyAutoGUI moveTo: {pyautogui.moveTo}\n")
    f.write(f"PyAutoGUI moveTo module: {pyautogui.moveTo.__module__}\n")

print(f"   ✅ Saved to: {log_path}")

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80 + "\n")

print("RECOMMENDATION:")
if backing_scale == 2.0:
    print("   Your system has 2x Retina scaling")
    print("   If mouse goes to (2475, 15) when you send (1236, 12),")
    print("   then something is applying the scale factor incorrectly.")
print("\n")