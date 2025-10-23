#!/usr/bin/env python3
"""
PyAutoGUI Intercept - Monkey-patch to log ALL coordinate operations
This will help us find where coordinates are getting doubled
"""

import pyautogui
import traceback

# Save originals
_original_moveTo = pyautogui.moveTo
_original_dragTo = pyautogui.dragTo
_original_click = pyautogui.click

LOG_FILE = "/tmp/pyautogui_intercept.log"

def log_call(func_name, args, kwargs, stack_depth=10):
    """Log a function call with its stack trace"""
    with open(LOG_FILE, "a") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"[INTERCEPT] {func_name} called\n")
        f.write(f"  Args: {args}\n")
        f.write(f"  Kwargs: {kwargs}\n")
        f.write(f"\nCall stack (last {stack_depth} frames):\n")

        stack = traceback.extract_stack()
        for frame in stack[-(stack_depth+1):-1]:  # Skip the log_call frame itself
            if 'backend' in frame.filename or 'JARVIS' in frame.filename:
                f.write(f"  {frame.filename}:{frame.lineno} in {frame.name}\n")
                f.write(f"    {frame.line}\n")
        f.write(f"\n")

def intercepted_moveTo(x, y, duration=None, **kwargs):
    """Intercepted moveTo"""
    log_call(f"moveTo({x}, {y}, duration={duration})", (x, y), kwargs)

    # Check if coordinates seem doubled
    screen_width, _ = pyautogui.size()
    if x > screen_width * 1.5:
        with open(LOG_FILE, "a") as f:
            f.write(f"🚨 WARNING: X coordinate {x} exceeds 1.5x screen width ({screen_width})!\n")
            f.write(f"   This looks like coordinate doubling!\n")

    result = _original_moveTo(x, y, duration=duration, **kwargs)

    # Log final position
    final_pos = pyautogui.position()
    with open(LOG_FILE, "a") as f:
        f.write(f"[INTERCEPT] After moveTo: Mouse at ({final_pos.x}, {final_pos.y})\n")

    return result

def intercepted_dragTo(x, y, duration=None, button='left', **kwargs):
    """Intercepted dragTo"""
    log_call(f"dragTo({x}, {y}, duration={duration}, button={button})", (x, y), kwargs)

    # Check if coordinates seem doubled
    screen_width, _ = pyautogui.size()
    if x > screen_width * 1.5:
        with open(LOG_FILE, "a") as f:
            f.write(f"🚨 WARNING: X coordinate {x} exceeds 1.5x screen width ({screen_width})!\n")
            f.write(f"   This looks like coordinate doubling!\n")

    result = _original_dragTo(x, y, duration=duration, button=button, **kwargs)

    # Log final position
    final_pos = pyautogui.position()
    with open(LOG_FILE, "a") as f:
        f.write(f"[INTERCEPT] After dragTo: Mouse at ({final_pos.x}, {final_pos.y})\n")

    return result

def intercepted_click(x=None, y=None, **kwargs):
    """Intercepted click"""
    if x is not None and y is not None:
        log_call(f"click({x}, {y})", (x, y), kwargs)

        # Check if coordinates seem doubled
        screen_width, _ = pyautogui.size()
        if x > screen_width * 1.5:
            with open(LOG_FILE, "a") as f:
                f.write(f"🚨 WARNING: X coordinate {x} exceeds 1.5x screen width ({screen_width})!\n")
                f.write(f"   This looks like coordinate doubling!\n")
    else:
        log_call(f"click() at current position", (), kwargs)

    return _original_click(x, y, **kwargs)

# Apply monkey-patch
def install_intercept():
    """Install the intercept"""
    pyautogui.moveTo = intercepted_moveTo
    pyautogui.dragTo = intercepted_dragTo
    pyautogui.click = intercepted_click

    # Clear log file
    with open(LOG_FILE, "w") as f:
        f.write("PyAutoGUI Intercept Log\n")
        f.write("="*80 + "\n\n")

    print(f"[INTERCEPT] Installed PyAutoGUI intercept. Logging to {LOG_FILE}")

# Auto-install when imported
with open(LOG_FILE, "w") as f:
    f.write("PyAutoGUI Intercept - Auto-installed on import\n")
    f.write("="*80 + "\n\n")

if __name__ == "__main__":
    install_intercept()
    print("Intercept installed. Import this module early in your code.")
else:
    # Auto-install when imported as a module
    install_intercept()
