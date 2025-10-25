#!/usr/bin/env python3
"""
Watch and log all mouse movements in real-time
This will show us exactly where JARVIS is moving the mouse
"""

import pyautogui
import time
import sys

print("\n" + "="*70)
print("🔍 LIVE MOUSE MOVEMENT TRACKER")
print("="*70)
print("\nThis will track mouse movements in real-time.")
print("When you tell JARVIS 'living room tv', watch the coordinates below.")
print("\nPress Ctrl+C to stop")
print("="*70)
print()

last_pos = None
movements = []

try:
    while True:
        current_pos = pyautogui.position()

        # Only log if position changed significantly
        if last_pos is None or (abs(current_pos.x - last_pos.x) > 5 or abs(current_pos.y - last_pos.y) > 5):
            timestamp = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] Mouse at: ({current_pos.x:4d}, {current_pos.y:4d})")

            # Track movements for analysis
            movements.append({
                'time': timestamp,
                'x': current_pos.x,
                'y': current_pos.y
            })

            # Check if near expected coordinates
            if abs(current_pos.x - 1235) < 10 and abs(current_pos.y - 10) < 10:
                print("  ✅ Near Control Center (1235, 10)")
            elif abs(current_pos.x - 1396) < 10 and abs(current_pos.y - 177) < 10:
                print("  ✅ Near Screen Mirroring (1396, 177)")
            elif abs(current_pos.x - 1223) < 10 and abs(current_pos.y - 115) < 10:
                print("  ✅ Near Living Room TV (1223, 115)")

            last_pos = current_pos

        time.sleep(0.05)  # Check every 50ms

except KeyboardInterrupt:
    print("\n" + "="*70)
    print("📊 MOVEMENT SUMMARY")
    print("="*70)

    if len(movements) > 1:
        print(f"\nTotal movements captured: {len(movements)}")
        print("\nFirst 10 positions:")
        for i, m in enumerate(movements[:10]):
            print(f"  {i+1}. ({m['x']:4d}, {m['y']:4d})")

        if len(movements) > 10:
            print(f"\n... and {len(movements) - 10} more movements")
    else:
        print("\nNo significant movements detected")

    print("="*70)