#!/usr/bin/env python3
"""
Clear the clicker singleton to force a fresh instance with the drag fix
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

print("Clearing clicker singletons...")

# Clear adaptive clicker singleton
try:
    import backend.display.adaptive_control_center_clicker as acc
    acc._adaptive_clicker = None
    print("✅ Cleared adaptive clicker singleton")
except Exception as e:
    print(f"❌ Could not clear adaptive clicker: {e}")

# Clear SAI clicker singleton
try:
    import backend.display.sai_enhanced_control_center_clicker as sai
    sai._sai_clicker = None
    print("✅ Cleared SAI clicker singleton")
except Exception as e:
    print(f"⚠️  Could not clear SAI clicker: {e}")

# Clear UAE clicker singleton
try:
    import backend.display.uae_enhanced_control_center_clicker as uae
    uae._uae_clicker = None
    print("✅ Cleared UAE clicker singleton")
except Exception as e:
    print(f"⚠️  Could not clear UAE clicker: {e}")

# Clear display monitor singleton
try:
    import backend.display.advanced_display_monitor as adm
    adm._monitor_instance = None
    print("✅ Cleared display monitor singleton")
except Exception as e:
    print(f"⚠️  Could not clear display monitor: {e}")

print("\nAll singletons cleared. JARVIS will create fresh instances with the drag fix.")