#!/usr/bin/env python3
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


Test script for closing applications on macOS
"""

import asyncio
from backend.system_control.macos_controller import MacOSController

def test_close_apps():
    """Test closing various applications"""
    controller = MacOSController()
    
    # Test apps
    test_apps = ["WhatsApp", "Preview", "Safari", "Notes"]
    
    print("Testing application closing functionality...\n")
    
    for app in test_apps:
        print(f"Attempting to close {app}...")
        success, message = controller.close_application(app)
        
        if success:
            print(f"✅ Success: {message}")
        else:
            print(f"❌ Failed: {message}")
        print()
    
    # List currently open apps
    print("\nCurrently open applications:")
    open_apps = controller.list_open_applications()
    for app in open_apps:
        print(f"  - {app}")

if __name__ == "__main__":
    test_close_apps()