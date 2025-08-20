#!/usr/bin/env python3
"""Debug test runner"""

import asyncio
import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.vision.jarvis_workspace_integration import JARVISWorkspaceIntelligence
from test_utils import WindowFixtures

async def test_basic():
    """Test basic functionality"""
    print("Creating JARVIS instance...")
    jarvis = JARVISWorkspaceIntelligence()
    
    print("Setting up mock windows...")
    jarvis.window_detector.get_all_windows = lambda: WindowFixtures.single_window()
    
    print("Testing workspace command...")
    try:
        response = await jarvis.handle_workspace_command("What am I working on?")
        print(f"Response: {response[:100]}...")
        print("✅ Test passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_basic())