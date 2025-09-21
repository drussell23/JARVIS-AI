#!/usr/bin/env python3
"""
Debug why search query is lost
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.unified_command_processor import get_unified_processor
from system_control.macos_controller import MacOSController

async def test_search_processing():
    processor = get_unified_processor()
    controller = MacOSController()
    
    # Test different search commands
    test_commands = [
        "search for dogs",
        "search in safari for dogs",
        "google dogs"
    ]
    
    print("Testing search command processing:\n")
    
    for cmd in test_commands:
        print(f"Command: '{cmd}'")
        
        # Test web_search directly
        if "dogs" in cmd:
            success, message = controller.web_search("dogs", browser="safari")
            print(f"  Direct web_search('dogs'): {message}")
        
        # Test through processor
        result = await processor._execute_system_command(cmd)
        print(f"  Via processor: {result.get('response')}")
        print()

if __name__ == "__main__":
    asyncio.run(test_search_processing())