#!/usr/bin/env python3
"""Test the live handler to see why it's not working"""

import asyncio
import sys
import os
sys.path.insert(0, 'backend')

async def test_handler():
    from api.vision_command_handler import vision_command_handler

    # Initialize the handler
    api_key = os.getenv("ANTHROPIC_API_KEY")
    await vision_command_handler.initialize_intelligence(api_key)

    print(f"Intelligence initialized: {vision_command_handler.intelligence is not None}")

    if vision_command_handler.intelligence:
        # Test multi-space detection
        test_query = "What's happening across my desktop spaces?"

        # Check if _should_use_multi_space exists
        has_method = hasattr(vision_command_handler.intelligence, "_should_use_multi_space")
        print(f"Has _should_use_multi_space method: {has_method}")

        if has_method:
            result = vision_command_handler.intelligence._should_use_multi_space(test_query)
            print(f"Query: '{test_query}'")
            print(f"Multi-space detection result: {result}")

            # Check multi_space_enabled
            print(f"Multi-space enabled: {vision_command_handler.intelligence.multi_space_enabled}")

            # Check if extension is present
            has_extension = hasattr(vision_command_handler.intelligence, "multi_space_extension")
            print(f"Has multi_space_extension: {has_extension}")

            if has_extension and vision_command_handler.intelligence.multi_space_extension:
                ext = vision_command_handler.intelligence.multi_space_extension
                ext_result = ext.should_use_multi_space(test_query)
                print(f"Extension detection result: {ext_result}")

asyncio.run(test_handler())