#!/usr/bin/env python3
"""
Debug Lock Command Issue
========================

Let's trace exactly what's happening.
"""

import asyncio
import logging
import sys

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress some noisy loggers
logging.getLogger('websockets').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


async def test_command_flow():
    """Test the exact flow of the lock command"""
    print("\n🔍 DEBUGGING LOCK COMMAND FLOW")
    print("="*60)
    
    # Import the handler directly
    from api.voice_unlock_integration import handle_voice_unlock_in_jarvis, voice_unlock_connector
    
    print(f"\n1️⃣ Initial state:")
    print(f"   voice_unlock_connector = {voice_unlock_connector}")
    print(f"   connected = {voice_unlock_connector.connected if voice_unlock_connector else 'None'}")
    
    # Test the command
    print(f"\n2️⃣ Sending 'lock my screen' command...")
    result = await handle_voice_unlock_in_jarvis("lock my screen")
    
    print(f"\n3️⃣ Result:")
    print(f"   Success: {result.get('success')}")
    print(f"   Response: {result.get('response')}")
    print(f"   Method: {result.get('method', 'not specified')}")
    
    # Check what happened
    print(f"\n4️⃣ After command:")
    from api import voice_unlock_integration as vui
    print(f"   voice_unlock_connector = {vui.voice_unlock_connector}")
    print(f"   connected = {vui.voice_unlock_connector.connected if vui.voice_unlock_connector else 'None'}")
    
    # Let's also check if the unified command processor is involved
    print(f"\n5️⃣ Checking command routing...")
    from api.unified_command_processor import UnifiedCommandProcessor
    processor = UnifiedCommandProcessor()
    
    # Check how the command is classified
    command_type, confidence = processor._classify_command("lock my screen")
    print(f"   Command type: {command_type}")
    print(f"   Confidence: {confidence}")
    
    # Check Voice Unlock patterns
    patterns = processor._detect_voice_unlock_patterns("lock my screen")
    print(f"   Voice unlock patterns detected: {patterns}")


async def test_direct_lock():
    """Test locking directly through Context Intelligence"""
    print(f"\n6️⃣ Testing direct Context Intelligence lock...")
    
    from context_intelligence.core.unlock_manager import get_unlock_manager
    unlock_manager = get_unlock_manager()
    
    success, message = await unlock_manager.lock_screen("Direct test")
    print(f"   Success: {success}")
    print(f"   Message: {message}")


async def main():
    """Run all tests"""
    await test_command_flow()
    await test_direct_lock()
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    # Ensure we're in the right directory
    import os
    os.chdir('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')
    
    asyncio.run(main())