#!/usr/bin/env python3
"""Test minimal voice initialization"""

import logging
logging.basicConfig(level=logging.DEBUG)

print("Starting minimal voice test...")

try:
    print("1. Importing VoiceAPI...")
    from api.voice_api import VoiceAPI
    print("   ✓ VoiceAPI imported")
    
    print("2. Creating mock chatbot...")
    class MockChatbot:
        pass
    chatbot = MockChatbot()
    print("   ✓ Mock chatbot created")
    
    print("3. Creating VoiceAPI instance...")
    voice_api = VoiceAPI(chatbot)
    print("   ✓ VoiceAPI instance created")
    
    print("\nTest completed successfully!")
    
except Exception as e:
    print(f"\nError: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()