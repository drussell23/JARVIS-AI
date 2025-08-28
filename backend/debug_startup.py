#!/usr/bin/env python3
"""Simple debug to find where backend is hanging"""

import sys
import time

print(">>> Starting debug imports...")

# Test imports one by one
print("\n1. Testing basic imports...")
import logging
logging.basicConfig(level=logging.INFO)

print("\n2. Testing memory manager...")
try:
    from memory.memory_manager import MemoryManager
    print("   ✓ Memory manager imported")
except Exception as e:
    print(f"   ✗ Memory manager failed: {e}")

print("\n3. Testing vision imports...")
try:
    from vision import vision_monitor
    print("   ✓ Vision monitor imported") 
except Exception as e:
    print(f"   ✗ Vision monitor failed: {e}")

print("\n4. Testing voice API...")
try:
    from api.voice_api import VoiceAPI
    print("   ✓ Voice API imported")
except Exception as e:
    print(f"   ✗ Voice API failed: {e}")

print("\n5. Creating test chatbot...")
from chatbots.claude_chatbot import ClaudeChatbot
api_key = "test-key"  
bot = ClaudeChatbot(api_key)
print("   ✓ Chatbot created")

print("\n6. Creating Voice API instance...")
start = time.time()
voice_api = VoiceAPI(bot)
elapsed = time.time() - start
print(f"   ✓ Voice API created in {elapsed:.2f}s")

print("\n7. Testing automation engine...")
try:
    from engines.automation_engine import AutomationEngine
    print("   ✓ Automation engine imported")
    engine = AutomationEngine()
    print("   ✓ Automation engine created")
except Exception as e:
    print(f"   ✗ Automation engine failed: {e}")

print("\n8. Testing JARVIS Voice API...")  
try:
    from api.jarvis_voice_api import JARVISVoiceAPI
    print("   ✓ JARVIS Voice API imported")
    jarvis_api = JARVISVoiceAPI()
    print("   ✓ JARVIS Voice API created")
except Exception as e:
    print(f"   ✗ JARVIS Voice API failed: {e}")

print("\n9. Testing enhanced voice routes...")
try:
    from api.enhanced_voice_routes import router as enhanced_voice_router
    print("   ✓ Enhanced voice routes imported")
except Exception as e:
    print(f"   ✗ Enhanced voice routes failed: {e}")

print("\n10. Testing ML audio API...")
try:
    from api.ml_audio_api import router as ml_audio_router
    print("   ✓ ML audio API imported")
except Exception as e:
    print(f"   ✗ ML audio API failed: {e}")

print("\n✅ Debug test completed!")