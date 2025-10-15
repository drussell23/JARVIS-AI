#!/usr/bin/env python3
"""Test Screen Monitoring Functionality"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add backend to path
sys.path.insert(0, '/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')

async def test_monitoring():
    """Test screen monitoring through JARVIS"""
    print("🖥️  Testing Screen Monitoring")
    print("="*60)
    
    # Load environment variables
    load_dotenv()
    
    # Test 1: Direct vision analyzer test
    print("\n🧪 Test 1: Direct Vision Analyzer")
    try:
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
        
        vision = ClaudeVisionAnalyzer(os.getenv('ANTHROPIC_API_KEY'))
        
        # Test video streaming initialization
        print("📹 Starting video streaming...")
        result = await vision.start_video_streaming()
        print(f"Result: {result}")
        
        if result.get('success'):
            print("✅ Video streaming started successfully!")
            
            # Get status
            status = await vision.get_video_streaming_status()
            print(f"📊 Status: {status}")
            
            # Stop after 3 seconds
            await asyncio.sleep(3)
            stop_result = await vision.stop_video_streaming()
            print(f"🛑 Stop result: {stop_result}")
        else:
            print(f"❌ Failed to start: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Vision analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Through Claude chatbot
    print("\n🧪 Test 2: Through Claude Chatbot")
    try:
        from chatbots.claude_vision_chatbot import ClaudeVisionChatbot
        
        # Create chatbot with vision analyzer
        vision = ClaudeVisionAnalyzer(os.getenv('ANTHROPIC_API_KEY'))
        chatbot = ClaudeVisionChatbot(vision_analyzer=vision)
        
        # Test monitoring command
        response = await chatbot.generate_response("Start monitoring my screen")
        print(f"🤖 Chatbot response: {response}")
        
        # Check if monitoring is active
        if chatbot._monitoring_active:
            print("✅ Monitoring is active!")
        else:
            print("❌ Monitoring not active")
            
    except Exception as e:
        print(f"❌ Chatbot test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Through JARVIS voice handler
    print("\n🧪 Test 3: Through JARVIS Voice Handler")
    try:
        from voice.jarvis_agent_voice import JARVISAgentVoice
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
        
        # Create JARVIS with vision
        vision = ClaudeVisionAnalyzer(os.getenv('ANTHROPIC_API_KEY'))
        jarvis = JARVISAgentVoice(vision_analyzer=vision)
        
        # Process monitoring command with wake word
        response = await jarvis.process_voice_input("Hey JARVIS, start monitoring my screen")
        print(f"🎤 JARVIS response: {response}")
        
    except Exception as e:
        print(f"❌ JARVIS test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)

if __name__ == "__main__":
    asyncio.run(test_monitoring())