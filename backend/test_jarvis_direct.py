#!/usr/bin/env python3
"""
Test JARVIS Weather Directly
Bypass WebSocket to see if the issue is there
"""

import asyncio
import logging
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Focus on weather-related modules
logging.getLogger("voice.jarvis_agent_voice").setLevel(logging.DEBUG)
logging.getLogger("workflows.weather_app_vision_unified").setLevel(logging.DEBUG)
logging.getLogger("system_control.unified_vision_weather").setLevel(logging.DEBUG)

async def test_jarvis_direct():
    """Test JARVIS weather handling directly without WebSocket"""
    print("🔍 Testing JARVIS Weather Directly")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ERROR: ANTHROPIC_API_KEY not found")
        return
    
    print("\n1. Initializing JARVIS components...")
    
    # Initialize vision analyzer
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzerMain
    vision_analyzer = ClaudeVisionAnalyzerMain(api_key)
    print("✅ Vision analyzer initialized")
    
    # Initialize JARVIS
    from voice.jarvis_agent_voice import JARVISAgentVoice
    jarvis = JARVISAgentVoice(user_name="Sir", vision_analyzer=vision_analyzer)
    print("✅ JARVIS initialized")
    
    # Test queries
    test_queries = [
        "what's the weather today",
        "jarvis what's the weather today",
        "hey jarvis what's today's weather"
    ]
    
    print("\n2. Testing weather queries directly:")
    print("-" * 60)
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        
        try:
            # Call process_voice_input directly
            print("   Calling process_voice_input...")
            response = await asyncio.wait_for(
                jarvis.process_voice_input(query),
                timeout=30.0  # 30 second timeout
            )
            
            print(f"   ✅ Response: {response}")
            
            # Check if it's stuck on "Processing..."
            if response == "Processing...":
                print("   ❌ ERROR: Got 'Processing...' response - something is hanging!")
            
        except asyncio.TimeoutError:
            print("   ❌ TIMEOUT: Query processing took more than 30 seconds!")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n3. Testing weather handler directly:")
    print("-" * 60)
    
    # Test the weather handler directly
    try:
        print("   Calling _handle_weather_command directly...")
        response = await asyncio.wait_for(
            jarvis._handle_weather_command("what's the weather today"),
            timeout=20.0
        )
        
        print(f"   ✅ Direct handler response: {response}")
        
    except asyncio.TimeoutError:
        print("   ❌ TIMEOUT: Weather handler took more than 20 seconds!")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n\n✅ Direct testing complete!")


if __name__ == "__main__":
    asyncio.run(test_jarvis_direct())