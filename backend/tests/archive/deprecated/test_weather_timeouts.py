#!/usr/bin/env python3
"""
Test Weather System Timeouts
Verify all components have proper timeouts to prevent hanging
"""

import asyncio
import time
import logging
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Enable detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_weather_system_timeouts():
    """Test that all weather system components have proper timeouts"""
    print("⏱️ Testing Weather System Timeouts")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ERROR: ANTHROPIC_API_KEY not found")
        return
    
    print("\n1. Testing AppleScript timeouts:")
    print("-" * 40)
    
    # Test AppleScript timeout
    from system_control.unified_vision_weather import UnifiedVisionWeather
    weather = UnifiedVisionWeather()
    
    # Test AppleScript with timeout
    start = time.time()
    result = await weather._run_applescript('tell application "System Events" to get name of every process')
    elapsed = time.time() - start
    print(f"✅ AppleScript completed in {elapsed:.2f}s")
    
    # Test a potentially hanging AppleScript
    print("\n2. Testing potentially slow AppleScript:")
    start = time.time()
    result = await weather._run_applescript('delay 10')  # This would normally take 10 seconds
    elapsed = time.time() - start
    
    if elapsed < 6:  # Should timeout at 5 seconds
        print(f"✅ AppleScript properly timed out after {elapsed:.2f}s")
    else:
        print(f"❌ AppleScript took too long: {elapsed:.2f}s")
    
    print("\n3. Testing vision handler timeouts:")
    print("-" * 40)
    
    # Initialize vision analyzer
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzerMain
    from system_control.macos_controller import MacOSController
    
    analyzer = ClaudeVisionAnalyzerMain(api_key)
    controller = MacOSController()
    
    # Initialize weather system with handlers
    weather_system = UnifiedVisionWeather(analyzer, controller)
    
    # Test weather app preparation
    print("\n4. Testing Weather app preparation timeout:")
    start = time.time()
    try:
        ready = await asyncio.wait_for(
            weather_system._ensure_weather_app_ready(),
            timeout=10.0
        )
        elapsed = time.time() - start
        print(f"✅ Weather app ready in {elapsed:.2f}s: {ready}")
    except asyncio.TimeoutError:
        print("❌ Weather app preparation timed out")
    
    print("\n5. Testing full weather query with timeouts:")
    print("-" * 40)
    
    # Test full weather query
    queries = [
        "What's the weather?",
        "What's the temperature?",
        "Will it rain today?"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        start = time.time()
        
        try:
            result = await asyncio.wait_for(
                weather_system.get_weather(query),
                timeout=20.0  # Overall 20 second timeout
            )
            elapsed = time.time() - start
            
            if result['success']:
                print(f"✅ Query completed in {elapsed:.2f}s")
                print(f"   Response: {result['formatted_response'][:100]}...")
            else:
                print(f"⚠️ Query failed in {elapsed:.2f}s: {result.get('error', 'Unknown error')}")
                
        except asyncio.TimeoutError:
            elapsed = time.time() - start
            print(f"❌ Query timed out after {elapsed:.2f}s")
    
    print("\n\n6. Testing JARVIS integration:")
    print("-" * 40)
    
    # Test JARVIS weather handler
    from voice.jarvis_agent_voice import JARVISAgentVoice
    
    jarvis = JARVISAgentVoice(vision_analyzer=analyzer)
    
    print("Testing JARVIS weather handler with timeout...")
    start = time.time()
    
    try:
        response = await asyncio.wait_for(
            jarvis._handle_weather_command("what's the weather today"),
            timeout=15.0
        )
        elapsed = time.time() - start
        
        print(f"✅ JARVIS responded in {elapsed:.2f}s")
        print(f"   Response: {response}")
        
        # Check if response is generic fallback
        if any(phrase in response for phrase in ["I've opened", "having difficulty", "unable to"]):
            print("⚠️ WARNING: Response is a fallback, not actual weather data")
        
    except asyncio.TimeoutError:
        elapsed = time.time() - start
        print(f"❌ JARVIS timed out after {elapsed:.2f}s")
    
    print("\n\n✅ Timeout testing complete!")
    print("\nSummary:")
    print("- All components should have timeouts to prevent hanging")
    print("- AppleScript: 5 second timeout")
    print("- Vision API: 30 second timeout (configurable)")
    print("- Weather extraction: 15 second timeout")
    print("- Overall weather query: 20 second timeout")


if __name__ == "__main__":
    asyncio.run(test_weather_system_timeouts())