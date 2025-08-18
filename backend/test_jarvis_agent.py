#!/usr/bin/env python3
"""
Test script for JARVIS AI Agent with System Control
Tests voice commands and system control capabilities
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test data
TEST_COMMANDS = [
    # Application control
    ("Open Chrome", "Should open Google Chrome browser"),
    ("Close Safari", "Should close Safari if open"),
    ("Switch to Finder", "Should bring Finder to front"),
    ("Show me all open apps", "Should list running applications"),
    
    # File operations
    ("Create a file called test.txt on my desktop", "Should create file"),
    ("Search for Python files", "Should find .py files"),
    
    # System control
    ("Set volume to 30%", "Should adjust system volume"),
    ("Mute the sound", "Should mute system"),
    ("Take a screenshot", "Should capture screen"),
    
    # Web operations
    ("Search Google for JARVIS AI", "Should open web search"),
    ("Open GitHub", "Should open github.com"),
    
    # Workflows
    ("Start my morning routine", "Should execute morning workflow"),
    
    # Mode switching
    ("Switch to system control mode", "Should focus on system commands"),
    ("Return to conversation mode", "Should go back to chat")
]


async def test_system_control():
    """Test system control components"""
    print("\n🧪 Testing System Control Components\n")
    
    # Test MacOSController
    print("1️⃣ Testing macOS Controller...")
    try:
        from system_control import MacOSController
        controller = MacOSController()
        
        # Test safe operations
        apps = controller.list_open_applications()
        print(f"✅ Found {len(apps)} open applications")
        
        # Test volume control
        success, message = controller.set_volume(20)
        print(f"✅ Volume control: {message}")
        
        print("✅ macOS Controller working\n")
    except Exception as e:
        print(f"❌ macOS Controller error: {e}\n")
        
    # Test Claude Command Interpreter
    print("2️⃣ Testing Claude Command Interpreter...")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  ANTHROPIC_API_KEY not set - skipping Claude tests\n")
        return
        
    try:
        from system_control import ClaudeCommandInterpreter
        interpreter = ClaudeCommandInterpreter(api_key)
        
        # Test basic command interpretation
        test_command = "Open Visual Studio Code"
        intent = await interpreter.interpret_command(test_command)
        
        print(f"✅ Interpreted: '{test_command}'")
        print(f"   Action: {intent.action}")
        print(f"   Target: {intent.target}")
        print(f"   Confidence: {intent.confidence:.2f}")
        print(f"   Category: {intent.category.value}")
        print(f"   Safety: {intent.safety_level.value}\n")
        
    except Exception as e:
        print(f"❌ Claude Interpreter error: {e}\n")


async def test_jarvis_agent():
    """Test JARVIS Agent Voice System"""
    print("3️⃣ Testing JARVIS Agent Voice System...")
    
    try:
        from voice.jarvis_agent_voice import JARVISAgentVoice
        jarvis = JARVISAgentVoice()
        
        if not jarvis.system_control_enabled:
            print("⚠️  System control not enabled - API key missing\n")
            return
            
        print("✅ JARVIS Agent initialized with system control\n")
        
        # Test capabilities
        capabilities = jarvis.get_capabilities()
        if "system_control" in capabilities:
            print("✅ System control capabilities available:")
            for category, items in capabilities["system_control"].items():
                print(f"   {category}: {len(items)} features")
        
        # Test sample commands
        print("\n🎯 Testing Voice Commands:\n")
        
        for command, description in TEST_COMMANDS[:5]:  # Test first 5 commands
            print(f"Command: '{command}'")
            print(f"Expected: {description}")
            
            try:
                # Simulate wake word detection
                jarvis.is_active = True
                response = await jarvis.process_voice_input(command)
                print(f"Response: {response}")
                print("-" * 50)
                
                # Small delay between commands
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"Error: {e}")
                print("-" * 50)
                
    except Exception as e:
        print(f"❌ JARVIS Agent error: {e}\n")


async def test_api_integration():
    """Test API integration"""
    print("\n4️⃣ Testing API Integration...")
    
    try:
        from api.jarvis_voice_api import JARVISVoiceAPI
        api = JARVISVoiceAPI()
        
        if api.jarvis_available:
            print("✅ JARVIS API initialized")
            
            # Test status endpoint
            status = await api.get_status()
            print(f"\nStatus: {status['status']}")
            print(f"Features: {len(status['features'])} available")
            
            if status.get('system_control', {}).get('enabled'):
                print("✅ System control enabled in API")
            else:
                print("⚠️  System control not enabled")
                
        else:
            print("❌ JARVIS API not available")
            
    except Exception as e:
        print(f"❌ API integration error: {e}")


async def interactive_test():
    """Interactive testing mode"""
    print("\n🎮 Interactive Test Mode")
    print("Type commands to test (or 'quit' to exit):\n")
    
    try:
        from voice.jarvis_agent_voice import JARVISAgentVoice
        jarvis = JARVISAgentVoice()
        
        if not jarvis.system_control_enabled:
            print("⚠️  System control disabled - configure API key")
            return
            
        # Activate JARVIS
        jarvis.is_active = True
        
        while True:
            command = input("\n> ").strip()
            
            if command.lower() in ['quit', 'exit', 'stop']:
                break
                
            if not command:
                continue
                
            try:
                response = await jarvis.process_voice_input(command)
                print(f"\nJARVIS: {response}")
            except Exception as e:
                print(f"\nError: {e}")
                
    except KeyboardInterrupt:
        print("\n\nExiting interactive mode...")
    except Exception as e:
        print(f"\nFatal error: {e}")


async def main():
    """Main test function"""
    print("=" * 60)
    print("🤖 JARVIS AI Agent System Control Test Suite")
    print("=" * 60)
    
    # Check environment
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  Warning: ANTHROPIC_API_KEY not set")
        print("   System control features will be limited")
        
    # Run tests
    await test_system_control()
    await test_jarvis_agent()
    await test_api_integration()
    
    # Ask for interactive mode
    response = input("\nRun interactive test mode? (y/n): ").strip().lower()
    if response == 'y':
        await interactive_test()
        
    print("\n✅ Test suite completed!")


if __name__ == "__main__":
    asyncio.run(main())