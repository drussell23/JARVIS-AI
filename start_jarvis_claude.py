#!/usr/bin/env python3
"""
Start JARVIS with Claude API integration
Perfect for M1 Macs with limited RAM - all processing happens in the cloud!
"""

import asyncio
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from chatbots.dynamic_chatbot import DynamicChatbot
from memory.memory_manager import M1MemoryManager
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main entry point for Claude-powered JARVIS"""
    
    print("🚀 Starting JARVIS with Claude API")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Error: ANTHROPIC_API_KEY not set!")
        print("\nTo use Claude with JARVIS:")
        print("1. Get an API key from: https://console.anthropic.com/")
        print("2. Set the environment variable:")
        print("   export ANTHROPIC_API_KEY='your-api-key-here'")
        print("\nAlternatively, you can create a .env file with:")
        print("   ANTHROPIC_API_KEY=your-api-key-here")
        return
        
    print("✅ Claude API key found")
    print("🧠 Using Claude for all responses (no local memory constraints!)")
    print("=" * 50)
    
    try:
        # Create memory manager (still needed for system monitoring)
        memory_manager = M1MemoryManager()
        
        # Create dynamic chatbot with Claude
        chatbot = DynamicChatbot(
            memory_manager=memory_manager,
            use_claude=True,
            claude_api_key=api_key,
            auto_switch=False,  # Don't auto-switch when using Claude
            preserve_context=True
        )
        
        # Start monitoring
        await chatbot.start_monitoring()
        
        # Force Claude mode
        await chatbot.force_mode("claude")
        
        print("\n🤖 JARVIS is ready! (Powered by Claude)")
        print("Type 'exit' or 'quit' to stop.")
        print("-" * 50)
        
        # Main conversation loop
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                    print("\n👋 Goodbye! Thanks for using JARVIS.")
                    break
                    
                if not user_input:
                    continue
                    
                # Generate response
                print("🤖 JARVIS: ", end="", flush=True)
                
                # For streaming response (if you want to see it word by word)
                if hasattr(chatbot, 'generate_response_stream'):
                    async for chunk in chatbot.generate_response_stream(user_input):
                        print(chunk, end="", flush=True)
                    print()  # New line after response
                else:
                    # Regular response
                    response = await chatbot.generate_response(user_input)
                    print(response)
                    
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"\n❌ Error: {e}")
                print("Please try again.")
                
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Fatal error: {e}")
        
    finally:
        # Cleanup
        if 'chatbot' in locals():
            print("\n🧹 Cleaning up...")
            await chatbot.stop_monitoring()
            await chatbot.cleanup()
            
        print("✅ JARVIS shutdown complete.")


def run():
    """Entry point for the script"""
    # Try to load .env file if it exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not installed, that's okay
        
    # Run the async main function
    asyncio.run(main())


if __name__ == "__main__":
    run()