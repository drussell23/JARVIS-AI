#!/usr/bin/env python3
"""
Test JARVIS with the specific "Where is the Terminal?" query
Simulating the actual JARVIS system flow
"""

import asyncio
import sys
import os
import traceback
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_terminal_query():
    """Test the specific Terminal query that's failing"""
    print("=== Testing 'Where is the Terminal?' Query ===\n")
    
    # Set up environment
    os.environ['VISION_MULTI_SPACE'] = 'true'
    
    try:
        # Import the chatbot that JARVIS uses
        from chatbots.claude_vision_chatbot import ClaudeVisionChatbot
        
        # Use the actual API key if available
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("⚠️  No ANTHROPIC_API_KEY found, using mock mode")
            api_key = 'test-key'
        
        # Create chatbot instance
        chatbot = ClaudeVisionChatbot(api_key=api_key, model='claude-3-5-sonnet-20241022')
        print("✓ ClaudeVisionChatbot created")
        
        # The query that's causing issues
        test_query = "Where is the Terminal?"
        print(f"\nQuery: {test_query}")
        
        # Analyze screen with vision (this is what JARVIS calls)
        try:
            response = await chatbot.analyze_screen_with_vision(test_query)
            print(f"\nResponse: {response}")
            
        except ValueError as ve:
            print(f"\n✗ ValueError occurred: {ve}")
            print("\nTraceback:")
            traceback.print_exc()
            
            # Try to identify where in the chain it failed
            print("\nDebugging the error chain...")
            
            # Test vision analyzer directly
            try:
                from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
                analyzer = ClaudeVisionAnalyzer(api_key)
                print("✓ ClaudeVisionAnalyzer created")
                
                # Try to capture screen
                screenshot = await analyzer.capture_screen()
                if screenshot:
                    print("✓ Screenshot captured")
                else:
                    print("✗ Failed to capture screenshot")
                    
                # Try smart_analyze with multi-space query
                if screenshot:
                    result = await analyzer.smart_analyze(screenshot, test_query)
                    print(f"✓ Smart analyze result: {result}")
                    
            except Exception as e2:
                print(f"✗ Vision analyzer error: {e2}")
                traceback.print_exc()
                
        except Exception as e:
            print(f"\n✗ Error: {type(e).__name__}: {e}")
            print("\nTraceback:")
            traceback.print_exc()
            
    except Exception as e:
        print(f"✗ Failed to set up test: {e}")
        traceback.print_exc()


async def test_direct_components():
    """Test components directly to isolate the issue"""
    print("\n\n=== Testing Components Directly ===\n")
    
    try:
        # Test multi-space window detector
        from vision.multi_space_window_detector import MultiSpaceWindowDetector
        
        detector = MultiSpaceWindowDetector()
        print("✓ MultiSpaceWindowDetector created")
        
        # Get window data
        window_data = detector.get_all_windows_across_spaces()
        print(f"✓ Got window data: {len(window_data.get('windows', []))} windows")
        
        # Check Terminal
        terminal_windows = [
            w for w in window_data.get('windows', [])
            if 'terminal' in w.app_name.lower()
        ]
        
        if terminal_windows:
            print(f"\n✓ Found Terminal windows:")
            for tw in terminal_windows:
                print(f"  - {tw.app_name} on Desktop {tw.space_id}")
                print(f"    Title: {tw.window_title}")
        else:
            print("\n✗ No Terminal windows found")
            
        # Show what's on each space
        print("\nWindows by space:")
        for space_id, window_ids in window_data.get('space_window_map', {}).items():
            print(f"\nDesktop {space_id}: {len(window_ids)} windows")
            space_windows = [
                w for w in window_data.get('windows', [])
                if hasattr(w, 'window_id') and w.window_id in window_ids
            ]
            apps = set(w.app_name for w in space_windows if hasattr(w, 'app_name'))
            for app in list(apps)[:5]:  # First 5 apps
                print(f"  - {app}")
                
    except Exception as e:
        print(f"✗ Component test failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # Test the actual query flow
    asyncio.run(test_terminal_query())
    
    # Test components directly
    asyncio.run(test_direct_components())