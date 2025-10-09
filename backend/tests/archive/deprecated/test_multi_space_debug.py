#!/usr/bin/env python3
"""Debug the multi-space ValueError issue"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_multi_space_flow():
    """Test the exact flow that's failing"""
    print("🔍 Debugging Multi-Space Query Flow")
    print("=" * 80)
    
    # Test 1: Check if multi_space_intelligence can be imported
    try:
        from vision.multi_space_intelligence import MultiSpaceIntelligenceExtension
        print("✅ Multi-space intelligence imported successfully")
    except Exception as e:
        print(f"❌ Failed to import multi_space_intelligence: {e}")
        return
        
    # Test 2: Check if pure_vision_intelligence has multi-space
    try:
        from api.pure_vision_intelligence import PureVisionIntelligence
        print("✅ PureVisionIntelligence imported")
        
        # Check if multi-space is properly initialized
        class MockClient:
            async def analyze_image_with_prompt(self, image, prompt, max_tokens=500):
                return {'content': 'test'}
                
        intelligence = PureVisionIntelligence(MockClient(), enable_multi_space=True)
        
        # Check attributes
        print(f"  - multi_space_enabled: {intelligence.multi_space_enabled}")
        print(f"  - has multi_space_extension: {hasattr(intelligence, 'multi_space_extension')}")
        
        # Test detection
        query = "can you see the Cursor IDE in the other desktop space?"
        should_use = intelligence._should_use_multi_space(query)
        print(f"  - Query detected as multi-space: {should_use}")
        
    except Exception as e:
        print(f"❌ PureVisionIntelligence error: {e}")
        import traceback
        traceback.print_exc()
        
    # Test 3: Check vision command handler flow
    try:
        from api.vision_command_handler import VisionCommandHandler
        print("\n✅ VisionCommandHandler imported")
        
        # Create handler
        handler = VisionCommandHandler()
        
        # Check if intelligence is initialized
        if not handler.intelligence:
            print("  - Intelligence not initialized, initializing...")
            await handler.initialize_intelligence()
            
        # Check multi-space detection
        if handler.intelligence:
            query = "can you see the Cursor IDE in the other desktop space?"
            needs_multi = handler.intelligence._should_use_multi_space(query) if hasattr(handler.intelligence, '_should_use_multi_space') else False
            print(f"  - Handler detects multi-space: {needs_multi}")
        
    except Exception as e:
        print(f"❌ VisionCommandHandler error: {e}")
        import traceback
        traceback.print_exc()
        
    # Test 4: Check the actual capture flow
    try:
        print("\n🔧 Testing capture flow...")
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer, VisionConfig
        
        # Create analyzer (without API key for testing)
        config = VisionConfig()
        analyzer = ClaudeVisionAnalyzer(api_key="test", config=config)
        
        # Test multi-space capture
        print("  - Testing multi-space capture...")
        # This might fail without proper setup, but we want to see where
        screenshots = await analyzer.capture_screen(multi_space=True)
        
        if isinstance(screenshots, dict):
            print(f"  ✅ Multi-space capture returned {len(screenshots)} spaces")
        else:
            print(f"  ❌ Multi-space capture returned single image")
            
    except Exception as e:
        print(f"  ⚠️  Capture test error (expected): {type(e).__name__}: {str(e)}")

async def check_actual_error():
    """Try to reproduce the actual error"""
    print("\n🎯 Reproducing the actual error flow...")
    print("=" * 80)
    
    try:
        # Import everything as it would be in production
        from api.vision_command_handler import VisionCommandHandler
        from api.vision_manager import VisionManager
        
        # Create handler
        handler = VisionCommandHandler()
        
        # Initialize if needed
        if not handler.intelligence:
            print("Initializing intelligence...")
            await handler.initialize_intelligence()
            
        # The actual query
        query = "can you see the Cursor IDE in the other desktop space?"
        
        # Check what happens
        print(f"\nProcessing query: '{query}'")
        
        # Step 1: Multi-space detection
        if handler.intelligence and hasattr(handler.intelligence, '_should_use_multi_space'):
            needs_multi = handler.intelligence._should_use_multi_space(query)
            print(f"Step 1: Multi-space detected = {needs_multi}")
        else:
            print("Step 1: No multi-space detection available")
            
        # Step 2: Try to capture
        try:
            screenshot = await handler._capture_screen(multi_space=True)
            if isinstance(screenshot, dict):
                print(f"Step 2: Captured {len(screenshot)} spaces")
            else:
                print("Step 2: Single screenshot captured")
        except Exception as e:
            print(f"Step 2 ERROR: {type(e).__name__}: {str(e)}")
            
        # Step 3: Try the full flow
        try:
            result = await handler.handle_command(query)
            print(f"Step 3: Command handled successfully")
            print(f"Response preview: {str(result.get('response', ''))[:100]}...")
        except Exception as e:
            print(f"Step 3 ERROR: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Setup error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_multi_space_flow())
    print("\n")
    asyncio.run(check_actual_error())