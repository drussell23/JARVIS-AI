#!/usr/bin/env python3
"""Comprehensive diagnostic for multi-space issues"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import logging

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

async def diagnose_issue():
    """Run comprehensive diagnostics"""
    print("🔍 Multi-Space Diagnostic Tool")
    print("=" * 80)
    
    # Test 1: Basic imports
    print("\n1️⃣ Testing basic imports...")
    try:
        from vision.multi_space_intelligence import MultiSpaceIntelligenceExtension
        from vision.multi_space_capture_engine import MultiSpaceCaptureEngine
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
        from api.pure_vision_intelligence import PureVisionIntelligence
        from api.vision_command_handler import VisionCommandHandler
        print("✅ All imports successful")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return
    
    # Test 2: Multi-space detection
    print("\n2️⃣ Testing multi-space detection...")
    extension = MultiSpaceIntelligenceExtension()
    query = "can you see the Cursor IDE in the other desktop space?"
    should_use = extension.should_use_multi_space(query)
    print(f"✅ Query detected as multi-space: {should_use}")
    
    # Test 3: Capture engine
    print("\n3️⃣ Testing capture engine...")
    try:
        engine = MultiSpaceCaptureEngine()
        spaces = await engine.enumerate_spaces()
        print(f"✅ Found {len(spaces)} spaces: {spaces}")
        
        # Try to capture current space
        from vision.multi_space_capture_engine import SpaceCaptureRequest, CaptureQuality
        request = SpaceCaptureRequest(
            space_ids=[spaces[0]] if spaces else [1],
            quality=CaptureQuality.FAST,
            use_cache=False
        )
        result = await engine.capture_all_spaces(request)
        print(f"✅ Capture test: success={result.success}, screenshots={len(result.screenshots)}")
    except Exception as e:
        print(f"❌ Capture engine error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Vision analyzer
    print("\n4️⃣ Testing vision analyzer...")
    try:
        # Create analyzer without API key for structure test
        from vision.claude_vision_analyzer_main import VisionConfig
        config = VisionConfig()
        analyzer = ClaudeVisionAnalyzer("test_key", config)
        
        # Check if multi-image method exists
        has_multi = hasattr(analyzer, 'analyze_multiple_images_with_prompt')
        print(f"✅ Vision analyzer has analyze_multiple_images_with_prompt: {has_multi}")
        
        # Test single capture
        screenshot = await analyzer.capture_screen()
        print(f"✅ Single capture: {'Success' if screenshot else 'Failed'}")
        
        # Test multi capture
        try:
            multi = await analyzer.capture_screen(multi_space=True)
            if isinstance(multi, dict):
                print(f"✅ Multi-space capture: {len(multi)} spaces")
            else:
                print("⚠️  Multi-space returned single image")
        except Exception as e:
            print(f"⚠️  Multi-space capture error: {e}")
    except Exception as e:
        print(f"❌ Vision analyzer error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Full flow simulation
    print("\n5️⃣ Testing full flow...")
    try:
        # Create mock vision analyzer
        class MockVisionAnalyzer:
            async def capture_screen(self, multi_space=False, space_number=None):
                if multi_space:
                    return {1: "screen1", 2: "screen2"}
                return "single_screen"
                
            async def analyze_image_with_prompt(self, image, prompt, **kwargs):
                return {'content': 'Single space response'}
                
            async def analyze_multiple_images_with_prompt(self, images, prompt, max_tokens=1000):
                return {'content': f'I can see {len(images)} desktop spaces. Cursor IDE is on Desktop 2.'}
        
        # Create handler with mock
        handler = VisionCommandHandler()
        handler.vision_analyzer = MockVisionAnalyzer()
        
        # Initialize intelligence
        await handler.initialize_intelligence()
        
        # Check if multi-space is enabled
        if handler.intelligence:
            print(f"✅ Intelligence initialized")
            print(f"   multi_space_enabled: {handler.intelligence.multi_space_enabled}")
            
            # Test the flow
            needs_multi = handler.intelligence._should_use_multi_space(query)
            print(f"   Should use multi-space: {needs_multi}")
            
            # Try the full command
            try:
                # Override capture method
                handler._capture_screen = MockVisionAnalyzer().capture_screen
                
                result = await handler.handle_command(query)
                print(f"✅ Command handled successfully")
                print(f"   Response preview: {result.get('response', '')[:100]}...")
            except Exception as e:
                print(f"❌ Command handling error: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ Intelligence not initialized")
            
    except Exception as e:
        print(f"❌ Full flow error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 6: Check actual error location
    print("\n6️⃣ Checking error locations...")
    try:
        # Test pure vision intelligence directly
        from api.pure_vision_intelligence import PureVisionIntelligence
        
        class TestClient:
            async def analyze_image_with_prompt(self, image, prompt, max_tokens=500):
                return {'content': 'test'}
            
            async def analyze_multiple_images_with_prompt(self, images, prompt, max_tokens=1000):
                return {'content': f'Multi-space test: {len(images)} spaces'}
        
        intel = PureVisionIntelligence(TestClient(), enable_multi_space=True)
        
        # Test single
        single_resp = await intel.understand_and_respond("single_img", "test")
        print(f"✅ Single response works: {single_resp[:50]}...")
        
        # Test multi
        multi_resp = await intel.understand_and_respond({1: "img1", 2: "img2"}, query)
        print(f"✅ Multi response works: {multi_resp[:50]}...")
        
    except Exception as e:
        print(f"❌ Direct intelligence test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(diagnose_issue())