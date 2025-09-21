#!/usr/bin/env python3
"""
Debug script for multi-space integration issues
"""

import sys
import os
import traceback

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test each import individually to find the issue"""
    print("=== Testing Multi-Space Component Imports ===\n")
    
    components = [
        ("MultiSpaceWindowDetector", "vision.multi_space_window_detector", "MultiSpaceWindowDetector"),
        ("EnhancedWindowInfo", "vision.multi_space_window_detector", "EnhancedWindowInfo"),
        ("MultiSpaceIntelligenceExtension", "vision.multi_space_intelligence", "MultiSpaceIntelligenceExtension"),
        ("SpaceQueryType", "vision.multi_space_intelligence", "SpaceQueryType"),
        ("SpaceQueryIntent", "vision.multi_space_intelligence", "SpaceQueryIntent"),
        ("SpaceScreenshotCache", "vision.space_screenshot_cache", "SpaceScreenshotCache"),
        ("CacheConfidence", "vision.space_screenshot_cache", "CacheConfidence"),
        ("MinimalSpaceSwitcher", "vision.minimal_space_switcher", "MinimalSpaceSwitcher"),
        ("SpaceCaptureIntegration", "vision.minimal_space_switcher", "SpaceCaptureIntegration"),
        ("SwitchRequest", "vision.minimal_space_switcher", "SwitchRequest"),
    ]
    
    failed_imports = []
    
    for name, module_path, class_name in components:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✓ {name} imported successfully from {module_path}")
        except Exception as e:
            print(f"✗ Failed to import {name} from {module_path}")
            print(f"  Error: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed_imports.append((name, str(e)))
            print()
    
    return failed_imports

def test_pure_vision_intelligence():
    """Test PureVisionIntelligence with multi-space"""
    print("\n=== Testing PureVisionIntelligence ===\n")
    
    try:
        from api.pure_vision_intelligence import PureVisionIntelligence, MULTI_SPACE_AVAILABLE
        print(f"✓ PureVisionIntelligence imported successfully")
        print(f"  MULTI_SPACE_AVAILABLE: {MULTI_SPACE_AVAILABLE}")
        
        # Try to create instance
        class MockClaude:
            async def analyze_image_with_prompt(self, image, prompt, max_tokens):
                return {'content': 'Test response'}
        
        intelligence = PureVisionIntelligence(MockClaude(), enable_multi_space=True)
        print(f"✓ PureVisionIntelligence instance created")
        print(f"  multi_space_enabled: {intelligence.multi_space_enabled}")
        
    except Exception as e:
        print(f"✗ Failed to test PureVisionIntelligence")
        print(f"  Error: {type(e).__name__}: {e}")
        traceback.print_exc()

def test_claude_vision_analyzer():
    """Test ClaudeVisionAnalyzer multi-space components"""
    print("\n=== Testing ClaudeVisionAnalyzer ===\n")
    
    try:
        from vision.claude_vision_analyzer_main import MULTI_SPACE_AVAILABLE
        print(f"✓ claude_vision_analyzer_main imported")
        print(f"  MULTI_SPACE_AVAILABLE: {MULTI_SPACE_AVAILABLE}")
        
    except Exception as e:
        print(f"✗ Failed to import claude_vision_analyzer_main")
        print(f"  Error: {type(e).__name__}: {e}")
        traceback.print_exc()

def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n=== Checking Dependencies ===\n")
    
    deps = [
        ("PIL/Pillow", "PIL", "Image"),
        ("NumPy", "numpy", None),
        ("PyObjC", "Quartz", None),
        ("AppKit", "AppKit", None),
    ]
    
    for name, module, attr in deps:
        try:
            if attr:
                mod = __import__(module, fromlist=[attr])
                getattr(mod, attr)
            else:
                __import__(module)
            print(f"✓ {name} is installed")
        except ImportError:
            print(f"✗ {name} is NOT installed - install with: pip install {module}")

def test_multi_space_query_detection():
    """Test query detection logic"""
    print("\n=== Testing Multi-Space Query Detection ===\n")
    
    try:
        from vision.multi_space_intelligence import MultiSpaceQueryDetector
        
        detector = MultiSpaceQueryDetector()
        print("✓ MultiSpaceQueryDetector created")
        
        test_queries = [
            "Where is the Terminal?",
            "Is VSCode open?",
            "What's on Desktop 2?",
            "Show me all workspaces"
        ]
        
        for query in test_queries:
            try:
                intent = detector.detect_intent(query)
                print(f"  Query: '{query}'")
                print(f"  Intent: {intent.query_type.value}")
                print(f"  Target app: {intent.target_app}")
                print(f"  Target space: {intent.target_space}")
                print()
            except Exception as e:
                print(f"  Error processing '{query}': {e}")
                
    except Exception as e:
        print(f"✗ Failed to test query detection")
        print(f"  Error: {type(e).__name__}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("=== JARVIS Multi-Space Debug Script ===\n")
    
    # Check dependencies first
    check_dependencies()
    
    # Test imports
    failed = test_imports()
    
    # Test main components
    test_pure_vision_intelligence()
    test_claude_vision_analyzer()
    
    # Test query detection
    test_multi_space_query_detection()
    
    print("\n=== Summary ===")
    if failed:
        print(f"\n{len(failed)} component(s) failed to import:")
        for name, error in failed:
            print(f"  - {name}: {error}")
    else:
        print("\nAll components imported successfully!")
    
    print("\nIf you're still getting ValueError, the issue might be in:")
    print("1. The window detection code (CGWindowListCopyWindowInfo)")
    print("2. Missing PyObjC components")
    print("3. Initialization timing issues")