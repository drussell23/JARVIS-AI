#!/usr/bin/env python3
"""
Diagnostic script to test vision capture and analysis in real-time
Tests if JARVIS can actually see and understand the screen
"""

import asyncio
import os
import sys
import subprocess
from datetime import datetime
import numpy as np
from PIL import Image
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def capture_screenshot_macos():
    """Capture a screenshot using macOS screencapture command"""
    import tempfile
    
    # Create temporary file for screenshot
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Use macOS screencapture command
        cmd = ['screencapture', '-x', tmp_path]  # -x prevents capture sound
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"screencapture failed: {result.stderr}")
            return None
        
        # Load and convert to numpy array
        image = Image.open(tmp_path)
        screenshot = np.array(image)
        
        logger.info(f"✅ Captured screenshot: {screenshot.shape}")
        return screenshot
        
    except Exception as e:
        logger.error(f"❌ Screenshot capture failed: {e}")
        return None
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

async def test_vision_analyzer():
    """Test the Claude Vision Analyzer with a real screenshot"""
    
    # Check for API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("❌ ANTHROPIC_API_KEY not set!")
        logger.info("Please set: export ANTHROPIC_API_KEY='your-key-here'")
        return False
    
    logger.info("🔍 Testing Claude Vision Analyzer...")
    
    try:
        # Import the analyzer
        from claude_vision_analyzer_main import ClaudeVisionAnalyzer
        
        # Initialize analyzer
        analyzer = ClaudeVisionAnalyzer(api_key)
        logger.info("✅ Vision analyzer initialized")
        
        # Capture a screenshot
        screenshot = await capture_screenshot_macos()
        if screenshot is None:
            return False
        
        # Test 1: Basic analysis
        logger.info("\n📊 Test 1: Basic screen analysis...")
        result = await analyzer.analyze_screenshot(
            screenshot,
            "What do you see on this screen? List the main elements visible."
        )
        
        if result and 'description' in result:
            logger.info(f"✅ Basic analysis successful!")
            logger.info(f"Description: {result['description'][:200]}...")
        else:
            logger.error("❌ Basic analysis failed")
            return False
        
        # Test 2: Window detection
        logger.info("\n📊 Test 2: Window detection...")
        result = await analyzer.analyze_screenshot(
            screenshot,
            "What applications or windows are visible? List each window/app you can see."
        )
        
        if result and 'description' in result:
            logger.info(f"✅ Window detection successful!")
            logger.info(f"Windows found: {result['description'][:200]}...")
        else:
            logger.error("❌ Window detection failed")
        
        # Test 3: Action detection
        logger.info("\n📊 Test 3: Action detection...")
        result = await analyzer.analyze_screenshot(
            screenshot,
            "What actions could I take on this screen? What buttons or interactive elements do you see?"
        )
        
        if result and 'description' in result:
            logger.info(f"✅ Action detection successful!")
            logger.info(f"Actions: {result['description'][:200]}...")
        else:
            logger.error("❌ Action detection failed")
        
        # Test memory stats
        stats = analyzer.get_all_memory_stats()
        logger.info(f"\n💾 Memory usage: {stats['system']['process_mb']:.1f}MB")
        
        # Cleanup
        await analyzer.cleanup_all_components()
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import vision analyzer: {e}")
        logger.info("Make sure you're in the correct directory")
        return False
    except Exception as e:
        logger.error(f"❌ Vision test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_swift_integration():
    """Test if Swift vision integration is working"""
    logger.info("\n🔍 Testing Swift vision integration...")
    
    try:
        # Check if Swift bridge exists
        swift_bridge_path = Path(__file__).parent.parent / "swift_bridge" / ".build"
        if not swift_bridge_path.exists():
            logger.warning("⚠️ Swift bridge not built. Vision might be limited.")
            return False
        
        # Try to import Swift integration
        try:
            from swift_vision_integration import MemoryAwareSwiftVisionIntegration
            integration = MemoryAwareSwiftVisionIntegration()
            
            # Capture and analyze
            image_data = integration.capture_screen()
            if image_data:
                logger.info("✅ Swift screen capture working!")
                
                # Test window detection
                windows = integration.get_window_info()
                logger.info(f"✅ Found {len(windows)} windows via Swift")
                for window in windows[:3]:  # Show first 3
                    logger.info(f"  - {window.get('app_name', 'Unknown')}: {window.get('title', 'No title')}")
                
                return True
            else:
                logger.error("❌ Swift capture returned no data")
                
        except ImportError:
            logger.warning("⚠️ Swift integration not available")
            
    except Exception as e:
        logger.error(f"❌ Swift test failed: {e}")
    
    return False

async def diagnose_vision_system():
    """Full diagnostic of the vision system"""
    print("="*60)
    print("🏥 JARVIS Vision System Diagnostic")
    print("="*60)
    
    # Check environment
    print("\n🔧 Environment Check:")
    api_key = os.getenv('ANTHROPIC_API_KEY')
    print(f"  ANTHROPIC_API_KEY: {'✅ Set' if api_key else '❌ Not set'}")
    print(f"  Working directory: {os.getcwd()}")
    
    # Check Python dependencies
    print("\n📦 Dependencies Check:")
    deps = ['PIL', 'numpy', 'anthropic']
    for dep in deps:
        try:
            __import__(dep)
            print(f"  {dep}: ✅ Installed")
        except ImportError:
            print(f"  {dep}: ❌ Not installed")
    
    # Test screen capture
    print("\n📸 Screen Capture Test:")
    screenshot = await capture_screenshot_macos()
    if screenshot is not None:
        print(f"  ✅ Screen capture working ({screenshot.shape})")
    else:
        print("  ❌ Screen capture failed")
    
    # Test vision analyzer
    print("\n🤖 Vision Analyzer Test:")
    vision_ok = await test_vision_analyzer()
    
    # Test Swift integration
    swift_ok = await test_swift_integration()
    
    # Summary
    print("\n" + "="*60)
    print("📊 Diagnostic Summary:")
    print("="*60)
    
    all_ok = api_key and screenshot is not None and vision_ok
    
    if all_ok:
        print("✅ Vision system is working properly!")
        print("\nYour JARVIS should be able to see and analyze the screen.")
    else:
        print("❌ Vision system has issues:")
        if not api_key:
            print("  - Set ANTHROPIC_API_KEY environment variable")
        if screenshot is None:
            print("  - Screen capture is not working")
        if not vision_ok:
            print("  - Vision analyzer is not functioning")
        if not swift_ok:
            print("  - Swift integration is not available (optional)")
    
    print("\n💡 Next steps:")
    if not all_ok:
        print("  1. Fix the issues listed above")
        print("  2. Run this diagnostic again")
    print("  3. Test with: python test_enhanced_vision_integration.py")
    print("  4. Check logs in your JARVIS interface")

if __name__ == "__main__":
    asyncio.run(diagnose_vision_system())