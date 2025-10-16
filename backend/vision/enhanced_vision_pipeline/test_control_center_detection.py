#!/usr/bin/env python3
"""
Test Control Center Detection
============================

Test the icon detection for Control Center specifically.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import pyautogui
import subprocess
from PIL import Image

logging.basicConfig(level=logging.INFO)


async def test_control_center_detection():
    """Test Control Center detection"""
    
    print("\n" + "="*70)
    print("🔍 Testing Control Center Detection")
    print("="*70)
    
    # Get screen info
    screen_width, screen_height = pyautogui.size()
    print(f"\n1️⃣  Screen Info:")
    print(f"   Resolution: {screen_width}x{screen_height}")
    
    # Expected Control Center position
    expected_x = screen_width - 100  # 100px from right
    expected_y = 15  # Center of menu bar
    
    print(f"\n2️⃣  Expected Control Center Position:")
    print(f"   X: {expected_x} (100px from right edge)")
    print(f"   Y: {expected_y} (center of menu bar)")
    
    # Capture menu bar
    print(f"\n3️⃣  Capturing menu bar...")
    
    temp_dir = Path.home() / '.jarvis' / 'screenshots' / 'test'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    menu_bar_path = temp_dir / 'menubar_test.png'
    
    # Capture just the menu bar (top 30px)
    process = await asyncio.create_subprocess_exec(
        'screencapture', '-R', f'0,0,{screen_width},30', '-x', str(menu_bar_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    await process.communicate()
    
    if menu_bar_path.exists():
        print(f"   ✅ Menu bar captured: {menu_bar_path}")
        
        # Load image
        img = Image.open(menu_bar_path)
        print(f"   Image size: {img.size}")
        
        # Test Enhanced Pipeline detection
        print(f"\n4️⃣  Testing Enhanced Pipeline detection...")
        
        try:
            from vision.enhanced_vision_pipeline import get_vision_pipeline
            
            pipeline = get_vision_pipeline()
            await pipeline.initialize()
            
            # Test Stage 1: Screen Region Analysis
            print(f"   Testing Stage 1: Screen Region Analysis...")
            stage1_result = await pipeline.screen_analyzer.analyze_region('control_center')
            
            if stage1_result['success']:
                region = stage1_result['segmented_region']
                print(f"   ✅ Region analyzed: {region.width}x{region.height}")
                print(f"   Region position: ({region.x}, {region.y})")
                
                # Test Stage 2: Icon Detection
                print(f"   Testing Stage 2: Icon Detection...")
                stage2_result = await pipeline.icon_detector.detect_icon(region, 'control_center')
                
                if stage2_result['success']:
                    best_detection = stage2_result['detection_results']['best']
                    print(f"   ✅ Icon detected via {best_detection.method}")
                    print(f"   Bounding box: {best_detection.bounding_box}")
                    print(f"   Center point: {best_detection.center_point}")
                    print(f"   Confidence: {best_detection.confidence:.2%}")
                    
                    # Check if coordinates are reasonable
                    if best_detection.center_point:
                        x, y = best_detection.center_point
                        print(f"\n5️⃣  Coordinate Analysis:")
                        print(f"   Detected: ({x}, {y})")
                        print(f"   Expected: ({expected_x}, {expected_y})")
                        
                        # Convert to global coordinates
                        global_x = x + region.x
                        global_y = y + region.y
                        print(f"   Global: ({global_x}, {global_y})")
                        
                        # Check if reasonable
                        if abs(global_x - expected_x) <= 50 and abs(global_y - expected_y) <= 10:
                            print(f"   ✅ Coordinates are reasonable!")
                        else:
                            print(f"   ⚠️  Coordinates are off by: ({global_x - expected_x}, {global_y - expected_y})")
                            
                            if global_y > 50:
                                print(f"   ❌ Y coordinate too large - Control Center should be in menu bar!")
                            if global_x > 2000:
                                print(f"   ❌ X coordinate too large - Control Center should be around {expected_x}!")
                else:
                    print(f"   ❌ Icon detection failed: {stage2_result.get('error', 'Unknown error')}")
            else:
                print(f"   ❌ Region analysis failed: {stage1_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n" + "="*70)
    print("✅ Test complete!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(test_control_center_detection())