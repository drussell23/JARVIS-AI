#!/usr/bin/env python3
"""
Test Coordinate Fix
==================

Test the coordinate calculation with proper DPI scaling.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import pyautogui

logging.basicConfig(level=logging.INFO)


async def test_coordinate_fix():
    """Test coordinate calculation"""
    
    print("\n" + "="*70)
    print("🔧 Testing Coordinate Fix")
    print("="*70)
    
    # Get screen info
    screen_width, screen_height = pyautogui.size()
    print(f"\n1️⃣  Screen Info:")
    print(f"   Resolution: {screen_width}x{screen_height}")
    
    # Expected Control Center position (from debug)
    expected_cc_x = 1340  # 100px from right edge
    expected_cc_y = 15    # Center of menu bar
    
    print(f"\n2️⃣  Expected Control Center Position:")
    print(f"   X: {expected_cc_x}")
    print(f"   Y: {expected_cc_y}")
    
    # Test Enhanced Pipeline coordinate calculation
    print(f"\n3️⃣  Testing Enhanced Pipeline...")
    
    try:
        from vision.enhanced_vision_pipeline import get_vision_pipeline
        
        pipeline = get_vision_pipeline()
        await pipeline.initialize()
        
        # Test coordinate calculator directly
        from vision.enhanced_vision_pipeline.coordinate_calculator import CoordinateCalculator
        from vision.enhanced_vision_pipeline.icon_detection_engine import DetectionResult
        
        calc = CoordinateCalculator(pipeline.config)
        await calc.initialize()
        
        # Simulate detection result
        mock_detection = {
            'best': DetectionResult(
                found=True,
                bounding_box=(expected_cc_x - 10, expected_cc_y - 10, 20, 20),  # 20x20 icon
                confidence=0.95,
                method='test',
                center_point=(expected_cc_x, expected_cc_y),
                metadata={}
            )
        }
        
        # Test coordinate calculation
        context = {
            'region_offset': (0, 0),  # Menu bar starts at (0,0)
            'dpi_scale': 2.0  # Retina scaling
        }
        
        result = await calc.calculate_coordinates(mock_detection, context)
        
        if result['success']:
            coords = result['calculated_coordinates']
            print(f"   ✅ Calculation successful")
            print(f"   Global coordinates: ({coords.global_x}, {coords.global_y})")
            print(f"   Expected: ({expected_cc_x}, {expected_cc_y})")
            
            # Check if coordinates match
            if abs(coords.global_x - expected_cc_x) <= 5 and abs(coords.global_y - expected_cc_y) <= 5:
                print(f"   ✅ Coordinates are correct!")
            else:
                print(f"   ⚠️  Coordinates are off by: ({coords.global_x - expected_cc_x}, {coords.global_y - expected_cc_y})")
        else:
            print(f"   ❌ Calculation failed: {result['error']}")
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n" + "="*70)
    print("✅ Test complete!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(test_coordinate_fix())