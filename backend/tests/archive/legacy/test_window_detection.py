#!/usr/bin/env python3
"""
Test window detection directly to identify ValueError
"""

import sys
import os
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_window_detection():
    """Test window detection directly"""
    print("=== Testing Window Detection ===\n")
    
    try:
        from vision.multi_space_window_detector import MultiSpaceWindowDetector
        
        detector = MultiSpaceWindowDetector()
        print("✓ MultiSpaceWindowDetector created")
        
        # Test the method that's likely causing ValueError
        print("\nTesting get_all_windows_across_spaces()...")
        result = detector.get_all_windows_across_spaces()
        
        print("✓ Window detection completed")
        print(f"\nResult structure:")
        print(f"  - current_space: {result.get('current_space')}")
        print(f"  - spaces: {len(result.get('spaces', []))} spaces found")
        print(f"  - windows: {len(result.get('windows', []))} windows found")
        print(f"  - space_window_map keys: {list(result.get('space_window_map', {}).keys())}")
        
        # Check for specific windows
        windows = result.get('windows', [])
        for w in windows[:5]:  # First 5 windows
            print(f"\n  Window: {w.app_name}")
            print(f"    Title: {w.window_title}")
            print(f"    Space: {w.space_id}")
            print(f"    Bounds: {w.bounds}")
            
    except Exception as e:
        print(f"✗ Window detection failed")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error message: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        
        # Check specific imports that might be failing
        print("\n\nChecking Quartz imports...")
        try:
            import Quartz
            print("✓ Quartz imported")
            
            # Try the specific function
            window_list = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionAll | 
                Quartz.kCGWindowListExcludeDesktopElements,
                Quartz.kCGNullWindowID
            )
            print(f"✓ CGWindowListCopyWindowInfo worked, got {len(window_list) if window_list else 0} windows")
            
        except Exception as e2:
            print(f"✗ Quartz error: {e2}")

if __name__ == "__main__":
    test_window_detection()