#!/usr/bin/env python3
"""
Test if screen recording permissions are working
"""

import os
import sys

try:
    import AVFoundation
    import CoreMedia
    from Quartz import CoreVideo
    print("✅ macOS frameworks imported successfully")
except ImportError as e:
    print(f"❌ Failed to import macOS frameworks: {e}")
    sys.exit(1)

def test_screen_recording():
    """Test if we can create a screen capture session"""
    print("\n🔍 Testing screen recording permissions...")
    
    try:
        # Create a capture session
        session = AVFoundation.AVCaptureSession.alloc().init()
        print("✅ Created AVCaptureSession")
        
        # Try to create screen input for main display (ID 0)
        screen_input = AVFoundation.AVCaptureScreenInput.alloc().initWithDisplayID_(0)
        
        if screen_input:
            print("✅ Created screen input successfully")
            
            # Check if we can add the input
            if session.canAddInput_(screen_input):
                print("✅ Can add screen input to session")
                session.addInput_(screen_input)
                print("✅ Added screen input to session")
                
                # Try to start the session
                session.startRunning()
                print("✅ Started capture session - screen recording is working!")
                
                # Stop the session
                session.stopRunning()
                print("✅ Stopped capture session")
                
                return True
            else:
                print("❌ Cannot add screen input to session")
                return False
        else:
            print("❌ Failed to create screen input - likely permissions issue")
            print("\n⚠️  Please check System Preferences:")
            print("   1. Go to System Preferences > Security & Privacy > Privacy")
            print("   2. Select 'Screen Recording' from the left sidebar")
            print("   3. Ensure Terminal (or your Python app) is checked")
            print("   4. You may need to restart Terminal after granting permission")
            return False
            
    except Exception as e:
        print(f"❌ Error testing screen recording: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Screen Recording Permissions Test\n")
    print("=" * 60)
    
    success = test_screen_recording()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Screen recording permissions are working correctly!")
    else:
        print("❌ Screen recording permissions test failed")
        print("\nTo fix this:")
        print("1. Open System Preferences > Security & Privacy > Privacy > Screen Recording")
        print("2. Add and check Terminal (or iTerm, VS Code, etc.)")
        print("3. Restart your terminal application")
        print("4. Run this test again")