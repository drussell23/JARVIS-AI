#!/usr/bin/env python3
"""
Diagnose vision system issues
"""

import sys
import os
import platform
sys.path.insert(0, os.path.abspath('.'))

print("🔍 JARVIS Vision System Diagnostics")
print("=" * 50)

# Check which application is running this script
print("\n📱 Process Information:")
print(f"  • Python executable: {sys.executable}")
print(f"  • Process ID: {os.getpid()}")

# Try to identify parent process (Cursor, Terminal, etc.)
try:
    import subprocess
    parent_info = subprocess.run(
        ["ps", "-p", str(os.getppid()), "-o", "comm="],
        capture_output=True,
        text=True
    )
    if parent_info.returncode == 0:
        parent = parent_info.stdout.strip()
        print(f"  • Parent process: {parent}")
        
        if "Cursor" in parent:
            print("\n⚠️  IMPORTANT: Running from Cursor AI!")
            print("   → Cursor needs Screen Recording permission (not Terminal)")
            print("   → After granting permission, you MUST restart Cursor completely")
        elif "Terminal" in parent:
            print("\n⚠️  Running from Terminal")
            print("   → Terminal needs Screen Recording permission")
except:
    pass

print()

# Step 1: Check imports
print("1️⃣ Checking package imports...")
packages_ok = True

try:
    import cv2
    print("✅ opencv-python (cv2) imported successfully")
except ImportError as e:
    print(f"❌ opencv-python import failed: {e}")
    packages_ok = False

try:
    import pytesseract
    print("✅ pytesseract imported successfully")
except ImportError as e:
    print(f"❌ pytesseract import failed: {e}")
    packages_ok = False

try:
    import Quartz
    print("✅ Quartz framework imported successfully")
except ImportError as e:
    print(f"❌ Quartz import failed: {e}")
    packages_ok = False

try:
    from PIL import Image
    print("✅ Pillow (PIL) imported successfully")
except ImportError as e:
    print(f"❌ Pillow import failed: {e}")
    packages_ok = False

# Step 2: Check Tesseract binary
print("\n2️⃣ Checking Tesseract OCR binary...")
import subprocess
try:
    result = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Tesseract binary found")
        print(f"   Version: {result.stdout.split()[1] if result.stdout else 'Unknown'}")
    else:
        print("❌ Tesseract binary not working properly")
except FileNotFoundError:
    print("❌ Tesseract binary not found - run: brew install tesseract")

# Step 3: Check screen capture permission
print("\n3️⃣ Checking screen capture permission...")
if packages_ok and 'Quartz' in sys.modules:
    try:
        screenshot = Quartz.CGDisplayCreateImage(Quartz.CGMainDisplayID())
        if screenshot is None:
            print("❌ Screen Recording permission NOT granted")
            print("\n🔧 TO FIX THIS:")
            print("1. Open System Preferences → Security & Privacy → Privacy → Screen Recording")
            print("2. Look for and check the box next to:")
            
            # Check parent process again for specific instructions
            try:
                parent_info = subprocess.run(
                    ["ps", "-p", str(os.getppid()), "-o", "comm="],
                    capture_output=True,
                    text=True
                )
                if parent_info.returncode == 0:
                    parent = parent_info.stdout.strip()
                    if "Cursor" in parent:
                        print("   ☐ Cursor (NOT Terminal - this is critical!)")
                        print("3. After checking the box, you MUST:")
                        print("   • Completely quit Cursor (Cmd+Q)")
                        print("   • Reopen Cursor")
                        print("   • Run JARVIS again")
                    else:
                        print("   ☐ Terminal (or your current terminal app)")
                        print("   ☐ Python/Python3 (if it appears)")
                        print("3. Restart your terminal/IDE after granting permission")
            except:
                print("   ☐ Terminal/IDE that's running this script")
                print("   ☐ Python/Python3 (if it appears)")
                print("3. Restart your terminal/IDE after granting permission")
        else:
            print("✅ Screen Recording permission granted!")
            width = Quartz.CGImageGetWidth(screenshot)
            height = Quartz.CGImageGetHeight(screenshot)
            print(f"   Screen resolution: {width}x{height}")
            print("   JARVIS can see your screen! 🎉")
    except Exception as e:
        print(f"❌ Error testing screen capture: {e}")

# Step 4: Test vision system initialization
print("\n4️⃣ Testing vision system initialization...")
try:
    from vision.screen_vision import ScreenVisionSystem, JARVISVisionIntegration
    print("✅ Vision modules imported successfully")
    
    # Try to initialize
    vision_system = ScreenVisionSystem()
    jarvis_integration = JARVISVisionIntegration(vision_system)
    print("✅ Vision system initialized successfully")
    
    # Test a simple capture
    import asyncio
    async def test_capture():
        try:
            context = await vision_system.get_screen_context()
            print(f"✅ Screen context captured: {context['screen_size']}")
            return True
        except Exception as e:
            print(f"❌ Screen capture failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    asyncio.run(test_capture())
    
except ImportError as e:
    print(f"❌ Vision module import failed: {e}")
except Exception as e:
    print(f"❌ Vision system initialization failed: {e}")
    import traceback
    traceback.print_exc()

# Step 5: Test JARVIS integration
print("\n5️⃣ Testing JARVIS vision integration...")
try:
    from voice.jarvis_agent_voice import JARVISAgentVoice
    jarvis = JARVISAgentVoice(user_name="Sir")
    print(f"✅ JARVIS initialized, vision_enabled: {jarvis.vision_enabled}")
    
    if jarvis.vision_enabled:
        # Test vision command
        async def test_command():
            try:
                response = await jarvis._handle_vision_command("can you see my screen")
                print(f"✅ Vision command response: {response[:100]}...")
            except Exception as e:
                print(f"❌ Vision command failed: {e}")
                import traceback
                traceback.print_exc()
        
        asyncio.run(test_command())
    else:
        print("❌ JARVIS vision is not enabled")
        
except Exception as e:
    print(f"❌ JARVIS integration test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n📋 Diagnostic Summary:")
print("=" * 50)

# Determine the main issue
main_issue = None
if not packages_ok:
    main_issue = "packages"
elif 'Quartz' in sys.modules:
    try:
        if Quartz.CGDisplayCreateImage(Quartz.CGMainDisplayID()) is None:
            main_issue = "permission"
    except:
        pass

# Provide targeted solution
if main_issue == "permission":
    print("❌ MAIN ISSUE: Screen Recording Permission")
    print("\n🚨 CURSOR AI USERS - CRITICAL:")
    print("1. You MUST grant permission to 'Cursor' (not Terminal)")
    print("2. System Preferences → Security & Privacy → Privacy → Screen Recording")
    print("3. Check the box next to 'Cursor'")
    print("4. COMPLETELY QUIT Cursor (Cmd+Q) and restart it")
    print("5. Run JARVIS again")
    print("\nThis is the #1 issue when running JARVIS from Cursor's terminal!")
elif main_issue == "packages":
    print("❌ MAIN ISSUE: Missing packages")
    print("\nRun these commands:")
    print("pip install opencv-python pytesseract Pillow pyobjc-framework-Quartz")
    print("brew install tesseract")
else:
    print("✅ Everything appears to be working!")
    print("JARVIS should be able to see your screen.")

print("\n💡 Quick Test:")
print("After fixing any issues above, run:")
print("python start_system.py")
print("Then say: 'Hey JARVIS, can you see my screen?'")