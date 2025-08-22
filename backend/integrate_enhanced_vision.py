#!/usr/bin/env python3
"""
Integration script to update JARVIS with enhanced vision and AI core
"""

import os
import shutil
from datetime import datetime


def backup_file(filepath):
    """Create backup of file before modifying"""
    if os.path.exists(filepath):
        backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(filepath, backup_path)
        print(f"✅ Backed up: {filepath} -> {backup_path}")
        return backup_path
    return None


def update_main_py():
    """Update main.py to use enhanced vision API"""
    main_path = "main.py"
    
    # Read current main.py
    with open(main_path, 'r') as f:
        content = f.read()
    
    # Replace old vision API import with enhanced
    replacements = [
        # Comment out old vision API
        (
            "from api.vision_api import router as vision_router\n    app.include_router(vision_router)",
            "# from api.vision_api import router as vision_router  # Replaced with enhanced_vision_api\n    # app.include_router(vision_router)"
        ),
        # Add enhanced vision API
        (
            "# Include Vision WebSocket API for real-time monitoring",
            """# Include Enhanced Vision API with Claude integration
try:
    from api.enhanced_vision_api import router as enhanced_vision_router
    app.include_router(enhanced_vision_router)
    logger.info("Enhanced Vision API routes added - Claude-powered vision system activated!")
    ENHANCED_VISION_AVAILABLE = True
except Exception as e:
    logger.warning(f"Failed to initialize Enhanced Vision API: {e}")
    ENHANCED_VISION_AVAILABLE = False

# Include Vision WebSocket API for real-time monitoring"""
        ),
    ]
    
    # Apply replacements
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"✅ Updated: {old[:50]}...")
    
    # Write updated content
    with open(main_path, 'w') as f:
        f.write(content)
    
    print("✅ Updated main.py with enhanced vision API")


def update_jarvis_voice_api():
    """Update jarvis_voice_api.py to use AI core"""
    api_path = "api/jarvis_voice_api.py"
    
    if not os.path.exists(api_path):
        print(f"⚠️  {api_path} not found")
        return
    
    # Read current file
    with open(api_path, 'r') as f:
        content = f.read()
    
    # Add AI core import
    if "from core.jarvis_ai_core import get_jarvis_ai_core" not in content:
        # Add import after other imports
        import_line = "from core.jarvis_ai_core import get_jarvis_ai_core\n"
        
        # Find a good place to insert
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith("logger = logging.getLogger"):
                lines.insert(i + 1, import_line)
                break
        
        content = '\n'.join(lines)
        print("✅ Added AI core import to jarvis_voice_api.py")
    
    # Write updated content
    with open(api_path, 'w') as f:
        f.write(content)


def create_requirements_additions():
    """Create additional requirements needed"""
    additions = """
# Enhanced Vision and AI Requirements
anthropic>=0.18.1
opencv-python>=4.8.0
pytesseract>=0.3.10
Pillow>=10.0.0
pyobjc-framework-Quartz>=9.0  # macOS screen capture
"""
    
    with open("requirements_enhanced.txt", 'w') as f:
        f.write(additions)
    
    print("✅ Created requirements_enhanced.txt")


def update_frontend_jarvis_voice():
    """Update JarvisVoice.js to use new speech recognition manager"""
    js_path = "../frontend/src/components/JarvisVoice.js"
    
    if not os.path.exists(js_path):
        print(f"⚠️  {js_path} not found")
        return
    
    # Read current file
    with open(js_path, 'r') as f:
        content = f.read()
    
    # Add import for SpeechRecognitionManager
    if "import SpeechRecognitionManager" not in content:
        import_line = "import SpeechRecognitionManager from '../utils/SpeechRecognitionManager';\n"
        
        # Insert after other imports
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith("import SpeechDebug"):
                lines.insert(i + 1, import_line)
                break
        
        content = '\n'.join(lines)
        print("✅ Added SpeechRecognitionManager import")
    
    # Write updated content
    with open(js_path, 'w') as f:
        f.write(content)


def main():
    """Run all integration updates"""
    print("🚀 Starting JARVIS Enhanced Vision Integration")
    print("=" * 50)
    
    # Change to backend directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Backup important files
    print("\n📦 Creating backups...")
    backup_file("main.py")
    backup_file("api/jarvis_voice_api.py")
    
    # Update files
    print("\n🔧 Updating files...")
    update_main_py()
    update_jarvis_voice_api()
    create_requirements_additions()
    update_frontend_jarvis_voice()
    
    print("\n✅ Integration complete!")
    print("\n📋 Next steps:")
    print("1. Install new requirements: pip install -r requirements_enhanced.txt")
    print("2. Restart the backend: ./start_jarvis_backend.sh")
    print("3. Refresh the frontend")
    print("4. Test with 'Hey JARVIS, activate full autonomy'")
    
    print("\n🎯 Expected improvements:")
    print("- Vision WebSocket will connect properly")
    print("- All AI operations use Claude API")
    print("- Continuous screen monitoring in autonomous mode")
    print("- Speech recognition state properly managed")
    print("- Multi-window analysis capabilities")


if __name__ == "__main__":
    main()