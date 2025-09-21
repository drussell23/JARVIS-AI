#!/usr/bin/env python3
"""
Fix vision import issues for chatbot
"""

import os
import sys

def fix_vision_imports():
    """Ensure vision imports work correctly"""
    
    # Add backend to Python path if not already there
    backend_path = os.path.dirname(os.path.abspath(__file__))
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    
    # Test imports
    print("🔧 Fixing vision imports...")
    
    try:
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
        print("✅ Vision analyzer import successful")
        
        # Test with API key
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            analyzer = ClaudeVisionAnalyzer(api_key, enable_realtime=True)
            print("✅ Vision analyzer initialized successfully")
            return True
        else:
            print("⚠️  ANTHROPIC_API_KEY not set")
            return False
            
    except Exception as e:
        print(f"❌ Import error: {e}")
        
        # Try to fix by updating __init__.py files
        vision_init = os.path.join(backend_path, "vision", "__init__.py")
        if not os.path.exists(vision_init):
            print(f"Creating {vision_init}")
            open(vision_init, 'a').close()
        
        # Try again
        try:
            from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
            print("✅ Vision analyzer import successful after fix")
            return True
        except:
            print("❌ Still failing after fix")
            return False

def update_chatbot_import():
    """Update chatbot to handle import issues"""
    
    chatbot_file = "chatbots/claude_vision_chatbot.py"
    
    print(f"\n📝 Updating {chatbot_file}...")
    
    # Read current content
    with open(chatbot_file, 'r') as f:
        content = f.read()
    
    # Check if already fixed
    if "sys.path.insert" in content:
        print("✅ Already fixed")
        return
    
    # Add import fix at the beginning
    fix_code = """# Fix import path for vision modules
import sys
import os
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

"""
    
    # Insert after the docstring
    lines = content.split('\n')
    insert_index = 0
    for i, line in enumerate(lines):
        if line.strip() == '"""' and i > 0:  # End of docstring
            insert_index = i + 1
            break
    
    # Insert the fix
    lines.insert(insert_index, fix_code)
    
    # Save updated file
    with open(chatbot_file + '.bak', 'w') as f:
        f.write(content)  # Backup
    
    with open(chatbot_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print("✅ Chatbot updated with import fix")

if __name__ == "__main__":
    # Fix imports
    success = fix_vision_imports()
    
    if success:
        print("\n✅ Vision imports are working correctly")
    else:
        print("\n🔧 Applying fixes...")
        update_chatbot_import()
        
        # Test again
        if fix_vision_imports():
            print("\n✅ Vision imports fixed!")
        else:
            print("\n❌ Could not fix imports automatically")
            print("\nManual fix:")
            print("1. Ensure you're running from the JARVIS root directory")
            print("2. Set ANTHROPIC_API_KEY environment variable")
            print("3. Check that vision/claude_vision_analyzer_main.py exists")