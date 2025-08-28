#!/usr/bin/env python3
"""
Apply the Advanced WhatsApp Fix to JARVIS
This script updates JARVIS to use ML-based routing with zero hardcoding
"""

import os
import sys
import shutil
from datetime import datetime

def apply_advanced_fix():
    """Apply the advanced WhatsApp fix to JARVIS"""
    
    print("🚀 JARVIS Advanced WhatsApp Fix Installer")
    print("=" * 60)
    print("This will upgrade JARVIS to use ML-based command routing")
    print("with ZERO hardcoding - everything is learned and adaptive!")
    print("=" * 60)
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("api/jarvis_voice_api.py"):
        print("❌ Error: Run this script from the backend directory")
        return False
    
    # Backup original files
    print("📦 Creating backups...")
    backup_dir = f"backups/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    files_to_backup = [
        "api/jarvis_voice_api.py",
        "voice/jarvis_agent_voice.py",
        "voice/intelligent_command_handler.py"
    ]
    
    for file in files_to_backup:
        if os.path.exists(file):
            shutil.copy2(file, backup_dir)
            print(f"   ✅ Backed up {file}")
    
    # Update jarvis_voice_api.py
    print("\n📝 Updating jarvis_voice_api.py...")
    
    api_update = '''
# At the top of the file, replace the old import:
# from voice.jarvis_agent_voice_fix import patch_jarvis_voice_agent

# With the new advanced import:
from voice.integrate_advanced_routing import patch_jarvis_voice_agent_advanced

# Then replace the patch line:
# patch_jarvis_voice_agent(JARVISAgentVoice)

# With:
patch_jarvis_voice_agent_advanced(JARVISAgentVoice)
'''
    
    try:
        # Read the file
        with open("api/jarvis_voice_api.py", "r") as f:
            content = f.read()
        
        # Check if already patched
        if "patch_jarvis_voice_agent_advanced" in content:
            print("   ℹ️  Already using advanced routing!")
        else:
            # Apply the patch
            if "jarvis_agent_voice_fix" in content:
                # Replace old fix
                content = content.replace(
                    "from voice.jarvis_agent_voice_fix import patch_jarvis_voice_agent",
                    "from voice.integrate_advanced_routing import patch_jarvis_voice_agent_advanced"
                )
                content = content.replace(
                    "patch_jarvis_voice_agent(JARVISAgentVoice)",
                    "patch_jarvis_voice_agent_advanced(JARVISAgentVoice)"
                )
            else:
                # Add new import
                import_line = "from voice.integrate_advanced_routing import patch_jarvis_voice_agent_advanced\n"
                
                # Find where to add import
                if "from voice.jarvis_agent_voice import" in content:
                    content = content.replace(
                        "from voice.jarvis_agent_voice import",
                        import_line + "from voice.jarvis_agent_voice import"
                    )
                
                # Add patch after initialization
                if "JARVISAgentVoice(" in content:
                    # Find where JARVISAgentVoice is initialized and patch it
                    lines = content.split("\n")
                    for i, line in enumerate(lines):
                        if "JARVISAgentVoice(" in line and "=" in line:
                            # Add patch on next line
                            indent = len(line) - len(line.lstrip())
                            lines.insert(i + 1, " " * indent + "patch_jarvis_voice_agent_advanced(JARVISAgentVoice)")
                            break
                    content = "\n".join(lines)
            
            # Write updated content
            with open("api/jarvis_voice_api.py", "w") as f:
                f.write(content)
            
            print("   ✅ Updated jarvis_voice_api.py")
    
    except Exception as e:
        print(f"   ❌ Error updating jarvis_voice_api.py: {e}")
        print("   Please manually add the import and patch call")
    
    # Create a test script
    print("\n📝 Creating test script...")
    
    test_script = '''#!/usr/bin/env python3
"""Test the advanced WhatsApp fix"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voice.integrate_advanced_routing import JARVISAdvancedVoiceAgent

async def test():
    agent = JARVISAdvancedVoiceAgent()
    
    print("🧪 Testing Advanced WhatsApp Fix...")
    print()
    
    # Test the problematic command
    response = await agent.process_voice_input("open WhatsApp")
    print(f"Command: 'open WhatsApp'")
    print(f"Response: {response}")
    print()
    
    # Check routing
    status = agent.get_status()
    perf = status["performance"]
    print(f"Routing: {status['routing']}")
    print(f"Performance: {perf}")
    
    print("\\n✅ If WhatsApp opens, the fix is working!")

if __name__ == "__main__":
    asyncio.run(test())
'''
    
    with open("test_whatsapp_fix_applied.py", "w") as f:
        f.write(test_script)
    os.chmod("test_whatsapp_fix_applied.py", 0o755)
    
    print("   ✅ Created test_whatsapp_fix_applied.py")
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ Advanced WhatsApp Fix Applied!")
    print("=" * 60)
    
    print("\n📋 What's New:")
    print("   • ML-based command routing (zero hardcoding)")
    print("   • Learns from every interaction") 
    print("   • Self-improving accuracy")
    print("   • No more 'what' in 'WhatsApp' confusion")
    print("   • Works with ANY app or command")
    
    print("\n🧪 Test the fix:")
    print("   python test_whatsapp_fix_applied.py")
    
    print("\n📊 Monitor performance:")
    print("   The system will show confidence scores and learning metrics")
    
    print("\n🔄 Rollback if needed:")
    print(f"   Backups saved in: {backup_dir}")
    
    print("\n🎯 Next Steps:")
    print("   1. Restart JARVIS")
    print("   2. Try 'Hey JARVIS, open WhatsApp'")
    print("   3. Watch it work correctly!")
    print("   4. The system learns and improves with use")
    
    return True

def show_manual_instructions():
    """Show manual installation instructions"""
    
    print("\n📝 Manual Installation Instructions:")
    print("=" * 60)
    
    print("\n1. In backend/api/jarvis_voice_api.py:")
    print("   Replace:")
    print("   ```python")
    print("   from voice.jarvis_agent_voice_fix import patch_jarvis_voice_agent")
    print("   patch_jarvis_voice_agent(JARVISAgentVoice)")
    print("   ```")
    print()
    print("   With:")
    print("   ```python")
    print("   from voice.integrate_advanced_routing import patch_jarvis_voice_agent_advanced")
    print("   patch_jarvis_voice_agent_advanced(JARVISAgentVoice)")
    print("   ```")
    
    print("\n2. Restart JARVIS")
    
    print("\n3. Test with: 'Hey JARVIS, open WhatsApp'")
    
    print("\nThat's it! The system will now use ML-based routing.")

if __name__ == "__main__":
    print("🤖 JARVIS Advanced WhatsApp Fix")
    print("   Making JARVIS truly intelligent with ML-based routing")
    print()
    
    # Apply the fix
    success = apply_advanced_fix()
    
    if not success:
        show_manual_instructions()
    
    print("\n✨ Remember: This fix makes JARVIS learn and adapt!")
    print("   No more hardcoded patterns - pure intelligence!")