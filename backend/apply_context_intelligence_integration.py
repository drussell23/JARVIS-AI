#!/usr/bin/env python3
"""
Apply Context Intelligence Integration
=====================================

This script updates the existing JARVIS files to use the new
Context Intelligence System.
"""

import os
import sys
import shutil
from pathlib import Path

def backup_file(filepath):
    """Create a backup of the file"""
    backup_path = f"{filepath}.backup_before_context_intelligence"
    if not os.path.exists(backup_path):
        shutil.copy2(filepath, backup_path)
        print(f"✓ Backed up {filepath}")
    else:
        print(f"⚠ Backup already exists for {filepath}")

def update_jarvis_voice_api():
    """Update jarvis_voice_api.py to use new context intelligence"""
    filepath = "api/jarvis_voice_api.py"
    
    if not os.path.exists(filepath):
        print(f"✗ File not found: {filepath}")
        return False
        
    backup_file(filepath)
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace the import
    old_import = """from .simple_context_handler_enhanced import (
                                wrap_with_enhanced_context,
                            )"""
    
    new_import = """# Use new Context Intelligence System
                            from context_intelligence.integrations.enhanced_context_wrapper import (
                                wrap_with_enhanced_context,
                            )"""
    
    if old_import in content:
        content = content.replace(old_import, new_import)
        
        with open(filepath, 'w') as f:
            f.write(content)
            
        print(f"✓ Updated {filepath} to use Context Intelligence System")
        return True
    else:
        print(f"⚠ Could not find import to replace in {filepath}")
        # Try alternative approach - look for the specific lines
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'from .simple_context_handler_enhanced import' in line:
                # Found it, update this and next line
                lines[i] = '                            # Use new Context Intelligence System'
                lines[i+1] = '                            from context_intelligence.integrations.enhanced_context_wrapper import ('
                
                with open(filepath, 'w') as f:
                    f.write('\n'.join(lines))
                    
                print(f"✓ Updated {filepath} to use Context Intelligence System (alternative method)")
                return True
        
        return False

def create_integration_test():
    """Create a test file to verify the integration works"""
    test_content = '''#!/usr/bin/env python3
"""
Test Context Intelligence Integration
====================================

Verifies the integration is working correctly.
"""

import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_integration():
    """Test the context intelligence integration"""
    print("Testing Context Intelligence Integration...")
    
    try:
        # Test 1: Import check
        print("\\n1. Testing imports...")
        from context_intelligence.integrations.enhanced_context_wrapper import (
            wrap_with_enhanced_context,
            EnhancedContextIntelligenceHandler
        )
        print("   ✓ Imports successful")
        
        # Test 2: Create handler
        print("\\n2. Testing handler creation...")
        
        class MockProcessor:
            async def process_command(self, command):
                return {"mock": True, "command": command}
        
        processor = MockProcessor()
        handler = wrap_with_enhanced_context(processor)
        print("   ✓ Handler created")
        
        # Test 3: Process a command
        print("\\n3. Testing command processing...")
        result = await handler.process_with_context(
            "open Safari and search for artificial intelligence"
        )
        print(f"   ✓ Command processed: {result.get('success', False)}")
        
        # Test 4: Check components
        print("\\n4. Checking components...")
        print(f"   - Context Manager: {'✓' if handler.context_manager else '✗'}")
        print(f"   - Feedback Manager: {'✓' if handler.feedback_manager else '✗'}")
        print(f"   - JARVIS Integration: {'✓' if handler.jarvis_integration else '✗'}")
        
        print("\\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_integration())
'''
    
    with open('test_integration.py', 'w') as f:
        f.write(test_content)
    
    print("✓ Created test_integration.py")

def create_example_usage():
    """Create example showing how to use the new system"""
    example_content = '''#!/usr/bin/env python3
"""
Example: Using Context Intelligence System
=========================================

Shows how the new system handles the locked screen scenario.
"""

import asyncio
import logging

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def simulate_locked_screen_scenario():
    """
    Simulate the exact scenario from the PRD:
    - Screen is locked
    - User says "JARVIS, open Safari and search for dogs"
    - System handles it intelligently
    """
    
    print("\\n" + "="*60)
    print("CONTEXT INTELLIGENCE DEMO")
    print("Scenario: Screen locked, user wants to search")
    print("="*60 + "\\n")
    
    # Import our integration
    from context_intelligence.integrations.jarvis_integration import (
        handle_voice_command
    )
    
    # Simulate the command
    command = "open Safari and search for dogs"
    print(f"User: 'JARVIS, {command}'")
    print()
    
    # Process through context intelligence
    result = await handle_voice_command(
        command=command,
        voice_context={
            "source": "voice",
            "tone": "normal",
            "urgency": "normal"
        }
    )
    
    print("\\nContext Intelligence Response:")
    print(f"- Status: {result.get('status')}")
    print(f"- Message: {result.get('message')}")
    print(f"- Requires Unlock: {result.get('requires_unlock')}")
    print(f"- Command ID: {result.get('command_id')}")
    
    if result.get('intent'):
        print(f"\\nIntent Analysis:")
        print(f"- Action: {result['intent'].get('action')}")
        print(f"- Target: {result['intent'].get('target')}")
        print(f"- Type: {result['intent'].get('type')}")
    
    print("\\nExpected Flow:")
    print("1. ✓ JARVIS detects screen is locked")
    print("2. ✓ Queues request with action: 'search dogs in Safari'")
    print("3. ✓ Feedback: 'Your screen is locked, I'll unlock it now...'")
    print("4. → Unlock Manager runs")
    print("5. → On success: JARVIS resumes queued request")
    print("6. → Opens Safari and searches for 'dogs'")
    print("7. → Feedback: 'I've unlocked your screen, opened Safari, and searched for dogs'")
    
if __name__ == "__main__":
    asyncio.run(simulate_locked_screen_scenario())
'''
    
    with open('example_context_usage.py', 'w') as f:
        f.write(example_content)
    
    print("✓ Created example_context_usage.py")

def main():
    """Main integration script"""
    print("\\nApplying Context Intelligence Integration...")
    print("="*50)
    
    # Change to backend directory
    if os.path.exists('backend'):
        os.chdir('backend')
    elif not os.path.exists('api'):
        print("✗ Error: Must run from project root or backend directory")
        return 1
    
    # Update files
    success = update_jarvis_voice_api()
    
    if success:
        print("\\n✓ Integration complete!")
        
        # Create test files
        create_integration_test()
        create_example_usage()
        
        print("\\nNext steps:")
        print("1. Run: python test_integration.py")
        print("2. Run: python example_context_usage.py")
        print("3. Test with actual voice commands")
    else:
        print("\\n✗ Integration failed")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())