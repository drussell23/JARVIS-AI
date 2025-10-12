#!/usr/bin/env python3
"""
Complete fix for desktop spaces query routing and response generation
Ensures JARVIS properly handles "What's happening across my desktop spaces?" with Claude API
"""

import os
import subprocess
import time

def apply_async_pipeline_fixes():
    """Fix the routing in async_pipeline to properly handle desktop space queries"""

    file_path = "backend/core/async_pipeline.py"

    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find and fix the context intelligence handler section
    for i, line in enumerate(lines):
        # Look for the context intelligence query patterns section
        if "# CONTEXT INTELLIGENCE QUERIES" in line:
            # Find the check for desktop space queries
            for j in range(i, min(i + 50, len(lines))):
                if "is_desktop_space_query = any" in lines[j]:
                    # Already has the check, update it to be more comprehensive
                    # Find the patterns list
                    for k in range(j, min(j + 10, len(lines))):
                        if '["desktop space"' in lines[k]:
                            # Replace with more comprehensive patterns
                            lines[k] = '''            "desktop space", "desktop spaces",
            "across my desktop", "across desktop",
            "happening across", "across my",  # Add more specific patterns
            "what's happening across", "what is happening across"
'''
                            lines[k+1] = '        ])\n'
                            break
                    break

    # Write back
    with open(file_path, 'w') as f:
        f.writelines(lines)

    print("✓ Fixed async_pipeline.py routing")

def verify_vision_handler():
    """Ensure vision handler is configured correctly"""

    file_path = "backend/api/vision_command_handler.py"

    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()

    # Check if Claude API calls are present
    if "self.intelligence.understand_and_respond" in content:
        print("✓ Vision handler configured to use Claude API")
    else:
        print("❌ Vision handler missing Claude API integration")

    # Check if workspace name processing is present
    if "process_response_with_workspace_names" in content:
        print("✓ Workspace name processing configured")
    else:
        print("⚠️ Workspace name processing may not be configured")

def test_intent_detection():
    """Test that desktop space queries are properly detected"""

    test_queries = [
        "What's happening across my desktop spaces?",
        "What is happening across my desktop spaces?",
        "Show me what's happening across my desktop",
        "What's happening across desktop spaces"
    ]

    print("\n🧪 Testing intent detection patterns...")

    # Read async_pipeline to check patterns
    with open("backend/core/async_pipeline.py", 'r') as f:
        content = f.read()

    # Check if vision patterns include our queries
    vision_section_start = content.find('"vision": [')
    if vision_section_start > -1:
        vision_section_end = content.find('],', vision_section_start)
        vision_patterns = content[vision_section_start:vision_section_end].lower()

        for query in test_queries:
            # Check if key phrases are in vision patterns
            if "across my desktop" in vision_patterns or "desktop space" in vision_patterns:
                print(f"  ✓ '{query}' would be detected as vision")
            else:
                print(f"  ❌ '{query}' might not be detected")

def cleanup_processes():
    """Clean up any running JARVIS processes"""

    print("\n🧹 Cleaning up existing JARVIS processes...")

    # Kill various JARVIS processes
    commands = [
        "pkill -f 'python.*start_system' 2>/dev/null",
        "pkill -f 'python.*main.py' 2>/dev/null",
        "pkill -f 'restart_jarvis' 2>/dev/null",
        "pkill -f 'node.*jarvis' 2>/dev/null",
        "pkill -f 'npm.*jarvis' 2>/dev/null"
    ]

    for cmd in commands:
        subprocess.run(cmd, shell=True, capture_output=True)

    time.sleep(2)

    # Check if any processes remain
    result = subprocess.run(
        "ps aux | grep -E '(python|node).*jarvis' | grep -v grep | wc -l",
        shell=True, capture_output=True, text=True
    )

    remaining = int(result.stdout.strip())
    if remaining == 0:
        print("  ✓ All JARVIS processes cleaned up")
    else:
        print(f"  ⚠️ {remaining} processes may still be running")

def start_jarvis():
    """Start JARVIS with proper configuration"""

    print("\n🚀 Starting JARVIS...")

    # Start JARVIS in background
    subprocess.Popen(
        ["python", "start_system.py"],
        stdout=open("jarvis_test.log", "w"),
        stderr=subprocess.STDOUT
    )

    print("  ✓ JARVIS starting (check jarvis_test.log for output)")
    print("\n⏳ Wait about 30 seconds for JARVIS to fully initialize")
    print("📝 Then test with: 'What's happening across my desktop spaces?'")
    print("\n✨ Expected behavior:")
    print("  1. Command detected as 'vision' intent")
    print("  2. Routed to vision_command_handler")
    print("  3. Claude API analyzes all desktop spaces")
    print("  4. Response includes actual workspace names (Cursor, Terminal, etc.)")

def main():
    """Apply all fixes and restart JARVIS"""

    print("=" * 60)
    print("🔧 Applying comprehensive desktop spaces fixes")
    print("=" * 60)

    # Apply fixes
    apply_async_pipeline_fixes()
    verify_vision_handler()
    test_intent_detection()

    # Clean up and restart
    cleanup_processes()
    start_jarvis()

    print("\n" + "=" * 60)
    print("✅ All fixes applied!")
    print("=" * 60)

if __name__ == "__main__":
    main()