#!/usr/bin/env python3
"""Add debug prints to main.py temporarily to find hang"""

import re

# Read main.py
with open('main.py', 'r') as f:
    content = f.read()

# Find the section after vision status endpoint
pattern = r'(logger\.info\("Vision status endpoint added.*?\))'
match = re.search(pattern, content)

if match:
    # Add debug print right after
    insert_pos = match.end()
    debug_code = '''
print("\\n>>> DEBUG: Vision status endpoint completed")
import sys
sys.stdout.flush()
'''
    
    # Also add debug before Voice API
    voice_pattern = r'(# Include Voice API routes if available with memory management)'
    voice_match = re.search(voice_pattern, content)
    
    if voice_match:
        voice_pos = voice_match.start()
        voice_debug = '''
print("\\n>>> DEBUG: About to check VOICE_API_AVAILABLE")
sys.stdout.flush()
'''
        content = content[:voice_pos] + voice_debug + content[voice_pos:]
    
    content = content[:insert_pos] + debug_code + content[insert_pos:]
    
    # Write debug version
    with open('main_debug.py', 'w') as f:
        f.write(content)
    
    print("Created main_debug.py with debug prints")
else:
    print("Could not find insertion point")