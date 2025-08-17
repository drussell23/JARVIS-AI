#!/usr/bin/env python3
"""
Fix pyttsx3 macOS driver issue
"""

import os
import site

# Find the pyttsx3 driver file
site_packages = site.getsitepackages()[0]
driver_file = os.path.join(site_packages, 'pyttsx3', 'drivers', 'nsss.py')

if os.path.exists(driver_file):
    # Read the file
    with open(driver_file, 'r') as f:
        content = f.read()
    
    # Check if objc import is missing
    if 'import objc' not in content:
        # Add the import at the beginning
        fixed_content = 'import objc\n' + content
        
        # Write back
        try:
            with open(driver_file, 'w') as f:
                f.write(fixed_content)
            print(f"✅ Fixed pyttsx3 driver at: {driver_file}")
        except PermissionError:
            print(f"❌ Permission denied. Run with: python {__file__}")
            print(f"   Or manually add 'import objc' to the top of:")
            print(f"   {driver_file}")
    else:
        print("✅ pyttsx3 driver already fixed")
else:
    print(f"❌ Driver file not found at: {driver_file}")

# Test if it works now
try:
    import pyttsx3
    engine = pyttsx3.init()
    print("✅ pyttsx3 is now working!")
except Exception as e:
    print(f"❌ Still having issues: {e}")