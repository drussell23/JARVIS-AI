#!/usr/bin/env python3
"""Check if cliclick is installed for better mouse control"""

import subprocess
import os

def check_cliclick():
    """Check if cliclick is installed"""
    print("🖱️  Checking cliclick installation...")
    print("="*60)
    
    try:
        result = subprocess.run(['which', 'cliclick'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ cliclick is installed at:", result.stdout.strip())
            
            # Test cliclick
            test = subprocess.run(['cliclick', '-V'], capture_output=True, text=True)
            print("   Version info:", test.stdout.strip())
        else:
            print("❌ cliclick is not installed")
            print("\nTo install cliclick (provides better mouse control):")
            print("1. Using Homebrew:")
            print("   brew install cliclick")
            print("\n2. Or download from:")
            print("   https://github.com/BlueM/cliclick")
            print("\ncliclick enables more reliable click-and-hold operations")
            print("which may help with selecting Toronto in the Weather app")
    
    except Exception as e:
        print(f"Error checking cliclick: {e}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    check_cliclick()