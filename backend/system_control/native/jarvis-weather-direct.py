#!/usr/bin/env python3
"""
Direct weather provider using Swift for city-based queries
and Python fallback for current location
"""

import json
import subprocess
import sys
import os
from pathlib import Path

# Find the Swift tool
SCRIPT_DIR = Path(__file__).parent
SWIFT_TOOL = SCRIPT_DIR / "jarvis-weather"
FALLBACK_TOOL = SCRIPT_DIR / "jarvis-weather-fallback.py"

def main():
    args = sys.argv[1:]
    command = args[0] if args else 'current'
    pretty = '--pretty' in args or '-p' in args
    
    # For current location, use fallback (avoids location permission issues)
    if command == 'current':
        # Use fallback for current location
        result = subprocess.run(
            [sys.executable, str(FALLBACK_TOOL), 'current'] + (['--pretty'] if pretty else []),
            capture_output=True,
            text=True
        )
        print(result.stdout)
        return
    
    # For city queries, use Swift tool if available
    elif command == 'city' and SWIFT_TOOL.exists():
        # Pass through to Swift tool
        result = subprocess.run(
            [str(SWIFT_TOOL)] + args,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            # Fallback if Swift fails
            result = subprocess.run(
                [sys.executable, str(FALLBACK_TOOL)] + args,
                capture_output=True,
                text=True
            )
            print(result.stdout)
        return
    
    # For everything else, use fallback
    result = subprocess.run(
        [sys.executable, str(FALLBACK_TOOL)] + args,
        capture_output=True,
        text=True
    )
    print(result.stdout)

if __name__ == '__main__':
    main()