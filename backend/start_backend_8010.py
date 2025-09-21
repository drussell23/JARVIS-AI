#!/usr/bin/env python3
"""
Start backend on port 8010 for frontend compatibility
"""

import subprocess
import sys
import os

def main():
    """Start the backend on port 8010"""
    print("🚀 Starting JARVIS Backend on port 8010 (for frontend compatibility)")
    print("=" * 60)
    
    # Set environment variable
    os.environ['BACKEND_PORT'] = '8010'
    
    # Start the main backend
    try:
        subprocess.run([
            sys.executable, 
            "main.py", 
            "--port", 
            "8010"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTry running directly:")
        print("  python main.py --port 8010")

if __name__ == "__main__":
    main()