#!/usr/bin/env python3
"""Test backend startup with debug output"""

import sys
import importlib
import logging
import time

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def timed_import(module_name, description):
    """Import a module with timing"""
    print(f"\n{'='*60}")
    print(f"Importing {description}...")
    start = time.time()
    try:
        module = importlib.import_module(module_name)
        elapsed = time.time() - start
        print(f"✓ {description} imported in {elapsed:.2f}s")
        return module
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ {description} failed after {elapsed:.2f}s: {e}")
        raise

def timed_call(func, args=(), kwargs={}, description=""):
    """Call a function with timing"""
    print(f"\n{'='*60}")
    print(f"Calling {description}...")
    start = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"✓ {description} completed in {elapsed:.2f}s")
        return result
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ {description} failed after {elapsed:.2f}s: {e}")
        raise

# Start testing
print("Testing backend startup sequence...")

# Test key imports
main = timed_import("main", "main module")

# Check what happens after vision status
print("\n" + "="*60)
print("Checking after vision status endpoint...")

# Look for the code after vision status
if hasattr(main, 'VOICE_API_AVAILABLE'):
    print(f"VOICE_API_AVAILABLE = {main.VOICE_API_AVAILABLE}")
    
if hasattr(main, 'voice_api'):
    print("voice_api exists")

print("\nChecking app routes...")
if hasattr(main, 'app'):
    print(f"Number of routes: {len(main.app.routes)}")
    for route in main.app.routes[-5:]:
        print(f"  - {route.path}")

print("\n✅ Backend startup test completed!")