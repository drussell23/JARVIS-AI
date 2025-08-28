#!/usr/bin/env python3
"""Test voice component imports"""

import sys

def test_import(module_path, component_name):
    print(f"\nTesting {component_name}...")
    try:
        __import__(module_path)
        print(f"  ✓ {component_name} imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ {component_name} failed: {type(e).__name__}: {e}")
        return False

print("Testing voice component imports...")

# Test imports in order
components = [
    ("engines.voice_engine", "Voice Engine"),
    ("api.voice_api", "Voice API"),
]

for module, name in components:
    if not test_import(module, name):
        print(f"\nStopping at {name}")
        break