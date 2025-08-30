#!/usr/bin/env python3
"""Test to find the exact import error"""

import sys
import os
import traceback

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), "backend")
sys.path.insert(0, backend_path)

# Set minimal environment
os.environ["JARVIS_MEMORY_LEVEL"] = "critical"
os.environ["DYLD_LIBRARY_PATH"] = os.path.join(backend_path, "swift_bridge/.build/release")

print("Testing imports to find the error...")
print("=" * 50)

# Test each import step by step
try:
    print("1. Testing vision.rust_integration...")
    from vision.rust_integration import RustAccelerator
    print("   ✅ Success")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    traceback.print_exc()

try:
    print("\n2. Testing unified_rust_service...")
    from unified_rust_service import UnifiedRustService
    print("   ✅ Success")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    traceback.print_exc()

try:
    print("\n3. Testing api.lazy_enhanced_vision_api...")
    from api.lazy_enhanced_vision_api import router
    print("   ✅ Success")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    traceback.print_exc()

try:
    print("\n4. Testing api.enhanced_vision_api...")
    from api.enhanced_vision_api import router
    print("   ✅ Success")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    traceback.print_exc()

print("\n" + "=" * 50)
print("Import testing complete.")