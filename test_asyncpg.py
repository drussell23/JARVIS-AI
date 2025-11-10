#!/usr/bin/env python3
"""Test asyncpg availability."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))

print("Testing asyncpg import...")
try:
    import asyncpg
    print(f"✅ asyncpg imported successfully: {asyncpg.__version__}")
except ImportError as e:
    print(f"❌ Failed to import asyncpg: {e}")

print("\nTesting cloud_database_adapter import...")
from backend.intelligence import cloud_database_adapter
print(f"ASYNCPG_AVAILABLE = {cloud_database_adapter.ASYNCPG_AVAILABLE}")

if cloud_database_adapter.ASYNCPG_AVAILABLE:
    print("✅ asyncpg is available in cloud_database_adapter")
else:
    print("❌ asyncpg is NOT available in cloud_database_adapter")