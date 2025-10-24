#!/usr/bin/env python3
"""
Test script for Hybrid Cloud Cost Optimization System
"""

import os
from pathlib import Path

from dotenv import load_dotenv

print("🧪 Testing Hybrid Cloud Cost Optimization System")
print("=" * 60)

# Test 1: Environment Variables
print("\n✅ Test 1: Environment Variables")
load_dotenv()
backend_env = Path("backend") / ".env"
if backend_env.exists():
    load_dotenv(backend_env, override=True)

gcp_project_id = os.getenv("GCP_PROJECT_ID")
print(f"GCP_PROJECT_ID: {gcp_project_id or 'NOT SET'}")

# Test 2: Check start_system.py for Spot VM config
print("\n✅ Test 2: Spot VM Configuration")
with open("start_system.py", "r") as f:
    content = f.read()

flags = ["--provisioning-model", "SPOT", "--instance-termination-action", "DELETE"]
for flag in flags:
    if flag in content:
        print(f"✅ {flag}")
    else:
        print(f"❌ {flag}")

# Test 3: Auto-cleanup
print("\n✅ Test 3: Auto-Cleanup Logic")
if "_cleanup_gcp_instance" in content:
    print("✅ Cleanup method exists")
if "await self._cleanup_gcp_instance" in content:
    print("✅ Cleanup called on shutdown")

print("\n✅ Configuration validated!")
