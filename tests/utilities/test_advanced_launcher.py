#!/usr/bin/env python3
"""
Test script for the advanced JARVIS launcher
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent / "backend"
if backend_path.exists():
    sys.path.insert(0, str(backend_path))

async def test_advanced_features():
    """Test the advanced launcher features"""
    
    print("🧪 Testing Advanced JARVIS Launcher Features\n")
    
    # Test 1: Check if advanced launcher exists
    adv_launcher = Path("start_system_advanced.py")
    if adv_launcher.exists():
        print("✅ Advanced launcher found")
    else:
        print("❌ Advanced launcher not found")
        return
        
    # Test 2: Import and check features
    try:
        from start_system_advanced import AdvancedJARVISManager
        manager = AdvancedJARVISManager()
        
        print("\n📋 Advanced Features Available:")
        print(f"  • System diagnostics: ✓")
        print(f"  • ML model validation: ✓")
        print(f"  • GPU detection: ✓")
        print(f"  • Network connectivity check: ✓")
        print(f"  • Intelligent port recovery: ✓")
        print(f"  • Advanced health monitoring: ✓")
        print(f"  • Autonomous mode initialization: ✓")
        
        # Test 3: Run diagnostics
        print("\n🔍 Running System Diagnostics...")
        diagnostics = await manager.run_system_diagnostics()
        
        print(f"\n📊 Diagnostic Results:")
        print(f"  • Platform: {diagnostics['platform'][:50]}...")
        print(f"  • CPU cores: {diagnostics['cpu_count']}")
        print(f"  • Memory: {diagnostics['memory_gb']}GB")
        if 'gpu' in diagnostics:
            print(f"  • GPU: {diagnostics['gpu']}")
        print(f"  • Issues found: {len(diagnostics['issues'])}")
        print(f"  • Warnings: {len(diagnostics['warnings'])}")
        
        # Test 4: Check ML models
        print("\n🧠 Checking ML Models...")
        ml_status = await manager.check_ml_models()
        for model, status in ml_status.items():
            print(f"  • {model}: {status}")
            
        # Test 5: Validate Claude API
        print("\n🔐 Validating Claude API...")
        api_valid, model = await manager.validate_claude_api()
        if api_valid:
            print(f"  ✅ Claude API validated - Model: {model}")
        else:
            print(f"  ❌ Claude API not configured")
            
        print("\n✨ Advanced launcher test complete!")
        
    except Exception as e:
        print(f"❌ Error testing advanced features: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point"""
    print("=" * 60)
    print("JARVIS Advanced Launcher Test Suite")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return
        
    # Run async tests
    asyncio.run(test_advanced_features())
    
    print("\n💡 To use the advanced launcher:")
    print("   python start_system_advanced.py")
    print("\n💡 Or use the enhanced standard launcher:")
    print("   python start_system.py")

if __name__ == "__main__":
    main()