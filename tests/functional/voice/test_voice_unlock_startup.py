#!/usr/bin/env python3
"""
Test Voice Unlock Startup Integration
====================================

Verifies that voice unlock is properly integrated with JARVIS.
"""

import asyncio
import sys
from pathlib import Path

# Add to path
sys.path.append(str(Path(__file__).parent))

from backend import get_backend_status, VOICE_UNLOCK_AVAILABLE
from backend.voice_unlock import check_dependencies, get_voice_unlock_status
from backend.voice_unlock.jarvis_integration import check_voice_unlock_requirements, setup_voice_unlock


async def test_voice_unlock_startup():
    """Test voice unlock startup integration"""
    print("🧪 Testing Voice Unlock Integration with JARVIS")
    print("=" * 60)
    
    # Check backend status
    print("\n📊 Backend Status:")
    status = get_backend_status()
    print(f"  Version: {status['version']}")
    print(f"  Voice Unlock Available: {status['modules']['voice_unlock']}")
    print(f"  Vision Available: {status['modules']['vision']}")
    print(f"  Cleanup Available: {status['modules']['cleanup']}")
    
    if not VOICE_UNLOCK_AVAILABLE:
        print("\n❌ Voice Unlock module not available!")
        return
        
    # Check dependencies
    print("\n📦 Voice Unlock Dependencies:")
    deps = check_dependencies()
    all_deps_ok = True
    for dep, available in deps.items():
        icon = "✅" if available else "❌"
        print(f"  {icon} {dep}")
        if not available:
            all_deps_ok = False
            
    # Check requirements
    print("\n🔍 System Requirements:")
    reqs = check_voice_unlock_requirements()
    for req, available in reqs.items():
        icon = "✅" if available else "❌"
        print(f"  {icon} {req}")
        
    # Try to setup voice unlock
    print("\n🚀 Setting up Voice Unlock...")
    result = setup_voice_unlock()
    
    print(f"\nSetup Result:")
    print(f"  Enabled: {result['enabled']}")
    if result['reason']:
        print(f"  Reason: {result['reason']}")
    print(f"  Apple Watch Support: {result['apple_watch']}")
    
    # Get voice unlock status
    print("\n📈 Voice Unlock Status:")
    vu_status = get_voice_unlock_status()
    
    if 'available' in vu_status:
        print(f"  Available: {vu_status['available']}")
        if 'error' in vu_status:
            print(f"  Error: {vu_status['error']}")
    else:
        # New system status
        print(f"  Active: {vu_status.get('is_active', False)}")
        print(f"  Locked: {vu_status.get('is_locked', True)}")
        print(f"  Current User: {vu_status.get('current_user', 'None')}")
        
        if 'ml_status' in vu_status:
            ml = vu_status['ml_status']
            print(f"\n  ML Status:")
            print(f"    Memory: {ml.get('memory_percent', 0):.1f}%")
            print(f"    ML Memory: {ml.get('ml_memory_mb', 0):.1f}MB")
            print(f"    Cache Size: {ml.get('cache_size_mb', 0):.1f}MB")
            
    # Check configuration
    print("\n⚙️  Configuration Test:")
    try:
        from backend.voice_unlock.config import get_config
        config = get_config()
        sys_info = config.get_system_info()
        
        print(f"  System RAM: {sys_info['ram_gb']}GB")
        print(f"  Available RAM: {sys_info['ram_available_gb']}GB")
        print(f"  Optimizations: {', '.join(sys_info['optimizations_applied']) or 'None'}")
        
        print(f"\n  Memory Settings:")
        print(f"    Max Memory: {config.performance.max_memory_mb}MB")
        print(f"    Cache Size: {config.performance.cache_size_mb}MB")
        print(f"    Quantization: {config.performance.enable_quantization}")
        
    except Exception as e:
        print(f"  ❌ Config error: {e}")
        
    print("\n" + "=" * 60)
    
    if all_deps_ok and result['enabled']:
        print("✅ Voice Unlock is properly integrated and ready!")
        print("\nNext steps:")
        print("1. Run: ./install_voice_unlock_deps.sh")
        print("2. Enroll: jarvis-voice-unlock enroll <name>")
        print("3. Test: jarvis-voice-unlock test")
    else:
        print("⚠️  Voice Unlock needs configuration")
        if not all_deps_ok:
            print("\nRun: pip install -r backend/voice_unlock/requirements.txt")


if __name__ == "__main__":
    asyncio.run(test_voice_unlock_startup())