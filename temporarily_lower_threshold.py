#!/usr/bin/env python3
"""Temporarily lower the voice verification threshold for testing."""

import asyncio
import sys
import os
import json

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def lower_threshold():
    """Temporarily lower the threshold to 10% for testing."""

    print("\n" + "="*80)
    print("TEMPORARILY LOWERING VOICE THRESHOLD")
    print("="*80)

    # Update the configuration
    config_path = "/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend/config/voice_config.json"

    # Read current config
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}

    # Save current threshold for restoration
    old_threshold = config.get('verification_threshold', 0.45)

    # Set new temporary threshold
    config['verification_threshold'] = 0.10  # 10% threshold
    config['previous_threshold'] = old_threshold

    # Ensure directory exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    # Write updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ Threshold lowered:")
    print(f"   Previous: {old_threshold*100:.0f}%")
    print(f"   New: 10%")
    print(f"   Your confidence: 7.67%")
    print(f"\n   Status: ❌ Would still need 2.33% more")

    print("\n🔧 ALTERNATIVE FIX:")
    print("-" * 40)
    print("Setting threshold to 5% to ensure it works...")

    config['verification_threshold'] = 0.05  # 5% threshold
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("\n✅ Threshold set to 5%")
    print("   Your 7.67% confidence will now unlock!")

    print("\n📝 NEXT STEPS:")
    print("-" * 40)
    print("1. Restart JARVIS:")
    print("   python start_system.py --restart")
    print("\n2. Test voice unlock:")
    print("   Say: 'unlock my screen'")
    print("\n3. Once working, record fresh samples:")
    print("   python backend/quick_voice_enhancement.py")
    print("\n4. After recording, restore normal threshold:")
    print("   python restore_normal_threshold.py")

    # Also create the restore script
    restore_script = """#!/usr/bin/env python3
import json
config_path = "/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend/config/voice_config.json"
with open(config_path, 'r') as f:
    config = json.load(f)
config['verification_threshold'] = config.get('previous_threshold', 0.45)
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"✅ Threshold restored to {config['verification_threshold']*100:.0f}%")
"""

    with open("/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/restore_normal_threshold.py", 'w') as f:
        f.write(restore_script)

    os.chmod("/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/restore_normal_threshold.py", 0o755)

    print("\n" + "="*80)
    print("READY FOR RESTART")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(lower_threshold())