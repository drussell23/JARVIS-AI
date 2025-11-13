#!/usr/bin/env python3
"""
Test Password Typing with Detailed Logging
==========================================
This script will test typing the password with character-by-character logging
"""

import asyncio
import logging
import sys

# Add backend to path
sys.path.insert(0, '/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_password_typing():
    """Test password typing with debug logging"""
    from voice_unlock.secure_password_typer import get_secure_typer, TypingConfig
    from macos_keychain_unlock import MacOSKeychainUnlock

    print("=" * 70)
    print("🔐 PASSWORD TYPING TEST")
    print("=" * 70)

    # Get password from keychain
    unlock_service = MacOSKeychainUnlock()
    password = await unlock_service.get_password_from_keychain()

    if not password:
        print("❌ Could not retrieve password from keychain")
        return

    print(f"\n✅ Retrieved password from keychain ({len(password)} characters)")
    print(f"   Password hint: {password[0]}{'*' * (len(password)-2)}{password[-1]}")

    # Analyze password characters
    print("\n📊 Password Character Analysis:")
    print("-" * 70)

    from voice_unlock.secure_password_typer import KEYCODE_MAP, SHIFT_CHARS

    for i, char in enumerate(password):
        keycode = KEYCODE_MAP.get(char)
        needs_shift = char in SHIFT_CHARS
        char_type = "letter" if char.isalpha() else "digit" if char.isdigit() else "special"

        if keycode:
            print(f"  {i+1:2d}. '{char}' ({char_type:7s}) -> keycode: 0x{keycode:02X} | shift: {needs_shift}")
        else:
            print(f"  {i+1:2d}. '{char}' ({char_type:7s}) -> ❌ NO KEYCODE MAPPING!")

    print("\n" + "=" * 70)
    print("⚠️  NOTE: This will type the password into whatever field has focus")
    print("   Make sure a text editor or test field is active, NOT the lock screen!")
    print("=" * 70)

    response = input("\nType 'yes' to proceed with typing test: ")
    if response.lower() != 'yes':
        print("Test cancelled")
        return

    print("\n🔄 Starting password typing test in 3 seconds...")
    print("   (Switch to a text editor now!)")
    await asyncio.sleep(3)

    # Get typer
    typer = get_secure_typer()

    # Create config with debug enabled
    config = TypingConfig(
        randomize_timing=False,  # Disable randomization for testing
        submit_after_typing=False,  # Don't press Enter
        wake_screen=False,  # Don't wake screen
        max_retries=1
    )

    print("\n🔐 Typing password...")
    success, metrics = await typer.type_password_secure(
        password=password,
        submit=False,
        config_override=config
    )

    print("\n" + "=" * 70)
    print("📈 RESULTS:")
    print("=" * 70)
    print(f"Success: {success}")
    print(f"Total time: {metrics.total_duration_ms:.1f}ms")
    print(f"Characters typed: {len(password)}")

    if success:
        print("\n✅ Password typed successfully!")
        print("   Check the text field to verify correctness")
    else:
        print(f"\n❌ Failed: {metrics.error_message}")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    asyncio.run(test_password_typing())
