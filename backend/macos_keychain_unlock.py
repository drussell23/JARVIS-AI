#!/usr/bin/env python3
"""
macOS Keychain Integration for Screen Unlock
Securely retrieves password from Keychain and performs actual screen unlock

NOTE: This is a lightweight fallback. For advanced features use:
- backend/voice_unlock/intelligent_voice_unlock_service.py (primary)
- backend/voice_unlock/objc/server/screen_lock_detector.py (detection)
"""

import asyncio
import logging
import os
import subprocess
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MacOSKeychainUnlock:
    """Secure screen unlock using macOS Keychain"""

    def __init__(self):
        # Use the actual keychain service that has the password stored
        # MUST match what's stored by enable_screen_unlock.sh
        self.service_name = "com.jarvis.voiceunlock"
        self.account_name = "unlock_token"
        self.keychain_item_name = "JARVIS Voice Unlock"

    async def store_password_in_keychain(self, password: str) -> bool:
        """Store password securely in macOS Keychain (one-time setup)"""
        try:
            # Add password to keychain
            cmd = [
                "security",
                "add-generic-password",
                "-a",
                self.account_name,
                "-s",
                self.service_name,
                "-w",
                password,
                "-T",
                "/usr/bin/security",  # Allow security tool access
                "-U",  # Update if exists
                "-l",
                self.keychain_item_name,
            ]

            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                logger.info(f"✅ Password stored in Keychain as '{self.keychain_item_name}'")
                return True
            else:
                logger.error(f"Failed to store password: {stderr.decode()}")
                return False

        except Exception as e:
            logger.error(f"Keychain storage error: {e}")
            return False

    async def get_password_from_keychain(self) -> Optional[str]:
        """Retrieve password from macOS Keychain"""
        try:
            cmd = [
                "security",
                "find-generic-password",
                "-a",
                self.account_name,
                "-s",
                self.service_name,
                "-w",  # Output password only
            ]

            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                password = stdout.decode().strip()
                logger.info("✅ Retrieved password from Keychain")
                return password
            else:
                logger.error(f"Password not found in Keychain: {stderr.decode()}")
                return None

        except Exception as e:
            logger.error(f"Keychain retrieval error: {e}")
            return None

    async def check_screen_locked(self) -> bool:
        """Check if screen is currently locked using voice_unlock detector"""
        try:
            # Use the advanced screen lock detector from voice_unlock
            from voice_unlock.objc.server.screen_lock_detector import is_screen_locked

            return is_screen_locked()

        except ImportError:
            # Fallback to AppleScript if detector not available
            logger.debug("Using fallback AppleScript for screen detection")
            script = """
            tell application "System Events"
                set isLocked to false

                -- Check for screensaver
                if (exists process "ScreenSaverEngine") then
                    set isLocked to true
                end if

                -- Check if at login window
                try
                    set frontApp to name of first application process whose frontmost is true
                    if frontApp is "loginwindow" then
                        set isLocked to true
                    end if
                end try

                return isLocked
            end tell
            """

            try:
                result = await asyncio.create_subprocess_exec(
                    "osascript", "-e", script, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )

                stdout, _ = await result.communicate()
                return stdout.decode().strip() == "true"

            except Exception as e:
                logger.error(f"Failed to check screen status: {e}")
                return False

    async def unlock_screen(self, verified_speaker: Optional[str] = None) -> Dict[str, Any]:
        """Perform actual screen unlock using Keychain password"""

        # Check if screen is locked
        is_locked = await self.check_screen_locked()

        if not is_locked:
            return {
                "success": True,
                "message": f"Screen already unlocked{f' for {verified_speaker}' if verified_speaker else ''}",
                "action": "none_needed",
            }

        logger.info(
            f"🔐 Screen locked, attempting unlock{f' for {verified_speaker}' if verified_speaker else ''}..."
        )

        # Get password from Keychain
        password = await self.get_password_from_keychain()

        if not password:
            return {
                "success": False,
                "message": "Password not found in Keychain. Run setup first.",
                "action": "setup_required",
            }

        try:
            # Use advanced secure password typer (Core Graphics, no AppleScript)
            from voice_unlock.secure_password_typer import (
                get_secure_typer,
                TypingConfig
            )

            logger.info("🔐 Using advanced secure password typer (Core Graphics)")

            # Get typer instance
            typer = get_secure_typer()

            # Configure for unlock (adaptive timing, auto-submit)
            config = TypingConfig(
                wake_screen=True,
                submit_after_typing=True,
                randomize_timing=True,
                adaptive_timing=True,
                detect_system_load=True,
                clear_memory_after=True,
                enable_applescript_fallback=True,
                max_retries=3
            )

            # Type password securely and get metrics
            success, metrics = await typer.type_password_secure(
                password=password,
                submit=True,
                config_override=config
            )

            # Log metrics
            logger.info(
                f"🔐 [METRICS] Typing: {metrics.typing_time_ms:.0f}ms, "
                f"Wake: {metrics.wake_time_ms:.0f}ms, "
                f"Submit: {metrics.submit_time_ms:.0f}ms, "
                f"Total: {metrics.total_duration_ms:.0f}ms, "
                f"Retries: {metrics.retries}, "
                f"Fallback: {metrics.fallback_used}"
            )

            if not success:
                logger.warning(f"⚠️ Secure typer failed: {metrics.error_message}")

                # Fallback to AppleScript if Core Graphics fails
                wake_script = """
                tell application "System Events"
                    key code 49  -- Space key to wake
                end tell
                """

                await asyncio.create_subprocess_exec(
                    "osascript", "-e", wake_script, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )

                await asyncio.sleep(0.5)

                # Secure AppleScript: Use stdin to avoid password in process list
                type_script = """
                tell application "System Events"
                    keystroke (system attribute "JARVIS_UNLOCK_PASS")
                    delay 0.1
                    key code 36
                end tell
                """

                # Set password in environment temporarily
                env = os.environ.copy()
                env["JARVIS_UNLOCK_PASS"] = password

                result = await asyncio.create_subprocess_exec(
                    "osascript", "-e", type_script,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env
                )

                await result.communicate()

                # Clear from environment
                if "JARVIS_UNLOCK_PASS" in env:
                    del env["JARVIS_UNLOCK_PASS"]

            # Wait for unlock
            await asyncio.sleep(1.5)

            # Verify unlock succeeded
            still_locked = await self.check_screen_locked()

            if not still_locked:
                logger.info(
                    f"✅ Screen unlocked successfully{f' for {verified_speaker}' if verified_speaker else ''}"
                )
                return {
                    "success": True,
                    "message": f"Screen unlocked{f' for {verified_speaker}' if verified_speaker else ''}",
                    "action": "unlocked",
                    "verified_speaker": verified_speaker,
                }
            else:
                logger.warning("⚠️ Screen still locked after unlock attempt")
                return {
                    "success": False,
                    "message": "Unlock failed - check password",
                    "action": "failed",
                }

        except Exception as e:
            logger.error(f"Unlock error: {e}")
            return {"success": False, "message": f"Unlock error: {str(e)}", "action": "error"}

    async def setup_keychain_password(self):
        """Interactive setup to store password in Keychain"""
        import getpass

        print("\n" + "=" * 60)
        print("🔐 JARVIS SCREEN UNLOCK SETUP")
        print("=" * 60)
        print("\nThis will securely store your login password in macOS Keychain")
        print("so JARVIS can unlock your screen when you're verified by voice.\n")

        username = input(f"macOS username [{self.account_name}]: ").strip() or self.account_name
        self.account_name = username

        password = getpass.getpass("Enter your macOS login password: ")

        if password:
            success = await self.store_password_in_keychain(password)

            if success:
                print("\n✅ Success! Password stored in Keychain")
                print(f"  - Service: {self.service_name}")
                print(f"  - Account: {self.account_name}")
                print("  - JARVIS can now unlock your screen\n")

                # Test retrieval
                test_pwd = await self.get_password_from_keychain()
                if test_pwd:
                    print("✅ Verified: Password retrieval working")
                else:
                    print("⚠️ Warning: Could not verify password retrieval")

                return True
            else:
                print("\n❌ Failed to store password in Keychain")
                return False
        else:
            print("\n❌ No password entered")
            return False


async def main():
    """Set up or test Keychain unlock"""
    logging.basicConfig(level=logging.INFO)

    unlock_service = MacOSKeychainUnlock()

    # Check if password is already stored
    password = await unlock_service.get_password_from_keychain()

    if not password:
        print("\n⚠️ No password found in Keychain")
        setup = input("Would you like to set it up now? (y/n): ").lower()

        if setup == "y":
            await unlock_service.setup_keychain_password()
        else:
            print("Setup cancelled")
            return

    # Test unlock
    print("\n🧪 Testing screen unlock...")
    result = await unlock_service.unlock_screen(verified_speaker="Derek")
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
