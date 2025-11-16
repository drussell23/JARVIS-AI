#!/usr/bin/env python3
"""
macOS HUD Launcher for Signed JARVIS-HUD Application
Simple launcher for Developer ID signed app - no complex workarounds needed!
"""

import subprocess
from pathlib import Path
from typing import Optional
import asyncio


class SignedHUDLauncher:
    """
    Simple launcher for properly signed macOS HUD application
    Uses the standard 'open' command since app is Developer ID signed
    """

    def __init__(self, hud_app_path: Path):
        self.hud_app_path = hud_app_path
        self.executable_path = hud_app_path / "Contents/MacOS/JARVIS-HUD"

    async def launch(self) -> bool:
        """Launch the signed HUD application using macOS 'open' command"""
        print("\n🚀 Launching signed JARVIS-HUD...")
        print(f"   App: {self.hud_app_path}")
        print(f"   Signed with: Developer ID Application")

        # Verify app exists
        if not self.hud_app_path.exists():
            print(f"   ❌ ERROR: HUD app not found at {self.hud_app_path}")
            return False

        if not self.executable_path.exists():
            print(f"   ❌ ERROR: Executable not found at {self.executable_path}")
            return False

        try:
            # Launch using macOS 'open' command (works for signed apps)
            # The app's Info.plist and code signature handle everything
            result = subprocess.run(
                ["open", "-a", str(self.hud_app_path)],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                print(f"   ✅ HUD launched successfully!")
                print(f"   → App will connect to ws://localhost:8010/ws automatically")

                # Give app a moment to start
                await asyncio.sleep(1)

                # Verify it's running
                verify = subprocess.run(
                    ["pgrep", "-f", "JARVIS-HUD"],
                    capture_output=True
                )

                if verify.returncode == 0:
                    pid = verify.stdout.decode().strip()
                    print(f"   ✓ HUD process confirmed (PID: {pid})")
                    return True
                else:
                    print(f"   ⚠️ HUD may be starting (process not yet visible)")
                    return True  # Still consider success - app may be initializing
            else:
                print(f"   ❌ Launch failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print(f"   ❌ Launch timed out")
            return False
        except Exception as e:
            print(f"   ❌ Launch error: {e}")
            return False


# Async entry point for use in start_system.py
async def launch_hud_async_safe(hud_app_path: Path) -> bool:
    """Async launcher that works in async context"""
    launcher = SignedHUDLauncher(hud_app_path)
    return await launcher.launch()


# Synchronous wrapper
def launch_hud_sync(hud_app_path: Path) -> bool:
    """Synchronous wrapper for the async launcher"""
    try:
        # Check if we're already in an async event loop
        loop = asyncio.get_running_loop()
        # We're in an async context, create a task
        return loop.create_task(launch_hud_async_safe(hud_app_path))
    except RuntimeError:
        # No running loop, use asyncio.run
        return asyncio.run(launch_hud_async_safe(hud_app_path))


if __name__ == "__main__":
    # Test the launcher
    import sys
    if len(sys.argv) > 1:
        app_path = Path(sys.argv[1])
    else:
        # Default to Release build
        app_path = Path(__file__).parent / "macos-hud/build/Build/Products/Release/JARVIS-HUD.app"

    print(f"Testing HUD launcher with: {app_path}")
    success = launch_hud_sync(app_path)
    sys.exit(0 if success else 1)
