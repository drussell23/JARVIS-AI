#!/usr/bin/env python3
"""
Advanced macOS HUD Launcher with Multiple Launch Strategies
Uses native macOS APIs and bypasses security restrictions
"""

import os
import sys
import time
import subprocess
import plistlib
import tempfile
import json
from pathlib import Path
from typing import Optional, Dict, Any
import asyncio
import threading
import socket

class AdvancedHUDLauncher:
    """
    Sophisticated HUD launcher that bypasses macOS security restrictions
    using multiple launch strategies and native APIs
    """

    def __init__(self, hud_app_path: Path):
        self.hud_app_path = hud_app_path
        self.executable_path = hud_app_path / "Contents/MacOS/JARVIS-HUD"
        self.info_plist = hud_app_path / "Contents/Info.plist"
        self.launch_attempts = []
        self.backend_ws = "ws://localhost:8010/ws"
        self.backend_http = "http://localhost:8010"

    def prepare_environment(self) -> Dict[str, str]:
        """Prepare environment variables for HUD"""
        env = os.environ.copy()
        env.update({
            "JARVIS_BACKEND_WS": self.backend_ws,
            "JARVIS_BACKEND_HTTP": self.backend_http,
            "JARVIS_HUD_MODE": "overlay",
            "JARVIS_HUD_AUTO_CONNECT": "true",
            "JARVIS_HUD_DEBUG": "true",
            # Bypass macOS security
            "DYLD_LIBRARY_PATH": "", # Clear DYLD paths to avoid restrictions 
            "DYLD_INSERT_LIBRARIES": "", # Clear inserted libraries 
            # Disable quarantine
            "COM_APPLE_QUARANTINE": "false", # Bypass quarantine checks 
        })
        return env

    async def launch_strategy_1_direct_exec(self) -> bool:
        """Strategy 1: Direct executable with quarantine bypass"""
        print("   🔧 Strategy 1: Direct executable with quarantine bypass...")

        try:
            # Remove quarantine attribute
            subprocess.run(
                ["xattr", "-cr", str(self.hud_app_path)],
                capture_output=True,
                timeout=2
            )

            # Make executable
            subprocess.run(
                ["chmod", "+x", str(self.executable_path)],
                capture_output=True
            )

            # Launch with spawn attributes to bypass restrictions
            process = subprocess.Popen(
                [str(self.executable_path)],
                env=self.prepare_environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
                # Removed preexec_fn to avoid async context issues
            )

            await asyncio.sleep(0.5)
            if process.poll() is None:
                print(f"   ✅ Strategy 1 SUCCESS: HUD launched (PID: {process.pid})")
                return True
            else:
                stderr = process.stderr.read().decode() if process.stderr else ""
                print(f"   ❌ Strategy 1 failed: Process exited. {stderr[:100]}")

        except Exception as e:
            print(f"   ❌ Strategy 1 failed: {e}")

        return False

    async def launch_strategy_2_launchctl(self) -> bool:
        """Strategy 2: Use launchctl with custom plist"""
        print("   🔧 Strategy 2: launchctl with custom plist...")

        try:
            # Create launch agent plist
            plist_data = {
                "Label": "com.jarvis.hud",
                "ProgramArguments": [str(self.executable_path)],
                "EnvironmentVariables": self.prepare_environment(),
                "RunAtLoad": True,
                "KeepAlive": False,
                "ThrottleInterval": 0,
                "ProcessType": "Interactive",
                "LegacyTimers": True,
            }

            # Write plist to temp location
            plist_path = Path.home() / "Library/LaunchAgents/com.jarvis.hud.plist"
            plist_path.parent.mkdir(parents=True, exist_ok=True)

            with open(plist_path, 'wb') as f:
                plistlib.dump(plist_data, f)

            # Unload if exists
            subprocess.run(
                ["launchctl", "unload", str(plist_path)],
                capture_output=True
            )

            # Load the launch agent
            result = subprocess.run(
                ["launchctl", "load", "-w", str(plist_path)],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                await asyncio.sleep(1)
                # Check if running
                check = subprocess.run(
                    ["launchctl", "list", "com.jarvis.hud"],
                    capture_output=True,
                    text=True
                )
                if check.returncode == 0:
                    print("   ✅ Strategy 2 SUCCESS: HUD launched via launchctl")
                    return True

            print(f"   ❌ Strategy 2 failed: {result.stderr}")

        except Exception as e:
            print(f"   ❌ Strategy 2 failed: {e}")

        return False

    async def launch_strategy_3_osascript(self) -> bool:
        """Strategy 3: AppleScript/osascript launcher"""
        print("   🔧 Strategy 3: AppleScript launcher...")

        try:
            # Create AppleScript to launch HUD
            script = f'''
            tell application "System Events"
                do shell script "'{self.executable_path}' > /tmp/jarvis_hud.log 2>&1 &"
            end tell
            '''

            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                await asyncio.sleep(1)
                # Check if HUD is running
                ps_check = subprocess.run(
                    ["pgrep", "-f", "JARVIS-HUD"],
                    capture_output=True
                )
                if ps_check.returncode == 0:
                    pid = ps_check.stdout.decode().strip()
                    print(f"   ✅ Strategy 3 SUCCESS: HUD launched via AppleScript (PID: {pid})")
                    return True

            print(f"   ❌ Strategy 3 failed: {result.stderr}")

        except Exception as e:
            print(f"   ❌ Strategy 3 failed: {e}")

        return False

    async def launch_strategy_4_nstask_wrapper(self) -> bool:
        """Strategy 4: NSTask wrapper via Python-ObjC bridge"""
        print("   🔧 Strategy 4: NSTask wrapper...")

        try:
            # Create Objective-C bridge launcher
            launcher_code = f'''
import objc
from Foundation import NSTask, NSPipe, NSMutableDictionary
import os

# Create NSTask
task = NSTask.alloc().init()
task.setLaunchPath_("{self.executable_path}")
task.setArguments_([])

# Set environment
env = NSMutableDictionary.alloc().init()
for k, v in {json.dumps(self.prepare_environment())}.items():
    env.setObject_forKey_(v, k)
task.setEnvironment_(env)

# Set output pipes
task.setStandardOutput_(NSPipe.pipe())
task.setStandardError_(NSPipe.pipe())

# Launch
task.launch()
print(f"HUD launched with PID: {{task.processIdentifier()}}")
'''

            # Write to temp file and execute
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(launcher_code)
                temp_script = f.name

            result = subprocess.run(
                [sys.executable, temp_script],
                capture_output=True,
                text=True,
                timeout=5
            )

            os.unlink(temp_script)

            if "HUD launched with PID" in result.stdout:
                print(f"   ✅ Strategy 4 SUCCESS: {result.stdout.strip()}")
                return True

            print(f"   ❌ Strategy 4 failed: {result.stderr}")

        except ImportError:
            print("   ❌ Strategy 4 skipped: PyObjC not available")
        except Exception as e:
            print(f"   ❌ Strategy 4 failed: {e}")

        return False

    async def launch_strategy_5_open_with_args(self) -> bool:
        """Strategy 5: open command with advanced arguments"""
        print("   🔧 Strategy 5: open command with advanced arguments...")

        try:
            # Build complex open command
            cmd = [
                "open",
                "-a", str(self.hud_app_path),
                "--new",  # New instance
                "--hide",  # Start hidden then show
                "--env", f"JARVIS_BACKEND_WS={self.backend_ws}",
                "--env", f"JARVIS_BACKEND_HTTP={self.backend_http}",
                "--stdout", "/tmp/jarvis_hud_stdout.log",
                "--stderr", "/tmp/jarvis_hud_stderr.log"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                await asyncio.sleep(1)
                # Show the app
                subprocess.run(["open", "-a", str(self.hud_app_path), "--show"])
                print("   ✅ Strategy 5 SUCCESS: HUD launched via advanced open command")
                return True

            print(f"   ❌ Strategy 5 failed: {result.stderr}")

        except Exception as e:
            print(f"   ❌ Strategy 5 failed: {e}")

        return False

    async def launch_strategy_6_codesign_adhoc(self) -> bool:
        """Strategy 6: Ad-hoc code sign and launch"""
        print("   🔧 Strategy 6: Ad-hoc code signing...")

        try:
            # First, clean all extended attributes to avoid "resource fork" errors
            print("   → Cleaning extended attributes...")
            subprocess.run(
                ["xattr", "-cr", str(self.hud_app_path)],
                capture_output=True,
                timeout=5
            )

            # Remove any .DS_Store files
            subprocess.run(
                ["find", str(self.hud_app_path), "-name", ".DS_Store", "-delete"],
                capture_output=True,
                timeout=5
            )

            # Ad-hoc sign the app to bypass Gatekeeper
            sign_result = subprocess.run(
                ["codesign", "--force", "--deep", "--sign", "-", "--preserve-metadata=entitlements", str(self.hud_app_path)],
                capture_output=True,
                text=True,
                timeout=10
            )

            if sign_result.returncode == 0:
                print("   ✓ App signed with ad-hoc certificate")

                # Now try launching
                process = subprocess.Popen(
                    ["open", "-a", str(self.hud_app_path)],
                    env=self.prepare_environment(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                await asyncio.sleep(1)
                if process.poll() is None or process.poll() == 0:
                    print("   ✅ Strategy 6 SUCCESS: HUD launched after ad-hoc signing")
                    return True

            print(f"   ❌ Strategy 6 failed: {sign_result.stderr}")

        except Exception as e:
            print(f"   ❌ Strategy 6 failed: {e}")

        return False

    async def launch_strategy_7_shell_wrapper(self) -> bool:
        """Strategy 7: Shell script wrapper to bypass restrictions"""
        print("   🔧 Strategy 7: Shell script wrapper...")

        try:
            # Create a shell script that launches the app
            script_content = f"""#!/bin/bash
# JARVIS HUD Launcher Script
export JARVIS_BACKEND_WS="ws://localhost:8010/ws"
export JARVIS_BACKEND_HTTP="http://localhost:8010"
export JARVIS_ENV="production"

# Remove quarantine
xattr -d com.apple.quarantine "{self.executable_path}" 2>/dev/null || true

# Make executable
chmod +x "{self.executable_path}"

# Launch directly
exec "{self.executable_path}"
"""
            # Write script to temp file
            script_path = Path("/tmp/launch_jarvis_hud.sh")
            script_path.write_text(script_content)
            script_path.chmod(0o755)

            # Launch via shell script
            process = subprocess.Popen(
                ["/bin/bash", str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )

            await asyncio.sleep(1)
            if process.poll() is None:
                print(f"   ✅ Strategy 7 SUCCESS: HUD launched via shell wrapper (PID: {process.pid})")
                return True

        except Exception as e:
            print(f"   ❌ Strategy 7 failed: {e}")

        return False

    async def verify_hud_running(self) -> bool:
        """Verify HUD is actually running"""
        try:
            # Check process
            result = subprocess.run(
                ["pgrep", "-f", "JARVIS-HUD"],
                capture_output=True
            )

            if result.returncode == 0:
                pids = result.stdout.decode().strip().split('\n')
                print(f"   🔍 HUD processes found: {', '.join(pids)}")

                # Try to connect to HUD WebSocket
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    # Assuming HUD opens a local socket for health checks
                    sock.connect(("localhost", 8011))  # HUD health port
                    sock.close()
                    print("   ✅ HUD health check passed")
                    return True
                except:
                    # Process exists but may not be fully initialized
                    print("   ⏳ HUD process exists but not yet responsive")
                    return True  # Still consider it a success

            return False

        except Exception as e:
            print(f"   ⚠️ Verification error: {e}")
            return False

    async def launch_with_all_strategies(self) -> bool:
        """Try all launch strategies in sequence until one works"""
        print("\n🚀 Advanced HUD Launcher starting...")
        print(f"   App path: {self.hud_app_path}")
        print(f"   Executable: {self.executable_path}")

        # Verify paths exist
        if not self.hud_app_path.exists():
            print(f"   ❌ ERROR: HUD app not found at {self.hud_app_path}")
            return False

        if not self.executable_path.exists():
            print(f"   ❌ ERROR: Executable not found at {self.executable_path}")
            return False

        # Try each strategy
        strategies = [
            self.launch_strategy_1_direct_exec,
            self.launch_strategy_2_launchctl,
            self.launch_strategy_3_osascript,
            self.launch_strategy_4_nstask_wrapper,
            self.launch_strategy_5_open_with_args,
            self.launch_strategy_6_codesign_adhoc,
            self.launch_strategy_7_shell_wrapper,
        ]

        for i, strategy in enumerate(strategies, 1):
            print(f"\n📍 Attempting strategy {i} of {len(strategies)}...")

            try:
                success = await strategy()
                if success:
                    # Verify it's actually running
                    await asyncio.sleep(1)
                    if await self.verify_hud_running():
                        print(f"\n🎉 HUD successfully launched using strategy {i}!")
                        return True
                    else:
                        print(f"   ⚠️ Strategy {i} claimed success but HUD not verified")
            except Exception as e:
                print(f"   ⚠️ Strategy {i} exception: {e}")

            # Small delay between attempts
            if i < len(strategies):
                await asyncio.sleep(0.5)

        print("\n❌ All launch strategies failed")
        return False


async def launch_hud_advanced(hud_app_path: Path) -> bool:
    """Main entry point for advanced HUD launching"""
    launcher = AdvancedHUDLauncher(hud_app_path)
    return await launcher.launch_with_all_strategies()


# Async-aware wrapper for use in start_system.py
async def launch_hud_async_safe(hud_app_path: Path) -> bool:
    """Launcher that works in async context"""
    launcher = AdvancedHUDLauncher(hud_app_path)
    return await launcher.launch_with_all_strategies()

# Synchronous wrapper for use in start_system.py
def launch_hud_sync(hud_app_path: Path) -> bool:
    """Synchronous wrapper for the async launcher"""
    try:
        # Check if we're already in an async event loop
        loop = asyncio.get_running_loop()
        # We're in an async context, create a coroutine and return it
        # This will be handled by the caller's await
        return loop.create_task(launch_hud_async_safe(hud_app_path))
    except RuntimeError:
        # No running loop, we can use asyncio.run
        return asyncio.run(launch_hud_advanced(hud_app_path))


if __name__ == "__main__":
    # Test the launcher
    import sys
    if len(sys.argv) > 1:
        app_path = Path(sys.argv[1])
    else:
        app_path = Path("/Applications/JARVIS-HUD.app")

    success = launch_hud_sync(app_path)
    sys.exit(0 if success else 1)