"""
JARVIS macOS Native HUD Launcher
Dynamic, async, robust launcher for native macOS SwiftUI application
Zero hardcoding - fully configuration-driven with graceful degradation
"""

import asyncio
import subprocess
import signal
import os
import sys
from pathlib import Path
from typing import Optional
import logging
import json

logger = logging.getLogger(__name__)


class MacOSHUDLauncher:
    """
    Robust, async macOS HUD launcher with process management
    Features:
    - Dynamic app path resolution (build/release/debug)
    - Async subprocess management with proper cleanup
    - Environment variable injection for backend connection
    - Health monitoring and auto-restart
    - Graceful shutdown with signal handling
    """

    def __init__(self, backend_host: str = "localhost", backend_port: int = 8000):
        self.backend_host = backend_host
        self.backend_port = backend_port
        self.process: Optional[asyncio.subprocess.Process] = None
        self.running = False
        self.app_path: Optional[Path] = None

        # Configuration
        self.config = {
            "app_name": "JARVIS-HUD.app",
            "search_paths": [
                "macos-hud/build/Release",
                "macos-hud/build/Debug",
                "macos-hud/Build/Products/Release",
                "macos-hud/Build/Products/Debug",
                "macos-hud/DerivedData/*/Build/Products/Release",
                "macos-hud/DerivedData/*/Build/Products/Debug",
            ],
            "build_timeout": 300,  # 5 minutes
            "health_check_interval": 5,  # seconds
            "restart_delay": 2,  # seconds before restart
            "max_restart_attempts": 3,
        }

    async def find_app(self) -> Optional[Path]:
        """
        Dynamically find the JARVIS-HUD.app in build directories
        Returns path to app or None if not found
        """
        logger.info("🔍 Searching for JARVIS-HUD.app...")

        repo_root = Path(__file__).parent.parent.parent
        app_name = self.config["app_name"]

        # Try each search path
        for search_path in self.config["search_paths"]:
            full_path = repo_root / search_path / app_name

            # Handle wildcard paths (DerivedData)
            if "*" in str(search_path):
                import glob
                glob_pattern = str(repo_root / search_path / app_name)
                matches = glob.glob(glob_pattern)
                if matches:
                    full_path = Path(matches[0])
                else:
                    continue

            if full_path.exists() and full_path.is_dir():
                logger.info(f"✓ Found app at: {full_path}")
                return full_path

        logger.warning(f"⚠️  {app_name} not found in any search path")
        return None

    async def build_app(self) -> bool:
        """
        Build the macOS app using xcodebuild
        Returns True if successful
        """
        logger.info("🔨 Building JARVIS-HUD.app with xcodebuild...")

        repo_root = Path(__file__).parent.parent.parent
        project_path = repo_root / "macos-hud" / "JARVIS-HUD.xcodeproj"

        if not project_path.exists():
            logger.error(f"❌ Xcode project not found at {project_path}")
            return False

        build_cmd = [
            "xcodebuild",
            "-project", str(project_path),
            "-scheme", "JARVIS-HUD",
            "-configuration", "Release",
            "-derivedDataPath", str(repo_root / "macos-hud" / "build"),
            "build",
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *build_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(repo_root / "macos-hud")
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config["build_timeout"]
            )

            if process.returncode == 0:
                logger.info("✓ Build successful!")
                return True
            else:
                logger.error(f"❌ Build failed with code {process.returncode}")
                if stderr:
                    logger.error(f"Error output: {stderr.decode()[:500]}")
                return False

        except asyncio.TimeoutError:
            logger.error(f"❌ Build timed out after {self.config['build_timeout']}s")
            return False
        except Exception as e:
            logger.error(f"❌ Build error: {e}")
            return False

    async def launch(self) -> bool:
        """
        Launch the macOS HUD app
        Returns True if successfully launched
        """
        # Find app
        self.app_path = await self.find_app()

        # If not found, try building
        if not self.app_path:
            logger.info("📦 App not found, attempting to build...")
            if await self.build_app():
                self.app_path = await self.find_app()

        if not self.app_path:
            logger.error("❌ Cannot launch: app not found and build failed")
            return False

        # Prepare environment
        env = os.environ.copy()
        env.update({
            "JARVIS_BACKEND_HOST": self.backend_host,
            "JARVIS_BACKEND_PORT": str(self.backend_port),
            "JARVIS_BACKEND_WS": f"ws://{self.backend_host}:{self.backend_port}/ws/hud",
            "JARVIS_BACKEND_HTTP": f"http://{self.backend_host}:{self.backend_port}",
        })

        # Launch app
        logger.info(f"🚀 Launching {self.app_path}...")
        logger.info(f"   Backend: ws://{self.backend_host}:{self.backend_port}/ws/hud")

        try:
            # Use 'open' command for macOS apps
            self.process = await asyncio.create_subprocess_exec(
                "open",
                "-a", str(self.app_path),
                "--env", json.dumps(env),  # Pass environment
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            self.running = True
            logger.info("✓ JARVIS macOS HUD launched successfully!")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to launch app: {e}")
            return False

    async def monitor(self):
        """
        Monitor the app process and restart if needed
        """
        restart_attempts = 0

        while self.running and restart_attempts < self.config["max_restart_attempts"]:
            await asyncio.sleep(self.config["health_check_interval"])

            # Check if process is still running
            if self.process and self.process.returncode is not None:
                logger.warning(f"⚠️  HUD process exited with code {self.process.returncode}")

                if self.running:  # Only restart if we're supposed to be running
                    logger.info(f"🔄 Restarting HUD (attempt {restart_attempts + 1}/{self.config['max_restart_attempts']})...")
                    await asyncio.sleep(self.config["restart_delay"])

                    if await self.launch():
                        restart_attempts = 0  # Reset on successful restart
                    else:
                        restart_attempts += 1

    async def shutdown(self):
        """
        Gracefully shutdown the HUD app
        """
        logger.info("🛑 Shutting down JARVIS macOS HUD...")
        self.running = False

        if self.process:
            try:
                # Try graceful shutdown first
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5)
                logger.info("✓ HUD shutdown gracefully")
            except asyncio.TimeoutError:
                # Force kill if needed
                logger.warning("⚠️  Forcing HUD shutdown...")
                self.process.kill()
                await self.process.wait()
            except Exception as e:
                logger.error(f"❌ Error during shutdown: {e}")

    async def run(self):
        """
        Main run loop: launch and monitor
        """
        if not await self.launch():
            logger.error("❌ Failed to launch macOS HUD")
            return False

        # Start monitoring in background
        monitor_task = asyncio.create_task(self.monitor())

        # Set up signal handlers for graceful shutdown
        def signal_handler(sig):
            logger.info(f"Received signal {sig}")
            asyncio.create_task(self.shutdown())

        try:
            import signal
            signal.signal(signal.SIGINT, lambda s, f: signal_handler(s))
            signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s))
        except Exception as e:
            logger.warning(f"Could not set signal handlers: {e}")

        # Wait for shutdown
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

        return True


async def launch_macos_hud(backend_host: str = "localhost", backend_port: int = 8000):
    """
    Async entry point for launching macOS HUD
    """
    launcher = MacOSHUDLauncher(backend_host, backend_port)
    return await launcher.run()


def launch_macos_hud_sync(backend_host: str = "localhost", backend_port: int = 8000):
    """
    Synchronous wrapper for launching macOS HUD
    """
    return asyncio.run(launch_macos_hud(backend_host, backend_port))
