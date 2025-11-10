#!/usr/bin/env python3
"""
Advanced Cloud SQL Proxy Manager
=================================

Dynamic, robust proxy lifecycle management with:
- Zero hardcoding - all config from database_config.json
- Auto-discovery of proxy binary location
- System-level persistence (launchd on macOS, systemd on Linux)
- Runtime health monitoring and auto-recovery
- Graceful degradation to SQLite fallback
- Port conflict resolution
- Multi-platform support
"""

import asyncio
import json
import logging
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class CloudSQLProxyManager:
    """
    Enterprise-grade Cloud SQL proxy lifecycle manager.

    Features:
    - Auto-discovers proxy binary and config
    - Manages proxy process lifecycle
    - Health monitoring with auto-recovery
    - System service integration (launchd/systemd)
    - Zero hardcoding - fully dynamic
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize proxy manager with dynamic configuration discovery.

        Args:
            config_path: Optional path to database_config.json
                        (auto-discovers from ~/.jarvis/gcp if not provided)
        """
        self.system = platform.system()
        self.config_path = config_path or self._discover_config_path()
        self.config = self._load_config()
        self.proxy_binary = self._discover_proxy_binary()
        # Use system temp directory for cross-platform compatibility
        temp_dir = Path(tempfile.gettempdir())
        self.log_path = temp_dir / "cloud-sql-proxy.log"
        self.pid_path = temp_dir / "cloud-sql-proxy.pid"
        self.service_name = "com.jarvis.cloudsql-proxy"
        self.process: Optional[subprocess.Popen] = None

    def _discover_config_path(self) -> Path:
        """
        Auto-discover database config file location.

        Searches in order:
        1. ~/.jarvis/gcp/database_config.json
        2. $JARVIS_HOME/gcp/database_config.json
        3. ./database_config.json
        """
        search_paths = [
            Path.home() / ".jarvis" / "gcp" / "database_config.json",
            Path(os.getenv("JARVIS_HOME", ".")) / "gcp" / "database_config.json",
            Path("database_config.json"),
        ]

        for path in search_paths:
            if path.exists():
                logger.info(f"📂 Found database config: {path}")
                return path

        raise FileNotFoundError(f"Could not find database_config.json in any of: {search_paths}")

    def _load_config(self) -> Dict:
        """Load and validate database configuration."""
        try:
            with open(self.config_path) as f:
                config = json.load(f)

            # Validate required fields
            required = ["cloud_sql", "project_id"]
            if not all(k in config for k in required):
                raise ValueError(f"Config missing required fields: {required}")

            cloud_sql = config["cloud_sql"]
            required_sql = ["connection_name", "port"]
            if not all(k in cloud_sql for k in required_sql):
                raise ValueError(f"cloud_sql config missing: {required_sql}")

            logger.info(
                f"✅ Loaded config: {cloud_sql['connection_name']} " f"on port {cloud_sql['port']}"
            )
            return config

        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            raise

    def _discover_proxy_binary(self) -> str:
        """
        Auto-discover cloud-sql-proxy binary location.

        Searches in order:
        1. $PATH (via which/where)
        2. ~/.local/bin/cloud-sql-proxy (common install location)
        3. ~/google-cloud-sdk/bin/cloud-sql-proxy
        4. /usr/local/bin/cloud-sql-proxy
        5. ~/bin/cloud-sql-proxy
        6. /opt/homebrew/bin/cloud-sql-proxy (Apple Silicon)
        """
        # Try PATH first
        binary = shutil.which("cloud-sql-proxy")
        if binary:
            logger.info(f"📍 Found proxy binary in PATH: {binary}")
            return binary

        # Try common locations (ENHANCED - includes ~/.local/bin first!)
        search_paths = [
            Path.home() / ".local" / "bin" / "cloud-sql-proxy",  # Most common
            Path.home() / "google-cloud-sdk" / "bin" / "cloud-sql-proxy",
            Path("/usr/local/bin/cloud-sql-proxy"),
            Path.home() / "bin" / "cloud-sql-proxy",
            Path("/opt/homebrew/bin/cloud-sql-proxy"),  # Apple Silicon Homebrew
        ]

        for path in search_paths:
            if path.exists() and os.access(path, os.X_OK):
                logger.info(f"📍 Found proxy binary: {path}")
                return str(path)

        # Enhanced error message with install instructions
        error_msg = (
            "❌ cloud-sql-proxy binary not found!\n\n"
            "Install options:\n"
            "  1. Download directly:\n"
            "     curl -o ~/.local/bin/cloud-sql-proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.8.2/cloud-sql-proxy.darwin.amd64\n"
            "     chmod +x ~/.local/bin/cloud-sql-proxy\n\n"
            "  2. Using gcloud:\n"
            "     gcloud components install cloud-sql-proxy\n\n"
            f"Searched locations: {[str(p) for p in search_paths]}"
        )
        raise FileNotFoundError(error_msg)

    def _is_port_in_use(self, port: int) -> bool:
        """Check if port is already in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port)) == 0

    def _find_proxy_processes(self) -> list:
        """Find all running cloud-sql-proxy processes."""
        try:
            if self.system == "Darwin" or self.system == "Linux":
                result = subprocess.run(
                    ["pgrep", "-f", "cloud-sql-proxy"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    return [int(pid) for pid in result.stdout.strip().split()]
            elif self.system == "Windows":
                result = subprocess.run(
                    ["tasklist", "/FI", "IMAGENAME eq cloud-sql-proxy*"],
                    capture_output=True,
                    text=True,
                )
                # Parse Windows tasklist output
                pids = []
                for line in result.stdout.split("\n")[3:]:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            pids.append(int(parts[1]))
                        except ValueError:
                            pass
                return pids
        except Exception as e:
            logger.debug(f"Error finding proxy processes: {e}")

        return []

    def is_running(self) -> bool:
        """Check if proxy is running and healthy."""
        # Check if port is in use
        port = self.config["cloud_sql"]["port"]
        if not self._is_port_in_use(port):
            return False

        # Check if PID file exists and process is alive
        if self.pid_path.exists():
            try:
                with open(self.pid_path) as f:
                    pid = int(f.read().strip())
                # Check if process exists
                os.kill(pid, 0)  # Doesn't actually kill, just checks
                return True
            except (ProcessLookupError, ValueError):
                self.pid_path.unlink(missing_ok=True)

        return False

    def _kill_conflicting_processes(self):
        """Kill any conflicting proxy processes."""
        port = self.config["cloud_sql"]["port"]

        if not self._is_port_in_use(port):
            return

        logger.warning(f"⚠️  Port {port} in use, killing conflicting processes...")
        pids = self._find_proxy_processes()

        for pid in pids:
            try:
                logger.info(f"🔪 Killing proxy process: PID {pid}")
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.5)
                # Force kill if still alive
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            except Exception as e:
                logger.debug(f"Error killing PID {pid}: {e}")

        # Wait for port to become available
        for _ in range(10):
            if not self._is_port_in_use(port):
                break
            time.sleep(0.5)
        else:
            logger.error(f"❌ Failed to free port {port}")

    async def start(self, force_restart: bool = False, max_retries: int = 3) -> bool:
        """
        Start Cloud SQL proxy with health monitoring and auto-recovery.

        Args:
            force_restart: Kill existing processes and start fresh
            max_retries: Maximum number of startup attempts

        Returns:
            True if started successfully, False otherwise
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f"🔄 Retry attempt {attempt + 1}/{max_retries}...")
                    await asyncio.sleep(2)  # Wait before retry

                # Check if already running
                if self.is_running() and not force_restart:
                    logger.info("✅ Cloud SQL proxy already running")
                    return True

                # Kill conflicting processes if needed
                if force_restart or attempt > 0:
                    self._kill_conflicting_processes()

                # Build proxy command (no hardcoding!)
                cloud_sql = self.config["cloud_sql"]
                port = cloud_sql["port"]
                connection_name = cloud_sql["connection_name"]

                cmd = [
                    self.proxy_binary,
                    connection_name,
                    f"--port={port}",
                ]

                # Add optional auth if specified in config
                if "auth_method" in cloud_sql:
                    if cloud_sql["auth_method"] == "service_account":
                        if "service_account_key" in cloud_sql:
                            cmd.append(f"--credentials-file={cloud_sql['service_account_key']}")

                logger.info(f"🚀 Starting Cloud SQL proxy (attempt {attempt + 1}/{max_retries})...")
                logger.info(f"   Binary: {self.proxy_binary}")
                logger.info(f"   Connection: {connection_name}")
                logger.info(f"   Port: {port}")
                logger.info(f"   Log: {self.log_path}")
                logger.info(f"   Command: {' '.join(cmd)}")

                # Ensure log directory exists
                self.log_path.parent.mkdir(parents=True, exist_ok=True)

                # Start proxy process (truncate log file for fresh start)
                log_file = open(self.log_path, "w")  # Use "w" to truncate old logs
                self.process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,  # Detach from parent
                )

                # Write PID file
                with open(self.pid_path, "w") as f:
                    f.write(str(self.process.pid))

                logger.info(f"   PID: {self.process.pid}")

                # Wait for proxy to be ready (max 15 seconds) - ASYNC!
                logger.info(f"⏳ Waiting for proxy to be ready...")
                for i in range(30):
                    await asyncio.sleep(0.5)  # Non-blocking async sleep

                    # Check if process crashed
                    if self.process.poll() is not None:
                        log_content = self.log_path.read_text() if self.log_path.exists() else "No log"
                        error_msg = f"Proxy process crashed (exit code: {self.process.returncode})\nLog:\n{log_content}"
                        logger.error(f"❌ {error_msg}")
                        last_error = error_msg
                        break

                    if self._is_port_in_use(port):
                        logger.info(f"✅ Cloud SQL proxy ready on port {port} (took {i * 0.5:.1f}s)")
                        return True

                # If we got here, proxy didn't start in time
                if self.process.poll() is None:
                    logger.error(f"❌ Proxy failed to start within 15 seconds")
                    last_error = "Startup timeout"
                    # Kill the slow process
                    self.process.terminate()

            except FileNotFoundError as e:
                logger.error(f"❌ Proxy binary not found: {e}")
                last_error = str(e)
                break  # No point retrying if binary doesn't exist

            except Exception as e:
                logger.error(f"❌ Failed to start Cloud SQL proxy: {e}", exc_info=True)
                last_error = str(e)

        # All retries failed
        logger.error(f"❌ Cloud SQL proxy failed to start after {max_retries} attempts")
        if last_error:
            logger.error(f"   Last error: {last_error}")

        # Show log file for debugging
        if self.log_path.exists():
            logger.error(f"   Check log file: {self.log_path}")
            try:
                log_tail = self.log_path.read_text().split("\n")[-20:]
                logger.error(f"   Last 20 lines of log:\n" + "\n".join(log_tail))
            except Exception:
                pass

        return False

    async def stop(self) -> bool:
        """Stop Cloud SQL proxy gracefully."""
        try:
            if not self.is_running():
                logger.info("Cloud SQL proxy not running")
                return True

            logger.info("🛑 Stopping Cloud SQL proxy...")

            # Try graceful shutdown first
            if self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            else:
                # Kill using PID file
                if self.pid_path.exists():
                    with open(self.pid_path) as f:
                        pid = int(f.read().strip())
                    os.kill(pid, signal.SIGTERM)

            self.pid_path.unlink(missing_ok=True)
            logger.info("✅ Cloud SQL proxy stopped")
            return True

        except Exception as e:
            logger.error(f"Error stopping proxy: {e}", exc_info=True)
            return False

    async def restart(self) -> bool:
        """Restart Cloud SQL proxy."""
        logger.info("🔄 Restarting Cloud SQL proxy...")
        await self.stop()
        await asyncio.sleep(1)
        return await self.start(force_restart=True)

    async def monitor(self, check_interval: int = 30, max_recovery_attempts: int = 3):
        """
        Monitor proxy health and auto-recover if needed.

        Args:
            check_interval: Seconds between health checks
            max_recovery_attempts: Maximum consecutive recovery attempts before giving up
        """
        logger.info(f"🔍 Starting proxy health monitor (interval: {check_interval}s)")

        consecutive_failures = 0
        last_check_time = time.time()

        while True:
            try:
                await asyncio.sleep(check_interval)

                # Health check
                is_healthy = self.is_running()
                current_time = time.time()
                elapsed = current_time - last_check_time
                last_check_time = current_time

                if not is_healthy:
                    consecutive_failures += 1
                    logger.warning(
                        f"⚠️  Cloud SQL proxy unhealthy "
                        f"(consecutive failures: {consecutive_failures}/{max_recovery_attempts})"
                    )

                    if consecutive_failures <= max_recovery_attempts:
                        logger.info(f"🔄 Attempting automatic recovery (attempt {consecutive_failures})...")
                        success = await self.start(force_restart=True)

                        if success:
                            logger.info("✅ Proxy recovered successfully")
                            consecutive_failures = 0  # Reset counter on success
                        else:
                            logger.error(f"❌ Proxy recovery attempt {consecutive_failures} failed")

                            # If max attempts reached, alert and wait longer
                            if consecutive_failures >= max_recovery_attempts:
                                logger.error(
                                    f"❌ Proxy recovery failed after {max_recovery_attempts} attempts"
                                )
                                logger.error("   Voice authentication will be unavailable")
                                logger.error("   Will continue monitoring and retry in 5 minutes...")
                                await asyncio.sleep(300)  # Wait 5 minutes before trying again
                                consecutive_failures = 0  # Reset to try again
                    else:
                        logger.error("❌ Max recovery attempts exceeded, waiting before retry...")
                else:
                    # Proxy is healthy
                    if consecutive_failures > 0:
                        logger.info("✅ Proxy health restored")
                        consecutive_failures = 0

                    # Log periodic health status
                    if int(current_time) % 300 == 0:  # Every 5 minutes
                        logger.debug(f"✅ Cloud SQL proxy healthy (uptime: {elapsed:.0f}s)")

            except asyncio.CancelledError:
                logger.info("Health monitor stopped")
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}", exc_info=True)

    def install_system_service(self) -> bool:
        """
        Install Cloud SQL proxy as system service (launchd/systemd).

        macOS: Creates launchd plist in ~/Library/LaunchAgents/
        Linux: Creates systemd unit in ~/.config/systemd/user/
        """
        try:
            if self.system == "Darwin":
                return self._install_launchd_service()
            elif self.system == "Linux":
                return self._install_systemd_service()
            else:
                logger.warning(f"System service not supported on {self.system}")
                return False
        except Exception as e:
            logger.error(f"Failed to install system service: {e}", exc_info=True)
            return False

    def _install_launchd_service(self) -> bool:
        """Install macOS launchd service."""
        plist_dir = Path.home() / "Library" / "LaunchAgents"
        plist_dir.mkdir(parents=True, exist_ok=True)
        plist_path = plist_dir / f"{self.service_name}.plist"

        cloud_sql = self.config["cloud_sql"]

        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{self.service_name}</string>

    <key>ProgramArguments</key>
    <array>
        <string>{self.proxy_binary}</string>
        <string>{cloud_sql['connection_name']}</string>
        <string>--port={cloud_sql['port']}</string>
    </array>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>StandardOutPath</key>
    <string>{self.log_path}</string>

    <key>StandardErrorPath</key>
    <string>{self.log_path}</string>

    <key>ProcessType</key>
    <string>Background</string>
</dict>
</plist>
"""

        with open(plist_path, "w") as f:
            f.write(plist_content)

        logger.info(f"✅ Created launchd service: {plist_path}")

        # Load service
        subprocess.run(["launchctl", "load", str(plist_path)], check=True)
        logger.info(f"✅ Loaded launchd service: {self.service_name}")

        return True

    def _install_systemd_service(self) -> bool:
        """Install Linux systemd user service."""
        systemd_dir = Path.home() / ".config" / "systemd" / "user"
        systemd_dir.mkdir(parents=True, exist_ok=True)
        service_path = systemd_dir / f"{self.service_name}.service"

        cloud_sql = self.config["cloud_sql"]

        service_content = f"""[Unit]
Description=JARVIS Cloud SQL Proxy
After=network.target

[Service]
Type=simple
ExecStart={self.proxy_binary} {cloud_sql['connection_name']} --port={cloud_sql['port']}
Restart=on-failure
RestartSec=5
StandardOutput=append:{self.log_path}
StandardError=append:{self.log_path}

[Install]
WantedBy=default.target
"""

        with open(service_path, "w") as f:
            f.write(service_content)

        logger.info(f"✅ Created systemd service: {service_path}")

        # Reload systemd and enable service
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(
            ["systemctl", "--user", "enable", f"{self.service_name}.service"], check=True
        )
        subprocess.run(["systemctl", "--user", "start", f"{self.service_name}.service"], check=True)

        logger.info(f"✅ Enabled systemd service: {self.service_name}")
        return True

    def uninstall_system_service(self) -> bool:
        """Uninstall system service."""
        try:
            if self.system == "Darwin":
                plist_path = Path.home() / "Library" / "LaunchAgents" / f"{self.service_name}.plist"
                if plist_path.exists():
                    subprocess.run(["launchctl", "unload", str(plist_path)])
                    plist_path.unlink()
                    logger.info(f"✅ Uninstalled launchd service")
            elif self.system == "Linux":
                subprocess.run(["systemctl", "--user", "stop", f"{self.service_name}.service"])
                subprocess.run(["systemctl", "--user", "disable", f"{self.service_name}.service"])
                service_path = (
                    Path.home() / ".config" / "systemd" / "user" / f"{self.service_name}.service"
                )
                if service_path.exists():
                    service_path.unlink()
                subprocess.run(["systemctl", "--user", "daemon-reload"])
                logger.info(f"✅ Uninstalled systemd service")
            return True
        except Exception as e:
            logger.error(f"Error uninstalling service: {e}", exc_info=True)
            return False


def get_proxy_manager() -> CloudSQLProxyManager:
    """Get singleton proxy manager instance."""
    if not hasattr(get_proxy_manager, "_instance"):
        get_proxy_manager._instance = CloudSQLProxyManager()
    return get_proxy_manager._instance


if __name__ == "__main__":
    # CLI for manual testing
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Cloud SQL Proxy Manager")
    parser.add_argument(
        "command", choices=["start", "stop", "restart", "status", "install", "uninstall"]
    )
    parser.add_argument("--force", action="store_true", help="Force restart if already running")

    args = parser.parse_args()

    manager = CloudSQLProxyManager()

    if args.command == "start":
        success = manager.start(force_restart=args.force)
        sys.exit(0 if success else 1)
    elif args.command == "stop":
        success = manager.stop()
        sys.exit(0 if success else 1)
    elif args.command == "restart":
        success = manager.restart()
        sys.exit(0 if success else 1)
    elif args.command == "status":
        if manager.is_running():
            print("✅ Cloud SQL proxy is running")
            sys.exit(0)
        else:
            print("❌ Cloud SQL proxy is not running")
            sys.exit(1)
    elif args.command == "install":
        success = manager.install_system_service()
        sys.exit(0 if success else 1)
    elif args.command == "uninstall":
        success = manager.uninstall_system_service()
        sys.exit(0 if success else 1)
