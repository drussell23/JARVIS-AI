#!/usr/bin/env python3
"""
AirPlay Route Picker Helper Client
===================================

Python client for the native Swift AirPlay Route Picker Helper.
Uses HTTP to communicate with the helper app for reliable AirPlay connections.

Features:
- Automatic helper process management
- Health checks and status monitoring
- Retry logic with exponential backoff
- Telemetry and error reporting

Author: Derek Russell
Date: 2025-10-16
Version: 1.0
"""

import asyncio
import subprocess
import os
import signal
import logging
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class RoutePickerHelperClient:
    """Client for AirPlay Route Picker Helper"""

    def __init__(self, helper_path: Optional[str] = None, helper_port: int = 8020):
        """
        Initialize the Route Picker Helper client

        Args:
            helper_path: Path to AirPlayRoutePickerHelper executable
            helper_port: Port the helper listens on (default: 8020)
        """
        if helper_path is None:
            helper_path = Path(__file__).parent / "native" / "AirPlayRoutePickerHelper"

        self.helper_path = str(helper_path)
        self.helper_port = helper_port
        self.base_url = f"http://localhost:{helper_port}"
        self.helper_process: Optional[subprocess.Popen] = None
        self.is_running = False

        logger.info(f"[ROUTE PICKER CLIENT] Initialized with helper at {self.helper_path}")

    async def start_helper(self) -> bool:
        """
        Start the helper process if not already running

        Returns:
            bool: True if helper is running, False otherwise
        """
        if self.is_running:
            logger.debug("[ROUTE PICKER CLIENT] Helper already running")
            return True

        # Check if helper exists
        if not os.path.exists(self.helper_path):
            logger.error(f"[ROUTE PICKER CLIENT] Helper not found at {self.helper_path}")
            return False

        try:
            logger.info("[ROUTE PICKER CLIENT] Starting helper process...")

            # Start the helper as a background process
            self.helper_process = subprocess.Popen(
                [self.helper_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True  # Detach from parent
            )

            # Wait for helper to be ready
            for i in range(20):  # 10 seconds max wait
                await asyncio.sleep(0.5)

                try:
                    status = await self.get_status()
                    if status and status.get("ready"):
                        self.is_running = True
                        logger.info(f"[ROUTE PICKER CLIENT] ✅ Helper ready on port {self.helper_port}")
                        return True
                except Exception:
                    pass

            logger.error("[ROUTE PICKER CLIENT] Helper failed to become ready")
            self.stop_helper()
            return False

        except Exception as e:
            logger.error(f"[ROUTE PICKER CLIENT] Failed to start helper: {e}")
            return False

    def stop_helper(self):
        """Stop the helper process"""
        if self.helper_process:
            try:
                # Try graceful termination first
                self.helper_process.terminate()
                self.helper_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if needed
                self.helper_process.kill()
                self.helper_process.wait()
            finally:
                self.helper_process = None
                self.is_running = False

            logger.info("[ROUTE PICKER CLIENT] Helper stopped")

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, str]] = None,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Make HTTP request to helper

        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint (e.g., "/status")
            params: Query parameters
            timeout: Request timeout in seconds

        Returns:
            Response dictionary

        Raises:
            Exception on request failure
        """
        url = f"{self.base_url}{endpoint}"

        if params:
            query = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
            url += f"?{query}"

        try:
            req = urllib.request.Request(url, method=method)
            req.add_header("Content-Type", "application/json")

            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = response.read().decode("utf-8")
                return json.loads(data)

        except urllib.error.URLError as e:
            logger.error(f"[ROUTE PICKER CLIENT] Request failed: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"[ROUTE PICKER CLIENT] Invalid JSON response: {e}")
            raise

    async def get_status(self) -> Optional[Dict[str, Any]]:
        """
        Get helper status

        Returns:
            Status dictionary or None on failure
        """
        try:
            return await self._make_request("GET", "/status", timeout=2.0)
        except Exception as e:
            logger.debug(f"[ROUTE PICKER CLIENT] Status check failed: {e}")
            return None

    async def connect_to_device(
        self,
        device_name: str,
        retry_attempts: int = 2,
        retry_delay: float = 1.0
    ) -> Dict[str, Any]:
        """
        Connect to AirPlay device

        Args:
            device_name: Name of device to connect to
            retry_attempts: Number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            Connection result dictionary with keys:
            - success: bool
            - message: str
            - deviceName: str (optional)
            - duration: float (optional)
            - telemetry: dict (optional)
        """
        # Ensure helper is running
        if not self.is_running:
            started = await self.start_helper()
            if not started:
                return {
                    "success": False,
                    "message": "Failed to start Route Picker Helper"
                }

        last_error = None

        for attempt in range(retry_attempts):
            if attempt > 0:
                logger.info(f"[ROUTE PICKER CLIENT] Retry attempt {attempt + 1}/{retry_attempts}")
                await asyncio.sleep(retry_delay)

            try:
                logger.info(f"[ROUTE PICKER CLIENT] Connecting to '{device_name}'...")

                result = await self._make_request(
                    "POST",
                    "/connect",
                    params={"device": device_name},
                    timeout=30.0
                )

                if result.get("success"):
                    logger.info(
                        f"[ROUTE PICKER CLIENT] ✅ Connected to '{device_name}' "
                        f"in {result.get('duration', 0):.2f}s"
                    )
                else:
                    logger.warning(f"[ROUTE PICKER CLIENT] Connection failed: {result.get('message')}")

                return result

            except Exception as e:
                last_error = str(e)
                logger.error(f"[ROUTE PICKER CLIENT] Connection attempt failed: {e}")

        return {
            "success": False,
            "message": f"All attempts failed: {last_error}"
        }

    async def disconnect(self) -> Dict[str, Any]:
        """
        Disconnect from current AirPlay device

        Returns:
            Disconnect result dictionary
        """
        if not self.is_running:
            return {
                "success": False,
                "message": "Helper not running"
            }

        try:
            logger.info("[ROUTE PICKER CLIENT] Disconnecting...")

            result = await self._make_request(
                "POST",
                "/disconnect",
                timeout=10.0
            )

            if result.get("success"):
                logger.info("[ROUTE PICKER CLIENT] ✅ Disconnected")
            else:
                logger.warning(f"[ROUTE PICKER CLIENT] Disconnect failed: {result.get('message')}")

            return result

        except Exception as e:
            logger.error(f"[ROUTE PICKER CLIENT] Disconnect failed: {e}")
            return {
                "success": False,
                "message": f"Disconnect error: {str(e)}"
            }

    async def ensure_running(self) -> bool:
        """
        Ensure helper is running and healthy

        Returns:
            bool: True if helper is healthy, False otherwise
        """
        status = await self.get_status()

        if status and status.get("ready"):
            self.is_running = True
            return True

        # Try to start if not running
        return await self.start_helper()

    def __del__(self):
        """Cleanup on deletion"""
        if self.helper_process:
            self.stop_helper()


# Singleton instance
_route_picker_client: Optional[RoutePickerHelperClient] = None


def get_route_picker_client() -> RoutePickerHelperClient:
    """Get singleton Route Picker Helper client"""
    global _route_picker_client
    if _route_picker_client is None:
        _route_picker_client = RoutePickerHelperClient()
    return _route_picker_client


if __name__ == "__main__":
    # Test the client
    async def test():
        logging.basicConfig(level=logging.INFO)

        client = get_route_picker_client()

        print("\n=== Starting Helper ===")
        started = await client.start_helper()
        print(f"Started: {started}")

        if started:
            print("\n=== Getting Status ===")
            status = await client.get_status()
            print(json.dumps(status, indent=2))

            print("\n=== Connecting to Living Room TV ===")
            result = await client.connect_to_device("Living Room TV")
            print(json.dumps(result, indent=2))

            if result.get("success"):
                print("\nWaiting 5 seconds...")
                await asyncio.sleep(5)

                print("\n=== Disconnecting ===")
                disconnect_result = await client.disconnect()
                print(json.dumps(disconnect_result, indent=2))

            client.stop_helper()

    asyncio.run(test())
