#!/usr/bin/env python3
"""
AirPlay RAOP Protocol Handler
==============================

Custom implementation of AirPlay/RAOP protocol for screen mirroring.

Protocol Stack:
- RAOP (Remote Audio Output Protocol) - Base protocol
- RTSP (Real Time Streaming Protocol) - Control channel
- HTTP - Data transport
- Fairplay - DRM/encryption (optional)

Features:
- Screen mirroring initiation via RAOP
- System AirPlay integration (macOS native)
- RTSP session management
- Connection lifecycle management
- Async/await throughout
- Comprehensive error handling

Note: For full screen mirroring, this integrates with macOS's native
AirPlay system rather than reimplementing the entire H.264 encoding pipeline.

Author: Derek Russell
Date: 2025-10-16
Version: 2.0
"""

import asyncio
import logging
import json
import socket
import time
import subprocess
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from enum import Enum
import hashlib
import base64

logger = logging.getLogger(__name__)


class AirPlayMethod(Enum):
    """AirPlay connection methods"""
    SYSTEM_NATIVE = "system_native"  # Use macOS native AirPlay
    RAOP_DIRECT = "raop_direct"       # Direct RAOP protocol
    COREMEDIASTREAM = "coremediastream"  # Private API (if available)


class ConnectionState(Enum):
    """Connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"


@dataclass
class ConnectionResult:
    """Result of connection attempt"""
    success: bool
    state: ConnectionState
    message: str
    method: AirPlayMethod
    duration: float
    metadata: Dict[str, Any] = None


class AirPlayProtocol:
    """
    AirPlay RAOP Protocol Handler

    Handles AirPlay connections using multiple strategies:
    1. macOS native AirPlay (preferred - no protocol implementation needed)
    2. Direct RAOP protocol (custom implementation)
    3. CoreMediaStream private API (fallback)
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize AirPlay protocol handler"""
        self.config = self._load_config(config_path)

        # Connection state
        self.connections: Dict[str, ConnectionState] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self.stats = {
            'total_connections': 0,
            'successful_connections': 0,
            'failed_connections': 0,
            'avg_connection_time': 0.0
        }

        logger.info("[AIRPLAY PROTOCOL] Initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'airplay_config.json'

        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"[AIRPLAY PROTOCOL] Config not found: {config_path}")
            raise

    async def connect(
        self,
        device_name: str,
        ip_address: str,
        port: int,
        mode: str = "extend",
        method: Optional[AirPlayMethod] = None
    ) -> ConnectionResult:
        """
        Connect to AirPlay device

        Args:
            device_name: Name of the AirPlay device
            ip_address: IP address of device
            port: Port number
            mode: Mirroring mode (mirror or extend)
            method: Preferred connection method

        Returns:
            ConnectionResult with status and details
        """
        start_time = time.time()
        device_id = self._get_device_id(device_name, ip_address)

        logger.info(f"[AIRPLAY PROTOCOL] Connecting to {device_name} ({ip_address}:{port}) mode={mode}")

        self.stats['total_connections'] += 1
        self.connections[device_id] = ConnectionState.CONNECTING

        try:
            # Try connection methods in order of preference
            if method is None:
                # Auto-select best method
                method = await self._select_best_method(device_name, ip_address)

            logger.info(f"[AIRPLAY PROTOCOL] Using method: {method.value}")

            if method == AirPlayMethod.SYSTEM_NATIVE:
                result = await self._connect_via_system(device_name, ip_address, port, mode)
            elif method == AirPlayMethod.RAOP_DIRECT:
                result = await self._connect_via_raop(device_name, ip_address, port, mode)
            elif method == AirPlayMethod.COREMEDIASTREAM:
                result = await self._connect_via_coremediastream(device_name, ip_address, port, mode)
            else:
                raise ValueError(f"Unknown method: {method}")

            duration = time.time() - start_time
            result.duration = duration

            if result.success:
                self.connections[device_id] = ConnectionState.CONNECTED
                self.active_sessions[device_id] = {
                    'device_name': device_name,
                    'ip_address': ip_address,
                    'port': port,
                    'mode': mode,
                    'method': method,
                    'connected_at': datetime.now(),
                    'duration': duration
                }
                self.stats['successful_connections'] += 1

                # Update average connection time
                total_success = self.stats['successful_connections']
                avg_time = self.stats['avg_connection_time']
                self.stats['avg_connection_time'] = (
                    (avg_time * (total_success - 1) + duration) / total_success
                )

                logger.info(f"[AIRPLAY PROTOCOL] ✅ Connected to {device_name} in {duration:.2f}s")
            else:
                self.connections[device_id] = ConnectionState.ERROR
                self.stats['failed_connections'] += 1
                logger.warning(f"[AIRPLAY PROTOCOL] ❌ Connection failed: {result.message}")

            return result

        except Exception as e:
            logger.error(f"[AIRPLAY PROTOCOL] Connection error: {e}", exc_info=True)
            self.connections[device_id] = ConnectionState.ERROR
            self.stats['failed_connections'] += 1

            return ConnectionResult(
                success=False,
                state=ConnectionState.ERROR,
                message=f"Connection error: {str(e)}",
                method=method or AirPlayMethod.SYSTEM_NATIVE,
                duration=time.time() - start_time
            )

    async def _select_best_method(self, device_name: str, ip_address: str) -> AirPlayMethod:
        """
        Select best connection method for device

        Strategy:
        1. System Native (preferred - most reliable)
        2. CoreMediaStream (if available)
        3. RAOP Direct (fallback)
        """
        # Always prefer system native on macOS
        if await self._is_macos():
            logger.debug("[AIRPLAY PROTOCOL] Selected system native method")
            return AirPlayMethod.SYSTEM_NATIVE

        # Fallback to RAOP
        logger.debug("[AIRPLAY PROTOCOL] Selected RAOP direct method")
        return AirPlayMethod.RAOP_DIRECT

    async def _is_macos(self) -> bool:
        """Check if running on macOS"""
        import platform
        return platform.system() == 'Darwin'

    async def _connect_via_system(
        self,
        device_name: str,
        ip_address: str,
        port: int,
        mode: str
    ) -> ConnectionResult:
        """
        Connect via macOS native AirPlay system

        This method triggers the system's built-in AirPlay functionality,
        which handles all the complex protocol details (H.264, encryption, etc.)
        """
        try:
            logger.info(f"[AIRPLAY PROTOCOL] Using macOS native AirPlay for {device_name}")

            # Strategy 1: Use CoreMediaStream private framework (if available)
            try:
                result = await self._trigger_system_airplay_coremediastream(device_name, mode)
                if result.success:
                    return result
                logger.debug("[AIRPLAY PROTOCOL] CoreMediaStream not available, trying AppleScript")
            except Exception as e:
                logger.debug(f"[AIRPLAY PROTOCOL] CoreMediaStream failed: {e}")

            # Strategy 2: Use AppleScript to trigger system AirPlay
            result = await self._trigger_system_airplay_applescript(device_name, mode)
            if result.success:
                return result

            # Strategy 3: Use system_profiler + networksetup (detection only)
            logger.warning("[AIRPLAY PROTOCOL] All system methods failed")

            return ConnectionResult(
                success=False,
                state=ConnectionState.ERROR,
                message="System native AirPlay failed - all strategies exhausted",
                method=AirPlayMethod.SYSTEM_NATIVE,
                duration=0.0
            )

        except Exception as e:
            logger.error(f"[AIRPLAY PROTOCOL] System connection error: {e}")
            return ConnectionResult(
                success=False,
                state=ConnectionState.ERROR,
                message=f"System connection error: {str(e)}",
                method=AirPlayMethod.SYSTEM_NATIVE,
                duration=0.0
            )

    async def _trigger_system_airplay_coremediastream(
        self,
        device_name: str,
        mode: str
    ) -> ConnectionResult:
        """
        Trigger system AirPlay via CoreMediaStream framework

        This uses macOS private APIs to control AirPlay directly.
        """
        try:
            # Try to import objc bridge
            import objc
            from Foundation import NSBundle

            # Load CoreMediaStream framework
            bundle_path = '/System/Library/PrivateFrameworks/CoreMediaStream.framework'
            bundle = NSBundle.bundleWithPath_(bundle_path)

            if not bundle:
                raise Exception("CoreMediaStream framework not available")

            if not bundle.load():
                raise Exception("Failed to load CoreMediaStream framework")

            logger.info("[AIRPLAY PROTOCOL] CoreMediaStream loaded successfully")

            # TODO: Implement actual CoreMediaStream API calls
            # This would require reverse engineering the private API
            # For now, return not implemented

            return ConnectionResult(
                success=False,
                state=ConnectionState.ERROR,
                message="CoreMediaStream implementation pending",
                method=AirPlayMethod.COREMEDIASTREAM,
                duration=0.0
            )

        except ImportError:
            logger.debug("[AIRPLAY PROTOCOL] PyObjC not available")
            raise
        except Exception as e:
            logger.debug(f"[AIRPLAY PROTOCOL] CoreMediaStream error: {e}")
            raise

    async def _trigger_system_airplay_applescript(
        self,
        device_name: str,
        mode: str
    ) -> ConnectionResult:
        """
        Trigger system AirPlay via AppleScript

        This uses AppleScript to automate the macOS System Preferences
        or Control Center to connect to the AirPlay device.
        """
        try:
            logger.info(f"[AIRPLAY PROTOCOL] Using AppleScript to connect to {device_name}")

            # AppleScript to open System Preferences and connect to AirPlay
            # Note: This may not work reliably on macOS Sequoia+ due to security restrictions
            script = f'''
            tell application "System Events"
                -- Try to use Control Center (macOS 11+)
                try
                    tell process "ControlCenter"
                        click menu bar item "Screen Mirroring" of menu bar 1
                        delay 0.5
                        click menu item "{device_name}" of menu 1 of menu bar item "Screen Mirroring" of menu bar 1
                        return "SUCCESS"
                    end tell
                on error errMsg
                    return "ERROR:" & errMsg
                end try
            end tell
            '''

            # Execute AppleScript
            proc = await asyncio.create_subprocess_exec(
                'osascript', '-e', script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config['connection']['timeout_seconds']
            )

            output = stdout.decode('utf-8').strip()

            if "SUCCESS" in output:
                return ConnectionResult(
                    success=True,
                    state=ConnectionState.CONNECTED,
                    message=f"Connected via AppleScript",
                    method=AirPlayMethod.SYSTEM_NATIVE,
                    duration=0.0
                )
            else:
                error_msg = output.replace("ERROR:", "").strip()
                return ConnectionResult(
                    success=False,
                    state=ConnectionState.ERROR,
                    message=f"AppleScript failed: {error_msg}",
                    method=AirPlayMethod.SYSTEM_NATIVE,
                    duration=0.0
                )

        except asyncio.TimeoutError:
            return ConnectionResult(
                success=False,
                state=ConnectionState.ERROR,
                message="AppleScript timeout",
                method=AirPlayMethod.SYSTEM_NATIVE,
                duration=0.0
            )
        except Exception as e:
            logger.error(f"[AIRPLAY PROTOCOL] AppleScript error: {e}")
            return ConnectionResult(
                success=False,
                state=ConnectionState.ERROR,
                message=f"AppleScript error: {str(e)}",
                method=AirPlayMethod.SYSTEM_NATIVE,
                duration=0.0
            )

    async def _connect_via_raop(
        self,
        device_name: str,
        ip_address: str,
        port: int,
        mode: str
    ) -> ConnectionResult:
        """
        Connect via direct RAOP protocol

        This implements a custom RAOP client.
        Note: Full screen mirroring via RAOP requires H.264 encoding,
        which is complex. This is a simplified implementation.
        """
        try:
            logger.info(f"[AIRPLAY PROTOCOL] Direct RAOP connection to {ip_address}:{port}")

            # RAOP connection steps:
            # 1. Open TCP connection to device
            # 2. Send RTSP ANNOUNCE request
            # 3. Send RTSP SETUP request
            # 4. Send RTSP RECORD request
            # 5. Start streaming

            # For now, just test connectivity
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(ip_address, port),
                timeout=self.config['connection']['timeout_seconds']
            )

            logger.info(f"[AIRPLAY PROTOCOL] TCP connection established to {ip_address}:{port}")

            # Send simple RTSP OPTIONS request
            request = (
                f"OPTIONS * RTSP/1.0\r\n"
                f"CSeq: 1\r\n"
                f"User-Agent: JARVIS/2.0\r\n"
                f"\r\n"
            )

            writer.write(request.encode())
            await writer.drain()

            # Read response
            response = await asyncio.wait_for(
                reader.read(1024),
                timeout=5.0
            )

            response_str = response.decode('utf-8', errors='ignore')
            logger.debug(f"[AIRPLAY PROTOCOL] RTSP response: {response_str[:200]}")

            writer.close()
            await writer.wait_closed()

            # TODO: Implement full RAOP handshake and streaming
            # For now, return success if we got a response

            if "RTSP/1.0" in response_str:
                return ConnectionResult(
                    success=True,
                    state=ConnectionState.CONNECTED,
                    message="RAOP connection established (streaming not yet implemented)",
                    method=AirPlayMethod.RAOP_DIRECT,
                    duration=0.0,
                    metadata={'response': response_str[:500]}
                )
            else:
                return ConnectionResult(
                    success=False,
                    state=ConnectionState.ERROR,
                    message="Invalid RTSP response",
                    method=AirPlayMethod.RAOP_DIRECT,
                    duration=0.0
                )

        except asyncio.TimeoutError:
            return ConnectionResult(
                success=False,
                state=ConnectionState.ERROR,
                message="RAOP connection timeout",
                method=AirPlayMethod.RAOP_DIRECT,
                duration=0.0
            )
        except Exception as e:
            logger.error(f"[AIRPLAY PROTOCOL] RAOP error: {e}")
            return ConnectionResult(
                success=False,
                state=ConnectionState.ERROR,
                message=f"RAOP error: {str(e)}",
                method=AirPlayMethod.RAOP_DIRECT,
                duration=0.0
            )

    async def _connect_via_coremediastream(
        self,
        device_name: str,
        ip_address: str,
        port: int,
        mode: str
    ) -> ConnectionResult:
        """
        Connect via CoreMediaStream private API

        This method uses macOS private frameworks to control AirPlay.
        """
        # This would require deeper integration with macOS private APIs
        # For now, delegate to system native method
        return await self._connect_via_system(device_name, ip_address, port, mode)

    async def disconnect(self, device_name: str, ip_address: str) -> bool:
        """Disconnect from AirPlay device"""
        device_id = self._get_device_id(device_name, ip_address)

        if device_id not in self.active_sessions:
            logger.warning(f"[AIRPLAY PROTOCOL] No active session for {device_name}")
            return False

        try:
            session = self.active_sessions[device_id]
            method = session['method']

            logger.info(f"[AIRPLAY PROTOCOL] Disconnecting from {device_name} (method: {method.value})")

            # Disconnect based on method
            if method == AirPlayMethod.SYSTEM_NATIVE:
                # Use AppleScript to disconnect
                await self._disconnect_via_applescript(device_name)
            elif method == AirPlayMethod.RAOP_DIRECT:
                # Send RTSP TEARDOWN
                await self._disconnect_via_raop(ip_address, session['port'])

            # Clean up session
            del self.active_sessions[device_id]
            self.connections[device_id] = ConnectionState.DISCONNECTED

            logger.info(f"[AIRPLAY PROTOCOL] ✅ Disconnected from {device_name}")
            return True

        except Exception as e:
            logger.error(f"[AIRPLAY PROTOCOL] Disconnect error: {e}")
            return False

    async def _disconnect_via_applescript(self, device_name: str):
        """Disconnect via AppleScript"""
        script = f'''
        tell application "System Events"
            tell process "ControlCenter"
                try
                    click menu bar item "Screen Mirroring" of menu bar 1
                    delay 0.3
                    click menu item "Stop Mirroring" of menu 1 of menu bar item "Screen Mirroring" of menu bar 1
                end try
            end tell
        end tell
        '''

        try:
            proc = await asyncio.create_subprocess_exec(
                'osascript', '-e', script,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await proc.wait()
        except Exception as e:
            logger.warning(f"[AIRPLAY PROTOCOL] AppleScript disconnect failed: {e}")

    async def _disconnect_via_raop(self, ip_address: str, port: int):
        """Disconnect via RAOP TEARDOWN"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(ip_address, port),
                timeout=5.0
            )

            request = "TEARDOWN * RTSP/1.0\r\nCSeq: 999\r\n\r\n"
            writer.write(request.encode())
            await writer.drain()

            writer.close()
            await writer.wait_closed()

        except Exception as e:
            logger.warning(f"[AIRPLAY PROTOCOL] RAOP disconnect failed: {e}")

    def _get_device_id(self, device_name: str, ip_address: str) -> str:
        """Generate device ID"""
        return hashlib.md5(f"{device_name}_{ip_address}".encode()).hexdigest()[:12]

    def get_connection_state(self, device_name: str, ip_address: str) -> ConnectionState:
        """Get connection state for device"""
        device_id = self._get_device_id(device_name, ip_address)
        return self.connections.get(device_id, ConnectionState.DISCONNECTED)

    def is_connected(self, device_name: str, ip_address: str) -> bool:
        """Check if connected to device"""
        state = self.get_connection_state(device_name, ip_address)
        return state in [ConnectionState.CONNECTED, ConnectionState.STREAMING]

    def get_stats(self) -> Dict[str, Any]:
        """Get protocol statistics"""
        return {
            **self.stats,
            'active_connections': len(self.active_sessions),
            'success_rate': (
                self.stats['successful_connections'] / self.stats['total_connections'] * 100
                if self.stats['total_connections'] > 0 else 0.0
            )
        }


# Singleton instance
_protocol_handler: Optional[AirPlayProtocol] = None


def get_airplay_protocol(config_path: Optional[str] = None) -> AirPlayProtocol:
    """Get singleton AirPlay protocol handler"""
    global _protocol_handler
    if _protocol_handler is None:
        _protocol_handler = AirPlayProtocol(config_path)
    return _protocol_handler


if __name__ == "__main__":
    # Test protocol handler
    async def test():
        logging.basicConfig(level=logging.INFO)

        protocol = get_airplay_protocol()

        # Test connection (requires AirPlay device on network)
        device_name = "Living Room TV"
        ip_address = "192.168.1.100"  # Replace with actual IP
        port = 7000

        print(f"\nTesting connection to {device_name} at {ip_address}:{port}")

        result = await protocol.connect(device_name, ip_address, port, mode="extend")

        print(f"\nConnection result:")
        print(f"  Success: {result.success}")
        print(f"  State: {result.state.value}")
        print(f"  Message: {result.message}")
        print(f"  Method: {result.method.value}")
        print(f"  Duration: {result.duration:.2f}s")

        # Get stats
        stats = protocol.get_stats()
        print(f"\nStats: {stats}")

        if result.success:
            # Wait a bit
            await asyncio.sleep(5)

            # Disconnect
            disconnected = await protocol.disconnect(device_name, ip_address)
            print(f"\nDisconnected: {disconnected}")

    asyncio.run(test())
