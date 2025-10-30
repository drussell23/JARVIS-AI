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
    """AirPlay connection methods.
    
    Defines the different strategies for connecting to AirPlay devices,
    ordered by preference and reliability.
    """
    SYSTEM_NATIVE = "system_native"  # Use macOS native AirPlay
    RAOP_DIRECT = "raop_direct"       # Direct RAOP protocol
    COREMEDIASTREAM = "coremediastream"  # Private API (if available)


class ConnectionState(Enum):
    """Connection states for AirPlay devices.
    
    Tracks the current state of each device connection throughout
    the connection lifecycle.
    """
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"


@dataclass
class ConnectionResult:
    """Result of connection attempt.
    
    Contains comprehensive information about a connection attempt,
    including success status, timing, and diagnostic information.
    
    Attributes:
        success: Whether the connection was successful
        state: Current connection state
        message: Human-readable status message
        method: Connection method used
        duration: Time taken for connection attempt in seconds
        metadata: Additional diagnostic information
    """
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
    
    This class provides a unified interface for connecting to AirPlay devices
    regardless of the underlying connection method. It automatically selects
    the best available method and provides comprehensive error handling and
    connection state management.
    
    Attributes:
        config: Configuration dictionary loaded from JSON file
        connections: Mapping of device IDs to connection states
        active_sessions: Mapping of device IDs to session information
        stats: Connection statistics and metrics
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize AirPlay protocol handler.
        
        Args:
            config_path: Path to configuration file. If None, uses default location.
            
        Raises:
            FileNotFoundError: If configuration file cannot be found
            json.JSONDecodeError: If configuration file is invalid JSON
        """
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

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            json.JSONDecodeError: If configuration file is invalid JSON
        """
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
        Connect to AirPlay device.
        
        Attempts to establish a connection to the specified AirPlay device using
        the best available method. Automatically handles method selection if not
        specified, and provides comprehensive error handling and statistics tracking.

        Args:
            device_name: Human-readable name of the AirPlay device
            ip_address: IP address of the device
            port: Port number for connection (typically 7000 for AirPlay)
            mode: Mirroring mode, either "mirror" or "extend"
            method: Preferred connection method. If None, auto-selects best method

        Returns:
            ConnectionResult containing success status, state, timing, and metadata

        Example:
            >>> protocol = AirPlayProtocol()
            >>> result = await protocol.connect("Living Room TV", "192.168.1.100", 7000)
            >>> if result.success:
            ...     print(f"Connected in {result.duration:.2f}s")
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
        Select best connection method for device.

        Automatically chooses the most appropriate connection method based on
        the current platform and device capabilities.

        Strategy:
        1. System Native (preferred - most reliable)
        2. CoreMediaStream (if available)
        3. RAOP Direct (fallback)
        
        Args:
            device_name: Name of the target device
            ip_address: IP address of the target device
            
        Returns:
            The best available AirPlayMethod for this device
        """
        # Always prefer system native on macOS
        if await self._is_macos():
            logger.debug("[AIRPLAY PROTOCOL] Selected system native method")
            return AirPlayMethod.SYSTEM_NATIVE

        # Fallback to RAOP
        logger.debug("[AIRPLAY PROTOCOL] Selected RAOP direct method")
        return AirPlayMethod.RAOP_DIRECT

    async def _is_macos(self) -> bool:
        """Check if running on macOS.
        
        Returns:
            True if running on macOS, False otherwise
        """
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
        Connect via macOS native AirPlay system.

        This method triggers the system's built-in AirPlay functionality,
        which handles all the complex protocol details (H.264, encryption, etc.)
        
        Uses multiple strategies in order of preference:
        1. CoreMediaStream private framework
        2. AppleScript automation
        3. System profiler detection
        
        Args:
            device_name: Name of the AirPlay device
            ip_address: IP address of the device
            port: Port number for connection
            mode: Mirroring mode ("mirror" or "extend")
            
        Returns:
            ConnectionResult with success status and details
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
        Trigger system AirPlay via CoreMediaStream framework.

        This uses macOS private APIs to control AirPlay directly.
        Requires PyObjC and access to private frameworks.
        
        Args:
            device_name: Name of the target device
            mode: Mirroring mode
            
        Returns:
            ConnectionResult with attempt status
            
        Raises:
            ImportError: If PyObjC is not available
            Exception: If CoreMediaStream framework cannot be loaded
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
        Trigger system AirPlay via AppleScript.

        This uses AppleScript to automate the macOS System Preferences
        or Control Center to connect to the AirPlay device.
        
        Note: May not work reliably on macOS Sequoia+ due to security restrictions.
        
        Args:
            device_name: Name of the target device
            mode: Mirroring mode
            
        Returns:
            ConnectionResult with connection status
            
        Raises:
            asyncio.TimeoutError: If AppleScript execution times out
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
        Connect via direct RAOP protocol.

        This implements a custom RAOP client using RTSP over TCP.
        
        RAOP connection steps:
        1. Open TCP connection to device
        2. Send RTSP ANNOUNCE request
        3. Send RTSP SETUP request
        4. Send RTSP RECORD request
        5. Start streaming
        
        Note: Full screen mirroring via RAOP requires H.264 encoding,
        which is complex. This is a simplified implementation for testing connectivity.
        
        Args:
            device_name: Name of the target device
            ip_address: IP address of the device
            port: Port number for RAOP connection
            mode: Mirroring mode
            
        Returns:
            ConnectionResult with connection status and response metadata
            
        Raises:
            asyncio.TimeoutError: If connection or communication times out
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
        Connect via CoreMediaStream private API.

        This method uses macOS private frameworks to control AirPlay.
        Currently delegates to system native method as a fallback.
        
        Args:
            device_name: Name of the target device
            ip_address: IP address of the device
            port: Port number for connection
            mode: Mirroring mode
            
        Returns:
            ConnectionResult from system native connection method
        """
        # This would require deeper integration with macOS private APIs
        # For now, delegate to system native method
        return await self._connect_via_system(device_name, ip_address, port, mode)

    async def disconnect(self, device_name: str, ip_address: str) -> bool:
        """Disconnect from AirPlay device.
        
        Terminates the connection to the specified device using the appropriate
        method based on how the connection was established.
        
        Args:
            device_name: Name of the device to disconnect from
            ip_address: IP address of the device
            
        Returns:
            True if disconnection was successful, False otherwise
            
        Example:
            >>> success = await protocol.disconnect("Living Room TV", "192.168.1.100")
            >>> if success:
            ...     print("Disconnected successfully")
        """
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

    async def _disconnect_via_applescript(self, device_name: str) -> None:
        """Disconnect via AppleScript.
        
        Uses AppleScript to automate the macOS Control Center to stop
        screen mirroring.
        
        Args:
            device_name: Name of the device to disconnect from
            
        Raises:
            Exception: If AppleScript execution fails (logged as warning)
        """
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

    async def _disconnect_via_raop(self, ip_address: str, port: int) -> None:
        """Disconnect via RAOP TEARDOWN.
        
        Sends an RTSP TEARDOWN request to cleanly terminate the RAOP session.
        
        Args:
            ip_address: IP address of the device
            port: Port number for the connection
            
        Raises:
            Exception: If RAOP disconnect fails (logged as warning)
        """
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
        """Generate unique device ID.
        
        Creates a unique identifier for a device based on its name and IP address.
        Used for tracking connections and sessions.
        
        Args:
            device_name: Name of the device
            ip_address: IP address of the device
            
        Returns:
            12-character hexadecimal device ID
        """
        return hashlib.md5(f"{device_name}_{ip_address}".encode()).hexdigest()[:12]

    def get_connection_state(self, device_name: str, ip_address: str) -> ConnectionState:
        """Get connection state for device.
        
        Args:
            device_name: Name of the device
            ip_address: IP address of the device
            
        Returns:
            Current ConnectionState for the device
        """
        device_id = self._get_device_id(device_name, ip_address)
        return self.connections.get(device_id, ConnectionState.DISCONNECTED)

    def is_connected(self, device_name: str, ip_address: str) -> bool:
        """Check if connected to device.
        
        Args:
            device_name: Name of the device
            ip_address: IP address of the device
            
        Returns:
            True if device is connected or streaming, False otherwise
        """
        state = self.get_connection_state(device_name, ip_address)
        return state in [ConnectionState.CONNECTED, ConnectionState.STREAMING]

    def get_stats(self) -> Dict[str, Any]:
        """Get protocol statistics.
        
        Returns comprehensive statistics about connection attempts,
        success rates, and performance metrics.
        
        Returns:
            Dictionary containing:
                - total_connections: Total connection attempts
                - successful_connections: Number of successful connections
                - failed_connections: Number of failed connections
                - avg_connection_time: Average connection time in seconds
                - active_connections: Current number of active connections
                - success_rate: Success rate as percentage
        """
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
    """Get singleton AirPlay protocol handler.
    
    Provides a singleton instance of the AirPlay protocol handler to ensure
    consistent state management across the application.
    
    Args:
        config_path: Path to configuration file. Only used on first call.
        
    Returns:
        Singleton AirPlayProtocol instance
        
    Example:
        >>> protocol = get_airplay_protocol()
        >>> result = await protocol.connect("TV", "192.168.1.100", 7000)
    """
    global _protocol_handler
    if _protocol_handler is None:
        _protocol_handler = AirPlayProtocol(config_path)
    return _protocol_handler


if __name__ == "__main__":
    # Test protocol handler
    async def test() -> None:
        """Test the AirPlay protocol handler.
        
        Demonstrates basic usage of the protocol handler including
        connection, statistics, and disconnection.
        """
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