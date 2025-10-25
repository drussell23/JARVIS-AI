#!/usr/bin/env python3
"""
AirPlay Connection Manager
===========================

High-level manager for AirPlay connections.
Integrates discovery service and protocol handler.

Features:
- Automatic device discovery
- Connection management
- Error handling and retry logic
- Connection pooling
- Event notifications
- Telemetry and statistics
- Async/await throughout

Author: Derek Russell
Date: 2025-10-16
Version: 2.0
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class ConnectionAttempt:
    """Record of a connection attempt"""
    device_name: str
    ip_address: str
    port: int
    mode: str
    timestamp: datetime
    success: bool
    duration: float
    method: str
    error: Optional[str] = None


class AirPlayManager:
    """
    AirPlay Connection Manager

    High-level API for managing AirPlay connections.
    Combines discovery, protocol handling, and connection lifecycle.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize AirPlay manager"""
        self.config = self._load_config(config_path)

        # Import components
        from display.airplay_discovery import get_airplay_discovery
        from display.airplay_protocol import get_airplay_protocol

        self.discovery = get_airplay_discovery(config_path)
        self.protocol = get_airplay_protocol(config_path)

        # State
        self.is_initialized = False
        self.connection_history: List[ConnectionAttempt] = []

        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            'connection_success': [],
            'connection_failed': [],
            'device_discovered': [],
            'device_lost': [],
            'error': []
        }

        # Statistics
        self.stats = {
            'total_connections': 0,
            'successful_connections': 0,
            'failed_connections': 0,
            'devices_discovered': 0,
            'avg_connection_time': 0.0,
            'uptime_start': datetime.now()
        }

        logger.info("[AIRPLAY MANAGER] Initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'airplay_config.json'

        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"[AIRPLAY MANAGER] Config not found: {config_path}")
            raise

    def register_callback(self, event: str, callback: Callable):
        """Register event callback"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
            logger.debug(f"[AIRPLAY MANAGER] Registered callback for {event}")

    async def _emit_event(self, event: str, **kwargs):
        """Emit event to callbacks"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(**kwargs)
                    else:
                        callback(**kwargs)
                except Exception as e:
                    logger.error(f"[AIRPLAY MANAGER] Callback error for {event}: {e}")

    async def initialize(self):
        """Initialize manager and start discovery"""
        if self.is_initialized:
            logger.warning("[AIRPLAY MANAGER] Already initialized")
            return

        logger.info("[AIRPLAY MANAGER] Initializing...")

        # Register discovery callbacks
        self.discovery.register_callback('device_discovered', self._on_device_discovered)
        self.discovery.register_callback('device_lost', self._on_device_lost)

        # Start discovery
        await self.discovery.start()

        self.is_initialized = True
        logger.info("[AIRPLAY MANAGER] ✅ Initialized and ready")

    async def shutdown(self):
        """Shutdown manager"""
        logger.info("[AIRPLAY MANAGER] Shutting down...")

        # Stop discovery
        await self.discovery.stop()

        # Disconnect all active connections
        for session in self.protocol.active_sessions.values():
            await self.protocol.disconnect(
                session['device_name'],
                session['ip_address']
            )

        self.is_initialized = False
        logger.info("[AIRPLAY MANAGER] Shutdown complete")

    async def _on_device_discovered(self, device):
        """Handle device discovered event"""
        self.stats['devices_discovered'] += 1
        logger.info(f"[AIRPLAY MANAGER] Device discovered: {device.name}")
        await self._emit_event('device_discovered', device=device)

    async def _on_device_lost(self, device):
        """Handle device lost event"""
        logger.info(f"[AIRPLAY MANAGER] Device lost: {device.name}")
        await self._emit_event('device_lost', device=device)

        # Disconnect if we have an active connection
        if self.protocol.is_connected(device.name, device.ip_address):
            await self.protocol.disconnect(device.name, device.ip_address)

    async def discover_devices(self, timeout: Optional[float] = None) -> List:
        """
        Discover AirPlay devices

        Args:
            timeout: Discovery timeout (uses config default if None)

        Returns:
            List of discovered devices
        """
        logger.info("[AIRPLAY MANAGER] Starting device discovery...")

        devices = await self.discovery.discover_devices(timeout)

        logger.info(f"[AIRPLAY MANAGER] Found {len(devices)} devices")
        return devices

    async def connect_to_device(
        self,
        device_name: str,
        mode: str = "extend",
        auto_discover: bool = True
    ) -> Dict[str, Any]:
        """
        Connect to AirPlay device

        Args:
            device_name: Name of device to connect to
            mode: Mirroring mode (mirror or extend)
            auto_discover: Auto-discover device if not found

        Returns:
            Connection result dictionary
        """
        start_time = time.time()

        logger.info(f"[AIRPLAY MANAGER] Connecting to '{device_name}' (mode: {mode})")

        try:
            # Find device
            device = self.discovery.get_device_by_name(device_name)

            if not device and auto_discover:
                logger.info("[AIRPLAY MANAGER] Device not found, discovering...")
                await self.discover_devices(timeout=5.0)
                device = self.discovery.get_device_by_name(device_name)

            if not device:
                return {
                    "success": False,
                    "message": f"Device '{device_name}' not found on network",
                    "available_devices": [d.name for d in self.discovery.get_all_devices()]
                }

            logger.info(f"[AIRPLAY MANAGER] Found device: {device.name} at {device.ip_address}:{device.port}")

            # Connect with retry logic
            retry_attempts = self.config['connection']['retry_attempts']
            retry_delay = self.config['connection']['retry_delay_seconds']

            last_error = None

            for attempt in range(retry_attempts):
                if attempt > 0:
                    logger.info(f"[AIRPLAY MANAGER] Retry attempt {attempt + 1}/{retry_attempts}")
                    await asyncio.sleep(retry_delay)

                try:
                    result = await self.protocol.connect(
                        device.name,
                        device.ip_address,
                        device.port,
                        mode
                    )

                    duration = time.time() - start_time

                    # Record attempt
                    self._record_connection_attempt(
                        device.name,
                        device.ip_address,
                        device.port,
                        mode,
                        result.success,
                        duration,
                        result.method.value,
                        None if result.success else result.message
                    )

                    if result.success:
                        self.stats['successful_connections'] += 1
                        self.stats['total_connections'] += 1

                        # Update average connection time
                        total_success = self.stats['successful_connections']
                        avg_time = self.stats['avg_connection_time']
                        self.stats['avg_connection_time'] = (
                            (avg_time * (total_success - 1) + duration) / total_success
                        )

                        logger.info(f"[AIRPLAY MANAGER] ✅ Connected to {device.name} in {duration:.2f}s")
                        await self._emit_event('connection_success', device=device, result=result)

                        return {
                            "success": True,
                            "message": f"Connected to {device.name}",
                            "device_name": device.name,
                            "ip_address": device.ip_address,
                            "mode": mode,
                            "method": result.method.value,
                            "duration": duration
                        }
                    else:
                        last_error = result.message
                        logger.warning(f"[AIRPLAY MANAGER] Connection attempt failed: {result.message}")

                except Exception as e:
                    last_error = str(e)
                    logger.error(f"[AIRPLAY MANAGER] Connection error: {e}")

            # All attempts failed
            duration = time.time() - start_time
            self.stats['failed_connections'] += 1
            self.stats['total_connections'] += 1

            self._record_connection_attempt(
                device.name,
                device.ip_address,
                device.port,
                mode,
                False,
                duration,
                "unknown",
                last_error
            )

            logger.error(f"[AIRPLAY MANAGER] ❌ Failed to connect after {retry_attempts} attempts")
            await self._emit_event('connection_failed', device=device, error=last_error)

            return {
                "success": False,
                "message": f"Connection failed after {retry_attempts} attempts: {last_error}",
                "device_name": device.name,
                "attempts": retry_attempts,
                "error": last_error
            }

        except Exception as e:
            logger.error(f"[AIRPLAY MANAGER] Connection error: {e}", exc_info=True)
            await self._emit_event('error', error=e)

            return {
                "success": False,
                "message": f"Connection error: {str(e)}",
                "error": str(e)
            }

    async def disconnect_from_device(self, device_name: str) -> Dict[str, Any]:
        """
        Disconnect from AirPlay device

        Args:
            device_name: Name of device to disconnect from

        Returns:
            Disconnect result dictionary
        """
        logger.info(f"[AIRPLAY MANAGER] Disconnecting from '{device_name}'")

        try:
            # Find device
            device = self.discovery.get_device_by_name(device_name)

            if not device:
                # Try to find in active sessions
                for session in self.protocol.active_sessions.values():
                    if session['device_name'].lower() == device_name.lower():
                        device_name = session['device_name']
                        ip_address = session['ip_address']
                        break
                else:
                    return {
                        "success": False,
                        "message": f"Device '{device_name}' not found"
                    }
            else:
                ip_address = device.ip_address

            # Disconnect
            success = await self.protocol.disconnect(device_name, ip_address)

            if success:
                logger.info(f"[AIRPLAY MANAGER] ✅ Disconnected from {device_name}")
                return {
                    "success": True,
                    "message": f"Disconnected from {device_name}"
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to disconnect from {device_name}"
                }

        except Exception as e:
            logger.error(f"[AIRPLAY MANAGER] Disconnect error: {e}")
            return {
                "success": False,
                "message": f"Disconnect error: {str(e)}",
                "error": str(e)
            }

    def _record_connection_attempt(
        self,
        device_name: str,
        ip_address: str,
        port: int,
        mode: str,
        success: bool,
        duration: float,
        method: str,
        error: Optional[str]
    ):
        """Record connection attempt for analysis"""
        attempt = ConnectionAttempt(
            device_name=device_name,
            ip_address=ip_address,
            port=port,
            mode=mode,
            timestamp=datetime.now(),
            success=success,
            duration=duration,
            method=method,
            error=error
        )

        self.connection_history.append(attempt)

        # Keep only recent history (last 100 attempts)
        if len(self.connection_history) > 100:
            self.connection_history = self.connection_history[-100:]

    def get_available_devices(self) -> List[Dict[str, Any]]:
        """Get list of available devices"""
        devices = self.discovery.get_all_devices(available_only=True)
        return [
            {
                'name': d.name,
                'ip_address': d.ip_address,
                'port': d.port,
                'model': d.model,
                'features': d.features,
                'is_connected': self.protocol.is_connected(d.name, d.ip_address)
            }
            for d in devices
        ]

    def get_active_connections(self) -> List[Dict[str, Any]]:
        """Get list of active connections"""
        return [
            {
                'device_name': session['device_name'],
                'ip_address': session['ip_address'],
                'port': session['port'],
                'mode': session['mode'],
                'method': session['method'].value,
                'connected_at': session['connected_at'].isoformat(),
                'duration': session['duration']
            }
            for session in self.protocol.active_sessions.values()
        ]

    def get_connection_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent connection history"""
        history = self.connection_history[-limit:]
        return [
            {
                'device_name': attempt.device_name,
                'ip_address': attempt.ip_address,
                'mode': attempt.mode,
                'timestamp': attempt.timestamp.isoformat(),
                'success': attempt.success,
                'duration': attempt.duration,
                'method': attempt.method,
                'error': attempt.error
            }
            for attempt in history
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        uptime = (datetime.now() - self.stats['uptime_start']).total_seconds()

        discovery_stats = self.discovery.get_stats()
        protocol_stats = self.protocol.get_stats()

        return {
            'manager': {
                **self.stats,
                'uptime_seconds': uptime,
                'is_initialized': self.is_initialized,
                'history_size': len(self.connection_history)
            },
            'discovery': discovery_stats,
            'protocol': protocol_stats
        }

    def is_connected_to(self, device_name: str) -> bool:
        """Check if connected to specific device"""
        device = self.discovery.get_device_by_name(device_name)
        if not device:
            # Check active sessions
            for session in self.protocol.active_sessions.values():
                if session['device_name'].lower() == device_name.lower():
                    return True
            return False

        return self.protocol.is_connected(device.name, device.ip_address)


# Singleton instance
_airplay_manager: Optional[AirPlayManager] = None


def get_airplay_manager(config_path: Optional[str] = None) -> AirPlayManager:
    """Get singleton AirPlay manager"""
    global _airplay_manager
    if _airplay_manager is None:
        _airplay_manager = AirPlayManager(config_path)
    return _airplay_manager


if __name__ == "__main__":
    # Test AirPlay manager
    async def test():
        logging.basicConfig(level=logging.INFO)

        manager = get_airplay_manager()

        # Initialize
        await manager.initialize()

        # Discover devices
        print("\n=== Discovering AirPlay devices ===")
        devices = await manager.discover_devices(timeout=10.0)

        print(f"\nFound {len(devices)} devices:")
        for device in devices:
            print(f"  - {device.name} ({device.ip_address}:{device.port})")

        if devices:
            # Try connecting to first device
            device = devices[0]
            print(f"\n=== Connecting to {device.name} ===")

            result = await manager.connect_to_device(device.name, mode="extend")

            print(f"\nConnection result:")
            print(f"  Success: {result['success']}")
            print(f"  Message: {result['message']}")

            if result['success']:
                # Wait a bit
                print("\nWaiting 5 seconds...")
                await asyncio.sleep(5)

                # Disconnect
                print(f"\n=== Disconnecting from {device.name} ===")
                disconnect_result = await manager.disconnect_from_device(device.name)
                print(f"  Success: {disconnect_result['success']}")
                print(f"  Message: {disconnect_result['message']}")

        # Get stats
        print("\n=== Statistics ===")
        stats = manager.get_stats()
        print(json.dumps(stats, indent=2, default=str))

        # Shutdown
        await manager.shutdown()

    asyncio.run(test())
