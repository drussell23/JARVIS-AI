"""
Advanced Async HUD Connection Manager
Ensures robust WebSocket connections between HUD and backend
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import weakref
from collections import deque
import time

logger = logging.getLogger(__name__)


class HUDConnectionManager:
    """
    Sophisticated async connection manager for HUD clients
    Features:
    - Automatic reconnection
    - Message buffering and replay
    - Health monitoring
    - Progress streaming
    - Dynamic state synchronization
    """

    def __init__(self, max_buffer_size: int = 1000):
        self.hud_clients: Dict[str, 'HUDClient'] = {}
        self.message_buffer = deque(maxlen=max_buffer_size)
        self.connection_callbacks: List[Callable] = []
        self.disconnection_callbacks: List[Callable] = []
        self.health_check_interval = 5  # seconds
        self.reconnect_interval = 2  # seconds
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._is_running = False
        self._lock = asyncio.Lock()
        self._startup_time = time.time()
        self._total_messages_sent = 0
        self._total_messages_buffered = 0

    async def start(self):
        """Start the connection manager"""
        if self._is_running:
            return

        self._is_running = True
        self._health_monitor_task = asyncio.create_task(self._health_monitor())
        logger.info("🚀 HUD Connection Manager started")

    async def stop(self):
        """Stop the connection manager"""
        self._is_running = False

        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        for client_id in list(self.hud_clients.keys()):
            await self.disconnect_client(client_id)

        logger.info("🛑 HUD Connection Manager stopped")

    async def register_client(
        self,
        client_id: str,
        websocket: Any,
        client_info: Optional[Dict[str, Any]] = None
    ) -> 'HUDClient':
        """Register a new HUD client"""
        async with self._lock:
            # Create client instance
            client = HUDClient(
                client_id=client_id,
                websocket=websocket,
                info=client_info or {},
                manager=self
            )

            # Store client
            self.hud_clients[client_id] = client

            # Run connection callbacks
            for callback in self.connection_callbacks:
                try:
                    await callback(client)
                except Exception as e:
                    logger.error(f"Connection callback error: {e}")

            logger.info(f"✅ HUD client registered: {client_id}")
            logger.info(f"   Active HUD clients: {len(self.hud_clients)}")

            # Replay buffered messages
            await self._replay_buffer(client)

            return client

    async def disconnect_client(self, client_id: str):
        """Disconnect a HUD client"""
        async with self._lock:
            if client_id not in self.hud_clients:
                return

            client = self.hud_clients[client_id]

            # Run disconnection callbacks
            for callback in self.disconnection_callbacks:
                try:
                    await callback(client)
                except Exception as e:
                    logger.error(f"Disconnection callback error: {e}")

            # Close WebSocket
            try:
                await client.close()
            except Exception as e:
                logger.error(f"Error closing client {client_id}: {e}")

            # Remove from registry
            del self.hud_clients[client_id]

            logger.info(f"🔌 HUD client disconnected: {client_id}")
            logger.info(f"   Active HUD clients: {len(self.hud_clients)}")

    async def broadcast_message(
        self,
        message_type: str,
        data: Dict[str, Any],
        buffer_if_no_clients: bool = True
    ):
        """Broadcast message to all HUD clients"""
        message = {
            "type": message_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }

        if not self.hud_clients:
            if buffer_if_no_clients:
                self._buffer_message(message)
                self._total_messages_buffered += 1
                logger.debug(f"📦 Buffered {message_type} message (no clients)")
            return

        # Send to all clients
        disconnected_clients = []

        for client_id, client in self.hud_clients.items():
            try:
                await client.send_message(message)
                self._total_messages_sent += 1
            except Exception as e:
                logger.error(f"Failed to send to {client_id}: {e}")
                disconnected_clients.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect_client(client_id)

    async def send_progress_update(
        self,
        percentage: int,
        status: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Send progress update to HUD clients"""
        data = {
            "percentage": percentage,
            "status": status,
            "details": details or {}
        }

        await self.broadcast_message("loading_progress", data)

        # Log significant milestones
        if percentage in [0, 25, 50, 75, 100]:
            logger.info(f"📊 HUD Progress: {percentage}% - {status}")

    async def send_loading_complete(self):
        """Send loading complete signal"""
        await self.broadcast_message("loading_complete", {
            "message": "System ready",
            "startup_time": time.time() - self._startup_time
        })
        logger.info("✅ HUD Loading Complete sent")

    async def send_system_state(self, state: Dict[str, Any]):
        """Send system state update"""
        await self.broadcast_message("system_state", state)

    def _buffer_message(self, message: Dict[str, Any]):
        """Buffer a message for later replay"""
        self.message_buffer.append(message)

    async def _replay_buffer(self, client: 'HUDClient'):
        """Replay buffered messages to a new client"""
        if not self.message_buffer:
            return

        logger.info(f"📼 Replaying {len(self.message_buffer)} buffered messages to {client.client_id}")

        for message in self.message_buffer:
            try:
                await client.send_message(message)
            except Exception as e:
                logger.error(f"Failed to replay message: {e}")
                break

        # Clear buffer after successful replay
        self.message_buffer.clear()

    async def _health_monitor(self):
        """Monitor health of connected clients"""
        while self._is_running:
            try:
                await asyncio.sleep(self.health_check_interval)

                if not self.hud_clients:
                    continue

                # Ping all clients
                disconnected = []
                for client_id, client in self.hud_clients.items():
                    try:
                        await client.ping()
                    except Exception:
                        disconnected.append(client_id)

                # Remove disconnected clients
                for client_id in disconnected:
                    logger.warning(f"⚠️ Client {client_id} failed health check")
                    await self.disconnect_client(client_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

    def on_connect(self, callback: Callable):
        """Register connection callback"""
        self.connection_callbacks.append(callback)

    def on_disconnect(self, callback: Callable):
        """Register disconnection callback"""
        self.disconnection_callbacks.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get connection manager statistics"""
        return {
            "active_clients": len(self.hud_clients),
            "buffered_messages": len(self.message_buffer),
            "total_sent": self._total_messages_sent,
            "total_buffered": self._total_messages_buffered,
            "uptime": time.time() - self._startup_time,
            "clients": {
                client_id: client.get_info()
                for client_id, client in self.hud_clients.items()
            }
        }


class HUDClient:
    """Represents a connected HUD client"""

    def __init__(
        self,
        client_id: str,
        websocket: Any,
        info: Dict[str, Any],
        manager: 'HUDConnectionManager'
    ):
        self.client_id = client_id
        self.websocket = websocket
        self.info = info
        self.manager = weakref.ref(manager)
        self.connected_at = datetime.now()
        self.last_activity = datetime.now()
        self.messages_sent = 0
        self.messages_received = 0
        self._is_connected = True

    async def send_message(self, message: Dict[str, Any]):
        """Send message to this client"""
        if not self._is_connected:
            raise ConnectionError(f"Client {self.client_id} is disconnected")

        try:
            await self.websocket.send_json(message)
            self.messages_sent += 1
            self.last_activity = datetime.now()
        except Exception as e:
            self._is_connected = False
            raise ConnectionError(f"Failed to send to {self.client_id}: {e}")

    async def send_text(self, text: str):
        """Send text message"""
        if not self._is_connected:
            raise ConnectionError(f"Client {self.client_id} is disconnected")

        try:
            await self.websocket.send_text(text)
            self.messages_sent += 1
            self.last_activity = datetime.now()
        except Exception as e:
            self._is_connected = False
            raise ConnectionError(f"Failed to send to {self.client_id}: {e}")

    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive message from client"""
        if not self._is_connected:
            return None

        try:
            message = await self.websocket.receive_json()
            self.messages_received += 1
            self.last_activity = datetime.now()
            return message
        except Exception as e:
            self._is_connected = False
            logger.error(f"Failed to receive from {self.client_id}: {e}")
            return None

    async def ping(self):
        """Send ping to check connection"""
        await self.send_message({"type": "ping"})

    async def close(self):
        """Close the client connection"""
        self._is_connected = False
        try:
            await self.websocket.close()
        except Exception:
            pass

    def get_info(self) -> Dict[str, Any]:
        """Get client information"""
        return {
            "id": self.client_id,
            "connected_at": self.connected_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "is_connected": self._is_connected,
            **self.info
        }


# Global singleton instance
_manager_instance: Optional[HUDConnectionManager] = None


def get_hud_manager() -> HUDConnectionManager:
    """Get the global HUD connection manager instance"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = HUDConnectionManager()
    return _manager_instance


async def initialize_hud_manager():
    """Initialize and start the HUD connection manager"""
    manager = get_hud_manager()
    await manager.start()
    logger.info("✅ HUD Connection Manager initialized")
    return manager


async def shutdown_hud_manager():
    """Shutdown the HUD connection manager"""
    manager = get_hud_manager()
    await manager.stop()
    logger.info("🛑 HUD Connection Manager shut down")