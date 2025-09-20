"""
Vision Status Manager - Coordinates purple indicator with WebSocket updates
"""

import asyncio
import logging
from typing import Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class VisionStatusManager:
    """Manages vision status coordination between purple indicator and WebSocket"""
    
    def __init__(self):
        self.is_connected = False
        self.websocket_manager = None
        self.status_change_callbacks = []
        self.last_status_change = None
        
    def set_websocket_manager(self, websocket_manager):
        """Set the WebSocket manager for broadcasting status updates"""
        self.websocket_manager = websocket_manager
        logger.info("WebSocket manager connected to vision status manager")
        
    def add_status_callback(self, callback: Callable):
        """Add a callback to be called when vision status changes"""
        self.status_change_callbacks.append(callback)
        
    async def update_vision_status(self, connected: bool):
        """Update vision status and notify all listeners"""
        if connected == self.is_connected:
            return  # No change
            
        self.is_connected = connected
        self.last_status_change = datetime.now()
        
        status_text = "connected" if connected else "disconnected"
        logger.info(f"🔵 Vision status changed to: {status_text.upper()}")
        
        # Notify all callbacks
        for callback in self.status_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(connected)
                else:
                    callback(connected)
            except Exception as e:
                logger.error(f"Error calling status callback: {e}")
        
        # Send WebSocket update
        await self._broadcast_status_update()
        
    async def _broadcast_status_update(self):
        """Broadcast vision status update via WebSocket"""
        if not self.websocket_manager:
            logger.debug("No WebSocket manager available for status broadcast")
            return
            
        try:
            status_message = {
                "type": "vision_status_update",
                "status": {
                    "connected": self.is_connected,
                    "text": "Vision: connected" if self.is_connected else "Vision: disconnected",
                    "color": "green" if self.is_connected else "red",
                    "indicator": "🟢" if self.is_connected else "🔴",
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Broadcast to all connected clients
            await self.websocket_manager.broadcast(status_message)
            logger.info(f"Broadcasted vision status: {status_message['status']['text']}")
            
        except Exception as e:
            logger.error(f"Failed to broadcast vision status: {e}")
            
    def get_status(self) -> dict:
        """Get current vision status"""
        return {
            "connected": self.is_connected,
            "text": "Vision: connected" if self.is_connected else "Vision: disconnected",
            "color": "green" if self.is_connected else "red",
            "indicator": "🟢" if self.is_connected else "🔴",
            "last_change": self.last_status_change.isoformat() if self.last_status_change else None
        }
        
    async def initialize_status(self):
        """Initialize and broadcast current status"""
        await self._broadcast_status_update()


# Global instance
_vision_status_manager = None


def get_vision_status_manager():
    """Get or create vision status manager instance"""
    global _vision_status_manager
    if _vision_status_manager is None:
        _vision_status_manager = VisionStatusManager()
    return _vision_status_manager