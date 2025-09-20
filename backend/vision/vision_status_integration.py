"""
Vision Status Integration - Connects vision status manager with WebSocket
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def initialize_vision_status(app) -> bool:
    """Initialize vision status manager and connect to WebSocket"""
    try:
        # Import vision status manager
        from .vision_status_manager import get_vision_status_manager
        
        # Import WebSocket manager
        try:
            from api.unified_websocket import ws_manager
        except ImportError:
            logger.warning("Unified WebSocket manager not available")
            return False
            
        # Get vision status manager instance
        status_manager = get_vision_status_manager()
        
        # Connect WebSocket manager
        status_manager.set_websocket_manager(ws_manager)
        logger.info("✅ Vision status manager connected to WebSocket")
        
        # Also try to connect to vision WebSocket manager
        try:
            from api.vision_api import ws_manager as vision_ws_manager
            # Add a secondary callback for vision WebSocket
            async def vision_ws_callback(connected: bool):
                await vision_ws_manager.broadcast({
                    "type": "vision_status_update",
                    "status": {
                        "connected": connected,
                        "text": "Vision: connected" if connected else "Vision: disconnected",
                        "color": "green" if connected else "red",
                        "indicator": "🟢" if connected else "🔴",
                        "timestamp": datetime.now().isoformat()
                    }
                })
            status_manager.add_status_callback(vision_ws_callback)
            logger.info("✅ Vision status also broadcasting to vision WebSocket")
        except Exception as e:
            logger.debug(f"Could not connect to vision WebSocket manager: {e}")
        
        # Store in app state for access
        app.state.vision_status_manager = status_manager
        
        # Initialize status broadcast
        await status_manager.initialize_status()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize vision status integration: {e}")
        return False


def setup_vision_status_callbacks(app):
    """Set up callbacks for vision status changes"""
    try:
        if not hasattr(app.state, 'vision_status_manager'):
            logger.warning("Vision status manager not initialized")
            return
            
        status_manager = app.state.vision_status_manager
        
        # Add callback to update JARVIS UI
        async def jarvis_ui_callback(connected: bool):
            """Update JARVIS UI with vision status"""
            try:
                # If JARVIS API is available, update UI
                if hasattr(app.state, 'jarvis_api'):
                    jarvis_api = app.state.jarvis_api
                    status_text = "Vision connected" if connected else "Vision disconnected"
                    
                    # Send status update through JARVIS
                    if hasattr(jarvis_api, 'update_ui_status'):
                        await jarvis_api.update_ui_status({
                            'vision': {
                                'connected': connected,
                                'text': status_text,
                                'color': 'green' if connected else 'red'
                            }
                        })
                    
            except Exception as e:
                logger.error(f"Error updating JARVIS UI: {e}")
                
        status_manager.add_status_callback(jarvis_ui_callback)
        logger.info("Vision status callbacks configured")
        
    except Exception as e:
        logger.error(f"Failed to setup vision status callbacks: {e}")


async def get_vision_status(app) -> dict:
    """Get current vision status"""
    try:
        if hasattr(app.state, 'vision_status_manager'):
            return app.state.vision_status_manager.get_status()
        else:
            return {
                "connected": False,
                "text": "Vision: disconnected",
                "color": "red",
                "indicator": "🔴",
                "error": "Vision status manager not initialized"
            }
    except Exception as e:
        logger.error(f"Error getting vision status: {e}")
        return {
            "connected": False,
            "text": "Vision: error",
            "color": "red",
            "indicator": "🔴",
            "error": str(e)
        }