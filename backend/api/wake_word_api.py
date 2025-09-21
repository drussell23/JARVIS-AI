"""
Wake Word API
=============

REST and WebSocket API endpoints for wake word detection.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import Dict, Any, Optional
import logging
import json

from wake_word import WakeWordService, get_config
from wake_word.services.wake_service import WakeWordAPI

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/wake-word", tags=["wake_word"])

# Global service instance (initialized from main app)
wake_service: Optional[WakeWordService] = None
wake_api: Optional[WakeWordAPI] = None


def initialize_wake_word(activation_callback=None):
    """Initialize wake word service"""
    global wake_service, wake_api
    
    try:
        config = get_config()
        
        if not config.enabled:
            logger.info("Wake word detection disabled in configuration")
            return False
        
        wake_service = WakeWordService()
        wake_api = WakeWordAPI(wake_service)
        
        logger.info("Wake word API initialized")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize wake word API: {e}")
        return False


@router.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get wake word service status"""
    if not wake_api:
        raise HTTPException(status_code=503, detail="Wake word service not available")
    
    return await wake_api.get_status()


@router.post("/enable")
async def enable_wake_word() -> Dict[str, Any]:
    """Enable wake word detection"""
    if not wake_api:
        raise HTTPException(status_code=503, detail="Wake word service not available")
    
    return await wake_api.enable()


@router.post("/disable")
async def disable_wake_word() -> Dict[str, Any]:
    """Disable wake word detection"""
    if not wake_api:
        raise HTTPException(status_code=503, detail="Wake word service not available")
    
    return await wake_api.disable()


@router.post("/test")
async def test_activation() -> Dict[str, Any]:
    """Test wake word activation response"""
    if not wake_api:
        raise HTTPException(status_code=503, detail="Wake word service not available")
    
    return await wake_api.test_activation()


@router.put("/settings")
async def update_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Update wake word settings"""
    if not wake_api:
        raise HTTPException(status_code=503, detail="Wake word service not available")
    
    # Validate settings
    valid_settings = {}
    
    if 'wake_words' in settings:
        if isinstance(settings['wake_words'], list):
            valid_settings['wake_words'] = settings['wake_words']
        else:
            raise HTTPException(status_code=400, detail="wake_words must be a list")
    
    if 'sensitivity' in settings:
        if settings['sensitivity'] in ['very_low', 'low', 'medium', 'high', 'very_high']:
            valid_settings['sensitivity'] = settings['sensitivity']
        else:
            raise HTTPException(status_code=400, detail="Invalid sensitivity level")
    
    if 'activation_responses' in settings:
        if isinstance(settings['activation_responses'], list):
            valid_settings['activation_responses'] = settings['activation_responses']
    
    return await wake_api.update_settings(valid_settings)


@router.get("/config")
async def get_configuration() -> Dict[str, Any]:
    """Get current wake word configuration"""
    config = get_config()
    return config.to_dict()


@router.post("/feedback/false-positive")
async def report_false_positive() -> Dict[str, Any]:
    """Report a false positive detection"""
    if not wake_service:
        raise HTTPException(status_code=503, detail="Wake word service not available")
    
    wake_service.report_false_positive()
    return {
        'success': True,
        'message': 'False positive reported. Adjusting sensitivity.'
    }


@router.websocket("/stream")
async def wake_word_websocket(websocket: WebSocket):
    """WebSocket for real-time wake word events"""
    await websocket.accept()
    
    if not wake_service:
        await websocket.send_json({
            'type': 'error',
            'message': 'Wake word service not available'
        })
        await websocket.close()
        return
    
    # Subscribe to wake word events
    async def send_event(event: Dict[str, Any]):
        """Send event to WebSocket client"""
        try:
            await websocket.send_json(event)
        except Exception as e:
            logger.error(f"Failed to send WebSocket event: {e}")
    
    # Set up event callback
    original_callback = wake_service.activation_callback
    wake_service.activation_callback = send_event
    
    try:
        while True:
            # Receive messages from client
            try:
                message = await websocket.receive_json()
                
                if message.get('type') == 'ping':
                    await websocket.send_json({'type': 'pong'})
                
                elif message.get('type') == 'command_received':
                    await wake_service.handle_command_received()
                
                elif message.get('type') == 'command_complete':
                    await wake_service.handle_command_complete()
                
            except json.JSONDecodeError:
                await websocket.send_json({
                    'type': 'error',
                    'message': 'Invalid JSON message'
                })
                
    except WebSocketDisconnect:
        logger.info("Wake word WebSocket disconnected")
    finally:
        # Restore original callback
        wake_service.activation_callback = original_callback


@router.get("/statistics")
async def get_statistics() -> Dict[str, Any]:
    """Get detection statistics"""
    if not wake_service or not wake_service.detector:
        raise HTTPException(status_code=503, detail="Wake word service not available")
    
    stats = wake_service.detector.get_statistics()
    
    # Add service-level stats
    stats['service'] = {
        'state': wake_service.state,
        'activation_history_count': len(wake_service.activation_history),
        'config': {
            'wake_words': wake_service.config.detection.wake_words,
            'sensitivity': wake_service.config.detection.sensitivity,
            'engine': wake_service.config.detection.engine
        }
    }
    
    return stats


# Export for main app
__all__ = ['router', 'initialize_wake_word', 'wake_service']