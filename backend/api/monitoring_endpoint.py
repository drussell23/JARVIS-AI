"""
Direct monitoring endpoint for simple start/stop commands
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitor", tags=["monitoring"])


class MonitoringCommand(BaseModel):
    action: str  # "start" or "stop"


@router.post("/control")
async def control_monitoring(command: MonitoringCommand) -> Dict[str, Any]:
    """Direct endpoint for monitoring control"""
    logger.info(f"[MONITOR] Direct monitoring command: {command.action}")
    
    try:
        from .vision_command_handler import vision_command_handler
        
        # Map action to command text
        if command.action == "start":
            cmd_text = "start monitoring my screen"
        elif command.action == "stop":
            cmd_text = "stop monitoring"
        else:
            raise HTTPException(status_code=400, detail=f"Invalid action: {command.action}")
        
        # Initialize if needed
        if not vision_command_handler.intelligence:
            logger.info("[MONITOR] Initializing vision handler...")
            await vision_command_handler.initialize_intelligence()
        
        # Handle the command
        result = await vision_command_handler.handle_command(cmd_text)
        
        if result.get('handled'):
            return {
                "success": True,
                "action": command.action,
                "response": result['response'],
                "monitoring_active": result.get('monitoring_active', False)
            }
        else:
            return {
                "success": False,
                "action": command.action,
                "response": "Failed to handle monitoring command",
                "monitoring_active": False
            }
            
    except Exception as e:
        logger.error(f"[MONITOR] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_monitoring_status() -> Dict[str, Any]:
    """Get current monitoring status"""
    try:
        from .vision_command_handler import vision_command_handler
        
        return {
            "monitoring_active": vision_command_handler.monitoring_active,
            "intelligence_initialized": vision_command_handler.intelligence is not None
        }
    except Exception as e:
        logger.error(f"[MONITOR] Status error: {e}")
        return {
            "monitoring_active": False,
            "intelligence_initialized": False,
            "error": str(e)
        }