"""
Unified Weather App Vision Workflow
Simplified integration with the unified weather system
"""

import logging
from typing import Dict, Any

from backend.system_control.unified_vision_weather import get_unified_weather_system

logger = logging.getLogger(__name__)


class UnifiedWeatherWorkflow:
    """Simplified weather workflow using unified system"""
    
    def __init__(self, controller, vision_handler):
        self.controller = controller
        self.vision_handler = vision_handler
        self.weather_system = get_unified_weather_system(vision_handler, controller)
    
    async def check_weather(self, query: str = "") -> Dict[str, Any]:
        """
        Check weather using the unified vision system
        
        Args:
            query: Natural language weather query
            
        Returns:
            Dict with weather information
        """
        try:
            # Let the unified system handle everything
            logger.info(f"[WEATHER WORKFLOW] Calling weather_system.get_weather with query: {query}")
            result = await self.weather_system.get_weather(query)
            logger.info(f"[WEATHER WORKFLOW] weather_system returned: {result}")
            
            # Return in workflow format
            return {
                'success': result.get('success', False),
                'message': result.get('formatted_response', 'Unable to get weather'),
                'source': 'unified_vision_weather',
                'data': result.get('data', {})
            }
            
        except Exception as e:
            logger.error(f"[WEATHER WORKFLOW] Weather workflow error: {e}", exc_info=True)
            return {
                'success': False,
                'message': f"I encountered an error checking the weather: {str(e)}",
                'source': 'error'
            }


# Integration function for compatibility
async def execute_weather_app_workflow(controller, vision_handler, query: str = "") -> str:
    """Execute the unified weather workflow and return formatted response"""
    logger.info(f"[WEATHER WORKFLOW] Starting workflow with query: {query}")
    logger.info(f"[WEATHER WORKFLOW] Controller: {controller}, Vision: {vision_handler}")
    
    workflow = UnifiedWeatherWorkflow(controller, vision_handler)
    
    # If no query provided, default to current weather
    if not query:
        query = "What's the weather today?"
    
    logger.info("[WEATHER WORKFLOW] Calling workflow.check_weather...")
    result = await workflow.check_weather(query)
    logger.info(f"[WEATHER WORKFLOW] Result: {result}")
    
    # Return just the message for compatibility
    message = result['message']
    logger.info(f"[WEATHER WORKFLOW] Returning message: {message[:100]}...")
    return message