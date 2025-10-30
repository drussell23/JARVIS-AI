"""
Unified Weather App Vision Workflow

This module provides a simplified integration with the unified weather system,
offering a streamlined workflow for weather-related queries through vision
and natural language processing.

The module contains:
- UnifiedWeatherWorkflow: Main workflow class for weather operations
- execute_weather_app_workflow: Compatibility function for legacy integration

Example:
    >>> workflow = UnifiedWeatherWorkflow(controller, vision_handler)
    >>> result = await workflow.check_weather("What's the weather today?")
    >>> print(result['message'])
"""

import logging
from typing import Dict, Any, Optional

from backend.system_control.unified_vision_weather import get_unified_weather_system

logger = logging.getLogger(__name__)


class UnifiedWeatherWorkflow:
    """
    Simplified weather workflow using unified system.
    
    This class provides a high-level interface for weather operations,
    integrating with the unified vision weather system to handle natural
    language queries and return formatted weather information.
    
    Attributes:
        controller: System controller instance for hardware/system operations
        vision_handler: Vision processing handler for image analysis
        weather_system: Unified weather system instance for weather operations
    """
    
    def __init__(self, controller: Any, vision_handler: Any) -> None:
        """
        Initialize the unified weather workflow.
        
        Args:
            controller: System controller instance for managing system operations
            vision_handler: Vision handler instance for processing visual data
        """
        self.controller = controller
        self.vision_handler = vision_handler
        self.weather_system = get_unified_weather_system(vision_handler, controller)
    
    async def check_weather(self, query: str = "") -> Dict[str, Any]:
        """
        Check weather using the unified vision system.
        
        Processes natural language weather queries through the unified system
        and returns structured weather information with success status and
        formatted response message.
        
        Args:
            query: Natural language weather query (e.g., "What's the weather today?")
                  If empty, defaults to current weather query
            
        Returns:
            Dictionary containing:
                - success (bool): Whether the weather check was successful
                - message (str): Formatted weather response for user display
                - source (str): Source of the response ('unified_vision_weather' or 'error')
                - data (dict): Raw weather data from the system
        
        Raises:
            Exception: Logs and handles any errors during weather retrieval
            
        Example:
            >>> result = await workflow.check_weather("Will it rain tomorrow?")
            >>> if result['success']:
            ...     print(result['message'])
            ... else:
            ...     print(f"Error: {result['message']}")
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


async def execute_weather_app_workflow(
    controller: Any, 
    vision_handler: Any, 
    query: str = ""
) -> str:
    """
    Execute the unified weather workflow and return formatted response.
    
    This is a compatibility function that provides a simple interface for
    executing weather workflows. It creates a workflow instance, processes
    the query, and returns just the formatted message string.
    
    Args:
        controller: System controller instance for managing system operations
        vision_handler: Vision handler instance for processing visual data
        query: Natural language weather query. Defaults to "What's the weather today?"
               if not provided
    
    Returns:
        Formatted weather response message as a string
    
    Raises:
        Exception: Any errors are caught and logged by the workflow, returning
                  an error message string instead of raising
    
    Example:
        >>> message = await execute_weather_app_workflow(
        ...     controller, vision_handler, "Is it sunny outside?"
        ... )
        >>> print(message)
        "It's currently sunny with a temperature of 75°F..."
    """
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