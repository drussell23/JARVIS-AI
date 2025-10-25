"""
Query Handler for JARVIS
Handles natural language queries using the Unified Awareness Engine
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


async def handle_query(command: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Handle a natural language query using UAE

    Args:
        command: The query text
        context: Optional context information

    Returns:
        Dict with success status and response
    """
    try:
        logger.info(f"[QUERY] Processing query: {command}")

        # Try to get UAE from app state
        from main import app
        uae_engine = getattr(app.state, 'uae_engine', None)

        if uae_engine:
            logger.info("[QUERY] Using UAE engine for query processing")
            # Process with UAE
            response = await uae_engine.process_query(command)

            return {
                "success": True,
                "response": response.get("response", "Query processed successfully"),
                "analysis": response.get("analysis", {}),
                "suggestions": response.get("suggestions", [])
            }
        else:
            logger.warning("[QUERY] UAE engine not available, using fallback")
            # Fallback response
            return {
                "success": True,
                "response": f"Received your query: '{command}'. UAE engine is currently loading...",
                "fallback": True
            }

    except Exception as e:
        logger.error(f"[QUERY] Error processing query: {e}")
        return {
            "success": False,
            "response": f"Sorry, I encountered an error: {str(e)}",
            "error": str(e)
        }
