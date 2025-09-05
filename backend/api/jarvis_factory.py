"""
Factory for creating JARVIS instances with proper dependency injection
"""
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Global reference to app state
_app_state: Optional[Any] = None

def set_app_state(app_state: Any):
    """Set the app state for dependency injection"""
    global _app_state
    _app_state = app_state
    logger.info("App state set for JARVIS factory")

def get_vision_analyzer():
    """Get vision analyzer from app state if available"""
    if _app_state and hasattr(_app_state, 'vision_analyzer'):
        logger.info("Using vision analyzer from app.state")
        return _app_state.vision_analyzer
    logger.warning("No vision analyzer available in app.state")
    return None

def create_jarvis_agent(user_name: str = "Sir"):
    """Create JARVIS agent with proper dependencies"""
    from voice.jarvis_agent_voice import JARVISAgentVoice
    
    vision_analyzer = get_vision_analyzer()
    logger.info(f"Creating JARVIS agent with vision_analyzer: {vision_analyzer is not None}")
    
    agent = JARVISAgentVoice(user_name=user_name, vision_analyzer=vision_analyzer)
    
    if vision_analyzer:
        logger.info("Created JARVIS agent with shared vision analyzer")
        # Verify the chatbot has the analyzer
        if hasattr(agent, 'claude_chatbot') and agent.claude_chatbot:
            logger.info(f"Chatbot vision analyzer: {agent.claude_chatbot.vision_analyzer is not None}")
    else:
        logger.info("Created JARVIS agent with independent vision analyzer")
    
    return agent