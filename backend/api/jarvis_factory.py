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

def get_app_state():
    """Get the app state for accessing shared resources"""
    return _app_state

def get_vision_analyzer():
    """Get vision analyzer from app state if available"""
    logger.info(f"[JARVIS FACTORY] Getting vision analyzer - app_state exists: {_app_state is not None}")
    if _app_state and hasattr(_app_state, 'vision_analyzer'):
        logger.info(f"[JARVIS FACTORY] Found vision analyzer in app.state: {_app_state.vision_analyzer}")
        logger.info(f"[JARVIS FACTORY] Vision analyzer type: {type(_app_state.vision_analyzer).__name__}")
        logger.info(f"[JARVIS FACTORY] Vision analyzer ID: {id(_app_state.vision_analyzer)}")
        return _app_state.vision_analyzer
    logger.warning("[JARVIS FACTORY] No vision analyzer available in app.state")
    return None

def create_jarvis_agent(user_name: str = "Sir"):
    """Create JARVIS agent with proper dependencies"""
    from voice.jarvis_agent_voice import JARVISAgentVoice
    
    logger.info(f"[JARVIS FACTORY] Creating JARVIS agent for user: {user_name}")
    vision_analyzer = get_vision_analyzer()
    logger.info(f"[JARVIS FACTORY] Creating JARVIS agent with vision_analyzer: {vision_analyzer is not None}")
    
    agent = JARVISAgentVoice(user_name=user_name, vision_analyzer=vision_analyzer)
    
    if vision_analyzer:
        logger.info("[JARVIS FACTORY] Created JARVIS agent with shared vision analyzer")
        # Verify the chatbot has the analyzer
        if hasattr(agent, 'claude_chatbot') and agent.claude_chatbot:
            logger.info(f"[JARVIS FACTORY] Chatbot vision analyzer: {agent.claude_chatbot.vision_analyzer is not None}")
            logger.info(f"[JARVIS FACTORY] Chatbot vision analyzer ID: {id(agent.claude_chatbot.vision_analyzer) if agent.claude_chatbot.vision_analyzer else None}")
    else:
        logger.info("[JARVIS FACTORY] Created JARVIS agent with independent vision analyzer")
    
    return agent