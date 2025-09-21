#!/usr/bin/env python3
"""
Vision Query Bypass Module
Ensures vision queries go directly to Claude's vision API without command interpretation
"""

import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class VisionQueryBypass:
    """Handles direct vision queries without action mapping"""
    
    @staticmethod
    def should_bypass_interpretation(query: str) -> bool:
        """
        Determine if a query should bypass command interpretation
        and go directly to vision analysis
        """
        query_lower = query.lower()
        
        # First check if it's definitely a command (not a question)
        command_verbs = ["open", "close", "launch", "quit", "start", "stop", "kill", "exit", "terminate"]
        is_command = any(query_lower.startswith(verb + " ") for verb in command_verbs)
        
        if is_command:
            # This is definitely a command, not a vision query
            return False
        
        # Patterns that indicate questions about screen content
        # These should NEVER go through command interpretation
        vision_query_patterns = [
            # Questions about quantities/counts
            "how many",
            "count",
            "number of",
            
            # Questions about presence/existence
            "do i have",
            "is there",
            "are there",
            "can you see",
            "what do you see",
            
            # Questions about status/state  
            "what's running",
            "what is running",
            "show me",
            "tell me about",
            
            # Questions about specific UI elements
            "windows",
            "tabs",
            "notifications",
            "messages",
            "alerts",
            
            # General screen queries
            "on my screen",
            "on the screen",
            "currently visible",
            "what's visible",
            "what is visible",
            
            # Application-specific queries
            "in chrome",
            "in safari", 
            "in firefox",
            "browser",
            "application",
        ]
        
        # Special handling for "open" in questions
        if "open" in query_lower:
            # Check if it's a question about what's open
            question_patterns = [
                "what.*open",
                "which.*open", 
                "how many.*open",
                ".*have.*open",
                ".*are open",
                ".*is open",
                "open in",  # "tabs open in chrome"
                "open on",  # "windows open on my screen"
            ]
            import re
            for pattern in question_patterns:
                if re.search(pattern, query_lower):
                    logger.info(f"Vision query with 'open' detected (pattern: '{pattern}'): {query}")
                    return True
        
        # Check if query matches vision patterns
        for pattern in vision_query_patterns:
            if pattern in query_lower:
                logger.info(f"Vision query detected (pattern: '{pattern}'): {query}")
                return True
        
        return False
    
    @staticmethod
    def extract_vision_context(query: str) -> Dict[str, Any]:
        """
        Extract context from vision query to help with analysis
        """
        query_lower = query.lower()
        context = {
            "query_type": "general",
            "target": None,
            "application": None,
            "ui_element": None
        }
        
        # Determine query type
        if "how many" in query_lower or "count" in query_lower:
            context["query_type"] = "count"
        elif "do i have" in query_lower or "is there" in query_lower:
            context["query_type"] = "existence"
        elif "what" in query_lower:
            context["query_type"] = "description"
        
        # Extract target application
        apps = ["chrome", "safari", "firefox", "slack", "discord", "terminal", "vscode", "whatsapp"]
        for app in apps:
            if app in query_lower:
                context["application"] = app
                break
        
        # Extract UI element
        ui_elements = ["window", "tab", "notification", "message", "alert", "popup", "dialog"]
        for element in ui_elements:
            if element in query_lower:
                context["ui_element"] = element
                break
        
        return context


# Integration point for vision command handler
def enhance_vision_routing(original_handler):
    """
    Decorator to enhance vision command routing with bypass logic
    """
    async def wrapper(self, command: str):
        # Check if this should bypass interpretation
        if VisionQueryBypass.should_bypass_interpretation(command):
            logger.info("Bypassing command interpretation for vision query")
            # Extract context for better analysis
            context = VisionQueryBypass.extract_vision_context(command)
            
            # Route directly to screen analysis
            if hasattr(self, 'analyze_screen'):
                return await self.analyze_screen(command)
            else:
                logger.warning("analyze_screen method not found, falling back to original handler")
        
        # Otherwise use original handler
        return await original_handler(self, command)
    
    return wrapper