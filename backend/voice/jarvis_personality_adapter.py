"""
JARVIS Personality Adapter for AI Agent
Provides compatibility layer for personality-based methods
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class PersonalityAdapter:
    """Adapter to provide personality interface for JARVISAgentVoice"""
    
    def __init__(self, agent):
        self.agent = agent
        self.user_preferences = {
            'name': agent.user_name,
            'humor_level': 'moderate',
            'work_hours': (9, 17),
            'break_reminder': True,
            'location': 'Unknown'
        }
        self.context = []
        
    async def process_voice_command(self, voice_command):
        """Process voice command through agent"""
        response = await self.agent.process_voice_input(voice_command.raw_text)
        return response
        
    def _get_context_info(self) -> Dict[str, Any]:
        """Get context information"""
        return {
            'time_of_day': self._get_time_of_day(),
            'user_name': self.user_preferences['name'],
            'last_interaction': datetime.now().isoformat(),
            'mode': getattr(self.agent, 'command_mode', 'conversation')
        }
        
    def _get_time_of_day(self) -> str:
        """Get current time of day description"""
        hour = datetime.now().hour
        if hour < 12:
            return "morning"
        elif hour < 17:
            return "afternoon"
        else:
            return "evening"
            
    def update_preference(self, key: str, value: Any):
        """Update user preference"""
        self.user_preferences[key] = value
        if key == 'name':
            self.agent.user_name = value