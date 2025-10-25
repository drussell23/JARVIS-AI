"""
Voice Prompt Manager for Proximity-Aware Display
=================================================

Manages voice prompts and responses for display connection decisions.
Integrates with JARVIS voice system to handle "Yes/No" responses.

Features:
- Automatic voice prompts when near displays
- Yes/No voice command handling
- Pending connection state management
- Timeout handling for unanswered prompts
- Natural language response generation

Author: Derek Russell
Date: 2025-10-14
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta
from enum import Enum

from .proximity_display_context import ConnectionDecision, ConnectionAction
from .auto_connection_manager import get_auto_connection_manager

logger = logging.getLogger(__name__)


class PromptState(Enum):
    """State of voice prompt"""
    IDLE = "idle"
    WAITING_FOR_RESPONSE = "waiting_for_response"
    ANSWERED = "answered"
    TIMEOUT = "timeout"


class VoicePromptManager:
    """
    Manages voice prompts for display connections
    
    Handles the full flow:
    1. Proximity detected → Generate voice prompt
    2. Wait for user response ("yes" / "no")
    3. Execute connection or skip
    4. Handle timeout if no response
    """
    
    def __init__(self, prompt_timeout_seconds: float = 30.0):
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.prompt_timeout_seconds = prompt_timeout_seconds
        
        # State tracking
        self.current_prompt: Optional[Dict[str, Any]] = None
        self.prompt_state: PromptState = PromptState.IDLE
        self.pending_decision: Optional[ConnectionDecision] = None
        self.prompt_timestamp: Optional[datetime] = None
        
        # Statistics
        self.total_prompts = 0
        self.accepted_prompts = 0
        self.rejected_prompts = 0
        self.timeout_prompts = 0
        
        self.logger.info("[VOICE PROMPT] Manager initialized")
    
    async def generate_prompt_for_decision(
        self,
        decision: ConnectionDecision
    ) -> Optional[str]:
        """
        Generate voice prompt for connection decision
        
        Args:
            decision: ConnectionDecision from bridge
            
        Returns:
            Voice prompt string or None
        """
        try:
            # Only prompt for PROMPT_USER actions
            if decision.action != ConnectionAction.PROMPT_USER:
                return None
            
            # Check if already waiting for response
            if self.prompt_state == PromptState.WAITING_FOR_RESPONSE:
                self.logger.debug("[VOICE PROMPT] Already waiting for response, skipping")
                return None
            
            # Generate natural language prompt
            prompt = self._generate_natural_prompt(decision)
            
            # Set pending state
            self.current_prompt = {
                "display_id": decision.display_id,
                "display_name": decision.display_name,
                "prompt": prompt,
                "decision": decision
            }
            self.pending_decision = decision
            self.prompt_state = PromptState.WAITING_FOR_RESPONSE
            self.prompt_timestamp = datetime.now()
            self.total_prompts += 1
            
            # Start timeout task
            asyncio.create_task(self._handle_prompt_timeout())
            
            self.logger.info(f"[VOICE PROMPT] Generated: {prompt}")
            return prompt
            
        except Exception as e:
            self.logger.error(f"[VOICE PROMPT] Error generating prompt: {e}")
            return None
    
    def _generate_natural_prompt(self, decision: ConnectionDecision) -> str:
        """
        Generate natural language prompt
        
        Args:
            decision: ConnectionDecision
            
        Returns:
            Natural prompt string
        """
        display_name = decision.display_name
        distance = decision.proximity_distance
        zone = decision.proximity_zone.value
        
        # Contextual prompts based on distance
        if distance < 2.0:
            return f"Sir, I see you're very close to the {display_name}. Would you like to connect?"
        elif distance < 4.0:
            return f"Sir, you're near the {display_name}, about {distance:.1f} meters away. Shall I connect?"
        else:
            return f"Sir, I detect you're in the {zone} zone with the {display_name}. Connect to this display?"
    
    async def handle_voice_response(self, response: str) -> Dict[str, Any]:
        """
        Handle user voice response ("yes" / "no")
        
        Args:
            response: User's voice command
            
        Returns:
            Response result with action taken
        """
        try:
            # Check if we're waiting for a response
            if self.prompt_state != PromptState.WAITING_FOR_RESPONSE:
                return {
                    "handled": False,
                    "reason": "No pending prompt"
                }
            
            # Check timeout
            if self._is_prompt_expired():
                self.prompt_state = PromptState.TIMEOUT
                self.timeout_prompts += 1
                return {
                    "handled": False,
                    "reason": "Prompt expired (timeout)"
                }
            
            # Parse response
            response_lower = response.lower().strip()
            
            # Affirmative responses
            if any(word in response_lower for word in ["yes", "yeah", "yep", "sure", "connect", "ok", "okay", "please"]):
                result = await self._handle_affirmative_response()
                self.accepted_prompts += 1
                return result
            
            # Negative responses
            elif any(word in response_lower for word in ["no", "nope", "don't", "skip", "cancel", "not now"]):
                result = await self._handle_negative_response()
                self.rejected_prompts += 1
                return result
            
            else:
                # Unclear response - ask again
                return {
                    "handled": True,
                    "action": "clarify",
                    "response": "Sir, I didn't quite catch that. Would you like to connect? Please say 'yes' or 'no'."
                }
                
        except Exception as e:
            self.logger.error(f"[VOICE PROMPT] Error handling response: {e}")
            self._reset_prompt_state()
            return {
                "handled": False,
                "error": str(e)
            }
    
    async def _handle_affirmative_response(self) -> Dict[str, Any]:
        """
        Handle affirmative response (yes, connect)
        
        Returns:
            Connection result
        """
        try:
            self.logger.info("[VOICE PROMPT] User said YES - connecting")
            
            decision = self.pending_decision
            display_name = decision.display_name
            
            # Execute connection
            manager = get_auto_connection_manager()
            result = await manager.evaluate_and_execute(decision, force=True)
            
            # Update state
            self.prompt_state = PromptState.ANSWERED
            
            # Generate response
            if result and result.success:
                response = f"Connecting to {display_name}... Done, sir."
            else:
                response = f"I encountered an issue connecting to {display_name}. Please check the display settings."
            
            # Reset state
            self._reset_prompt_state()
            
            return {
                "handled": True,
                "action": "connect",
                "response": response,
                "connection_result": result.to_dict() if result else None
            }
            
        except Exception as e:
            self.logger.error(f"[VOICE PROMPT] Error connecting: {e}")
            self._reset_prompt_state()
            return {
                "handled": True,
                "action": "error",
                "response": f"I'm sorry sir, I encountered an error: {str(e)}"
            }
    
    async def _handle_negative_response(self) -> Dict[str, Any]:
        """
        Handle negative response (no, don't connect)
        
        Returns:
            Skip result
        """
        try:
            self.logger.info("[VOICE PROMPT] User said NO - skipping connection")
            
            decision = self.pending_decision
            display_id = decision.display_id
            
            # Register user override (prevent asking again for 5 minutes)
            manager = get_auto_connection_manager()
            manager.register_user_override(display_id)
            
            # Update state
            self.prompt_state = PromptState.ANSWERED
            
            # Generate response
            response = "Understood, sir. I won't ask again for a few minutes."
            
            # Reset state
            self._reset_prompt_state()
            
            return {
                "handled": True,
                "action": "skip",
                "response": response
            }
            
        except Exception as e:
            self.logger.error(f"[VOICE PROMPT] Error handling negative: {e}")
            self._reset_prompt_state()
            return {
                "handled": True,
                "action": "error",
                "response": "Understood, sir."
            }
    
    async def _handle_prompt_timeout(self):
        """Handle prompt timeout (no response within timeout period)"""
        try:
            # Wait for timeout duration
            await asyncio.sleep(self.prompt_timeout_seconds)
            
            # Check if still waiting
            if self.prompt_state == PromptState.WAITING_FOR_RESPONSE:
                self.logger.info("[VOICE PROMPT] Prompt timed out (no response)")
                self.prompt_state = PromptState.TIMEOUT
                self.timeout_prompts += 1
                self._reset_prompt_state()
                
        except Exception as e:
            self.logger.error(f"[VOICE PROMPT] Error in timeout handler: {e}")
    
    def _is_prompt_expired(self) -> bool:
        """Check if current prompt has expired"""
        if not self.prompt_timestamp:
            return False
        
        elapsed = (datetime.now() - self.prompt_timestamp).total_seconds()
        return elapsed > self.prompt_timeout_seconds
    
    def _reset_prompt_state(self):
        """Reset prompt state"""
        self.current_prompt = None
        self.pending_decision = None
        self.prompt_state = PromptState.IDLE
        self.prompt_timestamp = None
    
    def get_prompt_stats(self) -> Dict[str, Any]:
        """Get prompt statistics"""
        return {
            "total_prompts": self.total_prompts,
            "accepted_prompts": self.accepted_prompts,
            "rejected_prompts": self.rejected_prompts,
            "timeout_prompts": self.timeout_prompts,
            "acceptance_rate": round(self.accepted_prompts / max(self.total_prompts, 1), 3),
            "current_state": self.prompt_state.value,
            "has_pending_prompt": self.prompt_state == PromptState.WAITING_FOR_RESPONSE
        }


# Singleton instance
_voice_prompt_manager: Optional[VoicePromptManager] = None

def get_voice_prompt_manager(prompt_timeout_seconds: float = 30.0) -> VoicePromptManager:
    """Get singleton VoicePromptManager instance"""
    global _voice_prompt_manager
    if _voice_prompt_manager is None:
        _voice_prompt_manager = VoicePromptManager(prompt_timeout_seconds)
    return _voice_prompt_manager
