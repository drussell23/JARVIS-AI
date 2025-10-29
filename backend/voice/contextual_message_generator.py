"""
Contextual Message Generator for JARVIS
Generates natural, context-aware responses for various system states

Uses LLM to create conversational messages that explain:
- Current state (screen locked, app closed, etc.)
- Actions being taken (unlocking screen, opening app, etc.)
- Expected outcomes (what will happen next)
"""

import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ContextualMessageGenerator:
    """
    Generates contextual messages for JARVIS responses

    Uses LLM for natural language generation
    """

    def __init__(self, llm_client=None):
        """
        Initialize message generator

        Args:
            llm_client: LLM client instance (optional, will use Qwen if available)
        """
        self.llm_client = llm_client
        self.initialized = False

        # Message templates (fallback if LLM not available)
        self.templates = {
            "screen_locked": "Good to see you, {speaker_name}. Your screen is locked. Let me unlock it to {action}.",
            "app_not_running": "I need to open {app_name} first to {action}, Sir.",
            "permission_required": "I need your permission to {action}, {speaker_name}.",
            "verification_needed": "For security, I need to verify your identity before I can {action}.",
            "action_in_progress": "I'm {action} for you now, Sir.",
            "action_complete": "I've {action} for you, Sir.",
            "action_failed": "I apologize, Sir. I couldn't {action}. {error_reason}",
            "greeting": "Good {time_of_day}, {speaker_name}. How may I assist you?",
        }

    async def initialize(self):
        """Initialize LLM client if not provided"""
        if self.initialized:
            return

        # Try to import LLM client
        if self.llm_client is None:
            try:
                from api.ml_model_manager import MLModelManager

                model_manager = MLModelManager.get_instance()
                if model_manager and hasattr(model_manager, "generate_text"):
                    self.llm_client = model_manager
                    logger.info("✅ Using MLModelManager for contextual messages")
            except Exception as e:
                logger.warning(f"LLM not available, using templates: {e}")

        self.initialized = True

    async def generate_screen_unlock_message(
        self,
        speaker_name: str,
        original_command: str,
        is_owner: bool = True,
    ) -> str:
        """
        Generate message for screen unlock scenario

        Args:
            speaker_name: Name of the speaker
            original_command: The command user wants to execute
            is_owner: Whether speaker is the device owner

        Returns:
            Contextual message
        """
        if not self.initialized:
            await self.initialize()

        context = {
            "state": "screen_locked",
            "speaker_name": speaker_name,
            "command": original_command,
            "is_owner": is_owner,
            "action": "unlock and execute your command",
        }

        if self.llm_client and hasattr(self.llm_client, "generate_text"):
            try:
                prompt = f"""Generate a brief, natural JARVIS response for this situation:
- Speaker: {speaker_name} (device owner: {is_owner})
- Current state: Screen is locked
- User command: "{original_command}"
- Action needed: Unlock screen, then execute command

Requirements:
- Address user as "{speaker_name}" or "Sir"
- Acknowledge the locked screen
- Explain what you'll do
- Keep it conversational and brief (1-2 sentences)
- Sound like JARVIS from Iron Man

Response:"""

                response = await self.llm_client.generate_text(
                    prompt,
                    max_tokens=100,
                    temperature=0.7,
                )

                return response.strip()

            except Exception as e:
                logger.debug(f"LLM generation failed, using template: {e}")

        # Fallback to template
        return self.templates["screen_locked"].format(**context)

    async def generate_action_message(
        self,
        action_type: str,
        speaker_name: str = "Sir",
        action_description: str = "",
        app_name: Optional[str] = None,
        error_reason: Optional[str] = None,
        custom_context: Optional[Dict] = None,
    ) -> str:
        """
        Generate message for various action types

        Args:
            action_type: Type of action (in_progress, complete, failed, etc.)
            speaker_name: Name of the speaker
            action_description: Description of the action
            app_name: Name of app involved (if any)
            error_reason: Reason for failure (if action_type is "failed")
            custom_context: Additional context for message generation

        Returns:
            Contextual message
        """
        if not self.initialized:
            await self.initialize()

        context = {
            "speaker_name": speaker_name,
            "action": action_description,
            "app_name": app_name or "the application",
            "error_reason": error_reason or "Please try again.",
            "time_of_day": self._get_time_of_day(),
        }

        if custom_context:
            context.update(custom_context)

        # Use LLM if available
        if self.llm_client and hasattr(self.llm_client, "generate_text"):
            try:
                prompt = self._build_prompt(action_type, context)
                response = await self.llm_client.generate_text(
                    prompt,
                    max_tokens=100,
                    temperature=0.7,
                )
                return response.strip()

            except Exception as e:
                logger.debug(f"LLM generation failed, using template: {e}")

        # Fallback to template
        template = self.templates.get(action_type, self.templates["action_in_progress"])
        return template.format(**context)

    async def generate_verification_message(
        self,
        speaker_name: str,
        action_description: str,
        verification_confidence: float,
    ) -> str:
        """
        Generate message for speaker verification

        Args:
            speaker_name: Identified speaker name
            action_description: Action requiring verification
            verification_confidence: Confidence of verification (0.0-1.0)

        Returns:
            Contextual message
        """
        if not self.initialized:
            await self.initialize()

        if verification_confidence >= 0.90:
            greeting = f"Welcome, {speaker_name}."
        elif verification_confidence >= 0.75:
            greeting = f"Good to see you, {speaker_name}."
        else:
            greeting = "I'm having trouble recognizing your voice."

        if self.llm_client and hasattr(self.llm_client, "generate_text"):
            try:
                prompt = f"""Generate a brief JARVIS greeting with verification status:
- Speaker: {speaker_name}
- Verification confidence: {verification_confidence:.0%}
- Action requested: {action_description}

Include:
- Greeting appropriate for confidence level
- Acknowledgment of verification
- What you'll do next

Keep it natural and brief (1-2 sentences).

Response:"""

                response = await self.llm_client.generate_text(
                    prompt,
                    max_tokens=100,
                    temperature=0.7,
                )
                return response.strip()

            except Exception as e:
                logger.debug(f"LLM generation failed, using simple greeting: {e}")

        # Fallback
        return f"{greeting} Let me {action_description} for you."

    def _build_prompt(self, action_type: str, context: Dict) -> str:
        """Build LLM prompt for message generation"""
        action_map = {
            "action_in_progress": "currently performing",
            "action_complete": "completed successfully",
            "action_failed": "encountered an error",
            "app_not_running": "needs to open an app first",
            "permission_required": "needs permission",
        }

        status = action_map.get(action_type, "performing an action")

        prompt = f"""Generate a brief, natural JARVIS response:
- Speaker: {context['speaker_name']}
- Status: {status}
- Action: {context['action']}
"""

        if context.get("app_name"):
            prompt += f"- App: {context['app_name']}\n"

        if context.get("error_reason"):
            prompt += f"- Error: {context['error_reason']}\n"

        prompt += """
Requirements:
- Sound like JARVIS from Iron Man
- Be helpful and professional
- Keep it brief (1-2 sentences)
- Address user appropriately

Response:"""

        return prompt

    def _get_time_of_day(self) -> str:
        """Get time of day for greetings"""
        hour = datetime.now().hour

        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"


# Global singleton
_message_generator: Optional[ContextualMessageGenerator] = None


def get_message_generator(llm_client=None) -> ContextualMessageGenerator:
    """Get global message generator instance"""
    global _message_generator

    if _message_generator is None:
        _message_generator = ContextualMessageGenerator(llm_client)

    return _message_generator


async def reset_message_generator():
    """Reset message generator (for testing)"""
    global _message_generator
    _message_generator = None
