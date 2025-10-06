"""
Screen Lock Context Detector for JARVIS
======================================

Detects screen lock state and provides context-aware responses.
Integrated with async_pipeline.py for dynamic, robust operation.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Tuple, Set
from datetime import datetime

from api.voice_unlock_integration import voice_unlock_connector, initialize_voice_unlock

logger = logging.getLogger(__name__)


class ScreenLockContextDetector:
    """
    Detects screen lock state and provides context for command execution.
    Dynamic pattern matching - NO hardcoding.
    """

    def __init__(self):
        self._last_check = None
        self._last_state = None

        # Dynamic action categories that require screen access
        self.screen_required_actions = {
            'browser': {'open', 'launch', 'start', 'navigate', 'browse', 'surf'},
            'search': {'search', 'google', 'look up', 'find online', 'query'},
            'navigation': {'go to', 'visit', 'navigate to', 'head to'},
            'application': {'open', 'launch', 'start', 'run', 'execute'},
            'ui_interaction': {'switch to', 'show', 'display', 'minimize', 'maximize', 'resize'},
            'file_ops': {'create', 'edit', 'save', 'close', 'open file', 'open document'},
            'system_ui': {'screenshot', 'capture', 'show desktop', 'move window'},
        }

        # Commands that DON'T require screen (voice-only)
        self.screen_exempt_patterns = {
            'lock screen', 'lock my screen', 'sleep', 'shutdown',
            'what time', 'what\'s the time', 'weather', 'temperature',
            'set timer', 'set alarm', 'remind me', 'play music'
        }
        
    async def is_screen_locked(self) -> bool:
        """Check if screen is currently locked"""
        try:
            from voice_unlock.objc.server.screen_lock_detector import is_screen_locked
            return is_screen_locked()
        except Exception as e:
            logger.debug(f"Could not check screen lock status: {e}")
            return False
        
    async def check_screen_context(self, command: str) -> Dict[str, Any]:
        """
        Check screen lock context for a command
        
        Args:
            command: The command to execute
            
        Returns:
            Context dict with screen state and recommendations
        """
        is_locked = await self.is_screen_locked()
        
        context = {
            "screen_locked": is_locked,
            "requires_unlock": False,
            "unlock_message": None,
            "command_requires_screen": self._command_requires_screen(command),
            "timestamp": datetime.now().isoformat()
        }
        
        # If screen is locked and command requires screen access
        if is_locked and context["command_requires_screen"]:
            context["requires_unlock"] = True
            context["unlock_message"] = self._generate_unlock_message(command)
            
        return context
        
    def _command_requires_screen(self, command: str) -> bool:
        """
        Dynamically determine if a command requires screen access.
        Uses compound action parser for intelligent detection.

        Args:
            command: The command to check

        Returns:
            True if command requires unlocked screen
        """
        command_lower = command.lower()

        # First check: Is this a voice-only/exempt command?
        for exempt_pattern in self.screen_exempt_patterns:
            if exempt_pattern in command_lower:
                logger.debug(f"Command exempt from screen requirement: {exempt_pattern}")
                return False

        # Second check: Parse compound actions using CompoundActionParser
        try:
            from context_intelligence.analyzers.compound_action_parser import get_compound_parser, ActionType

            parser = get_compound_parser()
            # Create a simple async wrapper
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            actions = loop.run_until_complete(parser.parse(command))

            # If parser found actions, check if any require screen
            if actions:
                screen_requiring_types = {
                    ActionType.OPEN_APP,
                    ActionType.SEARCH_WEB,
                    ActionType.NAVIGATE_URL,
                    ActionType.EXECUTE_SCRIPT,
                }

                for action in actions:
                    if action.type in screen_requiring_types:
                        logger.debug(f"Action {action.type.value} requires screen access")
                        return True

                # Parser found actions but none require screen
                return False

        except Exception as e:
            logger.debug(f"Compound parser check failed, falling back to pattern matching: {e}")

        # Third check: Fallback to dynamic category matching
        for category, keywords in self.screen_required_actions.items():
            for keyword in keywords:
                if keyword in command_lower:
                    logger.debug(f"Command matches screen-required category '{category}': {keyword}")
                    return True

        # Default: if uncertain, assume no screen required (safer for user experience)
        logger.debug(f"Command does not require screen access: {command}")
        return False
        
    def _generate_unlock_message(self, command: str) -> str:
        """
        Generate a dynamic, context-aware message for unlocking.
        Uses CompoundActionParser for intelligent action extraction.

        Args:
            command: The original command

        Returns:
            Message to speak to user
        """
        # Extract the action using CompoundActionParser for accuracy
        action = self._extract_action_dynamic(command)

        # Dynamic message templates
        templates = [
            f"Your screen is locked. I'll unlock it to {action}.",
            f"Let me unlock your screen so I can {action}.",
            f"Unlocking screen to {action}.",
        ]

        # Use voice dynamic response generator if available
        try:
            from voice.dynamic_response_generator import get_response_generator
            response_gen = get_response_generator()

            # Generate natural unlock message
            base_message = templates[0]  # Use first template as base
            return response_gen.get_contextual_response(
                base_message,
                context={'action': action, 'command': command}
            )
        except:
            # Fallback: use simple template
            import random
            return random.choice(templates)
        
    def _extract_action_dynamic(self, command: str) -> str:
        """
        Dynamically extract action description using CompoundActionParser.
        Falls back to simple extraction if parser fails.
        """
        try:
            from context_intelligence.analyzers.compound_action_parser import get_compound_parser

            parser = get_compound_parser()

            # Get execution plan
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            actions = loop.run_until_complete(parser.parse(command))

            if actions:
                # Get human-readable plan
                plan = parser.get_execution_plan(actions)
                return plan.lower()  # "open safari → search for 'dogs'"

        except Exception as e:
            logger.debug(f"Dynamic action extraction failed, using fallback: {e}")

        # Fallback to simple extraction
        return self._extract_action_simple(command)

    def _extract_action_simple(self, command: str) -> str:
        """Simple fallback action extraction"""
        command_lower = command.lower()

        # Common patterns
        if 'search for' in command_lower:
            search_term = command_lower.split('search for')[-1].strip()
            return f"search for {search_term}"
        elif 'open' in command_lower:
            app_or_site = command_lower.split('open')[-1].strip()
            return f"open {app_or_site}"
        elif 'go to' in command_lower:
            destination = command_lower.split('go to')[-1].strip()
            return f"go to {destination}"
        else:
            # Default: use the command as-is
            return "complete your request"
            
    async def handle_screen_lock_context(self, command: str) -> Tuple[bool, Optional[str]]:
        """
        Handle screen lock context and unlock if needed
        
        Args:
            command: The command to execute
            
        Returns:
            Tuple of (success, message)
        """
        context = await self.check_screen_context(command)
        
        if not context["requires_unlock"]:
            return True, None
            
        # Screen is locked and needs to be unlocked
        unlock_message = context["unlock_message"]
        
        try:
            # Try to unlock using the simple unlock handler directly
            logger.info(f"Attempting to unlock screen for command: {command}")

            # Use the simple unlock handler which has direct access to unlock functionality
            try:
                from api.simple_unlock_handler import handle_unlock_command

                # Send unlock command
                result = await handle_unlock_command("unlock my screen", None)

                if result and result.get('success'):
                    # Wait for unlock to complete
                    await asyncio.sleep(2.0)

                    # Verify unlock succeeded
                    is_still_locked = await self.is_screen_locked()
                    if not is_still_locked:
                        success_msg = "Now proceeding with your request."
                        return True, f"{unlock_message} {success_msg}"
                    else:
                        # Try one more time with a longer wait
                        await asyncio.sleep(1.0)
                        is_still_locked = await self.is_screen_locked()
                        if not is_still_locked:
                            success_msg = "Now proceeding with your request."
                            return True, f"{unlock_message} {success_msg}"
                        else:
                            return False, "I tried to unlock the screen but it appears to still be locked. Please try unlocking manually."
                else:
                    error_msg = result.get('response', 'Could not unlock screen') if result else 'Unlock service not available'
                    return False, f"I couldn't unlock the screen: {error_msg}"

            except ImportError:
                # Fallback: try using voice unlock if available
                if voice_unlock_connector and voice_unlock_connector.connected:
                    result = await voice_unlock_connector.send_command("unlock_screen", {
                        "source": "context_intelligence",
                        "reason": f"User command: {command}",
                        "authenticated": True
                    })

                    if result and result.get('success'):
                        await asyncio.sleep(2.0)
                        is_still_locked = await self.is_screen_locked()
                        if not is_still_locked:
                            return True, f"{unlock_message} Screen unlocked successfully."

                # If all else fails, provide manual instruction
                return False, f"{unlock_message} However, I need you to unlock your screen manually to proceed."
                
        except Exception as e:
            logger.error(f"Error handling screen lock context: {e}")
            return False, f"I encountered an error while trying to unlock the screen: {str(e)}"


# Global instance
_detector = None

def get_screen_lock_detector() -> ScreenLockContextDetector:
    """Get or create screen lock detector instance"""
    global _detector
    if _detector is None:
        _detector = ScreenLockContextDetector()
    return _detector