"""
Screen Lock Context Detector for JARVIS
======================================

Detects screen lock state and provides context-aware responses.
Integrated with async_pipeline.py for dynamic, robust operation.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from api.voice_unlock_integration import voice_unlock_connector

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
            "browser": {"open", "launch", "start", "navigate", "browse", "surf"},
            "search": {"search", "google", "look up", "find online", "query"},
            "navigation": {"go to", "visit", "navigate to", "head to"},
            "application": {"open", "launch", "start", "run", "execute"},
            "ui_interaction": {
                "switch to",
                "show",
                "display",
                "minimize",
                "maximize",
                "resize",
            },
            "file_ops": {
                "create",
                "edit",
                "save",
                "close",
                "open file",
                "open document",
            },
            "system_ui": {"screenshot", "capture", "show desktop", "move window"},
        }

        # Commands that DON'T require screen (voice-only)
        self.screen_exempt_patterns = {
            "lock screen",
            "lock my screen",
            "sleep",
            "shutdown",
            "what time",
            "what's the time",
            "weather",
            "temperature",
            "set timer",
            "set alarm",
            "remind me",
            "play music",
        }

    async def is_screen_locked(self) -> bool:
        """Check if screen is currently locked"""
        try:
            from voice_unlock.objc.server.screen_lock_detector import is_screen_locked

            return is_screen_locked()
        except Exception as e:
            logger.debug(f"Could not check screen lock status: {e}")
            return False

    async def check_screen_context(self, command: str, speaker_name: str = None) -> Dict[str, Any]:
        """
        Check screen lock context for a command

        Args:
            command: The command to execute
            speaker_name: Identified speaker name (for personalization)

        Returns:
            Context dict with screen state and recommendations
        """
        is_locked = await self.is_screen_locked()
        command_needs_screen = await self._command_requires_screen(command)

        logger.warning(f"[SCREEN LOCK DETECTOR] ━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.warning(f"[SCREEN LOCK DETECTOR] 🔍 Checking screen context")
        logger.warning(f"[SCREEN LOCK DETECTOR] 📝 Command: {command}")
        logger.warning(f"[SCREEN LOCK DETECTOR] 🔒 Screen locked: {is_locked}")
        logger.warning(f"[SCREEN LOCK DETECTOR] 📺 Requires screen: {command_needs_screen}")

        context = {
            "screen_locked": is_locked,
            "requires_unlock": False,
            "unlock_message": None,
            "command_requires_screen": command_needs_screen,
            "timestamp": datetime.now().isoformat(),
        }

        # If screen is locked and command requires screen access
        if is_locked and context["command_requires_screen"]:
            context["requires_unlock"] = True
            context["unlock_message"] = await self._generate_unlock_message(command, speaker_name)
            logger.warning(f"[SCREEN LOCK DETECTOR] ✅ UNLOCK REQUIRED")
            logger.warning(f"[SCREEN LOCK DETECTOR] 📢 Message: {context['unlock_message']}")
            if speaker_name:
                logger.warning(f"[SCREEN LOCK DETECTOR] 👤 Speaker: {speaker_name}")
        else:
            logger.warning(f"[SCREEN LOCK DETECTOR] ❌ NO UNLOCK NEEDED")
            if not is_locked:
                logger.warning(f"[SCREEN LOCK DETECTOR]    Reason: Screen not locked")
            if not command_needs_screen:
                logger.warning(f"[SCREEN LOCK DETECTOR]    Reason: Command doesn't need screen")

        logger.warning(f"[SCREEN LOCK DETECTOR] ━━━━━━━━━━━━━━━━━━━━━━━━━━")

        return context

    async def _command_requires_screen(self, command: str) -> bool:
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
            from context_intelligence.analyzers.compound_action_parser import (
                ActionType,
                get_compound_parser,
            )

            parser = get_compound_parser()
            # Since we're already in an async context, await directly
            actions = await parser.parse(command)

            # If parser found actions, check if any require screen
            if actions:
                screen_requiring_types = {
                    ActionType.OPEN_APP,
                    ActionType.CLOSE_APP,  # Closing apps requires screen access!
                    ActionType.SEARCH_WEB,
                    ActionType.NAVIGATE_URL,
                    ActionType.EXECUTE_SCRIPT,
                    ActionType.CREATE_DOCUMENT,  # Document creation requires screen access!
                    ActionType.TAKE_SCREENSHOT,
                    ActionType.CLICK,
                    ActionType.TYPE_TEXT,
                }

                for action in actions:
                    if action.type in screen_requiring_types:
                        logger.warning(
                            f"[SCREEN LOCK DETECTOR] ✅ Action '{action.type.value}' requires screen access"
                        )
                        return True

                # Parser found actions but none require screen
                logger.warning(
                    f"[SCREEN LOCK DETECTOR] ❌ Parser found actions but none require screen: {[a.type.value for a in actions]}"
                )
                return False
            else:
                logger.warning(
                    f"[SCREEN LOCK DETECTOR] ❌ Parser returned no actions for: {command}"
                )

        except Exception as e:
            logger.error(
                f"[SCREEN LOCK DETECTOR] ❌ Compound parser FAILED, falling back to pattern matching: {e}"
            )
            import traceback

            logger.error(f"[SCREEN LOCK DETECTOR] Traceback: {traceback.format_exc()}")

        # Third check: Fallback to dynamic category matching
        for category, keywords in self.screen_required_actions.items():
            for keyword in keywords:
                if keyword in command_lower:
                    logger.debug(
                        f"Command matches screen-required category '{category}': {keyword}"
                    )
                    return True

        # Default: if uncertain, assume no screen required (safer for user experience)
        logger.debug(f"Command does not require screen access: {command}")
        return False

    async def _generate_unlock_message(self, command: str, speaker_name: str = None) -> str:
        """
        Generate dynamic, contextual unlock message based on command analysis.
        Advanced, robust, and completely dynamic with no hardcoding.

        Args:
            command: The command that requires unlock
            speaker_name: Identified speaker name (for personalization)

        Returns:
            Contextual message to speak to user
        """
        # Parse command into actions for intelligent messaging
        actions = await self._extract_actions_dynamic(command)

        # Extract contextual information
        command_context = await self._analyze_command_context(command)

        # Generate contextual message based on detected actions and context
        return await self._generate_contextual_message(
            command, actions, command_context, speaker_name
        )

    async def _extract_actions_dynamic(self, command: str) -> List[str]:
        """Extract all actions from command using CompoundActionParser"""
        try:
            from context_intelligence.analyzers.compound_action_parser import get_compound_parser

            parser = get_compound_parser()
            actions = await parser.parse(command)

            # Convert to human-readable action descriptions
            action_descriptions = []
            for action in actions:
                if action.type.value == "open_app":
                    app_name = action.params.get("app_name", "application")
                    action_descriptions.append(f"open {app_name}")
                elif action.type.value == "search_web":
                    query = action.params.get("query", "online")
                    action_descriptions.append(f"search for {query}")
                elif action.type.value == "navigate_url":
                    url = action.params.get("url", "website")
                    action_descriptions.append(f"navigate to {url}")
                elif action.type.value == "create_document":
                    action_descriptions.append("create a document")
                else:
                    action_descriptions.append(action.type.value.replace("_", " "))

            return action_descriptions
        except Exception as e:
            logger.warning(f"Failed to parse actions: {e}")
            return ["complete your request"]

    async def _analyze_command_context(self, command: str) -> Dict[str, Any]:
        """Analyze command for contextual information"""
        command_lower = command.lower()

        context = {
            "is_document_creation": False,
            "is_web_search": False,
            "is_app_opening": False,
            "is_compound": False,
            "urgency": "normal",
            "topic": None,
            "app_name": None,
            "search_query": None,
        }

        # Detect document creation
        doc_keywords = [
            "write",
            "create",
            "draft",
            "compose",
            "essay",
            "document",
            "paper",
            "letter",
            "memo",
        ]
        if any(keyword in command_lower for keyword in doc_keywords):
            context["is_document_creation"] = True
            context["topic"] = self._extract_topic_from_command(command)

        # Detect web search
        search_keywords = ["search", "google", "look up", "find online", "browse"]
        if any(keyword in command_lower for keyword in search_keywords):
            context["is_web_search"] = True
            context["search_query"] = self._extract_search_query(command)

        # Detect app opening
        app_keywords = ["open", "launch", "start"]
        if any(keyword in command_lower for keyword in app_keywords):
            context["is_app_opening"] = True
            context["app_name"] = self._extract_app_name(command)

        # Detect compound commands
        compound_indicators = ["and", "then", "after", "also", "plus"]
        if any(indicator in command_lower for indicator in compound_indicators):
            context["is_compound"] = True

        # Detect urgency
        urgent_keywords = ["urgent", "asap", "quickly", "fast", "immediately"]
        if any(keyword in command_lower for keyword in urgent_keywords):
            context["urgency"] = "urgent"

        return context

    async def _generate_contextual_message(
        self, command: str, actions: List[str], context: Dict[str, Any], speaker_name: str = None
    ) -> str:
        """Generate contextual unlock message based on analysis with speaker recognition"""
        import random  # nosec B311 # UI message selection, not cryptographic

        # Determine greeting based on speaker
        if speaker_name:
            # First time greeting with name recognition
            greetings = [
                f"Good to see you, {speaker_name}.",
                f"Hello, {speaker_name}.",
                f"Welcome back, {speaker_name}.",
            ]
            greeting = random.choice(greetings)  # nosec B311 # UI message selection
        else:
            greeting = "Sir,"

        # Document creation messages
        if context["is_document_creation"] and context["topic"]:
            templates = [
                f"{greeting} Your screen is locked. Let me unlock it to create that {context['topic']}.",
                f"I notice your screen is locked, {speaker_name if speaker_name else 'Sir'}. Unlocking it now to write your {context['topic']}.",
                f"{greeting} Screen is locked. I'll unlock it to work on your {context['topic']}.",
            ]
            return random.choice(templates)  # nosec B311 # UI message selection

        # Web search messages
        if context["is_web_search"] and context["search_query"]:
            templates = [
                f"{greeting} Your screen is locked. Let me unlock it to search for {context['search_query']}.",
                f"I'll unlock your screen to search for {context['search_query']}, {speaker_name if speaker_name else 'Sir'}.",
                f"{greeting} Unlocking screen to search for {context['search_query']}.",
            ]
            return random.choice(templates)  # nosec B311 # UI message selection

        # App opening messages
        if context["is_app_opening"] and context["app_name"]:
            templates = [
                f"{greeting} Your screen is locked. Let me unlock it to open {context['app_name']}.",
                f"I'll unlock your screen to open {context['app_name']}, {speaker_name if speaker_name else 'Sir'}.",
                f"{greeting} Unlocking screen to open {context['app_name']}.",
            ]
            return random.choice(templates)  # nosec B311 # UI message selection

        # Compound command messages
        if context["is_compound"] and len(actions) > 1:
            action_text = " and ".join(actions[:2])  # Limit to first two actions
            templates = [
                f"{greeting} Your screen is locked. Let me unlock it to {action_text}.",
                f"I'll unlock your screen to {action_text}, {speaker_name if speaker_name else 'Sir'}.",
                f"{greeting} Unlocking screen to {action_text}.",
            ]
            return random.choice(templates)  # nosec B311 # UI message selection

        # Generic messages based on urgency
        if context["urgency"] == "urgent":
            templates = [
                f"{greeting} Your screen is locked. Unlocking it immediately for your urgent request.",
                f"I'll unlock your screen right away, {speaker_name if speaker_name else 'Sir'}.",
                f"{greeting} Unlocking screen immediately.",
            ]
        else:
            templates = [
                f"{greeting} Your screen is locked. Let me unlock it to complete your request.",
                f"I'll unlock your screen to proceed with your command, {speaker_name if speaker_name else 'Sir'}.",
                f"{greeting} Unlocking screen to continue.",
            ]

        return random.choice(templates)  # nosec B311 # UI message selection

    def _extract_search_query(self, command: str) -> str:
        """Extract search query from command"""
        import re

        # Common patterns for search queries
        patterns = [
            r"search for (.+)",
            r"google (.+)",
            r"look up (.+)",
            r"find (.+)",
            r"browse (.+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, command.lower())
            if match:
                query = match.group(1).strip()
                # Clean up the query
                query = query.replace(" online", "").replace(" on the web", "")
                return query[:50]  # Limit length

        return "your search"

    def _extract_app_name(self, command: str) -> str:
        """Extract app name from command"""
        import re

        # Common patterns for app names
        patterns = [
            r"open (.+)",
            r"launch (.+)",
            r"start (.+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, command.lower())
            if match:
                app_name = match.group(1).strip()
                # Clean up the app name
                app_name = app_name.replace(" application", "").replace(" app", "")
                return app_name.title()

        return "the application"

    async def _extract_action_dynamic(self, command: str) -> str:
        """
        Dynamically extract action description using CompoundActionParser.
        Falls back to simple extraction if parser fails.
        """
        try:
            from context_intelligence.analyzers.compound_action_parser import get_compound_parser

            parser = get_compound_parser()

            # Get execution plan - await directly since we're in async context
            actions = await parser.parse(command)

            if actions:
                # Get human-readable plan
                plan = parser.get_execution_plan(actions)
                return plan.lower()  # "open safari → search for 'dogs'"

        except Exception as e:
            logger.debug(f"Dynamic action extraction failed, using fallback: {e}")

        # Fallback to simple extraction
        return self._extract_action_simple(command)

    def _extract_topic_from_command(self, command: str) -> str:
        """Extract the document topic from command for natural messaging"""
        import re

        command_lower = command.lower()

        # Try to extract "essay/paper/document on/about TOPIC"
        patterns = [
            r"(?:essay|paper|document|article)\s+(?:on|about)\s+(.+?)(?:\s+in\s|$)",
            r"(?:write|create|draft|compose)\s+(?:an?|the)?\s*(?:essay|paper|document|article)?\s+(?:on|about)\s+(.+?)(?:\s+in\s|$)",
            r"(?:write|create|draft|compose)\s+(?:me\s+)?(?:an?|the)?\s*(essay|paper|document|article)\s+(?:on|about)\s+(.+?)(?:\s+in\s|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, command_lower)
            if match:
                # Get the last group (the topic)
                groups = match.groups()
                topic = groups[-1].strip() if groups else None
                if topic:
                    # Clean up common trailing words
                    topic = re.sub(r"\s+(for me|please)$", "", topic)
                    return f"essay on {topic}"

        # Fallback: just say "document" or "essay"
        if "essay" in command_lower:
            return "essay"
        elif "paper" in command_lower:
            return "paper"
        elif "document" in command_lower:
            return "document"
        else:
            return "document"

    def _extract_action_simple(self, command: str) -> str:
        """Simple fallback action extraction"""
        command_lower = command.lower()

        # Common patterns
        if "search for" in command_lower:
            search_term = command_lower.split("search for")[-1].strip()
            return f"search for {search_term}"
        elif "open" in command_lower:
            app_or_site = command_lower.split("open")[-1].strip()
            return f"open {app_or_site}"
        elif "go to" in command_lower:
            destination = command_lower.split("go to")[-1].strip()
            return f"go to {destination}"
        else:
            # Default: use the command as-is
            return "complete your request"

    async def handle_screen_lock_context(
        self, command: str, audio_data: bytes = None, speaker_name: str = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Handle screen lock context and unlock if needed using Intelligent Voice Unlock Service

        Args:
            command: The command to execute
            audio_data: Audio data for voice verification (optional)
            speaker_name: Known speaker name (optional)

        Returns:
            Tuple of (success, message)
        """
        context = await self.check_screen_context(command)

        if not context["requires_unlock"]:
            return True, None

        # Screen is locked and needs to be unlocked
        unlock_message = context["unlock_message"]

        try:
            # Try to unlock using the Intelligent Voice Unlock Service first
            logger.info(f"Attempting intelligent voice-authenticated unlock for command: {command}")

            # Try Intelligent Voice Unlock Service with full voice authentication
            try:
                from voice_unlock.intelligent_voice_unlock_service import (
                    get_intelligent_unlock_service,
                )

                intelligent_service = get_intelligent_unlock_service()

                # If we have audio data, use full intelligent unlock with voice verification
                if audio_data:
                    logger.info("Using Intelligent Voice Unlock Service with voice verification")

                    # Create context for the unlock
                    unlock_context = {
                        "source": "context_intelligence",
                        "reason": f"User command: {command}",
                        "original_command": command,
                        "screen_locked": True,
                    }

                    # Process with full intelligence stack (Speaker Recognition, CAI, SAI, etc.)
                    result = await intelligent_service.process_voice_unlock_command(
                        audio_data=audio_data, context=unlock_context
                    )

                    if result and result.get("success"):
                        logger.info("✅ Intelligent Voice Unlock succeeded with voice verification")
                        # Wait for unlock to complete
                        await asyncio.sleep(2.0)

                        # Verify unlock succeeded
                        is_still_locked = await self.is_screen_locked()
                        if not is_still_locked:
                            success_msg = "Now proceeding with your request."
                            return True, f"{unlock_message} {success_msg}"
                        else:
                            return (
                                False,
                                "Voice authentication succeeded but screen unlock failed. Please try again.",
                            )
                    else:
                        error_msg = result.get("message", "Voice authentication failed")
                        logger.warning(f"❌ Intelligent Voice Unlock failed: {error_msg}")
                        return False, f"I couldn't unlock the screen: {error_msg}"

                else:
                    # Fallback: Use simple unlock handler when no audio data available
                    logger.warning("No audio data available, falling back to simple unlock handler")
                    from api.simple_unlock_handler import handle_unlock_command

                    # Send unlock command
                    result = await handle_unlock_command("unlock my screen", None)

                if result and result.get("success"):
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
                            return (
                                False,
                                "I tried to unlock the screen but it appears to still be locked. Please try unlocking manually.",
                            )
                else:
                    error_msg = (
                        result.get("response", "Could not unlock screen")
                        if result
                        else "Unlock service not available"
                    )
                    return False, f"I couldn't unlock the screen: {error_msg}"

            except ImportError:
                # Fallback: try using voice unlock if available
                if voice_unlock_connector and voice_unlock_connector.connected:
                    result = await voice_unlock_connector.send_command(
                        "unlock_screen",
                        {
                            "source": "context_intelligence",
                            "reason": f"User command: {command}",
                            "authenticated": True,
                        },
                    )

                    if result and result.get("success"):
                        await asyncio.sleep(2.0)
                        is_still_locked = await self.is_screen_locked()
                        if not is_still_locked:
                            return (
                                True,
                                f"{unlock_message} Screen unlocked successfully.",
                            )

                # If all else fails, provide manual instruction
                return (
                    False,
                    f"{unlock_message} However, I need you to unlock your screen manually to proceed.",
                )

        except Exception as e:
            logger.error(f"Error handling screen lock context: {e}")
            return (
                False,
                f"I encountered an error while trying to unlock the screen: {str(e)}",
            )


# Global instance
_detector = None


def get_screen_lock_detector() -> ScreenLockContextDetector:
    """Get or create screen lock detector instance"""
    global _detector
    if _detector is None:
        _detector = ScreenLockContextDetector()
    return _detector
