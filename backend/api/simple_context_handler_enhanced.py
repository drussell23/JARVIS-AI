#!/usr/bin/env python3
"""
Enhanced Simple Context Handler for JARVIS
==========================================

Provides context-aware command processing with clear step-by-step feedback
"""

import asyncio
import logging
import re
from typing import Dict, Any, Optional, List
from datetime import datetime

# Use the fixed direct unlock handler
from api.direct_unlock_handler_fixed import (
    unlock_screen_direct,
    check_screen_locked_direct,
)

logger = logging.getLogger(__name__)


class EnhancedSimpleContextHandler:
    """Enhanced handler for context-aware command processing with step-by-step feedback"""

    def __init__(self, command_processor):
        self.command_processor = command_processor
        self.execution_steps = []
        self.screen_required_patterns = [
            # Browser operations
            "open safari",
            "open chrome",
            "open firefox",
            "open browser",
            "search for",
            "google",
            "look up",
            "find online",
            "go to",
            "navigate to",
            "visit",
            "browse",
            # Application operations
            "open",
            "launch",
            "start",
            "run",
            "quit",
            "close app",
            "switch to",
            "show me",
            "display",
            "bring up",
            # File operations
            "create",
            "edit",
            "save",
            "close file",
            "find file",
            "open file",
            "open document",
            # System UI operations
            "click",
            "type",
            "press",
            "select",
            "take screenshot",
            "show desktop",
            "minimize",
            "maximize",
        ]

    def _add_step(self, step: str, details: Dict[str, Any] = None):
        """Add an execution step for tracking"""
        self.execution_steps.append(
            {
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "details": details or {},
            }
        )
        logger.info(f"[CONTEXT STEP] {step}")

    async def process_with_context(
        self, command: str, websocket=None
    ) -> Dict[str, Any]:
        """Process command with enhanced context awareness and feedback"""
        try:
            # Reset steps for new command
            self.execution_steps = []

            logger.info(f"[ENHANCED CONTEXT] ========= START PROCESSING =========")
            logger.info(f"[ENHANCED CONTEXT] Command: '{command}'")
            self._add_step(f"Received command: {command}")

            # Check if command requires screen
            requires_screen = self._requires_screen(command)
            logger.info(f"[ENHANCED CONTEXT] Requires screen: {requires_screen}")

            if requires_screen:
                self._add_step(
                    "Command requires screen access", {"requires_screen": True}
                )

                # Check if screen is locked
                logger.info("[ENHANCED CONTEXT] Checking screen lock status...")
                is_locked = await self._check_screen_locked()
                logger.info(f"[ENHANCED CONTEXT] Screen locked: {is_locked}")
                self._add_step(
                    f"Screen status: {'LOCKED' if is_locked else 'UNLOCKED'}",
                    {"is_locked": is_locked},
                )

                if is_locked:
                    # Build context-aware response
                    action = self._extract_action_description(command)
                    context_message = f"I see your screen is locked. I'll unlock it now by typing in your password so I can {action}."

                    self._add_step(
                        "Screen unlock required", {"message": context_message}
                    )

                    # Send context message to user
                    if websocket:
                        # Send as response type to ensure it's spoken
                        await websocket.send_json(
                            {
                                "type": "response",
                                "text": context_message,
                                "command_type": "context_aware",
                                "status": "unlocking_screen",
                                "steps": self.execution_steps,
                                "speak": True,
                                "intermediate": True,  # Mark as intermediate response
                            }
                        )

                    # Perform unlock
                    logger.info("[ENHANCED CONTEXT] Attempting to unlock screen...")
                    unlock_success = await self._unlock_screen(command)

                    if unlock_success:
                        self._add_step(
                            "Screen unlocked successfully", {"success": True}
                        )

                        # Brief pause for unlock to fully complete
                        await asyncio.sleep(2.0)

                        # Send progress update
                        if websocket:
                            await websocket.send_json(
                                {
                                    "type": "response",
                                    "text": "Screen unlocked. Now executing your command...",
                                    "command_type": "context_aware",
                                    "status": "executing_command",
                                    "steps": self.execution_steps,
                                    "speak": True,
                                    "intermediate": True,
                                }
                            )

                        # Execute the original command
                        logger.info("[ENHANCED CONTEXT] Executing original command...")
                        result = await self.command_processor.process_command(
                            command, websocket
                        )

                        # Build comprehensive response
                        self._add_step(
                            "Command executed",
                            {"success": result.get("success", False)},
                        )

                        # Format the final response with all steps
                        if isinstance(result, dict):
                            original_response = result.get("response", "")

                            # Build step-by-step summary
                            steps_summary = self._build_steps_summary()

                            # Combine context handling with command result
                            result["response"] = (
                                f"{context_message} {original_response}"
                            )
                            result["context_handled"] = True
                            result["screen_unlocked"] = True
                            result["execution_steps"] = self.execution_steps
                            result["steps_summary"] = steps_summary

                        logger.info(
                            "[ENHANCED CONTEXT] Command completed with context handling"
                        )
                        return result
                    else:
                        self._add_step("Screen unlock failed", {"success": False})
                        return {
                            "success": False,
                            "response": "I tried to unlock your screen but couldn't. Please unlock it manually and try your command again.",
                            "context_handled": True,
                            "screen_unlocked": False,
                            "execution_steps": self.execution_steps,
                        }

            # No special context handling needed
            self._add_step("No context handling required")
            return await self.command_processor.process_command(command, websocket)

        except Exception as e:
            logger.error(f"[ENHANCED CONTEXT] Error: {e}", exc_info=True)
            self._add_step(f"Error occurred: {str(e)}", {"error": True})

            # Fallback to standard processing
            return await self.command_processor.process_command(command, websocket)

    def _requires_screen(self, command: str) -> bool:
        """Check if command requires screen access"""
        command_lower = command.lower()

        # Commands that explicitly don't need screen
        no_screen_patterns = [
            "lock screen",
            "lock my screen",
            "lock the screen",
            "what time",
            "weather",
            "temperature",
            "play music",
            "pause music",
            "stop music",
            "volume up",
            "volume down",
            "mute",
        ]

        if any(pattern in command_lower for pattern in no_screen_patterns):
            return False

        # Check if any screen-required pattern matches
        for pattern in self.screen_required_patterns:
            if pattern in command_lower:
                return True

        return False

    def _extract_action_description(self, command: str) -> str:
        """Extract a human-readable description of what the user wants to do"""
        command_lower = command.lower()

        # Common patterns and their descriptions
        patterns = [
            (r"open safari and (?:search for|google) (.+)", "search for {}"),
            (r"open (\w+)", "open {}"),
            (r"search for (.+)", "search for {}"),
            (r"go to (.+)", "navigate to {}"),
            (r"create (.+)", "create {}"),
            (r"show me (.+)", "show you {}"),
            (r"find (.+)", "find {}"),
        ]

        for pattern, template in patterns:
            match = re.search(pattern, command_lower)
            if match:
                return template.format(match.group(1))

        # Default: use the command as-is
        return f"execute your command: {command}"

    def _build_steps_summary(self) -> str:
        """Build a human-readable summary of execution steps"""
        if not self.execution_steps:
            return ""

        summary_parts = []
        for i, step in enumerate(self.execution_steps, 1):
            summary_parts.append(f"{i}. {step['step']}")

        return " ".join(summary_parts)

    async def _check_screen_locked(self) -> bool:
        """Check if screen is currently locked"""
        return await check_screen_locked_direct()

    async def _unlock_screen(self, command: str) -> bool:
        """Unlock the screen with context"""
        return await unlock_screen_direct(f"Context-aware execution: {command}")


def wrap_with_enhanced_context(processor):
    """Wrap a command processor with enhanced context handling"""
    handler = EnhancedSimpleContextHandler(processor)
    return handler

