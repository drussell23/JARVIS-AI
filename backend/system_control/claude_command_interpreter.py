#!/usr/bin/env python3
"""
Claude-Powered Command Interpreter for JARVIS
Analyzes natural language and converts it to system commands
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from anthropic import Anthropic

from .dynamic_app_controller import get_dynamic_app_controller
from .fast_app_launcher import get_fast_app_launcher
from .macos_controller import CommandCategory, MacOSController, SafetyLevel
from .vision_action_handler import get_vision_action_handler

# Removed to avoid circular import - will import locally when needed

logger = logging.getLogger(__name__)


@dataclass
class CommandIntent:
    """Represents a parsed command intent"""

    action: str
    target: str
    parameters: Dict[str, Any]
    confidence: float
    category: CommandCategory
    safety_level: SafetyLevel
    requires_confirmation: bool
    raw_command: str


@dataclass
class CommandResult:
    """Result of command execution"""

    success: bool
    message: str
    data: Optional[Any] = None
    follow_up_needed: bool = False


class ClaudeCommandInterpreter:
    """Interprets natural language commands using Claude API with intelligent model selection"""

    def __init__(self, api_key: str, use_intelligent_selection: bool = True):
        self.client = Anthropic(api_key=api_key)
        self.api_key = api_key
        self.use_intelligent_selection = use_intelligent_selection
        self.controller = MacOSController()
        self.dynamic_controller = get_dynamic_app_controller()
        self.fast_launcher = get_fast_app_launcher()
        self.vision_handler = get_vision_action_handler()
        self.conversation_history = []
        self.system_state = {}

        # Command patterns and examples for Claude
        self.command_examples = {
            "application": [
                "Open Chrome",
                "Close Spotify",
                "Switch to Visual Studio Code",
                "Show me all open applications",
                "Minimize everything",
            ],
            "file": [
                "Create a new file called notes.txt on my desktop",
                "Open the report.pdf from my documents",
                "Search for Python files in my projects folder",
                "Delete old_file.txt from downloads",
            ],
            "system": [
                "Set volume to 50%",
                "Mute the sound",
                "Take a screenshot",
                "Put the display to sleep",
                "Turn off WiFi",
            ],
            "web": [
                "Search Google for Python tutorials",
                "Open YouTube",
                "Go to github.com",
                "Search for weather in New York",
            ],
            "workflow": [
                "Start my morning routine",
                "Set up my development environment",
                "Prepare for a meeting",
            ],
            "vision": [
                "Describe what's on my screen",
                "What am I looking at",
                "Check my screen for errors",
                "Analyze the current window",
                "Tell me what you see",
            ],
        }

    async def interpret_command(
        self, voice_input: str, context: Optional[Dict] = None
    ) -> CommandIntent:
        """Interpret natural language command using Claude"""

        # Validate input
        if not voice_input:
            logger.warning("interpret_command called with None/empty voice_input")
            return CommandIntent(
                action="",
                target="",
                parameters={},
                raw_text="",
                category=CommandCategory.UNKNOWN,
                confidence=0.0,
                safety=SafetyLevel.SAFE,
            )

        # Build context for Claude
        system_prompt = """You are JARVIS, an AI assistant that controls macOS systems.
        Analyze the user's command and extract:
        1. The action to perform
        2. The target (application, file, URL, etc.) - if multiple targets use "and" between them
        3. Any parameters needed
        4. The command category (application, file, system, web, workflow, vision)
        5. Safety assessment (safe, caution, dangerous)
        6. Confidence level (0-1)

        For vision commands about describing or analyzing the screen, use category="vision" and action="describe_screen"

        For multiple targets (e.g., "close whatsapp and preview"), set target as "whatsapp and preview".

        Return a JSON response with this structure:
        {
            "action": "action_name",
            "target": "target_name",
            "parameters": {},
            "category": "category_name",
            "confidence": 0.95,
            "requires_confirmation": false,
            "interpretation": "human-readable explanation"
        }

        Common actions:
        - Applications: open_app, close_app, switch_to_app, list_apps
        - Files: open_file, create_file, delete_file, search_files
        - System: set_volume, mute, screenshot, sleep_display, toggle_wifi
        - Vision: describe_screen, analyze_window, check_screen
        - Web: open_url, web_search
        - Workflows: morning_routine, development_setup, meeting_prep
        Important: Weather requests (e.g., "what's the weather today") should be handled as a workflow:
        1. Category: "workflow"
        2. Action: "check_weather_app"
        3. This will open the Weather app and use vision to read the information

        Safety rules:
        - File deletions always require confirmation
        - System settings changes are generally safe
        - Opening system preferences or terminal requires confirmation
        - Shutdown/restart commands are dangerous
        """

        # Add conversation history for context
        messages = []

        # Add recent history
        for hist in self.conversation_history[-5:]:
            messages.append({"role": "user", "content": hist["command"]})
            messages.append({"role": "assistant", "content": hist["response"]})

        # Add current command
        user_message = f"Command: {voice_input}"
        if context:
            user_message += f"\nContext: {json.dumps(context)}"

        messages.append({"role": "user", "content": user_message})

        try:
            # Try intelligent model selection first
            if self.use_intelligent_selection:
                try:
                    response_text = await self._interpret_with_intelligent_selection(
                        voice_input, context, system_prompt, messages
                    )
                except Exception as e:
                    logger.warning(
                        f"Intelligent selection failed for command interpretation, falling back to Claude API: {e}"
                    )
                    # Fall through to direct API call below
                    response_text = await self._interpret_with_direct_api(system_prompt, messages)
            else:
                # Direct API call
                response_text = await self._interpret_with_direct_api(system_prompt, messages)

            # Parse Claude's response (response_text is now a string)

            # Extract JSON from response
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                intent_data = json.loads(json_match.group())
            else:
                # Fallback parsing
                intent_data = self._fallback_parse(voice_input)

            # Create CommandIntent
            intent = CommandIntent(
                action=intent_data.get("action", "unknown"),
                target=intent_data.get("target", ""),
                parameters=intent_data.get("parameters", {}),
                confidence=float(intent_data.get("confidence", 0.5)),
                category=CommandCategory(intent_data.get("category", "application")),
                safety_level=self._assess_safety(intent_data),
                requires_confirmation=intent_data.get("requires_confirmation", False),
                raw_command=voice_input,
            )

            # Store in history
            self.conversation_history.append(
                {
                    "command": voice_input,
                    "response": intent_data.get("interpretation", ""),
                    "intent": intent,
                }
            )

            return intent

        except Exception as e:
            logger.error(f"Claude interpretation error: {e}")
            # Fallback to basic parsing
            return self._basic_parse(voice_input)

    async def _interpret_with_intelligent_selection(
        self, voice_input: str, context: Optional[Dict], system_prompt: str, messages: List[Dict]
    ) -> str:
        """Use intelligent model selection for command interpretation"""
        from backend.core.hybrid_orchestrator import HybridOrchestrator

        orchestrator = HybridOrchestrator()
        if not orchestrator.is_running:
            await orchestrator.start()

        # Build full prompt combining system prompt and user message
        full_prompt = f"{system_prompt}\n\nUser command: {voice_input}"
        if context:
            full_prompt += f"\nContext: {json.dumps(context)}"

        # Execute with intelligent model selection
        result = await orchestrator.execute_with_intelligent_model_selection(
            query=full_prompt,
            intent="command_interpretation",
            required_capabilities={"nlp_analysis", "intent_classification", "conversational_ai"},
            context={"conversation_history": self.conversation_history[-3:]},
            max_tokens=500,
            temperature=0.3,
        )

        if not result.get("success"):
            raise Exception(result.get("error", "Unknown error"))

        response_text = result.get("text", "").strip()
        if not response_text:
            raise Exception("Empty response from intelligent model selection")

        model_used = result.get("model_used", "unknown")
        logger.info(f"Command interpreted using {model_used}")

        return response_text

    async def _interpret_with_direct_api(self, system_prompt: str, messages: List[Dict]) -> str:
        """Direct Claude API call (fallback)"""
        response = self.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            temperature=0.3,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text

    def _assess_safety(self, intent_data: Dict) -> SafetyLevel:
        """Assess safety level of command"""
        action = intent_data.get("action", "")
        target = intent_data.get("target", "")

        # Handle None values
        if action is None:
            action = ""
        if target is None:
            target = ""

        action = action.lower()
        target = target.lower()

        # Dangerous actions
        if any(word in action for word in ["delete", "remove", "shutdown", "restart", "format"]):
            return SafetyLevel.DANGEROUS

        # Forbidden targets
        if any(app in target for app in ["terminal", "system preferences", "disk utility"]):
            return SafetyLevel.DANGEROUS

        # File operations need caution
        if intent_data.get("category") == "file":
            return SafetyLevel.CAUTION

        return SafetyLevel.SAFE

    def _basic_parse(self, voice_input: str) -> CommandIntent:
        """Basic parsing fallback when Claude is unavailable"""
        voice_lower = voice_input.lower()

        # Simple pattern matching
        if "open" in voice_lower:
            # Extract app name after "open"
            match = re.search(r"open\s+(\w+)", voice_lower)
            if match:
                return CommandIntent(
                    action="open_app",
                    target=match.group(1),
                    parameters={},
                    confidence=0.7,
                    category=CommandCategory.APPLICATION,
                    safety_level=SafetyLevel.SAFE,
                    requires_confirmation=False,
                    raw_command=voice_input,
                )

        elif "close" in voice_lower:
            match = re.search(r"close\s+(\w+)", voice_lower)
            if match:
                return CommandIntent(
                    action="close_app",
                    target=match.group(1),
                    parameters={},
                    confidence=0.7,
                    category=CommandCategory.APPLICATION,
                    safety_level=SafetyLevel.SAFE,
                    requires_confirmation=False,
                    raw_command=voice_input,
                )

        elif "volume" in voice_lower:
            # Extract volume level
            match = re.search(r"(\d+)%?", voice_lower)
            if match:
                return CommandIntent(
                    action="set_volume",
                    target="system",
                    parameters={"level": int(match.group(1))},
                    confidence=0.8,
                    category=CommandCategory.SYSTEM,
                    safety_level=SafetyLevel.SAFE,
                    requires_confirmation=False,
                    raw_command=voice_input,
                )

        # Default unknown command
        return CommandIntent(
            action="unknown",
            target="",
            parameters={},
            confidence=0.1,
            category=CommandCategory.APPLICATION,
            safety_level=SafetyLevel.SAFE,
            requires_confirmation=True,
            raw_command=voice_input,
        )

    def _fallback_parse(self, voice_input: str) -> Dict:
        """Fallback parsing to return dict format"""
        intent = self._basic_parse(voice_input)
        return {
            "action": intent.action,
            "target": intent.target,
            "parameters": intent.parameters,
            "category": intent.category.value,
            "confidence": intent.confidence,
            "requires_confirmation": intent.requires_confirmation,
            "interpretation": f"Execute {intent.action} on {intent.target}",
        }

    async def execute_intent(self, intent: CommandIntent) -> CommandResult:
        """Execute the interpreted command intent"""

        # Check safety
        if intent.safety_level == SafetyLevel.FORBIDDEN:
            return CommandResult(
                success=False,
                message="This command is forbidden for safety reasons",
                follow_up_needed=False,
            )

        # Check if confirmation needed (but not for vision commands)
        if intent.requires_confirmation or intent.safety_level == SafetyLevel.DANGEROUS:
            # Exception for vision commands - they're always safe to execute
            if intent.category != CommandCategory.VISION and intent.action not in [
                "describe_screen",
                "analyze_window",
                "check_screen",
            ]:
                return CommandResult(
                    success=False,
                    message=f"This command requires confirmation: {intent.action} {intent.target}",
                    data={"intent": intent},
                    follow_up_needed=True,
                )

        # Execute based on category
        try:
            if intent.category == CommandCategory.APPLICATION:
                return await self._execute_app_command(intent)
            elif intent.category == CommandCategory.FILE:
                return await self._execute_file_command(intent)
            elif intent.category == CommandCategory.SYSTEM:
                # Check if this is actually a vision command
                if intent.action in ["describe_screen", "analyze_window", "check_screen"]:
                    return await self._execute_vision_command(intent)
                return await self._execute_system_command(intent)
            elif intent.category == CommandCategory.VISION:
                return await self._execute_vision_command(intent)
            elif intent.category == CommandCategory.WEB:
                return await self._execute_web_command(intent)
            elif intent.category == CommandCategory.WORKFLOW:
                return await self._execute_workflow_command(intent)
            else:
                return CommandResult(
                    success=False, message=f"Unknown command category: {intent.category}"
                )
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return CommandResult(success=False, message=f"Error executing command: {str(e)}")

    async def _execute_app_command(self, intent: CommandIntent) -> CommandResult:
        """Execute application commands"""
        if intent.action == "open_app":
            # Check if target contains multiple apps separated by "and"
            if " and " in intent.target:
                apps = [app.strip() for app in intent.target.split(" and ")]
                results = []
                all_success = True

                for app in apps:
                    # Use dynamic controller for intelligent app discovery
                    success, msg = await self.dynamic_controller.open_app_intelligently(app)
                    if not success:
                        # Try fast launcher for common apps
                        if self.fast_launcher.is_common_app(app):
                            success, msg = await self.fast_launcher.quick_open_app(app)
                        else:
                            # Last resort: standard controller
                            success, msg = self.controller.open_application(app)
                    results.append(f"{app}: {msg}")
                    if not success:
                        all_success = False

                return CommandResult(success=all_success, message="; ".join(results))
            else:
                # Use dynamic controller for intelligent app discovery
                success, message = await self.dynamic_controller.open_app_intelligently(
                    intent.target
                )
                if not success:
                    # Try fast launcher for common apps
                    if self.fast_launcher.is_common_app(intent.target):
                        success, message = await self.fast_launcher.quick_open_app(intent.target)
                    else:
                        # Last resort: standard controller
                        success, message = self.controller.open_application(intent.target)

        elif intent.action == "close_app":
            # Check if target contains multiple apps separated by "and"
            if " and " in intent.target:
                apps = [app.strip() for app in intent.target.split(" and ")]
                results = []
                all_success = True

                for app in apps:
                    # Use dynamic controller for intelligent app discovery
                    success, msg = await self.dynamic_controller.close_app_intelligently(app)
                    if not success:
                        # Last resort: standard controller
                        success, msg = self.controller.close_application(app)
                    results.append(f"{app}: {msg}")
                    if not success:
                        all_success = False

                return CommandResult(success=all_success, message="; ".join(results))
            else:
                # Use dynamic controller for intelligent app discovery
                success, message = await self.dynamic_controller.close_app_intelligently(
                    intent.target
                )
                if not success:
                    # Last resort: standard controller
                    success, message = self.controller.close_application(intent.target)

        elif intent.action == "switch_to_app":
            success, message = self.controller.switch_to_application(intent.target)
        elif intent.action == "list_apps":
            apps = self.controller.list_open_applications()
            return CommandResult(
                success=True, message=f"Open applications: {', '.join(apps)}", data=apps
            )
        elif intent.action == "minimize_all":
            success, message = self.controller.minimize_all_windows()
        else:
            return CommandResult(
                success=False, message=f"Unknown application action: {intent.action}"
            )

        return CommandResult(success=success, message=message)

    async def _execute_file_command(self, intent: CommandIntent) -> CommandResult:
        """Execute file commands"""
        if intent.action == "open_file":
            file_path = intent.parameters.get("path", intent.target)
            success, message = self.controller.open_file(file_path)
        elif intent.action == "create_file":
            file_path = intent.parameters.get("path", intent.target)
            content = intent.parameters.get("content", "")
            success, message = self.controller.create_file(file_path, content)
        elif intent.action == "delete_file":
            file_path = intent.parameters.get("path", intent.target)
            # Always require confirmation for deletion
            return CommandResult(
                success=False,
                message=f"Confirm deletion of {file_path}?",
                data={"action": "delete", "path": file_path},
                follow_up_needed=True,
            )
        elif intent.action == "search_files":
            query = intent.target
            directory = intent.parameters.get("directory")
            results = self.controller.search_files(query, directory)
            return CommandResult(
                success=True,
                message=f"Found {len(results)} files",
                data=results[:10],  # Limit results
            )
        else:
            return CommandResult(success=False, message=f"Unknown file action: {intent.action}")

        return CommandResult(success=success, message=message)

    async def _execute_system_command(self, intent: CommandIntent) -> CommandResult:
        """Execute system commands"""
        if intent.action == "set_volume":
            level = intent.parameters.get("level", 50)
            success, message = self.controller.set_volume(level)
        elif intent.action == "mute":
            mute = intent.parameters.get("mute", True)
            success, message = self.controller.mute_volume(mute)
        elif intent.action == "screenshot":
            path = intent.parameters.get("path")
            success, message = self.controller.take_screenshot(path)
        elif intent.action == "sleep_display":
            success, message = self.controller.sleep_display()
        elif intent.action == "toggle_wifi":
            enable = intent.parameters.get("enable", True)
            success, message = self.controller.toggle_wifi(enable)
        else:
            return CommandResult(success=False, message=f"Unknown system action: {intent.action}")

        return CommandResult(success=success, message=message)

    async def _execute_web_command(self, intent: CommandIntent) -> CommandResult:
        """Execute web commands"""
        if intent.action == "open_url":
            url = intent.target
            browser = intent.parameters.get("browser")
            success, message = self.controller.open_url(url, browser)
        elif intent.action == "web_search":
            query = intent.target
            engine = intent.parameters.get("engine", "google")
            success, message = self.controller.web_search(query, engine)
        else:
            return CommandResult(success=False, message=f"Unknown web action: {intent.action}")

        return CommandResult(success=success, message=message)

    async def _execute_workflow_command(self, intent: CommandIntent) -> CommandResult:
        """Execute workflow commands"""
        workflow_name = intent.target or intent.action

        # Special handling for weather workflow
        if workflow_name == "check_weather_app" or "weather" in intent.raw_command.lower():
            try:
                # Import locally to avoid circular import
                from backend.workflows.weather_app_vision_unified import (
                    execute_weather_app_workflow,
                )

                # Use the unified weather app vision workflow
                query = intent.raw_command if intent.raw_command else "What's the weather today?"
                message = await execute_weather_app_workflow(
                    self.controller, self.vision_handler, query
                )
                return CommandResult(success=True, message=message)
            except Exception as e:
                logger.error(f"Weather workflow error: {e}")
                # Return weather-specific error instead of fallback
                return CommandResult(
                    success=False, message="I need to access the Weather app to check the forecast."
                )

        # Regular workflow execution
        success, message = await self.controller.execute_workflow(workflow_name)
        return CommandResult(success=success, message=message)

    async def _execute_vision_command(self, intent: CommandIntent) -> CommandResult:
        """Execute vision commands"""
        try:
            # Add the raw command as query parameter if not already present
            if "query" not in intent.parameters:
                intent.parameters["query"] = intent.raw_command

            if intent.action == "describe_screen":
                result = await self.vision_handler.describe_screen(intent.parameters)
                return CommandResult(
                    success=result.success, message=result.description, data=result.data
                )
            elif intent.action == "analyze_window":
                result = await self.vision_handler.analyze_window(intent.parameters)
                return CommandResult(
                    success=result.success, message=result.description, data=result.data
                )
            elif intent.action == "check_screen":
                result = await self.vision_handler.check_screen(intent.parameters)
                return CommandResult(
                    success=result.success, message=result.description, data=result.data
                )
            else:
                # Try to handle any vision command through describe_screen
                result = await self.vision_handler.describe_screen(intent.parameters)
                return CommandResult(
                    success=result.success, message=result.description, data=result.data
                )
        except Exception as e:
            logger.error(f"Vision command error: {e}", exc_info=True)
            return CommandResult(success=False, message=f"Error executing vision command: {str(e)}")

    def get_suggestions(self, partial_command: str) -> List[str]:
        """Get command suggestions based on partial input"""
        suggestions = []

        # Search through command examples
        for category, examples in self.command_examples.items():
            for example in examples:
                if partial_command.lower() in example.lower():
                    suggestions.append(example)

        return suggestions[:5]  # Return top 5 suggestions

    def learn_from_feedback(
        self, intent: CommandIntent, success: bool, user_feedback: Optional[str] = None
    ):
        """Learn from command execution feedback"""
        # In a real implementation, this would update a learning model
        logger.info(f"Learning from feedback: {intent.action} - Success: {success}")
        if user_feedback:
            logger.info(f"User feedback: {user_feedback}")

    def get_help(self, topic: Optional[str] = None) -> str:
        """Get help on available commands"""
        if not topic:
            return """JARVIS System Control Commands:

Applications: Open, close, or switch between applications
Files: Create, open, delete, or search for files
System: Control volume, WiFi, screenshots, and display
Web: Open URLs or perform web searches
Workflows: Execute predefined routines

Say "Help with [topic]" for specific help."""

        topic_lower = topic.lower()
        if "application" in topic_lower or "app" in topic_lower:
            return "Application commands: 'Open Chrome', 'Close Spotify', 'Switch to Mail'"
        elif "file" in topic_lower:
            return "File commands: 'Create a file', 'Open document.pdf', 'Search for Python files'"
        elif "system" in topic_lower:
            return "System commands: 'Set volume to 50%', 'Take a screenshot', 'Turn off WiFi'"
        elif "web" in topic_lower:
            return "Web commands: 'Search for recipes', 'Open YouTube', 'Go to github.com'"
        elif "workflow" in topic_lower:
            return "Workflows: 'Start morning routine', 'Set up development environment'"
        else:
            return f"No specific help available for '{topic}'"
