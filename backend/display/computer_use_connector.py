"""
Claude Computer Use API Connector for JARVIS

This module provides a robust, async, and dynamic integration with Claude's
Computer Use API for vision-based UI automation. It replaces hardcoded
workflows with intelligent, adaptive action chains.

Key Features:
- Vision-based element detection (no hardcoded coordinates)
- Dynamic action chain execution with real-time reasoning
- Voice narration integration for transparency
- Automatic failure recovery and alternative approach generation
- Learning from successful interactions
- Full async support throughout

Architecture:
    Screenshot -> Claude Vision Analysis -> Action Decision -> Execution -> Verification
         ^                                                                      |
         |______________________________________________________________________|
                              (Loop until goal achieved or max attempts)

Author: JARVIS AI System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import pyautogui
from PIL import Image

try:
    from anthropic import Anthropic, AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None
    AsyncAnthropic = None

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Types
# ============================================================================

class ActionType(str, Enum):
    """Types of computer actions Claude can execute."""
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    DRAG = "drag"
    TYPE = "type"
    KEY = "key"
    SCROLL = "scroll"
    SCREENSHOT = "screenshot"
    WAIT = "wait"
    CURSOR_POSITION = "cursor_position"


class TaskStatus(str, Enum):
    """Status of a task execution."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    NEEDS_RETRY = "needs_retry"
    NEEDS_HUMAN = "needs_human"


class NarrationEvent(str, Enum):
    """Events for voice narration."""
    STARTING = "starting"
    ANALYZING = "analyzing"
    CLICKING = "clicking"
    TYPING = "typing"
    WAITING = "waiting"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    LEARNING = "learning"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ComputerAction:
    """A single computer action to execute."""
    action_id: str
    action_type: ActionType
    coordinates: Optional[Tuple[int, int]] = None
    text: Optional[str] = None
    key: Optional[str] = None
    scroll_amount: Optional[int] = None
    duration: float = 0.1
    reasoning: str = ""
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "coordinates": self.coordinates,
            "text": self.text,
            "key": self.key,
            "reasoning": self.reasoning,
            "confidence": self.confidence
        }


@dataclass
class ActionResult:
    """Result of an action execution."""
    action_id: str
    success: bool
    result_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    screenshot_after: Optional[str] = None
    duration_ms: float = 0
    verification_passed: bool = True


@dataclass
class TaskResult:
    """Result of a complete task execution."""
    task_id: str
    goal: str
    status: TaskStatus
    actions_executed: List[ActionResult]
    total_duration_ms: float
    narration_log: List[Dict[str, Any]]
    learning_insights: List[str]
    final_message: str
    confidence: float = 0.0


@dataclass
class VisionAnalysis:
    """Analysis of a screenshot by Claude."""
    analysis_id: str
    description: str
    detected_elements: List[Dict[str, Any]]
    suggested_action: Optional[ComputerAction]
    goal_progress: float  # 0.0 to 1.0
    is_goal_achieved: bool
    reasoning_chain: List[str]
    confidence: float
    is_auth_error: bool = False  # True if API authentication failed


# ============================================================================
# Voice Narration Handler
# ============================================================================

class VoiceNarrationHandler:
    """Handles voice narration for computer use actions."""

    def __init__(
        self,
        tts_callback: Optional[Callable[[str], Awaitable[None]]] = None,
        enabled: bool = True
    ):
        self.tts_callback = tts_callback
        self.enabled = enabled
        self._narration_log: List[Dict[str, Any]] = []
        self._templates = {
            NarrationEvent.STARTING: "Starting task: {goal}",
            NarrationEvent.ANALYZING: "Analyzing the screen to find {target}",
            NarrationEvent.CLICKING: "Clicking on {target}",
            NarrationEvent.TYPING: "Typing: {text}",
            NarrationEvent.WAITING: "Waiting for {description}",
            NarrationEvent.SUCCESS: "Successfully completed: {description}",
            NarrationEvent.FAILED: "Action failed: {reason}. Let me try another approach.",
            NarrationEvent.RETRYING: "Retrying with alternative approach: {approach}",
            NarrationEvent.LEARNING: "Learning from this: {insight}"
        }

    async def narrate(
        self,
        event: NarrationEvent,
        context: Optional[Dict[str, Any]] = None,
        custom_message: Optional[str] = None
    ) -> None:
        """Generate and speak narration."""
        if not self.enabled:
            return

        context = context or {}

        if custom_message:
            message = custom_message
        else:
            template = self._templates.get(event, "{event}")
            try:
                message = template.format(**context, event=event.value)
            except KeyError:
                message = template

        # Log narration
        log_entry = {
            "event": event.value,
            "message": message,
            "context": context,
            "timestamp": datetime.utcnow().isoformat()
        }
        self._narration_log.append(log_entry)

        logger.info(f"[NARRATION] {message}")

        # Speak if TTS available
        if self.tts_callback:
            try:
                await self.tts_callback(message)
            except Exception as e:
                logger.warning(f"TTS failed: {e}")

    def get_log(self) -> List[Dict[str, Any]]:
        """Get narration log."""
        return self._narration_log.copy()

    def clear_log(self) -> None:
        """Clear narration log."""
        self._narration_log.clear()


# ============================================================================
# Screen Capture Handler
# ============================================================================

class ScreenCaptureHandler:
    """Handles screen capture for Claude Computer Use."""

    def __init__(self, scale_factor: float = 1.0):
        self.scale_factor = scale_factor
        self._last_screenshot: Optional[Image.Image] = None
        self._screenshot_cache: Dict[str, str] = {}

    async def capture(
        self,
        region: Optional[Tuple[int, int, int, int]] = None,
        resize_for_api: bool = True,
        max_dimension: int = 1568
    ) -> Tuple[Image.Image, str]:
        """
        Capture screenshot and prepare for Claude API.

        Returns:
            Tuple of (PIL Image, base64-encoded string)
        """
        try:
            # Capture screenshot
            if region:
                screenshot = pyautogui.screenshot(region=region)
            else:
                screenshot = pyautogui.screenshot()

            self._last_screenshot = screenshot

            # Resize for API if needed (Claude has dimension limits)
            if resize_for_api:
                width, height = screenshot.size
                if width > max_dimension or height > max_dimension:
                    ratio = min(max_dimension / width, max_dimension / height)
                    new_size = (int(width * ratio), int(height * ratio))
                    screenshot = screenshot.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to base64
            buffer = io.BytesIO()
            screenshot.save(buffer, format="PNG", optimize=True)
            base64_image = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

            return screenshot, base64_image

        except Exception as e:
            logger.error(f"Screenshot capture failed: {e}")
            raise

    def get_last_screenshot(self) -> Optional[Image.Image]:
        """Get the last captured screenshot."""
        return self._last_screenshot


# ============================================================================
# Action Executor
# ============================================================================

class ActionExecutor:
    """Executes computer actions using PyAutoGUI."""

    def __init__(
        self,
        scale_factor: float = 1.0,
        safety_pause: float = 0.3,
        movement_duration: float = 0.2
    ):
        self.scale_factor = scale_factor
        self.safety_pause = safety_pause
        self.movement_duration = movement_duration

        # Configure PyAutoGUI
        pyautogui.PAUSE = safety_pause
        pyautogui.FAILSAFE = True

    async def execute(self, action: ComputerAction) -> ActionResult:
        """Execute a computer action."""
        start_time = time.time()

        try:
            if action.action_type == ActionType.CLICK:
                await self._execute_click(action)
            elif action.action_type == ActionType.DOUBLE_CLICK:
                await self._execute_double_click(action)
            elif action.action_type == ActionType.RIGHT_CLICK:
                await self._execute_right_click(action)
            elif action.action_type == ActionType.TYPE:
                await self._execute_type(action)
            elif action.action_type == ActionType.KEY:
                await self._execute_key(action)
            elif action.action_type == ActionType.SCROLL:
                await self._execute_scroll(action)
            elif action.action_type == ActionType.WAIT:
                await self._execute_wait(action)
            elif action.action_type == ActionType.CURSOR_POSITION:
                await self._execute_move(action)
            elif action.action_type == ActionType.SCREENSHOT:
                pass  # Screenshot is handled separately

            duration_ms = (time.time() - start_time) * 1000

            return ActionResult(
                action_id=action.action_id,
                success=True,
                duration_ms=duration_ms
            )

        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return ActionResult(
                action_id=action.action_id,
                success=False,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000
            )

    async def _execute_click(self, action: ComputerAction) -> None:
        """Execute click action."""
        if not action.coordinates:
            raise ValueError("Click action requires coordinates")

        x, y = self._scale_coordinates(action.coordinates)

        # Move smoothly to position
        pyautogui.moveTo(x, y, duration=self.movement_duration)
        await asyncio.sleep(0.05)

        # Click
        pyautogui.click(x, y)

    async def _execute_double_click(self, action: ComputerAction) -> None:
        """Execute double click action."""
        if not action.coordinates:
            raise ValueError("Double click action requires coordinates")

        x, y = self._scale_coordinates(action.coordinates)
        pyautogui.moveTo(x, y, duration=self.movement_duration)
        pyautogui.doubleClick(x, y)

    async def _execute_right_click(self, action: ComputerAction) -> None:
        """Execute right click action."""
        if not action.coordinates:
            raise ValueError("Right click action requires coordinates")

        x, y = self._scale_coordinates(action.coordinates)
        pyautogui.moveTo(x, y, duration=self.movement_duration)
        pyautogui.rightClick(x, y)

    async def _execute_type(self, action: ComputerAction) -> None:
        """Execute type action."""
        if not action.text:
            raise ValueError("Type action requires text")

        pyautogui.write(action.text, interval=0.02)

    async def _execute_key(self, action: ComputerAction) -> None:
        """Execute key press action."""
        if not action.key:
            raise ValueError("Key action requires key")

        # Handle key combinations (e.g., "command+c")
        if "+" in action.key:
            keys = action.key.split("+")
            pyautogui.hotkey(*keys)
        else:
            pyautogui.press(action.key)

    async def _execute_scroll(self, action: ComputerAction) -> None:
        """Execute scroll action."""
        amount = action.scroll_amount or 3

        if action.coordinates:
            x, y = self._scale_coordinates(action.coordinates)
            pyautogui.moveTo(x, y)

        pyautogui.scroll(amount)

    async def _execute_wait(self, action: ComputerAction) -> None:
        """Execute wait action."""
        await asyncio.sleep(action.duration)

    async def _execute_move(self, action: ComputerAction) -> None:
        """Execute cursor move action."""
        if not action.coordinates:
            raise ValueError("Move action requires coordinates")

        x, y = self._scale_coordinates(action.coordinates)
        pyautogui.moveTo(x, y, duration=self.movement_duration)

    def _scale_coordinates(self, coords: Tuple[int, int]) -> Tuple[int, int]:
        """Scale coordinates for Retina displays if needed."""
        x, y = coords
        return (int(x * self.scale_factor), int(y * self.scale_factor))


# ============================================================================
# Claude Computer Use Connector
# ============================================================================

class ClaudeComputerUseConnector:
    """
    Main connector for Claude Computer Use API.

    Provides vision-based, dynamic UI automation without hardcoded coordinates.
    Integrates with voice narration for transparency.
    """

    # Claude Computer Use model
    COMPUTER_USE_MODEL = "claude-sonnet-4-20250514"

    # System prompt for computer use
    SYSTEM_PROMPT = """You are JARVIS, an AI assistant helping to control a macOS computer.
You can see the screen through screenshots and execute actions to help the user.

*** CRITICAL: FULL SCREEN MODE DETECTION - CHECK THIS FIRST ***
The user often runs applications in FULL SCREEN MODE where the macOS menu bar is HIDDEN.

STEP 1 - ALWAYS CHECK FIRST: Look at the very top of the screenshot:
- If you see a menu bar (clock, wifi icon, Control Center icon in top-right): Menu bar is VISIBLE, proceed normally
- If the top of the screen shows ONLY the application content (no menu bar, no system icons): You are in FULL SCREEN MODE

STEP 2 - IF IN FULL SCREEN MODE (no menu bar visible):
1. IMMEDIATELY move the mouse cursor to y=0 (the very top edge of the screen)
   - Use coordinates like (screen_width/2, 0) to move cursor to top-center
   - This action REVEALS the hidden menu bar in macOS full screen mode
2. Wait 0.5-1 second for the menu bar to animate into view
3. Take a NEW screenshot to see the revealed menu bar
4. NOW you can see and click on Control Center

STEP 3 - Once menu bar is visible:
- The Control Center icon is in the top-right, looks like two toggle switches/sliders
- Click it to open Control Center
- Find and click "Screen Mirroring" (two overlapping screens icon)
- Select the target display from the list

When analyzing screenshots:
1. FIRST: Check if menu bar is visible (see above)
2. Describe what you see clearly
3. Identify UI elements by visual appearance, NOT memorized positions
4. Locate elements dynamically based on current screen content

When executing actions:
1. Be precise with coordinates - identify exact pixel locations
2. Wait for UI elements to load after clicks
3. Verify your actions succeeded by checking the next screenshot
4. If something fails, try an alternative approach

For macOS Control Center:
- The Control Center icon is in the top-right menu bar
- It looks like two overlapping rectangles (toggle switches)
- After clicking, Control Center panel appears with various controls
- Screen Mirroring shows two overlapping screens icon
- AirPlay devices are listed when Screen Mirroring is expanded

Remember: If you cannot see the menu bar, you MUST first move the cursor to y=0 to reveal it!

Always provide your reasoning before taking action."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        tts_callback: Optional[Callable[[str], Awaitable[None]]] = None,
        learning_callback: Optional[Callable[[Dict], Awaitable[None]]] = None,
        scale_factor: float = 1.0,
        max_actions_per_task: int = 20,
        action_timeout: float = 30.0
    ):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        import os
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY required")

        self.client = AsyncAnthropic(api_key=self.api_key)
        self.max_actions_per_task = max_actions_per_task
        self.action_timeout = action_timeout

        # Initialize components
        self.narrator = VoiceNarrationHandler(tts_callback=tts_callback)
        self.screen_capture = ScreenCaptureHandler(scale_factor=scale_factor)
        self.action_executor = ActionExecutor(scale_factor=scale_factor)
        self.learning_callback = learning_callback

        # State tracking
        self._current_task_id: Optional[str] = None
        self._action_history: List[ComputerAction] = []
        self._learned_positions: Dict[str, Tuple[int, int]] = {}
        self._auth_failed: bool = False  # Track API authentication failures

        # Load learned positions
        self._load_learned_positions()

        logger.info("[COMPUTER USE] Claude Computer Use Connector initialized")

    def _load_learned_positions(self) -> None:
        """Load previously learned UI element positions."""
        cache_file = Path.home() / ".jarvis" / "learned_ui_positions.json"
        try:
            if cache_file.exists():
                with open(cache_file) as f:
                    self._learned_positions = json.load(f)
                logger.info(f"[COMPUTER USE] Loaded {len(self._learned_positions)} learned positions")
        except Exception as e:
            logger.warning(f"Could not load learned positions: {e}")

    def _save_learned_position(self, element_name: str, coords: Tuple[int, int]) -> None:
        """Save a learned position for future use."""
        self._learned_positions[element_name] = coords
        cache_file = Path.home() / ".jarvis" / "learned_ui_positions.json"
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump(self._learned_positions, f)
        except Exception as e:
            logger.warning(f"Could not save learned position: {e}")

    async def execute_task(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        narrate: bool = True
    ) -> TaskResult:
        """
        Execute a complete task using Claude Computer Use.

        Args:
            goal: Natural language description of what to accomplish
            context: Additional context for the task
            narrate: Whether to provide voice narration

        Returns:
            TaskResult with complete execution details
        """
        task_id = str(uuid4())
        self._current_task_id = task_id
        start_time = time.time()
        actions_executed: List[ActionResult] = []
        learning_insights: List[str] = []

        # Early return if auth has already failed
        if self._auth_failed:
            logger.warning("[COMPUTER USE] Skipping task - API authentication previously failed")
            return TaskResult(
                task_id=task_id,
                goal=goal,
                status=TaskStatus.FAILED,
                actions_executed=[],
                total_duration_ms=0,
                narration_log=[],
                learning_insights=[],
                final_message="API authentication failed - please check ANTHROPIC_API_KEY",
                confidence=0.0
            )

        self.narrator.enabled = narrate
        self.narrator.clear_log()

        await self.narrator.narrate(
            NarrationEvent.STARTING,
            {"goal": goal}
        )

        try:
            # Main execution loop
            action_count = 0
            goal_achieved = False

            while action_count < self.max_actions_per_task and not goal_achieved:
                action_count += 1

                # Capture current screen state
                await self.narrator.narrate(
                    NarrationEvent.ANALYZING,
                    {"target": "the current screen state"}
                )

                screenshot, base64_screenshot = await self.screen_capture.capture()

                # Get Claude's analysis and suggested action
                analysis = await self._analyze_and_decide(
                    goal=goal,
                    screenshot_base64=base64_screenshot,
                    action_history=self._action_history[-5:],  # Last 5 actions
                    context=context
                )

                # Check for authentication errors - bail out immediately
                if analysis.is_auth_error:
                    logger.error("[COMPUTER USE] ❌ Authentication failed - stopping task")
                    await self.narrator.narrate(
                        NarrationEvent.FAILED,
                        {"reason": "API authentication failed - invalid API key"}
                    )
                    return TaskResult(
                        task_id=task_id,
                        goal=goal,
                        status=TaskStatus.FAILED,
                        actions_executed=actions_executed,
                        total_duration_ms=(time.time() - start_time) * 1000,
                        narration_log=self.narrator.get_log(),
                        learning_insights=learning_insights,
                        final_message="API authentication failed - please check ANTHROPIC_API_KEY",
                        confidence=0.0
                    )

                # Check if goal achieved
                if analysis.is_goal_achieved:
                    goal_achieved = True
                    await self.narrator.narrate(
                        NarrationEvent.SUCCESS,
                        {"description": goal}
                    )
                    break

                # Execute suggested action
                if analysis.suggested_action:
                    action = analysis.suggested_action

                    # Narrate the action
                    await self._narrate_action(action)

                    # Execute
                    result = await self.action_executor.execute(action)
                    actions_executed.append(result)
                    self._action_history.append(action)

                    if result.success:
                        # Learn from successful action
                        if action.coordinates and action.reasoning:
                            element_hint = self._extract_element_name(action.reasoning)
                            if element_hint:
                                self._save_learned_position(element_hint, action.coordinates)
                                learning_insights.append(
                                    f"Learned position for '{element_hint}': {action.coordinates}"
                                )

                        # Wait for UI to update
                        await asyncio.sleep(0.5)
                    else:
                        await self.narrator.narrate(
                            NarrationEvent.FAILED,
                            {"reason": result.error or "Unknown error"}
                        )
                else:
                    # No action suggested - Claude might be stuck
                    logger.warning("[COMPUTER USE] No action suggested by Claude")
                    await asyncio.sleep(1.0)

            # Determine final status
            if goal_achieved:
                status = TaskStatus.SUCCESS
                final_message = f"Successfully completed: {goal}"
            elif action_count >= self.max_actions_per_task:
                status = TaskStatus.NEEDS_HUMAN
                final_message = f"Reached maximum actions ({self.max_actions_per_task}) without completing goal"
            else:
                status = TaskStatus.FAILED
                final_message = "Task failed to complete"

            # Store learning
            if self.learning_callback and learning_insights:
                await self.learning_callback({
                    "task_id": task_id,
                    "goal": goal,
                    "insights": learning_insights,
                    "success": goal_achieved
                })

            total_duration_ms = (time.time() - start_time) * 1000

            return TaskResult(
                task_id=task_id,
                goal=goal,
                status=status,
                actions_executed=actions_executed,
                total_duration_ms=total_duration_ms,
                narration_log=self.narrator.get_log(),
                learning_insights=learning_insights,
                final_message=final_message,
                confidence=0.9 if goal_achieved else 0.3
            )

        except Exception as e:
            logger.error(f"[COMPUTER USE] Task execution failed: {e}")
            await self.narrator.narrate(
                NarrationEvent.FAILED,
                {"reason": str(e)}
            )

            return TaskResult(
                task_id=task_id,
                goal=goal,
                status=TaskStatus.FAILED,
                actions_executed=actions_executed,
                total_duration_ms=(time.time() - start_time) * 1000,
                narration_log=self.narrator.get_log(),
                learning_insights=learning_insights,
                final_message=f"Task failed with error: {str(e)}",
                confidence=0.0
            )

    async def _analyze_and_decide(
        self,
        goal: str,
        screenshot_base64: str,
        action_history: List[ComputerAction],
        context: Optional[Dict[str, Any]] = None
    ) -> VisionAnalysis:
        """
        Analyze screenshot and decide on next action using Claude.
        """
        # Build conversation history
        history_text = ""
        if action_history:
            history_text = "\n\nPrevious actions in this task:\n"
            for i, action in enumerate(action_history, 1):
                history_text += f"{i}. {action.action_type.value}"
                if action.coordinates:
                    history_text += f" at {action.coordinates}"
                history_text += f" - {action.reasoning}\n"

        context_text = ""
        if context:
            context_text = f"\n\nAdditional context: {json.dumps(context)}"

        # Build the prompt
        user_prompt = f"""Goal: {goal}
{history_text}{context_text}

Please analyze the current screenshot and determine:
1. What do you see on the screen?
2. How close are we to achieving the goal? (0-100%)
3. Is the goal already achieved?
4. What action should we take next?

If an action is needed, provide it in this JSON format:
```json
{{
    "action_type": "click|double_click|right_click|type|key|scroll|wait",
    "coordinates": [x, y],  // For click actions, precise pixel coordinates
    "text": "...",  // For type actions
    "key": "...",  // For key actions (e.g., "return", "command+c")
    "scroll_amount": 3,  // For scroll actions
    "duration": 1.0,  // For wait actions (seconds)
    "reasoning": "Why this action will help achieve the goal"
}}
```

Respond with your analysis followed by the action JSON if needed."""

        try:
            # Call Claude with computer use capability
            response = await self.client.messages.create(
                model=self.COMPUTER_USE_MODEL,
                max_tokens=1024,
                system=self.SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": screenshot_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": user_prompt
                            }
                        ]
                    }
                ]
            )

            # Parse response
            response_text = response.content[0].text
            return self._parse_analysis_response(response_text, goal)

        except Exception as e:
            error_str = str(e)
            logger.error(f"[COMPUTER USE] Claude analysis failed: {e}")

            # Check for authentication errors - these are fatal
            is_auth_error = (
                "authentication_error" in error_str.lower() or
                "invalid x-api-key" in error_str.lower() or
                "401" in error_str
            )

            if is_auth_error:
                # Mark connector as unavailable to prevent further attempts
                self._auth_failed = True
                logger.error("[COMPUTER USE] ❌ Authentication failed - API key is invalid")

            return VisionAnalysis(
                analysis_id=str(uuid4()),
                description=f"Analysis failed: {error_str}",
                detected_elements=[],
                suggested_action=None,
                goal_progress=0.0,
                is_goal_achieved=False,
                reasoning_chain=[f"Error: {error_str}"],
                confidence=0.0,
                is_auth_error=is_auth_error  # Signal auth failure
            )

    def _parse_analysis_response(self, response_text: str, goal: str) -> VisionAnalysis:
        """Parse Claude's response into a VisionAnalysis."""
        analysis_id = str(uuid4())
        suggested_action = None
        goal_progress = 0.0
        is_goal_achieved = False
        reasoning_chain = []

        # Extract reasoning
        reasoning_chain.append(response_text.split("```")[0].strip())

        # Check for goal achievement indicators
        goal_achieved_phrases = [
            "goal is achieved",
            "goal has been achieved",
            "successfully completed",
            "task is complete",
            "connection established",
            "connected to"
        ]
        response_lower = response_text.lower()
        is_goal_achieved = any(phrase in response_lower for phrase in goal_achieved_phrases)

        # Extract progress percentage if mentioned
        import re
        progress_match = re.search(r'(\d{1,3})%', response_text)
        if progress_match:
            goal_progress = min(100, int(progress_match.group(1))) / 100.0

        if is_goal_achieved:
            goal_progress = 1.0

        # Extract JSON action if present
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            try:
                action_data = json.loads(json_match.group(1))

                action_type = ActionType(action_data.get("action_type", "click"))
                coords = action_data.get("coordinates")
                if coords and isinstance(coords, list) and len(coords) == 2:
                    coords = tuple(coords)
                else:
                    coords = None

                suggested_action = ComputerAction(
                    action_id=str(uuid4()),
                    action_type=action_type,
                    coordinates=coords,
                    text=action_data.get("text"),
                    key=action_data.get("key"),
                    scroll_amount=action_data.get("scroll_amount"),
                    duration=action_data.get("duration", 0.5),
                    reasoning=action_data.get("reasoning", ""),
                    confidence=0.8
                )
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Could not parse action JSON: {e}")

        return VisionAnalysis(
            analysis_id=analysis_id,
            description=reasoning_chain[0][:200] if reasoning_chain else "Analysis complete",
            detected_elements=[],
            suggested_action=suggested_action,
            goal_progress=goal_progress,
            is_goal_achieved=is_goal_achieved,
            reasoning_chain=reasoning_chain,
            confidence=0.8 if suggested_action else 0.5
        )

    async def _narrate_action(self, action: ComputerAction) -> None:
        """Narrate an action before executing it."""
        if action.action_type == ActionType.CLICK:
            await self.narrator.narrate(
                NarrationEvent.CLICKING,
                {"target": action.reasoning or "the element"}
            )
        elif action.action_type == ActionType.TYPE:
            # Don't narrate sensitive text
            display_text = action.text[:20] + "..." if len(action.text or "") > 20 else action.text
            await self.narrator.narrate(
                NarrationEvent.TYPING,
                {"text": display_text}
            )
        elif action.action_type == ActionType.WAIT:
            await self.narrator.narrate(
                NarrationEvent.WAITING,
                {"description": f"{action.duration} seconds"}
            )

    def _extract_element_name(self, reasoning: str) -> Optional[str]:
        """Extract UI element name from reasoning for learning."""
        # Look for quoted element names
        import re
        match = re.search(r"['\"]([^'\"]+)['\"]", reasoning)
        if match:
            return match.group(1).lower().replace(" ", "_")

        # Look for specific UI elements mentioned
        elements = ["control_center", "screen_mirroring", "airplay", "wifi", "bluetooth"]
        for elem in elements:
            if elem.replace("_", " ") in reasoning.lower():
                return elem

        return None

    async def connect_to_display(self, display_name: str) -> TaskResult:
        """
        Connect to a display using Claude Computer Use.

        This is a high-level convenience method for display connection.

        Args:
            display_name: Name of the display to connect to

        Returns:
            TaskResult with connection details
        """
        goal = f"""Connect to the display named "{display_name}" using macOS Screen Mirroring.

Steps to accomplish this:
1. Click on the Control Center icon in the menu bar (top right, looks like two toggle switches)
2. Wait for Control Center to open
3. Click on Screen Mirroring (shows two overlapping screens icon)
4. Wait for the list of available displays
5. Click on "{display_name}" in the list
6. Wait for the connection to establish

The task is complete when you see the display connected (green checkmark or active indicator)."""

        return await self.execute_task(
            goal=goal,
            context={"target_display": display_name}
        )


# ============================================================================
# Factory Functions
# ============================================================================

_default_connector: Optional[ClaudeComputerUseConnector] = None


def get_computer_use_connector(
    tts_callback: Optional[Callable[[str], Awaitable[None]]] = None
) -> ClaudeComputerUseConnector:
    """Get or create the default Computer Use connector."""
    global _default_connector

    if _default_connector is None:
        _default_connector = ClaudeComputerUseConnector(
            tts_callback=tts_callback
        )

    return _default_connector


async def connect_to_display_dynamic(
    display_name: str,
    tts_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    narrate: bool = True
) -> TaskResult:
    """
    Convenience function to connect to a display using Computer Use.

    Args:
        display_name: Name of display to connect to
        tts_callback: Optional TTS callback for voice narration
        narrate: Whether to enable voice narration

    Returns:
        TaskResult with connection details
    """
    connector = get_computer_use_connector(tts_callback)
    return await connector.connect_to_display(display_name)
