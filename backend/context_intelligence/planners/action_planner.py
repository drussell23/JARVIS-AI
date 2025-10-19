"""
Action Planner for JARVIS
==========================

Plans action execution with implicit reference resolution

Key Features:
1. **Reference Resolution** - Uses implicit_reference_resolver to resolve "it", "that", "the error"
2. **Context Integration** - Integrates with context graph for workspace awareness
3. **Step Planning** - Breaks down complex actions into atomic steps
4. **Dependency Management** - Handles action dependencies
5. **Safety Validation** - Validates actions are safe to execute

Author: Derek Russell
Date: 2025-10-19
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from backend.context_intelligence.analyzers.action_analyzer import (
    ActionIntent,
    ActionType,
    ActionTarget,
    ActionSafety
)

logger = logging.getLogger(__name__)


# ============================================================================
# EXECUTION PLAN
# ============================================================================

class StepStatus(Enum):
    """Status of an execution step"""
    PENDING = "pending"
    RESOLVING = "resolving"
    READY = "ready"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ExecutionStep:
    """A single step in an action execution plan"""
    step_id: str
    description: str
    action_type: str  # yabai, applescript, shell, etc.
    command: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)  # Step IDs this depends on
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """Complete execution plan for an action"""
    plan_id: str
    original_query: str
    action_intent: ActionIntent
    steps: List[ExecutionStep]
    resolved_references: Dict[str, Any] = field(default_factory=dict)
    safety_level: ActionSafety = ActionSafety.NEEDS_CONFIRMATION
    requires_confirmation: bool = True
    estimated_duration: float = 0.0  # seconds
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_step(self, step_id: str) -> Optional[ExecutionStep]:
        """Get a step by ID"""
        return next((s for s in self.steps if s.step_id == step_id), None)

    def get_pending_steps(self) -> List[ExecutionStep]:
        """Get all pending steps with satisfied dependencies"""
        pending = []
        for step in self.steps:
            if step.status != StepStatus.PENDING:
                continue

            # Check if dependencies are satisfied
            deps_satisfied = all(
                self.get_step(dep_id).status == StepStatus.COMPLETED
                for dep_id in step.depends_on
            )

            if deps_satisfied:
                pending.append(step)

        return pending


# ============================================================================
# ACTION PLANNER
# ============================================================================

class ActionPlanner:
    """
    Plans action execution with implicit reference resolution

    Works with:
    - implicit_reference_resolver for resolving ambiguous references
    - context_graph for workspace awareness
    - action_analyzer for intent understanding
    """

    def __init__(self, context_graph=None, implicit_resolver=None):
        """Initialize the action planner"""
        self.context_graph = context_graph
        self.implicit_resolver = implicit_resolver
        self._plan_counter = 0

        logger.info("[ACTION-PLANNER] Initialized")

    async def create_plan(
        self,
        action_intent: ActionIntent,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        """
        Create an execution plan for an action

        Args:
            action_intent: Parsed action intent
            context: Additional context

        Returns:
            ExecutionPlan with steps to execute
        """
        logger.info(f"[ACTION-PLANNER] Creating plan for: {action_intent.action_type.value}")

        self._plan_counter += 1
        plan_id = f"plan_{self._plan_counter}_{int(datetime.now().timestamp())}"

        # Step 1: Resolve implicit references if needed
        resolved_refs = {}
        if action_intent.requires_resolution:
            resolved_refs = await self._resolve_references(action_intent, context)
            logger.info(f"[ACTION-PLANNER] Resolved references: {list(resolved_refs.keys())}")

        # Step 2: Generate execution steps
        steps = await self._generate_steps(action_intent, resolved_refs, context)

        # Step 3: Determine overall safety level
        safety_level = self._determine_safety_level(action_intent, steps)

        # Step 4: Calculate if confirmation is needed
        requires_confirmation = safety_level in [ActionSafety.NEEDS_CONFIRMATION, ActionSafety.RISKY]

        # Step 5: Estimate duration
        estimated_duration = self._estimate_duration(steps)

        plan = ExecutionPlan(
            plan_id=plan_id,
            original_query=action_intent.original_query,
            action_intent=action_intent,
            steps=steps,
            resolved_references=resolved_refs,
            safety_level=safety_level,
            requires_confirmation=requires_confirmation,
            estimated_duration=estimated_duration,
            metadata={
                "context": context,
                "resolution_confidence": self._calculate_resolution_confidence(resolved_refs)
            }
        )

        logger.info(f"[ACTION-PLANNER] Plan created: {len(steps)} steps, safety={safety_level.value}")

        return plan

    async def _resolve_references(
        self,
        action_intent: ActionIntent,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Resolve implicit references using the resolver"""
        resolved = {}

        # If we have an implicit resolver, use it
        if self.implicit_resolver:
            try:
                logger.info("[ACTION-PLANNER] Using implicit reference resolver")
                result = await self.implicit_resolver.resolve_query(action_intent.original_query)

                if result.get("referent"):
                    referent = result["referent"]
                    resolved["referent"] = referent
                    resolved["referent_type"] = referent.get("type")
                    resolved["referent_entity"] = referent.get("entity")
                    resolved["space_id"] = referent.get("space_id")
                    resolved["app_name"] = referent.get("app_name")

                logger.info(f"[ACTION-PLANNER] Resolver found: {referent.get('type')} - {referent.get('entity', '')[:50]}")

            except Exception as e:
                logger.error(f"[ACTION-PLANNER] Error resolving references: {e}")

        # Fallback: Use context graph to find recent entities
        if not resolved and self.context_graph:
            # Try to find recent error
            if action_intent.action_type == ActionType.FIX_ERROR:
                error = self._find_recent_error(context)
                if error:
                    resolved["referent_type"] = "error"
                    resolved["referent_entity"] = error
                    logger.info(f"[ACTION-PLANNER] Found recent error via context graph")

            # Try to find active space
            if not action_intent.has_param("space_id"):
                active_space = await self._get_active_space()
                if active_space:
                    resolved["space_id"] = active_space
                    logger.info(f"[ACTION-PLANNER] Using active space: {active_space}")

        # Use explicit parameters from intent
        for param_name, param in action_intent.parameters.items():
            if param.explicit and param_name not in resolved:
                resolved[param_name] = param.value

        return resolved

    async def _generate_steps(
        self,
        action_intent: ActionIntent,
        resolved_refs: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> List[ExecutionStep]:
        """Generate execution steps based on action type"""
        steps = []

        action_type = action_intent.action_type

        if action_type == ActionType.SWITCH_SPACE:
            steps.extend(await self._plan_switch_space(action_intent, resolved_refs))
        elif action_type == ActionType.CLOSE_WINDOW:
            steps.extend(await self._plan_close_window(action_intent, resolved_refs))
        elif action_type == ActionType.MOVE_WINDOW:
            steps.extend(await self._plan_move_window(action_intent, resolved_refs))
        elif action_type == ActionType.LAUNCH_APP:
            steps.extend(await self._plan_launch_app(action_intent, resolved_refs))
        elif action_type == ActionType.QUIT_APP:
            steps.extend(await self._plan_quit_app(action_intent, resolved_refs))
        elif action_type == ActionType.RUN_TESTS:
            steps.extend(await self._plan_run_tests(action_intent, resolved_refs, context))
        elif action_type == ActionType.RUN_BUILD:
            steps.extend(await self._plan_run_build(action_intent, resolved_refs, context))
        elif action_type == ActionType.FIX_ERROR:
            steps.extend(await self._plan_fix_error(action_intent, resolved_refs, context))
        elif action_type == ActionType.OPEN_URL:
            steps.extend(await self._plan_open_url(action_intent, resolved_refs))
        elif action_type == ActionType.FOCUS_WINDOW:
            steps.extend(await self._plan_focus_window(action_intent, resolved_refs))
        else:
            # Generic action
            steps.append(ExecutionStep(
                step_id="step_1",
                description=f"Execute {action_type.value}",
                action_type="generic",
                command=action_intent.original_query,
                metadata={"action_intent": action_intent}
            ))

        return steps

    # ========================================================================
    # STEP PLANNING FOR EACH ACTION TYPE
    # ========================================================================

    async def _plan_switch_space(self, intent: ActionIntent, refs: Dict[str, Any]) -> List[ExecutionStep]:
        """Plan switching to a space"""
        space_id = refs.get("space_id") or intent.get_param("space_id")

        if not space_id:
            return [ExecutionStep(
                step_id="step_1_error",
                description="Cannot switch space: no space ID specified",
                action_type="error",
                command="",
                status=StepStatus.FAILED,
                error="No space ID specified"
            )]

        return [ExecutionStep(
            step_id="step_1",
            description=f"Switch to space {space_id}",
            action_type="yabai",
            command=f"yabai -m space --focus {space_id}",
            parameters={"space_id": space_id}
        )]

    async def _plan_close_window(self, intent: ActionIntent, refs: Dict[str, Any]) -> List[ExecutionStep]:
        """Plan closing a window"""
        app_name = refs.get("app_name") or intent.get_param("app_name")
        space_id = refs.get("context_space") or intent.get_param("context_space")

        steps = []

        if space_id:
            # First focus the space
            steps.append(ExecutionStep(
                step_id="step_1",
                description=f"Focus space {space_id}",
                action_type="yabai",
                command=f"yabai -m space --focus {space_id}",
                parameters={"space_id": space_id}
            ))

        if app_name:
            # Close the app using AppleScript
            steps.append(ExecutionStep(
                step_id="step_2" if space_id else "step_1",
                description=f"Close {app_name}",
                action_type="applescript",
                command=f'tell application "{app_name}" to quit',
                parameters={"app_name": app_name},
                depends_on=["step_1"] if space_id else []
            ))
        else:
            # Close focused window
            steps.append(ExecutionStep(
                step_id="step_2" if space_id else "step_1",
                description="Close focused window",
                action_type="yabai",
                command="yabai -m window --close",
                depends_on=["step_1"] if space_id else []
            ))

        return steps

    async def _plan_move_window(self, intent: ActionIntent, refs: Dict[str, Any]) -> List[ExecutionStep]:
        """Plan moving a window to a different space"""
        target_space = refs.get("space_id") or intent.get_param("space_id")
        app_name = refs.get("app_name") or intent.get_param("app_name")

        if not target_space:
            return [ExecutionStep(
                step_id="step_1_error",
                description="Cannot move window: no target space specified",
                action_type="error",
                command="",
                status=StepStatus.FAILED,
                error="No target space specified"
            )]

        steps = []

        if app_name:
            # Focus the app first
            steps.append(ExecutionStep(
                step_id="step_1",
                description=f"Focus {app_name}",
                action_type="applescript",
                command=f'tell application "{app_name}" to activate',
                parameters={"app_name": app_name}
            ))

        # Move focused window to space
        steps.append(ExecutionStep(
            step_id="step_2" if app_name else "step_1",
            description=f"Move window to space {target_space}",
            action_type="yabai",
            command=f"yabai -m window --space {target_space}",
            parameters={"target_space": target_space},
            depends_on=["step_1"] if app_name else []
        ))

        return steps

    async def _plan_launch_app(self, intent: ActionIntent, refs: Dict[str, Any]) -> List[ExecutionStep]:
        """Plan launching an application"""
        app_name = refs.get("app_name") or intent.get_param("app_name")

        if not app_name:
            return [ExecutionStep(
                step_id="step_1_error",
                description="Cannot launch app: no app name specified",
                action_type="error",
                command="",
                status=StepStatus.FAILED,
                error="No app name specified"
            )]

        return [ExecutionStep(
            step_id="step_1",
            description=f"Launch {app_name}",
            action_type="applescript",
            command=f'tell application "{app_name}" to activate',
            parameters={"app_name": app_name}
        )]

    async def _plan_quit_app(self, intent: ActionIntent, refs: Dict[str, Any]) -> List[ExecutionStep]:
        """Plan quitting an application"""
        app_name = refs.get("app_name") or intent.get_param("app_name")

        if not app_name:
            return [ExecutionStep(
                step_id="step_1_error",
                description="Cannot quit app: no app name specified",
                action_type="error",
                command="",
                status=StepStatus.FAILED,
                error="No app name specified"
            )]

        return [ExecutionStep(
            step_id="step_1",
            description=f"Quit {app_name}",
            action_type="applescript",
            command=f'tell application "{app_name}" to quit',
            parameters={"app_name": app_name}
        )]

    async def _plan_run_tests(self, intent: ActionIntent, refs: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ExecutionStep]:
        """Plan running tests"""
        # Determine test command from context or use default
        test_command = "pytest"  # Default
        if context and "test_command" in context:
            test_command = context["test_command"]

        return [ExecutionStep(
            step_id="step_1",
            description="Run test suite",
            action_type="shell",
            command=test_command,
            parameters={"cwd": refs.get("repo_path", ".")}
        )]

    async def _plan_run_build(self, intent: ActionIntent, refs: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ExecutionStep]:
        """Plan running build"""
        # Determine build command from context or use default
        build_command = "npm run build"  # Default
        if context and "build_command" in context:
            build_command = context["build_command"]

        return [ExecutionStep(
            step_id="step_1",
            description="Run build process",
            action_type="shell",
            command=build_command,
            parameters={"cwd": refs.get("repo_path", ".")}
        )]

    async def _plan_fix_error(self, intent: ActionIntent, refs: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ExecutionStep]:
        """Plan fixing an error (v1.0: suggest steps only)"""
        error_entity = refs.get("referent_entity", "unknown error")
        space_id = refs.get("space_id")

        # v1.0: Provide suggestions only (read-only)
        suggestion = f"I can see the error: {error_entity[:100]}, but cannot automatically fix it yet. "
        suggestion += "Suggested manual steps:\n"
        suggestion += "1. Review the error message carefully\n"
        suggestion += "2. Check recent code changes\n"
        suggestion += "3. Verify dependencies are installed\n"
        suggestion += "4. Check logs for more details"

        return [ExecutionStep(
            step_id="step_1",
            description="Provide error fix suggestions (v1.0 - read-only)",
            action_type="suggestion",
            command="",
            parameters={
                "error": error_entity,
                "space_id": space_id,
                "suggestion": suggestion
            },
            metadata={"version": "1.0", "read_only": True}
        )]

    async def _plan_open_url(self, intent: ActionIntent, refs: Dict[str, Any]) -> List[ExecutionStep]:
        """Plan opening a URL"""
        url = refs.get("url") or intent.get_param("url")

        if not url:
            return [ExecutionStep(
                step_id="step_1_error",
                description="Cannot open URL: no URL specified",
                action_type="error",
                command="",
                status=StepStatus.FAILED,
                error="No URL specified"
            )]

        # Ensure URL has protocol
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        return [ExecutionStep(
            step_id="step_1",
            description=f"Open {url} in default browser",
            action_type="shell",
            command=f'open "{url}"',
            parameters={"url": url}
        )]

    async def _plan_focus_window(self, intent: ActionIntent, refs: Dict[str, Any]) -> List[ExecutionStep]:
        """Plan focusing a window"""
        app_name = refs.get("app_name") or intent.get_param("app_name")

        if not app_name:
            return [ExecutionStep(
                step_id="step_1",
                description="Focus current window",
                action_type="yabai",
                command="yabai -m window --focus",
            )]

        return [ExecutionStep(
            step_id="step_1",
            description=f"Focus {app_name}",
            action_type="applescript",
            command=f'tell application "{app_name}" to activate',
            parameters={"app_name": app_name}
        )]

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _find_recent_error(self, context: Optional[Dict[str, Any]]) -> Optional[str]:
        """Find the most recent error from context graph"""
        if not self.context_graph:
            return None

        try:
            # Try to find recent error from context graph
            error = self.context_graph.find_most_recent_error(within_seconds=300)
            if error:
                _, _, details = error
                return details.get("error", "Unknown error")
        except Exception as e:
            logger.warning(f"[ACTION-PLANNER] Could not find recent error: {e}")

        return None

    async def _get_active_space(self) -> Optional[int]:
        """Get the currently active space"""
        try:
            from vision.yabai_space_detector import get_yabai_detector
            yabai = get_yabai_detector()

            if not yabai.is_available():
                return None

            spaces = yabai.enumerate_all_spaces(include_display_info=True)
            active_space = next((s for s in spaces if s.get('has-focus', False)), None)

            if active_space:
                return active_space.get('index', active_space.get('id', 1))
        except Exception as e:
            logger.warning(f"[ACTION-PLANNER] Could not get active space: {e}")

        return None

    def _determine_safety_level(self, intent: ActionIntent, steps: List[ExecutionStep]) -> ActionSafety:
        """Determine overall safety level of the plan"""
        # Start with intent's safety level
        safety = intent.safety_level

        # Check if any steps are risky
        for step in steps:
            if step.action_type == "shell" and "rm" in step.command:
                safety = ActionSafety.RISKY
            elif step.status == StepStatus.FAILED:
                safety = ActionSafety.BLOCKED

        return safety

    def _estimate_duration(self, steps: List[ExecutionStep]) -> float:
        """Estimate execution duration in seconds"""
        # Simple estimation based on step types
        durations = {
            "yabai": 0.5,
            "applescript": 1.0,
            "shell": 2.0,
            "suggestion": 0.1,
            "error": 0.0,
        }

        total = sum(durations.get(step.action_type, 1.0) for step in steps)
        return total

    def _calculate_resolution_confidence(self, resolved_refs: Dict[str, Any]) -> float:
        """Calculate confidence in reference resolution"""
        if not resolved_refs:
            return 0.0

        # Simple heuristic based on what was resolved
        confidence = 0.5

        if "referent_type" in resolved_refs:
            confidence += 0.2
        if "space_id" in resolved_refs:
            confidence += 0.1
        if "app_name" in resolved_refs:
            confidence += 0.2

        return min(1.0, confidence)


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_global_planner: Optional[ActionPlanner] = None


def get_action_planner() -> Optional[ActionPlanner]:
    """Get the global action planner instance"""
    return _global_planner


def initialize_action_planner(context_graph=None, implicit_resolver=None) -> ActionPlanner:
    """Initialize the global action planner"""
    global _global_planner
    _global_planner = ActionPlanner(context_graph, implicit_resolver)
    logger.info("[ACTION-PLANNER] Global instance initialized")
    return _global_planner
