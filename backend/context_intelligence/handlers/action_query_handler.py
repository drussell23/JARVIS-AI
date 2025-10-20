"""
Action Query Handler for JARVIS
================================

Main coordinator for action-oriented queries

Integrates:
- Action Analyzer (intent parsing)
- Action Planner (with implicit_reference_resolver) ⭐
- Action Safety Manager (confirmations)
- Action Executor (yabai/AppleScript/shell)

Handles queries like:
- "Fix the error in space 3"
- "Switch to space 5"
- "Close the browser"
- "Run the tests"

Author: Derek Russell
Date: 2025-10-19
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from context_intelligence.analyzers.action_analyzer import (
    get_action_analyzer,
    initialize_action_analyzer,
    ActionIntent
)
from context_intelligence.planners.action_planner import (
    get_action_planner,
    initialize_action_planner,
    ExecutionPlan
)
from context_intelligence.executors.action_executor import (
    get_action_executor,
    initialize_action_executor,
    ExecutionResult
)
from context_intelligence.safety.action_safety_manager import (
    get_action_safety_manager,
    initialize_action_safety_manager,
    ConfirmationResult
)

logger = logging.getLogger(__name__)


# ============================================================================
# RESPONSE MODELS
# ============================================================================

@dataclass
class ActionQueryResponse:
    """Response from action query handling"""
    success: bool
    message: str
    query: str
    action_type: str
    execution_result: Optional[ExecutionResult] = None
    plan: Optional[ExecutionPlan] = None
    confirmation: Optional[ConfirmationResult] = None
    requires_confirmation: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# ACTION QUERY HANDLER
# ============================================================================

class ActionQueryHandler:
    """
    Main handler for action-oriented queries

    Coordinates the full action execution pipeline:
    1. Analyze query intent
    2. Create execution plan (with implicit reference resolution)
    3. Request safety confirmation
    4. Execute plan
    5. Return results
    """

    def __init__(
        self,
        context_graph=None,
        implicit_resolver=None,
        dry_run: bool = False
    ):
        """
        Initialize the action query handler

        Args:
            context_graph: Context graph for workspace awareness
            implicit_resolver: Implicit reference resolver (CRITICAL!)
            dry_run: If True, don't actually execute actions
        """
        self.context_graph = context_graph
        self.implicit_resolver = implicit_resolver

        # Initialize components
        self.analyzer = get_action_analyzer()
        if not self.analyzer:
            self.analyzer = initialize_action_analyzer()

        self.planner = get_action_planner()
        if not self.planner:
            self.planner = initialize_action_planner(
                context_graph=context_graph,
                implicit_resolver=implicit_resolver  # ⭐ KEY INTEGRATION!
            )

        self.executor = get_action_executor()
        if not self.executor:
            self.executor = initialize_action_executor(dry_run=dry_run)

        self.safety_manager = get_action_safety_manager()
        if not self.safety_manager:
            self.safety_manager = initialize_action_safety_manager()

        logger.info("[ACTION-HANDLER] Initialized")

    async def handle_action_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ActionQueryResponse:
        """
        Handle an action query end-to-end

        Args:
            query: The action query
            context: Additional context

        Returns:
            ActionQueryResponse with results
        """
        logger.info(f"[ACTION-HANDLER] Handling query: '{query}'")

        try:
            # Step 1: Analyze intent
            intent = await self.analyzer.analyze(query, context)
            logger.info(f"[ACTION-HANDLER] Intent: {intent.action_type.value}, confidence={intent.confidence:.2f}")

            # Step 2: Create execution plan
            # This is where implicit_reference_resolver is used!
            plan = await self.planner.create_plan(intent, context)
            logger.info(f"[ACTION-HANDLER] Plan created: {len(plan.steps)} steps, safety={plan.safety_level.value}")

            # Step 3: Safety check & confirmation
            if plan.requires_confirmation:
                confirmation = await self.safety_manager.request_confirmation(plan, context)

                if not confirmation.approved:
                    logger.info("[ACTION-HANDLER] Action cancelled by user")
                    return ActionQueryResponse(
                        success=False,
                        message="Action cancelled",
                        query=query,
                        action_type=intent.action_type.value,
                        plan=plan,
                        confirmation=confirmation,
                        requires_confirmation=True,
                        metadata={"cancelled": True}
                    )
            else:
                logger.info("[ACTION-HANDLER] No confirmation needed - proceeding")
                confirmation = None

            # Step 4: Execute plan
            execution_result = await self.executor.execute_plan(plan)
            logger.info(f"[ACTION-HANDLER] Execution complete: {execution_result.status.value}")

            # Step 5: Generate response
            success = execution_result.status.value in ["success", "partial_success"]
            message = self._generate_response_message(plan, execution_result, intent)

            return ActionQueryResponse(
                success=success,
                message=message,
                query=query,
                action_type=intent.action_type.value,
                execution_result=execution_result,
                plan=plan,
                confirmation=confirmation,
                requires_confirmation=plan.requires_confirmation,
                metadata={
                    "resolved_references": plan.resolved_references,
                    "safety_level": plan.safety_level.value,
                    "step_count": len(plan.steps)
                }
            )

        except Exception as e:
            logger.error(f"[ACTION-HANDLER] Error handling query: {e}", exc_info=True)
            return ActionQueryResponse(
                success=False,
                message=f"Error processing action: {str(e)}",
                query=query,
                action_type="unknown",
                metadata={"error": str(e)}
            )

    def _generate_response_message(
        self,
        plan: ExecutionPlan,
        execution_result: ExecutionResult,
        intent: ActionIntent
    ) -> str:
        """Generate human-readable response message"""
        # Start with execution message
        message = execution_result.message

        # Add context about what was resolved
        if plan.resolved_references:
            refs = plan.resolved_references

            if "referent_entity" in refs:
                entity = refs["referent_entity"]
                if len(str(entity)) > 100:
                    entity = str(entity)[:100] + "..."
                message += f"\n\n📍 Resolved reference: {entity}"

            if "space_id" in refs and intent.action_type.value != "switch_space":
                message += f"\n📍 Target space: {refs['space_id']}"

        # Add step details if partially successful
        if execution_result.status.value == "partial_success":
            succeeded = [r for r in execution_result.steps_executed if r.success]
            failed = [r for r in execution_result.steps_executed if not r.success]

            message += f"\n\n✅ Completed: {len(succeeded)} steps"
            message += f"\n❌ Failed: {len(failed)} steps"

            if failed:
                first_failure = failed[0]
                if first_failure.error:
                    message += f"\n\nFirst error: {first_failure.error}"

        return message

    # ========================================================================
    # CONVENIENCE METHODS
    # ========================================================================

    async def switch_space(self, space_id: int) -> ActionQueryResponse:
        """Convenience: Switch to a space"""
        return await self.handle_action_query(f"Switch to space {space_id}")

    async def close_window(self, app_name: Optional[str] = None, space_id: Optional[int] = None) -> ActionQueryResponse:
        """Convenience: Close a window"""
        query = "Close"
        if app_name:
            query += f" {app_name}"
        if space_id:
            query += f" in space {space_id}"
        return await self.handle_action_query(query)

    async def launch_app(self, app_name: str) -> ActionQueryResponse:
        """Convenience: Launch an app"""
        return await self.handle_action_query(f"Launch {app_name}")

    async def run_tests(self) -> ActionQueryResponse:
        """Convenience: Run tests"""
        return await self.handle_action_query("Run the tests")


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_global_handler: Optional[ActionQueryHandler] = None


def get_action_query_handler() -> Optional[ActionQueryHandler]:
    """Get the global action query handler instance"""
    return _global_handler


def initialize_action_query_handler(
    context_graph=None,
    implicit_resolver=None,
    **kwargs
) -> ActionQueryHandler:
    """
    Initialize the global action query handler

    Args:
        context_graph: Context graph for workspace awareness
        implicit_resolver: Implicit reference resolver ⭐ IMPORTANT!
        **kwargs: Additional arguments

    Returns:
        ActionQueryHandler instance
    """
    global _global_handler
    _global_handler = ActionQueryHandler(
        context_graph=context_graph,
        implicit_resolver=implicit_resolver,
        **kwargs
    )
    logger.info("[ACTION-HANDLER] Global instance initialized")
    return _global_handler


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

async def handle_action_query(query: str, **kwargs) -> ActionQueryResponse:
    """Convenience function to handle an action query"""
    handler = get_action_query_handler()
    if not handler:
        # Need to initialize with implicit resolver
        raise RuntimeError(
            "ActionQueryHandler not initialized. "
            "Call initialize_action_query_handler() first with implicit_resolver"
        )
    return await handler.handle_action_query(query, **kwargs)
