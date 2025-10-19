"""
Action Safety Manager for JARVIS
=================================

Manages safety confirmations and risk assessment for actions

Features:
- Safety level evaluation
- User confirmation requests
- Automatic approval for safe actions
- Extra warnings for risky actions
- Trusted action allowlist

Author: Derek Russell
Date: 2025-10-19
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from backend.context_intelligence.analyzers.action_analyzer import ActionSafety
from backend.context_intelligence.planners.action_planner import ExecutionPlan, ExecutionStep

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIRMATION RESULTS
# ============================================================================

@dataclass
class ConfirmationResult:
    """Result of a confirmation request"""
    approved: bool
    confirmation_message: str
    user_response: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# ACTION SAFETY MANAGER
# ============================================================================

class ActionSafetyManager:
    """
    Manages safety confirmations for actions

    Handles:
    - Automatic approval for safe actions
    - User confirmation for risky actions
    - Trusted action patterns
    - Safety level evaluation
    """

    def __init__(
        self,
        auto_approve_safe: bool = True,
        enable_confirmations: bool = True
    ):
        """
        Initialize the safety manager

        Args:
            auto_approve_safe: Automatically approve SAFE actions
            enable_confirmations: Enable user confirmations (False = auto-approve all)
        """
        self.auto_approve_safe = auto_approve_safe
        self.enable_confirmations = enable_confirmations

        # Trusted actions that can be auto-approved
        self.trusted_actions = self._initialize_trusted_actions()

        # Confirmation callback (set by integration layer)
        self.confirmation_callback = None

        logger.info(f"[SAFETY-MANAGER] Initialized (auto_approve_safe={auto_approve_safe}, confirmations={enable_confirmations})")

    def _initialize_trusted_actions(self) -> Set[str]:
        """Initialize set of trusted action patterns"""
        return {
            "yabai -m space --focus",  # Safe space switching
            "yabai -m window --focus",  # Safe window focusing
            "open http",  # Safe URL opening
        }

    async def request_confirmation(
        self,
        plan: ExecutionPlan,
        context: Optional[Dict[str, Any]] = None
    ) -> ConfirmationResult:
        """
        Request confirmation for an action plan

        Args:
            plan: The execution plan
            context: Additional context

        Returns:
            ConfirmationResult with approval decision
        """
        logger.info(f"[SAFETY-MANAGER] Requesting confirmation for plan: {plan.plan_id}")

        # Check if confirmations are disabled
        if not self.enable_confirmations:
            logger.info("[SAFETY-MANAGER] Confirmations disabled - auto-approving")
            return ConfirmationResult(
                approved=True,
                confirmation_message="Auto-approved (confirmations disabled)",
                metadata={"auto_approved": True}
            )

        # Auto-approve SAFE actions
        if plan.safety_level == ActionSafety.SAFE and self.auto_approve_safe:
            logger.info("[SAFETY-MANAGER] Auto-approving SAFE action")
            return ConfirmationResult(
                approved=True,
                confirmation_message="Auto-approved (safe action)",
                metadata={"auto_approved": True, "safety_level": "SAFE"}
            )

        # Check if all steps are trusted
        if self._all_steps_trusted(plan.steps):
            logger.info("[SAFETY-MANAGER] Auto-approving trusted actions")
            return ConfirmationResult(
                approved=True,
                confirmation_message="Auto-approved (trusted actions)",
                metadata={"auto_approved": True, "trusted": True}
            )

        # Generate confirmation message
        message = self._generate_confirmation_message(plan)

        # Request confirmation from user
        if self.confirmation_callback:
            approved = await self.confirmation_callback(message, plan)
        else:
            # Default: approve for now (in real system, would wait for user input)
            logger.warning("[SAFETY-MANAGER] No confirmation callback set - auto-approving")
            approved = True

        logger.info(f"[SAFETY-MANAGER] Confirmation result: approved={approved}")

        return ConfirmationResult(
            approved=approved,
            confirmation_message=message,
            metadata={
                "safety_level": plan.safety_level.value,
                "step_count": len(plan.steps)
            }
        )

    def _generate_confirmation_message(self, plan: ExecutionPlan) -> str:
        """Generate human-readable confirmation message"""
        action_name = plan.action_intent.action_type.value.replace("_", " ").title()

        msg = f"I'm about to {action_name.lower()}:\n\n"

        # List steps
        for i, step in enumerate(plan.steps, 1):
            msg += f"  {i}. {step.description}\n"

        # Add safety warning
        if plan.safety_level == ActionSafety.RISKY:
            msg += "\n⚠️  WARNING: This action may be irreversible!\n"
        elif plan.safety_level == ActionSafety.NEEDS_CONFIRMATION:
            msg += "\n⚠️  This action requires confirmation.\n"

        # Add resolution info if references were resolved
        if plan.resolved_references:
            msg += "\n📍 Resolved references:\n"
            for key, value in plan.resolved_references.items():
                if key in ["referent_entity", "app_name", "space_id"]:
                    msg += f"  - {key}: {value}\n"

        msg += "\nProceed? (yes/no)"

        return msg

    def _all_steps_trusted(self, steps: List[ExecutionStep]) -> bool:
        """Check if all steps are trusted"""
        for step in steps:
            # Check if command matches trusted patterns
            if not any(
                step.command.startswith(trusted)
                for trusted in self.trusted_actions
            ):
                return False

        return True

    def set_confirmation_callback(self, callback):
        """
        Set the confirmation callback function

        Args:
            callback: Async function(message, plan) -> bool
        """
        self.confirmation_callback = callback
        logger.info("[SAFETY-MANAGER] Confirmation callback set")

    def add_trusted_action(self, action_pattern: str):
        """Add a trusted action pattern"""
        self.trusted_actions.add(action_pattern)
        logger.info(f"[SAFETY-MANAGER] Added trusted action: {action_pattern}")

    def is_action_safe(self, plan: ExecutionPlan) -> bool:
        """Check if an action is safe to execute without confirmation"""
        return (
            plan.safety_level == ActionSafety.SAFE
            or self._all_steps_trusted(plan.steps)
        )


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_global_safety_manager: Optional[ActionSafetyManager] = None


def get_action_safety_manager() -> Optional[ActionSafetyManager]:
    """Get the global action safety manager instance"""
    return _global_safety_manager


def initialize_action_safety_manager(**kwargs) -> ActionSafetyManager:
    """Initialize the global action safety manager"""
    global _global_safety_manager
    _global_safety_manager = ActionSafetyManager(**kwargs)
    logger.info("[SAFETY-MANAGER] Global instance initialized")
    return _global_safety_manager
