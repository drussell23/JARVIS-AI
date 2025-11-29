"""
Phase 3: Autonomous Action Execution Module

This module provides the autonomous action execution layer for JARVIS,
enabling intelligent, safe, and adaptive execution of system actions.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     PHASE 3: AUTONOMOUS ACTIONS                      │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                       │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
    │  │   Action    │───▶│  Permission │───▶│   Safety    │              │
    │  │  Registry   │    │   System    │    │  Validator  │              │
    │  └─────────────┘    └─────────────┘    └─────────────┘              │
    │         │                  │                  │                      │
    │         ▼                  ▼                  ▼                      │
    │  ┌─────────────────────────────────────────────────────────────┐   │
    │  │                  ADVANCED ACTION EXECUTOR                     │   │
    │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │   │
    │  │  │ Multi-Retry  │  │   Circuit    │  │   Rollback   │       │   │
    │  │  │  Strategy    │  │   Breaker    │  │   Manager    │       │   │
    │  │  └──────────────┘  └──────────────┘  └──────────────┘       │   │
    │  └─────────────────────────────────────────────────────────────┘   │
    │         │                                                           │
    │         ▼                                                           │
    │  ┌─────────────────────────────────────────────────────────────┐   │
    │  │                   ACTION LEARNING SYSTEM                      │   │
    │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │   │
    │  │  │   Success    │  │   Pattern    │  │   Adaptive   │       │   │
    │  │  │  Predictor   │  │   Learning   │  │  Optimization│       │   │
    │  │  └──────────────┘  └──────────────┘  └──────────────┘       │   │
    │  └─────────────────────────────────────────────────────────────┘   │
    │                                                                       │
    └─────────────────────────────────────────────────────────────────────┘

Components:
    - ActionRegistry: Centralized registry of all action types with metadata
    - PermissionSystem: Fine-grained permission management
    - SafetyValidator: Constraint checking and risk assessment
    - AdvancedActionExecutor: Multi-strategy execution with error recovery
    - ActionLearningSystem: Pattern learning and success prediction

Key Features:
    - Async-first architecture with background task management
    - Dynamic configuration via environment variables
    - Multi-strategy retry with exponential backoff
    - Circuit breaker pattern for failure isolation
    - Rollback capabilities for reversible actions
    - Machine learning-based success prediction
    - Real-time learning from execution outcomes

Environment Variables:
    JARVIS_ACTION_EXECUTOR_MAX_RETRIES: Maximum retry attempts (default: 3)
    JARVIS_ACTION_EXECUTOR_TIMEOUT: Default action timeout in seconds (default: 30)
    JARVIS_ACTION_CIRCUIT_BREAKER_THRESHOLD: Failure threshold for circuit breaker (default: 5)
    JARVIS_ACTION_CIRCUIT_BREAKER_RESET_SECONDS: Circuit breaker reset time (default: 60)
    JARVIS_ACTION_LEARNING_ENABLED: Enable learning system (default: true)
    JARVIS_ACTION_SAFETY_LEVEL: Default safety level (default: standard)
    JARVIS_ACTION_PERMISSION_MODE: Permission mode (default: paranoid)
    JARVIS_ACTION_DRY_RUN_MODE: Enable dry-run mode (default: false)
"""

from .action_registry import (
    ActionRegistry,
    ActionType,
    ActionMetadata,
    ActionCategory,
    ActionRiskLevel,
    ActionHandler,
    get_action_registry,
    start_action_registry,
    stop_action_registry,
)

from .permission_system import (
    PermissionSystem,
    Permission,
    PermissionLevel,
    PermissionScope,
    PermissionContext,
    PermissionDecision,
    get_permission_system,
    start_permission_system,
    stop_permission_system,
)

from .safety_validator import (
    SafetyValidator,
    SafetyConstraint,
    SafetyCheck,
    SafetyCheckResult,
    SafetyLevel,
    RiskAssessment,
    get_safety_validator,
    start_safety_validator,
    stop_safety_validator,
)

from .advanced_executor import (
    AdvancedActionExecutor,
    ExecutionContext,
    ExecutionResult,
    ExecutionStatus,
    RetryStrategy,
    CircuitBreakerState,
    RollbackManager,
    get_advanced_executor,
    start_advanced_executor,
    stop_advanced_executor,
)

from .action_learning import (
    ActionLearningSystem,
    ActionOutcome,
    PredictionResult,
    LearningFeatures,
    PatternType,
    get_action_learning_system,
    start_action_learning_system,
    stop_action_learning_system,
)

from .autonomy_coordinator import (
    AutonomyCoordinator,
    get_autonomy_coordinator,
    start_autonomy_coordinator,
    stop_autonomy_coordinator,
)

__all__ = [
    # Action Registry
    "ActionRegistry",
    "ActionType",
    "ActionMetadata",
    "ActionCategory",
    "ActionRiskLevel",
    "ActionHandler",
    "get_action_registry",
    "start_action_registry",
    "stop_action_registry",

    # Permission System
    "PermissionSystem",
    "Permission",
    "PermissionLevel",
    "PermissionScope",
    "PermissionContext",
    "PermissionDecision",
    "get_permission_system",
    "start_permission_system",
    "stop_permission_system",

    # Safety Validator
    "SafetyValidator",
    "SafetyConstraint",
    "SafetyCheck",
    "SafetyCheckResult",
    "SafetyLevel",
    "RiskAssessment",
    "get_safety_validator",
    "start_safety_validator",
    "stop_safety_validator",

    # Advanced Executor
    "AdvancedActionExecutor",
    "ExecutionContext",
    "ExecutionResult",
    "ExecutionStatus",
    "RetryStrategy",
    "CircuitBreakerState",
    "RollbackManager",
    "get_advanced_executor",
    "start_advanced_executor",
    "stop_advanced_executor",

    # Action Learning
    "ActionLearningSystem",
    "ActionOutcome",
    "PredictionResult",
    "LearningFeatures",
    "PatternType",
    "get_action_learning_system",
    "start_action_learning_system",
    "stop_action_learning_system",

    # Autonomy Coordinator
    "AutonomyCoordinator",
    "get_autonomy_coordinator",
    "start_autonomy_coordinator",
    "stop_autonomy_coordinator",
]
