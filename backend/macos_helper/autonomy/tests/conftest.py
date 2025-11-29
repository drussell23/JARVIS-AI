"""
Pytest configuration for autonomy module tests.

Uses direct module imports to avoid macos_helper __init__ side effects.
"""

import pytest
import sys
import os
import importlib.util

# Ensure we're importing from the right place
backend_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)


def load_module_directly(module_name: str, file_path: str):
    """Load a module directly without triggering parent __init__."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Pre-load autonomy modules directly
autonomy_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load each module directly
_action_registry = load_module_directly(
    "macos_helper.autonomy.action_registry",
    os.path.join(autonomy_path, "action_registry.py")
)

_permission_system = load_module_directly(
    "macos_helper.autonomy.permission_system",
    os.path.join(autonomy_path, "permission_system.py")
)

_safety_validator = load_module_directly(
    "macos_helper.autonomy.safety_validator",
    os.path.join(autonomy_path, "safety_validator.py")
)

_action_learning = load_module_directly(
    "macos_helper.autonomy.action_learning",
    os.path.join(autonomy_path, "action_learning.py")
)


@pytest.fixture
def action_registry_module():
    """Provide action registry module."""
    return _action_registry


@pytest.fixture
def permission_system_module():
    """Provide permission system module."""
    return _permission_system


@pytest.fixture
def safety_validator_module():
    """Provide safety validator module."""
    return _safety_validator


@pytest.fixture
def action_learning_module():
    """Provide action learning module."""
    return _action_learning


@pytest.fixture
def registry_config(action_registry_module):
    """Create a test registry configuration."""
    return action_registry_module.ActionRegistryConfig(
        auto_discover=False,
        cache_ttl_seconds=5.0,
        validation_mode="strict"
    )


@pytest.fixture
def permission_config(permission_system_module):
    """Create a test permission configuration."""
    return permission_system_module.PermissionSystemConfig(
        mode="standard",
        auto_approve_minimal=True,
        rate_limit_enabled=False,  # Disable for tests
        quiet_hours_enabled=False,
        cache_enabled=False,  # Disable caching for tests
        audit_enabled=False,
    )


@pytest.fixture
def safety_config(safety_validator_module):
    """Create a test safety configuration."""
    return safety_validator_module.SafetyValidatorConfig(
        level=safety_validator_module.SafetyLevel.STANDARD,
        max_risk_score=0.8,
        audit_enabled=False,
    )


@pytest.fixture
def learning_config(action_learning_module):
    """Create a test learning configuration."""
    return action_learning_module.ActionLearningConfig(
        enabled=True,
        min_samples_for_prediction=3,  # Lower for tests
        decay_factor=0.95,
        max_history_size=100,
    )
