"""
Pytest configuration for intelligence module tests.

Uses direct module imports to avoid macos_helper __init__ side effects.
"""

import pytest
import sys
import os

# Ensure we're importing from the right place
backend_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# Block macos_helper from importing problematic modules during tests
# by pre-loading just the intelligence submodule
import importlib.util

def load_module_directly(module_name, file_path):
    """Load a module directly without triggering parent __init__."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Pre-load intelligence modules directly
intelligence_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load each module directly
_screen_context = load_module_directly(
    "macos_helper.intelligence.screen_context_analyzer",
    os.path.join(intelligence_path, "screen_context_analyzer.py")
)

_suggestion_engine = load_module_directly(
    "macos_helper.intelligence.proactive_suggestion_engine",
    os.path.join(intelligence_path, "proactive_suggestion_engine.py")
)

_notification_triage = load_module_directly(
    "macos_helper.intelligence.notification_triage",
    os.path.join(intelligence_path, "notification_triage.py")
)

_focus_tracker = load_module_directly(
    "macos_helper.intelligence.focus_tracker",
    os.path.join(intelligence_path, "focus_tracker.py")
)

_coordinator = load_module_directly(
    "macos_helper.intelligence.intelligence_coordinator",
    os.path.join(intelligence_path, "intelligence_coordinator.py")
)


@pytest.fixture
def screen_context_module():
    """Provide screen context analyzer module."""
    return _screen_context


@pytest.fixture
def suggestion_engine_module():
    """Provide suggestion engine module."""
    return _suggestion_engine


@pytest.fixture
def notification_triage_module():
    """Provide notification triage module."""
    return _notification_triage


@pytest.fixture
def focus_tracker_module():
    """Provide focus tracker module."""
    return _focus_tracker


@pytest.fixture
def coordinator_module():
    """Provide intelligence coordinator module."""
    return _coordinator
