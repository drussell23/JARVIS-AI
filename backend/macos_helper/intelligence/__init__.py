"""
JARVIS macOS Helper - Phase 2 Real-Time Intelligence

Advanced intelligence layer that transforms raw macOS events into
actionable insights and proactive suggestions.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    Phase 2: Real-Time Intelligence                       │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  ┌──────────────────────────────────────────────────────────────────┐   │
    │  │                    Screen Context Analyzer                         │   │
    │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────────┐   │   │
    │  │  │ Capture │ │   OCR   │ │Semantic │ │ Element │ │  Context  │   │   │
    │  │  │ Manager │ │ Engine  │ │ Parser  │ │ Tracker │ │ Enricher  │   │   │
    │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └───────────┘   │   │
    │  └──────────────────────────────────────────────────────────────────┘   │
    │                               │                                          │
    │  ┌──────────────────────────────────────────────────────────────────┐   │
    │  │                    Proactive Suggestion Engine                     │   │
    │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────────┐   │   │
    │  │  │ Pattern │ │ Intent  │ │Predict- │ │Suggest- │ │  Timing   │   │   │
    │  │  │ Learner │ │ Inferer │ │  tion   │ │  ion    │ │ Optimizer │   │   │
    │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └───────────┘   │   │
    │  └──────────────────────────────────────────────────────────────────┘   │
    │                               │                                          │
    │  ┌──────────────────────────────────────────────────────────────────┐   │
    │  │                    Notification Triage System                      │   │
    │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────────┐   │   │
    │  │  │Category │ │Priority │ │ Smart   │ │ Focus   │ │ Delivery  │   │   │
    │  │  │ Engine  │ │ Scorer  │ │Batching │ │ Guard   │ │ Scheduler │   │   │
    │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └───────────┘   │   │
    │  └──────────────────────────────────────────────────────────────────┘   │
    │                               │                                          │
    │  ┌──────────────────────────────────────────────────────────────────┐   │
    │  │                    Focus & Productivity Tracker                    │   │
    │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────────┐   │   │
    │  │  │Activity │ │  Focus  │ │ Session │ │ Insight │ │  Report   │   │   │
    │  │  │ Monitor │ │ Scorer  │ │ Tracker │ │ Engine  │ │ Generator │   │   │
    │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └───────────┘   │   │
    │  └──────────────────────────────────────────────────────────────────┘   │
    │                               │                                          │
    │  ┌──────────────────────────────────────────────────────────────────┐   │
    │  │                       UAE Integration                              │   │
    │  │  ┌─────────────────────────────────────────────────────────────┐  │   │
    │  │  │    Context × Situation × Learning = True Intelligence       │  │   │
    │  │  └─────────────────────────────────────────────────────────────┘  │   │
    │  └──────────────────────────────────────────────────────────────────┘   │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

Features:
- Real-time screen context understanding with Claude Vision
- Proactive suggestions based on learned patterns
- Intelligent notification batching and prioritization
- Focus tracking and productivity insights
- Deep integration with UAE for context fusion
- Calendar-aware and context-aware timing
- Zero hardcoding - all behavior learned dynamically

Usage:
    from macos_helper.intelligence import start_intelligence_layer

    # Start the intelligence layer
    intel = await start_intelligence_layer()

    # Subscribe to suggestions
    intel.on_suggestion(handle_suggestion)

    # Get current context
    context = await intel.get_current_context()

    # Stop
    await stop_intelligence_layer()

Version: 1.0.0
"""

from __future__ import annotations

__version__ = "1.0.0"

from .screen_context_analyzer import (
    ScreenContextAnalyzer,
    ScreenContext,
    VisualElement,
    ActivityType,
    get_screen_context_analyzer,
    start_screen_context_analyzer,
    stop_screen_context_analyzer,
)

from .proactive_suggestion_engine import (
    ProactiveSuggestionEngine,
    Suggestion,
    SuggestionType,
    SuggestionPriority,
    get_suggestion_engine,
    start_suggestion_engine,
    stop_suggestion_engine,
)

from .notification_triage import (
    NotificationTriageSystem,
    TriagedNotification,
    NotificationBatch,
    UrgencyLevel,
    get_notification_triage,
    start_notification_triage,
    stop_notification_triage,
)

from .focus_tracker import (
    FocusTracker,
    FocusSession,
    ProductivityInsight,
    FocusState,
    get_focus_tracker,
    start_focus_tracker,
    stop_focus_tracker,
)

from .intelligence_coordinator import (
    IntelligenceCoordinator,
    IntelligenceConfig,
    get_intelligence_coordinator,
    start_intelligence_layer,
    stop_intelligence_layer,
)

__all__ = [
    # Version
    "__version__",
    # Screen Context
    "ScreenContextAnalyzer",
    "ScreenContext",
    "VisualElement",
    "ActivityType",
    "get_screen_context_analyzer",
    "start_screen_context_analyzer",
    "stop_screen_context_analyzer",
    # Proactive Suggestions
    "ProactiveSuggestionEngine",
    "Suggestion",
    "SuggestionType",
    "SuggestionPriority",
    "get_suggestion_engine",
    "start_suggestion_engine",
    "stop_suggestion_engine",
    # Notification Triage
    "NotificationTriageSystem",
    "TriagedNotification",
    "NotificationBatch",
    "UrgencyLevel",
    "get_notification_triage",
    "start_notification_triage",
    "stop_notification_triage",
    # Focus Tracker
    "FocusTracker",
    "FocusSession",
    "ProductivityInsight",
    "FocusState",
    "get_focus_tracker",
    "start_focus_tracker",
    "stop_focus_tracker",
    # Main Coordinator
    "IntelligenceCoordinator",
    "IntelligenceConfig",
    "get_intelligence_coordinator",
    "start_intelligence_layer",
    "stop_intelligence_layer",
]
