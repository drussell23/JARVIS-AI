"""
Context Intelligence Handlers
==============================

Handlers for complex context-aware queries
"""

from .multi_space_query_handler import (
    MultiSpaceQueryHandler,
    get_multi_space_handler,
    initialize_multi_space_handler,
    MultiSpaceQueryType,
    SpaceAnalysisResult,
    MultiSpaceQueryResult
)

from .temporal_query_handler import (
    TemporalQueryHandler,
    get_temporal_query_handler,
    initialize_temporal_handler,
    TemporalQueryType,
    TemporalQueryResult,
    ScreenshotManager,
    ImageDiffer
)

from .predictive_query_handler import (
    PredictiveQueryHandler,
    PredictiveQueryRequest,
    PredictiveQueryResponse,
    ClaudeVisionAnalyzer,
    get_predictive_handler,
    initialize_predictive_handler,
    handle_predictive_query
)

from .action_query_handler import (
    ActionQueryHandler,
    ActionQueryResponse,
    get_action_query_handler,
    initialize_action_query_handler,
    handle_action_query
)

from .query_complexity_manager import (
    QueryComplexityManager,
    QueryComplexityClassifier,
    QueryRouter,
    ClassifiedQuery,
    ComplexityMetrics,
    QueryComplexity,
    get_query_complexity_manager,
    initialize_query_complexity_manager
)

from .medium_complexity_handler import (
    MediumComplexityHandler,
    MediumQueryType,
    SpaceCapture,
    MediumQueryResult,
    get_medium_complexity_handler,
    initialize_medium_complexity_handler
)

from .complex_complexity_handler import (
    ComplexComplexityHandler,
    ComplexQueryType,
    SpaceSnapshot,
    TemporalAnalysis,
    CrossSpaceAnalysis,
    PredictiveAnalysis,
    ComplexQueryResult,
    get_complex_complexity_handler,
    initialize_complex_complexity_handler
)

from .multi_monitor_query_handler import (
    MultiMonitorQueryHandler,
    MultiMonitorQueryType,
    MonitorContentResult,
    DisplayListResult,
    WindowLocationResult,
    MoveSpaceResult,
    SpaceContent,
    get_multi_monitor_query_handler,
    initialize_multi_monitor_query_handler
)

__all__ = [
    'MultiSpaceQueryHandler',
    'get_multi_space_handler',
    'initialize_multi_space_handler',
    'MultiSpaceQueryType',
    'SpaceAnalysisResult',
    'MultiSpaceQueryResult',
    'TemporalQueryHandler',
    'get_temporal_query_handler',
    'initialize_temporal_handler',
    'TemporalQueryType',
    'TemporalQueryResult',
    'ScreenshotManager',
    'ImageDiffer',
    'PredictiveQueryHandler',
    'PredictiveQueryRequest',
    'PredictiveQueryResponse',
    'ClaudeVisionAnalyzer',
    'get_predictive_handler',
    'initialize_predictive_handler',
    'handle_predictive_query',
    'ActionQueryHandler',
    'ActionQueryResponse',
    'get_action_query_handler',
    'initialize_action_query_handler',
    'handle_action_query',
    'QueryComplexityManager',
    'QueryComplexityClassifier',
    'QueryRouter',
    'ClassifiedQuery',
    'ComplexityMetrics',
    'QueryComplexity',
    'get_query_complexity_manager',
    'initialize_query_complexity_manager',
    'MediumComplexityHandler',
    'MediumQueryType',
    'SpaceCapture',
    'MediumQueryResult',
    'get_medium_complexity_handler',
    'initialize_medium_complexity_handler',
    'ComplexComplexityHandler',
    'ComplexQueryType',
    'SpaceSnapshot',
    'TemporalAnalysis',
    'CrossSpaceAnalysis',
    'PredictiveAnalysis',
    'ComplexQueryResult',
    'get_complex_complexity_handler',
    'initialize_complex_complexity_handler',
    'MultiMonitorQueryHandler',
    'MultiMonitorQueryType',
    'MonitorContentResult',
    'DisplayListResult',
    'WindowLocationResult',
    'MoveSpaceResult',
    'SpaceContent',
    'get_multi_monitor_query_handler',
    'initialize_multi_monitor_query_handler',
]
