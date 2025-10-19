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
]
