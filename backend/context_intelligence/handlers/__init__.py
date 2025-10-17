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

__all__ = [
    'MultiSpaceQueryHandler',
    'get_multi_space_handler',
    'initialize_multi_space_handler',
    'MultiSpaceQueryType',
    'SpaceAnalysisResult',
    'MultiSpaceQueryResult'
]
