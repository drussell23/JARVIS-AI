"""
Context Intelligence Resolvers
==============================

Resolvers for ambiguous and contextual queries
"""

from .contextual_query_resolver import (
    ContextualQueryResolver,
    get_contextual_resolver,
    QueryResolution,
    ResolutionStrategy,
    ContextualReference
)

__all__ = [
    'ContextualQueryResolver',
    'get_contextual_resolver',
    'QueryResolution',
    'ResolutionStrategy',
    'ContextualReference'
]
