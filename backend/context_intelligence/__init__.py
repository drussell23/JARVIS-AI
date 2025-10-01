"""
Context Intelligence Module
============================

Provides intelligent task execution and automation capabilities.
"""

from .executors import get_document_writer, parse_document_request, DocumentRequest
from .automation import get_browser_controller, get_google_docs_client, get_claude_streamer

__all__ = [
    "get_document_writer",
    "parse_document_request",
    "DocumentRequest",
    "get_browser_controller",
    "get_google_docs_client",
    "get_claude_streamer",
]
