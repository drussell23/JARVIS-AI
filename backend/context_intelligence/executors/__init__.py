"""
Executors Module
================

Provides task execution capabilities for Context Intelligence.
"""

from .document_writer import get_document_writer, parse_document_request, DocumentRequest

__all__ = ["get_document_writer", "parse_document_request", "DocumentRequest"]
