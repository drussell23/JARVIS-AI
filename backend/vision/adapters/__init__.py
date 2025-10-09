"""
Vision Adapters Package
Provides unified interfaces for vision-related operations.
"""
from .ocr import ocr_text_from_snapshot, clear_ocr_cache
from .analysis import extract_errors, suggest_fix, summarize_terminal_state
from .page import get_page_title, get_readable_text, extract_page_content
from .code import analyze_code_window

__all__ = [
    # OCR
    'ocr_text_from_snapshot',
    'clear_ocr_cache',

    # Analysis
    'extract_errors',
    'suggest_fix',
    'summarize_terminal_state',

    # Page
    'get_page_title',
    'get_readable_text',
    'extract_page_content',

    # Code
    'analyze_code_window',
]
