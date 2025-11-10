"""
Browser Automation Module
=========================

Provides browser automation capabilities for document creation
and other screen-based tasks.
"""

from .browser_controller import get_browser_controller
from .google_docs_api import get_google_docs_client

# Claude streamer causes import hangs - import directly when needed
# from .claude_streamer import get_claude_streamer

__all__ = ['get_browser_controller', 'get_google_docs_client']
