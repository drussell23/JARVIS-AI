"""
Claude API Content Streamer
===========================

Streams content generation from Claude API for real-time document writing.
"""

import asyncio
import logging
from typing import AsyncIterator, Optional, Callable, Dict, Any
import os

logger = logging.getLogger(__name__)

# Try to import anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic package not available - install with: pip install anthropic")


class ClaudeContentStreamer:
    """Streams content generation from Claude API"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Claude streamer

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self._client = None

        if not self.api_key:
            logger.warning("No Anthropic API key provided - streaming will use mock data")

    def _ensure_client(self):
        """Ensure we have a Claude API client"""
        if not ANTHROPIC_AVAILABLE:
            return False

        if self._client is None and self.api_key:
            self._client = anthropic.Anthropic(api_key=self.api_key)

        return self._client is not None

    async def stream_content(self,
                           prompt: str,
                           max_tokens: int = 4096,
                           model: str = "claude-3-5-sonnet-20241022",
                           chunk_callback: Optional[Callable[[str], None]] = None) -> AsyncIterator[str]:
        """
        Stream content generation from Claude

        Args:
            prompt: The content generation prompt
            max_tokens: Maximum tokens to generate
            model: Claude model to use
            chunk_callback: Optional callback for each chunk

        Yields:
            Content chunks as they're generated
        """
        if not self._ensure_client():
            # Fallback to mock streaming if API not available
            logger.info("Using mock content streaming")
            async for chunk in self._mock_stream(prompt):
                if chunk_callback:
                    chunk_callback(chunk)
                yield chunk
            return

        try:
            # Use the streaming API
            with self._client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            ) as stream:
                for text in stream.text_stream:
                    if text:
                        if chunk_callback:
                            chunk_callback(text)
                        yield text

                        # Small delay to prevent overwhelming the system
                        await asyncio.sleep(0.01)

        except Exception as e:
            logger.error(f"Error streaming from Claude API: {e}")
            # Fallback to mock on error
            async for chunk in self._mock_stream(prompt):
                if chunk_callback:
                    chunk_callback(chunk)
                yield chunk

    async def generate_outline(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a document outline

        Args:
            prompt: Outline generation prompt

        Returns:
            Structured outline dictionary
        """
        if not self._ensure_client():
            return self._mock_outline()

        try:
            response = self._client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2048,
                messages=[{
                    "role": "user",
                    "content": prompt + "\n\nProvide the outline in a clear, structured format."
                }]
            )

            outline_text = response.content[0].text

            # Parse the outline
            return self._parse_outline(outline_text)

        except Exception as e:
            logger.error(f"Error generating outline: {e}")
            return self._mock_outline()

    def _parse_outline(self, outline_text: str) -> Dict[str, Any]:
        """Parse outline text into structured format"""
        lines = outline_text.strip().split('\n')

        title = ""
        sections = []
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Title (usually first line or starts with #)
            if line.startswith('# ') or (not title and not sections):
                title = line.lstrip('#').strip()
                continue

            # Section heading (## or numbered)
            if line.startswith('## ') or (line[0].isdigit() and '.' in line[:3]):
                if current_section:
                    sections.append(current_section)

                section_name = line.lstrip('#').lstrip('0123456789. ').strip()
                current_section = {
                    "name": section_name,
                    "points": []
                }
                continue

            # Section point (- or * or numbered sub-item)
            if current_section and (line.startswith('- ') or line.startswith('* ') or
                                   (line[0:2].strip() and line[0:2].strip()[0].isdigit())):
                point = line.lstrip('-*0123456789. ').strip()
                current_section["points"].append(point)

        # Add final section
        if current_section:
            sections.append(current_section)

        return {
            "title": title or "Untitled Document",
            "sections": sections
        }

    def _mock_outline(self) -> Dict[str, Any]:
        """Generate a mock outline for testing"""
        return {
            "title": "Sample Document",
            "sections": [
                {
                    "name": "Introduction",
                    "points": ["Context and background", "Thesis statement"]
                },
                {
                    "name": "Main Content",
                    "points": ["Key point 1", "Key point 2", "Key point 3"]
                },
                {
                    "name": "Conclusion",
                    "points": ["Summary", "Final thoughts"]
                }
            ]
        }

    async def _mock_stream(self, prompt: str) -> AsyncIterator[str]:
        """Mock content streaming for testing"""
        mock_content = """# Sample Document

## Introduction

This is a sample document generated for testing purposes. The content would normally come from the Claude API, but we're using this mock version for development.

## Main Content

Here we would have the main body of the document, with detailed paragraphs covering the topic in depth.

This section would be tailored to the specific prompt provided by the user.

## Conclusion

In conclusion, this demonstrates the basic structure and flow of the document generation system.

The actual content would be much more comprehensive and relevant to the user's request.
"""

        # Stream the mock content word by word
        words = mock_content.split()
        for i, word in enumerate(words):
            # Add space except for first word
            chunk = word if i == 0 else " " + word
            yield chunk
            await asyncio.sleep(0.05)  # Simulate streaming delay


# Global instance
_claude_streamer: Optional[ClaudeContentStreamer] = None


def get_claude_streamer(api_key: Optional[str] = None) -> ClaudeContentStreamer:
    """Get or create global Claude streamer instance"""
    global _claude_streamer
    if _claude_streamer is None:
        _claude_streamer = ClaudeContentStreamer(api_key)
    return _claude_streamer
