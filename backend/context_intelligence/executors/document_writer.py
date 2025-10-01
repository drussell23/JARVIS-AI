"""
Document Writer Executor
========================

Orchestrates end-to-end document creation including:
- Browser automation (Google Docs/Word Online)
- Content streaming from Claude API
- Real-time narration and progress updates
"""

import asyncio
import logging
import re
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Supported document types"""
    ESSAY = "essay"
    REPORT = "report"
    PAPER = "paper"
    ARTICLE = "article"
    BLOG_POST = "blog_post"
    GENERAL = "general"


class DocumentPlatform(Enum):
    """Supported document platforms"""
    GOOGLE_DOCS = "google_docs"
    WORD_ONLINE = "word_online"
    LOCAL_WORD = "local_word"
    TEXT_EDITOR = "text_editor"


@dataclass
class DocumentRequest:
    """Represents a document creation request"""
    topic: str
    document_type: DocumentType
    word_count: Optional[int] = None
    page_count: Optional[int] = None
    platform: DocumentPlatform = DocumentPlatform.GOOGLE_DOCS
    additional_requirements: str = ""
    original_command: str = ""


class DocumentWriterExecutor:
    """Orchestrates document creation workflow"""

    def __init__(self):
        """Initialize document writer"""
        self._browser_automation = None
        self._claude_client = None

    async def create_document(self,
                            request: DocumentRequest,
                            progress_callback: Optional[Callable] = None,
                            websocket = None) -> Dict[str, Any]:
        """
        Execute end-to-end document creation

        Args:
            request: Document creation request
            progress_callback: Callback for narration updates
            websocket: WebSocket for real-time updates

        Returns:
            Execution result with document details
        """
        try:
            # Phase 1: Announce intention
            await self._narrate(progress_callback, websocket,
                f"Understood, Sir. I'll create a {request.document_type.value} about {request.topic}.")

            # Phase 2: Open document platform
            await self._narrate(progress_callback, websocket,
                f"Opening {request.platform.value.replace('_', ' ')}...")

            doc_url = await self._open_document_platform(request.platform)

            if not doc_url:
                return {
                    "success": False,
                    "error": f"Failed to open {request.platform.value}"
                }

            await self._narrate(progress_callback, websocket,
                "Document ready. Beginning content generation...")

            # Phase 3: Generate content structure
            await self._narrate(progress_callback, websocket,
                "Analyzing topic and creating document outline...")

            outline = await self._generate_outline(request)

            # Phase 4: Stream content
            await self._narrate(progress_callback, websocket,
                "Writing content. You'll see it appear in real-time, Sir.")

            content_result = await self._stream_content_to_document(
                request, outline, doc_url, progress_callback, websocket
            )

            if not content_result["success"]:
                return content_result

            # Phase 5: Finalization
            await self._narrate(progress_callback, websocket,
                f"Document complete, Sir. {content_result['stats']['word_count']} words written.")

            return {
                "success": True,
                "document_url": doc_url,
                "stats": content_result["stats"],
                "platform": request.platform.value,
                "topic": request.topic
            }

        except Exception as e:
            logger.error(f"Error creating document: {e}", exc_info=True)
            await self._narrate(progress_callback, websocket,
                f"I encountered an error, Sir: {str(e)}")

            return {
                "success": False,
                "error": str(e)
            }

    async def _open_document_platform(self, platform: DocumentPlatform) -> Optional[str]:
        """Open the target document platform"""
        if platform == DocumentPlatform.GOOGLE_DOCS:
            return await self._open_google_docs()
        elif platform == DocumentPlatform.WORD_ONLINE:
            return await self._open_word_online()
        else:
            logger.warning(f"Platform {platform.value} not yet implemented")
            return None

    async def _open_google_docs(self) -> Optional[str]:
        """
        Open Google Docs in browser

        Returns URL of the new document
        """
        try:
            # Import browser automation
            from ..automation.browser_controller import get_browser_controller
            browser = get_browser_controller()

            # Navigate to Google Docs new document
            doc_url = "https://docs.google.com/document/create"
            await browser.navigate(doc_url)

            # Wait for document to load
            await asyncio.sleep(3)

            # Get actual document URL (will have document ID)
            actual_url = await browser.get_current_url()

            logger.info(f"Opened Google Docs: {actual_url}")
            return actual_url

        except ImportError:
            logger.warning("Browser automation not available, will create placeholder")
            # For now, just return a placeholder URL
            return "https://docs.google.com/document/d/PLACEHOLDER"
        except Exception as e:
            logger.error(f"Failed to open Google Docs: {e}")
            return None

    async def _open_word_online(self) -> Optional[str]:
        """Open Microsoft Word Online"""
        # TODO: Implement Word Online automation
        logger.warning("Word Online not yet implemented")
        return None

    async def _generate_outline(self, request: DocumentRequest) -> Dict[str, Any]:
        """
        Generate document outline using Claude

        Returns structured outline
        """
        # Build prompt for outline generation
        word_target = f"{request.word_count} words" if request.word_count else ""
        page_target = f"{request.page_count} pages" if request.page_count else ""
        length_spec = word_target or page_target or "appropriate length"

        outline_prompt = f"""Create a detailed outline for a {request.document_type.value} about "{request.topic}".
Target length: {length_spec}

The outline should include:
- Title
- Introduction points
- Main sections with key points
- Conclusion points

{request.additional_requirements}

Provide the outline in a structured format."""

        # Use Claude API to generate outline
        try:
            from ..automation.claude_streamer import get_claude_streamer
            claude = get_claude_streamer()
            outline = await claude.generate_outline(outline_prompt)
            return outline
        except Exception as e:
            logger.error(f"Error generating outline: {e}")
            # Fallback to basic outline
            return {
                "title": f"{request.topic}",
                "sections": [
                    {"name": "Introduction", "points": []},
                    {"name": "Main Content", "points": []},
                    {"name": "Conclusion", "points": []}
                ]
            }

    async def _stream_content_to_document(self,
                                         request: DocumentRequest,
                                         outline: Dict[str, Any],
                                         doc_url: str,
                                         progress_callback: Optional[Callable],
                                         websocket) -> Dict[str, Any]:
        """
        Stream content generation to the document in real-time

        This is the core function that:
        1. Generates content via Claude API
        2. Types it into the document
        3. Provides progress updates
        """
        try:
            # Build comprehensive prompt
            content_prompt = self._build_content_prompt(request, outline)

            # Initialize stats
            total_words = 0
            sections_completed = 0
            content_buffer = ""

            # Get Claude streamer and browser
            from ..automation.claude_streamer import get_claude_streamer
            from ..automation.browser_controller import get_browser_controller

            claude = get_claude_streamer()
            browser = get_browser_controller()

            # Ensure browser is focused
            await browser.focus_browser()
            await asyncio.sleep(1)  # Wait for document to be ready

            # Stream content from Claude and type into document
            chunk_buffer = ""
            async for chunk in claude.stream_content(content_prompt):
                chunk_buffer += chunk
                content_buffer += chunk

                # Type in chunks of ~50 characters for smooth streaming
                if len(chunk_buffer) >= 50:
                    await browser.type_text(chunk_buffer, delay=0)
                    total_words += len(chunk_buffer.split())
                    chunk_buffer = ""

                    # Update progress occasionally
                    if total_words % 100 == 0:
                        await self._narrate(progress_callback, websocket,
                            f"Writing... {total_words} words so far.")

            # Type remaining buffer
            if chunk_buffer:
                await browser.type_text(chunk_buffer, delay=0)
                total_words += len(chunk_buffer.split())

            return {
                "success": True,
                "stats": {
                    "word_count": total_words,
                    "sections_completed": len(outline['sections']),
                    "content_length": len(content_buffer)
                }
            }

        except Exception as e:
            logger.error(f"Error streaming content: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    def _build_content_prompt(self, request: DocumentRequest, outline: Dict[str, Any]) -> str:
        """Build comprehensive prompt for content generation"""
        word_target = f"Target length: {request.word_count} words. " if request.word_count else ""
        page_target = f"Target length: {request.page_count} pages. " if request.page_count else ""

        prompt = f"""Write a complete {request.document_type.value} about "{request.topic}".

{word_target}{page_target}

Structure:
{self._format_outline(outline)}

Requirements:
- Professional, well-researched content
- Clear structure with proper headings
- Engaging and informative writing
- {request.additional_requirements}

Write the complete {request.document_type.value} now:"""

        return prompt

    def _format_outline(self, outline: Dict[str, Any]) -> str:
        """Format outline for prompt"""
        formatted = f"Title: {outline['title']}\n\n"

        for section in outline['sections']:
            formatted += f"- {section['name']}\n"
            for point in section.get('points', []):
                formatted += f"  * {point}\n"

        return formatted

    async def _narrate(self, progress_callback: Optional[Callable],
                      websocket, message: str):
        """Send narration update"""
        logger.info(f"[DOCUMENT WRITER] {message}")

        if progress_callback:
            await progress_callback(message)

        if websocket:
            try:
                import json
                await websocket.send_text(json.dumps({
                    "type": "narration",
                    "message": message
                }))
            except Exception as e:
                logger.debug(f"Could not send narration to websocket: {e}")


def parse_document_request(command: str, intent: Dict[str, Any]) -> DocumentRequest:
    """
    Parse a document creation command into a structured request

    Args:
        command: Original command string
        intent: Parsed intent information

    Returns:
        DocumentRequest object
    """
    command_lower = command.lower()

    # Extract document type
    doc_type = DocumentType.GENERAL
    if "essay" in command_lower:
        doc_type = DocumentType.ESSAY
    elif "report" in command_lower:
        doc_type = DocumentType.REPORT
    elif "paper" in command_lower:
        doc_type = DocumentType.PAPER
    elif "article" in command_lower:
        doc_type = DocumentType.ARTICLE
    elif "blog post" in command_lower or "blog" in command_lower:
        doc_type = DocumentType.BLOG_POST

    # Extract word/page count
    word_count = None
    page_count = None

    word_match = re.search(r'(\d+)\s*words?', command_lower)
    if word_match:
        word_count = int(word_match.group(1))

    page_match = re.search(r'(\d+)\s*pages?', command_lower)
    if page_match:
        page_count = int(page_match.group(1))

    # Extract topic
    topic = _extract_topic(command, doc_type)

    # Determine platform (default to Google Docs)
    platform = DocumentPlatform.GOOGLE_DOCS
    if "word" in command_lower and "online" not in command_lower:
        platform = DocumentPlatform.LOCAL_WORD
    elif "word online" in command_lower:
        platform = DocumentPlatform.WORD_ONLINE

    return DocumentRequest(
        topic=topic,
        document_type=doc_type,
        word_count=word_count,
        page_count=page_count,
        platform=platform,
        original_command=command
    )


def _extract_topic(command: str, doc_type: DocumentType) -> str:
    """Extract document topic from command"""
    command_lower = command.lower()

    # Common patterns for topic extraction
    patterns = [
        r'(?:write|create|draft|compose|generate).*?(?:essay|report|paper|article|document).*?(?:about|on|regarding)\s+(.+)',
        r'(?:write|create|draft|compose|generate).*?(?:about|on|regarding)\s+(.+)',
        r'(?:essay|report|paper|article|document).*?(?:about|on|regarding)\s+(.+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, command_lower)
        if match:
            topic = match.group(1).strip()
            # Clean up the topic
            topic = re.sub(r'\s+in\s+(?:google\s+)?docs?$', '', topic)
            topic = re.sub(r'\s+in\s+word$', '', topic)
            return topic

    # Fallback: use entities from intent
    return "the requested topic"


# Global instance
_document_writer: Optional[DocumentWriterExecutor] = None


def get_document_writer() -> DocumentWriterExecutor:
    """Get or create global document writer instance"""
    global _document_writer
    if _document_writer is None:
        _document_writer = DocumentWriterExecutor()
    return _document_writer
