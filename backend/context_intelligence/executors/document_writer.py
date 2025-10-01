"""
Document Writer Executor
========================

Robust, dynamic document creation with:
- Chrome browser automation (non-incognito)
- Google Docs API for title + content
- Claude API streaming for content generation
- Smart error handling and retry logic
- Real-time progress with dynamic responses
- Zero hardcoding - fully configurable
"""

import asyncio
import logging
import re
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

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
    # Content specs
    topic: str
    document_type: DocumentType
    word_count: Optional[int] = None
    page_count: Optional[int] = None
    platform: DocumentPlatform = DocumentPlatform.GOOGLE_DOCS

    # Configuration
    browser: str = "Chrome"
    use_google_docs_api: bool = True
    claude_model: str = "claude-3-5-sonnet-20241022"
    claude_max_tokens: int = 4096
    chunk_size: int = 300
    stream_delay: float = 0.3
    max_retries: int = 3
    retry_delay: float = 2.0

    # Metadata
    title: str = ""
    original_command: str = ""
    additional_requirements: str = ""

    def __post_init__(self):
        """Generate title if not provided"""
        if not self.title:
            self.title = self._generate_title()

    def _generate_title(self) -> str:
        """Generate intelligent title from topic and type"""
        topic_words = self.topic.split()
        capitalized = ' '.join(word.capitalize() for word in topic_words)

        type_formats = {
            DocumentType.ESSAY: f"{capitalized}: An Essay",
            DocumentType.REPORT: f"Report on {capitalized}",
            DocumentType.PAPER: f"Research Paper: {capitalized}",
            DocumentType.ARTICLE: capitalized,
            DocumentType.BLOG_POST: capitalized,
            DocumentType.GENERAL: capitalized
        }
        return type_formats.get(self.document_type, capitalized)

    def get_length_spec(self) -> str:
        """Get length specification for prompts"""
        if self.word_count:
            return f"approximately {self.word_count} words"
        elif self.page_count:
            return f"approximately {self.page_count} pages"

        # Intelligent defaults based on document type
        defaults = {
            DocumentType.ESSAY: "500-750 words",
            DocumentType.REPORT: "1000-1500 words",
            DocumentType.PAPER: "1500-2500 words",
            DocumentType.ARTICLE: "800-1200 words",
            DocumentType.BLOG_POST: "600-1000 words"
        }
        return defaults.get(self.document_type, "750-1000 words")


class DocumentWriterExecutor:
    """Orchestrates document creation workflow with Google Docs API"""

    def __init__(self):
        """Initialize document writer"""
        self._google_docs = None
        self._claude = None
        self._browser = None

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
            # Phase 1: Announce
            article = "an" if request.document_type.value[0] in 'aeiou' else "a"
            await self._narrate(progress_callback, websocket,
                f"Understood, Sir. I'll create {article} {request.document_type.value} about {request.topic}.")

            # Phase 2: Initialize services
            if not await self._initialize_services(request):
                return {
                    "success": False,
                    "error": "Failed to initialize required services"
                }

            # Phase 3: Create Google Doc via API
            await self._narrate(progress_callback, websocket,
                f"Creating new document in Google Docs...")

            doc_info = await self._create_google_doc(request)
            if not doc_info:
                return {
                    "success": False,
                    "error": "Failed to create Google Doc"
                }

            document_id = doc_info['document_id']
            document_url = doc_info['document_url']

            # Phase 4: Open in Chrome
            await self._narrate(progress_callback, websocket,
                f"Opening document in Chrome...")

            await self._open_in_browser(document_url, request)
            await asyncio.sleep(1.5)  # Let user see the document open

            # Phase 5: Generate outline
            await self._narrate(progress_callback, websocket,
                f"Analyzing the topic and creating an outline...")

            outline = await self._generate_outline(request)

            # Phase 6: Stream content
            await self._narrate(progress_callback, websocket,
                f"Writing your {request.document_type.value}. "
                f"You'll see the content appear in real-time, Sir.")

            word_count = await self._stream_content(
                document_id, request, outline,
                progress_callback, websocket
            )

            # Phase 7: Completion
            await self._narrate(progress_callback, websocket,
                f"Your {request.document_type.value} is complete, Sir. "
                f"{word_count} words written on '{request.topic}'.")

            return {
                "success": True,
                "document_id": document_id,
                "document_url": document_url,
                "title": request.title,
                "word_count": word_count,
                "platform": request.platform.value,
                "topic": request.topic
            }

        except Exception as e:
            logger.error(f"Error in document creation: {e}", exc_info=True)
            await self._narrate(progress_callback, websocket,
                f"I encountered an error, Sir: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _initialize_services(self, request: DocumentRequest) -> bool:
        """Initialize required services"""
        try:
            # Initialize Google Docs API
            if request.use_google_docs_api:
                from ..automation.google_docs_api import get_google_docs_client
                self._google_docs = get_google_docs_client()

                if not await self._google_docs.authenticate():
                    logger.error("Failed to authenticate with Google Docs API")
                    return False

            # Initialize Claude
            from ..automation.claude_streamer import get_claude_streamer
            self._claude = get_claude_streamer()

            # Initialize browser
            from ..automation.browser_controller import get_browser_controller
            self._browser = get_browser_controller(request.browser)

            return True

        except Exception as e:
            logger.error(f"Error initializing services: {e}")
            return False

    async def _create_google_doc(self, request: DocumentRequest) -> Optional[Dict[str, Any]]:
        """Create Google Doc via API with retry logic"""
        for attempt in range(request.max_retries):
            try:
                doc_info = await self._google_docs.create_document(request.title)
                if doc_info:
                    logger.info(f"Created Google Doc: {doc_info['document_id']}")
                    return doc_info
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < request.max_retries - 1:
                    await asyncio.sleep(request.retry_delay)
        return None

    async def _open_in_browser(self, url: str, request: DocumentRequest) -> bool:
        """Open document in browser"""
        try:
            success = await self._browser.navigate(url)
            if success:
                logger.info(f"Opened document in {request.browser}")
            return success
        except Exception as e:
            logger.error(f"Error opening in browser: {e}")
            return False

    async def _generate_outline(self, request: DocumentRequest) -> Dict[str, Any]:
        """Generate document outline using Claude"""
        article = "an" if request.document_type.value[0] in 'aeiou' else "a"
        outline_prompt = f"""Create a detailed outline for {article} {request.document_type.value} about "{request.topic}".

Target length: {request.get_length_spec()}

The outline should include:
- A compelling title (if different from "{request.title}")
- Introduction points
- Main sections with key arguments/points
- Conclusion points

{request.additional_requirements}

Provide a clear, structured outline."""

        try:
            outline = await self._claude.generate_outline(outline_prompt)
            return outline
        except Exception as e:
            logger.error(f"Error generating outline: {e}")
            return {
                "title": request.title,
                "sections": [
                    {"name": "Introduction", "points": []},
                    {"name": "Main Content", "points": []},
                    {"name": "Conclusion", "points": []}
                ]
            }

    async def _stream_content(self,
                            document_id: str,
                            request: DocumentRequest,
                            outline: Dict[str, Any],
                            progress_callback: Optional[Callable],
                            websocket) -> int:
        """Stream content to Google Doc via API"""
        content_prompt = self._build_content_prompt(request, outline)

        word_count = 0
        buffer = ""
        progress_interval = 100
        next_milestone = progress_interval

        try:
            async for chunk in self._claude.stream_content(
                content_prompt,
                max_tokens=request.claude_max_tokens,
                model=request.claude_model
            ):
                buffer += chunk

                # Write in chunks to Google Docs API
                if len(buffer) >= request.chunk_size:
                    success = await self._google_docs.append_text(document_id, buffer)

                    if success:
                        word_count += len(buffer.split())

                        # Progress update
                        if word_count >= next_milestone:
                            await self._narrate(progress_callback, websocket,
                                f"Writing... {word_count} words so far.")
                            next_milestone += progress_interval

                    buffer = ""
                    await asyncio.sleep(request.stream_delay)

            # Write remaining buffer
            if buffer:
                await self._google_docs.append_text(document_id, buffer)
                word_count += len(buffer.split())

            return word_count

        except Exception as e:
            logger.error(f"Error streaming content: {e}")
            return word_count

    def _build_content_prompt(self, request: DocumentRequest, outline: Dict[str, Any]) -> str:
        """Build comprehensive prompt for content generation"""
        prompt = f"""Write a complete, high-quality {request.document_type.value} about "{request.topic}".

Target length: {request.get_length_spec()}

Title: {request.title}

Outline:
{self._format_outline(outline)}

Requirements:
- Professional, well-researched content
- Clear structure with proper transitions
- Engaging and informative writing
- Academic/professional tone appropriate for a {request.document_type.value}
{f"- {request.additional_requirements}" if request.additional_requirements else ""}

Write the complete {request.document_type.value} now, starting with the introduction:"""

        return prompt

    def _format_outline(self, outline: Dict[str, Any]) -> str:
        """Format outline for prompt"""
        lines = []
        for section in outline.get('sections', []):
            lines.append(f"- {section['name']}")
            for point in section.get('points', []):
                lines.append(f"  * {point}")
        return '\n'.join(lines)

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
    Parse command into DocumentRequest (zero hardcoding)

    Args:
        command: Original command string
        intent: Parsed intent information

    Returns:
        DocumentRequest object
    """
    command_lower = command.lower()

    # Detect document type
    type_patterns = {
        DocumentType.ESSAY: r'\bessay\b',
        DocumentType.REPORT: r'\breport\b',
        DocumentType.PAPER: r'\bpaper\b',
        DocumentType.ARTICLE: r'\barticle\b',
        DocumentType.BLOG_POST: r'\bblog\s*post\b|\bblog\b'
    }

    doc_type = DocumentType.GENERAL
    for dtype, pattern in type_patterns.items():
        if re.search(pattern, command_lower):
            doc_type = dtype
            break

    # Extract word count
    word_count = None
    word_match = re.search(r'(\d+)\s*words?', command_lower)
    if word_match:
        word_count = int(word_match.group(1))

    # Extract page count
    page_count = None
    page_match = re.search(r'(\d+)\s*pages?', command_lower)
    if page_match:
        page_count = int(page_match.group(1))

    # Extract topic dynamically
    topic = _extract_topic(command, doc_type)

    # Determine platform
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
    """Dynamically extract topic from command"""
    command_lower = command.lower()

    # Pattern list (ordered by specificity)
    patterns = [
        r'(?:write|create|draft|compose|generate)\s+(?:me\s+)?(?:an?\s+)?(?:\d+\s+(?:word|page)s?\s+)?(?:essay|report|paper|article|document|blog\s*post)?\s+(?:about|on|regarding)\s+(.+?)(?:\s+in\s+(?:google\s*)?docs?|\s+for\s+me|$)',
        r'(?:essay|report|paper|article|document)\s+(?:about|on|regarding)\s+(.+?)(?:\s+in\s+(?:google\s*)?docs?|$)',
        r'(?:write|create|draft)\s+(?:me\s+)?(?:an?\s+)?(.+?)(?:\s+essay|\s+report|\s+paper|\s+article|$)',
    ]

    for pattern in patterns:
        match = re.search(pattern, command_lower, re.IGNORECASE)
        if match:
            topic = match.group(1).strip()
            # Clean up the topic
            topic = re.sub(r'\s+in\s+(?:google\s+)?docs?$', '', topic)
            topic = re.sub(r'\s+for\s+me$', '', topic)
            topic = re.sub(r'^\d+\s+(?:word|page)s?\s+', '', topic)
            if topic:
                return topic

    return "the requested topic"


# Global instance
_document_writer: Optional[DocumentWriterExecutor] = None


def get_document_writer() -> DocumentWriterExecutor:
    """Get or create global document writer instance"""
    global _document_writer
    if _document_writer is None:
        _document_writer = DocumentWriterExecutor()
    return _document_writer
