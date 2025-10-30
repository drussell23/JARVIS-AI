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

This module provides comprehensive document creation capabilities with intelligent
narration, async pipeline processing, and support for multiple academic formats.
"""

import asyncio
import logging
import re
from typing import Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# Import async pipeline for non-blocking document operations
try:
    from core.async_pipeline import get_async_pipeline, AdvancedAsyncPipeline
except ImportError:
    # Fallback - define stub functions
    def get_async_pipeline():
        return None
    class AdvancedAsyncPipeline:
        pass

logger = logging.getLogger(__name__)


# Formatting templates for different academic styles
FORMAT_SPECIFICATIONS = {
    "mla": {
        "name": "MLA (Modern Language Association)",
        "header": "Last name and page number in upper right corner",
        "title": "Centered, no bold/italics/underline",
        "font": "Times New Roman, 12pt",
        "spacing": "Double-spaced throughout",
        "margins": "1-inch margins on all sides",
        "heading": "Name, instructor, course, date (day Month year) in upper left",
        "citations": "In-text citations with author's last name and page number (Smith 123)",
        "works_cited": "Works Cited page at end, alphabetical by author's last name",
        "paragraphs": "First line indented 0.5 inches",
        "requirements": """
- Times New Roman 12pt font
- Double-spaced throughout (including Works Cited)
- 1-inch margins on all sides
- Header: Your last name and page number (flush right)
- First page heading (flush left): Your Name, Instructor Name, Course, Date
- Title centered (no bold, italics, or underline)
- Indent first line of each paragraph 0.5 inch
- In-text citations: (Author Page#)
- Works Cited page at end with hanging indent"""
    },
    "apa": {
        "name": "APA (American Psychological Association)",
        "header": "Running head on every page (left) and page number (right)",
        "title": "Centered, bold, title case",
        "font": "Times New Roman, 12pt or Calibri, 11pt",
        "spacing": "Double-spaced throughout",
        "margins": "1-inch margins on all sides",
        "heading": "Title page with title, author, institutional affiliation",
        "citations": "In-text citations with author and year (Smith, 2023)",
        "references": "References page at end, alphabetical by author's last name, hanging indent",
        "paragraphs": "First line indented 0.5 inches",
        "requirements": """
- Times New Roman 12pt or Calibri 11pt font
- Double-spaced throughout
- 1-inch margins on all sides
- Running head (shortened title) on all pages, flush left
- Page number on all pages, flush right
- Title page with title (bold, centered), author name, institutional affiliation
- Headings: Level 1 (Centered, Bold), Level 2 (Flush Left, Bold)
- In-text citations: (Author, Year) or (Author, Year, p. #)
- References page with hanging indent, alphabetical order"""
    },
    "chicago": {
        "name": "Chicago/Turabian",
        "header": "Page number in top right or bottom center",
        "title": "Centered, 1/3 down the page",
        "font": "Times New Roman, 12pt",
        "spacing": "Double-spaced",
        "margins": "1-inch margins",
        "heading": "Title page with title, author, course, instructor, date",
        "citations": "Footnotes or endnotes with superscript numbers",
        "bibliography": "Bibliography page at end, alphabetical, hanging indent",
        "paragraphs": "First line indented 0.5 inches",
        "requirements": """
- Times New Roman 12pt font
- Double-spaced (except footnotes/endnotes)
- 1-inch margins
- Title page with title centered 1/3 down page
- Page numbers in header (top right) or footer (bottom center)
- Footnotes or endnotes with full citation on first reference
- Shortened citations for subsequent references
- Bibliography page with hanging indent, alphabetical order"""
    },
    "ieee": {
        "name": "IEEE (Institute of Electrical and Electronics Engineers)",
        "header": "Paper title on left, page number on right",
        "title": "Centered, 18pt Times New Roman",
        "font": "Times New Roman, 10pt",
        "spacing": "Single-spaced",
        "margins": "3/4-inch margins",
        "heading": "Author names centered below title",
        "citations": "Numbered citations in brackets [1]",
        "references": "References numbered in order of appearance",
        "paragraphs": "First line indented 0.25 inches",
        "requirements": """
- Times New Roman 10pt font
- Single-spaced
- Two-column format for most sections
- Title: 18pt, centered
- Author names below title
- Abstract in italics
- In-text citations: numbered [1], [2]
- References numbered in order of appearance"""
    },
    "harvard": {
        "name": "Harvard Referencing",
        "header": "Page number centered at bottom",
        "title": "Centered, bold",
        "font": "Times New Roman or Arial, 12pt",
        "spacing": "1.5 or double-spaced",
        "margins": "1-inch margins",
        "heading": "Title page optional",
        "citations": "In-text: (Author Year) or (Author Year: page)",
        "references": "Reference list at end, alphabetical, hanging indent",
        "paragraphs": "Justified or left-aligned",
        "requirements": """
- Times New Roman or Arial 12pt font
- 1.5 or double-spaced
- 1-inch margins
- In-text citations: (Author Year) or (Author, Year, p.#)
- Reference list at end with hanging indent
- Alphabetical order by author's surname
- Multiple works by same author ordered chronologically"""
    },
    "none": {
        "name": "No specific format",
        "requirements": "Write in a clear, professional manner without specific formatting requirements"
    }
}


class DocumentType(Enum):
    """Supported document types for creation.
    
    Attributes:
        ESSAY: Academic essay format
        REPORT: Business or technical report
        PAPER: General academic paper
        RESEARCH_PAPER: Research-focused academic paper
        ARTICLE: Article or publication format
        BLOG_POST: Blog post format
        GENERAL: General document type
    """
    ESSAY = "essay"
    REPORT = "report"
    PAPER = "paper"
    RESEARCH_PAPER = "research_paper"
    ARTICLE = "article"
    BLOG_POST = "blog_post"
    GENERAL = "general"


class DocumentFormat(Enum):
    """Academic and professional formatting styles.
    
    Attributes:
        MLA: Modern Language Association format
        APA: American Psychological Association format
        CHICAGO: Chicago/Turabian format
        IEEE: Institute of Electrical and Electronics Engineers format
        HARVARD: Harvard referencing format
        NONE: No specific formatting requirements
    """
    MLA = "mla"
    APA = "apa"
    CHICAGO = "chicago"
    IEEE = "ieee"
    HARVARD = "harvard"
    NONE = "none"


class DocumentPlatform(Enum):
    """Supported document platforms for creation.
    
    Attributes:
        GOOGLE_DOCS: Google Docs platform
        WORD_ONLINE: Microsoft Word Online
        LOCAL_WORD: Local Microsoft Word installation
        TEXT_EDITOR: Plain text editor
    """
    GOOGLE_DOCS = "google_docs"
    WORD_ONLINE = "word_online"
    LOCAL_WORD = "local_word"
    TEXT_EDITOR = "text_editor"


@dataclass
class DocumentRequest:
    """Represents a document creation request with all necessary parameters.
    
    This class encapsulates all the information needed to create a document,
    including content specifications, formatting requirements, and metadata.
    
    Attributes:
        topic: The main topic or subject of the document
        document_type: Type of document to create (essay, report, etc.)
        word_count: Target word count for the document
        page_count: Target page count for the document
        platform: Platform to create the document on
        formatting: Academic formatting style to use
        browser: Browser to use for document creation
        use_google_docs_api: Whether to use Google Docs API
        claude_model: Claude AI model to use for content generation
        claude_max_tokens: Maximum tokens for Claude API calls
        chunk_size: Size of content chunks for streaming
        stream_delay: Delay between content chunks
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retry attempts
        title: Document title (auto-generated if empty)
        original_command: Original user command
        additional_requirements: Additional formatting or content requirements
        author_name: Author name for document metadata
        instructor_name: Instructor name for academic formats
        course_name: Course name for academic formats
        institution: Institution name for academic formats
    """
    # Content specs
    topic: str
    document_type: DocumentType
    word_count: Optional[int] = None
    page_count: Optional[int] = None
    platform: DocumentPlatform = DocumentPlatform.GOOGLE_DOCS
    formatting: DocumentFormat = DocumentFormat.NONE

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
    author_name: str = "Student Name"
    instructor_name: str = ""
    course_name: str = ""
    institution: str = ""

    def __post_init__(self) -> None:
        """Initialize derived attributes after object creation.
        
        Generates title and auto-detects format if not provided.
        """
        if not self.title:
            self.title = self._generate_title()

        # Auto-detect format based on document type if not explicitly set
        if self.formatting == DocumentFormat.NONE:
            self.formatting = self._auto_detect_format()

    def _auto_detect_format(self) -> DocumentFormat:
        """Intelligently auto-detect formatting style based on document type.
        
        Returns:
            DocumentFormat: The most appropriate format for the document type
        """
        # Smart mapping: document type -> most common format
        format_mapping = {
            DocumentType.ESSAY: DocumentFormat.MLA,  # Essays typically use MLA
            DocumentType.RESEARCH_PAPER: DocumentFormat.MLA,  # Academic research papers often use MLA
            DocumentType.PAPER: DocumentFormat.APA,  # General papers often use APA
            DocumentType.REPORT: DocumentFormat.APA,  # Business/scientific reports use APA
            DocumentType.ARTICLE: DocumentFormat.CHICAGO,  # Articles often use Chicago
            DocumentType.BLOG_POST: DocumentFormat.NONE,  # Blog posts are informal
            DocumentType.GENERAL: DocumentFormat.NONE
        }

        detected = format_mapping.get(self.document_type, DocumentFormat.NONE)
        logger.info(f"Auto-detected format: {detected.value} for {self.document_type.value}")
        return detected

    def _generate_title(self) -> str:
        """Generate intelligent title from topic and document type.
        
        Returns:
            str: Generated title appropriate for the document type
        """
        topic_words = self.topic.split()
        capitalized = ' '.join(word.capitalize() for word in topic_words)

        type_formats = {
            DocumentType.ESSAY: f"{capitalized}: An Essay",
            DocumentType.REPORT: f"Report on {capitalized}",
            DocumentType.PAPER: f"Research Paper: {capitalized}",
            DocumentType.RESEARCH_PAPER: f"{capitalized}",
            DocumentType.ARTICLE: capitalized,
            DocumentType.BLOG_POST: capitalized,
            DocumentType.GENERAL: capitalized
        }
        return type_formats.get(self.document_type, capitalized)

    def get_length_spec(self) -> str:
        """Get length specification for content generation prompts.
        
        Returns:
            str: Human-readable length specification
        """
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
    """Orchestrates document creation workflow with Google Docs API and AI content generation.
    
    This class manages the complete document creation process, including service initialization,
    document creation, content generation with Claude AI, and real-time progress narration.
    
    Attributes:
        _google_docs: Google Docs API client
        _claude: Claude AI streaming client
        _browser: Browser automation controller
        _intelligent_narrator: AI-powered narration system
        _speech_in_progress: Flag to prevent overlapping speech
        _speech_queue: Queue for managing speech requests
        _last_speech_end_time: Timestamp of last speech completion
        _narration_context: Context for fallback narration
        _last_narration_time: Timestamp of last narration
        pipeline: Async pipeline for non-blocking operations
    """

    def __init__(self) -> None:
        """Initialize document writer with default configuration."""
        self._google_docs = None
        self._claude = None
        self._browser = None
        self._intelligent_narrator = None  # Will be initialized with Claude
        
        # Speech queue management to prevent overlapping
        self._speech_in_progress = False
        self._speech_queue = []
        self._last_speech_end_time = 0
        
        # Fallback context for non-intelligent mode
        self._narration_context = {
            "phase": "initializing",
            "topic": "",
            "progress": 0,
            "current_section": "",
            "word_count": 0,
            "total_words_target": 0
        }
        self._last_narration_time = 0

        # Initialize async pipeline for non-blocking document operations
        self.pipeline = get_async_pipeline()
        self._register_pipeline_stages()

    def _register_pipeline_stages(self) -> None:
        """Register async pipeline stages for document operations.
        
        Sets up the pipeline stages for service initialization, document creation,
        content generation, and content streaming with appropriate timeouts and retry logic.
        """
        # Service initialization stage
        self.pipeline.register_stage(
            "service_init",
            self._init_services_async,
            timeout=15.0,
            retry_count=2,
            required=True
        )

        # Google Doc creation stage
        self.pipeline.register_stage(
            "doc_creation",
            self._create_doc_async,
            timeout=20.0,
            retry_count=3,
            required=True
        )

        # Content generation stage
        self.pipeline.register_stage(
            "content_generation",
            self._generate_content_async,
            timeout=120.0,  # Longer timeout for AI content generation
            retry_count=1,
            required=True
        )

        # Content streaming stage
        self.pipeline.register_stage(
            "content_streaming",
            self._stream_to_doc_async,
            timeout=60.0,
            retry_count=1,
            required=True
        )

    async def _init_services_async(self, context) -> None:
        """Non-blocking service initialization via async pipeline.
        
        Args:
            context: Pipeline context containing request metadata
            
        Raises:
            Exception: If service initialization fails
        """
        request = context.metadata.get("request")

        try:
            # Initialize Google Docs API
            if request.use_google_docs_api:
                from ..automation.google_docs_api import get_google_docs_client
                self._google_docs = get_google_docs_client()

                if not await self._google_docs.authenticate():
                    context.metadata["error"] = "Failed to authenticate with Google Docs API"
                    return

            # Initialize Claude
            from ..automation.claude_streamer import get_claude_streamer
            self._claude = get_claude_streamer()

            # Initialize browser
            from ..automation.browser_controller import get_browser_controller
            self._browser = get_browser_controller(request.browser)

            context.metadata["services_ready"] = True

        except Exception as e:
            logger.error(f"Error initializing services: {e}")
            context.metadata["error"] = str(e)

    async def _create_doc_async(self, context) -> None:
        """Non-blocking Google Doc creation via async pipeline.
        
        Args:
            context: Pipeline context containing request metadata
        """
        request = context.metadata.get("request")

        for attempt in range(request.max_retries):
            try:
                doc_info = await self._google_docs.create_document(request.title)
                if doc_info:
                    logger.info(f"Created Google Doc: {doc_info['document_id']}")
                    context.metadata["doc_info"] = doc_info
                    context.metadata["document_id"] = doc_info['document_id']
                    context.metadata["document_url"] = doc_info['document_url']
                    return
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < request.max_retries - 1:
                    await asyncio.sleep(request.retry_delay)

        context.metadata["error"] = "Failed to create Google Doc after retries"

    async def _generate_content_async(self, context) -> None:
        """Non-blocking content generation via async pipeline.
        
        Args:
            context: Pipeline context containing request and outline metadata
        """
        request = context.metadata.get("request")
        outline = context.metadata.get("outline")

        content_prompt = self._build_content_prompt(request, outline)
        context.metadata["content_prompt"] = content_prompt
        context.metadata["ready_for_streaming"] = True

    async def _stream_to_doc_async(self, context) -> None:
        """Non-blocking content streaming to Google Doc via async pipeline.
        
        Args:
            context: Pipeline context containing document and streaming metadata
        """
        document_id = context.metadata.get("document_id")
        request = context.metadata.get("request")
        content_prompt = context.metadata.get("content_prompt")
        progress_callback = context.metadata.get("progress_callback")
        websocket = context.metadata.get("websocket")

        word_count = 0
        buffer = ""

        try:
            async for chunk in self._claude.stream_content(
                content_prompt,
                max_tokens=request.claude_max_tokens,
                model=request.claude_model
            ):
                buffer += chunk

                if len(buffer) >= request.chunk_size:
                    success = await self._google_docs.append_text(document_id, buffer)

                    if success:
                        current_words = len(buffer.split())
                        word_count += current_words

                    buffer = ""
                    await asyncio.sleep(request.stream_delay)

            # Write remaining buffer
            if buffer:
                await self._google_docs.append_text(document_id, buffer)
                word_count += len(buffer.split())

            context.metadata["word_count"] = word_count

        except Exception as e:
            logger.error(f"Error streaming content: {e}")
            context.metadata["error"] = str(e)
            context.metadata["word_count"] = word_count

    async def create_document(self,
                            request: DocumentRequest,
                            progress_callback: Optional[Callable] = None,
                            websocket = None) -> Dict[str, Any]:
        """Execute end-to-end document creation with detailed real-time communication.

        This is the main entry point for document creation. It orchestrates the entire
        process from service initialization to final document completion with real-time
        progress updates and intelligent narration.

        Args:
            request: Document creation request with all specifications
            progress_callback: Optional callback function for progress updates
            websocket: Optional WebSocket connection for real-time updates

        Returns:
            Dict containing:
                - success: Boolean indicating if creation was successful
                - document_id: Google Docs document ID (if successful)
                - document_url: URL to the created document (if successful)
                - title: Document title
                - word_count: Final word count
                - platform: Platform used for creation
                - topic: Document topic
                - formatting: Formatting style used
                - error: Error message (if unsuccessful)

        Raises:
            Exception: If critical errors occur during document creation
            
        Example:
            >>> request = DocumentRequest(
            ...     topic="Climate Change",
            ...     document_type=DocumentType.ESSAY,
            ...     word_count=1000
            ... )
            >>> result = await executor.create_document(request)
            >>> print(result['success'])
            True
        """
        try:
            # Initialize narration context
            self._narration_context = {
                "phase": "starting",
                "topic": request.topic,
                "document_type": request.document_type.value,
                "format": request.formatting.value,
                "progress": 0,
                "word_count": 0,
                "total_words_target": request.word_count or 1000,
                "current_section": "initialization"
            }

            # Phase 1: Dynamic announcement
            await self._narrate(progress_callback, websocket, context={
                "phase": "acknowledging_request",
                "topic": request.topic,
                "document_type": request.document_type.value,
                "format": request.formatting.value,
                "total_words_target": request.word_count or 1000
            })

            # Phase 2: Initialize services via async pipeline
            await self._narrate(progress_callback, websocket, context={
                "phase": "initializing_services",
                "progress": 10
            })

            # Route through async pipeline for service initialization
            init_result = await self.pipeline.process_async(
                text=f"Initialize document services for {request.topic}",
                metadata={
                    "request": request,
                    "stage": "service_init"
                }
            )

            if init_result.get("metadata", {}).get("error"):
                return {
                    "success": False,
                    "error": init_result["metadata"]["error"]
                }

            await self._narrate(progress_callback, websocket, context={
                "phase": "services_ready",
                "progress": 20
            })

            # Phase 3: Create Google Doc via async pipeline
            await self._narrate(progress_callback, websocket, context={
                "phase": "creating_document",
                "progress": 25,
                "current_section": f"Google Doc: {request.title}"
            })

            # Route through async pipeline for doc creation
            doc_result = await self.pipeline.process_async(
                text=f"Create Google Doc: {request.title}",
                metadata={
                    "request": request,
                    "stage": "doc_creation"
                }
            )

            doc_metadata = doc_result.get("metadata", {})
            if doc_metadata.get("error"):
                return {
                    "success": False,
                    "error": doc_metadata["error"]
                }

            document_id = doc_metadata.get("document_id")
            document_url = doc_metadata.get("document_url")

            if not document_id or not document_url:
                return {
                    "success": False,
                    "error": "Failed to create Google Doc"
                }

            await self._narrate(progress_callback, websocket, context={
                "phase": "document_created",
                "progress": 30
            })

            # Phase 4: Open in Chrome
            await self._narrate(progress_callback, websocket, context={
                "phase": "opening_browser",
                "progress": 35
            })

            await self._open_in_browser(document_url, request)
            await asyncio.sleep(1.5)

            await self._narrate(progress_callback, websocket, context={
                "phase": "browser_ready",
                "progress": 40
            })

            # Phase 5: Generate outline with detail
            await self._narrate(progress_callback, websocket, context={
                "phase": "analyzing_topic",
                "progress": 45,
                "current_section": "outline generation"
            })

            outline = await self._generate_outline(request)

            section_count = len(outline.get('sections', []))
            sections = outline.get('sections', [])
            section_names = ', '.join([s['name'] for s in sections])

            await self._narrate(progress_callback, websocket, context={
                "phase": "outline_complete",
                "progress": 50,
                "section_count": section_count,
                "sections": section_names
            })

            # Phase 6: Stream content with detailed progress
            await self._narrate(progress_callback, websocket, context={
                "phase": "starting_writing",
                "progress": 55,
                "current_section": sections[0]['name'] if sections else "introduction"
            })

            word_count = await self._stream_content(
                document_id, request, outline,
                progress_callback, websocket
            )

            # Phase 7: Completion with summary - CRITICAL announcements
            await self._narrate(progress_callback, websocket, context={
                "phase": "writing_complete",
                "progress": 95,
                "word_count": word_count,
                "topic": request.topic,
                "document_type": request.document_type.value
            })

            # Small delay for natural pacing between completion messages
            await asyncio.sleep(1.5)

            await self._narrate(progress_callback, websocket, context={
                "phase": "document_ready",
                "progress": 100,
                "word_count": word_count,
                "section_count": section_count,
                "topic": request.topic,
                "document_type": request.document_type.value
            })

            return {
                "success": True,
                "document_id": document_id,
                "document_url": document_url,
                "title": request.title,
                "word_count": word_count,
                "platform": request.platform.value,
                "topic": request.topic,
                "formatting": request.formatting.value
            }

        except Exception as e:
            logger.error(f"Error in document creation: {e}", exc_info=True)
            await self._narrate(progress_callback, websocket,
                f"I apologize, Sir. I encountered an error: {str(e)}")
            await self._narrate(progress_callback, websocket,
                "I'll attempt to recover and continue...")
            return {
                "success": False,
                "error": str(e)
            }

    async def _initialize_services(self, request: DocumentRequest) -> bool:
        """Initialize required services for document creation.
        
        Args:
            request: Document request containing service configuration
            
        Returns:
            bool: True if all services initialized successfully, False otherwise
        """
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

            # Initialize Intelligent Narrator with Claude
            from .intelligent_narrator import get_intelligent_narrator
            self._intelligent_narrator = get_intelligent_narrator(self._claude)
            await self._intelligent_narrator.initialize(
                topic=request.topic,
                doc_type=request.document_type.value,
                format_style=request.formatting.value,
                target_words=request.word_count or 1000,
                claude_client=self._claude
            )
            logger.info("[DOCUMENT WRITER] ✅ Intelligent Narrator initialized")

            # Initialize browser
            from ..automation.browser_controller import get_browser_controller
            self._browser = get_browser_controller(request.browser)

            return True

        except Exception as e:
            logger.error(f"Error initializing services: {e}")
            return False

    async def _create_google_doc(self, request: DocumentRequest) -> Optional[Dict[str, Any]]:
        """Create Google Doc via API with retry logic.
        
        Args:
            request: Document request containing title and retry configuration
            
        Returns:
            Optional[Dict[str, Any]]: Document info dict with ID and URL, or None if failed
        """
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
        """Open document in browser.
        
        Args:
            url: Document URL to open
            request: Document request containing browser configuration
            
        Returns:
            bool: True if successfully opened, False otherwise
        """
        try:
            success = await self._browser.navigate(url)
            if success:
                logger.info(f"Opened document in {request.browser}")
            return success
        except Exception as e:
            logger.error(f"Error opening in browser: {e}")
            return False

    async def _generate_outline(self, request: DocumentRequest) -> Dict[str, Any]:
        """Generate document outline using Claude AI.
        
        Args:
            request: Document request containing topic and requirements
            
        Returns:
            Dict[str, Any]: Outline structure with title and sections
        """
        article = "an" if request.document_type.value[0] in 'aeiou' else "a"
        outline_prompt = f"""Create a detailed outline for {article} {request.document_type.value} about "{request.topic}".

Target length: {request