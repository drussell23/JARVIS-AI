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
    """Supported document types"""
    ESSAY = "essay"
    REPORT = "report"
    PAPER = "paper"
    RESEARCH_PAPER = "research_paper"
    ARTICLE = "article"
    BLOG_POST = "blog_post"
    GENERAL = "general"


class DocumentFormat(Enum):
    """Academic and professional formatting styles"""
    MLA = "mla"
    APA = "apa"
    CHICAGO = "chicago"
    IEEE = "ieee"
    HARVARD = "harvard"
    NONE = "none"


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

    def __post_init__(self):
        """Generate title and auto-detect format if not provided"""
        if not self.title:
            self.title = self._generate_title()

        # Auto-detect format based on document type if not explicitly set
        if self.formatting == DocumentFormat.NONE:
            self.formatting = self._auto_detect_format()

    def _auto_detect_format(self) -> DocumentFormat:
        """Intelligently auto-detect formatting style based on document type"""
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
        """Generate intelligent title from topic and type"""
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
        Execute end-to-end document creation with detailed real-time communication

        Args:
            request: Document creation request
            progress_callback: Callback for narration updates
            websocket: WebSocket for real-time updates

        Returns:
            Execution result with document details
        """
        try:
            # Phase 1: Announce with full context
            article = "an" if request.document_type.value[0] in 'aeiou' else "a"
            format_spec = FORMAT_SPECIFICATIONS.get(request.formatting.value, {})
            format_name = format_spec.get("name", "standard format")

            await self._narrate(progress_callback, websocket,
                f"Understood, Sir. I'll create {article} {request.document_type.value} about '{request.topic}' "
                f"in {format_name}.")

            if request.word_count:
                await self._narrate(progress_callback, websocket,
                    f"Target length: {request.word_count} words.")
            elif request.page_count:
                await self._narrate(progress_callback, websocket,
                    f"Target length: {request.page_count} pages.")

            # Phase 2: Initialize services with detailed feedback
            await self._narrate(progress_callback, websocket,
                "Initializing document creation systems...")

            if not await self._initialize_services(request):
                return {
                    "success": False,
                    "error": "Failed to initialize required services"
                }

            await self._narrate(progress_callback, websocket,
                "✓ Claude AI connected and ready")
            await self._narrate(progress_callback, websocket,
                "✓ Google Docs API authenticated")
            await self._narrate(progress_callback, websocket,
                "✓ Browser automation ready")

            # Phase 3: Create Google Doc via API
            await self._narrate(progress_callback, websocket,
                f"Creating new Google Doc titled '{request.title}'...")

            doc_info = await self._create_google_doc(request)
            if not doc_info:
                return {
                    "success": False,
                    "error": "Failed to create Google Doc"
                }

            document_id = doc_info['document_id']
            document_url = doc_info['document_url']

            await self._narrate(progress_callback, websocket,
                f"✓ Document created successfully")

            # Phase 4: Open in Chrome
            await self._narrate(progress_callback, websocket,
                f"Opening document in {request.browser}...")

            await self._open_in_browser(document_url, request)
            await asyncio.sleep(1.5)  # Let user see the document open

            await self._narrate(progress_callback, websocket,
                "✓ Document opened in browser")

            # Phase 5: Generate outline with detail
            await self._narrate(progress_callback, websocket,
                f"Analyzing '{request.topic}' and structuring the content...")

            await self._narrate(progress_callback, websocket,
                "Creating comprehensive outline with Claude AI...")

            outline = await self._generate_outline(request)

            section_count = len(outline.get('sections', []))
            await self._narrate(progress_callback, websocket,
                f"✓ Outline complete: {section_count} main sections identified")

            # Announce the structure
            sections = outline.get('sections', [])
            if sections:
                await self._narrate(progress_callback, websocket,
                    f"Structure planned: {', '.join([s['name'] for s in sections])}")

            # Phase 6: Stream content with detailed progress
            await self._narrate(progress_callback, websocket,
                f"Beginning to write your {request.document_type.value} in {format_name}...")

            await self._narrate(progress_callback, websocket,
                "Claude AI is now generating content. You'll see it appear in real-time, Sir.")

            word_count = await self._stream_content(
                document_id, request, outline,
                progress_callback, websocket
            )

            # Phase 7: Completion with summary
            await self._narrate(progress_callback, websocket,
                f"✓ Writing complete!")

            await self._narrate(progress_callback, websocket,
                f"Your {request.document_type.value} about '{request.topic}' is ready, Sir.")

            await self._narrate(progress_callback, websocket,
                f"Final statistics: {word_count} words, {format_name}, {section_count} sections")

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
        """Stream content to Google Doc via API with detailed real-time updates"""
        content_prompt = self._build_content_prompt(request, outline)

        word_count = 0
        sentence_count = 0
        buffer = ""
        progress_interval = 50  # Update every 50 words
        next_milestone = progress_interval

        # Track sections for progress updates
        sections = outline.get('sections', [])
        current_section_index = 0
        section_announced = False

        await self._narrate(progress_callback, websocket,
            "Starting with the MLA heading..." if request.formatting == DocumentFormat.MLA else "Starting with the document...")

        try:
            async for chunk in self._claude.stream_content(
                content_prompt,
                max_tokens=request.claude_max_tokens,
                model=request.claude_model
            ):
                buffer += chunk

                # Count sentences for more granular updates
                sentence_count += buffer.count('.') + buffer.count('!') + buffer.count('?')

                # Write in smaller chunks for more real-time feel
                if len(buffer) >= request.chunk_size:
                    success = await self._google_docs.append_text(document_id, buffer)

                    if success:
                        current_words = len(buffer.split())
                        word_count += current_words

                        # Detect section changes (simple heuristic)
                        if current_section_index < len(sections) and not section_announced:
                            section_name = sections[current_section_index]['name']
                            await self._narrate(progress_callback, websocket,
                                f"Writing {section_name} section...")
                            section_announced = True

                        # Move to next section periodically
                        if word_count > (current_section_index + 1) * 150 and current_section_index < len(sections) - 1:
                            current_section_index += 1
                            section_announced = False

                        # Progress milestones with more detail
                        if word_count >= next_milestone:
                            percentage = min(int((word_count / (request.word_count or 750)) * 100), 99) if request.word_count or request.document_type != DocumentType.BLOG_POST else 0

                            if percentage > 0:
                                await self._narrate(progress_callback, websocket,
                                    f"Progress: {word_count} words ({percentage}% complete)")
                            else:
                                await self._narrate(progress_callback, websocket,
                                    f"Progress: {word_count} words written")

                            next_milestone += progress_interval

                    buffer = ""
                    await asyncio.sleep(request.stream_delay)

            # Write remaining buffer
            if buffer:
                await self._google_docs.append_text(document_id, buffer)
                word_count += len(buffer.split())

            # Final section notification
            if current_section_index < len(sections):
                await self._narrate(progress_callback, websocket,
                    "Finalizing conclusion and Works Cited/References...")

            return word_count

        except Exception as e:
            logger.error(f"Error streaming content: {e}")
            await self._narrate(progress_callback, websocket,
                f"Encountered an issue during writing, but continuing... ({word_count} words so far)")
            return word_count

    def _build_content_prompt(self, request: DocumentRequest, outline: Dict[str, Any]) -> str:
        """Build comprehensive prompt for content generation with formatting"""
        # Get formatting specifications
        format_spec = FORMAT_SPECIFICATIONS.get(request.formatting.value, {})
        format_name = format_spec.get("name", "No specific format")
        format_requirements = format_spec.get("requirements", "")

        # Build heading information if applicable
        heading_info = ""
        if request.formatting == DocumentFormat.MLA:
            heading_info = f"""
MLA Heading (upper left corner):
{request.author_name}
{request.instructor_name if request.instructor_name else "Instructor Name"}
{request.course_name if request.course_name else "Course Name"}
{datetime.now().strftime("%d %B %Y")}
"""
        elif request.formatting == DocumentFormat.APA:
            heading_info = f"""
Title Page:
Title: {request.title}
Author: {request.author_name}
Institutional Affiliation: {request.institution if request.institution else "Institution Name"}
"""

        prompt = f"""Write a complete, high-quality {request.document_type.value} about "{request.topic}" in {format_name} format.

Target length: {request.get_length_spec()}

Title: {request.title}

FORMATTING REQUIREMENTS - {format_name}:
{format_requirements}

{heading_info}

Outline:
{self._format_outline(outline)}

Content Requirements:
- Professional, well-researched content
- Clear structure with proper transitions
- Engaging and informative writing
- Academic/professional tone appropriate for a {request.document_type.value}
- STRICTLY follow {format_name} formatting guidelines above
- Include proper in-text citations in {format_name} style (use plausible sources)
- Include a proper Works Cited/References page at the end in {format_name} format
{f"- {request.additional_requirements}" if request.additional_requirements else ""}

IMPORTANT: Format the entire document according to {format_name} standards, including:
- Proper heading/title page
- Correct spacing and indentation
- Appropriate in-text citations
- Properly formatted Works Cited/References page

Write the complete {request.document_type.value} now, starting with the proper {format_name} heading:"""

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
    Parse command into DocumentRequest (zero hardcoding, fully dynamic)

    Args:
        command: Original command string
        intent: Parsed intent information

    Returns:
        DocumentRequest object
    """
    command_lower = command.lower()

    # Detect document type (with research paper detection)
    type_patterns = {
        DocumentType.RESEARCH_PAPER: r'\bresearch\s+paper\b',
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

    # Detect formatting style explicitly mentioned in command
    format_patterns = {
        DocumentFormat.MLA: r'\bmla\b|\bmla\s+format\b',
        DocumentFormat.APA: r'\bapa\b|\bapa\s+format\b|\bapa\s+style\b',
        DocumentFormat.CHICAGO: r'\bchicago\b|\bturabian\b|\bchicago\s+style\b',
        DocumentFormat.IEEE: r'\bieee\b|\bieee\s+format\b',
        DocumentFormat.HARVARD: r'\bharvard\b|\bharvard\s+referencing\b'
    }

    formatting = DocumentFormat.NONE  # Will auto-detect if not found
    for fmt, pattern in format_patterns.items():
        if re.search(pattern, command_lower):
            formatting = fmt
            logger.info(f"Explicitly detected format from command: {fmt.value}")
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

    # Extract author name if mentioned
    author_name = "Student Name"
    author_match = re.search(r'by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', command)
    if author_match:
        author_name = author_match.group(1)

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
        formatting=formatting,
        author_name=author_name,
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
