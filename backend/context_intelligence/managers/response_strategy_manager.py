"""
Response Strategy Manager
=========================

Transforms vague responses into clear, actionable ones with specific details.

Good Response:
✅ "Space 3 has a TypeError on line 421 in test_vision.py.
   The error is: 'NoneType' object has no attribute 'get'"

Bad Response:
❌ "There's an error."
❌ "I see some text in a code editor."

Strategy:
- Extract specific details (file names, line numbers, error messages, space IDs)
- Provide context and actionable information
- Score response specificity
- Enhance vague responses with detailed analysis
"""

import asyncio
import re
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ResponseQuality(Enum):
    """Quality levels for responses"""
    VAGUE = "vague"  # "There's an error"
    BASIC = "basic"  # "I see an error in your code"
    SPECIFIC = "specific"  # "TypeError on line 421"
    ACTIONABLE = "actionable"  # "TypeError on line 421 in test_vision.py: fix by checking if obj is not None"
    EXCELLENT = "excellent"  # Full context + specific details + actionable steps


@dataclass
class ExtractedDetail:
    """A specific detail extracted from text"""
    type: str  # error, warning, file, line, space, app, etc.
    value: str
    context: Optional[str] = None
    location: Optional[str] = None
    confidence: float = 1.0


@dataclass
class ResponseAnalysis:
    """Analysis of a response's quality and details"""
    quality: ResponseQuality
    specificity_score: float  # 0.0 (vague) to 1.0 (excellent)
    details_found: List[ExtractedDetail]
    missing_details: List[str]
    suggestions: List[str]
    is_actionable: bool


@dataclass
class EnhancedResponse:
    """Enhanced response with improved clarity and actionability"""
    original_response: str
    enhanced_response: str
    analysis: ResponseAnalysis
    improvements: List[str]
    enhancement_time: float = 0.0


class DetailExtractor:
    """
    Extracts specific details from text and vision results.

    Extracts:
    - File names and paths
    - Line numbers
    - Error types and messages
    - Space/window IDs
    - Application names
    - Code snippets
    - URLs
    """

    def __init__(self):
        """Initialize detail extractor with dynamic patterns"""
        # Compile regex patterns for efficiency
        self.patterns = {
            # File paths and names
            'file_path': re.compile(r'(?:/[\w\-./]+\.[\w]+|[\w\-]+\.(?:py|js|ts|jsx|tsx|java|cpp|c|go|rs|rb|php|swift|kt|cs))'),
            'file_name': re.compile(r'([\w\-]+\.(?:py|js|ts|jsx|tsx|java|cpp|c|go|rs|rb|php|swift|kt|cs))\b'),

            # Line and column numbers
            'line_number': re.compile(r'line\s+(\d+)|:(\d+):|line\s*:\s*(\d+)'),
            'line_col': re.compile(r':(\d+):(\d+)'),

            # Error types and messages
            'error_type': re.compile(r'\b([A-Z][a-zA-Z]*(?:Error|Exception|Warning))\b'),
            'error_message': re.compile(r'(?:Error|Exception|Warning):\s*(.+?)(?:\n|$)'),

            # Space and window IDs
            'space_id': re.compile(r'[Ss]pace\s+(\d+)'),
            'window_id': re.compile(r'[Ww]indow\s+(\d+)'),

            # Application names (common patterns)
            'app_name': re.compile(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\.(app|exe)\b'),

            # Code context
            'function_name': re.compile(r'\bdef\s+(\w+)|function\s+(\w+)|fn\s+(\w+)|func\s+(\w+)'),
            'class_name': re.compile(r'\bclass\s+(\w+)'),
            'variable': re.compile(r"'(\w+)'\s+object|object\s+'(\w+)'"),

            # URLs
            'url': re.compile(r'https?://[\w\-._~:/?#[\]@!$&\'()*+,;=]+'),
        }

    def extract_details(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[ExtractedDetail]:
        """
        Extract all specific details from text.

        Args:
            text: Text to extract from
            context: Additional context (space_id, etc.)

        Returns:
            List of extracted details
        """
        details = []

        # Extract file paths
        for match in self.patterns['file_path'].finditer(text):
            details.append(ExtractedDetail(
                type='file_path',
                value=match.group(0),
                confidence=0.9
            ))

        # Extract file names (if not already in paths)
        file_paths = {d.value for d in details if d.type == 'file_path'}
        for match in self.patterns['file_name'].finditer(text):
            filename = match.group(0)
            if not any(filename in path for path in file_paths):
                details.append(ExtractedDetail(
                    type='file_name',
                    value=filename,
                    confidence=0.8
                ))

        # Extract line numbers
        for match in self.patterns['line_number'].finditer(text):
            line_num = next((g for g in match.groups() if g), None)
            if line_num:
                details.append(ExtractedDetail(
                    type='line_number',
                    value=line_num,
                    confidence=0.95
                ))

        # Extract line:col references
        for match in self.patterns['line_col'].finditer(text):
            details.append(ExtractedDetail(
                type='line_col',
                value=f"{match.group(1)}:{match.group(2)}",
                confidence=0.95
            ))

        # Extract error types
        for match in self.patterns['error_type'].finditer(text):
            details.append(ExtractedDetail(
                type='error_type',
                value=match.group(1),
                confidence=0.9
            ))

        # Extract error messages
        for match in self.patterns['error_message'].finditer(text):
            details.append(ExtractedDetail(
                type='error_message',
                value=match.group(1).strip(),
                confidence=0.85
            ))

        # Extract space IDs
        for match in self.patterns['space_id'].finditer(text):
            details.append(ExtractedDetail(
                type='space_id',
                value=match.group(1),
                confidence=1.0
            ))

        # Extract window IDs
        for match in self.patterns['window_id'].finditer(text):
            details.append(ExtractedDetail(
                type='window_id',
                value=match.group(1),
                confidence=1.0
            ))

        # Extract application names
        for match in self.patterns['app_name'].finditer(text):
            details.append(ExtractedDetail(
                type='app_name',
                value=match.group(0),
                confidence=0.8
            ))

        # Extract function names
        for match in self.patterns['function_name'].finditer(text):
            func_name = next((g for g in match.groups() if g), None)
            if func_name:
                details.append(ExtractedDetail(
                    type='function_name',
                    value=func_name,
                    confidence=0.7
                ))

        # Extract class names
        for match in self.patterns['class_name'].finditer(text):
            details.append(ExtractedDetail(
                type='class_name',
                value=match.group(1),
                confidence=0.7
            ))

        # Extract variable names from error messages
        for match in self.patterns['variable'].finditer(text):
            var_name = next((g for g in match.groups() if g), None)
            if var_name:
                details.append(ExtractedDetail(
                    type='variable',
                    value=var_name,
                    confidence=0.8
                ))

        # Extract URLs
        for match in self.patterns['url'].finditer(text):
            details.append(ExtractedDetail(
                type='url',
                value=match.group(0),
                confidence=0.95
            ))

        # Add context if provided
        if context:
            if 'space_id' in context:
                details.append(ExtractedDetail(
                    type='space_id',
                    value=str(context['space_id']),
                    confidence=1.0
                ))
            if 'window_id' in context:
                details.append(ExtractedDetail(
                    type='window_id',
                    value=str(context['window_id']),
                    confidence=1.0
                ))

        return details

    def extract_error_details(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract structured error information.

        Returns:
            Dictionary with error_type, message, file, line, etc.
        """
        error_info = {}

        # Extract error type
        error_type_match = self.patterns['error_type'].search(text)
        if error_type_match:
            error_info['error_type'] = error_type_match.group(1)

        # Extract error message
        error_msg_match = self.patterns['error_message'].search(text)
        if error_msg_match:
            error_info['message'] = error_msg_match.group(1).strip()

        # Extract file
        file_match = self.patterns['file_path'].search(text) or self.patterns['file_name'].search(text)
        if file_match:
            error_info['file'] = file_match.group(0)

        # Extract line number
        line_match = self.patterns['line_number'].search(text)
        if line_match:
            line_num = next((g for g in line_match.groups() if g), None)
            if line_num:
                error_info['line'] = int(line_num)

        return error_info if error_info else None


class SpecificityScorer:
    """
    Scores response specificity and quality.

    Scoring factors:
    - Presence of specific details (file, line, error type)
    - Length and context
    - Actionable information
    - Vague language detection
    """

    def __init__(self):
        """Initialize scorer with vague language patterns"""
        self.vague_patterns = [
            r'\bthere\'s\s+(?:a|an|some)\b',
            r'\bi\s+see\s+(?:a|an|some)\b',
            r'\blooks\s+like\b',
            r'\bseems\s+to\s+be\b',
            r'\bappears\s+to\b',
            r'\bmight\s+be\b',
            r'\bcould\s+be\b',
            r'\bprobably\b',
            r'\bmaybe\b',
            r'\bsomething\b',
            r'\bsomewhere\b',
            r'\ban?\s+(?:error|issue|problem)\b(?!\s+(?:in|on|at|with))',  # Vague if not followed by location
        ]
        self.vague_regex = re.compile('|'.join(self.vague_patterns), re.IGNORECASE)

    def score_response(
        self, response: str, details: List[ExtractedDetail]
    ) -> Tuple[float, ResponseQuality]:
        """
        Score response specificity.

        Args:
            response: The response text
            details: Extracted details from response

        Returns:
            Tuple of (score, quality_level)
        """
        score = 0.0

        # Factor 1: Detail density (40% of score)
        detail_types = set(d.type for d in details)
        detail_score = min(len(detail_types) / 5.0, 1.0) * 0.4  # Max 5 different detail types
        score += detail_score

        # Factor 2: Specific detail presence (30% of score)
        important_details = {'file_path', 'file_name', 'line_number', 'error_type', 'space_id'}
        found_important = important_details & detail_types
        specific_score = len(found_important) / len(important_details) * 0.3
        score += specific_score

        # Factor 3: Vague language penalty (20% of score)
        vague_matches = self.vague_regex.findall(response)
        vague_penalty = min(len(vague_matches) * 0.05, 0.2)  # Max 20% penalty
        score += (0.2 - vague_penalty)

        # Factor 4: Length and context (10% of score)
        # Longer responses with context are generally better
        word_count = len(response.split())
        length_score = min(word_count / 50.0, 1.0) * 0.1  # Optimal around 50 words
        score += length_score

        # Determine quality level
        if score >= 0.9:
            quality = ResponseQuality.EXCELLENT
        elif score >= 0.7:
            quality = ResponseQuality.ACTIONABLE
        elif score >= 0.5:
            quality = ResponseQuality.SPECIFIC
        elif score >= 0.3:
            quality = ResponseQuality.BASIC
        else:
            quality = ResponseQuality.VAGUE

        return score, quality

    def analyze_response(
        self, response: str, details: List[ExtractedDetail]
    ) -> ResponseAnalysis:
        """
        Full analysis of response quality.

        Args:
            response: The response text
            details: Extracted details

        Returns:
            ResponseAnalysis with quality assessment
        """
        score, quality = self.score_response(response, details)

        # Determine missing details
        detail_types = set(d.type for d in details)
        expected_details = {'space_id', 'file_name', 'line_number', 'error_type'}
        missing = []

        # Check if response is about an error
        if 'error' in response.lower() or 'exception' in response.lower():
            if 'error_type' not in detail_types:
                missing.append('error_type')
            if 'file_name' not in detail_types and 'file_path' not in detail_types:
                missing.append('file_name')
            if 'line_number' not in detail_types:
                missing.append('line_number')

        # Check if response mentions spaces
        if 'space' in response.lower() and 'space_id' not in detail_types:
            missing.append('space_id')

        # Generate suggestions
        suggestions = []
        if quality in [ResponseQuality.VAGUE, ResponseQuality.BASIC]:
            suggestions.append("Add specific location (file, line number)")
            suggestions.append("Include exact error type and message")
            suggestions.append("Specify which space/window")

        if 'error_type' in missing:
            suggestions.append("Include the specific error type (e.g., TypeError, ValueError)")

        if 'line_number' in missing:
            suggestions.append("Add the line number where the issue occurs")

        # Check if actionable
        is_actionable = (
            quality in [ResponseQuality.ACTIONABLE, ResponseQuality.EXCELLENT] and
            len(details) >= 3
        )

        return ResponseAnalysis(
            quality=quality,
            specificity_score=score,
            details_found=details,
            missing_details=missing,
            suggestions=suggestions,
            is_actionable=is_actionable
        )


class ActionableFormatter:
    """
    Formats responses to be clear and actionable.

    Transforms:
    - "There's an error" → "Space 3 has a TypeError on line 421 in test_vision.py"
    - "I see text" → "Space 2 shows code in editor.py with function 'process_command'"
    """

    def __init__(self, detail_extractor: DetailExtractor):
        """Initialize formatter"""
        self.extractor = detail_extractor

    def format_response(
        self,
        vague_response: str,
        details: List[ExtractedDetail],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format a vague response into a clear, actionable one.

        Args:
            vague_response: Original vague response
            details: Extracted details
            context: Additional context

        Returns:
            Enhanced, actionable response
        """
        # Group details by type
        detail_map = {}
        for detail in details:
            if detail.type not in detail_map:
                detail_map[detail.type] = []
            detail_map[detail.type].append(detail)

        # Build structured response
        parts = []

        # Start with space/window context
        if 'space_id' in detail_map:
            space_id = detail_map['space_id'][0].value
            parts.append(f"Space {space_id}")
        elif context and 'space_id' in context:
            parts.append(f"Space {context['space_id']}")

        # Add error information if present
        if 'error_type' in detail_map:
            error_type = detail_map['error_type'][0].value

            # Build error description
            error_parts = [f"has a {error_type}"]

            if 'line_number' in detail_map:
                line_num = detail_map['line_number'][0].value
                error_parts.append(f"on line {line_num}")

            if 'file_name' in detail_map or 'file_path' in detail_map:
                file_detail = (detail_map.get('file_path', []) or detail_map.get('file_name', []))[0]
                error_parts.append(f"in {file_detail.value}")

            parts.append(' '.join(error_parts))

            # Add error message if available
            if 'error_message' in detail_map:
                error_msg = detail_map['error_message'][0].value
                parts.append(f"\nThe error is: {error_msg}")

        # Add file/code context if not error-related
        elif 'file_name' in detail_map or 'file_path' in detail_map:
            file_detail = (detail_map.get('file_path', []) or detail_map.get('file_name', []))[0]

            code_parts = [f"shows {file_detail.value}"]

            if 'function_name' in detail_map:
                func_name = detail_map['function_name'][0].value
                code_parts.append(f"with function '{func_name}'")

            if 'class_name' in detail_map:
                class_name = detail_map['class_name'][0].value
                code_parts.append(f"containing class '{class_name}'")

            parts.append(' '.join(code_parts))

        # Add app context
        if 'app_name' in detail_map:
            app_name = detail_map['app_name'][0].value
            parts.append(f"(in {app_name})")

        # If we built a structured response, use it; otherwise use original
        if len(parts) > 1:  # More than just space ID
            return ' '.join(parts)
        else:
            # Couldn't improve, return original
            return vague_response

    def format_error_response(self, error_info: Dict[str, Any]) -> str:
        """
        Format structured error information into a clear response.

        Args:
            error_info: Dictionary with error details

        Returns:
            Formatted error response
        """
        parts = []

        # Error type
        if 'error_type' in error_info:
            parts.append(f"{error_info['error_type']}")

        # Location
        location_parts = []
        if 'line' in error_info:
            location_parts.append(f"line {error_info['line']}")
        if 'file' in error_info:
            location_parts.append(f"in {error_info['file']}")

        if location_parts:
            parts.append(f"on {' '.join(location_parts)}")

        # Message
        if 'message' in error_info:
            parts.append(f"\nThe error is: {error_info['message']}")

        return ' '.join(parts) if parts else "Error detected"


class ResponseEnhancer:
    """
    Enhances vague responses using vision analysis.

    Uses Claude Vision to extract more details when initial response is vague.
    """

    def __init__(self, vision_client: Optional[Any] = None):
        """
        Initialize enhancer.

        Args:
            vision_client: Optional Claude Vision client for deep analysis
        """
        self.vision_client = vision_client

    async def enhance_response(
        self,
        vague_response: str,
        image_path: Optional[str] = None,
        ocr_text: Optional[str] = None
    ) -> str:
        """
        Enhance a vague response with more specific details.

        Args:
            vague_response: The vague response to enhance
            image_path: Path to screenshot for vision analysis
            ocr_text: OCR text if available

        Returns:
            Enhanced response
        """
        if not self.vision_client or not image_path:
            return vague_response

        try:
            # Use Claude Vision for detailed analysis
            prompt = f"""
Analyze this screenshot and provide a SPECIFIC, ACTIONABLE description.

Original vague response: "{vague_response}"

Please provide:
1. Exact file names if visible
2. Specific line numbers if visible
3. Exact error types and messages
4. Application or window names
5. Any other specific details

Format: Clear, specific, actionable response (not vague).
"""

            # Call Claude Vision (implementation depends on your vision client)
            # This is a placeholder - actual implementation would use your vision client
            enhanced = vague_response  # Fallback

            return enhanced

        except Exception as e:
            logger.warning(f"Vision enhancement failed: {e}")
            return vague_response


class ResponseStrategyManager:
    """
    Main manager for response quality strategies.

    Transforms vague responses into clear, actionable ones.
    """

    def __init__(
        self,
        vision_client: Optional[Any] = None,
        min_quality: ResponseQuality = ResponseQuality.SPECIFIC
    ):
        """
        Initialize Response Strategy Manager.

        Args:
            vision_client: Optional vision client for enhancement
            min_quality: Minimum acceptable response quality
        """
        self.detail_extractor = DetailExtractor()
        self.specificity_scorer = SpecificityScorer()
        self.actionable_formatter = ActionableFormatter(self.detail_extractor)
        self.response_enhancer = ResponseEnhancer(vision_client)
        self.min_quality = min_quality

    async def improve_response(
        self,
        response: str,
        context: Optional[Dict[str, Any]] = None,
        image_path: Optional[str] = None,
        ocr_text: Optional[str] = None
    ) -> EnhancedResponse:
        """
        Improve response quality to meet minimum standards.

        Args:
            response: Original response
            context: Additional context (space_id, etc.)
            image_path: Optional screenshot for vision enhancement
            ocr_text: Optional OCR text

        Returns:
            EnhancedResponse with improvements
        """
        import time
        start_time = time.time()

        # Step 1: Extract details from response and context
        details = self.detail_extractor.extract_details(response, context)

        # Add details from OCR text if available
        if ocr_text:
            ocr_details = self.detail_extractor.extract_details(ocr_text, context)
            details.extend(ocr_details)

        # Step 2: Analyze response quality
        analysis = self.specificity_scorer.analyze_response(response, details)

        # Step 3: Improve if below minimum quality
        improved_response = response
        improvements = []

        if analysis.quality.value < self.min_quality.value:
            # Try formatting with extracted details
            formatted = self.actionable_formatter.format_response(
                response, details, context
            )

            if formatted != response:
                improved_response = formatted
                improvements.append("Added specific details from context")

            # If still not good enough and vision available, enhance further
            if analysis.quality in [ResponseQuality.VAGUE, ResponseQuality.BASIC] and image_path:
                enhanced = await self.response_enhancer.enhance_response(
                    improved_response, image_path, ocr_text
                )
                if enhanced != improved_response:
                    improved_response = enhanced
                    improvements.append("Enhanced with vision analysis")

        execution_time = time.time() - start_time

        return EnhancedResponse(
            original_response=response,
            enhanced_response=improved_response,
            analysis=analysis,
            improvements=improvements,
            enhancement_time=execution_time
        )

    def extract_error_details(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract structured error information from text"""
        return self.detail_extractor.extract_error_details(text)

    def format_error_response(self, error_info: Dict[str, Any]) -> str:
        """Format error information into clear response"""
        return self.actionable_formatter.format_error_response(error_info)


# Global instance
_response_strategy_manager: Optional[ResponseStrategyManager] = None


def get_response_strategy_manager() -> Optional[ResponseStrategyManager]:
    """Get the global ResponseStrategyManager instance"""
    return _response_strategy_manager


def initialize_response_strategy_manager(
    vision_client: Optional[Any] = None,
    min_quality: ResponseQuality = ResponseQuality.SPECIFIC
) -> ResponseStrategyManager:
    """
    Initialize the global ResponseStrategyManager instance.

    Args:
        vision_client: Optional vision client for enhancement
        min_quality: Minimum acceptable response quality

    Returns:
        ResponseStrategyManager instance
    """
    global _response_strategy_manager

    _response_strategy_manager = ResponseStrategyManager(
        vision_client=vision_client,
        min_quality=min_quality
    )

    return _response_strategy_manager
