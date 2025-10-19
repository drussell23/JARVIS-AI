"""
Confidence Manager
==================

Expresses uncertainty in responses based on data quality and confidence scores.

High Confidence (0.8-1.0):
✅ "Space 3 has 15 visible lines of Python code."

Medium Confidence (0.5-0.8):
⚠️ "Space 3 appears to have a syntax error, though the text is partially obscured."

Low Confidence (0.0-0.5):
❓ "Space 3 may contain a terminal, but the resolution is too low to confirm."

Strategy:
- Calculate confidence from multiple sources (OCR, image quality, data completeness)
- Classify into HIGH/MEDIUM/LOW levels
- Format responses with appropriate hedging language
- Provide visual indicators (✅ ⚠️ ❓)
- Include reasoning when confidence is not high
"""

import re
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence levels for responses"""
    HIGH = "high"  # 0.8-1.0: Strong confidence
    MEDIUM = "medium"  # 0.5-0.8: Moderate confidence
    LOW = "low"  # 0.0-0.5: Weak confidence


@dataclass
class ConfidenceScore:
    """Confidence score with breakdown"""
    overall: float  # 0.0-1.0
    level: ConfidenceLevel
    sources: Dict[str, float]  # Individual confidence sources
    reasoning: List[str]  # Why this confidence level
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfidenceFormattedResponse:
    """Response formatted with confidence level"""
    original_response: str
    formatted_response: str
    confidence_score: ConfidenceScore
    visual_indicator: str  # ✅ ⚠️ ❓
    hedging_applied: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConfidenceCalculator:
    """
    Calculates confidence scores from multiple sources.

    Sources:
    - OCR confidence
    - Image quality metrics
    - Text clarity/readability
    - Data completeness
    - Pattern matching confidence
    - Verification against multiple sources
    """

    def __init__(self):
        """Initialize confidence calculator"""
        self.weights = {
            'ocr_confidence': 0.35,  # 35% weight
            'image_quality': 0.25,   # 25% weight
            'data_completeness': 0.20,  # 20% weight
            'text_clarity': 0.15,    # 15% weight
            'verification': 0.05     # 5% weight
        }

    def calculate_confidence(
        self,
        ocr_confidence: Optional[float] = None,
        image_quality: Optional[float] = None,
        text_clarity: Optional[float] = None,
        data_completeness: Optional[float] = None,
        verification_score: Optional[float] = None,
        additional_sources: Optional[Dict[str, float]] = None
    ) -> ConfidenceScore:
        """
        Calculate overall confidence score.

        Args:
            ocr_confidence: OCR quality score (0.0-1.0)
            image_quality: Image quality score (0.0-1.0)
            text_clarity: Text readability score (0.0-1.0)
            data_completeness: Completeness score (0.0-1.0)
            verification_score: Cross-verification score (0.0-1.0)
            additional_sources: Additional confidence sources

        Returns:
            ConfidenceScore with overall score and breakdown
        """
        sources = {}
        reasoning = []
        total_score = 0.0
        total_weight = 0.0

        # OCR confidence
        if ocr_confidence is not None:
            sources['ocr_confidence'] = ocr_confidence
            total_score += ocr_confidence * self.weights['ocr_confidence']
            total_weight += self.weights['ocr_confidence']

            if ocr_confidence < 0.5:
                reasoning.append(f"Low OCR confidence ({ocr_confidence:.1%})")
            elif ocr_confidence < 0.8:
                reasoning.append(f"Moderate OCR confidence ({ocr_confidence:.1%})")

        # Image quality
        if image_quality is not None:
            sources['image_quality'] = image_quality
            total_score += image_quality * self.weights['image_quality']
            total_weight += self.weights['image_quality']

            if image_quality < 0.5:
                reasoning.append("Low image quality")
            elif image_quality < 0.8:
                reasoning.append("Moderate image quality")

        # Text clarity
        if text_clarity is not None:
            sources['text_clarity'] = text_clarity
            total_score += text_clarity * self.weights['text_clarity']
            total_weight += self.weights['text_clarity']

            if text_clarity < 0.5:
                reasoning.append("Text is unclear or obscured")
            elif text_clarity < 0.8:
                reasoning.append("Some text is partially obscured")

        # Data completeness
        if data_completeness is not None:
            sources['data_completeness'] = data_completeness
            total_score += data_completeness * self.weights['data_completeness']
            total_weight += self.weights['data_completeness']

            if data_completeness < 0.5:
                reasoning.append("Incomplete data")
            elif data_completeness < 0.8:
                reasoning.append("Some information missing")

        # Verification
        if verification_score is not None:
            sources['verification'] = verification_score
            total_score += verification_score * self.weights['verification']
            total_weight += self.weights['verification']

        # Additional sources (weighted equally)
        if additional_sources:
            additional_weight = 0.1 / len(additional_sources)
            for name, score in additional_sources.items():
                sources[name] = score
                total_score += score * additional_weight
                total_weight += additional_weight

        # Normalize score
        overall = total_score / total_weight if total_weight > 0 else 0.5

        # Determine confidence level
        level = self._classify_confidence(overall)

        return ConfidenceScore(
            overall=overall,
            level=level,
            sources=sources,
            reasoning=reasoning
        )

    def _classify_confidence(self, score: float) -> ConfidenceLevel:
        """Classify confidence level from score"""
        if score >= 0.8:
            return ConfidenceLevel.HIGH
        elif score >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def calculate_from_ocr_result(self, ocr_result: Any) -> ConfidenceScore:
        """Calculate confidence from OCR result object"""
        ocr_confidence = None
        text_clarity = None

        # Extract OCR confidence if available
        if hasattr(ocr_result, 'confidence'):
            ocr_confidence = ocr_result.confidence

        # Estimate text clarity from text length and structure
        if hasattr(ocr_result, 'text') and ocr_result.text:
            text_clarity = self._estimate_text_clarity(ocr_result.text)

        # Calculate data completeness (did we get text?)
        data_completeness = 1.0 if (hasattr(ocr_result, 'text') and ocr_result.text) else 0.0

        return self.calculate_confidence(
            ocr_confidence=ocr_confidence,
            text_clarity=text_clarity,
            data_completeness=data_completeness
        )

    def _estimate_text_clarity(self, text: str) -> float:
        """Estimate text clarity from content"""
        if not text:
            return 0.0

        clarity_score = 1.0

        # Reduce score for excessive special characters (corrupted text)
        special_char_ratio = len(re.findall(r'[^\w\s.,;:!?()\[\]{}"\'`-]', text)) / max(len(text), 1)
        if special_char_ratio > 0.3:
            clarity_score -= 0.4

        # Reduce score for very short text (might be incomplete)
        if len(text) < 10:
            clarity_score -= 0.2

        # Reduce score for excessive repetition
        words = text.split()
        if len(words) > 1:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                clarity_score -= 0.3

        return max(0.0, min(1.0, clarity_score))


class UncertaintyFormatter:
    """
    Formats responses with appropriate hedging language based on confidence level.

    Hedging strategies:
    - HIGH: Definitive statements
    - MEDIUM: Qualified statements (appears, seems, likely)
    - LOW: Uncertain statements (may, might, possibly)
    """

    def __init__(self):
        """Initialize uncertainty formatter"""
        # Hedging words by confidence level
        self.hedging = {
            ConfidenceLevel.MEDIUM: [
                'appears to', 'seems to', 'likely', 'probably',
                'suggests', 'indicates', 'though'
            ],
            ConfidenceLevel.LOW: [
                'may', 'might', 'possibly', 'could',
                'uncertain', 'unclear', 'but'
            ]
        }

    def format_with_confidence(
        self,
        response: str,
        confidence_score: ConfidenceScore,
        include_reasoning: bool = True
    ) -> str:
        """
        Format response with appropriate hedging language.

        Args:
            response: Original response
            confidence_score: Confidence score
            include_reasoning: Include reasoning when confidence is not high

        Returns:
            Formatted response with hedging
        """
        level = confidence_score.level

        # HIGH confidence: no changes needed
        if level == ConfidenceLevel.HIGH:
            return response

        # MEDIUM/LOW confidence: add hedging
        hedged = self._add_hedging(response, level)

        # Add reasoning if requested and confidence is not high
        if include_reasoning and confidence_score.reasoning:
            reasoning_text = ", ".join(confidence_score.reasoning)
            # Add reasoning as qualifier
            if level == ConfidenceLevel.MEDIUM:
                hedged = f"{hedged}, though {reasoning_text.lower()}"
            else:
                hedged = f"{hedged}, but {reasoning_text.lower()}"

        return hedged

    def _add_hedging(self, response: str, level: ConfidenceLevel) -> str:
        """Add hedging language to response"""
        if level == ConfidenceLevel.HIGH:
            return response

        # Detect if response already has hedging
        response_lower = response.lower()
        has_hedging = any(
            hedge in response_lower
            for hedges in self.hedging.values()
            for hedge in hedges
        )

        if has_hedging:
            return response  # Already hedged

        # Add appropriate hedging based on level
        if level == ConfidenceLevel.MEDIUM:
            # Replace definitive verbs with qualified ones
            hedged = self._replace_definitive_verbs(response, 'appears to')
        else:  # LOW
            hedged = self._replace_definitive_verbs(response, 'may')

        return hedged

    def _replace_definitive_verbs(self, response: str, hedge_word: str) -> str:
        """Replace definitive verbs with hedged versions"""
        # Common definitive patterns
        patterns = [
            (r'\b(has|have)\b', f'{hedge_word} have'),
            (r'\b(is|are)\b', f'{hedge_word} be'),
            (r'\b(shows|show)\b', f'{hedge_word} show'),
            (r'\b(contains|contain)\b', f'{hedge_word} contain'),
        ]

        # Try to replace first definitive verb
        for pattern, replacement in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                # Replace only the first occurrence
                hedged = re.sub(pattern, replacement, response, count=1, flags=re.IGNORECASE)
                return hedged

        # If no patterns matched, prepend hedge to beginning
        return f"{hedge_word.capitalize()} {response.lower()}"


class ConfidenceVisualIndicator:
    """
    Provides visual indicators for confidence levels.

    Indicators:
    - HIGH: ✅ (checkmark)
    - MEDIUM: ⚠️ (warning)
    - LOW: ❓ (question mark)
    """

    INDICATORS = {
        ConfidenceLevel.HIGH: "✅",
        ConfidenceLevel.MEDIUM: "⚠️",
        ConfidenceLevel.LOW: "❓"
    }

    @classmethod
    def get_indicator(cls, level: ConfidenceLevel) -> str:
        """Get visual indicator for confidence level"""
        return cls.INDICATORS.get(level, "")

    @classmethod
    def format_with_indicator(cls, response: str, level: ConfidenceLevel) -> str:
        """Format response with visual indicator"""
        indicator = cls.get_indicator(level)
        return f"{indicator} {response}"


class ConfidenceManager:
    """
    Main manager for confidence-based response formatting.

    Calculates confidence, classifies level, and formats responses with
    appropriate hedging language and visual indicators.
    """

    def __init__(
        self,
        include_visual_indicators: bool = True,
        include_reasoning: bool = True,
        min_confidence_for_high: float = 0.8,
        min_confidence_for_medium: float = 0.5
    ):
        """
        Initialize Confidence Manager.

        Args:
            include_visual_indicators: Add ✅ ⚠️ ❓ indicators
            include_reasoning: Include reasoning for non-high confidence
            min_confidence_for_high: Minimum score for HIGH level
            min_confidence_for_medium: Minimum score for MEDIUM level
        """
        self.include_visual_indicators = include_visual_indicators
        self.include_reasoning = include_reasoning
        self.min_confidence_for_high = min_confidence_for_high
        self.min_confidence_for_medium = min_confidence_for_medium

        self.calculator = ConfidenceCalculator()
        self.formatter = UncertaintyFormatter()

    def format_with_confidence(
        self,
        response: str,
        confidence_score: Optional[ConfidenceScore] = None,
        ocr_confidence: Optional[float] = None,
        image_quality: Optional[float] = None,
        data_completeness: Optional[float] = None,
        **kwargs
    ) -> ConfidenceFormattedResponse:
        """
        Format response with confidence level.

        Args:
            response: Original response
            confidence_score: Pre-calculated confidence score (optional)
            ocr_confidence: OCR quality score
            image_quality: Image quality score
            data_completeness: Data completeness score
            **kwargs: Additional confidence sources

        Returns:
            ConfidenceFormattedResponse with formatted text
        """
        # Calculate confidence if not provided
        if confidence_score is None:
            confidence_score = self.calculator.calculate_confidence(
                ocr_confidence=ocr_confidence,
                image_quality=image_quality,
                data_completeness=data_completeness,
                **kwargs
            )

        # Format with hedging language
        formatted = self.formatter.format_with_confidence(
            response,
            confidence_score,
            include_reasoning=self.include_reasoning
        )

        # Add visual indicator if enabled
        visual_indicator = ConfidenceVisualIndicator.get_indicator(confidence_score.level)
        if self.include_visual_indicators:
            formatted = ConfidenceVisualIndicator.format_with_indicator(
                formatted, confidence_score.level
            )

        hedging_applied = confidence_score.level != ConfidenceLevel.HIGH

        return ConfidenceFormattedResponse(
            original_response=response,
            formatted_response=formatted,
            confidence_score=confidence_score,
            visual_indicator=visual_indicator,
            hedging_applied=hedging_applied,
            metadata={
                'confidence_sources_count': len(confidence_score.sources),
                'reasoning_count': len(confidence_score.reasoning)
            }
        )

    def calculate_confidence_from_capture(
        self, capture_result: Any
    ) -> ConfidenceScore:
        """
        Calculate confidence from a capture result.

        Args:
            capture_result: Capture result with OCR data

        Returns:
            ConfidenceScore
        """
        ocr_confidence = None
        data_completeness = 0.0

        # Extract OCR confidence
        if hasattr(capture_result, 'ocr_confidence'):
            ocr_confidence = capture_result.ocr_confidence
        elif hasattr(capture_result, 'confidence'):
            ocr_confidence = capture_result.confidence

        # Check data completeness
        if hasattr(capture_result, 'ocr_text') and capture_result.ocr_text:
            data_completeness = 1.0
            # Estimate text clarity
            text_clarity = self.calculator._estimate_text_clarity(capture_result.ocr_text)
        else:
            data_completeness = 0.0
            text_clarity = 0.0

        # Check if capture was successful
        image_quality = 1.0 if (hasattr(capture_result, 'success') and capture_result.success) else 0.0

        return self.calculator.calculate_confidence(
            ocr_confidence=ocr_confidence,
            image_quality=image_quality,
            text_clarity=text_clarity,
            data_completeness=data_completeness
        )

    def format_multiple_captures(
        self,
        response: str,
        captures: List[Any]
    ) -> ConfidenceFormattedResponse:
        """
        Format response based on multiple capture results.

        Uses the minimum confidence from all captures.

        Args:
            response: Response text
            captures: List of capture results

        Returns:
            ConfidenceFormattedResponse
        """
        if not captures:
            # No captures = low confidence
            return self.format_with_confidence(
                response,
                confidence_score=ConfidenceScore(
                    overall=0.3,
                    level=ConfidenceLevel.LOW,
                    sources={},
                    reasoning=["No capture data available"]
                )
            )

        # Calculate confidence for each capture
        confidence_scores = [
            self.calculate_confidence_from_capture(capture)
            for capture in captures
        ]

        # Use minimum confidence (most conservative)
        min_confidence = min(score.overall for score in confidence_scores)

        # Aggregate reasoning
        all_reasoning = []
        for score in confidence_scores:
            all_reasoning.extend(score.reasoning)

        # Remove duplicates while preserving order
        unique_reasoning = list(dict.fromkeys(all_reasoning))

        # Aggregate sources (average)
        all_sources = {}
        for score in confidence_scores:
            for source, value in score.sources.items():
                if source not in all_sources:
                    all_sources[source] = []
                all_sources[source].append(value)

        averaged_sources = {
            source: sum(values) / len(values)
            for source, values in all_sources.items()
        }

        # Create aggregate confidence score
        aggregate_score = ConfidenceScore(
            overall=min_confidence,
            level=self.calculator._classify_confidence(min_confidence),
            sources=averaged_sources,
            reasoning=unique_reasoning[:3]  # Limit to top 3 reasons
        )

        return self.format_with_confidence(response, aggregate_score)

    def set_visual_indicators(self, enabled: bool):
        """Enable or disable visual indicators"""
        self.include_visual_indicators = enabled

    def set_reasoning_inclusion(self, enabled: bool):
        """Enable or disable reasoning text"""
        self.include_reasoning = enabled


# Global instance
_confidence_manager: Optional[ConfidenceManager] = None


def get_confidence_manager() -> Optional[ConfidenceManager]:
    """Get the global ConfidenceManager instance"""
    return _confidence_manager


def initialize_confidence_manager(
    include_visual_indicators: bool = True,
    include_reasoning: bool = True,
    min_confidence_for_high: float = 0.8,
    min_confidence_for_medium: float = 0.5
) -> ConfidenceManager:
    """
    Initialize the global ConfidenceManager instance.

    Args:
        include_visual_indicators: Add ✅ ⚠️ ❓ indicators
        include_reasoning: Include reasoning for non-high confidence
        min_confidence_for_high: Minimum score for HIGH level
        min_confidence_for_medium: Minimum score for MEDIUM level

    Returns:
        ConfidenceManager instance
    """
    global _confidence_manager

    _confidence_manager = ConfidenceManager(
        include_visual_indicators=include_visual_indicators,
        include_reasoning=include_reasoning,
        min_confidence_for_high=min_confidence_for_high,
        min_confidence_for_medium=min_confidence_for_medium
    )

    return _confidence_manager
