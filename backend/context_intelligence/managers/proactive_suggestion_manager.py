"""
Proactive Suggestion Manager
=============================

Generates intelligent next-step suggestions based on conversation context.

Good Response (Proactive):
✅ Query: "What's in space 5?"
   Response: "Space 5 shows Chrome with error documentation for NoneType.
              Would you like me to compare this with the error in Space 3?"

Bad Response (Not Proactive):
❌ Query: "What's in space 5?"
   Response: "Space 5 shows Chrome with error documentation for NoneType."
   (No suggestions for next steps)

Strategy:
- Analyze query and response for patterns
- Use conversation context to find related entities
- Generate contextual suggestions (compare, deep dive, related, action)
- Rank suggestions by relevance
- Format suggestions as natural questions
"""

import asyncio
import re
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SuggestionType(Enum):
    """Types of proactive suggestions"""
    COMPARISON = "comparison"  # "Compare with X?"
    DEEP_DIVE = "deep_dive"  # "Want more details?"
    RELATED = "related"  # "See related X?"
    ACTION = "action"  # "Should I fix this?"
    SEARCH = "search"  # "Search for similar?"
    NAVIGATE = "navigate"  # "Go to X?"
    ANALYZE = "analyze"  # "Analyze X?"


@dataclass
class Suggestion:
    """A proactive suggestion"""
    type: SuggestionType
    text: str  # The suggestion text
    confidence: float  # 0.0-1.0
    reasoning: str  # Why this suggestion
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher = more important


@dataclass
class SuggestionResult:
    """Result of suggestion generation"""
    suggestions: List[Suggestion]
    top_suggestion: Optional[Suggestion]
    formatted_text: str  # Ready-to-append text
    metadata: Dict[str, Any] = field(default_factory=dict)


class PatternAnalyzer:
    """
    Analyzes query and response patterns to identify suggestion opportunities.

    Patterns:
    - Error detected → Suggest comparison, search, fix
    - Multiple entities → Suggest comparison
    - Surface-level info → Suggest deep dive
    - Code mentioned → Suggest navigation, analysis
    """

    def __init__(self):
        """Initialize pattern analyzer"""
        # Compile patterns for efficiency
        self.error_pattern = re.compile(r'(\w+Error|\w+Exception)', re.IGNORECASE)
        self.file_pattern = re.compile(r'[\w\-]+\.(?:py|js|ts|java|cpp|go|rs|rb)')
        self.line_pattern = re.compile(r'line\s+(\d+)', re.IGNORECASE)
        self.function_pattern = re.compile(r'function\s+[\'"]?(\w+)[\'"]?', re.IGNORECASE)
        self.space_pattern = re.compile(r'[Ss]pace\s+(\d+)')

    def detect_patterns(
        self, query: str, response: str
    ) -> Dict[str, Any]:
        """
        Detect patterns in query and response.

        Args:
            query: User query
            response: System response

        Returns:
            Dictionary with detected patterns
        """
        patterns = {
            'has_error': False,
            'error_types': [],
            'has_files': False,
            'files': [],
            'has_line_number': False,
            'line_numbers': [],
            'has_function': False,
            'functions': [],
            'spaces_mentioned': [],
            'is_informational': False,
            'is_surface_level': True,
            'has_code': False,
        }

        # Combine query and response for analysis
        combined = f"{query} {response}"

        # Detect errors
        errors = self.error_pattern.findall(combined)
        if errors:
            patterns['has_error'] = True
            patterns['error_types'] = list(set(errors))

        # Detect files
        files = self.file_pattern.findall(combined)
        if files:
            patterns['has_files'] = True
            patterns['files'] = list(set(files))

        # Detect line numbers
        lines = self.line_pattern.findall(combined)
        if lines:
            patterns['has_line_number'] = True
            patterns['line_numbers'] = [int(l) for l in lines]

        # Detect functions
        functions = self.function_pattern.findall(combined)
        if functions:
            patterns['has_function'] = True
            patterns['functions'] = list(set(functions))

        # Detect spaces
        spaces = self.space_pattern.findall(combined)
        if spaces:
            patterns['spaces_mentioned'] = [int(s) for s in spaces]

        # Check if informational query
        info_words = ['what', 'show', 'tell', 'display', 'list']
        if any(word in query.lower() for word in info_words):
            patterns['is_informational'] = True

        # Check if surface level (short response without details)
        if len(response.split()) < 20 and not patterns['has_line_number']:
            patterns['is_surface_level'] = True
        else:
            patterns['is_surface_level'] = False

        # Check for code indicators
        code_indicators = ['function', 'class', 'variable', 'code', 'syntax']
        if any(word in combined.lower() for word in code_indicators):
            patterns['has_code'] = True

        return patterns


class SuggestionGenerator:
    """
    Generates proactive suggestions based on patterns and context.

    Suggestion strategies:
    - Comparison: Compare entities of same type
    - Deep dive: Get more details about surface-level info
    - Related: Find related entities
    - Action: Suggest actions to fix/improve
    """

    def __init__(self):
        """Initialize suggestion generator"""
        self.pattern_analyzer = PatternAnalyzer()

    def generate_suggestions(
        self,
        query: str,
        response: str,
        conversation_context: Optional[Any] = None,
        patterns: Optional[Dict[str, Any]] = None
    ) -> List[Suggestion]:
        """
        Generate proactive suggestions.

        Args:
            query: User query
            response: System response
            conversation_context: ConversationContext from tracker
            patterns: Pre-detected patterns (optional)

        Returns:
            List of suggestions
        """
        # Detect patterns if not provided
        if patterns is None:
            patterns = self.pattern_analyzer.detect_patterns(query, response)

        suggestions = []

        # Generate comparison suggestions
        suggestions.extend(
            self._generate_comparison_suggestions(patterns, conversation_context)
        )

        # Generate deep dive suggestions
        suggestions.extend(
            self._generate_deep_dive_suggestions(patterns, conversation_context)
        )

        # Generate related suggestions
        suggestions.extend(
            self._generate_related_suggestions(patterns, conversation_context)
        )

        # Generate action suggestions
        suggestions.extend(
            self._generate_action_suggestions(patterns, conversation_context)
        )

        # Generate search suggestions
        suggestions.extend(
            self._generate_search_suggestions(patterns, conversation_context)
        )

        return suggestions

    def _generate_comparison_suggestions(
        self, patterns: Dict[str, Any], context: Optional[Any]
    ) -> List[Suggestion]:
        """Generate comparison suggestions"""
        suggestions = []

        # If error detected and other spaces in context
        if patterns['has_error'] and context:
            current_spaces = set(patterns['spaces_mentioned'])
            recent_spaces = set(getattr(context, 'recent_spaces', []))

            # Suggest comparing with other spaces that have errors
            other_spaces = recent_spaces - current_spaces
            if other_spaces and len(current_spaces) > 0:
                other_space = sorted(other_spaces)[0]  # Get first other space
                current_space = sorted(current_spaces)[0]

                suggestions.append(Suggestion(
                    type=SuggestionType.COMPARISON,
                    text=f"Would you like me to compare this with the error in Space {other_space}?",
                    confidence=0.8,
                    reasoning=f"Both Space {current_space} and Space {other_space} were recently discussed and have errors",
                    context={'spaces': [current_space, other_space]},
                    priority=3
                ))

        # If multiple spaces mentioned, suggest comparison
        if len(patterns['spaces_mentioned']) >= 2:
            spaces = sorted(patterns['spaces_mentioned'])[:2]
            suggestions.append(Suggestion(
                type=SuggestionType.COMPARISON,
                text=f"Should I compare Space {spaces[0]} and Space {spaces[1]}?",
                confidence=0.85,
                reasoning="Multiple spaces mentioned in conversation",
                context={'spaces': spaces},
                priority=4
            ))

        return suggestions

    def _generate_deep_dive_suggestions(
        self, patterns: Dict[str, Any], context: Optional[Any]
    ) -> List[Suggestion]:
        """Generate deep dive suggestions"""
        suggestions = []

        # If surface-level response, suggest more details
        if patterns['is_surface_level'] and patterns['is_informational']:
            suggestions.append(Suggestion(
                type=SuggestionType.DEEP_DIVE,
                text="Would you like more detailed information?",
                confidence=0.6,
                reasoning="Response is surface-level, user might want details",
                priority=2
            ))

        # If error without line number, suggest details
        if patterns['has_error'] and not patterns['has_line_number']:
            suggestions.append(Suggestion(
                type=SuggestionType.DEEP_DIVE,
                text="Should I analyze the error in detail?",
                confidence=0.75,
                reasoning="Error detected but no line number specified",
                priority=3
            ))

        # If file mentioned without function, suggest code dive
        if patterns['has_files'] and not patterns['has_function']:
            file_name = patterns['files'][0]
            suggestions.append(Suggestion(
                type=SuggestionType.DEEP_DIVE,
                text=f"Would you like to see the code in {file_name}?",
                confidence=0.7,
                reasoning="File mentioned but no specific function",
                context={'file': file_name},
                priority=2
            ))

        return suggestions

    def _generate_related_suggestions(
        self, patterns: Dict[str, Any], context: Optional[Any]
    ) -> List[Suggestion]:
        """Generate related suggestions"""
        suggestions = []

        # If error detected, suggest seeing related code
        if patterns['has_error'] and patterns['has_files']:
            error_type = patterns['error_types'][0]
            file_name = patterns['files'][0]

            if patterns['has_line_number']:
                line_num = patterns['line_numbers'][0]
                suggestions.append(Suggestion(
                    type=SuggestionType.RELATED,
                    text=f"Want to see the code around line {line_num} in {file_name}?",
                    confidence=0.85,
                    reasoning="Error with specific location - user might want to see code",
                    context={'file': file_name, 'line': line_num},
                    priority=4
                ))

        # If multiple files in context, suggest exploring related files
        if context and hasattr(context, 'recent_files') and len(context.recent_files) > 1:
            other_files = [f for f in context.recent_files if f not in patterns.get('files', [])]
            if other_files:
                suggestions.append(Suggestion(
                    type=SuggestionType.RELATED,
                    text=f"Should I check {other_files[0]} as well?",
                    confidence=0.65,
                    reasoning="Multiple files in recent context",
                    context={'file': other_files[0]},
                    priority=2
                ))

        return suggestions

    def _generate_action_suggestions(
        self, patterns: Dict[str, Any], context: Optional[Any]
    ) -> List[Suggestion]:
        """Generate action suggestions"""
        suggestions = []

        # If error with specific location, suggest fix
        if patterns['has_error'] and patterns['has_line_number'] and patterns['has_files']:
            error_type = patterns['error_types'][0]
            suggestions.append(Suggestion(
                type=SuggestionType.ACTION,
                text=f"Would you like suggestions to fix the {error_type}?",
                confidence=0.7,
                reasoning="Specific error location identified",
                priority=3
            ))

        # If NoneType error, suggest adding null checks
        if any('NoneType' in e for e in patterns.get('error_types', [])):
            suggestions.append(Suggestion(
                type=SuggestionType.ACTION,
                text="Should I suggest where to add null/None checks?",
                confidence=0.75,
                reasoning="NoneType error indicates missing null checks",
                priority=3
            ))

        return suggestions

    def _generate_search_suggestions(
        self, patterns: Dict[str, Any], context: Optional[Any]
    ) -> List[Suggestion]:
        """Generate search suggestions"""
        suggestions = []

        # If error detected, suggest searching for similar errors
        if patterns['has_error']:
            error_type = patterns['error_types'][0]
            suggestions.append(Suggestion(
                type=SuggestionType.SEARCH,
                text=f"Want me to search for other {error_type}s in your code?",
                confidence=0.65,
                reasoning="Error detected - might be systemic",
                context={'error_type': error_type},
                priority=2
            ))

        # If error across multiple spaces, suggest cross-space search
        if patterns['has_error'] and context and hasattr(context, 'recent_spaces'):
            if len(context.recent_spaces) > 1:
                suggestions.append(Suggestion(
                    type=SuggestionType.SEARCH,
                    text="Should I check all spaces for similar errors?",
                    confidence=0.7,
                    reasoning="Multiple spaces in context with errors",
                    priority=3
                ))

        return suggestions


class SuggestionRanker:
    """
    Ranks and filters suggestions by relevance.

    Ranking factors:
    - Confidence score
    - Priority level
    - Recency of context
    - Diversity (don't suggest same type multiple times)
    """

    def rank_suggestions(
        self,
        suggestions: List[Suggestion],
        max_suggestions: int = 2
    ) -> List[Suggestion]:
        """
        Rank and filter suggestions.

        Args:
            suggestions: List of suggestions to rank
            max_suggestions: Maximum number to return

        Returns:
            Ranked and filtered list
        """
        if not suggestions:
            return []

        # Calculate combined score
        for suggestion in suggestions:
            suggestion.context['score'] = (
                suggestion.confidence * 0.6 +  # 60% confidence
                (suggestion.priority / 5.0) * 0.4  # 40% priority
            )

        # Sort by score
        ranked = sorted(
            suggestions,
            key=lambda s: s.context.get('score', 0.0),
            reverse=True
        )

        # Filter for diversity (prefer different types)
        diverse = []
        seen_types = set()

        for suggestion in ranked:
            if suggestion.type not in seen_types or len(diverse) == 0:
                diverse.append(suggestion)
                seen_types.add(suggestion.type)

            if len(diverse) >= max_suggestions:
                break

        return diverse


class SuggestionFormatter:
    """
    Formats suggestions for display.

    Formats:
    - Natural questions
    - Optional bullets
    - Inline or separate section
    """

    def format_suggestions(
        self,
        suggestions: List[Suggestion],
        format_style: str = "inline"
    ) -> str:
        """
        Format suggestions for display.

        Args:
            suggestions: List of suggestions
            format_style: "inline" or "section"

        Returns:
            Formatted text
        """
        if not suggestions:
            return ""

        if format_style == "inline":
            # Single line after main response
            if len(suggestions) == 1:
                return f"\n\n💡 {suggestions[0].text}"
            else:
                # Multiple suggestions
                lines = ["\n\n💡 Suggestions:"]
                for i, suggestion in enumerate(suggestions, 1):
                    lines.append(f"   {i}. {suggestion.text}")
                return "\n".join(lines)

        elif format_style == "section":
            # Separate section
            lines = ["\n\n**Suggested Next Steps:**"]
            for suggestion in suggestions:
                lines.append(f"- {suggestion.text}")
            return "\n".join(lines)

        else:
            # Default: simple inline
            return f"\n\n{suggestions[0].text}"


class ProactiveSuggestionManager:
    """
    Main manager for proactive suggestion generation.

    Generates contextual next-step suggestions to enhance user experience.
    """

    def __init__(
        self,
        conversation_tracker: Optional[Any] = None,
        implicit_resolver: Optional[Any] = None,
        max_suggestions: int = 2,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize Proactive Suggestion Manager.

        Args:
            conversation_tracker: ConversationTracker instance
            implicit_resolver: ImplicitReferenceResolver instance
            max_suggestions: Maximum suggestions to generate
            confidence_threshold: Minimum confidence for suggestions
        """
        self.conversation_tracker = conversation_tracker
        self.implicit_resolver = implicit_resolver
        self.max_suggestions = max_suggestions
        self.confidence_threshold = confidence_threshold

        self.pattern_analyzer = PatternAnalyzer()
        self.suggestion_generator = SuggestionGenerator()
        self.suggestion_ranker = SuggestionRanker()
        self.suggestion_formatter = SuggestionFormatter()

    async def generate_suggestions(
        self,
        query: str,
        response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SuggestionResult:
        """
        Generate proactive suggestions for a query/response pair.

        Args:
            query: User query
            response: System response
            context: Additional context

        Returns:
            SuggestionResult with ranked suggestions
        """
        # Step 1: Get conversation context
        conversation_context = None
        if self.conversation_tracker:
            conversation_context = self.conversation_tracker.get_recent_context()

        # Step 2: Detect patterns
        patterns = self.pattern_analyzer.detect_patterns(query, response)

        # Step 3: Generate suggestions
        all_suggestions = self.suggestion_generator.generate_suggestions(
            query, response, conversation_context, patterns
        )

        # Step 4: Filter by confidence threshold
        filtered = [
            s for s in all_suggestions
            if s.confidence >= self.confidence_threshold
        ]

        # Step 5: Rank and limit
        ranked = self.suggestion_ranker.rank_suggestions(
            filtered, self.max_suggestions
        )

        # Step 6: Format
        formatted_text = self.suggestion_formatter.format_suggestions(
            ranked, format_style="inline"
        )

        return SuggestionResult(
            suggestions=ranked,
            top_suggestion=ranked[0] if ranked else None,
            formatted_text=formatted_text,
            metadata={
                'patterns': patterns,
                'total_generated': len(all_suggestions),
                'filtered_count': len(filtered),
                'final_count': len(ranked)
            }
        )

    def set_max_suggestions(self, max_suggestions: int):
        """Set maximum number of suggestions"""
        self.max_suggestions = max_suggestions

    def set_confidence_threshold(self, threshold: float):
        """Set minimum confidence threshold"""
        self.confidence_threshold = threshold


# Global instance
_proactive_suggestion_manager: Optional[ProactiveSuggestionManager] = None


def get_proactive_suggestion_manager() -> Optional[ProactiveSuggestionManager]:
    """Get the global ProactiveSuggestionManager instance"""
    return _proactive_suggestion_manager


def initialize_proactive_suggestion_manager(
    conversation_tracker: Optional[Any] = None,
    implicit_resolver: Optional[Any] = None,
    max_suggestions: int = 2,
    confidence_threshold: float = 0.5
) -> ProactiveSuggestionManager:
    """
    Initialize the global ProactiveSuggestionManager instance.

    Args:
        conversation_tracker: ConversationTracker instance
        implicit_resolver: ImplicitReferenceResolver instance
        max_suggestions: Maximum suggestions to generate
        confidence_threshold: Minimum confidence for suggestions

    Returns:
        ProactiveSuggestionManager instance
    """
    global _proactive_suggestion_manager

    _proactive_suggestion_manager = ProactiveSuggestionManager(
        conversation_tracker=conversation_tracker,
        implicit_resolver=implicit_resolver,
        max_suggestions=max_suggestions,
        confidence_threshold=confidence_threshold
    )

    return _proactive_suggestion_manager
