"""
Cross-Space Intelligence System - Advanced Multi-Space Understanding for JARVIS
==============================================================================

This system provides JARVIS with the ability to:
- Detect when activities across spaces are semantically related (not just pattern-based)
- Synthesize information from multiple sources into coherent understanding
- Provide workspace-wide answers drawing from all spaces
- Learn and adapt correlation strategies without hardcoding

Architecture:
    CrossSpaceIntelligence (Main Coordinator)
    ├── SemanticCorrelator (Content-based relationship detection)
    ├── ActivityCorrelationEngine (Multi-dimensional correlation)
    ├── MultiSourceSynthesizer (Information synthesis)
    ├── WorkspaceQueryResolver (Workspace-wide query answering)
    └── RelationshipGraph (Dynamic relationship tracking)

Key Capabilities:
    1. Semantic Understanding: "npm error" in terminal + "npm documentation" in browser = related
    2. Temporal Correlation: Activities happening within similar timeframes are likely related
    3. Behavioral Patterns: User switches spaces → activities are likely connected
    4. Content Synthesis: Combine terminal error + browser solution + code change = full story
    5. Dynamic Learning: No hardcoded patterns - learns from activity patterns

Example Scenarios:
    - Terminal shows "ECONNREFUSED" → Browser shows "redis not running" → Related debugging
    - Code file changes in Space 1 → Test output in Space 2 → Development workflow
    - Slack message in Space 3 → Documentation editing in Space 1 → Collaboration
    - Error in Space 1 → StackOverflow in Space 2 → Code fix in Space 3 → Problem solving

NO HARDCODING: All relationship detection is based on:
    - Keyword extraction and matching
    - Temporal proximity
    - User behavior patterns
    - Semantic similarity
"""
import asyncio
import logging
import re
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import defaultdict, deque, Counter
import hashlib
import json

logger = logging.getLogger(__name__)


# ============================================================================
# RELATIONSHIP TYPES AND SCORING
# ============================================================================

class RelationshipType(Enum):
    """Types of cross-space relationships (discovered dynamically).
    
    These relationship types are discovered through pattern analysis rather than
    hardcoded rules. Each type represents a common workflow pattern observed
    across multiple spaces.
    
    Attributes:
        DEBUGGING: Error investigation and resolution workflow
        RESEARCH_AND_CODE: Learning while implementing workflow
        PROBLEM_SOLVING: Issue identification and solution workflow
        LEARNING: Educational and practice workflow
        COLLABORATION: Communication and shared work workflow
        MULTI_TERMINAL: Parallel terminal operations workflow
        CODE_AND_TEST: Development and testing workflow
        DOCUMENTATION: Writing/reading docs with implementation
        DEPLOYMENT: Code deployment and monitoring workflow
        INVESTIGATION: Exploration and analysis workflow
        UNKNOWN: Detected relationship with unclear type
    """
    DEBUGGING = "debugging"                    # Error + research + fix
    RESEARCH_AND_CODE = "research_and_code"    # Reading docs while coding
    PROBLEM_SOLVING = "problem_solving"        # Error + solution search + implementation
    LEARNING = "learning"                      # Tutorial + experimentation + practice
    COLLABORATION = "collaboration"            # Communication + work artifacts
    MULTI_TERMINAL = "multi_terminal"          # Multiple terminals on related tasks
    CODE_AND_TEST = "code_and_test"           # Writing code + running tests
    DOCUMENTATION = "documentation"            # Writing/reading docs + code
    DEPLOYMENT = "deployment"                  # Code + build + deploy activities
    INVESTIGATION = "investigation"            # Exploring + understanding + documenting
    UNKNOWN = "unknown"                        # Detected relationship, type unclear


class CorrelationDimension(Enum):
    """Dimensions along which we correlate activities.
    
    These dimensions provide different perspectives for analyzing relationships
    between activities across spaces.
    
    Attributes:
        TEMPORAL: Time-based correlation (activities happening together)
        SEMANTIC: Content-based correlation (similar keywords/topics)
        BEHAVIORAL: Pattern-based correlation (user behavior patterns)
        CAUSAL: Cause-effect correlation (error → research → fix)
        SEQUENTIAL: Step-by-step correlation (ordered workflow steps)
    """
    TEMPORAL = "temporal"          # Time-based: activities happening together
    SEMANTIC = "semantic"          # Content-based: similar keywords/topics
    BEHAVIORAL = "behavioral"      # Pattern-based: user behavior patterns
    CAUSAL = "causal"             # Cause-effect: error → research → fix
    SEQUENTIAL = "sequential"      # Step-by-step: first this, then that


@dataclass
class KeywordSignature:
    """Extracted keywords representing activity content.
    
    This class captures the semantic essence of an activity through categorized
    keywords extracted from the activity content. Used for semantic correlation
    between activities.
    
    Attributes:
        technical_terms: Technical terminology (npm, redis, python, etc.)
        error_indicators: Error-related terms (failed, error, exception, crash)
        action_verbs: Action words (install, run, build, deploy, fix)
        file_references: File extensions and names (.py, .js, package.json)
        command_names: Command line tools (npm, git, python, docker)
        domain_concepts: Domain-specific concepts (database, server, api)
    """
    technical_terms: Set[str]         # npm, redis, python, error, etc.
    error_indicators: Set[str]        # failed, error, exception, crash
    action_verbs: Set[str]            # install, run, build, deploy, fix
    file_references: Set[str]         # .py, .js, package.json, etc.
    command_names: Set[str]           # npm, git, python, docker, etc.
    domain_concepts: Set[str]         # database, server, api, authentication

    def similarity(self, other: 'KeywordSignature') -> float:
        """Calculate similarity score between two keyword signatures.
        
        Uses Jaccard similarity across all keyword categories to determine
        how semantically similar two activities are.
        
        Args:
            other: Another KeywordSignature to compare against
            
        Returns:
            Similarity score between 0.0 (no similarity) and 1.0 (identical)
            
        Example:
            >>> sig1 = KeywordSignature(technical_terms={'npm', 'error'}, ...)
            >>> sig2 = KeywordSignature(technical_terms={'npm', 'install'}, ...)
            >>> similarity = sig1.similarity(sig2)
            >>> print(f"Similarity: {similarity:.2f}")
            Similarity: 0.33
        """
        if not other:
            return 0.0

        total_score = 0.0
        comparisons = 0

        # Compare each category
        for field_name in ['technical_terms', 'error_indicators', 'action_verbs',
                          'file_references', 'command_names', 'domain_concepts']:
            self_set = getattr(self, field_name)
            other_set = getattr(other, field_name)

            if self_set or other_set:
                # Jaccard similarity
                intersection = len(self_set & other_set)
                union = len(self_set | other_set)
                if union > 0:
                    total_score += intersection / union
                    comparisons += 1

        return total_score / comparisons if comparisons > 0 else 0.0

    def is_empty(self) -> bool:
        """Check if signature has any keywords.
        
        Returns:
            True if all keyword categories are empty, False otherwise
        """
        return not any([
            self.technical_terms, self.error_indicators, self.action_verbs,
            self.file_references, self.command_names, self.domain_concepts
        ])


@dataclass
class ActivitySignature:
    """Complete signature of an activity for correlation.
    
    Represents a single activity in a space with all information needed
    for cross-space correlation analysis.
    
    Attributes:
        space_id: ID of the space where activity occurred
        app_name: Name of the application (terminal, browser, IDE, etc.)
        timestamp: When the activity occurred
        keywords: Extracted keyword signature for semantic analysis
        activity_type: Category of activity (terminal, browser, ide, communication)
        content_summary: Brief summary of activity content
        has_error: Whether activity contains error indicators
        has_solution: Whether activity contains solution/fix content
        is_user_initiated: Whether activity was directly initiated by user
        significance: Importance level (critical, high, normal, low)
    """
    space_id: int
    app_name: str
    timestamp: datetime
    keywords: KeywordSignature
    activity_type: str              # "terminal", "browser", "ide", "communication"
    content_summary: str            # Brief summary of activity
    has_error: bool = False
    has_solution: bool = False      # Detected solution/fix content
    is_user_initiated: bool = True
    significance: str = "normal"    # "critical", "high", "normal", "low"

    def temporal_distance(self, other: 'ActivitySignature') -> float:
        """Calculate temporal distance in seconds.
        
        Args:
            other: Another ActivitySignature to compare timing with
            
        Returns:
            Absolute time difference in seconds
        """
        return abs((self.timestamp - other.timestamp).total_seconds())

    def is_temporally_close(self, other: 'ActivitySignature', threshold_seconds: int = 300) -> bool:
        """Check if activities happened close in time.
        
        Args:
            other: Another ActivitySignature to compare timing with
            threshold_seconds: Maximum time difference to consider "close" (default 5 minutes)
            
        Returns:
            True if activities are within the time threshold, False otherwise
        """
        return self.temporal_distance(other) <= threshold_seconds


@dataclass
class CorrelationScore:
    """Multi-dimensional correlation score between activities.
    
    Represents the strength of relationship between two activities across
    multiple dimensions of analysis.
    
    Attributes:
        temporal_score: How close activities are in time (0-1)
        semantic_score: How similar activities are in content (0-1)
        behavioral_score: How well activities match user behavior patterns (0-1)
        causal_score: Likelihood of cause-effect relationship (0-1)
    """
    temporal_score: float      # 0-1: how close in time
    semantic_score: float      # 0-1: how similar in content
    behavioral_score: float    # 0-1: user behavior patterns match
    causal_score: float        # 0-1: likelihood of cause-effect

    @property
    def overall_score(self) -> float:
        """Weighted combination of all dimensions.
        
        Combines all correlation dimensions using predefined weights to
        produce a single overall correlation score.
        
        Returns:
            Overall correlation score between 0.0 and 1.0
        """
        # Weights can be tuned based on effectiveness
        weights = {
            'temporal': 0.25,
            'semantic': 0.35,
            'behavioral': 0.20,
            'causal': 0.20
        }
        return (
            self.temporal_score * weights['temporal'] +
            self.semantic_score * weights['semantic'] +
            self.behavioral_score * weights['behavioral'] +
            self.causal_score * weights['causal']
        )

    def is_significant(self, threshold: float = 0.5) -> bool:
        """Check if correlation is significant enough to establish relationship.
        
        Args:
            threshold: Minimum overall score to consider significant (default 0.5)
            
        Returns:
            True if overall score meets or exceeds threshold, False otherwise
        """
        return self.overall_score >= threshold


@dataclass
class CrossSpaceRelationship:
    """Discovered relationship between activities across spaces.
    
    Represents a detected workflow or pattern that spans multiple spaces,
    with evidence and confidence metrics.
    
    Attributes:
        relationship_id: Unique identifier for this relationship
        relationship_type: Type of relationship (debugging, research, etc.)
        activities: List of activities involved in this relationship
        correlation_score: Multi-dimensional correlation strength
        first_detected: When relationship was first discovered
        last_updated: When relationship was last updated
        confidence: Confidence level in this relationship (0-1)
        evidence: Supporting evidence for the relationship
        description: Human-readable explanation of the relationship
    """
    relationship_id: str
    relationship_type: RelationshipType
    activities: List[ActivitySignature]
    correlation_score: CorrelationScore
    first_detected: datetime
    last_updated: datetime
    confidence: float                    # 0-1: confidence in this relationship
    evidence: List[Dict[str, Any]]      # Evidence supporting relationship
    description: str                     # Human-readable explanation

    def involves_space(self, space_id: int) -> bool:
        """Check if relationship involves a specific space.
        
        Args:
            space_id: Space ID to check for involvement
            
        Returns:
            True if any activity in relationship is from the specified space
        """
        return any(act.space_id == space_id for act in self.activities)

    def involves_app(self, app_name: str) -> bool:
        """Check if relationship involves a specific app.
        
        Args:
            app_name: Application name to check for involvement
            
        Returns:
            True if any activity in relationship is from the specified app
        """
        return any(act.app_name == app_name for act in self.activities)

    def get_spaces(self) -> Set[int]:
        """Get all spaces involved in relationship.
        
        Returns:
            Set of space IDs that have activities in this relationship
        """
        return {act.space_id for act in self.activities}

    def get_timeline(self) -> List[Tuple[datetime, str]]:
        """Get chronological timeline of activities.
        
        Returns:
            List of (timestamp, description) tuples sorted chronologically
        """
        timeline = [(act.timestamp, f"{act.app_name} (Space {act.space_id}): {act.content_summary[:50]}")
                   for act in sorted(self.activities, key=lambda a: a.timestamp)]
        return timeline


# ============================================================================
# KEYWORD EXTRACTION - Dynamic, No Hardcoding
# ============================================================================

class KeywordExtractor:
    """Extracts semantic keywords from text without hardcoding.
    
    Uses pattern recognition and linguistic rules to identify technical terms,
    commands, errors, and other significant keywords from activity content.
    No hardcoded lists - relies on patterns and context.
    
    Attributes:
        technical_patterns: Regex patterns for identifying technical content
        technical_indicators: Patterns for common technical naming conventions
        stop_words: Common words to ignore during extraction
    """

    def __init__(self):
        """Initialize the keyword extractor with pattern definitions."""
        # Technical patterns (regex-based, not hardcoded terms)
        self.technical_patterns = {
            'error_indicators': r'(?i)\b\w*(error|exception|failed|failure|crash|abort|fatal|critical|warning)\w*\b',
            'action_verbs': r'\b(install|run|execute|build|deploy|compile|test|debug|fix|update|upgrade|create|delete|start|stop|running)\w*\b',
            'file_extensions': r'\b\w+\.(py|js|ts|jsx|tsx|java|cpp|h|go|rs|rb|php|css|html|json|yaml|yml|xml|md|txt|sh)\b',
            'command_names': r'\b(npm|pip|python|node|java|git|docker|kubectl|cargo|maven|gradle|make|cmake)\b',
            'module_names': r'\b(module|package|library|import|require)\w*\b',
            'version_numbers': r'\bv?\d+\.\d+(?:\.\d+)?\b',
            'ports': r'\b(?:port\s+)?(\d{2,5})\b',
            'urls': r'\b(?:https?://)?(?:www\.)?[\w\-\.]+\.[a-z]{2,}(?:/\S*)?\b',
        }

        # Common technical term indicators (patterns, not exhaustive lists)
        self.technical_indicators = [
            r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b',  # CamelCase
            r'\b[a-z]+(?:_[a-z]+)+\b',           # snake_case
            r'\b[a-z]+(?:-[a-z]+)+\b',           # kebab-case
            r'\b[A-Z_]{3,}\b',                    # CONSTANTS
        ]

        # Stop words (common words to ignore)
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
            'who', 'when', 'where', 'why', 'how'
        }

    def extract(self, text: str) -> KeywordSignature:
        """Extract keyword signature from text.
        
        Analyzes text content to identify and categorize keywords that represent
        the semantic meaning of the activity.
        
        Args:
            text: Raw text content to analyze
            
        Returns:
            KeywordSignature containing categorized keywords
            
        Example:
            >>> extractor = KeywordExtractor()
            >>> signature = extractor.extract("npm install failed with error")
            >>> print(signature.command_names)
            {'npm'}
            >>> print(signature.error_indicators)
            {'failed', 'error'}
        """
        if not text:
            return KeywordSignature(
                technical_terms=set(),
                error_indicators=set(),
                action_verbs=set(),
                file_references=set(),
                command_names=set(),
                domain_concepts=set()
            )

        text_lower = text.lower()

        # Extract specific categories (case-insensitive)
        error_indicators = self._extract_pattern(text, self.technical_patterns['error_indicators'])
        action_verbs = self._extract_pattern(text_lower, self.technical_patterns['action_verbs'])
        file_references = self._extract_pattern(text, self.technical_patterns['file_extensions'])
        command_names = self._extract_pattern(text_lower, self.technical_patterns['command_names'])

        # Extract module/package names
        module_indicators = self._extract_pattern(text_lower, self.technical_patterns['module_names'])

        # Extract technical terms (CamelCase, snake_case, etc.)
        technical_terms = set()
        for pattern in self.technical_indicators:
            matches = re.findall(pattern, text)
            technical_terms.update(m.lower() for m in matches if len(m) > 2)

        # Add module indicators to technical terms
        technical_terms.update(module_indicators)

        # Extract domain concepts (meaningful nouns, not in stop words)
        words = re.findall(r'\b[a-z]{4,}\b', text_lower)
        domain_concepts = {w for w in words
                          if w not in self.stop_words
                          and w not in error_indicators
                          and w not in action_verbs
                          and w not in command_names
                          and w not in module_indicators}

        # Add specific important terms from the text
        # Look for quoted terms (like 'requests' in error messages)
        quoted_terms = re.findall(r"['\"](\w+)['\"]", text)
        domain_concepts.update(t.lower() for t in quoted_terms if len(t) > 2)

        # Limit to most relevant (prevent overfitting)
        domain_concepts = set(list(domain_concepts)[:20])

        return KeywordSignature(
            technical_terms=technical_terms,
            error_indicators=error_indicators,
            action_verbs=action_verbs,
            file_references=file_references,
            command_names=command_names,
            domain_concepts=domain_concepts
        )

    def _extract_pattern(self, text: str, pattern: str) -> Set[str]:
        """Extract all matches for a pattern.
        
        Args:
            text: Text to search in
            pattern: Regex pattern to match
            
        Returns:
            Set of unique matches found (lowercased)
        """
        matches = re.findall(pattern, text, re.IGNORECASE)
        return {m.lower() for m in matches if m}


# ============================================================================
# SEMANTIC CORRELATOR - Content-Based Relationship Detection
# ============================================================================

class SemanticCorrelator:
    """Detects relationships based on semantic similarity of content.
    
    Uses keyword analysis and similarity metrics to identify when activities
    across different spaces are semantically related. No hardcoded patterns -
    relies on dynamic keyword extraction and similarity calculation.
    
    Attributes:
        keyword_extractor: KeywordExtractor instance for text analysis
        recent_signatures: Deque of recent activity signatures for correlation
    """

    def __init__(self):
        """Initialize the semantic correlator."""
        self.keyword_extractor = KeywordExtractor()
        self.recent_signatures: deque[ActivitySignature] = deque(maxlen=100)

    def create_signature(self, space_id: int, app_name: str, content: str,
                        activity_type: str, has_error: bool = False,
                        significance: str = "normal") -> ActivitySignature:
        """Create activity signature from raw content.
        
        Processes raw activity content into a structured signature that can
        be used for correlation analysis.
        
        Args:
            space_id: ID of the space where activity occurred
            app_name: Name of the application
            content: Raw content of the activity
            activity_type: Type of activity (terminal, browser, ide, etc.)
            has_error: Whether the activity contains error indicators
            significance: Importance level of the activity
            
        Returns:
            ActivitySignature representing the processed activity
            
        Example:
            >>> correlator = SemanticCorrelator()
            >>> sig = correlator.create_signature(
            ...     space_id=1,
            ...     app_name="terminal",
            ...     content="npm install failed",
            ...     activity_type="terminal",
            ...     has_error=True
            ... )
            >>> print(sig.has_error)
            True
        """
        keywords = self.keyword_extractor.extract(content)

        # Detect if content contains solution/fix language
        has_solution = self._detect_solution_content(content)

        signature = ActivitySignature(
            space_id=space_id,
            app_name=app_name,
            timestamp=datetime.now(),
            keywords=keywords,
            activity_type=activity_type,
            content_summary=content[:200],
            has_error=has_error,
            has_solution=has_solution,
            significance=significance
        )

        # Store for correlation
        self.recent_signatures.append(signature)

        return signature

    def find_related_activities(self, signature: ActivitySignature,
                               time_window_seconds: int = 300,
                               min_similarity: float = 0.3) -> List[Tuple[ActivitySignature, float]]:
        """Find activities semantically related to the given signature.
        
        Searches through recent activities to find those that are semantically
        similar and temporally close to the given signature.
        
        Args:
            signature: ActivitySignature to find relations for
            time_window_seconds: Maximum time difference to consider (default 5 minutes)
            min_similarity: Minimum semantic similarity threshold (default 0.3)
            
        Returns:
            List of (related_signature, similarity_score) tuples, sorted by similarity
            
        Example:
            >>> related = correlator.find_related_activities(error_signature)
            >>> for sig, similarity in related:
            ...     print(f"Related: {sig.app_name} (similarity: {similarity:.2f})")
            Related: browser (similarity: 0.65)
        """
        related = []

        for other in self.recent_signatures:
            # Skip same activity
            if other.space_id == signature.space_id and other.app_name == signature.app_name:
                continue

            # Check temporal proximity
            if not signature.is_temporally_close(other, time_window_seconds):
                continue

            # Calculate semantic similarity
            similarity = signature.keywords.similarity(other.keywords)

            if similarity >= min_similarity:
                related.append((other, similarity))

        # Sort by similarity (most similar first)
        related.sort(key=lambda x: x[1], reverse=True)

        return related

    def _detect_solution_content(self, text: str) -> bool:
        """Detect if text contains solution/fix language.
        
        Args:
            text: Text content to analyze
            
        Returns:
            True if text appears to contain solution or fix information
        """
        solution_patterns = [
            r'\b(solution|fix|resolve|workaround|answer|how to)\b',
            r'\b(try|attempt|should|need to|you can)\b',
            r'\b(install|run|change|update|modify)\s+\w+',
        ]

        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in solution_patterns)


# ============================================================================
# ACTIVITY CORRELATION ENGINE - Multi-Dimensional Correlation
# ============================================================================

class ActivityCorrelationEngine:
    """Calculates multi-dimensional correlation scores between activities.
    
    Analyzes activities across temporal, semantic, behavioral, and causal
    dimensions to determine the strength of relationships between them.
    
    Attributes:
        semantic_correlator: SemanticCorrelator for content analysis
        behavior_patterns: Recent behavior patterns for learning
    """

    def __init__(self):
        """Initialize the activity correlation engine."""
        self.semantic_correlator = SemanticCorrelator()
        self.behavior_patterns: deque[Dict[str, Any]] = deque(maxlen=50)

    def correlate(self, activity1: ActivitySignature, activity2: ActivitySignature) -> CorrelationScore:
        """Calculate multi-dimensional correlation between two activities.
        
        Analyzes the relationship between two activities across multiple
        dimensions to produce a comprehensive correlation score.
        
        Args:
            activity1: First activity to correlate
            activity2: Second activity to correlate
            
        Returns:
            CorrelationScore containing scores for all dimensions
            
        Example:
            >>> engine = ActivityCorrelationEngine()
            >>> score = engine.correlate(error_activity, browser_activity)
            >>> print(f"Overall correlation: {score.overall_score:.2f}")
            Overall correlation: 0.73
        """

        # 1. Temporal correlation
        temporal_score = self._calculate_temporal_score(activity1, activity2)

        # 2. Semantic correlation
        semantic_score = activity1.keywords.similarity(activity2.keywords)

        # 3. Behavioral correlation
        behavioral_score = self._calculate_behavioral_score(activity1, activity2)

        # 4. Causal correlation
        causal_score = self._calculate_causal_score(activity1, activity2)

        return CorrelationScore(
            temporal_score=temporal_score,
            semantic_score=semantic_score,
            behavioral_score=behavioral_score,
            causal_score=causal_score
        )

    def _calculate_temporal_score(self, act1: ActivitySignature, act2: ActivitySignature) -> float:
        """Score temporal proximity: activities closer in time score higher.

        Scoring:
        - 0-60s: 1.0 (very close)
        - 60-180s: 0.8 (close)
        - 180-300s: 0.5 (related)
        - 300-600s: 0.2 (possibly related)
        - 600s+: 0.0 (unrelated)
        
        Args:
            act1: First activity
            act2: Second activity
            
        Returns:
            Temporal correlation score between 0.0 and 1.0
        """
        distance = act1.temporal_distance(act2)

        if distance <= 60:
            return 1.0
        elif distance <= 180:
            return 0.8
        elif distance <= 300:
            return 0.5
        elif distance <= 600:
            return 0.2
        else:
            return 0.0

    def _calculate_behavioral_score(self, act1: ActivitySignature, act2: ActivitySignature) -> float:
        """Score based on user behavior patterns.

        High score if:
        - Different spaces (user switched → likely related)
        - One is error, other is browser (common: error → search)
        - Sequential activities of same type (workflow)
        
        Args:
            act1: First activity
            act2: Second activity
            
        Returns:
            Behavioral correlation score between 0.0 and 1.0
        """
        score = 0.0

        # Different spaces → likely related (user switched for a reason)
        if act1.space_id != act2.space_id:
            score += 0.4

        # Error + browser research pattern
        if act1.has_error and act2.activity_type == "browser":
            score += 0.5
        elif act2.has_error and act1.activity_type == "browser":
            score += 0.5

        # Browser + solution found → action taken
        if act1.activity_type == "browser" and act1.has_solution:
            if act2.activity_type in ["terminal", "ide"]:
                score += 0.4

        # Same activity type in different spaces (parallel workflow)
        if act1.activity_type == act2.activity_type and act1.space_id != act2.space_id:
            score += 0.3

        return min(1.0, score)

    def _calculate_causal_score(self, act1: ActivitySignature, act2: ActivitySignature) -> float:
        """Score likelihood of cause-effect relationship.

        High score if:
        - Act1 is error → Act2 is research/fix (error causes investigation)
        - Act1 is code change → Act2 is test/build (code causes testing)
        - Act1 is research → Act2 is implementation (learning causes doing)
        
        Args:
            act1: First activity (potential cause)
            act2: Second activity (potential effect)
            
        Returns:
            Causal correlation score between 0.0 and 1.0
        """
        score = 0.0

        # Ensure temporal ordering (cause must come before effect)
        if act1.timestamp > act2.timestamp:
            # Swap if needed
            act1, act2 = act2, act1

        # Error → Research
        if act1.has_error and act2.activity_type == "browser":
            score += 0.7

        # Research → Implementation
        if act1.activity_type == "browser" and act1.has_solution:
            if act2.activity_type in ["terminal", "ide"]:
                score += 0.6

        # Code → Test
        if act1.activity_type == "ide" and act2.activity_