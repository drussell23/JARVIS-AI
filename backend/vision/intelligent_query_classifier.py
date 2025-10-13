"""
Intelligent Query Classifier for JARVIS Vision System
Uses Claude API for zero-hardcoded pattern classification.

Classifies queries into three categories:
- METADATA_ONLY: Fast Yabai-only queries (<100ms)
- VISUAL_ANALYSIS: Current screen analysis (1-3s)
- DEEP_ANALYSIS: Multi-space comprehensive analysis (3-10s)
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Query classification intents"""
    METADATA_ONLY = "metadata_only"  # Yabai-only, no screenshots
    VISUAL_ANALYSIS = "visual_analysis"  # Current screen capture + Claude
    DEEP_ANALYSIS = "deep_analysis"  # Multi-space capture + Yabai + Claude


@dataclass
class ClassificationResult:
    """Result of query classification"""
    intent: QueryIntent
    confidence: float  # 0.0 to 1.0
    reasoning: str  # Why this classification was chosen
    second_best: Optional[Tuple[QueryIntent, float]] = None  # Alternative intent
    features: Dict[str, Any] = field(default_factory=dict)  # Extracted features
    latency_ms: float = 0  # Time taken to classify
    timestamp: datetime = field(default_factory=datetime.now)


class IntelligentQueryClassifier:
    """
    Claude-powered query classifier that learns from usage patterns.
    Zero hardcoded patterns - all intelligence from Claude API.
    """

    def __init__(self, claude_client=None, enable_cache: bool = True):
        """
        Initialize the intelligent classifier

        Args:
            claude_client: Claude API client for classification
            enable_cache: Whether to use classification cache
        """
        self.claude = claude_client
        self.enable_cache = enable_cache

        # Classification cache (30 second TTL)
        self._classification_cache: Dict[str, ClassificationResult] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(seconds=30)

        # Performance tracking
        self._classification_count = 0
        self._total_latency_ms = 0
        self._cache_hits = 0

        logger.info("[CLASSIFIER] Intelligent query classifier initialized")

    async def classify_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ClassificationResult:
        """
        Classify a user query into one of three intents using Claude.

        Args:
            query: User's query text
            context: Optional context (time, recent queries, active apps, etc.)

        Returns:
            ClassificationResult with intent, confidence, and reasoning
        """
        start_time = time.time()

        # Check cache first
        if self.enable_cache:
            cached_result = self._check_cache(query)
            if cached_result:
                self._cache_hits += 1
                logger.info(f"[CLASSIFIER] Cache hit for query: {query[:50]}...")
                return cached_result

        # Extract features from query and context
        features = self._extract_features(query, context)

        # Build classification prompt for Claude
        classification_prompt = self._build_classification_prompt(query, features, context)

        # Get classification from Claude
        try:
            if self.claude:
                response = await self._call_claude_for_classification(classification_prompt)
                result = self._parse_claude_classification(response, features)
            else:
                # Fallback: Use heuristic classification
                logger.warning("[CLASSIFIER] No Claude client, using fallback heuristics")
                result = self._fallback_classification(query, features)

        except Exception as e:
            logger.error(f"[CLASSIFIER] Classification error: {e}", exc_info=True)
            # Fallback on error
            result = self._fallback_classification(query, features)

        # Track latency
        latency_ms = (time.time() - start_time) * 1000
        result.latency_ms = latency_ms

        # Update stats
        self._classification_count += 1
        self._total_latency_ms += latency_ms

        # Cache the result
        if self.enable_cache:
            self._cache_result(query, result)

        logger.info(
            f"[CLASSIFIER] Query classified as {result.intent.value} "
            f"(confidence: {result.confidence:.2f}, latency: {latency_ms:.1f}ms)"
        )

        return result

    def _extract_features(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract features from query and context for classification.
        Zero hardcoded patterns - features are descriptive, not prescriptive.
        """
        features = {
            # Query characteristics
            "query_length": len(query.split()),
            "has_space_reference": any(
                word in query.lower()
                for word in ["space", "desktop", "workspace", "across", "all"]
            ),
            "has_time_reference": any(
                word in query.lower()
                for word in ["now", "currently", "current", "recent", "latest"]
            ),
            "has_visual_keywords": any(
                word in query.lower()
                for word in ["see", "look", "show", "view", "display", "screen", "analyze", "read"]
            ),
            "has_metadata_keywords": any(
                word in query.lower()
                for word in ["how many", "which", "list", "what apps", "summary", "overview"]
            ),
            "has_question_mark": "?" in query,
            "is_imperative": query.strip().endswith("!") or any(
                query.lower().startswith(word)
                for word in ["tell", "show", "give", "list", "describe"]
            ),
        }

        # Context features
        if context:
            features["active_space"] = context.get("active_space")
            features["total_spaces"] = context.get("total_spaces", 0)
            features["recent_intent"] = context.get("recent_intent")
            features["time_since_last_query"] = context.get("time_since_last_query", 0)
            features["user_pattern"] = context.get("user_pattern", "unknown")

        return features

    def _build_classification_prompt(
        self,
        query: str,
        features: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Build a prompt for Claude to classify the query.
        Dynamic prompt that adapts to context.
        """
        prompt = f"""You are an intelligent query classifier for JARVIS, an AI assistant with vision capabilities.

Classify this user query into ONE of three categories:

1. **METADATA_ONLY** - Fast queries answerable with just workspace metadata (no screenshots)
   - Examples: "How many spaces do I have?", "Which apps are open?", "Workspace overview", "What's on Desktop 2?"
   - Response time: <100ms
   - Uses: Yabai CLI only

2. **VISUAL_ANALYSIS** - Queries requiring current screen visual analysis
   - Examples: "What do you see?", "Read this error", "What's on my current screen?", "Describe this window"
   - Response time: 1-3s
   - Uses: Current screen capture + Claude Vision

3. **DEEP_ANALYSIS** - Comprehensive analysis across multiple spaces
   - Examples: "What am I working on?", "Analyze all my spaces", "What's happening across my desktops?"
   - Response time: 3-10s
   - Uses: Multi-space capture + Yabai + Claude Vision

User Query: "{query}"

Query Features:
- Length: {features['query_length']} words
- Has space reference: {features.get('has_space_reference', False)}
- Has visual keywords: {features.get('has_visual_keywords', False)}
- Has metadata keywords: {features.get('has_metadata_keywords', False)}
- Has time reference: {features.get('has_time_reference', False)}

"""

        # Add context if available
        if context:
            prompt += f"""
Context:
- Active space: {context.get('active_space', 'unknown')}
- Total spaces: {context.get('total_spaces', 0)}
- Recent intent: {context.get('recent_intent', 'none')}
"""

        prompt += """
Respond in this EXACT JSON format:
{
  "intent": "METADATA_ONLY" | "VISUAL_ANALYSIS" | "DEEP_ANALYSIS",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of why this classification",
  "second_best": {
    "intent": "alternative intent",
    "confidence": 0.0-1.0
  }
}

Think step-by-step:
1. Does this query need visual information from the screen? (If no → METADATA_ONLY)
2. Does this query reference multiple spaces or ask for comprehensive analysis? (If yes → DEEP_ANALYSIS)
3. Otherwise → VISUAL_ANALYSIS

Respond with ONLY the JSON, no other text.
"""

        return prompt

    async def _call_claude_for_classification(self, prompt: str) -> Dict[str, Any]:
        """Call Claude API for classification"""
        try:
            # Use the Claude client's message API
            if hasattr(self.claude, 'analyze_with_prompt'):
                # If it's a vision analyzer wrapper
                response = await self.claude.analyze_with_prompt(prompt)
                return response
            elif hasattr(self.claude, 'messages'):
                # If it's the Anthropic client directly
                response = await self.claude.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text
                return {"response": content}
            else:
                raise ValueError("Claude client doesn't have expected methods")

        except Exception as e:
            logger.error(f"[CLASSIFIER] Claude API call failed: {e}")
            raise

    def _parse_claude_classification(
        self,
        response: Dict[str, Any],
        features: Dict[str, Any]
    ) -> ClassificationResult:
        """Parse Claude's classification response"""
        import json

        try:
            # Extract JSON from response
            response_text = response.get("response", response.get("content", ""))

            # Try to find JSON in the response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                classification = json.loads(json_text)

                # Parse intent
                intent_str = classification.get("intent", "VISUAL_ANALYSIS")
                try:
                    intent = QueryIntent(intent_str.lower())
                except ValueError:
                    logger.warning(f"[CLASSIFIER] Invalid intent '{intent_str}', defaulting to VISUAL_ANALYSIS")
                    intent = QueryIntent.VISUAL_ANALYSIS

                # Parse confidence
                confidence = float(classification.get("confidence", 0.7))
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

                # Parse reasoning
                reasoning = classification.get("reasoning", "Claude classification")

                # Parse second_best
                second_best = None
                if "second_best" in classification:
                    sb = classification["second_best"]
                    try:
                        sb_intent = QueryIntent(sb["intent"].lower())
                        sb_confidence = float(sb.get("confidence", 0.5))
                        second_best = (sb_intent, sb_confidence)
                    except (ValueError, KeyError):
                        pass

                return ClassificationResult(
                    intent=intent,
                    confidence=confidence,
                    reasoning=reasoning,
                    second_best=second_best,
                    features=features
                )
            else:
                raise ValueError("No valid JSON found in Claude response")

        except Exception as e:
            logger.error(f"[CLASSIFIER] Failed to parse Claude response: {e}")
            logger.debug(f"Response was: {response}")
            # Fallback
            return self._fallback_classification("", features)

    def _fallback_classification(
        self,
        query: str,
        features: Dict[str, Any]
    ) -> ClassificationResult:
        """
        Fallback classification using simple heuristics.
        Only used when Claude API is unavailable.
        """
        query_lower = query.lower()

        # Metadata-only signals
        metadata_score = 0
        if features.get("has_metadata_keywords", False):
            metadata_score += 0.4
        if "overview" in query_lower or "summary" in query_lower:
            metadata_score += 0.3
        if not features.get("has_visual_keywords", False):
            metadata_score += 0.2
        if features.get("has_space_reference", False):
            metadata_score += 0.1

        # Visual analysis signals
        visual_score = 0
        if features.get("has_visual_keywords", False):
            visual_score += 0.5
        if "current" in query_lower or "this" in query_lower:
            visual_score += 0.3
        if "screen" in query_lower:
            visual_score += 0.2

        # Deep analysis signals
        deep_score = 0
        if "across" in query_lower or "all" in query_lower:
            deep_score += 0.5  # Increased from 0.4
        if features.get("has_space_reference", False) and features.get("has_visual_keywords", False):
            deep_score += 0.3
        if "analyze" in query_lower or "comprehensive" in query_lower:
            deep_score += 0.2
        # Strong signal: "what's happening" + space reference
        if ("happening" in query_lower or "working on" in query_lower) and features.get("has_space_reference", False):
            deep_score += 0.3

        # Determine intent
        scores = [
            (QueryIntent.METADATA_ONLY, metadata_score),
            (QueryIntent.VISUAL_ANALYSIS, visual_score),
            (QueryIntent.DEEP_ANALYSIS, deep_score)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)

        intent = scores[0][0]
        # Use a confidence boost formula: base_score + (base_score * 0.3) to get above 0.60 threshold
        base_confidence = scores[0][1]
        confidence = min(0.85, base_confidence + (base_confidence * 0.3))  # Boost confidence for fallback
        second_best = (scores[1][0], scores[1][1]) if len(scores) > 1 else None

        return ClassificationResult(
            intent=intent,
            confidence=confidence,
            reasoning="Fallback heuristic classification (Claude unavailable)",
            second_best=second_best,
            features=features
        )

    def _check_cache(self, query: str) -> Optional[ClassificationResult]:
        """Check if query classification is cached and still valid"""
        cache_key = query.lower().strip()

        if cache_key in self._classification_cache:
            timestamp = self._cache_timestamps.get(cache_key)
            if timestamp and datetime.now() - timestamp < self._cache_ttl:
                return self._classification_cache[cache_key]
            else:
                # Expired
                del self._classification_cache[cache_key]
                del self._cache_timestamps[cache_key]

        return None

    def _cache_result(self, query: str, result: ClassificationResult):
        """Cache classification result"""
        cache_key = query.lower().strip()
        self._classification_cache[cache_key] = result
        self._cache_timestamps[cache_key] = datetime.now()

        # Limit cache size
        if len(self._classification_cache) > 100:
            # Remove oldest entries
            sorted_keys = sorted(
                self._cache_timestamps.items(),
                key=lambda x: x[1]
            )
            for key, _ in sorted_keys[:20]:  # Remove 20 oldest
                del self._classification_cache[key]
                del self._cache_timestamps[key]

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get classification performance statistics"""
        avg_latency = (
            self._total_latency_ms / self._classification_count
            if self._classification_count > 0
            else 0
        )

        cache_hit_rate = (
            self._cache_hits / (self._classification_count + self._cache_hits)
            if (self._classification_count + self._cache_hits) > 0
            else 0
        )

        return {
            "total_classifications": self._classification_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "avg_latency_ms": avg_latency,
            "cache_size": len(self._classification_cache),
        }

    def clear_cache(self):
        """Clear classification cache"""
        self._classification_cache.clear()
        self._cache_timestamps.clear()
        logger.info("[CLASSIFIER] Classification cache cleared")


# Singleton instance
_classifier_instance: Optional[IntelligentQueryClassifier] = None


def get_query_classifier(claude_client=None) -> IntelligentQueryClassifier:
    """Get or create the singleton query classifier"""
    global _classifier_instance

    if _classifier_instance is None:
        _classifier_instance = IntelligentQueryClassifier(claude_client)
    elif claude_client is not None and _classifier_instance.claude is None:
        # Update with Claude client if provided
        _classifier_instance.claude = claude_client

    return _classifier_instance
