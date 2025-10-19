"""
Complex Complexity Handler
==========================

Handles Level 3 (COMPLEX) queries that require:
- Temporal analysis ("What changed in the last 5 minutes?")
- Cross-space intelligence ("Find all errors across all spaces")
- Predictive queries ("Am I making progress?")

Processing:
- Query all spaces (1-10+)
- Capture each space
- Run OCR + analysis
- Apply temporal/semantic logic
- Synthesize high-level answer

Latency: 10-30s
API Calls: 5-15+
Requires: v2.0 features (caching, session memory)
"""

import asyncio
import time
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .temporal_query_handler import (
    TemporalQueryHandler,
    TemporalQueryType,
    TemporalQueryResult
)
from .multi_space_query_handler import (
    MultiSpaceQueryHandler,
    MultiSpaceQueryType,
    MultiSpaceQueryResult
)
from .predictive_query_handler import (
    PredictiveQueryHandler,
    PredictiveQueryResponse
)
from ..managers.capture_strategy_manager import CaptureStrategyManager
from ..managers.ocr_strategy_manager import OCRStrategyManager

try:
    from ..implicit_reference_resolver import ImplicitReferenceResolver
    IMPLICIT_RESOLVER_AVAILABLE = True
except ImportError:
    IMPLICIT_RESOLVER_AVAILABLE = False
    ImplicitReferenceResolver = None


class ComplexQueryType(Enum):
    """Types of complex queries"""
    TEMPORAL = "temporal"  # "What changed in the last 5 minutes?"
    CROSS_SPACE = "cross_space"  # "Find all errors across all spaces"
    PREDICTIVE = "predictive"  # "Am I making progress?"
    ANALYTICAL = "analytical"  # "What's the overall status?"


@dataclass
class SpaceSnapshot:
    """Snapshot of a single space"""
    space_id: int
    capture_path: Optional[str] = None
    ocr_text: Optional[str] = None
    ocr_confidence: float = 0.0
    capture_timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class TemporalAnalysis:
    """Results of temporal analysis"""
    changes_detected: bool
    change_count: int
    change_summary: str
    changed_spaces: List[int]
    change_details: List[Dict[str, Any]]
    time_range: Dict[str, Any]


@dataclass
class CrossSpaceAnalysis:
    """Results of cross-space analysis"""
    total_spaces_scanned: int
    matches_found: int
    match_summary: str
    matching_spaces: List[int]
    match_details: List[Dict[str, Any]]


@dataclass
class PredictiveAnalysis:
    """Results of predictive analysis"""
    prediction: str
    confidence: float
    reasoning: str
    supporting_evidence: List[str]
    recommendations: List[str]


@dataclass
class ComplexQueryResult:
    """Result of complex query processing"""
    success: bool
    query_type: ComplexQueryType
    synthesis: str
    snapshots: List[SpaceSnapshot]
    temporal_analysis: Optional[TemporalAnalysis] = None
    cross_space_analysis: Optional[CrossSpaceAnalysis] = None
    predictive_analysis: Optional[PredictiveAnalysis] = None
    execution_time: float = 0.0
    api_calls: int = 0
    spaces_processed: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComplexComplexityHandler:
    """
    Handler for Level 3 (COMPLEX) queries.

    Orchestrates specialized handlers for temporal, cross-space, and predictive queries.
    Provides high-level synthesis and intelligent caching for 5-15+ API calls.
    """

    def __init__(
        self,
        temporal_handler: Optional[TemporalQueryHandler] = None,
        multi_space_handler: Optional[MultiSpaceQueryHandler] = None,
        predictive_handler: Optional[PredictiveQueryHandler] = None,
        capture_manager: Optional[CaptureStrategyManager] = None,
        ocr_manager: Optional[OCRStrategyManager] = None,
        multi_monitor_manager: Optional[Any] = None,
        implicit_resolver: Optional[Any] = None,
        cache_ttl: float = 60.0,
        max_concurrent_captures: int = 5
    ):
        """
        Initialize Complex Complexity Handler.

        Args:
            temporal_handler: Handler for temporal queries
            multi_space_handler: Handler for multi-space queries
            predictive_handler: Handler for predictive queries
            capture_manager: Manager for intelligent capture
            ocr_manager: Manager for intelligent OCR
            multi_monitor_manager: Manager for multi-monitor support
            implicit_resolver: Resolver for implicit references
            cache_ttl: Cache TTL in seconds
            max_concurrent_captures: Max parallel captures
        """
        self.temporal_handler = temporal_handler
        self.multi_space_handler = multi_space_handler
        self.predictive_handler = predictive_handler
        self.capture_manager = capture_manager
        self.ocr_manager = ocr_manager
        self.multi_monitor_manager = multi_monitor_manager
        self.implicit_resolver = implicit_resolver
        self.cache_ttl = cache_ttl
        self.max_concurrent_captures = max_concurrent_captures

        # Snapshot cache for temporal queries
        self._snapshot_cache: Dict[int, SpaceSnapshot] = {}
        self._cache_timestamps: Dict[int, float] = {}

    async def process_query(
        self,
        query: str,
        query_type: ComplexQueryType,
        space_ids: Optional[List[int]] = None,
        time_range: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ComplexQueryResult:
        """
        Process a complex query.

        Args:
            query: The query string
            query_type: Type of complex query
            space_ids: List of space IDs to analyze (None = all spaces)
            time_range: Time range for temporal queries {"minutes": 5, "hours": 1}
            context: Additional context

        Returns:
            ComplexQueryResult with synthesis and analysis
        """
        start_time = time.time()
        api_calls = 0

        try:
            # Step 1: Resolve implicit references
            resolved_query = await self._resolve_references(query, context)

            # Step 2: Determine spaces to analyze
            target_spaces = await self._determine_target_spaces(
                resolved_query, space_ids, context
            )

            if not target_spaces:
                return ComplexQueryResult(
                    success=False,
                    query_type=query_type,
                    synthesis="No spaces available to analyze",
                    snapshots=[],
                    error="No target spaces determined"
                )

            # Step 3: Route to appropriate handler based on query type
            if query_type == ComplexQueryType.TEMPORAL:
                result, calls = await self._handle_temporal_query(
                    resolved_query, target_spaces, time_range, context
                )
            elif query_type == ComplexQueryType.CROSS_SPACE:
                result, calls = await self._handle_cross_space_query(
                    resolved_query, target_spaces, context
                )
            elif query_type == ComplexQueryType.PREDICTIVE:
                result, calls = await self._handle_predictive_query(
                    resolved_query, target_spaces, context
                )
            elif query_type == ComplexQueryType.ANALYTICAL:
                result, calls = await self._handle_analytical_query(
                    resolved_query, target_spaces, context
                )
            else:
                return ComplexQueryResult(
                    success=False,
                    query_type=query_type,
                    synthesis=f"Unsupported query type: {query_type}",
                    snapshots=[],
                    error=f"Unsupported query type: {query_type}"
                )

            api_calls += calls
            result.execution_time = time.time() - start_time
            result.api_calls = api_calls

            return result

        except Exception as e:
            return ComplexQueryResult(
                success=False,
                query_type=query_type,
                synthesis=f"Error processing complex query: {str(e)}",
                snapshots=[],
                execution_time=time.time() - start_time,
                api_calls=api_calls,
                error=str(e)
            )

    async def _resolve_references(
        self, query: str, context: Optional[Dict[str, Any]]
    ) -> str:
        """Resolve implicit references in the query"""
        if not self.implicit_resolver:
            return query

        try:
            resolved = await self.implicit_resolver.resolve(query, context or {})
            return resolved
        except Exception as e:
            # If resolution fails, use original query
            return query

    async def _determine_target_spaces(
        self,
        query: str,
        space_ids: Optional[List[int]],
        context: Optional[Dict[str, Any]]
    ) -> List[int]:
        """Determine which spaces to analyze"""
        # If space_ids provided, use them
        if space_ids:
            return space_ids

        # Extract from context
        if context:
            if "spaces" in context:
                return context["spaces"]
            if "current_space" in context:
                return [context["current_space"]]
            if "system_context" in context:
                sys_ctx = context["system_context"]
                if "current_space_id" in sys_ctx:
                    return [sys_ctx["current_space_id"]]

        # Default: analyze current space only
        # In a real implementation, this would query yabai for current space
        return [1]  # Fallback to space 1

    async def _capture_spaces_parallel(
        self, space_ids: List[int]
    ) -> List[SpaceSnapshot]:
        """Capture multiple spaces in parallel with intelligent caching"""
        current_time = time.time()
        snapshots = []
        spaces_to_capture = []

        # Check cache first
        for space_id in space_ids:
            if space_id in self._snapshot_cache:
                cache_age = current_time - self._cache_timestamps.get(space_id, 0)
                if cache_age < self.cache_ttl:
                    # Use cached snapshot
                    snapshots.append(self._snapshot_cache[space_id])
                    continue

            spaces_to_capture.append(space_id)

        # Capture spaces that need fresh snapshots
        if spaces_to_capture:
            # Limit concurrent captures
            for i in range(0, len(spaces_to_capture), self.max_concurrent_captures):
                batch = spaces_to_capture[i:i + self.max_concurrent_captures]
                batch_snapshots = await asyncio.gather(
                    *[self._capture_single_space(space_id) for space_id in batch],
                    return_exceptions=True
                )

                for snapshot in batch_snapshots:
                    if isinstance(snapshot, SpaceSnapshot):
                        snapshots.append(snapshot)
                        # Update cache
                        self._snapshot_cache[snapshot.space_id] = snapshot
                        self._cache_timestamps[snapshot.space_id] = current_time

        return snapshots

    async def _capture_single_space(self, space_id: int) -> SpaceSnapshot:
        """Capture a single space"""
        snapshot = SpaceSnapshot(
            space_id=space_id,
            capture_timestamp=datetime.now()
        )

        try:
            if self.capture_manager:
                success, result, message = await self.capture_manager.capture_with_fallbacks(
                    space_id=space_id
                )

                if success and result:
                    snapshot.capture_path = result
                else:
                    snapshot.error = message
            else:
                snapshot.error = "Capture manager not available"

        except Exception as e:
            snapshot.error = f"Capture failed: {str(e)}"

        return snapshot

    async def _extract_text_parallel(
        self, snapshots: List[SpaceSnapshot]
    ) -> List[SpaceSnapshot]:
        """Extract text from all snapshots in parallel"""
        # Limit concurrent OCR operations
        results = []
        snapshots_to_process = [s for s in snapshots if s.capture_path and not s.error]

        for i in range(0, len(snapshots_to_process), self.max_concurrent_captures):
            batch = snapshots_to_process[i:i + self.max_concurrent_captures]
            batch_results = await asyncio.gather(
                *[self._extract_text_single(snapshot) for snapshot in batch],
                return_exceptions=True
            )
            results.extend(batch_results)

        # Add snapshots that were skipped (errors or no capture)
        for snapshot in snapshots:
            if snapshot not in snapshots_to_process:
                results.append(snapshot)

        return results

    async def _extract_text_single(self, snapshot: SpaceSnapshot) -> SpaceSnapshot:
        """Extract text from a single snapshot"""
        try:
            if self.ocr_manager and snapshot.capture_path:
                result = await self.ocr_manager.extract_text_with_fallbacks(
                    image_path=snapshot.capture_path,
                    cache_max_age=self.cache_ttl
                )

                if result.success:
                    snapshot.ocr_text = result.text
                    snapshot.ocr_confidence = result.confidence
                else:
                    snapshot.error = result.error or "OCR failed"
            else:
                snapshot.error = "OCR manager not available"

        except Exception as e:
            snapshot.error = f"OCR failed: {str(e)}"

        return snapshot

    async def _handle_temporal_query(
        self,
        query: str,
        space_ids: List[int],
        time_range: Optional[Dict[str, Any]],
        context: Optional[Dict[str, Any]]
    ) -> Tuple[ComplexQueryResult, int]:
        """Handle temporal queries using TemporalQueryHandler"""
        api_calls = 0

        if not self.temporal_handler:
            # Fallback: manual temporal analysis
            return await self._handle_temporal_fallback(
                query, space_ids, time_range, context
            )

        try:
            # Determine temporal query type
            temporal_type = self._classify_temporal_query(query)

            # Process with TemporalQueryHandler
            result = await self.temporal_handler.process_query(
                query=query,
                query_type=temporal_type,
                space_ids=space_ids,
                time_range=time_range,
                context=context
            )

            api_calls += getattr(result, 'api_calls', len(space_ids))

            # Convert to ComplexQueryResult
            temporal_analysis = TemporalAnalysis(
                changes_detected=result.changes_detected if hasattr(result, 'changes_detected') else False,
                change_count=result.change_count if hasattr(result, 'change_count') else 0,
                change_summary=result.summary if hasattr(result, 'summary') else str(result),
                changed_spaces=space_ids,
                change_details=[],
                time_range=time_range or {}
            )

            return ComplexQueryResult(
                success=result.success if hasattr(result, 'success') else True,
                query_type=ComplexQueryType.TEMPORAL,
                synthesis=result.summary if hasattr(result, 'summary') else str(result),
                snapshots=[],
                temporal_analysis=temporal_analysis,
                spaces_processed=len(space_ids)
            ), api_calls

        except Exception as e:
            return ComplexQueryResult(
                success=False,
                query_type=ComplexQueryType.TEMPORAL,
                synthesis=f"Temporal query failed: {str(e)}",
                snapshots=[],
                error=str(e)
            ), api_calls

    async def _handle_temporal_fallback(
        self,
        query: str,
        space_ids: List[int],
        time_range: Optional[Dict[str, Any]],
        context: Optional[Dict[str, Any]]
    ) -> Tuple[ComplexQueryResult, int]:
        """Fallback temporal analysis when TemporalQueryHandler not available"""
        # Capture current snapshots
        current_snapshots = await self._capture_spaces_parallel(space_ids)
        current_snapshots = await self._extract_text_parallel(current_snapshots)

        api_calls = len([s for s in current_snapshots if s.ocr_text])

        # Compare with cached snapshots for change detection
        changes_detected = False
        changed_spaces = []
        change_details = []

        for snapshot in current_snapshots:
            if snapshot.space_id in self._snapshot_cache:
                cached = self._snapshot_cache[snapshot.space_id]
                if cached.ocr_text != snapshot.ocr_text:
                    changes_detected = True
                    changed_spaces.append(snapshot.space_id)
                    change_details.append({
                        "space_id": snapshot.space_id,
                        "previous_text": cached.ocr_text,
                        "current_text": snapshot.ocr_text
                    })

        synthesis = self._synthesize_temporal_changes(
            changes_detected, changed_spaces, change_details
        )

        temporal_analysis = TemporalAnalysis(
            changes_detected=changes_detected,
            change_count=len(changed_spaces),
            change_summary=synthesis,
            changed_spaces=changed_spaces,
            change_details=change_details,
            time_range=time_range or {}
        )

        return ComplexQueryResult(
            success=True,
            query_type=ComplexQueryType.TEMPORAL,
            synthesis=synthesis,
            snapshots=current_snapshots,
            temporal_analysis=temporal_analysis,
            spaces_processed=len(space_ids)
        ), api_calls

    async def _handle_cross_space_query(
        self,
        query: str,
        space_ids: List[int],
        context: Optional[Dict[str, Any]]
    ) -> Tuple[ComplexQueryResult, int]:
        """Handle cross-space queries using MultiSpaceQueryHandler"""
        api_calls = 0

        if not self.multi_space_handler:
            # Fallback: manual cross-space analysis
            return await self._handle_cross_space_fallback(query, space_ids, context)

        try:
            # Determine multi-space query type
            multi_space_type = self._classify_cross_space_query(query)

            # Process with MultiSpaceQueryHandler
            result = await self.multi_space_handler.process_query(
                query=query,
                query_type=multi_space_type,
                space_ids=space_ids,
                context=context
            )

            api_calls += getattr(result, 'api_calls', len(space_ids))

            # Convert to ComplexQueryResult
            cross_space_analysis = CrossSpaceAnalysis(
                total_spaces_scanned=len(space_ids),
                matches_found=result.matches_found if hasattr(result, 'matches_found') else 0,
                match_summary=result.summary if hasattr(result, 'summary') else str(result),
                matching_spaces=result.matching_spaces if hasattr(result, 'matching_spaces') else [],
                match_details=result.details if hasattr(result, 'details') else []
            )

            return ComplexQueryResult(
                success=result.success if hasattr(result, 'success') else True,
                query_type=ComplexQueryType.CROSS_SPACE,
                synthesis=result.summary if hasattr(result, 'summary') else str(result),
                snapshots=[],
                cross_space_analysis=cross_space_analysis,
                spaces_processed=len(space_ids)
            ), api_calls

        except Exception as e:
            return ComplexQueryResult(
                success=False,
                query_type=ComplexQueryType.CROSS_SPACE,
                synthesis=f"Cross-space query failed: {str(e)}",
                snapshots=[],
                error=str(e)
            ), api_calls

    async def _handle_cross_space_fallback(
        self,
        query: str,
        space_ids: List[int],
        context: Optional[Dict[str, Any]]
    ) -> Tuple[ComplexQueryResult, int]:
        """Fallback cross-space analysis when MultiSpaceQueryHandler not available"""
        # Capture all spaces
        snapshots = await self._capture_spaces_parallel(space_ids)
        snapshots = await self._extract_text_parallel(snapshots)

        api_calls = len([s for s in snapshots if s.ocr_text])

        # Extract search terms from query
        search_terms = self._extract_search_terms(query)

        # Search across all spaces
        matching_spaces = []
        match_details = []

        for snapshot in snapshots:
            if not snapshot.ocr_text or snapshot.error:
                continue

            matches = self._find_matches_in_text(snapshot.ocr_text, search_terms)
            if matches:
                matching_spaces.append(snapshot.space_id)
                match_details.append({
                    "space_id": snapshot.space_id,
                    "matches": matches
                })

        synthesis = self._synthesize_cross_space_results(
            query, matching_spaces, match_details, len(space_ids)
        )

        cross_space_analysis = CrossSpaceAnalysis(
            total_spaces_scanned=len(space_ids),
            matches_found=len(matching_spaces),
            match_summary=synthesis,
            matching_spaces=matching_spaces,
            match_details=match_details
        )

        return ComplexQueryResult(
            success=True,
            query_type=ComplexQueryType.CROSS_SPACE,
            synthesis=synthesis,
            snapshots=snapshots,
            cross_space_analysis=cross_space_analysis,
            spaces_processed=len(space_ids)
        ), api_calls

    async def _handle_predictive_query(
        self,
        query: str,
        space_ids: List[int],
        context: Optional[Dict[str, Any]]
    ) -> Tuple[ComplexQueryResult, int]:
        """Handle predictive queries using PredictiveQueryHandler"""
        api_calls = 0

        if not self.predictive_handler:
            # Fallback: manual predictive analysis
            return await self._handle_predictive_fallback(query, space_ids, context)

        try:
            # Capture current state
            snapshots = await self._capture_spaces_parallel(space_ids)
            snapshots = await self._extract_text_parallel(snapshots)

            api_calls += len([s for s in snapshots if s.ocr_text])

            # Process with PredictiveQueryHandler
            result = await self.predictive_handler.handle_predictive_query(
                query=query,
                context=context or {}
            )

            api_calls += 1  # Claude Vision API call

            # Convert to ComplexQueryResult
            predictive_analysis = PredictiveAnalysis(
                prediction=result.prediction if hasattr(result, 'prediction') else "",
                confidence=result.confidence if hasattr(result, 'confidence') else 0.0,
                reasoning=result.reasoning if hasattr(result, 'reasoning') else "",
                supporting_evidence=result.evidence if hasattr(result, 'evidence') else [],
                recommendations=result.recommendations if hasattr(result, 'recommendations') else []
            )

            synthesis = f"**Prediction:** {predictive_analysis.prediction}\n\n"
            synthesis += f"**Confidence:** {predictive_analysis.confidence:.1%}\n\n"
            synthesis += f"**Reasoning:** {predictive_analysis.reasoning}"

            return ComplexQueryResult(
                success=True,
                query_type=ComplexQueryType.PREDICTIVE,
                synthesis=synthesis,
                snapshots=snapshots,
                predictive_analysis=predictive_analysis,
                spaces_processed=len(space_ids)
            ), api_calls

        except Exception as e:
            return ComplexQueryResult(
                success=False,
                query_type=ComplexQueryType.PREDICTIVE,
                synthesis=f"Predictive query failed: {str(e)}",
                snapshots=[],
                error=str(e)
            ), api_calls

    async def _handle_predictive_fallback(
        self,
        query: str,
        space_ids: List[int],
        context: Optional[Dict[str, Any]]
    ) -> Tuple[ComplexQueryResult, int]:
        """Fallback predictive analysis when PredictiveQueryHandler not available"""
        # Capture current state
        snapshots = await self._capture_spaces_parallel(space_ids)
        snapshots = await self._extract_text_parallel(snapshots)

        api_calls = len([s for s in snapshots if s.ocr_text])

        # Simple heuristic-based prediction
        prediction = "Unable to make detailed prediction without PredictiveQueryHandler"
        confidence = 0.3
        reasoning = "Using fallback heuristic analysis"

        # Basic progress detection
        if "progress" in query.lower():
            # Count successful vs failed snapshots
            successful = len([s for s in snapshots if s.ocr_text and not s.error])
            total = len(snapshots)
            confidence = successful / total if total > 0 else 0.0
            prediction = f"Making progress: {successful}/{total} spaces accessible"
            reasoning = "Based on space accessibility metrics"

        predictive_analysis = PredictiveAnalysis(
            prediction=prediction,
            confidence=confidence,
            reasoning=reasoning,
            supporting_evidence=[f"Analyzed {len(snapshots)} spaces"],
            recommendations=["Install PredictiveQueryHandler for detailed analysis"]
        )

        synthesis = f"**Prediction:** {prediction}\n\n"
        synthesis += f"**Confidence:** {confidence:.1%}\n\n"
        synthesis += f"**Reasoning:** {reasoning}"

        return ComplexQueryResult(
            success=True,
            query_type=ComplexQueryType.PREDICTIVE,
            synthesis=synthesis,
            snapshots=snapshots,
            predictive_analysis=predictive_analysis,
            spaces_processed=len(space_ids)
        ), api_calls

    async def _handle_analytical_query(
        self,
        query: str,
        space_ids: List[int],
        context: Optional[Dict[str, Any]]
    ) -> Tuple[ComplexQueryResult, int]:
        """Handle general analytical queries"""
        # Capture all spaces
        snapshots = await self._capture_spaces_parallel(space_ids)
        snapshots = await self._extract_text_parallel(snapshots)

        api_calls = len([s for s in snapshots if s.ocr_text])

        # Aggregate statistics
        total_spaces = len(snapshots)
        accessible_spaces = len([s for s in snapshots if s.ocr_text and not s.error])
        failed_spaces = total_spaces - accessible_spaces

        synthesis = f"**Overall Analysis:**\n\n"
        synthesis += f"- Total spaces analyzed: {total_spaces}\n"
        synthesis += f"- Accessible spaces: {accessible_spaces}\n"
        synthesis += f"- Failed captures: {failed_spaces}\n"

        return ComplexQueryResult(
            success=True,
            query_type=ComplexQueryType.ANALYTICAL,
            synthesis=synthesis,
            snapshots=snapshots,
            spaces_processed=total_spaces
        ), api_calls

    def _classify_temporal_query(self, query: str) -> Any:
        """Classify temporal query type"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["changed", "change", "different"]):
            return TemporalQueryType.CHANGE_DETECTION if hasattr(TemporalQueryType, 'CHANGE_DETECTION') else "change_detection"
        elif any(word in query_lower for word in ["history", "past", "before"]):
            return TemporalQueryType.HISTORICAL if hasattr(TemporalQueryType, 'HISTORICAL') else "historical"
        else:
            return TemporalQueryType.CHANGE_DETECTION if hasattr(TemporalQueryType, 'CHANGE_DETECTION') else "change_detection"

    def _classify_cross_space_query(self, query: str) -> Any:
        """Classify cross-space query type"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["find", "search", "locate"]):
            return MultiSpaceQueryType.SEARCH if hasattr(MultiSpaceQueryType, 'SEARCH') else "search"
        elif any(word in query_lower for word in ["all", "every", "across"]):
            return MultiSpaceQueryType.AGGREGATE if hasattr(MultiSpaceQueryType, 'AGGREGATE') else "aggregate"
        else:
            return MultiSpaceQueryType.SEARCH if hasattr(MultiSpaceQueryType, 'SEARCH') else "search"

    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract search terms from query"""
        # Simple extraction: look for quoted terms and common keywords
        terms = []

        # Extract quoted terms
        import re
        quoted = re.findall(r'"([^"]*)"', query)
        terms.extend(quoted)

        # Common search keywords
        keywords = ["error", "warning", "fail", "exception", "bug", "issue", "problem"]
        query_lower = query.lower()
        for keyword in keywords:
            if keyword in query_lower and keyword not in terms:
                terms.append(keyword)

        return terms if terms else [query]  # Fallback to full query

    def _find_matches_in_text(
        self, text: str, search_terms: List[str]
    ) -> List[str]:
        """Find search term matches in text"""
        matches = []
        text_lower = text.lower()

        for term in search_terms:
            if term.lower() in text_lower:
                matches.append(term)

        return matches

    def _synthesize_temporal_changes(
        self,
        changes_detected: bool,
        changed_spaces: List[int],
        change_details: List[Dict[str, Any]]
    ) -> str:
        """Synthesize temporal change results"""
        if not changes_detected:
            return "No changes detected in the analyzed time period."

        synthesis = f"**Changes detected in {len(changed_spaces)} space(s):**\n\n"

        for detail in change_details[:3]:  # Limit to first 3 for brevity
            space_id = detail["space_id"]
            synthesis += f"- Space {space_id}: Content changed\n"

        if len(change_details) > 3:
            synthesis += f"- ... and {len(change_details) - 3} more spaces\n"

        return synthesis

    def _synthesize_cross_space_results(
        self,
        query: str,
        matching_spaces: List[int],
        match_details: List[Dict[str, Any]],
        total_spaces: int
    ) -> str:
        """Synthesize cross-space search results"""
        if not matching_spaces:
            return f"No matches found across {total_spaces} spaces."

        synthesis = f"**Found matches in {len(matching_spaces)} of {total_spaces} spaces:**\n\n"

        for detail in match_details[:3]:  # Limit to first 3
            space_id = detail["space_id"]
            matches = detail.get("matches", [])
            synthesis += f"- Space {space_id}: {', '.join(matches)}\n"

        if len(match_details) > 3:
            synthesis += f"- ... and {len(match_details) - 3} more spaces\n"

        return synthesis


# Global instance
_complex_handler: Optional[ComplexComplexityHandler] = None


def get_complex_complexity_handler() -> Optional[ComplexComplexityHandler]:
    """Get the global ComplexComplexityHandler instance"""
    return _complex_handler


def initialize_complex_complexity_handler(
    temporal_handler: Optional[TemporalQueryHandler] = None,
    multi_space_handler: Optional[MultiSpaceQueryHandler] = None,
    predictive_handler: Optional[PredictiveQueryHandler] = None,
    capture_manager: Optional[CaptureStrategyManager] = None,
    ocr_manager: Optional[OCRStrategyManager] = None,
    multi_monitor_manager: Optional[Any] = None,
    implicit_resolver: Optional[Any] = None,
    cache_ttl: float = 60.0,
    max_concurrent_captures: int = 5
) -> ComplexComplexityHandler:
    """
    Initialize the global ComplexComplexityHandler instance.

    Args:
        temporal_handler: Handler for temporal queries
        multi_space_handler: Handler for multi-space queries
        predictive_handler: Handler for predictive queries
        capture_manager: Manager for intelligent capture
        ocr_manager: Manager for intelligent OCR
        multi_monitor_manager: Manager for multi-monitor support
        implicit_resolver: Resolver for implicit references
        cache_ttl: Cache TTL in seconds
        max_concurrent_captures: Max parallel captures

    Returns:
        ComplexComplexityHandler instance
    """
    global _complex_handler

    _complex_handler = ComplexComplexityHandler(
        temporal_handler=temporal_handler,
        multi_space_handler=multi_space_handler,
        predictive_handler=predictive_handler,
        capture_manager=capture_manager,
        ocr_manager=ocr_manager,
        multi_monitor_manager=multi_monitor_manager,
        implicit_resolver=implicit_resolver,
        cache_ttl=cache_ttl,
        max_concurrent_captures=max_concurrent_captures
    )

    return _complex_handler
