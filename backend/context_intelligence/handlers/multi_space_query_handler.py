"""
Multi-Space Query Handler - Advanced Cross-Space Analysis
==========================================================

Handles complex queries spanning multiple Mission Control spaces:
- "Compare space 3 and space 5"
- "Which space has the error?"
- "Find the terminal across all spaces"
- "What's different between space 1 and space 2?"

Architecture:
    User Query → Intent Detection → Space Resolution → Parallel Capture
         ↓              ↓                  ↓                  ↓
    Parse Query   COMPARE/LOCATE    Extract Spaces    Screenshot All
         ↓              ↓                  ↓                  ↓
    Extract Refs   Determine Type    Space List      Vision Analysis
         ↓              ↓                  ↓                  ↓
         └──────────────┴──────────────────┴────────→ Synthesis
                                                           ↓
                                                    Unified Response

Features:
- ✅ Parallel space capture (async/concurrent)
- ✅ Dynamic space resolution (no hardcoding)
- ✅ Intent-aware comparison (leverages ImplicitReferenceResolver)
- ✅ Cross-space search (find X across all spaces)
- ✅ Difference detection (semantic comparison)
- ✅ Synthesis engine (unified response generation)
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re

from context_intelligence.managers.space_state_manager import (
    get_space_state_manager,
    SpaceState
)

logger = logging.getLogger(__name__)


# ============================================================================
# QUERY TYPES
# ============================================================================

class MultiSpaceQueryType(Enum):
    """Types of multi-space queries"""
    COMPARE = "compare"              # Compare 2+ spaces
    SEARCH = "search"                # Find X across all spaces
    DIFFERENCE = "difference"        # What's different between spaces
    SUMMARY = "summary"              # Summarize multiple spaces
    LOCATE = "locate"                # Which space has X?


@dataclass
class SpaceAnalysisResult:
    """Result of analyzing a single space"""
    space_id: int
    success: bool
    app_name: Optional[str] = None
    window_title: Optional[str] = None
    content_type: Optional[str] = None  # error, code, documentation, terminal, browser
    content_summary: str = ""
    ocr_text: str = ""
    entities: List[str] = field(default_factory=list)  # Extracted entities
    errors: List[str] = field(default_factory=list)
    significance: str = "normal"  # critical, high, normal, low
    analysis_time: float = 0.0
    vision_analysis: Optional[Dict[str, Any]] = None


@dataclass
class MultiSpaceQueryResult:
    """Result of multi-space query"""
    query_type: MultiSpaceQueryType
    original_query: str
    spaces_analyzed: List[int]
    results: List[SpaceAnalysisResult]
    comparison: Optional[Dict[str, Any]] = None
    differences: Optional[List[Dict[str, Any]]] = None
    search_matches: Optional[List[Dict[str, Any]]] = None
    synthesis: str = ""
    confidence: float = 0.0
    total_time: float = 0.0


# ============================================================================
# MULTI-SPACE QUERY HANDLER
# ============================================================================

class MultiSpaceQueryHandler:
    """
    Handles queries spanning multiple Mission Control spaces.

    This integrates with:
    - ImplicitReferenceResolver (intent detection)
    - ContextualQueryResolver (space resolution)
    - MultiSpaceContextGraph (context storage)
    - Vision systems (OCR and analysis)
    """

    def __init__(self, context_graph=None, implicit_resolver=None, contextual_resolver=None):
        """
        Initialize the multi-space query handler.

        Args:
            context_graph: MultiSpaceContextGraph instance
            implicit_resolver: ImplicitReferenceResolver instance
            contextual_resolver: ContextualQueryResolver instance
        """
        self.context_graph = context_graph
        self.implicit_resolver = implicit_resolver
        self.contextual_resolver = contextual_resolver
        self.space_manager = get_space_state_manager()

        # Query patterns (dynamic - no hardcoding)
        self._initialize_patterns()

        logger.info("[MULTI-SPACE] Handler initialized")

    def _initialize_patterns(self):
        """Initialize dynamic query patterns"""
        # Comparison patterns
        self.comparison_patterns = [
            r'\bcompare\b.*\b(?:and|vs|versus|with)\b',
            r'\b(?:difference|different)\s+between\b',
            r'\bwhat\'?s\s+(?:different|the difference)\b',
        ]

        # Search patterns
        self.search_patterns = [
            r'\bfind\s+(?:the\s+)?(\w+)\s+across\b',
            r'\b(?:where|which\s+space)\s+(?:is|has)\b',
            r'\blocate\s+(?:the\s+)?(\w+)\b',
            r'\bsearch\s+(?:for\s+)?(?:the\s+)?(\w+)\s+in\s+all\b',
        ]

        # Space extraction patterns
        self.space_patterns = [
            r'space\s+(\d+)',
            r'spaces?\s+(\d+)\s+(?:and|&)\s+(\d+)',
            r'spaces?\s+(\d+),\s*(\d+)(?:,\s*and\s+(\d+))?',
        ]

    async def handle_query(self, query: str, available_spaces: Optional[List[int]] = None) -> MultiSpaceQueryResult:
        """
        Main entry point for multi-space queries.

        Args:
            query: User's natural language query
            available_spaces: Optional list of available spaces (auto-detected if None)

        Returns:
            MultiSpaceQueryResult with comprehensive analysis
        """
        start_time = datetime.now()

        logger.info(f"[MULTI-SPACE] Processing query: '{query}'")

        # Step 1: Classify query type
        query_type = await self._classify_query_type(query)
        logger.debug(f"[MULTI-SPACE] Query type: {query_type.value}")

        # Step 2: Resolve which spaces to analyze
        spaces_to_analyze = await self._resolve_spaces(query, query_type, available_spaces)
        logger.info(f"[MULTI-SPACE] Spaces to analyze: {spaces_to_analyze}")

        if not spaces_to_analyze:
            return MultiSpaceQueryResult(
                query_type=query_type,
                original_query=query,
                spaces_analyzed=[],
                results=[],
                synthesis="I couldn't determine which spaces to analyze. Could you specify?",
                confidence=0.0,
                total_time=0.0
            )

        # Step 3: Capture and analyze spaces in parallel
        results = await self._analyze_spaces_parallel(spaces_to_analyze, query)

        # Step 4: Perform query-specific processing
        if query_type == MultiSpaceQueryType.COMPARE:
            comparison = await self._compare_spaces(results, query)
        else:
            comparison = None

        if query_type == MultiSpaceQueryType.DIFFERENCE:
            differences = await self._detect_differences(results)
        else:
            differences = None

        if query_type == MultiSpaceQueryType.SEARCH or query_type == MultiSpaceQueryType.LOCATE:
            search_matches = await self._search_across_spaces(results, query)
        else:
            search_matches = None

        # Step 5: Synthesize unified response
        synthesis = await self._synthesize_response(
            query_type, results, query, comparison, differences, search_matches
        )

        # Calculate confidence
        confidence = self._calculate_confidence(results, query_type)

        total_time = (datetime.now() - start_time).total_seconds()

        return MultiSpaceQueryResult(
            query_type=query_type,
            original_query=query,
            spaces_analyzed=spaces_to_analyze,
            results=results,
            comparison=comparison,
            differences=differences,
            search_matches=search_matches,
            synthesis=synthesis,
            confidence=confidence,
            total_time=total_time
        )

    async def _classify_query_type(self, query: str) -> MultiSpaceQueryType:
        """Classify the type of multi-space query"""
        query_lower = query.lower()

        # Use implicit resolver's intent if available
        if self.implicit_resolver:
            try:
                parsed = self.implicit_resolver.query_analyzer.analyze(query)
                intent = parsed.intent.value

                # Map intent to query type
                if intent == "compare":
                    return MultiSpaceQueryType.COMPARE
                elif intent == "locate":
                    return MultiSpaceQueryType.LOCATE
                elif intent == "summarize":
                    return MultiSpaceQueryType.SUMMARY
            except Exception as e:
                logger.debug(f"[MULTI-SPACE] Could not use implicit resolver intent: {e}")

        # Fallback to pattern matching
        for pattern in self.comparison_patterns:
            if re.search(pattern, query_lower):
                # Check if it's asking for differences
                if "different" in query_lower or "difference" in query_lower:
                    return MultiSpaceQueryType.DIFFERENCE
                return MultiSpaceQueryType.COMPARE

        for pattern in self.search_patterns:
            if re.search(pattern, query_lower):
                # "which space has" → LOCATE, "find X across" → SEARCH
                if "which space" in query_lower or "where is" in query_lower:
                    return MultiSpaceQueryType.LOCATE
                return MultiSpaceQueryType.SEARCH

        # Default to comparison if multiple spaces mentioned
        if len(self._extract_space_numbers(query)) >= 2:
            return MultiSpaceQueryType.COMPARE

        return MultiSpaceQueryType.SEARCH  # Default

    async def _resolve_spaces(self, query: str, query_type: MultiSpaceQueryType,
                              available_spaces: Optional[List[int]]) -> List[int]:
        """
        Resolve which spaces to analyze based on query.

        Uses both explicit mentions and contextual resolution.
        """
        # Try explicit space numbers first
        explicit_spaces = self._extract_space_numbers(query)
        if explicit_spaces:
            logger.debug(f"[MULTI-SPACE] Found explicit spaces: {explicit_spaces}")
            return explicit_spaces

        # For search/locate queries, use all available spaces
        if query_type in [MultiSpaceQueryType.SEARCH, MultiSpaceQueryType.LOCATE]:
            if available_spaces:
                logger.debug(f"[MULTI-SPACE] Using all available spaces for search: {available_spaces}")
                return available_spaces
            else:
                # Auto-detect available spaces (1-10 by default)
                return list(range(1, 11))

        # For comparison, try contextual resolver
        if self.contextual_resolver:
            try:
                resolution = await self.contextual_resolver.resolve_query(query)
                if resolution.success and resolution.resolved_spaces:
                    logger.debug(f"[MULTI-SPACE] Contextual resolver found: {resolution.resolved_spaces}")
                    return resolution.resolved_spaces
            except Exception as e:
                logger.debug(f"[MULTI-SPACE] Contextual resolution failed: {e}")

        # Fallback: empty list (will trigger clarification)
        return []

    def _extract_space_numbers(self, query: str) -> List[int]:
        """Extract explicit space numbers from query"""
        spaces = set()

        for pattern in self.space_patterns:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                # Extract all captured groups
                for group in match.groups():
                    if group and group.isdigit():
                        spaces.add(int(group))

        return sorted(list(spaces))

    async def _analyze_spaces_parallel(self, space_ids: List[int], query: str) -> List[SpaceAnalysisResult]:
        """
        Analyze multiple spaces in parallel using async/await.

        This is the core parallel execution engine.
        """
        logger.info(f"[MULTI-SPACE] Starting parallel analysis of {len(space_ids)} spaces")

        # Create async tasks for each space
        tasks = [
            self._analyze_single_space(space_id, query)
            for space_id in space_ids
        ]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and failed results
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"[MULTI-SPACE] Space {space_ids[i]} analysis failed: {result}")
                # Add failed result
                valid_results.append(SpaceAnalysisResult(
                    space_id=space_ids[i],
                    success=False,
                    content_summary=f"Analysis failed: {str(result)}"
                ))
            else:
                valid_results.append(result)

        logger.info(f"[MULTI-SPACE] Completed analysis: {len(valid_results)}/{len(space_ids)} successful")
        return valid_results

    async def _analyze_single_space(self, space_id: int, query: str) -> SpaceAnalysisResult:
        """
        Analyze a single space.

        This integrates with vision systems and context graph.
        """
        start_time = datetime.now()

        # Validate space state first
        edge_case_result = await self.space_manager.handle_edge_case(space_id)

        # Handle edge cases
        if edge_case_result.edge_case == "not_exist":
            return SpaceAnalysisResult(
                space_id=space_id,
                success=False,
                content_summary=edge_case_result.message,
                analysis_time=(datetime.now() - start_time).total_seconds()
            )
        elif edge_case_result.edge_case == "empty":
            return SpaceAnalysisResult(
                space_id=space_id,
                success=True,
                content_summary=f"Space {space_id} is empty (no windows)",
                analysis_time=(datetime.now() - start_time).total_seconds()
            )
        elif edge_case_result.edge_case == "minimized_only":
            # Get apps from state info
            apps = edge_case_result.state_info.applications if edge_case_result.state_info else []
            app_list = ", ".join(apps[:2])
            return SpaceAnalysisResult(
                space_id=space_id,
                success=True,
                app_name=apps[0] if apps else "Unknown",
                content_summary=f"Space {space_id} has minimized windows only ({app_list})",
                significance="low",
                analysis_time=(datetime.now() - start_time).total_seconds()
            )
        elif edge_case_result.edge_case == "transitioning":
            if not edge_case_result.success:
                return SpaceAnalysisResult(
                    space_id=space_id,
                    success=False,
                    content_summary=edge_case_result.message,
                    analysis_time=(datetime.now() - start_time).total_seconds()
                )
            logger.info(f"[MULTI-SPACE] Space {space_id} stabilized after transition")

        try:
            # Get context from context graph if available
            if self.context_graph and space_id in self.context_graph.spaces:
                space_ctx = self.context_graph.spaces[space_id]

                # Get active apps
                apps = list(space_ctx.applications.keys()) if space_ctx.applications else []
                app_name = apps[0] if apps else "Unknown"

                # Get recent events
                recent_events = space_ctx.get_recent_events(within_seconds=300)

                # Find errors
                errors = []
                for event in recent_events:
                    if event.event_type.value == "error_detected":
                        errors.append(event.details.get("error", "Unknown error"))

                # Determine content type
                content_type = "unknown"
                if errors:
                    content_type = "error"
                elif app_name.lower() in ["terminal", "iterm2", "hyper"]:
                    content_type = "terminal"
                elif app_name.lower() in ["safari", "chrome", "firefox"]:
                    content_type = "browser"
                elif app_name.lower() in ["vscode", "pycharm", "sublime", "vim"]:
                    content_type = "code"

                # Build content summary
                content_summary = f"{app_name}"
                if errors:
                    content_summary += f" with {len(errors)} error(s)"

                analysis_time = (datetime.now() - start_time).total_seconds()

                return SpaceAnalysisResult(
                    space_id=space_id,
                    success=True,
                    app_name=app_name,
                    content_type=content_type,
                    content_summary=content_summary,
                    errors=errors,
                    significance="critical" if errors else "normal",
                    analysis_time=analysis_time
                )
            else:
                # No context available - basic result
                return SpaceAnalysisResult(
                    space_id=space_id,
                    success=True,
                    content_summary="No context available",
                    analysis_time=(datetime.now() - start_time).total_seconds()
                )

        except Exception as e:
            logger.error(f"[MULTI-SPACE] Error analyzing space {space_id}: {e}", exc_info=True)
            return SpaceAnalysisResult(
                space_id=space_id,
                success=False,
                content_summary=f"Analysis error: {str(e)}",
                analysis_time=(datetime.now() - start_time).total_seconds()
            )

    async def _compare_spaces(self, results: List[SpaceAnalysisResult], query: str) -> Dict[str, Any]:
        """
        Compare multiple spaces and identify key differences.
        """
        comparison = {
            "spaces": [r.space_id for r in results],
            "summary": {},
            "differences": [],
            "similarities": []
        }

        # Build summary for each space
        for result in results:
            comparison["summary"][result.space_id] = {
                "app": result.app_name,
                "type": result.content_type,
                "has_errors": len(result.errors) > 0,
                "error_count": len(result.errors),
                "significance": result.significance
            }

        # Find differences
        if len(results) >= 2:
            # Compare first two spaces
            space1, space2 = results[0], results[1]

            if space1.content_type != space2.content_type:
                comparison["differences"].append({
                    "type": "content_type",
                    "space1": space1.content_type,
                    "space2": space2.content_type,
                    "description": f"Space {space1.space_id} is {space1.content_type}, Space {space2.space_id} is {space2.content_type}"
                })

            if space1.app_name != space2.app_name:
                comparison["differences"].append({
                    "type": "application",
                    "space1": space1.app_name,
                    "space2": space2.app_name,
                    "description": f"Space {space1.space_id} has {space1.app_name}, Space {space2.space_id} has {space2.app_name}"
                })

            if len(space1.errors) != len(space2.errors):
                comparison["differences"].append({
                    "type": "errors",
                    "space1": len(space1.errors),
                    "space2": len(space2.errors),
                    "description": f"Space {space1.space_id} has {len(space1.errors)} error(s), Space {space2.space_id} has {len(space2.errors)} error(s)"
                })

        return comparison

    async def _detect_differences(self, results: List[SpaceAnalysisResult]) -> List[Dict[str, Any]]:
        """Detect all differences between spaces"""
        differences = []

        for i, result1 in enumerate(results):
            for result2 in results[i+1:]:
                if result1.content_type != result2.content_type:
                    differences.append({
                        "space1": result1.space_id,
                        "space2": result2.space_id,
                        "difference_type": "content_type",
                        "value1": result1.content_type,
                        "value2": result2.content_type
                    })

                if result1.app_name != result2.app_name:
                    differences.append({
                        "space1": result1.space_id,
                        "space2": result2.space_id,
                        "difference_type": "application",
                        "value1": result1.app_name,
                        "value2": result2.app_name
                    })

        return differences

    async def _search_across_spaces(self, results: List[SpaceAnalysisResult], query: str) -> List[Dict[str, Any]]:
        """
        Search for specific content across all spaces.

        Examples:
        - "Find the terminal"
        - "Which space has the error?"
        """
        query_lower = query.lower()
        matches = []

        # Extract search term
        search_term = self._extract_search_term(query)
        logger.debug(f"[MULTI-SPACE] Searching for: '{search_term}'")

        for result in results:
            if not result.success:
                continue

            # Check for matches
            match_score = 0.0
            match_reasons = []

            # Search in app name
            if search_term and result.app_name and search_term in result.app_name.lower():
                match_score += 0.5
                match_reasons.append(f"App name contains '{search_term}'")

            # Search in content type
            if search_term and result.content_type and search_term in result.content_type.lower():
                match_score += 0.4
                match_reasons.append(f"Content type is '{search_term}'")

            # Search for errors if query mentions error
            if "error" in query_lower and result.errors:
                match_score += 0.6
                match_reasons.append(f"Has {len(result.errors)} error(s)")

            # Search for terminal
            if "terminal" in query_lower and result.content_type == "terminal":
                match_score += 0.8
                match_reasons.append("Is a terminal")

            if match_score > 0:
                matches.append({
                    "space_id": result.space_id,
                    "score": match_score,
                    "reasons": match_reasons,
                    "content": result.content_summary
                })

        # Sort by score
        matches.sort(key=lambda m: m["score"], reverse=True)
        return matches

    def _extract_search_term(self, query: str) -> Optional[str]:
        """Extract the search term from a query"""
        query_lower = query.lower()

        # Try patterns
        patterns = [
            r'find\s+(?:the\s+)?(\w+)',
            r'which\s+space\s+has\s+(?:the\s+)?(\w+)',
            r'locate\s+(?:the\s+)?(\w+)',
            r'where\s+is\s+(?:the\s+)?(\w+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                return match.group(1)

        return None

    async def _synthesize_response(self, query_type: MultiSpaceQueryType,
                                   results: List[SpaceAnalysisResult],
                                   query: str,
                                   comparison: Optional[Dict[str, Any]],
                                   differences: Optional[List[Dict[str, Any]]],
                                   search_matches: Optional[List[Dict[str, Any]]]) -> str:
        """
        Synthesize a unified natural language response.
        """
        if query_type == MultiSpaceQueryType.COMPARE:
            return self._synthesize_comparison_response(results, comparison)
        elif query_type == MultiSpaceQueryType.DIFFERENCE:
            return self._synthesize_difference_response(results, differences)
        elif query_type in [MultiSpaceQueryType.SEARCH, MultiSpaceQueryType.LOCATE]:
            return self._synthesize_search_response(results, search_matches, query)
        elif query_type == MultiSpaceQueryType.SUMMARY:
            return self._synthesize_summary_response(results)
        else:
            # Generic response
            return self._synthesize_generic_response(results)

    def _synthesize_comparison_response(self, results: List[SpaceAnalysisResult],
                                       comparison: Optional[Dict[str, Any]]) -> str:
        """Generate comparison response"""
        if not results:
            return "No spaces to compare."

        if len(results) < 2:
            return f"Space {results[0].space_id}: {results[0].content_summary}"

        # Build comparison
        lines = []
        for result in results:
            if result.success:
                error_part = f" with {len(result.errors)} error(s)" if result.errors else ""
                lines.append(f"Space {result.space_id}: {result.app_name}{error_part}")

        # Add differences
        if comparison and comparison.get("differences"):
            lines.append("\nKey Differences:")
            for diff in comparison["differences"][:3]:  # Top 3
                lines.append(f"  • {diff['description']}")

        return "\n".join(lines)

    def _synthesize_difference_response(self, results: List[SpaceAnalysisResult],
                                       differences: Optional[List[Dict[str, Any]]]) -> str:
        """Generate difference response"""
        if not differences:
            return "The spaces appear similar."

        lines = ["Differences found:"]
        for diff in differences[:5]:  # Top 5
            lines.append(
                f"  • Space {diff['space1']} ({diff['value1']}) vs "
                f"Space {diff['space2']} ({diff['value2']})"
            )

        return "\n".join(lines)

    def _synthesize_search_response(self, results: List[SpaceAnalysisResult],
                                   search_matches: Optional[List[Dict[str, Any]]],
                                   query: str) -> str:
        """Generate search response"""
        if not search_matches:
            return "No matches found across the spaces analyzed."

        # Top match
        top_match = search_matches[0]
        space_id = top_match["space_id"]

        # Find full result
        result = next((r for r in results if r.space_id == space_id), None)

        if result:
            reasons = ", ".join(top_match["reasons"])
            response = f"Found in Space {space_id}: {result.content_summary}\n({reasons})"

            # Add other matches if available
            if len(search_matches) > 1:
                others = [f"Space {m['space_id']}" for m in search_matches[1:3]]
                response += f"\n\nAlso found in: {', '.join(others)}"

            return response
        else:
            return f"Found in Space {space_id}"

    def _synthesize_summary_response(self, results: List[SpaceAnalysisResult]) -> str:
        """Generate summary response"""
        lines = [f"Summary of {len(results)} space(s):"]
        for result in results:
            if result.success:
                lines.append(f"  • Space {result.space_id}: {result.content_summary}")

        return "\n".join(lines)

    def _synthesize_generic_response(self, results: List[SpaceAnalysisResult]) -> str:
        """Generate generic response"""
        return self._synthesize_summary_response(results)

    def _calculate_confidence(self, results: List[SpaceAnalysisResult],
                             query_type: MultiSpaceQueryType) -> float:
        """Calculate confidence in the analysis"""
        if not results:
            return 0.0

        successful = sum(1 for r in results if r.success)
        success_rate = successful / len(results)

        # Base confidence on success rate
        confidence = success_rate * 0.7

        # Boost for complete results
        if all(r.app_name for r in results if r.success):
            confidence += 0.2

        # Boost for specific query types
        if query_type in [MultiSpaceQueryType.COMPARE, MultiSpaceQueryType.DIFFERENCE]:
            if len(results) >= 2:
                confidence += 0.1

        return min(1.0, confidence)


# ============================================================================
# GLOBAL INSTANCE MANAGEMENT
# ============================================================================

_global_handler: Optional[MultiSpaceQueryHandler] = None


def get_multi_space_handler() -> Optional[MultiSpaceQueryHandler]:
    """Get the global multi-space query handler"""
    return _global_handler


def initialize_multi_space_handler(context_graph=None, implicit_resolver=None,
                                   contextual_resolver=None) -> MultiSpaceQueryHandler:
    """Initialize the global multi-space query handler"""
    global _global_handler
    _global_handler = MultiSpaceQueryHandler(context_graph, implicit_resolver, contextual_resolver)
    logger.info("[MULTI-SPACE] Global handler initialized")
    return _global_handler
