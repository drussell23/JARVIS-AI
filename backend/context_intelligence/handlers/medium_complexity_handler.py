"""
Medium Complexity Handler for JARVIS
=====================================

Handles Level 2 (Moderate) complexity queries:
- Multiple spaces or context
- Comparisons between spaces
- Cross-space searches

Processing Pipeline:
1. Parse multiple spaces
2. Capture in parallel
3. Run OCR on each (with intelligent fallbacks)
4. Synthesize comparison/results

Latency: 3-6s
API Calls: 2-6 (depending on spaces)

Examples:
- "Compare space 3 and space 5"
- "Which space has the terminal?"
- "Show me spaces 1, 2, 3"
- "What's different between those spaces?"

Uses:
- CaptureStrategyManager for intelligent screen capture
- OCRStrategyManager for intelligent OCR with fallbacks
- ImplicitReferenceResolver for entity resolution
- Error Handling Matrix for graceful degradation

Author: Derek Russell
Date: 2025-10-19
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

# Import managers and resolvers
try:
    from context_intelligence.managers import (
        get_capture_strategy_manager,
        get_ocr_strategy_manager,
        CaptureStrategyManager,
        OCRStrategyManager
    )
    CAPTURE_STRATEGY_AVAILABLE = True
    OCR_STRATEGY_AVAILABLE = True
except ImportError:
    CAPTURE_STRATEGY_AVAILABLE = False
    OCR_STRATEGY_AVAILABLE = False
    get_capture_strategy_manager = lambda: None
    get_ocr_strategy_manager = lambda: None
    logger.warning("Strategy managers not available")

try:
    from backend.context_intelligence.resolvers import (
        get_implicit_reference_resolver
    )
    IMPLICIT_RESOLVER_AVAILABLE = True
except ImportError:
    IMPLICIT_RESOLVER_AVAILABLE = False
    get_implicit_reference_resolver = lambda: None
    logger.warning("ImplicitReferenceResolver not available")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class MediumQueryType(Enum):
    """Medium query types"""
    COMPARISON = "comparison"          # Compare multiple spaces
    MULTI_SPACE = "multi_space"       # Show multiple spaces
    CROSS_SPACE_SEARCH = "cross_space_search"  # Find entity across spaces


@dataclass
class SpaceCapture:
    """Captured space data"""
    space_id: int
    success: bool
    image: Optional[Any] = None
    ocr_text: Optional[str] = None
    ocr_confidence: float = 0.0
    capture_method: str = "unknown"
    ocr_method: str = "unknown"
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MediumQueryResult:
    """Result from medium complexity query"""
    success: bool
    query_type: MediumQueryType
    spaces_processed: List[int]
    captures: List[SpaceCapture]
    synthesis: str  # Synthesized result/comparison
    execution_time: float
    total_api_calls: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# MEDIUM COMPLEXITY HANDLER
# ============================================================================

class MediumComplexityHandler:
    """
    Handles Level 2 (Moderate) complexity queries

    Features:
    - Parallel space capture
    - Intelligent OCR with fallbacks
    - Comparison synthesis
    - Cross-space search
    - Reference resolution
    """

    def __init__(
        self,
        capture_manager: Optional[CaptureStrategyManager] = None,
        ocr_manager: Optional[OCRStrategyManager] = None,
        implicit_resolver: Optional[Any] = None
    ):
        """
        Initialize medium complexity handler

        Args:
            capture_manager: CaptureStrategyManager instance
            ocr_manager: OCRStrategyManager instance
            implicit_resolver: ImplicitReferenceResolver instance
        """
        self.capture_manager = capture_manager or get_capture_strategy_manager()
        self.ocr_manager = ocr_manager or get_ocr_strategy_manager()
        self.implicit_resolver = implicit_resolver or get_implicit_reference_resolver()

        logger.info("[MEDIUM-HANDLER] Initialized")
        logger.info(f"  Capture Manager: {'✅' if self.capture_manager else '❌'}")
        logger.info(f"  OCR Manager: {'✅' if self.ocr_manager else '❌'}")
        logger.info(f"  Implicit Resolver: {'✅' if self.implicit_resolver else '❌'}")

    async def process_query(
        self,
        query: str,
        space_ids: List[int],
        query_type: MediumQueryType,
        context: Optional[Dict[str, Any]] = None
    ) -> MediumQueryResult:
        """
        Process medium complexity query

        Args:
            query: Original user query
            space_ids: List of space IDs to process
            query_type: Type of medium query
            context: Optional context

        Returns:
            MediumQueryResult with captures and synthesis
        """
        start_time = time.time()
        api_calls = 0

        logger.info(f"[MEDIUM-HANDLER] Processing {query_type.value} query: '{query}'")
        logger.info(f"  Spaces: {space_ids}")

        # Step 1: Resolve references if needed
        resolved_query = await self._resolve_references(query, context)
        if resolved_query != query:
            logger.info(f"[MEDIUM-HANDLER] Resolved: '{query}' → '{resolved_query}'")

        # Step 2: Capture all spaces in parallel
        captures = await self._capture_spaces_parallel(space_ids)

        # Count successful captures
        successful_captures = [c for c in captures if c.success]
        logger.info(f"[MEDIUM-HANDLER] Captured {len(successful_captures)}/{len(space_ids)} spaces")

        # Step 3: Run OCR on all captures in parallel
        captures_with_ocr = await self._extract_text_parallel(captures)

        # Count API calls (rough estimate)
        for capture in captures_with_ocr:
            if capture.ocr_method == "claude_vision":
                api_calls += 1
            elif capture.capture_method != "cached":
                api_calls += 0.5  # Partial API call for capture

        # Step 4: Synthesize results based on query type
        synthesis = await self._synthesize_results(
            resolved_query,
            query_type,
            captures_with_ocr,
            context
        )

        execution_time = time.time() - start_time

        logger.info(
            f"[MEDIUM-HANDLER] ✅ Completed in {execution_time:.2f}s "
            f"(api_calls={api_calls:.1f})"
        )

        return MediumQueryResult(
            success=len(successful_captures) > 0,
            query_type=query_type,
            spaces_processed=space_ids,
            captures=captures_with_ocr,
            synthesis=synthesis,
            execution_time=execution_time,
            total_api_calls=int(api_calls),
            metadata={
                "original_query": query,
                "resolved_query": resolved_query,
                "successful_captures": len(successful_captures),
                "failed_captures": len(space_ids) - len(successful_captures)
            }
        )

    async def _resolve_references(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Resolve ambiguous references in query"""
        if not self.implicit_resolver:
            return query

        try:
            resolved = await asyncio.to_thread(
                self.implicit_resolver.resolve_query,
                query,
                context or {}
            )
            return resolved if resolved else query
        except Exception as e:
            logger.error(f"Reference resolution failed: {e}")
            return query

    async def _capture_spaces_parallel(
        self,
        space_ids: List[int]
    ) -> List[SpaceCapture]:
        """
        Capture multiple spaces in parallel

        Uses CaptureStrategyManager for intelligent fallbacks:
        - Try window capture
        - Fallback to space capture
        - Fallback to cache
        - Return error
        """
        logger.info(f"[MEDIUM-HANDLER] Capturing {len(space_ids)} spaces in parallel")

        if self.capture_manager:
            # Use capture strategy manager with fallbacks
            tasks = [
                self._capture_space_with_strategy(space_id)
                for space_id in space_ids
            ]
        else:
            # Fallback to basic capture
            tasks = [
                self._capture_space_basic(space_id)
                for space_id in space_ids
            ]

        captures = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed captures
        results = []
        for i, capture in enumerate(captures):
            if isinstance(capture, Exception):
                results.append(SpaceCapture(
                    space_id=space_ids[i],
                    success=False,
                    error=str(capture)
                ))
            else:
                results.append(capture)

        return results

    async def _capture_space_with_strategy(self, space_id: int) -> SpaceCapture:
        """Capture space using CaptureStrategyManager"""
        try:
            success, image, message = await self.capture_manager.capture_with_fallbacks(
                space_id=space_id,
                window_id=None,
                window_capture_func=None,  # Would need actual capture functions
                space_capture_func=None,
                cache_max_age=60.0
            )

            return SpaceCapture(
                space_id=space_id,
                success=success,
                image=image,
                capture_method=message.split("via")[-1].strip() if "via" in message else "unknown",
                metadata={"message": message}
            )

        except Exception as e:
            logger.error(f"Capture failed for space {space_id}: {e}")
            return SpaceCapture(
                space_id=space_id,
                success=False,
                error=str(e)
            )

    async def _capture_space_basic(self, space_id: int) -> SpaceCapture:
        """Basic space capture fallback (legacy)"""
        logger.warning(f"[MEDIUM-HANDLER] Using basic capture for space {space_id}")

        # Placeholder - would integrate with actual capture system
        return SpaceCapture(
            space_id=space_id,
            success=False,
            error="Capture managers not available"
        )

    async def _extract_text_parallel(
        self,
        captures: List[SpaceCapture]
    ) -> List[SpaceCapture]:
        """
        Extract text from all captures in parallel

        Uses OCRStrategyManager for intelligent fallbacks:
        - Try Claude Vision
        - Fallback to cache
        - Fallback to Tesseract
        - Return metadata
        """
        logger.info(f"[MEDIUM-HANDLER] Extracting text from {len(captures)} captures")

        # Only process successful captures
        successful_captures = [c for c in captures if c.success and c.image]

        if not successful_captures:
            logger.warning("[MEDIUM-HANDLER] No successful captures to process")
            return captures

        if self.ocr_manager:
            # Use OCR strategy manager with fallbacks
            tasks = [
                self._extract_text_with_strategy(capture)
                for capture in successful_captures
            ]
        else:
            # Fallback to basic OCR
            tasks = [
                self._extract_text_basic(capture)
                for capture in successful_captures
            ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Update captures with OCR results
        capture_map = {c.space_id: c for c in captures}

        for i, result in enumerate(results):
            space_id = successful_captures[i].space_id

            if isinstance(result, Exception):
                capture_map[space_id].ocr_text = ""
                capture_map[space_id].ocr_confidence = 0.0
                capture_map[space_id].error = str(result)
            else:
                text, confidence, method = result
                capture_map[space_id].ocr_text = text
                capture_map[space_id].ocr_confidence = confidence
                capture_map[space_id].ocr_method = method

        return list(capture_map.values())

    async def _extract_text_with_strategy(
        self,
        capture: SpaceCapture
    ) -> Tuple[str, float, str]:
        """Extract text using OCRStrategyManager"""
        try:
            # Save image to temp file for OCR
            import tempfile
            from PIL import Image
            from pathlib import Path

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name

                # Convert image to PIL if needed
                if not isinstance(capture.image, Image.Image):
                    # Assume numpy array
                    img = Image.fromarray(capture.image)
                else:
                    img = capture.image

                img.save(tmp_path)

            try:
                result = await self.ocr_manager.extract_text_with_fallbacks(
                    image_path=tmp_path,
                    cache_max_age=300.0
                )

                return (result.text, result.confidence, result.method)

            finally:
                # Clean up temp file
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"OCR failed for space {capture.space_id}: {e}")
            return ("", 0.0, "failed")

    async def _extract_text_basic(
        self,
        capture: SpaceCapture
    ) -> Tuple[str, float, str]:
        """Basic OCR fallback (legacy)"""
        logger.warning(f"[MEDIUM-HANDLER] Using basic OCR for space {capture.space_id}")
        return ("", 0.0, "unavailable")

    async def _synthesize_results(
        self,
        query: str,
        query_type: MediumQueryType,
        captures: List[SpaceCapture],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Synthesize results based on query type

        Args:
            query: Resolved query
            query_type: Type of query
            captures: All space captures with OCR
            context: Optional context

        Returns:
            Synthesized result string
        """
        logger.info(f"[MEDIUM-HANDLER] Synthesizing {query_type.value} results")

        if query_type == MediumQueryType.COMPARISON:
            return await self._synthesize_comparison(query, captures, context)
        elif query_type == MediumQueryType.CROSS_SPACE_SEARCH:
            return await self._synthesize_search(query, captures, context)
        else:  # MULTI_SPACE
            return await self._synthesize_multi_space(query, captures, context)

    async def _synthesize_comparison(
        self,
        query: str,
        captures: List[SpaceCapture],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Synthesize comparison between spaces"""
        successful = [c for c in captures if c.success and c.ocr_text]

        if len(successful) < 2:
            return "❌ Unable to compare: need at least 2 successful captures"

        # Simple comparison based on text differences
        comparison_parts = []
        comparison_parts.append(f"**Comparison of {len(successful)} spaces:**\n")

        for capture in successful:
            text_preview = capture.ocr_text[:200] + "..." if len(capture.ocr_text) > 200 else capture.ocr_text
            comparison_parts.append(
                f"**Space {capture.space_id}:**\n"
                f"  - Text length: {len(capture.ocr_text)} characters\n"
                f"  - OCR confidence: {capture.ocr_confidence:.2f}\n"
                f"  - Method: {capture.ocr_method}\n"
                f"  - Preview: {text_preview}\n"
            )

        # Find differences
        if len(successful) == 2:
            space1, space2 = successful[0], successful[1]

            # Simple text difference
            if space1.ocr_text == space2.ocr_text:
                comparison_parts.append("\n✅ **Text is identical** between both spaces")
            else:
                comparison_parts.append(
                    f"\n⚠️ **Text differs** between spaces "
                    f"({abs(len(space1.ocr_text) - len(space2.ocr_text))} character difference)"
                )

        return "\n".join(comparison_parts)

    async def _synthesize_search(
        self,
        query: str,
        captures: List[SpaceCapture],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Synthesize cross-space search results"""
        successful = [c for c in captures if c.success and c.ocr_text]

        if not successful:
            return "❌ Unable to search: no successful captures"

        # Extract search term from query
        import re
        # Simple heuristic: look for quoted terms or last word
        quoted = re.findall(r'"([^"]+)"', query)
        if quoted:
            search_term = quoted[0]
        else:
            # Use last significant word
            words = query.lower().split()
            search_term = words[-1] if words else ""

        logger.info(f"[MEDIUM-HANDLER] Searching for: '{search_term}'")

        # Search in all spaces
        matches = []
        for capture in successful:
            if search_term.lower() in capture.ocr_text.lower():
                # Find context around match
                idx = capture.ocr_text.lower().find(search_term.lower())
                start = max(0, idx - 50)
                end = min(len(capture.ocr_text), idx + len(search_term) + 50)
                context_text = capture.ocr_text[start:end]

                matches.append({
                    "space_id": capture.space_id,
                    "context": context_text,
                    "confidence": capture.ocr_confidence
                })

        if not matches:
            return f"❌ **'{search_term}' not found** in any of the {len(successful)} spaces searched"

        # Build result
        result_parts = []
        result_parts.append(f"✅ **Found '{search_term}' in {len(matches)} space(s):**\n")

        for match in matches:
            result_parts.append(
                f"**Space {match['space_id']}:**\n"
                f"  - Context: ...{match['context']}...\n"
                f"  - Confidence: {match['confidence']:.2f}\n"
            )

        return "\n".join(result_parts)

    async def _synthesize_multi_space(
        self,
        query: str,
        captures: List[SpaceCapture],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Synthesize multi-space view results"""
        successful = [c for c in captures if c.success]

        if not successful:
            return "❌ Unable to process: no successful captures"

        result_parts = []
        result_parts.append(f"**Processed {len(successful)} space(s):**\n")

        for capture in successful:
            text_length = len(capture.ocr_text) if capture.ocr_text else 0
            result_parts.append(
                f"**Space {capture.space_id}:**\n"
                f"  - Status: ✅ Captured\n"
                f"  - Text: {text_length} characters\n"
                f"  - Confidence: {capture.ocr_confidence:.2f}\n"
                f"  - Methods: {capture.capture_method} + {capture.ocr_method}\n"
            )

        # Add failed captures
        failed = [c for c in captures if not c.success]
        if failed:
            result_parts.append(f"\n❌ **Failed to capture {len(failed)} space(s):**")
            for capture in failed:
                result_parts.append(f"  - Space {capture.space_id}: {capture.error or 'Unknown error'}")

        return "\n".join(result_parts)


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_global_handler: Optional[MediumComplexityHandler] = None


def get_medium_complexity_handler() -> Optional[MediumComplexityHandler]:
    """Get the global medium complexity handler instance"""
    return _global_handler


def initialize_medium_complexity_handler(
    capture_manager: Optional[CaptureStrategyManager] = None,
    ocr_manager: Optional[OCRStrategyManager] = None,
    implicit_resolver: Optional[Any] = None
) -> MediumComplexityHandler:
    """Initialize the global medium complexity handler"""
    global _global_handler
    _global_handler = MediumComplexityHandler(
        capture_manager=capture_manager,
        ocr_manager=ocr_manager,
        implicit_resolver=implicit_resolver
    )
    logger.info("[MEDIUM-HANDLER] Global instance initialized")
    return _global_handler
