"""
OCR Adapter for Follow-Up System
Provides unified interface for OCR text extraction.

Uses OCRStrategyManager for intelligent fallbacks:
1. Primary: Claude Vision API
2. Fallback 1: Use cached OCR (if <5min old)
3. Fallback 2: Local OCR (Tesseract)
4. Fallback 3: Return image metadata only
"""
import logging
from pathlib import Path
from typing import Optional
import asyncio

logger = logging.getLogger(__name__)

# Import OCR Strategy Manager
try:
    from context_intelligence.managers import (
        get_ocr_strategy_manager,
        initialize_ocr_strategy_manager
    )
    OCR_STRATEGY_AVAILABLE = True
except ImportError:
    OCR_STRATEGY_AVAILABLE = False
    logger.warning("OCR Strategy Manager not available, falling back to legacy OCR")

# Cache for OCR results (legacy - now using OCRStrategyManager cache)
_ocr_cache: dict[str, str] = {}


async def ocr_text_from_snapshot(snapshot_id: str, use_claude: bool = True) -> str:
    """
    Extract OCR text from a snapshot ID.

    Uses OCRStrategyManager for intelligent fallbacks:
    1. Claude Vision API (if available and use_claude=True)
    2. Cached OCR (if <5min old)
    3. Tesseract OCR
    4. Image metadata

    Args:
        snapshot_id: Snapshot identifier (could be file path, hash, or UUID)
        use_claude: Use Claude Vision API as primary method (default: True)

    Returns:
        Extracted text, or empty string if extraction fails
    """
    # Try to resolve snapshot ID to file path
    snapshot_path = _resolve_snapshot_path(snapshot_id)

    if not snapshot_path or not snapshot_path.exists():
        logger.warning(f"[OCR] Snapshot not found: {snapshot_id}")
        return ""

    try:
        # Use OCR Strategy Manager if available
        if OCR_STRATEGY_AVAILABLE and use_claude:
            manager = get_ocr_strategy_manager()

            if manager:
                logger.info(f"[OCR] Using OCR Strategy Manager for {snapshot_id}")

                # Extract text with full fallback chain
                result = await manager.extract_text_with_fallbacks(
                    image_path=str(snapshot_path),
                    cache_max_age=300.0,  # 5 minutes
                    skip_cache=False
                )

                if result.success:
                    logger.info(
                        f"[OCR] ✅ Extracted {len(result.text)} chars using {result.method} "
                        f"(confidence={result.confidence:.2f})"
                    )
                    return result.text
                else:
                    logger.warning(f"[OCR] Strategy manager failed: {result.error}")
                    # Fall through to legacy method

        # Fallback to legacy OCR (Tesseract only)
        logger.info(f"[OCR] Using legacy OCR for {snapshot_id}")

        # Check legacy cache first
        if snapshot_id in _ocr_cache:
            logger.debug(f"[OCR] Legacy cache hit for snapshot {snapshot_id}")
            return _ocr_cache[snapshot_id]

        # Use existing OCR processor
        from backend.vision.ocr_processor import OCRProcessor
        from PIL import Image

        processor = OCRProcessor()

        # Load image
        image = Image.open(snapshot_path)

        # Run OCR
        result = await processor.process_image(image)

        # Extract full text
        ocr_text = result.full_text if result else ""

        # Cache result
        _ocr_cache[snapshot_id] = ocr_text

        # Limit cache size
        if len(_ocr_cache) > 100:
            # Remove oldest entries
            oldest_keys = list(_ocr_cache.keys())[:50]
            for key in oldest_keys:
                del _ocr_cache[key]

        logger.info(f"[OCR] Extracted {len(ocr_text)} chars from {snapshot_id} (legacy)")
        return ocr_text

    except Exception as e:
        logger.error(f"[OCR] Failed to extract text from {snapshot_id}: {e}", exc_info=True)
        return ""


def _resolve_snapshot_path(snapshot_id: str) -> Optional[Path]:
    """
    Resolve snapshot ID to file path.

    Supports:
    - Direct file paths
    - Relative paths (searches common directories)
    - UUIDs (looks in temp/capture directories)
    """
    # Try as direct path
    path = Path(snapshot_id)
    if path.exists():
        return path

    # Try common snapshot directories
    common_dirs = [
        Path.home() / "Library" / "Application Support" / "JARVIS" / "screenshots",
        Path.home() / ".jarvis" / "screenshots",
        Path("/tmp") / "jarvis_screenshots",
        Path.cwd() / "screenshots",
    ]

    for base_dir in common_dirs:
        if not base_dir.exists():
            continue

        # Try exact match
        candidate = base_dir / snapshot_id
        if candidate.exists():
            return candidate

        # Try with common extensions
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = base_dir / f"{snapshot_id}{ext}"
            if candidate.exists():
                return candidate

    logger.warning(f"[OCR] Could not resolve snapshot path: {snapshot_id}")
    return None


def clear_ocr_cache():
    """Clear the OCR cache."""
    global _ocr_cache
    _ocr_cache.clear()
    logger.info("[OCR] Cache cleared")
