"""
OCR Adapter for Follow-Up System
Provides unified interface for OCR text extraction.
"""
import logging
from pathlib import Path
from typing import Optional
import asyncio

logger = logging.getLogger(__name__)

# Cache for OCR results
_ocr_cache: dict[str, str] = {}


async def ocr_text_from_snapshot(snapshot_id: str) -> str:
    """
    Extract OCR text from a snapshot ID.

    Args:
        snapshot_id: Snapshot identifier (could be file path, hash, or UUID)

    Returns:
        Extracted text, or empty string if extraction fails
    """
    # Check cache first
    if snapshot_id in _ocr_cache:
        logger.debug(f"[OCR] Cache hit for snapshot {snapshot_id}")
        return _ocr_cache[snapshot_id]

    try:
        # Try to resolve snapshot ID to file path
        snapshot_path = _resolve_snapshot_path(snapshot_id)

        if not snapshot_path or not snapshot_path.exists():
            logger.warning(f"[OCR] Snapshot not found: {snapshot_id}")
            return ""

        # Use existing OCR processor
        from backend.vision.ocr_processor import OCRProcessor
        from PIL import Image

        processor = OCRProcessor()

        # Load image
        image = Image.open(snapshot_path)

        # Run OCR in thread pool (tesseract is blocking)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, processor.process_image, image
        )

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

        logger.info(f"[OCR] Extracted {len(ocr_text)} chars from {snapshot_id}")
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
