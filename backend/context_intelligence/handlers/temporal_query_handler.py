"""
Temporal Query Handler - Time-Based Intelligence & Change Detection
===================================================================

Handles temporal queries like:
- ❓ "What changed in space 3?"
- ❓ "Has the error been fixed?"
- ❓ "What's new in the last 5 minutes?"
- ❓ "When did this error first appear?"

Features:
✅ Screenshot caching with timestamps
✅ Image diffing to detect changes
✅ Error appearance/resolution tracking
✅ Integration with ImplicitReferenceResolver for "the error", "it", etc.
✅ Integration with TemporalContextEngine for time-series data
✅ Dynamic, robust, async with zero hardcoding

Architecture:

    User Query → ImplicitReferenceResolver → TemporalQueryHandler
         ↓               ↓                            ↓
    "what changed?"  Resolve refs          Classify query type
         ↓               ↓                            ↓
    "the error"     → "error #1234"        CHANGE_DETECTION
         ↓               ↓                            ↓
    space 3         → space_id: 3          Get time range
         ↓               ↓                            ↓
    Resolved Query  Screenshots          Diff images
         ↓               ↓                            ↓
    "What changed   Events over time      Detect changes
     in space 3?"        ↓                            ↓
         └───────────────┴────────────────────────→ Response
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
import numpy as np
from collections import deque, defaultdict
import hashlib
import json

# Window capture manager
from backend.context_intelligence.managers.window_capture_manager import (
    get_window_capture_manager,
    CaptureStatus
)

# Image processing
try:
    from PIL import Image
    import imagehash
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    logging.warning("PIL not available - image diffing disabled")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available - advanced image diffing disabled")

logger = logging.getLogger(__name__)


# ============================================================================
# TEMPORAL QUERY TYPES
# ============================================================================

class TemporalQueryType(Enum):
    """Types of temporal queries"""
    CHANGE_DETECTION = auto()      # "What changed?"
    ERROR_TRACKING = auto()        # "Has the error been fixed?"
    TIMELINE = auto()              # "What's new in last 5 minutes?"
    FIRST_APPEARANCE = auto()      # "When did this first appear?"
    LAST_OCCURRENCE = auto()       # "When did I last see X?"
    COMPARISON = auto()            # "How is this different from before?"
    TREND_ANALYSIS = auto()        # "Is CPU usage increasing?"
    STATE_HISTORY = auto()         # "Show me history of space 3"


class ChangeType(Enum):
    """Types of changes detected"""
    CONTENT_CHANGE = "content_change"        # Text/UI content changed
    LAYOUT_CHANGE = "layout_change"          # UI layout changed
    ERROR_APPEARED = "error_appeared"        # New error
    ERROR_RESOLVED = "error_resolved"        # Error fixed
    WINDOW_ADDED = "window_added"            # New window
    WINDOW_REMOVED = "window_removed"        # Window closed
    VALUE_CHANGED = "value_changed"          # Numeric value changed
    STATUS_CHANGED = "status_changed"        # Status indicator changed
    NO_CHANGE = "no_change"                  # Nothing changed


@dataclass
class TimeRange:
    """Represents a time range for queries"""
    start: datetime
    end: datetime

    @property
    def duration(self) -> timedelta:
        return self.end - self.start

    @property
    def duration_seconds(self) -> float:
        return self.duration.total_seconds()

    @classmethod
    def from_natural_language(cls, text: str, reference_time: Optional[datetime] = None) -> 'TimeRange':
        """Parse natural language time references"""
        now = reference_time or datetime.now()

        # Parse common patterns
        text_lower = text.lower()

        # "last X minutes/hours/days"
        if "last" in text_lower:
            if "minute" in text_lower:
                # Extract number
                match = re.search(r'(\d+)\s*minute', text_lower)
                minutes = int(match.group(1)) if match else 5
                start = now - timedelta(minutes=minutes)
            elif "hour" in text_lower:
                match = re.search(r'(\d+)\s*hour', text_lower)
                hours = int(match.group(1)) if match else 1
                start = now - timedelta(hours=hours)
            elif "day" in text_lower:
                match = re.search(r'(\d+)\s*day', text_lower)
                days = int(match.group(1)) if match else 1
                start = now - timedelta(days=days)
            else:
                start = now - timedelta(minutes=5)  # Default

        # "since X minutes/hours ago"
        elif "ago" in text_lower:
            match = re.search(r'(\d+)\s*(\w+)\s*ago', text_lower)
            if match:
                value = int(match.group(1))
                unit = match.group(2)
                if "minute" in unit:
                    start = now - timedelta(minutes=value)
                elif "hour" in unit:
                    start = now - timedelta(hours=value)
                elif "day" in unit:
                    start = now - timedelta(days=value)
                else:
                    start = now - timedelta(minutes=5)
            else:
                start = now - timedelta(minutes=5)

        # "recently" / "just now"
        elif any(word in text_lower for word in ["recently", "just now", "latest"]):
            start = now - timedelta(minutes=2)

        # "today"
        elif "today" in text_lower:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Default: last 5 minutes
        else:
            start = now - timedelta(minutes=5)

        return cls(start=start, end=now)


@dataclass
class ScreenshotCache:
    """Cached screenshot with metadata"""
    screenshot_id: str
    timestamp: datetime
    space_id: Optional[int]
    app_id: Optional[str]
    image_path: Path
    image_hash: str  # Perceptual hash for quick comparison
    ocr_text: Optional[str] = None
    detected_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectedChange:
    """Represents a detected change between two screenshots"""
    change_type: ChangeType
    timestamp: datetime
    space_id: Optional[int]
    description: str
    confidence: float
    before_screenshot_id: Optional[str] = None
    after_screenshot_id: Optional[str] = None
    diff_regions: List[Tuple[int, int, int, int]] = field(default_factory=list)  # (x, y, w, h)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalQueryResult:
    """Result of a temporal query"""
    query_type: TemporalQueryType
    time_range: TimeRange
    changes: List[DetectedChange]
    summary: str
    timeline: List[Dict[str, Any]]
    screenshots: List[ScreenshotCache]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# SCREENSHOT MANAGER
# ============================================================================

class ScreenshotManager:
    """Manages screenshot caching and retrieval"""

    def __init__(self, cache_dir: Path = Path("/tmp/jarvis_screenshots")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache (last 100 screenshots)
        self.screenshot_cache: deque[ScreenshotCache] = deque(maxlen=100)
        self.screenshot_index: Dict[str, ScreenshotCache] = {}

        # Space-based index
        self.space_screenshots: Dict[int, List[str]] = defaultdict(list)

        # Window capture manager integration
        self.window_capture_manager = get_window_capture_manager()

        # Load existing cache
        self._load_cache_index()

    async def capture_screenshot(
        self,
        space_id: Optional[int] = None,
        window_id: Optional[int] = None,
        app_id: Optional[str] = None,
        ocr_text: Optional[str] = None,
        detected_errors: Optional[List[str]] = None
    ) -> Optional[ScreenshotCache]:
        """
        Capture and cache a screenshot with robust edge case handling.

        Args:
            space_id: Optional space ID for context
            window_id: Optional specific window ID to capture
            app_id: Optional app ID
            ocr_text: Optional OCR text
            detected_errors: Optional list of detected errors

        Returns:
            ScreenshotCache entry or None if capture failed
        """
        import uuid

        screenshot_id = str(uuid.uuid4())
        timestamp = datetime.now()

        # Determine output path
        output_path = str(self.cache_dir / f"{screenshot_id}.png")

        # Capture using WindowCaptureManager if window_id provided
        if window_id is not None:
            logger.info(f"[SCREENSHOT-MANAGER] Capturing window {window_id}")
            capture_result = await self.window_capture_manager.capture_window(
                window_id=window_id,
                output_path=output_path,
                space_id=space_id,
                use_fallback=True
            )

            if not capture_result.success:
                logger.error(f"[SCREENSHOT-MANAGER] Window capture failed: {capture_result.error}")
                return None

            image_path = Path(capture_result.image_path)

            # Add capture metadata
            metadata = {
                "capture_status": capture_result.status.value,
                "original_size": capture_result.original_size,
                "resized_size": capture_result.resized_size,
                "window_id": window_id,
                "fallback_used": capture_result.fallback_window_id is not None
            }
            if capture_result.fallback_window_id:
                metadata["fallback_window_id"] = capture_result.fallback_window_id

        else:
            # Fallback to full screen capture (legacy behavior)
            logger.info(f"[SCREENSHOT-MANAGER] Capturing full screen for space {space_id}")
            try:
                import pyautogui
                screenshot = pyautogui.screenshot()
                screenshot.save(output_path)
                image_path = Path(output_path)
                metadata = {"capture_type": "full_screen"}
            except Exception as e:
                logger.error(f"[SCREENSHOT-MANAGER] Full screen capture failed: {e}")
                return None

        # Calculate perceptual hash
        if PILLOW_AVAILABLE:
            try:
                img = Image.open(image_path)
                image_hash = str(imagehash.average_hash(img))
            except Exception as e:
                logger.warning(f"[SCREENSHOT-MANAGER] Failed to calculate perceptual hash: {e}")
                with open(image_path, 'rb') as f:
                    image_hash = hashlib.md5(f.read()).hexdigest()
        else:
            # Fallback to simple hash
            with open(image_path, 'rb') as f:
                image_hash = hashlib.md5(f.read()).hexdigest()

        # Create cache entry
        cache_entry = ScreenshotCache(
            screenshot_id=screenshot_id,
            timestamp=timestamp,
            space_id=space_id,
            app_id=app_id,
            image_path=image_path,
            image_hash=image_hash,
            ocr_text=ocr_text,
            detected_errors=detected_errors or [],
            metadata=metadata
        )

        # Add to cache
        self.screenshot_cache.append(cache_entry)
        self.screenshot_index[screenshot_id] = cache_entry

        if space_id is not None:
            self.space_screenshots[space_id].append(screenshot_id)
            # Keep only last 20 per space
            self.space_screenshots[space_id] = self.space_screenshots[space_id][-20:]

        logger.debug(f"[SCREENSHOT-MANAGER] Captured screenshot {screenshot_id} for space {space_id}")

        return cache_entry

    def get_screenshots_in_range(
        self,
        time_range: TimeRange,
        space_id: Optional[int] = None,
        app_id: Optional[str] = None
    ) -> List[ScreenshotCache]:
        """Get screenshots within a time range"""
        screenshots = []

        for screenshot in self.screenshot_cache:
            # Check time range
            if not (time_range.start <= screenshot.timestamp <= time_range.end):
                continue

            # Check space filter
            if space_id is not None and screenshot.space_id != space_id:
                continue

            # Check app filter
            if app_id is not None and screenshot.app_id != app_id:
                continue

            screenshots.append(screenshot)

        return sorted(screenshots, key=lambda s: s.timestamp)

    def get_latest_screenshot(
        self,
        space_id: Optional[int] = None,
        app_id: Optional[str] = None
    ) -> Optional[ScreenshotCache]:
        """Get the most recent screenshot"""
        candidates = []

        for screenshot in reversed(self.screenshot_cache):
            if space_id is not None and screenshot.space_id != space_id:
                continue
            if app_id is not None and screenshot.app_id != app_id:
                continue
            candidates.append(screenshot)

        return candidates[0] if candidates else None

    def _load_cache_index(self) -> None:
        """Load screenshot cache index from disk"""
        index_path = self.cache_dir / "cache_index.json"

        if not index_path.exists():
            return

        try:
            with open(index_path, 'r') as f:
                cache_data = json.load(f)

            for entry_data in cache_data:
                # Reconstruct ScreenshotCache object
                entry = ScreenshotCache(
                    screenshot_id=entry_data['screenshot_id'],
                    timestamp=datetime.fromisoformat(entry_data['timestamp']),
                    space_id=entry_data.get('space_id'),
                    app_id=entry_data.get('app_id'),
                    image_path=Path(entry_data['image_path']),
                    image_hash=entry_data['image_hash'],
                    ocr_text=entry_data.get('ocr_text'),
                    detected_errors=entry_data.get('detected_errors', []),
                    metadata=entry_data.get('metadata', {})
                )

                # Only add if image file still exists
                if entry.image_path.exists():
                    self.screenshot_cache.append(entry)
                    self.screenshot_index[entry.screenshot_id] = entry

                    if entry.space_id is not None:
                        self.space_screenshots[entry.space_id].append(entry.screenshot_id)

        except Exception as e:
            logger.error(f"Failed to load cache index: {e}")

    def _save_cache_index(self) -> None:
        """Save screenshot cache index to disk"""
        index_path = self.cache_dir / "cache_index.json"

        try:
            cache_data = []
            for screenshot in self.screenshot_cache:
                cache_data.append({
                    'screenshot_id': screenshot.screenshot_id,
                    'timestamp': screenshot.timestamp.isoformat(),
                    'space_id': screenshot.space_id,
                    'app_id': screenshot.app_id,
                    'image_path': str(screenshot.image_path),
                    'image_hash': screenshot.image_hash,
                    'ocr_text': screenshot.ocr_text,
                    'detected_errors': screenshot.detected_errors,
                    'metadata': screenshot.metadata
                })

            with open(index_path, 'w') as f:
                json.dump(cache_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")


# ============================================================================
# IMAGE DIFFER
# ============================================================================

class ImageDiffer:
    """Detects changes between screenshots"""

    def __init__(self):
        self.difference_threshold = 0.05  # 5% change threshold

    async def detect_changes(
        self,
        before: ScreenshotCache,
        after: ScreenshotCache
    ) -> List[DetectedChange]:
        """Detect changes between two screenshots"""
        changes = []

        # Quick hash comparison
        if before.image_hash == after.image_hash:
            # Identical images
            return [DetectedChange(
                change_type=ChangeType.NO_CHANGE,
                timestamp=after.timestamp,
                space_id=after.space_id,
                description="No changes detected",
                confidence=1.0
            )]

        # Load images
        try:
            before_img = Image.open(before.image_path)
            after_img = Image.open(after.image_path)
        except Exception as e:
            logger.error(f"Failed to load images for comparison: {e}")
            return []

        # Perceptual hash difference
        if PILLOW_AVAILABLE:
            before_hash = imagehash.average_hash(before_img)
            after_hash = imagehash.average_hash(after_img)
            hash_diff = before_hash - after_hash

            if hash_diff > 10:  # Significant difference
                changes.append(DetectedChange(
                    change_type=ChangeType.CONTENT_CHANGE,
                    timestamp=after.timestamp,
                    space_id=after.space_id,
                    description=f"Significant visual changes detected (hash difference: {hash_diff})",
                    confidence=min(1.0, hash_diff / 20.0),
                    before_screenshot_id=before.screenshot_id,
                    after_screenshot_id=after.screenshot_id
                ))

        # OCR text comparison
        if before.ocr_text and after.ocr_text:
            text_changes = await self._compare_text(before.ocr_text, after.ocr_text)
            changes.extend(text_changes)

        # Error detection comparison
        error_changes = await self._compare_errors(before, after)
        changes.extend(error_changes)

        # Pixel-level comparison (if OpenCV available)
        if CV2_AVAILABLE:
            pixel_changes = await self._compare_pixels(before_img, after_img, after.timestamp, after.space_id)
            changes.extend(pixel_changes)

        return changes

    async def _compare_text(self, before_text: str, after_text: str) -> List[DetectedChange]:
        """Compare OCR text between screenshots"""
        changes = []

        if before_text == after_text:
            return changes

        # Calculate text similarity
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, before_text, after_text).ratio()

        if similarity < 0.8:  # Significant text change
            changes.append(DetectedChange(
                change_type=ChangeType.CONTENT_CHANGE,
                timestamp=datetime.now(),
                space_id=None,
                description=f"Text content changed (similarity: {similarity:.2f})",
                confidence=1.0 - similarity,
                metadata={
                    'before_text_length': len(before_text),
                    'after_text_length': len(after_text),
                    'similarity': similarity
                }
            ))

        return changes

    async def _compare_errors(self, before: ScreenshotCache, after: ScreenshotCache) -> List[DetectedChange]:
        """Compare detected errors between screenshots"""
        changes = []

        before_errors = set(before.detected_errors)
        after_errors = set(after.detected_errors)

        # New errors
        new_errors = after_errors - before_errors
        for error in new_errors:
            changes.append(DetectedChange(
                change_type=ChangeType.ERROR_APPEARED,
                timestamp=after.timestamp,
                space_id=after.space_id,
                description=f"New error appeared: {error}",
                confidence=0.9,
                before_screenshot_id=before.screenshot_id,
                after_screenshot_id=after.screenshot_id,
                metadata={'error': error}
            ))

        # Resolved errors
        resolved_errors = before_errors - after_errors
        for error in resolved_errors:
            changes.append(DetectedChange(
                change_type=ChangeType.ERROR_RESOLVED,
                timestamp=after.timestamp,
                space_id=after.space_id,
                description=f"Error resolved: {error}",
                confidence=0.9,
                before_screenshot_id=before.screenshot_id,
                after_screenshot_id=after.screenshot_id,
                metadata={'error': error}
            ))

        return changes

    async def _compare_pixels(
        self,
        before_img: Image.Image,
        after_img: Image.Image,
        timestamp: datetime,
        space_id: Optional[int]
    ) -> List[DetectedChange]:
        """Pixel-level comparison using OpenCV"""
        changes = []

        try:
            # Convert to numpy arrays
            before_np = np.array(before_img)
            after_np = np.array(after_img)

            # Ensure same dimensions
            if before_np.shape != after_np.shape:
                return changes

            # Calculate difference
            diff = cv2.absdiff(before_np, after_np)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

            # Threshold
            _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Significant changes
            significant_regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    significant_regions.append((x, y, w, h))

            if significant_regions:
                # Calculate total change percentage
                total_pixels = before_np.shape[0] * before_np.shape[1]
                changed_pixels = np.count_nonzero(thresh)
                change_percentage = changed_pixels / total_pixels

                if change_percentage > self.difference_threshold:
                    changes.append(DetectedChange(
                        change_type=ChangeType.LAYOUT_CHANGE,
                        timestamp=timestamp,
                        space_id=space_id,
                        description=f"{len(significant_regions)} regions changed ({change_percentage:.1%} of screen)",
                        confidence=min(1.0, change_percentage / 0.5),
                        diff_regions=significant_regions,
                        metadata={
                            'change_percentage': change_percentage,
                            'region_count': len(significant_regions)
                        }
                    ))

        except Exception as e:
            logger.error(f"Pixel comparison failed: {e}")

        return changes


# ============================================================================
# TEMPORAL QUERY HANDLER
# ============================================================================

class TemporalQueryHandler:
    """Main handler for temporal queries"""

    def __init__(
        self,
        screenshot_manager: Optional[ScreenshotManager] = None,
        image_differ: Optional[ImageDiffer] = None
    ):
        self.screenshot_manager = screenshot_manager or ScreenshotManager()
        self.image_differ = image_differ or ImageDiffer()

        # Will be injected
        self.implicit_resolver = None
        self.temporal_engine = None

    def set_implicit_resolver(self, resolver):
        """Inject ImplicitReferenceResolver"""
        self.implicit_resolver = resolver

    def set_temporal_engine(self, engine):
        """Inject TemporalContextEngine"""
        self.temporal_engine = engine

    async def handle_query(self, query: str, space_id: Optional[int] = None) -> TemporalQueryResult:
        """Handle a temporal query"""

        # Step 1: Classify query type
        query_type = self._classify_query_type(query)

        # Step 2: Extract time range
        time_range = TimeRange.from_natural_language(query)

        # Step 3: Use implicit resolver to resolve references
        resolved_query = await self._resolve_references(query, space_id)

        # Step 4: Execute query based on type
        if query_type == TemporalQueryType.CHANGE_DETECTION:
            result = await self._handle_change_detection(resolved_query, time_range, space_id)

        elif query_type == TemporalQueryType.ERROR_TRACKING:
            result = await self._handle_error_tracking(resolved_query, time_range, space_id)

        elif query_type == TemporalQueryType.TIMELINE:
            result = await self._handle_timeline(resolved_query, time_range, space_id)

        elif query_type == TemporalQueryType.FIRST_APPEARANCE:
            result = await self._handle_first_appearance(resolved_query, time_range, space_id)

        elif query_type == TemporalQueryType.COMPARISON:
            result = await self._handle_comparison(resolved_query, time_range, space_id)

        else:
            result = await self._handle_generic_temporal_query(resolved_query, time_range, space_id)

        return result

    def _classify_query_type(self, query: str) -> TemporalQueryType:
        """Classify the type of temporal query"""
        query_lower = query.lower()

        # Change detection
        if any(word in query_lower for word in ["changed", "change", "different", "new"]):
            return TemporalQueryType.CHANGE_DETECTION

        # Error tracking
        if any(word in query_lower for word in ["error", "fixed", "bug", "issue"]):
            return TemporalQueryType.ERROR_TRACKING

        # Timeline
        if any(word in query_lower for word in ["timeline", "history", "show me"]):
            return TemporalQueryType.TIMELINE

        # First appearance
        if any(word in query_lower for word in ["when", "first", "appeared", "started"]):
            return TemporalQueryType.FIRST_APPEARANCE

        # Comparison
        if any(word in query_lower for word in ["compare", "vs", "versus", "before", "after"]):
            return TemporalQueryType.COMPARISON

        # Default
        return TemporalQueryType.TIMELINE

    async def _resolve_references(self, query: str, space_id: Optional[int]) -> Dict[str, Any]:
        """Use ImplicitReferenceResolver to resolve references in query"""

        if self.implicit_resolver:
            # Use the implicit resolver
            resolution = await self.implicit_resolver.resolve_query(query)

            return {
                'original_query': query,
                'intent': resolution.get('intent'),
                'referents': resolution.get('referents', []),
                'space_id': space_id or resolution.get('space_id'),
                'resolved_entities': resolution.get('entities', {})
            }
        else:
            # Fallback: simple resolution
            return {
                'original_query': query,
                'space_id': space_id,
                'referents': [],
                'resolved_entities': {}
            }

    async def _handle_change_detection(
        self,
        resolved_query: Dict[str, Any],
        time_range: TimeRange,
        space_id: Optional[int]
    ) -> TemporalQueryResult:
        """Handle 'What changed?' queries"""

        # Get screenshots in time range
        screenshots = self.screenshot_manager.get_screenshots_in_range(
            time_range,
            space_id=space_id
        )

        if len(screenshots) < 2:
            return TemporalQueryResult(
                query_type=TemporalQueryType.CHANGE_DETECTION,
                time_range=time_range,
                changes=[],
                summary="Not enough screenshots to detect changes. Need at least 2 screenshots.",
                timeline=[],
                screenshots=screenshots
            )

        # Compare consecutive screenshots
        all_changes = []
        for i in range(len(screenshots) - 1):
            before = screenshots[i]
            after = screenshots[i + 1]

            changes = await self.image_differ.detect_changes(before, after)
            all_changes.extend(changes)

        # Build summary
        summary = self._build_change_summary(all_changes, time_range)

        # Build timeline
        timeline = self._build_timeline(screenshots, all_changes)

        return TemporalQueryResult(
            query_type=TemporalQueryType.CHANGE_DETECTION,
            time_range=time_range,
            changes=all_changes,
            summary=summary,
            timeline=timeline,
            screenshots=screenshots
        )

    async def _handle_error_tracking(
        self,
        resolved_query: Dict[str, Any],
        time_range: TimeRange,
        space_id: Optional[int]
    ) -> TemporalQueryResult:
        """Handle 'Has the error been fixed?' queries"""

        screenshots = self.screenshot_manager.get_screenshots_in_range(
            time_range,
            space_id=space_id
        )

        # Track error states
        error_timeline = []
        for screenshot in screenshots:
            if screenshot.detected_errors:
                error_timeline.append({
                    'timestamp': screenshot.timestamp.isoformat(),
                    'errors': screenshot.detected_errors,
                    'screenshot_id': screenshot.screenshot_id
                })

        # Detect changes
        all_changes = []
        for i in range(len(screenshots) - 1):
            changes = await self.image_differ.detect_changes(screenshots[i], screenshots[i + 1])
            error_changes = [c for c in changes if c.change_type in [ChangeType.ERROR_APPEARED, ChangeType.ERROR_RESOLVED]]
            all_changes.extend(error_changes)

        # Build summary
        if not all_changes:
            summary = "No error state changes detected in the time range."
        else:
            appeared = sum(1 for c in all_changes if c.change_type == ChangeType.ERROR_APPEARED)
            resolved = sum(1 for c in all_changes if c.change_type == ChangeType.ERROR_RESOLVED)

            if resolved > appeared:
                summary = f"✅ Errors have been fixed! {resolved} error(s) resolved, {appeared} new error(s) appeared."
            elif appeared > resolved:
                summary = f"❌ New errors appeared. {appeared} error(s) appeared, {resolved} error(s) resolved."
            else:
                summary = f"Error status mixed: {appeared} error(s) appeared and resolved."

        return TemporalQueryResult(
            query_type=TemporalQueryType.ERROR_TRACKING,
            time_range=time_range,
            changes=all_changes,
            summary=summary,
            timeline=error_timeline,
            screenshots=screenshots
        )

    async def _handle_timeline(
        self,
        resolved_query: Dict[str, Any],
        time_range: TimeRange,
        space_id: Optional[int]
    ) -> TemporalQueryResult:
        """Handle 'What's new in last 5 minutes?' queries"""

        screenshots = self.screenshot_manager.get_screenshots_in_range(
            time_range,
            space_id=space_id
        )

        # Detect all changes
        all_changes = []
        for i in range(len(screenshots) - 1):
            changes = await self.image_differ.detect_changes(screenshots[i], screenshots[i + 1])
            all_changes.extend(changes)

        # Build timeline
        timeline = []
        for screenshot in screenshots:
            timeline.append({
                'timestamp': screenshot.timestamp.isoformat(),
                'space_id': screenshot.space_id,
                'app_id': screenshot.app_id,
                'has_errors': len(screenshot.detected_errors) > 0,
                'error_count': len(screenshot.detected_errors)
            })

        # Build summary
        summary = f"Timeline of {len(screenshots)} screenshot(s) over {time_range.duration_seconds:.0f} seconds. "
        summary += f"{len(all_changes)} change(s) detected."

        return TemporalQueryResult(
            query_type=TemporalQueryType.TIMELINE,
            time_range=time_range,
            changes=all_changes,
            summary=summary,
            timeline=timeline,
            screenshots=screenshots
        )

    async def _handle_first_appearance(
        self,
        resolved_query: Dict[str, Any],
        time_range: TimeRange,
        space_id: Optional[int]
    ) -> TemporalQueryResult:
        """Handle 'When did this first appear?' queries"""

        # Get entity from resolved query
        entity = resolved_query.get('resolved_entities', {}).get('error') or \
                 resolved_query.get('resolved_entities', {}).get('element')

        screenshots = self.screenshot_manager.get_screenshots_in_range(
            time_range,
            space_id=space_id
        )

        # Find first appearance
        first_appearance = None
        for screenshot in screenshots:
            # Check OCR text
            if entity and screenshot.ocr_text and entity.lower() in screenshot.ocr_text.lower():
                first_appearance = screenshot
                break

            # Check errors
            if screenshot.detected_errors:
                for error in screenshot.detected_errors:
                    if entity and entity.lower() in error.lower():
                        first_appearance = screenshot
                        break

        if first_appearance:
            summary = f"First appeared at {first_appearance.timestamp.strftime('%I:%M:%S %p')} ({(datetime.now() - first_appearance.timestamp).total_seconds():.0f} seconds ago)"
        else:
            summary = "Could not determine first appearance in the given time range."

        return TemporalQueryResult(
            query_type=TemporalQueryType.FIRST_APPEARANCE,
            time_range=time_range,
            changes=[],
            summary=summary,
            timeline=[],
            screenshots=screenshots,
            metadata={'first_appearance': first_appearance.screenshot_id if first_appearance else None}
        )

    async def _handle_comparison(
        self,
        resolved_query: Dict[str, Any],
        time_range: TimeRange,
        space_id: Optional[int]
    ) -> TemporalQueryResult:
        """Handle 'How is this different from before?' queries"""

        screenshots = self.screenshot_manager.get_screenshots_in_range(
            time_range,
            space_id=space_id
        )

        if len(screenshots) < 2:
            return TemporalQueryResult(
                query_type=TemporalQueryType.COMPARISON,
                time_range=time_range,
                changes=[],
                summary="Not enough screenshots for comparison.",
                timeline=[],
                screenshots=screenshots
            )

        # Compare first and last screenshot
        before = screenshots[0]
        after = screenshots[-1]

        changes = await self.image_differ.detect_changes(before, after)

        summary = f"Comparison from {before.timestamp.strftime('%I:%M:%S %p')} to {after.timestamp.strftime('%I:%M:%S %p')}: "

        if not changes or all(c.change_type == ChangeType.NO_CHANGE for c in changes):
            summary += "No significant changes detected."
        else:
            summary += f"{len(changes)} change(s) detected - "
            change_types = set(c.change_type for c in changes)
            summary += ", ".join(ct.value.replace('_', ' ') for ct in change_types)

        return TemporalQueryResult(
            query_type=TemporalQueryType.COMPARISON,
            time_range=time_range,
            changes=changes,
            summary=summary,
            timeline=[],
            screenshots=[before, after]
        )

    async def _handle_generic_temporal_query(
        self,
        resolved_query: Dict[str, Any],
        time_range: TimeRange,
        space_id: Optional[int]
    ) -> TemporalQueryResult:
        """Handle generic temporal queries"""

        screenshots = self.screenshot_manager.get_screenshots_in_range(
            time_range,
            space_id=space_id
        )

        return TemporalQueryResult(
            query_type=TemporalQueryType.TIMELINE,
            time_range=time_range,
            changes=[],
            summary=f"Found {len(screenshots)} screenshot(s) in the time range.",
            timeline=[],
            screenshots=screenshots
        )

    def _build_change_summary(self, changes: List[DetectedChange], time_range: TimeRange) -> str:
        """Build a human-readable summary of changes"""

        if not changes:
            return f"No changes detected in the last {time_range.duration_seconds:.0f} seconds."

        # Count by type
        change_counts = defaultdict(int)
        for change in changes:
            change_counts[change.change_type] += 1

        summary_parts = []

        for change_type, count in change_counts.items():
            if change_type == ChangeType.NO_CHANGE:
                continue
            summary_parts.append(f"{count} {change_type.value.replace('_', ' ')}")

        if not summary_parts:
            return f"No significant changes detected."

        summary = f"Detected {len(changes)} change(s): " + ", ".join(summary_parts)

        return summary

    def _build_timeline(self, screenshots: List[ScreenshotCache], changes: List[DetectedChange]) -> List[Dict[str, Any]]:
        """Build a timeline from screenshots and changes"""

        timeline = []

        for screenshot in screenshots:
            entry = {
                'timestamp': screenshot.timestamp.isoformat(),
                'screenshot_id': screenshot.screenshot_id,
                'space_id': screenshot.space_id,
                'app_id': screenshot.app_id,
                'changes': []
            }

            # Find changes associated with this screenshot
            for change in changes:
                if change.after_screenshot_id == screenshot.screenshot_id or \
                   change.before_screenshot_id == screenshot.screenshot_id:
                    entry['changes'].append({
                        'type': change.change_type.value,
                        'description': change.description,
                        'confidence': change.confidence
                    })

            timeline.append(entry)

        return timeline


# ============================================================================
# INITIALIZATION
# ============================================================================

# Global instance
_temporal_query_handler_instance = None

def get_temporal_query_handler() -> TemporalQueryHandler:
    """Get or create the global temporal query handler"""
    global _temporal_query_handler_instance
    if _temporal_query_handler_instance is None:
        _temporal_query_handler_instance = TemporalQueryHandler()
    return _temporal_query_handler_instance


def initialize_temporal_handler(implicit_resolver=None, temporal_engine=None) -> TemporalQueryHandler:
    """Initialize temporal query handler with dependencies"""
    handler = get_temporal_query_handler()

    if implicit_resolver:
        handler.set_implicit_resolver(implicit_resolver)

    if temporal_engine:
        handler.set_temporal_engine(temporal_engine)

    return handler
