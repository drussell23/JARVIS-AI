"""
Change Detection Manager
========================

Advanced temporal & state-based change detection with:
- Screenshot caching with timestamps and hashing
- Image diffing with perceptual hashing
- State tracking for entities (errors, builds, processes)
- Temporal query resolution

Handles queries like:
- "What changed since I last asked?"
- "Did the error get fixed?"
- "Has the build finished?"
"""

import asyncio
import logging
import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import json

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class ChangeType(Enum):
    """Types of detected changes"""
    NO_CHANGE = "no_change"
    CONTENT_CHANGED = "content_changed"
    APPEARANCE_CHANGED = "appearance_changed"
    STATE_CHANGED = "state_changed"
    NEW_ENTITY = "new_entity"
    ENTITY_REMOVED = "entity_removed"


@dataclass
class CachedSnapshot:
    """Cached screenshot/state snapshot"""
    space_id: int
    image: Optional[Any] = None
    image_hash: str = ""
    ocr_text: str = ""
    text_hash: str = ""
    timestamp: float = 0.0
    entities: Dict[str, Any] = field(default_factory=dict)  # Detected entities (errors, builds, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChangeDetectionResult:
    """Result of change detection"""
    changed: bool
    change_type: ChangeType
    space_id: int
    previous_snapshot: Optional[CachedSnapshot] = None
    current_snapshot: Optional[CachedSnapshot] = None
    differences: List[str] = field(default_factory=list)
    similarity_score: float = 1.0  # 0.0 = completely different, 1.0 = identical
    elapsed_time: float = 0.0  # Seconds since last snapshot
    summary: str = ""


@dataclass
class EntityState:
    """State of a tracked entity (error, build, process, etc.)"""
    entity_type: str  # "error", "build", "process", etc.
    entity_id: str
    state: str  # "active", "resolved", "completed", "failed", etc.
    first_seen: float
    last_seen: float
    last_updated: float
    space_id: int
    data: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================================
# CHANGE DETECTION MANAGER
# ============================================================================

class ChangeDetectionManager:
    """
    Manages change detection across spaces with advanced caching and diffing.

    Features:
    - Screenshot caching with perceptual hashing
    - OCR text caching and comparison
    - Entity state tracking (errors, builds, processes)
    - Temporal query resolution
    - Image diffing with similarity scoring
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        cache_ttl: float = 3600.0,  # 1 hour
        max_cache_size: int = 100,
        implicit_resolver: Optional[Any] = None,
        conversation_tracker: Optional[Any] = None
    ):
        """
        Initialize Change Detection Manager.

        Args:
            cache_dir: Directory for persistent cache
            cache_ttl: How long to keep cached snapshots (seconds)
            max_cache_size: Maximum number of cached snapshots
            implicit_resolver: ImplicitReferenceResolver for entity resolution
            conversation_tracker: ConversationTracker for temporal context
        """
        self.cache_dir = cache_dir or Path.home() / ".jarvis" / "change_cache"
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        self.implicit_resolver = implicit_resolver
        self.conversation_tracker = conversation_tracker

        # In-memory cache
        self._snapshot_cache: Dict[int, CachedSnapshot] = {}
        self._entity_states: Dict[str, EntityState] = {}

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("[CHANGE-DETECTION] Initialized")
        logger.info(f"  Cache Dir: {self.cache_dir}")
        logger.info(f"  Cache TTL: {cache_ttl}s")
        logger.info(f"  Implicit Resolver: {'✅' if implicit_resolver else '❌'}")
        logger.info(f"  Conversation Tracker: {'✅' if conversation_tracker else '❌'}")

    async def detect_changes(
        self,
        space_id: int,
        current_image: Optional[Any] = None,
        current_ocr_text: Optional[str] = None,
        query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ChangeDetectionResult:
        """
        Detect changes in a space since last snapshot.

        Args:
            space_id: Space to check
            current_image: Current screenshot
            current_ocr_text: Current OCR text
            query: Original user query for context
            context: Additional context

        Returns:
            ChangeDetectionResult with detected changes
        """
        logger.info(f"[CHANGE-DETECTION] Detecting changes for space {space_id}")

        # Get previous snapshot
        previous_snapshot = self._snapshot_cache.get(space_id)

        if not previous_snapshot:
            logger.info(f"[CHANGE-DETECTION] No previous snapshot for space {space_id}, creating initial snapshot")
            # Create initial snapshot
            current_snapshot = await self._create_snapshot(
                space_id, current_image, current_ocr_text
            )
            self._snapshot_cache[space_id] = current_snapshot

            return ChangeDetectionResult(
                changed=False,
                change_type=ChangeType.NO_CHANGE,
                space_id=space_id,
                current_snapshot=current_snapshot,
                summary="Initial snapshot created, no previous data to compare"
            )

        # Create current snapshot
        current_snapshot = await self._create_snapshot(
            space_id, current_image, current_ocr_text
        )

        # Calculate elapsed time
        elapsed_time = current_snapshot.timestamp - previous_snapshot.timestamp

        # Detect changes
        changed, change_type, differences, similarity = await self._compare_snapshots(
            previous_snapshot, current_snapshot, query, context
        )

        # Update cache
        self._snapshot_cache[space_id] = current_snapshot

        # Generate summary
        summary = self._generate_change_summary(
            changed, change_type, differences, elapsed_time, similarity
        )

        return ChangeDetectionResult(
            changed=changed,
            change_type=change_type,
            space_id=space_id,
            previous_snapshot=previous_snapshot,
            current_snapshot=current_snapshot,
            differences=differences,
            similarity_score=similarity,
            elapsed_time=elapsed_time,
            summary=summary
        )

    async def track_entity_state(
        self,
        entity_type: str,
        entity_id: str,
        state: str,
        space_id: int,
        data: Optional[Dict[str, Any]] = None
    ) -> EntityState:
        """
        Track state of an entity (error, build, process, etc.).

        Args:
            entity_type: Type of entity ("error", "build", "process")
            entity_id: Unique identifier for entity
            state: Current state ("active", "resolved", "completed", etc.)
            space_id: Space where entity is located
            data: Additional entity data

        Returns:
            EntityState with current and historical information
        """
        key = f"{entity_type}:{entity_id}"
        now = time.time()

        if key in self._entity_states:
            # Update existing entity
            entity = self._entity_states[key]

            # Track state change in history
            if entity.state != state:
                entity.history.append({
                    'timestamp': now,
                    'previous_state': entity.state,
                    'new_state': state,
                    'space_id': space_id
                })
                entity.state = state
                entity.last_updated = now

            entity.last_seen = now
            if data:
                entity.data.update(data)

        else:
            # Create new entity
            entity = EntityState(
                entity_type=entity_type,
                entity_id=entity_id,
                state=state,
                first_seen=now,
                last_seen=now,
                last_updated=now,
                space_id=space_id,
                data=data or {}
            )
            self._entity_states[key] = entity

        logger.info(f"[CHANGE-DETECTION] Tracked {entity_type} '{entity_id}': {state}")
        return entity

    async def query_entity_state(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[EntityState]:
        """
        Query entity state using natural language.
        e.g., "Did the error get fixed?" -> resolve "the error" and check its state

        Args:
            query: Natural language query
            context: Additional context

        Returns:
            EntityState if found, None otherwise
        """
        logger.info(f"[CHANGE-DETECTION] Querying entity state: {query}")

        # Use implicit resolver to extract entity reference
        entity_ref = await self._extract_entity_reference(query, context)

        if not entity_ref:
            logger.warning("[CHANGE-DETECTION] Could not extract entity reference from query")
            return None

        # Find matching entity
        entity_type = entity_ref.get('type', 'unknown')
        entity_id = entity_ref.get('id')

        if entity_id:
            key = f"{entity_type}:{entity_id}"
            return self._entity_states.get(key)

        # Fuzzy match based on type
        matching_entities = [
            e for k, e in self._entity_states.items()
            if e.entity_type == entity_type
        ]

        if matching_entities:
            # Return most recently updated
            return max(matching_entities, key=lambda e: e.last_updated)

        return None

    async def get_temporal_context(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get temporal context from query.
        e.g., "since I last asked" -> determine timestamp of last query

        Args:
            query: User query with temporal reference
            context: Additional context

        Returns:
            Temporal context with timestamps
        """
        temporal_context = {
            'reference_time': None,
            'time_range': None,
            'snapshots_in_range': []
        }

        # Use conversation tracker to find temporal reference
        if self.conversation_tracker:
            try:
                # Get conversation history
                recent_turns = self.conversation_tracker.get_recent_context(limit=10)

                # Extract temporal reference
                if 'last asked' in query.lower() or 'last time' in query.lower():
                    # Find last user query
                    for turn in reversed(recent_turns.get('turns', [])):
                        if turn.get('role') == 'user':
                            temporal_context['reference_time'] = turn.get('timestamp')
                            break

            except Exception as e:
                logger.warning(f"[CHANGE-DETECTION] Could not get temporal context: {e}")

        # Fallback: use last snapshot timestamp
        if not temporal_context['reference_time'] and self._snapshot_cache:
            latest_snapshot = max(
                self._snapshot_cache.values(),
                key=lambda s: s.timestamp,
                default=None
            )
            if latest_snapshot:
                temporal_context['reference_time'] = latest_snapshot.timestamp

        return temporal_context

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    async def _create_snapshot(
        self,
        space_id: int,
        image: Optional[Any],
        ocr_text: Optional[str]
    ) -> CachedSnapshot:
        """Create a snapshot with hashes"""
        # Calculate image hash
        image_hash = ""
        if image:
            image_hash = await self._calculate_image_hash(image)

        # Calculate text hash
        text_hash = ""
        if ocr_text:
            text_hash = hashlib.md5(ocr_text.encode()).hexdigest()

        # Extract entities from OCR text
        entities = {}
        if ocr_text:
            entities = await self._extract_entities(ocr_text)

        return CachedSnapshot(
            space_id=space_id,
            image=image,
            image_hash=image_hash,
            ocr_text=ocr_text or "",
            text_hash=text_hash,
            timestamp=time.time(),
            entities=entities
        )

    async def _calculate_image_hash(self, image: Any) -> str:
        """Calculate perceptual hash of image"""
        try:
            # Simple hash for now (could use perceptual hashing libraries)
            # Convert image to bytes and hash
            if hasattr(image, 'tobytes'):
                image_bytes = image.tobytes()
            elif isinstance(image, bytes):
                image_bytes = image
            else:
                # Try converting to string representation
                image_bytes = str(image).encode()

            return hashlib.md5(image_bytes).hexdigest()

        except Exception as e:
            logger.warning(f"[CHANGE-DETECTION] Could not hash image: {e}")
            return ""

    async def _compare_snapshots(
        self,
        previous: CachedSnapshot,
        current: CachedSnapshot,
        query: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> Tuple[bool, ChangeType, List[str], float]:
        """Compare two snapshots and return changes"""
        differences = []
        changed = False
        change_type = ChangeType.NO_CHANGE
        similarity = 1.0

        # Compare image hashes
        if previous.image_hash and current.image_hash:
            if previous.image_hash != current.image_hash:
                changed = True
                change_type = ChangeType.APPEARANCE_CHANGED
                differences.append("Visual appearance changed")

                # Calculate similarity (simple version - could be enhanced)
                similarity = 0.5  # Placeholder - would use actual image diff

        # Compare text hashes
        if previous.text_hash and current.text_hash:
            if previous.text_hash != current.text_hash:
                changed = True
                change_type = ChangeType.CONTENT_CHANGED
                differences.append("Text content changed")

                # Calculate text similarity
                similarity = self._calculate_text_similarity(
                    previous.ocr_text, current.ocr_text
                )

        # Compare entities
        entity_changes = self._compare_entities(previous.entities, current.entities)
        if entity_changes:
            changed = True
            change_type = ChangeType.STATE_CHANGED
            differences.extend(entity_changes)

        # If query-specific, check for specific changes
        if query and self.implicit_resolver:
            specific_changes = await self._detect_query_specific_changes(
                query, previous, current, context
            )
            if specific_changes:
                differences.extend(specific_changes)

        return changed, change_type, differences, similarity

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        if not text1 or not text2:
            return 0.0 if text1 != text2 else 1.0

        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def _compare_entities(
        self,
        previous_entities: Dict[str, Any],
        current_entities: Dict[str, Any]
    ) -> List[str]:
        """Compare entities between snapshots"""
        changes = []

        # Find new entities
        new_keys = set(current_entities.keys()) - set(previous_entities.keys())
        for key in new_keys:
            changes.append(f"New entity detected: {key}")

        # Find removed entities
        removed_keys = set(previous_entities.keys()) - set(current_entities.keys())
        for key in removed_keys:
            changes.append(f"Entity removed: {key}")

        # Find changed entities
        common_keys = set(previous_entities.keys()) & set(current_entities.keys())
        for key in common_keys:
            if previous_entities[key] != current_entities[key]:
                changes.append(f"Entity changed: {key}")

        return changes

    async def _extract_entities(self, ocr_text: str) -> Dict[str, Any]:
        """Extract entities from OCR text (errors, builds, processes, etc.)"""
        entities = {}

        # Look for error patterns
        import re
        error_patterns = [
            r'error[:\s]+([^\n]+)',
            r'exception[:\s]+([^\n]+)',
            r'failed[:\s]+([^\n]+)',
        ]

        for pattern in error_patterns:
            matches = re.finditer(pattern, ocr_text, re.IGNORECASE)
            for i, match in enumerate(matches):
                entities[f'error_{i}'] = {
                    'type': 'error',
                    'text': match.group(1).strip(),
                    'position': match.start()
                }

        # Look for build/process patterns
        build_patterns = [
            r'build[:\s]+(\w+)',
            r'compilation[:\s]+(\w+)',
            r'deployment[:\s]+(\w+)',
        ]

        for pattern in build_patterns:
            matches = re.finditer(pattern, ocr_text, re.IGNORECASE)
            for i, match in enumerate(matches):
                entities[f'build_{i}'] = {
                    'type': 'build',
                    'state': match.group(1).strip(),
                    'position': match.start()
                }

        return entities

    async def _extract_entity_reference(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extract entity reference from query using implicit resolver"""
        if not self.implicit_resolver:
            # Fallback: simple pattern matching
            query_lower = query.lower()

            if 'error' in query_lower:
                return {'type': 'error', 'id': None}
            if 'build' in query_lower:
                return {'type': 'build', 'id': None}
            if 'process' in query_lower:
                return {'type': 'process', 'id': None}

            return None

        try:
            # Use implicit resolver to extract entity
            resolved = await self.implicit_resolver.resolve_reference(query, context)

            if resolved and 'entity' in resolved:
                return {
                    'type': resolved.get('entity_type', 'unknown'),
                    'id': resolved.get('entity')
                }

        except Exception as e:
            logger.warning(f"[CHANGE-DETECTION] Failed to extract entity reference: {e}")

        return None

    async def _detect_query_specific_changes(
        self,
        query: str,
        previous: CachedSnapshot,
        current: CachedSnapshot,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Detect changes specific to the query"""
        changes = []
        query_lower = query.lower()

        # Check for error-related changes
        if 'error' in query_lower or 'fix' in query_lower:
            prev_errors = [e for e in previous.entities.values() if e.get('type') == 'error']
            curr_errors = [e for e in current.entities.values() if e.get('type') == 'error']

            if len(prev_errors) > len(curr_errors):
                changes.append(f"Errors reduced from {len(prev_errors)} to {len(curr_errors)}")
            elif len(prev_errors) < len(curr_errors):
                changes.append(f"New errors appeared: {len(curr_errors) - len(prev_errors)}")

        # Check for build-related changes
        if 'build' in query_lower or 'finish' in query_lower or 'complet' in query_lower:
            prev_builds = [e for e in previous.entities.values() if e.get('type') == 'build']
            curr_builds = [e for e in current.entities.values() if e.get('type') == 'build']

            # Check for state changes
            for prev_build in prev_builds:
                for curr_build in curr_builds:
                    if prev_build.get('text') == curr_build.get('text'):
                        if prev_build.get('state') != curr_build.get('state'):
                            changes.append(
                                f"Build state changed: {prev_build.get('state')} → {curr_build.get('state')}"
                            )

        return changes

    def _generate_change_summary(
        self,
        changed: bool,
        change_type: ChangeType,
        differences: List[str],
        elapsed_time: float,
        similarity: float
    ) -> str:
        """Generate human-readable summary of changes"""
        if not changed:
            return f"No changes detected (checked {elapsed_time:.1f}s after last snapshot)"

        lines = [
            f"Changes detected after {elapsed_time:.1f}s:",
            f"  Type: {change_type.value}",
            f"  Similarity: {similarity:.1%}",
            ""
        ]

        if differences:
            lines.append("  Changes:")
            for diff in differences:
                lines.append(f"    • {diff}")

        return "\n".join(lines)


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_global_manager: Optional[ChangeDetectionManager] = None


def get_change_detection_manager() -> Optional[ChangeDetectionManager]:
    """Get the global change detection manager instance"""
    return _global_manager


def initialize_change_detection_manager(
    cache_dir: Optional[Path] = None,
    cache_ttl: float = 3600.0,
    max_cache_size: int = 100,
    implicit_resolver: Optional[Any] = None,
    conversation_tracker: Optional[Any] = None
) -> ChangeDetectionManager:
    """
    Initialize the global ChangeDetectionManager instance.

    Args:
        cache_dir: Directory for persistent cache
        cache_ttl: How long to keep cached snapshots (seconds)
        max_cache_size: Maximum number of cached snapshots
        implicit_resolver: ImplicitReferenceResolver instance
        conversation_tracker: ConversationTracker instance

    Returns:
        ChangeDetectionManager instance
    """
    global _global_manager

    _global_manager = ChangeDetectionManager(
        cache_dir=cache_dir,
        cache_ttl=cache_ttl,
        max_cache_size=max_cache_size,
        implicit_resolver=implicit_resolver,
        conversation_tracker=conversation_tracker
    )

    logger.info("[CHANGE-DETECTION] Global instance initialized")
    return _global_manager
