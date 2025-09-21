#!/usr/bin/env python3
"""
Multi-Space Performance Optimizer for JARVIS Vision System
Implements intelligent caching, pre-fetching, and quality optimization
According to PRD Phase 4 requirements
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Tuple, Deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import statistics
import heapq

logger = logging.getLogger(__name__)

class AccessPattern(Enum):
    """Types of space access patterns"""
    FREQUENT = "frequent"          # Accessed often
    PERIODIC = "periodic"          # Accessed at regular intervals
    SEQUENTIAL = "sequential"      # Accessed in sequence with others
    RANDOM = "random"             # No clear pattern
    WORKFLOW = "workflow"         # Part of a workflow pattern

@dataclass
class SpaceAccessMetrics:
    """Metrics for space access patterns"""
    space_id: int
    access_count: int = 0
    last_access: Optional[datetime] = None
    access_history: Deque[datetime] = field(default_factory=lambda: deque(maxlen=100))
    average_interval: Optional[timedelta] = None
    access_pattern: AccessPattern = AccessPattern.RANDOM
    priority_score: float = 0.0
    linked_spaces: Set[int] = field(default_factory=set)
    typical_apps: Dict[str, int] = field(default_factory=dict)
    
    def update_access(self, timestamp: Optional[datetime] = None):
        """Update access metrics"""
        if timestamp is None:
            timestamp = datetime.now()
            
        self.access_history.append(timestamp)
        self.last_access = timestamp
        self.access_count += 1
        
        # Calculate average interval
        if len(self.access_history) > 1:
            intervals = []
            for i in range(1, len(self.access_history)):
                interval = self.access_history[i] - self.access_history[i-1]
                intervals.append(interval.total_seconds())
            
            if intervals:
                avg_seconds = statistics.mean(intervals)
                self.average_interval = timedelta(seconds=avg_seconds)
                
        # Update pattern detection
        self._detect_pattern()
        
    def _detect_pattern(self):
        """Detect access pattern based on history"""
        if len(self.access_history) < 5:
            return
            
        # Check for frequent access (many accesses in short time)
        recent_accesses = sum(
            1 for t in self.access_history
            if datetime.now() - t < timedelta(minutes=30)
        )
        if recent_accesses > 10:
            self.access_pattern = AccessPattern.FREQUENT
            return
            
        # Check for periodic access (regular intervals)
        if self.average_interval and len(self.access_history) > 10:
            intervals = []
            for i in range(1, min(11, len(self.access_history))):
                interval = self.access_history[-i] - self.access_history[-(i+1)]
                intervals.append(interval.total_seconds())
                
            if intervals:
                std_dev = statistics.stdev(intervals) if len(intervals) > 1 else 0
                mean = statistics.mean(intervals)
                
                # Low standard deviation indicates periodic pattern
                if mean > 0 and std_dev / mean < 0.3:
                    self.access_pattern = AccessPattern.PERIODIC
                    return
                    
        # Check for sequential access (accessed with other spaces)
        if len(self.linked_spaces) >= 2:
            self.access_pattern = AccessPattern.SEQUENTIAL
            return
            
        # Default to random
        self.access_pattern = AccessPattern.RANDOM

@dataclass
class CacheOptimizationStrategy:
    """Strategy for cache optimization"""
    max_cache_size_mb: int = 500
    min_cache_time: timedelta = field(default_factory=lambda: timedelta(seconds=30))
    max_cache_time: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    quality_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'high_priority': 0.8,
        'medium_priority': 0.5,
        'low_priority': 0.3
    })
    prefetch_threshold: float = 0.6
    eviction_policy: str = "adaptive_lru"  # lru, lfu, adaptive_lru

class MultiSpaceOptimizer:
    """
    Performance optimization system for multi-space capture
    Implements PRD Phase 4 requirements
    """
    
    def __init__(self, capture_engine=None, monitor=None):
        self.capture_engine = capture_engine
        self.monitor = monitor
        
        # Access pattern tracking
        self.space_metrics: Dict[int, SpaceAccessMetrics] = {}
        self.global_access_history = deque(maxlen=1000)
        
        # Optimization settings
        self.strategy = CacheOptimizationStrategy()
        self.optimization_active = False
        self.optimization_task = None
        
        # Performance metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_capture_time = 0.0
        self.capture_count = 0
        
        # Predictive models (simple for now)
        self.next_space_predictions: Dict[int, List[Tuple[int, float]]] = {}
        self.workflow_sequences: List[List[int]] = []
        
    async def start_optimization(self):
        """Start the optimization system"""
        if self.optimization_active:
            return
            
        self.optimization_active = True
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        logger.info("Multi-space optimization started")
        
    async def stop_optimization(self):
        """Stop the optimization system"""
        self.optimization_active = False
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
        logger.info("Multi-space optimization stopped")
        
    def track_space_access(self, space_id: int, linked_spaces: Optional[Set[int]] = None):
        """Track access to a space"""
        if space_id not in self.space_metrics:
            self.space_metrics[space_id] = SpaceAccessMetrics(space_id=space_id)
            
        metrics = self.space_metrics[space_id]
        metrics.update_access()
        
        # Track linked spaces
        if linked_spaces:
            metrics.linked_spaces.update(linked_spaces)
            
        # Update global history
        self.global_access_history.append((space_id, datetime.now()))
        
        # Update predictions
        self._update_predictions(space_id)
        
        # Calculate priority score
        self._calculate_priority_score(space_id)
        
    def track_capture_performance(self, space_id: int, capture_time: float, cache_hit: bool):
        """Track capture performance metrics"""
        self.total_capture_time += capture_time
        self.capture_count += 1
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            
        # Update space-specific metrics
        if space_id in self.space_metrics:
            metrics = self.space_metrics[space_id]
            # Adjust priority based on capture time
            if capture_time > 1.0:  # Slow capture
                metrics.priority_score *= 1.1
                
    async def _optimization_loop(self):
        """Main optimization loop"""
        while self.optimization_active:
            try:
                # Run optimization tasks
                await self._optimize_cache()
                await self._predictive_prefetch()
                await self._adjust_capture_quality()
                
                # Sleep with adaptive interval
                interval = self._calculate_optimization_interval()
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(30)  # Default interval on error
                
    async def _optimize_cache(self):
        """Optimize cache based on access patterns"""
        if not self.capture_engine or not hasattr(self.capture_engine, 'cache'):
            return
            
        try:
            # Get current cache stats
            cache_stats = self.capture_engine.get_cache_stats()
            cache_size_mb = cache_stats.get('size_mb', 0)
            
            # Check if optimization needed
            if cache_size_mb > self.strategy.max_cache_size_mb * 0.9:
                # Need to evict some entries
                await self._evict_cache_entries()
                
            # Adjust cache TTL based on patterns
            for space_id, metrics in self.space_metrics.items():
                if metrics.access_pattern == AccessPattern.FREQUENT:
                    # Extend cache time for frequently accessed spaces
                    ttl = int(self.strategy.max_cache_time.total_seconds())
                elif metrics.access_pattern == AccessPattern.RANDOM:
                    # Shorter cache time for random access
                    ttl = int(self.strategy.min_cache_time.total_seconds())
                else:
                    # Default cache time
                    ttl = int((self.strategy.min_cache_time.total_seconds() + 
                              self.strategy.max_cache_time.total_seconds()) / 2)
                    
                # This would update cache TTL for specific space
                # Implementation depends on cache system
                
        except Exception as e:
            logger.error(f"Cache optimization error: {e}")
            
    async def _predictive_prefetch(self):
        """Predictively prefetch spaces based on patterns"""
        if not self.capture_engine:
            return
            
        try:
            # Get spaces to prefetch
            spaces_to_prefetch = self._get_prefetch_candidates()
            
            if spaces_to_prefetch:
                # Use capture engine's prefetch method
                from .multi_space_capture_engine import (
                    SpaceCaptureRequest, CaptureQuality
                )
                
                # Determine quality based on priority
                quality = CaptureQuality.OPTIMIZED
                high_priority = [s for s in spaces_to_prefetch 
                               if self.space_metrics[s].priority_score > 0.8]
                if not high_priority:
                    quality = CaptureQuality.FAST
                    
                request = SpaceCaptureRequest(
                    space_ids=spaces_to_prefetch,
                    quality=quality,
                    use_cache=True,
                    priority=3,  # Low priority for prefetch
                    reason="predictive_prefetch",
                    require_permission=False
                )
                
                # Run prefetch in background
                asyncio.create_task(self.capture_engine.capture_all_spaces(request))
                logger.debug(f"Prefetching spaces: {spaces_to_prefetch}")
                
        except Exception as e:
            logger.error(f"Predictive prefetch error: {e}")
            
    async def _adjust_capture_quality(self):
        """Adjust capture quality based on usage patterns"""
        if not self.capture_engine:
            return
            
        try:
            # Analyze performance metrics
            if self.capture_count > 0:
                avg_capture_time = self.total_capture_time / self.capture_count
                cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses)
                
                # Adjust strategy based on metrics
                if avg_capture_time > 2.0:  # Captures taking too long
                    # Lower quality thresholds
                    self.strategy.quality_thresholds['high_priority'] = min(0.9, 
                        self.strategy.quality_thresholds['high_priority'] + 0.1)
                elif cache_hit_rate > 0.8:  # Good cache performance
                    # Can afford higher quality
                    self.strategy.quality_thresholds['high_priority'] = max(0.7,
                        self.strategy.quality_thresholds['high_priority'] - 0.05)
                        
        except Exception as e:
            logger.error(f"Quality adjustment error: {e}")
            
    async def _evict_cache_entries(self):
        """Evict cache entries based on adaptive strategy"""
        if not self.capture_engine:
            return
            
        # Build eviction candidates with scores
        eviction_candidates = []
        
        for space_id, metrics in self.space_metrics.items():
            # Calculate eviction score (lower = more likely to evict)
            score = self._calculate_eviction_score(metrics)
            eviction_candidates.append((score, space_id))
            
        # Sort by score (ascending)
        eviction_candidates.sort()
        
        # Evict lowest scoring entries
        spaces_to_evict = [space_id for score, space_id in eviction_candidates[:3]]
        if spaces_to_evict and hasattr(self.capture_engine, 'clear_cache'):
            await self.capture_engine.clear_cache(space_ids=spaces_to_evict)
            logger.debug(f"Evicted cache for spaces: {spaces_to_evict}")
            
    def _calculate_priority_score(self, space_id: int):
        """Calculate priority score for a space"""
        if space_id not in self.space_metrics:
            return 0.0
            
        metrics = self.space_metrics[space_id]
        score = 0.0
        
        # Factors for priority
        # 1. Access frequency
        if metrics.access_count > 0:
            recent_accesses = sum(
                1 for _, t in self.global_access_history
                if datetime.now() - t < timedelta(hours=1)
            )
            frequency_factor = min(1.0, recent_accesses / 10)
            score += frequency_factor * 0.3
            
        # 2. Access pattern
        pattern_scores = {
            AccessPattern.FREQUENT: 0.9,
            AccessPattern.WORKFLOW: 0.8,
            AccessPattern.PERIODIC: 0.6,
            AccessPattern.SEQUENTIAL: 0.5,
            AccessPattern.RANDOM: 0.2
        }
        score += pattern_scores.get(metrics.access_pattern, 0.2) * 0.3
        
        # 3. Recency
        if metrics.last_access:
            age = datetime.now() - metrics.last_access
            recency_factor = max(0, 1 - (age.total_seconds() / 3600))  # 1 hour decay
            score += recency_factor * 0.2
            
        # 4. Linked spaces (workflow indicator)
        if metrics.linked_spaces:
            workflow_factor = min(1.0, len(metrics.linked_spaces) / 3)
            score += workflow_factor * 0.2
            
        metrics.priority_score = score
        return score
        
    def _calculate_eviction_score(self, metrics: SpaceAccessMetrics) -> float:
        """Calculate eviction score (lower = more likely to evict)"""
        score = metrics.priority_score
        
        # Adjust for cache age
        if metrics.last_access:
            age = datetime.now() - metrics.last_access
            age_penalty = min(1.0, age.total_seconds() / 1800)  # 30 min max
            score *= (1 - age_penalty * 0.5)
            
        return score
        
    def _get_prefetch_candidates(self) -> List[int]:
        """Get spaces that should be prefetched"""
        candidates = []
        
        # Check each space's metrics
        for space_id, metrics in self.space_metrics.items():
            if metrics.priority_score > self.strategy.prefetch_threshold:
                candidates.append(space_id)
                
        # Add predicted next spaces
        if self.global_access_history:
            last_space = self.global_access_history[-1][0]
            if last_space in self.next_space_predictions:
                for predicted_space, confidence in self.next_space_predictions[last_space]:
                    if confidence > 0.5 and predicted_space not in candidates:
                        candidates.append(predicted_space)
                        
        return candidates[:5]  # Limit to 5 spaces
        
    def _update_predictions(self, current_space: int):
        """Update space transition predictions"""
        # Simple markov chain-like prediction
        if len(self.global_access_history) < 2:
            return
            
        # Get previous space
        prev_space = None
        for space_id, _ in reversed(list(self.global_access_history)[:-1]):
            if space_id != current_space:
                prev_space = space_id
                break
                
        if prev_space is None:
            return
            
        # Update transition count
        if prev_space not in self.next_space_predictions:
            self.next_space_predictions[prev_space] = []
            
        # Simple frequency-based prediction
        transitions = defaultdict(int)
        for i in range(len(self.global_access_history) - 1):
            if self.global_access_history[i][0] == prev_space:
                next_space = self.global_access_history[i + 1][0]
                transitions[next_space] += 1
                
        # Calculate probabilities
        total = sum(transitions.values())
        if total > 0:
            predictions = [
                (space, count / total)
                for space, count in transitions.items()
            ]
            predictions.sort(key=lambda x: x[1], reverse=True)
            self.next_space_predictions[prev_space] = predictions[:3]
            
    def _calculate_optimization_interval(self) -> float:
        """Calculate adaptive optimization interval"""
        # Base interval
        interval = 30.0
        
        # Adjust based on activity
        if self.capture_count > 0:
            recent_captures = sum(
                1 for _, t in self.global_access_history
                if datetime.now() - t < timedelta(minutes=5)
            )
            
            if recent_captures > 10:
                interval = 10.0  # High activity, optimize more frequently
            elif recent_captures < 2:
                interval = 60.0  # Low activity, optimize less frequently
                
        return interval
        
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        cache_hit_rate = 0.0
        if self.cache_hits + self.cache_misses > 0:
            cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses)
            
        return {
            "cache_hit_rate": cache_hit_rate,
            "total_captures": self.capture_count,
            "average_capture_time": self.total_capture_time / max(1, self.capture_count),
            "tracked_spaces": len(self.space_metrics),
            "optimization_active": self.optimization_active,
            "space_patterns": {
                space_id: {
                    "pattern": metrics.access_pattern.value,
                    "priority": metrics.priority_score,
                    "access_count": metrics.access_count
                }
                for space_id, metrics in list(self.space_metrics.items())[:10]
            }
        }
        
    async def optimize_capture_request(self, request) -> Any:
        """Optimize a capture request based on patterns"""
        # Determine optimal quality based on space priorities
        if hasattr(request, 'space_ids'):
            max_priority = max(
                self.space_metrics.get(sid, SpaceAccessMetrics(space_id=sid)).priority_score
                for sid in request.space_ids
            )
            
            # Adjust quality based on priority
            from .multi_space_capture_engine import CaptureQuality
            if max_priority > self.strategy.quality_thresholds['high_priority']:
                request.quality = CaptureQuality.FULL
            elif max_priority < self.strategy.quality_thresholds['low_priority']:
                request.quality = CaptureQuality.THUMBNAIL
                
            # Track access
            for space_id in request.space_ids:
                self.track_space_access(space_id, set(request.space_ids))
                
        return request