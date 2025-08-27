#!/usr/bin/env python3
"""
Experience Replay System for Vision System v2.0
Stores and manages interaction history for continuous learning
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import pickle
import heapq
import random
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib
import lz4.frame

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """Single experience in the replay buffer"""
    experience_id: str
    timestamp: datetime
    command: str
    command_embedding: np.ndarray
    intent: str
    confidence: float
    handler: str
    response: str
    success: bool
    latency_ms: float
    user_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    feedback: Optional[Dict[str, Any]] = None
    importance_score: float = 1.0
    replay_count: int = 0
    last_replayed: Optional[datetime] = None
    
    def __hash__(self):
        return hash(self.experience_id)


@dataclass
class Pattern:
    """Extracted pattern from experiences"""
    pattern_id: str
    pattern_type: str  # command, sequence, context, failure
    description: str
    examples: List[str]
    frequency: int
    confidence: float
    first_seen: datetime
    last_seen: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplayBatch:
    """Batch of experiences for training"""
    experiences: List[Experience]
    sampling_method: str  # uniform, prioritized, recent, pattern_based
    metadata: Dict[str, Any] = field(default_factory=dict)


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer with efficient sampling
    Uses importance sampling to replay more important experiences
    """
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.alpha = 0.6  # Prioritization exponent
        self.beta = 0.4   # Importance sampling exponent
        self.beta_increment = 0.001
        self.max_priority = 1.0
        
    def add(self, experience: Experience):
        """Add experience with max priority"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        # Set priority
        priority = experience.importance_score ** self.alpha
        self.priorities[self.position] = priority
        self.max_priority = max(self.max_priority, priority)
        
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple[List[Experience], List[float], List[int]]:
        """Sample batch with importance weights"""
        if len(self.buffer) == 0:
            return [], [], []
            
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(
            len(self.buffer),
            size=min(batch_size, len(self.buffer)),
            replace=False,
            p=probabilities
        )
        
        # Calculate importance weights
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, weights.tolist(), indices.tolist()
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities after training"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)


class ExperienceReplaySystem:
    """
    Advanced experience replay system with pattern extraction
    and intelligent sampling strategies
    """
    
    def __init__(self, capacity: int = 10000):
        # Main storage
        self.capacity = capacity
        self.replay_buffer = PrioritizedReplayBuffer(capacity)
        self.experience_index: Dict[str, Experience] = {}
        
        # Pattern extraction
        self.patterns: Dict[str, Pattern] = {}
        self.pattern_extractors: Dict[str, Callable] = {}
        self._register_default_extractors()
        
        # Compressed storage for long-term memory
        self.compressed_storage = Path("backend/data/experience_replay")
        self.compressed_storage.mkdir(exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'total_experiences': 0,
            'successful_experiences': 0,
            'failed_experiences': 0,
            'patterns_extracted': 0,
            'replay_counts': defaultdict(int),
            'compression_ratio': 1.0
        }
        
        # Threading for background tasks
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.pattern_extraction_thread: Optional[threading.Thread] = None
        self.running = True
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info(f"Experience Replay System initialized with capacity: {capacity}")
    
    def _register_default_extractors(self):
        """Register default pattern extraction functions"""
        self.pattern_extractors = {
            'command_patterns': self._extract_command_patterns,
            'failure_patterns': self._extract_failure_patterns,
            'sequence_patterns': self._extract_sequence_patterns,
            'context_patterns': self._extract_context_patterns,
            'performance_patterns': self._extract_performance_patterns
        }
    
    def _start_background_tasks(self):
        """Start background pattern extraction"""
        def extraction_loop():
            while self.running:
                try:
                    # Extract patterns every 5 minutes
                    asyncio.run(self._extract_all_patterns())
                    asyncio.sleep(300)
                except Exception as e:
                    logger.error(f"Pattern extraction error: {e}")
                    asyncio.sleep(600)  # Wait longer on error
        
        self.pattern_extraction_thread = threading.Thread(
            target=extraction_loop,
            daemon=True
        )
        self.pattern_extraction_thread.start()
    
    async def add_experience(
        self,
        command: str,
        command_embedding: np.ndarray,
        intent: str,
        confidence: float,
        handler: str,
        response: str,
        success: bool,
        latency_ms: float,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        feedback: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add new experience to replay buffer"""
        # Generate experience ID
        exp_id = hashlib.sha256(
            f"{command}{datetime.now().timestamp()}{random.random()}".encode()
        ).hexdigest()[:16]
        
        # Calculate importance score
        importance = self._calculate_importance(
            confidence, success, latency_ms, feedback
        )
        
        # Create experience
        experience = Experience(
            experience_id=exp_id,
            timestamp=datetime.now(),
            command=command,
            command_embedding=command_embedding,
            intent=intent,
            confidence=confidence,
            handler=handler,
            response=response,
            success=success,
            latency_ms=latency_ms,
            user_id=user_id,
            context=context or {},
            feedback=feedback,
            importance_score=importance
        )
        
        # Add to buffer
        self.replay_buffer.add(experience)
        self.experience_index[exp_id] = experience
        
        # Update statistics
        self.stats['total_experiences'] += 1
        if success:
            self.stats['successful_experiences'] += 1
        else:
            self.stats['failed_experiences'] += 1
        
        # Check if compression needed
        if self.stats['total_experiences'] % 1000 == 0:
            self.executor.submit(self._compress_old_experiences)
        
        logger.debug(f"Added experience {exp_id} with importance {importance:.2f}")
        
        return exp_id
    
    def _calculate_importance(
        self,
        confidence: float,
        success: bool,
        latency_ms: float,
        feedback: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate importance score for prioritized replay"""
        importance = 1.0
        
        # Failed experiences are more important to learn from
        if not success:
            importance *= 2.0
        
        # Low confidence experiences are important
        if confidence < 0.5:
            importance *= 1.5
        elif confidence > 0.9:
            importance *= 0.8
        
        # High latency is important to optimize
        if latency_ms > 100:
            importance *= 1.3
        
        # User feedback increases importance
        if feedback:
            if feedback.get('rating', 0) < 3:
                importance *= 1.5
            elif feedback.get('rating', 0) >= 4:
                importance *= 0.9
        
        return min(importance, 5.0)  # Cap at 5x normal importance
    
    async def sample_batch(
        self,
        batch_size: int,
        method: str = "prioritized"
    ) -> ReplayBatch:
        """Sample batch of experiences for training"""
        if method == "prioritized":
            experiences, weights, indices = self.replay_buffer.sample(batch_size)
        
        elif method == "recent":
            # Sample from recent experiences
            recent = list(self.replay_buffer.buffer)[-batch_size:]
            experiences = recent
            weights = [1.0] * len(experiences)
            indices = list(range(len(self.replay_buffer.buffer) - len(recent), len(self.replay_buffer.buffer)))
        
        elif method == "failure":
            # Sample mostly failures
            failures = [exp for exp in self.replay_buffer.buffer if not exp.success]
            if failures:
                sample_size = min(batch_size, len(failures))
                experiences = random.sample(failures, sample_size)
                weights = [2.0] * len(experiences)  # Higher weight for failures
                indices = [i for i, exp in enumerate(self.replay_buffer.buffer) if exp in experiences]
            else:
                experiences, weights, indices = [], [], []
        
        elif method == "pattern_based":
            # Sample based on extracted patterns
            experiences = self._sample_pattern_based(batch_size)
            weights = [1.0] * len(experiences)
            indices = [i for i, exp in enumerate(self.replay_buffer.buffer) if exp in experiences]
        
        else:  # uniform
            sample_size = min(batch_size, len(self.replay_buffer.buffer))
            indices = random.sample(range(len(self.replay_buffer.buffer)), sample_size)
            experiences = [self.replay_buffer.buffer[i] for i in indices]
            weights = [1.0] * len(experiences)
        
        # Update replay counts
        for exp in experiences:
            exp.replay_count += 1
            exp.last_replayed = datetime.now()
            self.stats['replay_counts'][exp.experience_id] += 1
        
        return ReplayBatch(
            experiences=experiences,
            sampling_method=method,
            metadata={
                'weights': weights,
                'indices': indices,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def _sample_pattern_based(self, batch_size: int) -> List[Experience]:
        """Sample experiences based on learned patterns"""
        if not self.patterns:
            return []
        
        # Get experiences that match important patterns
        pattern_experiences = []
        
        for pattern in self.patterns.values():
            if pattern.confidence > 0.7:  # High confidence patterns
                # Find experiences matching this pattern
                for exp in self.replay_buffer.buffer:
                    if self._matches_pattern(exp, pattern):
                        pattern_experiences.append(exp)
        
        # Sample from pattern matches
        if pattern_experiences:
            sample_size = min(batch_size, len(pattern_experiences))
            return random.sample(pattern_experiences, sample_size)
        
        return []
    
    def _matches_pattern(self, experience: Experience, pattern: Pattern) -> bool:
        """Check if experience matches a pattern"""
        if pattern.pattern_type == "command":
            # Check if command contains pattern keywords
            pattern_keywords = set(word.lower() for desc in pattern.examples for word in desc.split())
            command_words = set(experience.command.lower().split())
            return len(pattern_keywords & command_words) > 0
        
        elif pattern.pattern_type == "failure":
            return not experience.success and experience.intent in pattern.metadata.get('intents', [])
        
        elif pattern.pattern_type == "context":
            # Check context similarity
            pattern_context = pattern.metadata.get('context_features', {})
            return any(key in experience.context for key in pattern_context.keys())
        
        return False
    
    async def _extract_all_patterns(self):
        """Extract all types of patterns from buffer"""
        logger.info("Starting pattern extraction...")
        
        for extractor_name, extractor_func in self.pattern_extractors.items():
            try:
                patterns = await extractor_func()
                for pattern in patterns:
                    self.patterns[pattern.pattern_id] = pattern
                    self.stats['patterns_extracted'] += 1
                
                logger.info(f"Extracted {len(patterns)} patterns using {extractor_name}")
            except Exception as e:
                logger.error(f"Pattern extraction failed for {extractor_name}: {e}")
    
    async def _extract_command_patterns(self) -> List[Pattern]:
        """Extract common command patterns"""
        patterns = []
        
        # Group experiences by intent
        intent_groups = defaultdict(list)
        for exp in self.replay_buffer.buffer:
            intent_groups[exp.intent].append(exp)
        
        # Find common words/phrases per intent
        for intent, experiences in intent_groups.items():
            if len(experiences) < 5:  # Need minimum examples
                continue
            
            # Extract common words
            word_freq = defaultdict(int)
            for exp in experiences:
                for word in exp.command.lower().split():
                    word_freq[word] += 1
            
            # Get top words
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            
            if top_words:
                pattern = Pattern(
                    pattern_id=f"cmd_pattern_{intent}_{datetime.now().timestamp()}",
                    pattern_type="command",
                    description=f"Common words for {intent} intent",
                    examples=[word for word, _ in top_words],
                    frequency=len(experiences),
                    confidence=sum(exp.confidence for exp in experiences) / len(experiences),
                    first_seen=min(exp.timestamp for exp in experiences),
                    last_seen=max(exp.timestamp for exp in experiences),
                    metadata={'intent': intent, 'word_frequencies': dict(top_words)}
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _extract_failure_patterns(self) -> List[Pattern]:
        """Extract patterns from failed experiences"""
        patterns = []
        
        # Group failures by handler and error type
        failure_groups = defaultdict(list)
        
        for exp in self.replay_buffer.buffer:
            if not exp.success:
                error_type = exp.context.get('error_type', 'unknown')
                key = f"{exp.handler}_{error_type}"
                failure_groups[key].append(exp)
        
        # Create patterns for common failures
        for key, failures in failure_groups.items():
            if len(failures) >= 3:  # Repeated failure pattern
                handler, error_type = key.rsplit('_', 1)
                
                pattern = Pattern(
                    pattern_id=f"failure_pattern_{key}_{datetime.now().timestamp()}",
                    pattern_type="failure",
                    description=f"Repeated failures in {handler} with {error_type}",
                    examples=[f.command for f in failures[:5]],
                    frequency=len(failures),
                    confidence=0.9,  # High confidence for repeated failures
                    first_seen=min(f.timestamp for f in failures),
                    last_seen=max(f.timestamp for f in failures),
                    metadata={
                        'handler': handler,
                        'error_type': error_type,
                        'intents': list(set(f.intent for f in failures))
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _extract_sequence_patterns(self) -> List[Pattern]:
        """Extract patterns from command sequences"""
        patterns = []
        
        # Group experiences by user
        user_sequences = defaultdict(list)
        for exp in sorted(self.replay_buffer.buffer, key=lambda x: x.timestamp):
            if exp.user_id:
                user_sequences[exp.user_id].append(exp)
        
        # Look for common sequences
        sequence_counts = defaultdict(int)
        
        for user_id, experiences in user_sequences.items():
            # Extract bi-grams and tri-grams
            for i in range(len(experiences) - 1):
                # Check if commands are within 5 minutes
                if (experiences[i+1].timestamp - experiences[i].timestamp) < timedelta(minutes=5):
                    bigram = (experiences[i].intent, experiences[i+1].intent)
                    sequence_counts[bigram] += 1
                    
                    if i < len(experiences) - 2:
                        if (experiences[i+2].timestamp - experiences[i+1].timestamp) < timedelta(minutes=5):
                            trigram = (experiences[i].intent, experiences[i+1].intent, experiences[i+2].intent)
                            sequence_counts[trigram] += 1
        
        # Create patterns for common sequences
        for sequence, count in sequence_counts.items():
            if count >= 3:  # Repeated sequence
                pattern = Pattern(
                    pattern_id=f"seq_pattern_{'_'.join(sequence)}_{datetime.now().timestamp()}",
                    pattern_type="sequence",
                    description=f"Common sequence: {' -> '.join(sequence)}",
                    examples=list(sequence),
                    frequency=count,
                    confidence=0.7,
                    first_seen=datetime.now() - timedelta(days=7),  # Approximate
                    last_seen=datetime.now(),
                    metadata={'sequence_length': len(sequence)}
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _extract_context_patterns(self) -> List[Pattern]:
        """Extract patterns from context information"""
        patterns = []
        
        # Analyze context features
        context_features = defaultdict(lambda: defaultdict(int))
        
        for exp in self.replay_buffer.buffer:
            for key, value in exp.context.items():
                if isinstance(value, (str, int, float, bool)):
                    context_features[key][str(value)] += 1
        
        # Find dominant context patterns
        for feature, values in context_features.items():
            if len(values) > 1:
                # Find most common value
                dominant_value = max(values.items(), key=lambda x: x[1])
                
                if dominant_value[1] > len(self.replay_buffer.buffer) * 0.3:  # 30% threshold
                    pattern = Pattern(
                        pattern_id=f"ctx_pattern_{feature}_{datetime.now().timestamp()}",
                        pattern_type="context",
                        description=f"Common context: {feature}={dominant_value[0]}",
                        examples=[f"{feature}={dominant_value[0]}"],
                        frequency=dominant_value[1],
                        confidence=dominant_value[1] / len(self.replay_buffer.buffer),
                        first_seen=datetime.now() - timedelta(days=7),
                        last_seen=datetime.now(),
                        metadata={'context_features': {feature: dominant_value[0]}}
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _extract_performance_patterns(self) -> List[Pattern]:
        """Extract patterns related to performance"""
        patterns = []
        
        # Analyze latency patterns
        latency_by_intent = defaultdict(list)
        for exp in self.replay_buffer.buffer:
            latency_by_intent[exp.intent].append(exp.latency_ms)
        
        # Find intents with consistent high/low latency
        for intent, latencies in latency_by_intent.items():
            if len(latencies) >= 5:
                avg_latency = np.mean(latencies)
                std_latency = np.std(latencies)
                
                if avg_latency > 100 and std_latency < 20:  # Consistently slow
                    pattern = Pattern(
                        pattern_id=f"perf_pattern_slow_{intent}_{datetime.now().timestamp()}",
                        pattern_type="performance",
                        description=f"Consistently slow intent: {intent}",
                        examples=[f"avg={avg_latency:.1f}ms", f"std={std_latency:.1f}ms"],
                        frequency=len(latencies),
                        confidence=0.8,
                        first_seen=datetime.now() - timedelta(days=1),
                        last_seen=datetime.now(),
                        metadata={'intent': intent, 'avg_latency': avg_latency}
                    )
                    patterns.append(pattern)
                
                elif avg_latency < 50 and std_latency < 10:  # Consistently fast
                    pattern = Pattern(
                        pattern_id=f"perf_pattern_fast_{intent}_{datetime.now().timestamp()}",
                        pattern_type="performance",
                        description=f"Consistently fast intent: {intent}",
                        examples=[f"avg={avg_latency:.1f}ms", f"std={std_latency:.1f}ms"],
                        frequency=len(latencies),
                        confidence=0.8,
                        first_seen=datetime.now() - timedelta(days=1),
                        last_seen=datetime.now(),
                        metadata={'intent': intent, 'avg_latency': avg_latency}
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def update_priorities(self, experience_ids: List[str], td_errors: List[float]):
        """Update priorities based on temporal difference errors"""
        indices = []
        for exp_id in experience_ids:
            # Find index in buffer
            for i, exp in enumerate(self.replay_buffer.buffer):
                if exp.experience_id == exp_id:
                    indices.append(i)
                    break
        
        # Update priorities
        priorities = [abs(td_error) + 0.01 for td_error in td_errors]  # Small epsilon
        self.replay_buffer.update_priorities(indices, priorities)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about replay system"""
        buffer_size = len(self.replay_buffer.buffer)
        
        if buffer_size > 0:
            avg_confidence = np.mean([exp.confidence for exp in self.replay_buffer.buffer])
            avg_latency = np.mean([exp.latency_ms for exp in self.replay_buffer.buffer])
            success_rate = self.stats['successful_experiences'] / max(1, self.stats['total_experiences'])
            
            # Intent distribution
            intent_counts = defaultdict(int)
            for exp in self.replay_buffer.buffer:
                intent_counts[exp.intent] += 1
            
            # User distribution
            user_counts = defaultdict(int)
            for exp in self.replay_buffer.buffer:
                user_counts[exp.user_id or 'anonymous'] += 1
            
        else:
            avg_confidence = avg_latency = success_rate = 0.0
            intent_counts = user_counts = {}
        
        return {
            'buffer_stats': {
                'current_size': buffer_size,
                'capacity': self.capacity,
                'utilization': buffer_size / self.capacity,
                'total_experiences': self.stats['total_experiences'],
                'compression_ratio': self.stats['compression_ratio']
            },
            'performance_stats': {
                'success_rate': success_rate,
                'avg_confidence': avg_confidence,
                'avg_latency_ms': avg_latency
            },
            'pattern_stats': {
                'total_patterns': len(self.patterns),
                'pattern_types': defaultdict(int, {
                    p.pattern_type: 1 for p in self.patterns.values()
                })
            },
            'distribution_stats': {
                'intent_distribution': dict(intent_counts),
                'user_distribution': dict(user_counts),
                'top_replayed': sorted(
                    self.stats['replay_counts'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            }
        }
    
    def _compress_old_experiences(self):
        """Compress old experiences to save memory"""
        # Find experiences older than 1 day
        cutoff_time = datetime.now() - timedelta(days=1)
        old_experiences = []
        
        for exp in self.replay_buffer.buffer:
            if exp.timestamp < cutoff_time and exp.replay_count > 0:
                old_experiences.append(exp)
        
        if not old_experiences:
            return
        
        # Compress and save
        for exp in old_experiences:
            try:
                # Serialize experience
                exp_data = {
                    'experience_id': exp.experience_id,
                    'timestamp': exp.timestamp.isoformat(),
                    'command': exp.command,
                    'intent': exp.intent,
                    'confidence': exp.confidence,
                    'success': exp.success,
                    'latency_ms': exp.latency_ms,
                    'importance_score': exp.importance_score
                }
                
                # Compress embedding separately
                embedding_bytes = exp.command_embedding.tobytes()
                compressed_embedding = lz4.frame.compress(embedding_bytes)
                
                # Save to file
                exp_file = self.compressed_storage / f"{exp.experience_id}.json"
                emb_file = self.compressed_storage / f"{exp.experience_id}.emb"
                
                with open(exp_file, 'w') as f:
                    json.dump(exp_data, f)
                
                with open(emb_file, 'wb') as f:
                    f.write(compressed_embedding)
                
                # Calculate compression ratio
                original_size = len(embedding_bytes)
                compressed_size = len(compressed_embedding)
                self.stats['compression_ratio'] = compressed_size / original_size
                
                logger.debug(f"Compressed experience {exp.experience_id}, ratio: {self.stats['compression_ratio']:.2f}")
                
            except Exception as e:
                logger.error(f"Failed to compress experience {exp.experience_id}: {e}")
    
    async def shutdown(self):
        """Gracefully shutdown the replay system"""
        logger.info("Shutting down Experience Replay System")
        
        self.running = False
        
        # Wait for threads
        if self.pattern_extraction_thread:
            self.pattern_extraction_thread.join(timeout=5)
        
        # Final pattern extraction
        await self._extract_all_patterns()
        
        # Compress remaining experiences
        self._compress_old_experiences()
        
        # Save patterns
        patterns_file = self.compressed_storage / "patterns.pkl"
        with open(patterns_file, 'wb') as f:
            pickle.dump(self.patterns, f)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Experience Replay System shutdown complete")


# Singleton instance
_replay_system: Optional[ExperienceReplaySystem] = None


def get_experience_replay_system() -> ExperienceReplaySystem:
    """Get singleton instance of experience replay system"""
    global _replay_system
    if _replay_system is None:
        _replay_system = ExperienceReplaySystem()
    return _replay_system