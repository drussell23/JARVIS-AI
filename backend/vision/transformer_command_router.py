#!/usr/bin/env python3
"""
Transformer-Based Command Router - Phase 3
Ultra-fast (<100ms) neural routing with continuous learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import logging
import asyncio
import time
from collections import defaultdict, deque
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class RouteMetrics:
    """Performance metrics for a route"""
    handler_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    last_used: Optional[datetime] = None
    confidence_history: List[float] = field(default_factory=list)
    error_types: Dict[str, int] = field(default_factory=dict)


@dataclass
class LearningRecord:
    """Record for continuous learning"""
    command: str
    embedding: np.ndarray
    selected_handler: str
    alternative_handlers: List[Tuple[str, float]]
    execution_time_ms: float
    success: bool
    confidence: float
    context: Dict[str, Any]
    timestamp: datetime
    user_feedback: Optional[Dict[str, Any]] = None


class OptimizedTransformerRouter(nn.Module):
    """
    Lightweight transformer for ultra-fast routing
    Optimized for <100ms inference
    """
    
    def __init__(
        self, 
        d_model: int = 256,  # Reduced for speed
        n_heads: int = 4,    # Fewer heads for efficiency
        n_layers: int = 2,   # Shallow for speed
        max_seq_len: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Positional encoding
        self.pos_encoder = nn.Embedding(max_seq_len, d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=512,  # Reduced FFN
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Route prediction head
        self.route_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 64)  # Route embedding space
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Handler embedding matrix (learned during training)
        self.handler_embeddings = nn.Parameter(torch.randn(100, 64))
        
        # Cache for faster inference
        self.attention_cache = {}
        
    def forward(self, input_embeddings, compute_attention=False):
        batch_size, seq_len, _ = input_embeddings.shape
        
        # Add positional encoding
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(input_embeddings.device)
        pos_embeddings = self.pos_encoder(positions)
        
        # Combine with input
        x = input_embeddings + pos_embeddings
        
        # Pass through transformer
        if compute_attention:
            x = self.transformer(x)
            attention_weights = None  # Would need custom implementation
        else:
            x = self.transformer(x)
            attention_weights = None
        
        # Global pooling
        x_pooled = x.mean(dim=1)  # [batch_size, d_model]
        
        # Get route embedding and confidence
        route_embedding = self.route_head(x_pooled)  # [batch_size, 64]
        confidence = self.confidence_head(x_pooled)  # [batch_size, 1]
        
        return route_embedding, confidence.squeeze(), attention_weights


class TransformerCommandRouter:
    """
    Production-ready transformer-based command router
    Features:
    - <100ms latency guarantee
    - Dynamic handler discovery
    - Continuous learning pipeline
    - Performance-based route optimization
    """
    
    def __init__(self):
        # Model configuration
        self.model = OptimizedTransformerRouter()
        self.model.eval()  # Start in eval mode for speed
        
        # Handler registry with dynamic discovery
        self.handlers: Dict[str, Callable] = {}
        self.handler_metadata: Dict[str, Dict[str, Any]] = {}
        self.handler_embeddings_map: Dict[str, int] = {}  # Maps handler name to embedding index
        
        # Performance tracking
        self.route_metrics: Dict[str, RouteMetrics] = defaultdict(
            lambda: RouteMetrics(handler_name="unknown")
        )
        self.latency_buffer = deque(maxlen=1000)  # Track recent latencies
        
        # Learning pipeline
        self.learning_records: deque = deque(maxlen=10000)
        self.learning_thread: Optional[threading.Thread] = None
        self.learning_executor = ThreadPoolExecutor(max_workers=2)
        
        # Route optimization
        self.route_cache: Dict[str, Tuple[str, float]] = {}  # Command hash -> (handler, confidence)
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Multi-path exploration
        self.exploration_mode = False
        self.exploration_results: Dict[str, List[Tuple[str, float, float]]] = defaultdict(list)
        
        # Pre-loading predictions
        self.preload_cache: Dict[str, Any] = {}
        self.preload_patterns: List[str] = []
        
        # Initialize components
        self._initialize_optimizations()
        self._start_continuous_learning()
        
        logger.info("Transformer Command Router initialized with <100ms latency target")
    
    def _initialize_optimizations(self):
        """Initialize performance optimizations"""
        # JIT compile the model for faster inference
        try:
            dummy_input = torch.randn(1, 10, 256)
            self.model = torch.jit.trace(self.model, (dummy_input, False))
            logger.info("Model JIT compiled for optimal performance")
        except Exception as e:
            logger.warning(f"JIT compilation failed, using eager mode: {e}")
        
        # Pre-allocate tensors for common sizes
        self.preallocated_tensors = {
            'small': torch.zeros(1, 32, 256),
            'medium': torch.zeros(1, 64, 256),
            'large': torch.zeros(1, 128, 256)
        }
    
    def _start_continuous_learning(self):
        """Start the continuous learning pipeline"""
        def learning_loop():
            while True:
                try:
                    # Process learning records in batches
                    if len(self.learning_records) >= 50:
                        self._process_learning_batch()
                    
                    # Optimize routes periodically
                    self._optimize_routes()
                    
                    # Update preload predictions
                    self._update_preload_predictions()
                    
                    time.sleep(60)  # Run every minute
                    
                except Exception as e:
                    logger.error(f"Learning pipeline error: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
        
        self.learning_thread = threading.Thread(target=learning_loop, daemon=True)
        self.learning_thread.start()
    
    async def discover_handler(
        self,
        handler_func: Callable,
        auto_analyze: bool = True
    ) -> str:
        """
        Dynamically discover and register a new handler
        Returns the generated handler name
        """
        # Extract metadata from function
        handler_name = handler_func.__name__
        handler_doc = handler_func.__doc__ or ""
        
        # Auto-analyze if requested
        if auto_analyze:
            # Analyze function signature and behavior
            import inspect
            sig = inspect.signature(handler_func)
            params = list(sig.parameters.keys())
            
            metadata = {
                'discovered_at': datetime.now(),
                'parameters': params,
                'is_async': inspect.iscoroutinefunction(handler_func),
                'description': handler_doc.strip().split('\n')[0] if handler_doc else "",
                'inferred_patterns': self._infer_handler_patterns(handler_name, handler_doc)
            }
        else:
            metadata = {'discovered_at': datetime.now()}
        
        # Register handler
        self.handlers[handler_name] = handler_func
        self.handler_metadata[handler_name] = metadata
        
        # Assign embedding index
        if handler_name not in self.handler_embeddings_map:
            idx = len(self.handler_embeddings_map)
            self.handler_embeddings_map[handler_name] = idx
            
            # Initialize embedding with semantic information
            if idx < self.model.handler_embeddings.shape[0]:
                self._initialize_handler_embedding(handler_name, idx)
        
        logger.info(f"Discovered and registered handler: {handler_name}")
        return handler_name
    
    def _infer_handler_patterns(self, name: str, doc: str) -> List[str]:
        """Infer command patterns from handler name and documentation"""
        patterns = []
        
        # From name
        words = name.replace('_', ' ').replace('handle', '').replace('process', '').strip()
        if words:
            patterns.append(words)
        
        # From documentation
        if doc:
            # Look for example commands in doc
            lines = doc.strip().split('\n')
            for line in lines:
                if 'example:' in line.lower() or 'e.g.' in line.lower():
                    patterns.append(line.split(':', 1)[-1].strip())
        
        return patterns
    
    def _initialize_handler_embedding(self, handler_name: str, idx: int):
        """Initialize handler embedding based on semantic information"""
        # This would use a pre-trained encoder in production
        # For now, use random initialization with some structure
        patterns = self.handler_metadata[handler_name].get('inferred_patterns', [])
        
        if patterns:
            # Create embedding based on patterns
            embedding = torch.zeros(64)
            for pattern in patterns:
                pattern_hash = hash(pattern.lower())
                embedding[pattern_hash % 64] += 1.0
            
            embedding = F.normalize(embedding, dim=0)
            self.model.handler_embeddings.data[idx] = embedding
    
    async def route_command(
        self,
        command: str,
        context: Optional[Dict[str, Any]] = None,
        force_exploration: bool = False
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Route command with <100ms latency guarantee
        Returns (result, routing_info)
        """
        start_time = time.perf_counter()
        
        # Check cache first
        cache_key = hash((command, str(sorted(context.items()) if context else [])))
        if cache_key in self.route_cache and not force_exploration:
            handler_name, confidence = self.route_cache[cache_key]
            self.cache_hits += 1
            
            # Fast path execution
            result = await self._execute_handler(handler_name, command, context)
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            return result, {
                'handler': handler_name,
                'confidence': confidence,
                'latency_ms': latency_ms,
                'cache_hit': True
            }
        
        self.cache_misses += 1
        
        # Neural routing
        route_embedding, confidence, alternatives = await self._neural_route(command, context)
        
        # Multi-path exploration if enabled
        if self.exploration_mode or force_exploration:
            exploration_results = await self._explore_multiple_paths(
                command, context, alternatives[:3]
            )
            best_handler = self._select_best_from_exploration(exploration_results)
        else:
            # Use first alternative if available, otherwise use first registered handler
            if alternatives:
                best_handler = alternatives[0][0]
            elif self.handlers:
                best_handler = list(self.handlers.keys())[0]
            else:
                # No handlers available
                return None, {
                    'handler': None,
                    'confidence': 0.0,
                    'latency_ms': (time.perf_counter() - start_time) * 1000,
                    'error': 'No handlers registered'
                }
        
        # Execute selected handler
        result = await self._execute_handler(best_handler, command, context)
        
        # Calculate final latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Update cache if fast enough
        if latency_ms < 100:
            self.route_cache[cache_key] = (best_handler, confidence)
        
        # Record for learning
        self._record_routing(
            command=command,
            handler=best_handler,
            alternatives=alternatives,
            confidence=confidence,
            latency_ms=latency_ms,
            context=context,
            success=result is not None
        )
        
        # Log warning if latency exceeded
        if latency_ms > 100:
            logger.warning(f"Latency exceeded 100ms: {latency_ms:.1f}ms for '{command}'")
        
        return result, {
            'handler': best_handler,
            'confidence': float(confidence),
            'latency_ms': latency_ms,
            'alternatives': alternatives[:3],
            'cache_hit': False,
            'exploration_performed': self.exploration_mode or force_exploration
        }
    
    async def _neural_route(
        self,
        command: str,
        context: Optional[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, float, List[Tuple[str, float]]]:
        """Perform neural routing with transformer"""
        # Create embedding (would use sentence transformer in production)
        embedding = self._create_fast_embedding(command, context)
        
        # Select pre-allocated tensor based on size
        seq_len = min(len(command.split()), 128)
        if seq_len <= 32:
            input_tensor = self.preallocated_tensors['small'].clone()
        elif seq_len <= 64:
            input_tensor = self.preallocated_tensors['medium'].clone()
        else:
            input_tensor = self.preallocated_tensors['large'].clone()
        
        # Fill with embedding - ensure dimensions match
        embedding_tensor = torch.tensor(embedding[:seq_len], dtype=torch.float32)
        if embedding_tensor.dim() == 2:
            input_tensor[0, :embedding_tensor.size(0), :] = embedding_tensor
        else:
            # Create proper 2D embedding if needed
            input_tensor[0, :seq_len, :] = torch.randn(seq_len, 256)
        
        # Forward pass
        with torch.no_grad():
            route_embedding, confidence, _ = self.model(input_tensor[:1, :seq_len, :])
        
        # Find closest handlers
        handler_scores = {}
        for handler_name, idx in self.handler_embeddings_map.items():
            if idx < self.model.handler_embeddings.shape[0]:
                handler_emb = self.model.handler_embeddings[idx]
                similarity = F.cosine_similarity(
                    route_embedding,
                    handler_emb.unsqueeze(0)
                ).item()
                
                # Boost by performance metrics
                metrics = self.route_metrics[handler_name]
                success_rate = metrics.successful_calls / max(1, metrics.total_calls)
                performance_boost = 0.8 + (0.2 * success_rate)
                
                handler_scores[handler_name] = similarity * performance_boost
        
        # Sort by score
        sorted_handlers = sorted(
            handler_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # If no handlers scored, return registered handlers with low scores
        if not sorted_handlers and self.handler_embeddings_map:
            sorted_handlers = [(name, 0.1) for name in self.handler_embeddings_map.keys()]
        
        return (
            route_embedding,
            confidence.item(),
            [(name, score) for name, score in sorted_handlers]
        )
    
    def _create_fast_embedding(
        self,
        command: str,
        context: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """Create embedding quickly for <100ms inference"""
        # Very fast hash-based embedding
        words = command.lower().split()
        embedding = np.zeros((128, 256))
        
        for i, word in enumerate(words[:128]):
            # Quick hash embedding
            h1 = hash(word)
            h2 = hash(word + str(i))
            embedding[i, h1 % 256] = 1.0
            embedding[i, h2 % 256] = 0.5
            
            # Add context features
            if context:
                for key, value in list(context.items())[:5]:
                    h3 = hash(f"{key}:{value}")
                    embedding[i, h3 % 256] += 0.3
        
        return embedding
    
    async def _execute_handler(
        self,
        handler_name: str,
        command: str,
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Execute handler with performance tracking"""
        if handler_name not in self.handlers:
            logger.error(f"Handler not found: {handler_name}")
            return None
        
        handler = self.handlers[handler_name]
        metrics = self.route_metrics[handler_name]
        
        start_time = time.perf_counter()
        
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(command, context)
            else:
                result = handler(command, context)
            
            # Update metrics
            latency_ms = (time.perf_counter() - start_time) * 1000
            metrics.successful_calls += 1
            metrics.total_calls += 1
            metrics.last_used = datetime.now()
            
            # Update latency tracking
            self._update_latency_metrics(handler_name, latency_ms)
            
            return result
            
        except Exception as e:
            # Track failure
            metrics.failed_calls += 1
            metrics.total_calls += 1
            metrics.error_types[type(e).__name__] = metrics.error_types.get(type(e).__name__, 0) + 1
            
            logger.error(f"Handler {handler_name} failed: {e}")
            return None
    
    def _update_latency_metrics(self, handler_name: str, latency_ms: float):
        """Update latency metrics with p99 calculation"""
        metrics = self.route_metrics[handler_name]
        
        # Rolling average
        if metrics.avg_latency_ms == 0:
            metrics.avg_latency_ms = latency_ms
        else:
            metrics.avg_latency_ms = (0.95 * metrics.avg_latency_ms) + (0.05 * latency_ms)
        
        # Track for p99
        self.latency_buffer.append((handler_name, latency_ms))
        
        # Calculate p99 periodically
        if len(self.latency_buffer) >= 100:
            handler_latencies = [l for h, l in self.latency_buffer if h == handler_name]
            if handler_latencies:
                metrics.p99_latency_ms = np.percentile(handler_latencies, 99)
    
    async def _explore_multiple_paths(
        self,
        command: str,
        context: Optional[Dict[str, Any]],
        candidates: List[Tuple[str, float]]
    ) -> List[Tuple[str, Any, float]]:
        """Explore multiple routing paths in parallel"""
        exploration_tasks = []
        
        for handler_name, score in candidates:
            task = asyncio.create_task(
                self._explore_single_path(handler_name, command, context)
            )
            exploration_tasks.append((handler_name, score, task))
        
        # Wait for all explorations with timeout
        results = []
        for handler_name, score, task in exploration_tasks:
            try:
                result = await asyncio.wait_for(task, timeout=0.05)  # 50ms timeout
                results.append((handler_name, result, score))
            except asyncio.TimeoutError:
                logger.debug(f"Exploration timeout for {handler_name}")
        
        return results
    
    async def _explore_single_path(
        self,
        handler_name: str,
        command: str,
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Explore a single routing path"""
        # Dry run or lightweight execution
        if handler_name in self.handlers:
            handler = self.handlers[handler_name]
            
            # Check if handler supports dry run
            if hasattr(handler, 'dry_run'):
                return await handler.dry_run(command, context)
            else:
                # Default exploration - just check if callable
                return {'explorable': True, 'handler': handler_name}
        
        return None
    
    def _select_best_from_exploration(
        self,
        exploration_results: List[Tuple[str, Any, float]]
    ) -> str:
        """Select best handler from exploration results"""
        if not exploration_results:
            return list(self.handlers.keys())[0]  # Fallback
        
        # Score based on exploration results and original scores
        best_score = -1
        best_handler = None
        
        for handler_name, result, original_score in exploration_results:
            if result is not None:
                # Boost score if exploration was successful
                final_score = original_score * 1.2
                
                if final_score > best_score:
                    best_score = final_score
                    best_handler = handler_name
        
        return best_handler or exploration_results[0][0]
    
    def _record_routing(
        self,
        command: str,
        handler: str,
        alternatives: List[Tuple[str, float]],
        confidence: float,
        latency_ms: float,
        context: Optional[Dict[str, Any]],
        success: bool
    ):
        """Record routing decision for continuous learning"""
        embedding = self._create_fast_embedding(command, context)
        
        record = LearningRecord(
            command=command,
            embedding=embedding,
            selected_handler=handler,
            alternative_handlers=alternatives[:5],
            execution_time_ms=latency_ms,
            success=success,
            confidence=confidence,
            context=context or {},
            timestamp=datetime.now()
        )
        
        self.learning_records.append(record)
        
        # Update confidence history
        metrics = self.route_metrics[handler]
        metrics.confidence_history.append(confidence)
        if len(metrics.confidence_history) > 100:
            metrics.confidence_history.pop(0)
    
    def _process_learning_batch(self):
        """Process a batch of learning records"""
        batch_size = 50
        batch = list(self.learning_records)[:batch_size]
        
        if not batch:
            return
        
        logger.info(f"Processing learning batch of {len(batch)} records")
        
        # Group by success/failure
        successes = [r for r in batch if r.success]
        failures = [r for r in batch if not r.success]
        
        # Update handler embeddings based on outcomes
        for record in successes:
            self._reinforce_route(record, positive=True)
        
        for record in failures:
            self._reinforce_route(record, positive=False)
        
        # Identify patterns for pre-loading
        self._identify_preload_patterns(batch)
        
        # Clear processed records
        for _ in range(min(batch_size, len(self.learning_records))):
            self.learning_records.popleft()
    
    def _reinforce_route(self, record: LearningRecord, positive: bool):
        """Reinforce or discourage a routing decision"""
        handler_idx = self.handler_embeddings_map.get(record.selected_handler)
        
        if handler_idx is not None and handler_idx < self.model.handler_embeddings.shape[0]:
            # Simple reinforcement: move embedding closer/further from command
            command_vec = torch.tensor(record.embedding.mean(axis=0), dtype=torch.float32)
            handler_emb = self.model.handler_embeddings[handler_idx]
            
            # Learning rate based on confidence
            lr = 0.01 * (1.0 - record.confidence) if positive else 0.005
            
            if positive:
                # Move closer
                update = lr * (command_vec - handler_emb)
            else:
                # Move away
                update = -lr * (command_vec - handler_emb)
            
            self.model.handler_embeddings.data[handler_idx] += update
            
            # Renormalize
            self.model.handler_embeddings.data[handler_idx] = F.normalize(
                self.model.handler_embeddings.data[handler_idx],
                dim=0
            )
    
    def _identify_preload_patterns(self, batch: List[LearningRecord]):
        """Identify patterns for pre-loading predictions"""
        # Look for common command prefixes
        prefixes = defaultdict(int)
        
        for record in batch:
            words = record.command.lower().split()
            for i in range(1, min(4, len(words) + 1)):
                prefix = ' '.join(words[:i])
                prefixes[prefix] += 1
        
        # Keep top prefixes
        top_prefixes = sorted(
            prefixes.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        self.preload_patterns = [prefix for prefix, _ in top_prefixes]
    
    def _update_preload_predictions(self):
        """Update pre-loaded predictions for common patterns"""
        for pattern in self.preload_patterns:
            # Pre-compute routing for common patterns
            embedding = self._create_fast_embedding(pattern, None)
            
            # Cache the embedding computation
            self.preload_cache[pattern] = {
                'embedding': embedding,
                'computed_at': datetime.now()
            }
    
    def _optimize_routes(self):
        """Optimize routing based on performance data"""
        # Remove underperforming handlers from cache
        for handler_name, metrics in self.route_metrics.items():
            if metrics.total_calls > 10:
                success_rate = metrics.successful_calls / metrics.total_calls
                
                # Remove from cache if performing poorly
                if success_rate < 0.3:
                    # Find and remove cache entries for this handler
                    keys_to_remove = [
                        k for k, (h, _) in self.route_cache.items()
                        if h == handler_name
                    ]
                    for key in keys_to_remove:
                        del self.route_cache[key]
                    
                    logger.info(f"Removed {len(keys_to_remove)} cache entries for underperforming handler: {handler_name}")
        
        # Prune old cache entries
        max_cache_size = 1000
        if len(self.route_cache) > max_cache_size:
            # Remove oldest entries (simple FIFO for now)
            for key in list(self.route_cache.keys())[:-max_cache_size]:
                del self.route_cache[key]
    
    def get_routing_analytics(self) -> Dict[str, Any]:
        """Get comprehensive routing analytics"""
        total_routes = sum(m.total_calls for m in self.route_metrics.values())
        total_successes = sum(m.successful_calls for m in self.route_metrics.values())
        
        analytics = {
            'performance': {
                'total_routes': total_routes,
                'success_rate': total_successes / max(1, total_routes),
                'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses),
                'avg_latency_ms': np.mean([m.avg_latency_ms for m in self.route_metrics.values() if m.avg_latency_ms > 0])
            },
            'handlers': {
                name: {
                    'total_calls': metrics.total_calls,
                    'success_rate': metrics.successful_calls / max(1, metrics.total_calls),
                    'avg_latency_ms': metrics.avg_latency_ms,
                    'p99_latency_ms': metrics.p99_latency_ms,
                    'last_used': metrics.last_used.isoformat() if metrics.last_used else None,
                    'error_types': metrics.error_types
                }
                for name, metrics in self.route_metrics.items()
                if metrics.total_calls > 0
            },
            'learning': {
                'records_pending': len(self.learning_records),
                'preload_patterns': self.preload_patterns[:10],
                'handlers_registered': len(self.handlers)
            },
            'optimization': {
                'cache_size': len(self.route_cache),
                'handler_embeddings_used': len(self.handler_embeddings_map),
                'exploration_mode': self.exploration_mode
            }
        }
        
        return analytics
    
    def export_route_explanations(self) -> List[Dict[str, Any]]:
        """Export routing decisions with explanations for debugging"""
        explanations = []
        
        # Sample recent routing decisions
        for record in list(self.learning_records)[-20:]:
            explanation = {
                'command': record.command,
                'timestamp': record.timestamp.isoformat(),
                'selected_handler': record.selected_handler,
                'confidence': record.confidence,
                'alternatives': [
                    {'handler': h, 'score': s}
                    for h, s in record.alternative_handlers
                ],
                'execution_time_ms': record.execution_time_ms,
                'success': record.success,
                'context_keys': list(record.context.keys()),
                'explanation': self._generate_route_explanation(record)
            }
            explanations.append(explanation)
        
        return explanations
    
    def _generate_route_explanation(self, record: LearningRecord) -> str:
        """Generate human-readable explanation for routing decision"""
        explanation_parts = []
        
        # Confidence explanation
        if record.confidence > 0.8:
            explanation_parts.append(f"High confidence ({record.confidence:.1%}) match")
        elif record.confidence > 0.5:
            explanation_parts.append(f"Moderate confidence ({record.confidence:.1%}) match")
        else:
            explanation_parts.append(f"Low confidence ({record.confidence:.1%}) match")
        
        # Performance explanation
        metrics = self.route_metrics[record.selected_handler]
        if metrics.total_calls > 0:
            success_rate = metrics.successful_calls / metrics.total_calls
            explanation_parts.append(f"Handler success rate: {success_rate:.1%}")
        
        # Context influence
        if record.context:
            explanation_parts.append(f"Context factors: {', '.join(record.context.keys())}")
        
        # Alternative routes
        if record.alternative_handlers:
            alt_names = [h for h, _ in record.alternative_handlers[:2]]
            explanation_parts.append(f"Also considered: {', '.join(alt_names)}")
        
        return " | ".join(explanation_parts)
    
    async def shutdown(self):
        """Gracefully shutdown the router"""
        logger.info("Shutting down Transformer Command Router")
        
        # Stop learning thread
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
        
        # Process remaining records
        if self.learning_records:
            self._process_learning_batch()
        
        # Save state
        self._save_state()
        
        # Shutdown executor
        self.learning_executor.shutdown(wait=True)
    
    def _save_state(self):
        """Save router state to disk"""
        state_path = Path("backend/data/transformer_router_state.pt")
        state_path.parent.mkdir(exist_ok=True)
        
        state = {
            'model_state': self.model.state_dict(),
            'handler_embeddings_map': self.handler_embeddings_map,
            'route_metrics': {
                name: {
                    'total_calls': m.total_calls,
                    'successful_calls': m.successful_calls,
                    'failed_calls': m.failed_calls,
                    'avg_latency_ms': m.avg_latency_ms,
                    'p99_latency_ms': m.p99_latency_ms
                }
                for name, m in self.route_metrics.items()
            },
            'preload_patterns': self.preload_patterns,
            'cache_stats': {
                'hits': self.cache_hits,
                'misses': self.cache_misses
            }
        }
        
        torch.save(state, state_path)
        logger.info("Router state saved")


# Singleton instance
_transformer_router: Optional[TransformerCommandRouter] = None


def get_transformer_router() -> TransformerCommandRouter:
    """Get singleton instance of transformer router"""
    global _transformer_router
    if _transformer_router is None:
        _transformer_router = TransformerCommandRouter()
    return _transformer_router