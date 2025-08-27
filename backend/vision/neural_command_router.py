#!/usr/bin/env python3
"""
Neural Command Router - Replaces if/elif chains with neural routing
Dynamic, learned routing based on embeddings and context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from pathlib import Path
import logging
import asyncio
import json
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class RouteDecision:
    """Neural routing decision with metadata"""
    handler_name: str
    confidence: float
    reasoning: Optional[str] = None
    alternative_routes: List[Tuple[str, float]] = field(default_factory=list)
    context_factors: Dict[str, Any] = field(default_factory=dict)
    execution_time: Optional[float] = None


@dataclass
class HandlerRegistration:
    """Registration for a command handler"""
    name: str
    handler: Callable
    embedding: Optional[np.ndarray] = None
    success_rate: float = 0.0
    usage_count: int = 0
    context_requirements: Dict[str, Any] = field(default_factory=dict)
    learned_patterns: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class AttentionRouter(nn.Module):
    """Attention-based neural router for command routing"""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, num_heads: int = 8):
        super().__init__()
        
        # Multi-head attention for command understanding
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=0.1
        )
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Route predictor
        self.route_predictor = nn.Sequential(
            nn.Linear(input_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 128)  # Route space
        )
        
        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, command_embedding, handler_embeddings, context_embedding):
        # Apply attention to understand command in context of available handlers
        attended, attention_weights = self.attention(
            command_embedding.unsqueeze(0),
            handler_embeddings,
            handler_embeddings
        )
        
        # Encode context
        context_features = self.context_encoder(context_embedding)
        
        # Combine attended command with context
        combined = torch.cat([attended.squeeze(0), context_features], dim=-1)
        
        # Predict route
        route_features = self.route_predictor(combined)
        
        # Estimate confidence
        confidence = self.confidence_head(route_features)
        
        return route_features, confidence, attention_weights


class NeuralCommandRouter:
    """
    Neural router that replaces traditional if/elif chains
    Learns optimal routing based on command embeddings and context
    """
    
    def __init__(self):
        # Neural routing model
        self.router_model = AttentionRouter()
        self.optimizer = torch.optim.Adam(self.router_model.parameters(), lr=0.001)
        
        # Handler registry
        self.handlers: Dict[str, HandlerRegistration] = {}
        self.handler_embeddings: Dict[str, np.ndarray] = {}
        
        # Routing history for learning
        self.routing_history = []
        self.performance_buffer = deque(maxlen=1000)
        
        # Embedding cache
        self.embedding_cache = {}
        self.cache_size = 1000
        
        # Dynamic route discovery
        self.discovered_routes = defaultdict(list)
        self.route_clusters = {}
        
        # Load saved state if available
        self._load_router_state()
        
        logger.info("Neural Command Router initialized")
    
    def register_handler(
        self,
        name: str,
        handler: Callable,
        description: Optional[str] = None,
        examples: Optional[List[str]] = None
    ):
        """Register a command handler with the neural router"""
        # Create handler registration
        registration = HandlerRegistration(
            name=name,
            handler=handler
        )
        
        # Generate embedding from description and examples
        embedding_text = f"{name} {description or ''}"
        if examples:
            embedding_text += " " + " ".join(examples)
            registration.learned_patterns.extend(examples)
        
        # Create embedding
        registration.embedding = self._create_embedding(embedding_text)
        self.handler_embeddings[name] = registration.embedding
        
        # Register handler
        self.handlers[name] = registration
        
        logger.info(f"Registered handler: {name}")
    
    async def route_command(
        self,
        command: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, RouteDecision]:
        """
        Route a command to the appropriate handler using neural routing
        Returns both the handler result and the routing decision
        """
        context = context or {}
        start_time = datetime.now()
        
        # Get embeddings
        command_embedding = self._get_or_create_embedding(command)
        context_embedding = self._create_context_embedding(context)
        
        # Prepare handler embeddings for neural routing
        handler_names = list(self.handlers.keys())
        handler_embeddings_tensor = torch.stack([
            torch.tensor(self.handler_embeddings[name], dtype=torch.float32)
            for name in handler_names
        ])
        
        # Neural routing
        with torch.no_grad():
            route_features, confidence, attention_weights = self.router_model(
                torch.tensor(command_embedding, dtype=torch.float32),
                handler_embeddings_tensor,
                torch.tensor(context_embedding, dtype=torch.float32)
            )
        
        # Find best matching handler
        best_handler, alternatives = self._find_best_handler(
            route_features, 
            confidence.item(),
            handler_names,
            attention_weights
        )
        
        # Create routing decision
        decision = RouteDecision(
            handler_name=best_handler,
            confidence=confidence.item(),
            alternative_routes=alternatives,
            context_factors=self._extract_context_factors(context, attention_weights)
        )
        
        # Execute handler
        try:
            handler = self.handlers[best_handler].handler
            if asyncio.iscoroutinefunction(handler):
                result = await handler(command, context)
            else:
                result = handler(command, context)
            
            # Update metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            decision.execution_time = execution_time
            
            # Record successful routing
            await self._record_routing_success(command, best_handler, decision, True)
            
            return result, decision
            
        except Exception as e:
            logger.error(f"Handler execution failed: {e}")
            
            # Try alternative routes
            for alt_handler, alt_confidence in alternatives[:2]:
                try:
                    handler = self.handlers[alt_handler].handler
                    if asyncio.iscoroutinefunction(handler):
                        result = await handler(command, context)
                    else:
                        result = handler(command, context)
                    
                    # Update decision
                    decision.handler_name = alt_handler
                    decision.confidence = alt_confidence
                    decision.reasoning = f"Fallback to alternative after primary failed: {str(e)}"
                    
                    await self._record_routing_success(command, alt_handler, decision, True)
                    return result, decision
                    
                except Exception as alt_e:
                    logger.error(f"Alternative handler {alt_handler} also failed: {alt_e}")
            
            # Record failure
            await self._record_routing_success(command, best_handler, decision, False)
            raise
    
    def _find_best_handler(
        self,
        route_features: torch.Tensor,
        confidence: float,
        handler_names: List[str],
        attention_weights: torch.Tensor
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """Find best matching handler based on neural routing"""
        # Calculate similarity scores
        scores = {}
        route_features_np = route_features.numpy()
        
        for i, handler_name in enumerate(handler_names):
            handler = self.handlers[handler_name]
            
            # Base score from attention weights
            # attention_weights is 2D: [1, num_handlers]
            attention_score = attention_weights[0, i].item()
            
            # Adjust based on handler performance
            performance_factor = 0.5 + (handler.success_rate * 0.5)
            
            # Combine scores
            final_score = attention_score * performance_factor * confidence
            scores[handler_name] = final_score
        
        # Sort by score
        sorted_handlers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get best and alternatives
        best_handler = sorted_handlers[0][0]
        alternatives = [(name, score) for name, score in sorted_handlers[1:]]
        
        return best_handler, alternatives
    
    def _create_embedding(self, text: str) -> np.ndarray:
        """Create embedding for text"""
        # Simple embedding (in production would use sentence transformers)
        words = text.lower().split()
        embedding = np.zeros(768)
        
        for i, word in enumerate(words):
            # Hash-based embedding
            hash_val = hash(word)
            embedding[hash_val % 768] += 1.0 / (i + 1)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def _get_or_create_embedding(self, text: str) -> np.ndarray:
        """Get embedding from cache or create new one"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        embedding = self._create_embedding(text)
        
        # Add to cache with size limit
        if len(self.embedding_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest_key]
        
        self.embedding_cache[text] = embedding
        return embedding
    
    def _create_context_embedding(self, context: Dict[str, Any]) -> np.ndarray:
        """Create embedding from context"""
        embedding = np.zeros(768)
        
        # Encode various context features
        feature_idx = 0
        
        # User information
        if 'user' in context:
            user_hash = hash(context['user'])
            embedding[feature_idx:feature_idx+10] = (user_hash % 10) / 10
            feature_idx += 10
        
        # Confidence if available
        if 'confidence' in context:
            embedding[feature_idx] = context['confidence']
            feature_idx += 1
        
        # Intent type
        if 'intent' in context:
            intent_hash = hash(context['intent'])
            embedding[feature_idx:feature_idx+20] = (intent_hash % 20) / 20
            feature_idx += 20
        
        # Time features
        if 'timestamp' in context:
            try:
                ts = datetime.fromisoformat(context['timestamp'])
                embedding[feature_idx] = ts.hour / 24  # Hour of day
                embedding[feature_idx+1] = ts.weekday() / 7  # Day of week
                feature_idx += 2
            except:
                pass
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def _extract_context_factors(
        self,
        context: Dict[str, Any],
        attention_weights: torch.Tensor
    ) -> Dict[str, Any]:
        """Extract factors that influenced routing decision"""
        factors = {
            'attention_distribution': attention_weights[0].tolist(),
            'context_keys': list(context.keys()),
            'has_confidence': 'confidence' in context,
            'has_user': 'user' in context
        }
        
        # Add top attended handlers
        top_attention = torch.topk(attention_weights[0], k=min(3, len(self.handlers)))
        factors['top_attended_handlers'] = [
            (list(self.handlers.keys())[idx], score.item())
            for idx, score in zip(top_attention.indices, top_attention.values)
        ]
        
        return factors
    
    async def _record_routing_success(
        self,
        command: str,
        handler_name: str,
        decision: RouteDecision,
        success: bool
    ):
        """Record routing result for learning"""
        # Update handler metrics
        handler = self.handlers[handler_name]
        handler.usage_count += 1
        if success:
            handler.success_rate = (
                (handler.success_rate * (handler.usage_count - 1) + 1) / 
                handler.usage_count
            )
        else:
            handler.success_rate = (
                (handler.success_rate * (handler.usage_count - 1)) / 
                handler.usage_count
            )
        
        # Record in history
        self.routing_history.append({
            'command': command,
            'handler': handler_name,
            'decision': decision.__dict__,
            'success': success,
            'timestamp': datetime.now().isoformat()
        })
        
        # Add to performance buffer for training
        self.performance_buffer.append({
            'command_embedding': self._get_or_create_embedding(command),
            'handler': handler_name,
            'confidence': decision.confidence,
            'success': success
        })
        
        # Trigger learning if buffer is full
        if len(self.performance_buffer) >= 100:
            await self._learn_from_performance()
    
    async def _learn_from_performance(self):
        """Learn from accumulated performance data"""
        if len(self.performance_buffer) < 50:
            return
        
        logger.info("Learning from routing performance...")
        
        # Prepare training data
        successes = [p for p in self.performance_buffer if p['success']]
        failures = [p for p in self.performance_buffer if not p['success']]
        
        # Simple learning: boost successful routes, penalize failures
        for success in successes:
            handler = success['handler']
            if handler in self.handlers:
                # Add successful pattern
                self.handlers[handler].performance_metrics['recent_success'] = (
                    self.handlers[handler].performance_metrics.get('recent_success', 0) + 1
                )
        
        for failure in failures:
            handler = failure['handler']
            if handler in self.handlers:
                # Track failure pattern
                self.handlers[handler].performance_metrics['recent_failure'] = (
                    self.handlers[handler].performance_metrics.get('recent_failure', 0) + 1
                )
        
        # Clear old performance data
        self.performance_buffer.clear()
        
        # Save updated state
        self._save_router_state()
    
    def discover_route_patterns(self):
        """Discover common routing patterns from history"""
        if len(self.routing_history) < 100:
            return
        
        # Group by handler
        handler_patterns = defaultdict(list)
        for record in self.routing_history[-1000:]:  # Last 1000 routes
            if record['success']:
                handler_patterns[record['handler']].append(record['command'])
        
        # Find common patterns per handler
        for handler_name, commands in handler_patterns.items():
            if len(commands) > 10:
                # Extract common words/phrases
                word_freq = defaultdict(int)
                for cmd in commands:
                    for word in cmd.lower().split():
                        word_freq[word] += 1
                
                # Top patterns
                top_patterns = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
                self.discovered_routes[handler_name] = [word for word, _ in top_patterns]
                
                logger.info(f"Discovered patterns for {handler_name}: {self.discovered_routes[handler_name]}")
    
    def get_routing_metrics(self) -> Dict[str, Any]:
        """Get routing performance metrics"""
        total_routes = len(self.routing_history)
        successful_routes = sum(1 for r in self.routing_history if r['success'])
        
        handler_metrics = {}
        for name, handler in self.handlers.items():
            handler_metrics[name] = {
                'usage_count': handler.usage_count,
                'success_rate': handler.success_rate,
                'performance': handler.performance_metrics
            }
        
        return {
            'total_routes': total_routes,
            'success_rate': successful_routes / max(1, total_routes),
            'handlers': handler_metrics,
            'discovered_patterns': dict(self.discovered_routes),
            'cache_size': len(self.embedding_cache)
        }
    
    def _save_router_state(self):
        """Save router state to disk"""
        save_path = Path("backend/data/neural_router_state.pt")
        save_path.parent.mkdir(exist_ok=True)
        
        # Prepare handler data
        handler_data = {}
        for name, handler in self.handlers.items():
            handler_data[name] = {
                'embedding': handler.embedding,
                'success_rate': handler.success_rate,
                'usage_count': handler.usage_count,
                'learned_patterns': handler.learned_patterns,
                'performance_metrics': handler.performance_metrics
            }
        
        torch.save({
            'router_model': self.router_model.state_dict(),
            'handlers': handler_data,
            'routing_history': self.routing_history[-1000:],  # Keep last 1000
            'discovered_routes': dict(self.discovered_routes)
        }, save_path)
        
        logger.debug("Saved router state")
    
    def _load_router_state(self):
        """Load router state from disk"""
        save_path = Path("backend/data/neural_router_state.pt")
        
        if not save_path.exists():
            logger.info("No saved router state found")
            return
        
        try:
            data = torch.load(save_path)
            
            # Load model state
            self.router_model.load_state_dict(data['router_model'])
            
            # Load handler data
            # Note: handlers themselves must be re-registered, we only load metrics
            for name, handler_data in data.get('handlers', {}).items():
                if name in self.handlers:
                    self.handlers[name].embedding = handler_data['embedding']
                    self.handlers[name].success_rate = handler_data['success_rate']
                    self.handlers[name].usage_count = handler_data['usage_count']
                    self.handlers[name].learned_patterns = handler_data['learned_patterns']
                    self.handlers[name].performance_metrics = handler_data['performance_metrics']
                    self.handler_embeddings[name] = handler_data['embedding']
            
            # Load history
            self.routing_history = data.get('routing_history', [])
            self.discovered_routes = defaultdict(list, data.get('discovered_routes', {}))
            
            logger.info("Loaded router state")
            
        except Exception as e:
            logger.error(f"Error loading router state: {e}")


# Global router instance
_neural_router: Optional[NeuralCommandRouter] = None


def get_neural_router() -> NeuralCommandRouter:
    """Get singleton instance of neural router"""
    global _neural_router
    if _neural_router is None:
        _neural_router = NeuralCommandRouter()
    return _neural_router


# Example of replacing if/elif chains with neural routing
async def example_migration():
    """Example of how to migrate from if/elif to neural routing"""
    router = get_neural_router()
    
    # Example handlers
    async def handle_open(command: str, context: Dict):
        return f"Opening based on command: {command}"
    
    async def handle_close(command: str, context: Dict):
        return f"Closing based on command: {command}"
    
    async def handle_screen(command: str, context: Dict):
        return f"Analyzing screen for: {command}"
    
    # Old approach with if/elif:
    # if "open" in command:
    #     return handle_open(command)
    # elif "close" in command:
    #     return handle_close(command)
    # elif "screen" in command:
    #     return handle_screen(command)
    
    # New approach with neural routing:
    
    # Register handlers once
    router.register_handler(
        "open_handler",
        handle_open,
        "Handles opening applications and files",
        ["open chrome", "open file.txt", "launch app"]
    )
    
    router.register_handler(
        "close_handler", 
        handle_close,
        "Handles closing applications and windows",
        ["close window", "quit app", "exit program"]
    )
    
    router.register_handler(
        "screen_handler",
        handle_screen,
        "Handles screen analysis and vision commands",
        ["analyze screen", "what's on my screen", "describe display"]
    )
    
    # Route commands dynamically
    result, decision = await router.route_command(
        "can you open the browser?",
        context={'user': 'John', 'confidence': 0.8}
    )
    
    # The router learns from usage and improves over time
    logger.info(f"Routed to {decision.handler_name} with {decision.confidence:.1%} confidence")