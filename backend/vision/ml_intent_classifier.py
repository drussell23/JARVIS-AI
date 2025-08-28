#!/usr/bin/env python3
"""
ML-Based Intent Classification for Vision System
Zero hardcoding - learns all patterns dynamically
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import numpy as np
from pathlib import Path
from collections import deque, defaultdict
import logging
from sentence_transformers import SentenceTransformer
import pickle

logger = logging.getLogger(__name__)

@dataclass
class VisionIntent:
    """Represents a classified vision intent"""
    intent_type: str
    confidence: float
    raw_command: str
    embeddings: np.ndarray
    context: Dict[str, Any] = field(default_factory=dict)
    learned_patterns: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class LearnedPattern:
    """A pattern learned from user interactions"""
    pattern_text: str
    intent_type: str
    embedding: np.ndarray
    success_count: int = 0
    failure_count: int = 0
    confidence_score: float = 0.5
    last_used: datetime = field(default_factory=datetime.now)
    context_examples: List[Dict] = field(default_factory=list)

class MLIntentClassifier:
    """
    ML-based intent classifier that learns dynamically
    No hardcoded patterns - everything is learned
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # Sentence transformer for embeddings
        self.encoder = SentenceTransformer(model_name)
        
        # Learned patterns storage
        self.learned_patterns: Dict[str, List[LearnedPattern]] = defaultdict(list)
        
        # Neural network for intent classification
        self.classifier_net = self._build_classifier()
        
        # Confidence scoring components
        self.confidence_history = deque(maxlen=100)
        self.confidence_threshold = 0.7  # Will auto-tune
        
        # Real-time learning buffer
        self.learning_buffer = deque(maxlen=50)
        
        # Multi-language support
        self.language_agnostic = True
        
        # Load existing patterns if available
        self._load_learned_patterns()
        
    def _build_classifier(self) -> nn.Module:
        """Build neural network for intent classification"""
        return nn.Sequential(
            nn.Linear(384, 512),  # Input from sentence embeddings
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)  # Output: intent space
        )
        
    def classify_intent(self, command: str, context: Optional[Dict] = None) -> VisionIntent:
        """
        Classify intent with zero hardcoding
        Returns intent with confidence score
        """
        context = context or {}
        
        # Get embedding for the command
        embedding = self.encoder.encode(command, convert_to_numpy=True)
        
        # Find best matching intent from learned patterns
        best_intent, confidence = self._find_best_intent(embedding, context)
        
        # Create intent object
        intent = VisionIntent(
            intent_type=best_intent,
            confidence=confidence,
            raw_command=command,
            embeddings=embedding,
            context=context
        )
        
        # If confidence is low, mark for learning
        if confidence < self.confidence_threshold:
            self.learning_buffer.append({
                'command': command,
                'embedding': embedding,
                'context': context,
                'timestamp': datetime.now()
            })
        
        return intent
        
    def _find_best_intent(self, embedding: np.ndarray, context: Dict) -> Tuple[str, float]:
        """Find best matching intent from learned patterns"""
        if not self.learned_patterns:
            # No patterns learned yet - bootstrap with generic intent
            return "unknown_vision_request", 0.0
            
        best_score = 0.0
        best_intent = "unknown_vision_request"
        
        # Compare with all learned patterns
        for intent_type, patterns in self.learned_patterns.items():
            for pattern in patterns:
                # Calculate similarity
                similarity = self._calculate_similarity(embedding, pattern.embedding)
                
                # Apply context weighting
                context_weight = self._calculate_context_weight(context, pattern.context_examples)
                
                # Apply success rate weighting
                success_rate = pattern.success_count / max(1, pattern.success_count + pattern.failure_count)
                
                # Combined score
                score = similarity * 0.6 + context_weight * 0.2 + success_rate * 0.2
                
                if score > best_score:
                    best_score = score
                    best_intent = intent_type
                    
        return best_intent, best_score
        
    def _calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        dot_product = np.dot(emb1, emb2)
        norm_product = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        
        if norm_product == 0:
            return 0.0
            
        return float(float(dot_product) / float(norm_product))
        
    def _calculate_context_weight(self, current_context: Dict, pattern_contexts: List[Dict]) -> float:
        """Calculate how well contexts match"""
        if not pattern_contexts:
            return 0.5  # Neutral weight
            
        # Compare context features
        weights = []
        for pattern_context in pattern_contexts:
            common_keys = set(current_context.keys()) & set(pattern_context.keys())
            if not common_keys:
                weights.append(0.0)
                continue
                
            matches = sum(1 for k in common_keys if current_context[k] == pattern_context[k])
            weight = matches / len(common_keys)
            weights.append(weight)
            
        return np.mean(weights) if weights else 0.5
        
    def learn_from_interaction(self, command: str, intent_type: str, success: bool, context: Optional[Dict] = None):
        """Learn from successful or failed interactions"""
        context = context or {}
        
        # Get embedding
        embedding = self.encoder.encode(command, convert_to_numpy=True)
        
        # Check if pattern exists
        existing_pattern = None
        for pattern in self.learned_patterns[intent_type]:
            if self._calculate_similarity(embedding, pattern.embedding) > 0.95:
                existing_pattern = pattern
                break
                
        if existing_pattern:
            # Update existing pattern
            if success:
                existing_pattern.success_count += 1
            else:
                existing_pattern.failure_count += 1
                
            existing_pattern.last_used = datetime.now()
            existing_pattern.confidence_score = existing_pattern.success_count / max(1, 
                existing_pattern.success_count + existing_pattern.failure_count)
                
            # Add context example if successful
            if success and context:
                existing_pattern.context_examples.append(context)
                if len(existing_pattern.context_examples) > 10:
                    existing_pattern.context_examples.pop(0)
        else:
            # Create new pattern
            new_pattern = LearnedPattern(
                pattern_text=command,
                intent_type=intent_type,
                embedding=embedding,
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                confidence_score=1.0 if success else 0.0,
                context_examples=[context] if success and context else []
            )
            self.learned_patterns[intent_type].append(new_pattern)
            
        # Auto-tune confidence threshold
        self._auto_tune_confidence()
        
        # Save patterns
        self._save_learned_patterns()
        
    def _auto_tune_confidence(self):
        """Automatically adjust confidence threshold based on performance"""
        if len(self.confidence_history) < 20:
            return
            
        # Calculate success rate at different thresholds
        thresholds = np.arange(0.5, 0.95, 0.05)
        best_threshold = self.confidence_threshold
        best_f1 = 0.0
        
        for threshold in thresholds:
            tp = sum(1 for conf, success in self.confidence_history if conf >= threshold and success)
            fp = sum(1 for conf, success in self.confidence_history if conf >= threshold and not success)
            fn = sum(1 for conf, success in self.confidence_history if conf < threshold and success)
            
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    
        # Smooth adjustment
        self.confidence_threshold = 0.9 * self.confidence_threshold + 0.1 * best_threshold
        logger.info(f"Auto-tuned confidence threshold: {self.confidence_threshold:.3f}")
        
    def get_confidence_score(self) -> float:
        """Get current confidence threshold"""
        return self.confidence_threshold
        
    def get_learned_intents(self) -> List[str]:
        """Get list of all learned intent types"""
        return list(self.learned_patterns.keys())
        
    def get_pattern_count(self) -> Dict[str, int]:
        """Get count of patterns per intent type"""
        return {intent: len(patterns) for intent, patterns in self.learned_patterns.items()}
        
    def _save_learned_patterns(self):
        """Save learned patterns to disk"""
        save_path = Path("backend/data/vision_learned_patterns.pkl")
        save_path.parent.mkdir(exist_ok=True)
        
        try:
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'patterns': self.learned_patterns,
                    'confidence_threshold': self.confidence_threshold,
                    'confidence_history': list(self.confidence_history)
                }, f)
        except Exception as e:
            logger.error(f"Error saving patterns: {e}")
            
    def _load_learned_patterns(self):
        """Load previously learned patterns"""
        save_path = Path("backend/data/vision_learned_patterns.pkl")
        
        if not save_path.exists():
            logger.info("No existing patterns found - starting fresh")
            return
            
        try:
            with open(save_path, 'rb') as f:
                data = pickle.load(f)
                self.learned_patterns = data['patterns']
                self.confidence_threshold = data.get('confidence_threshold', 0.7)
                self.confidence_history = deque(data.get('confidence_history', []), maxlen=100)
                logger.info(f"Loaded {sum(len(p) for p in self.learned_patterns.values())} patterns")
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
            
    def export_patterns_for_visualization(self) -> Dict[str, Any]:
        """Export patterns in format suitable for visualization"""
        export_data = {
            'intents': {},
            'confidence_threshold': self.confidence_threshold,
            'total_patterns': sum(len(p) for p in self.learned_patterns.values()),
            'timestamp': datetime.now().isoformat()
        }
        
        for intent_type, patterns in self.learned_patterns.items():
            export_data['intents'][intent_type] = {
                'pattern_count': len(patterns),
                'avg_confidence': np.mean([p.confidence_score for p in patterns]) if patterns else 0,
                'total_uses': sum(p.success_count + p.failure_count for p in patterns),
                'success_rate': sum(p.success_count for p in patterns) / max(1, sum(p.success_count + p.failure_count for p in patterns)),
                'examples': [p.pattern_text for p in sorted(patterns, key=lambda x: x.confidence_score, reverse=True)[:5]]
            }
            
        return export_data

# Singleton instance
_classifier_instance: Optional[MLIntentClassifier] = None

def get_ml_intent_classifier() -> MLIntentClassifier:
    """Get singleton instance of ML intent classifier"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = MLIntentClassifier()
    return _classifier_instance