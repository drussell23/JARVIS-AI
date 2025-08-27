#!/usr/bin/env python3
"""
Personalization Engine for Vision System
Learns user communication styles and preferences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import numpy as np
from pathlib import Path
import logging
from collections import defaultdict, deque
import re
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """Complete user profile with preferences and patterns"""
    user_id: str
    communication_style: str = "balanced"  # concise, balanced, verbose
    tone_preference: str = "professional"  # casual, professional, technical
    response_format: str = "text"  # text, json, markdown
    language: str = "en"
    
    # Learned patterns
    vocab_complexity: float = 0.5  # 0-1 scale
    avg_response_length: float = 100.0
    interaction_times: List[datetime] = field(default_factory=list)
    preferred_phrases: List[str] = field(default_factory=list)
    avoided_phrases: List[str] = field(default_factory=list)
    
    # Behavioral patterns
    patience_level: float = 0.7  # 0-1, affects response timing
    detail_preference: float = 0.5  # 0-1, affects response depth
    emoji_preference: float = 0.0  # 0-1, likelihood of emoji use
    
    # Performance metrics
    satisfaction_scores: List[float] = field(default_factory=list)
    interaction_count: int = 0
    last_interaction: Optional[datetime] = None
    
    # Context preferences
    context_awareness: Dict[str, Any] = field(default_factory=dict)
    topic_interests: Dict[str, float] = field(default_factory=dict)


@dataclass
class InteractionPattern:
    """Pattern learned from user interactions"""
    pattern_type: str  # query_style, response_preference, timing, etc.
    pattern_value: Any
    confidence: float
    occurrences: int = 1
    last_seen: datetime = field(default_factory=datetime.now)
    context: Optional[Dict[str, Any]] = None


class UserStyleAnalyzer(nn.Module):
    """Neural network for analyzing user communication style"""
    
    def __init__(self, input_dim: int = 768, style_dim: int = 128):
        super().__init__()
        
        self.style_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, style_dim)
        )
        
        # Style-specific classifiers
        self.verbosity_classifier = nn.Linear(style_dim, 3)  # concise, balanced, verbose
        self.tone_classifier = nn.Linear(style_dim, 3)  # casual, professional, technical
        self.complexity_regressor = nn.Linear(style_dim, 1)  # 0-1 complexity score
        
    def forward(self, interaction_embedding):
        style_features = self.style_encoder(interaction_embedding)
        
        verbosity = F.softmax(self.verbosity_classifier(style_features), dim=-1)
        tone = F.softmax(self.tone_classifier(style_features), dim=-1)
        complexity = torch.sigmoid(self.complexity_regressor(style_features))
        
        return style_features, verbosity, tone, complexity


class PersonalizationEngine:
    """
    Learns and applies user-specific preferences and communication styles
    Provides personalized response generation guidance
    """
    
    def __init__(self):
        # User profiles
        self.user_profiles: Dict[str, UserProfile] = {}
        self.profile_embeddings: Dict[str, np.ndarray] = {}
        
        # Pattern recognition
        self.interaction_patterns: Dict[str, List[InteractionPattern]] = defaultdict(list)
        self.pattern_clusters: Dict[str, KMeans] = {}
        
        # Neural components
        self.style_analyzer = UserStyleAnalyzer()
        
        # Learning buffers
        self.interaction_buffer = deque(maxlen=1000)
        self.feedback_buffer = deque(maxlen=500)
        
        # Phrase analysis
        self.phrase_frequencies: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.response_effectiveness: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Time pattern analysis
        self.time_patterns: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        
        # Load saved profiles
        self._load_profiles()
        
        logger.info("Personalization Engine initialized")
    
    def get_or_create_profile(self, user_id: str) -> UserProfile:
        """Get existing profile or create new one"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)
            logger.info(f"Created new profile for user: {user_id}")
        
        return self.user_profiles[user_id]
    
    async def analyze_user_style(
        self,
        user_id: str,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze user's communication style from their message"""
        profile = self.get_or_create_profile(user_id)
        
        # Extract style features
        style_features = self._extract_style_features(message)
        
        # Create embedding
        embedding = self._create_message_embedding(message, context)
        
        # Neural style analysis
        with torch.no_grad():
            style_tensor = torch.tensor(embedding, dtype=torch.float32)
            style_repr, verbosity, tone, complexity = self.style_analyzer(style_tensor)
        
        # Update profile based on analysis
        self._update_profile_from_analysis(profile, style_features, verbosity, tone, complexity)
        
        # Detect patterns
        patterns = self._detect_interaction_patterns(user_id, message, context)
        
        # Record interaction
        self._record_interaction(user_id, message, style_features, context)
        
        return {
            'communication_style': profile.communication_style,
            'tone_preference': profile.tone_preference,
            'complexity_level': float(complexity.item()),
            'detected_patterns': patterns,
            'style_features': style_features
        }
    
    def _extract_style_features(self, message: str) -> Dict[str, Any]:
        """Extract style features from message"""
        words = message.split()
        sentences = re.split(r'[.!?]', message)
        
        features = {
            'message_length': len(message),
            'word_count': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'sentence_count': len([s for s in sentences if s.strip()]),
            'question_marks': message.count('?'),
            'exclamations': message.count('!'),
            'emojis': len(re.findall(r'[\U00010000-\U0010ffff]', message)),
            'technical_terms': self._count_technical_terms(message),
            'politeness_markers': self._count_politeness_markers(message),
            'urgency_markers': self._detect_urgency(message)
        }
        
        return features
    
    def _count_technical_terms(self, message: str) -> int:
        """Count technical terminology"""
        technical_terms = {
            'api', 'algorithm', 'backend', 'frontend', 'database', 'function',
            'variable', 'parameter', 'debug', 'compile', 'execute', 'process'
        }
        
        message_lower = message.lower()
        return sum(1 for term in technical_terms if term in message_lower)
    
    def _count_politeness_markers(self, message: str) -> int:
        """Count politeness markers"""
        politeness_markers = {
            'please', 'thank you', 'thanks', 'could you', 'would you',
            'kindly', 'appreciate', 'grateful'
        }
        
        message_lower = message.lower()
        return sum(1 for marker in politeness_markers if marker in message_lower)
    
    def _detect_urgency(self, message: str) -> float:
        """Detect urgency level (0-1)"""
        urgency_words = {
            'urgent': 1.0, 'asap': 1.0, 'immediately': 0.9,
            'quickly': 0.7, 'soon': 0.5, 'now': 0.8,
            'hurry': 0.9, 'emergency': 1.0
        }
        
        message_lower = message.lower()
        urgency_scores = [score for word, score in urgency_words.items() if word in message_lower]
        
        return max(urgency_scores) if urgency_scores else 0.0
    
    def _create_message_embedding(self, message: str, context: Optional[Dict]) -> np.ndarray:
        """Create embedding from message and context"""
        # Simple embedding (in production would use sentence transformers)
        embedding = np.zeros(768)
        
        # Word-based features
        words = message.lower().split()
        for i, word in enumerate(words):
            hash_val = hash(word)
            embedding[hash_val % 768] += 1.0 / (i + 1)
        
        # Context features
        if context:
            # Time of day
            if 'timestamp' in context:
                try:
                    ts = datetime.fromisoformat(context['timestamp'])
                    embedding[0] = ts.hour / 24
                    embedding[1] = ts.weekday() / 7
                except:
                    pass
            
            # Other context
            for key in ['confidence', 'urgency', 'sentiment']:
                if key in context:
                    embedding[hash(key) % 768] = float(context[key])
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def _update_profile_from_analysis(
        self,
        profile: UserProfile,
        style_features: Dict[str, Any],
        verbosity: torch.Tensor,
        tone: torch.Tensor,
        complexity: torch.Tensor
    ):
        """Update user profile based on style analysis"""
        # Update communication style
        verbosity_classes = ['concise', 'balanced', 'verbose']
        verbosity_idx = verbosity.argmax().item()
        
        # Smooth update
        if profile.interaction_count > 0:
            # Weighted average with existing preference
            current_weight = 0.7
            new_weight = 0.3
        else:
            current_weight = 0
            new_weight = 1
        
        # Update style if confident
        if verbosity[verbosity_idx] > 0.6:
            profile.communication_style = verbosity_classes[verbosity_idx]
        
        # Update tone preference
        tone_classes = ['casual', 'professional', 'technical']
        tone_idx = tone.argmax().item()
        if tone[tone_idx] > 0.6:
            profile.tone_preference = tone_classes[tone_idx]
        
        # Update complexity
        profile.vocab_complexity = (
            current_weight * profile.vocab_complexity +
            new_weight * complexity.item()
        )
        
        # Update average response length preference
        if 'word_count' in style_features:
            profile.avg_response_length = (
                current_weight * profile.avg_response_length +
                new_weight * style_features['word_count']
            )
        
        # Update emoji preference
        if 'emojis' in style_features:
            emoji_rate = style_features['emojis'] / max(1, style_features['word_count'])
            profile.emoji_preference = (
                current_weight * profile.emoji_preference +
                new_weight * emoji_rate
            )
        
        # Update detail preference based on message length
        if style_features['word_count'] > 50:
            profile.detail_preference = min(1.0, profile.detail_preference + 0.1)
        elif style_features['word_count'] < 10:
            profile.detail_preference = max(0.0, profile.detail_preference - 0.1)
    
    def _detect_interaction_patterns(
        self,
        user_id: str,
        message: str,
        context: Optional[Dict]
    ) -> List[InteractionPattern]:
        """Detect patterns in user interaction"""
        patterns = []
        
        # Query style patterns
        if message.endswith('?'):
            pattern = InteractionPattern(
                pattern_type='query_style',
                pattern_value='direct_question',
                confidence=1.0
            )
            patterns.append(pattern)
        
        # Time patterns
        if context and 'timestamp' in context:
            try:
                ts = datetime.fromisoformat(context['timestamp'])
                hour = ts.hour
                
                # Track interaction times
                self.time_patterns[user_id].append((hour, 1.0))
                
                # Detect peak hours
                if len(self.time_patterns[user_id]) > 10:
                    peak_hour = self._find_peak_hour(self.time_patterns[user_id])
                    if abs(hour - peak_hour) <= 2:
                        pattern = InteractionPattern(
                            pattern_type='timing',
                            pattern_value=f'peak_hour_{peak_hour}',
                            confidence=0.8
                        )
                        patterns.append(pattern)
            except:
                pass
        
        # Language patterns
        formal_words = {'please', 'kindly', 'would', 'could', 'shall'}
        if any(word in message.lower() for word in formal_words):
            pattern = InteractionPattern(
                pattern_type='language_style',
                pattern_value='formal',
                confidence=0.7
            )
            patterns.append(pattern)
        
        # Store patterns
        self.interaction_patterns[user_id].extend(patterns)
        
        return patterns
    
    def _find_peak_hour(self, time_data: List[Tuple[int, float]]) -> int:
        """Find peak interaction hour"""
        hour_counts = defaultdict(float)
        for hour, weight in time_data:
            hour_counts[hour] += weight
        
        return max(hour_counts.items(), key=lambda x: x[1])[0]
    
    def _record_interaction(
        self,
        user_id: str,
        message: str,
        style_features: Dict[str, Any],
        context: Optional[Dict]
    ):
        """Record interaction for learning"""
        profile = self.user_profiles[user_id]
        profile.interaction_count += 1
        profile.last_interaction = datetime.now()
        
        # Add to interaction times
        profile.interaction_times.append(datetime.now())
        if len(profile.interaction_times) > 100:
            profile.interaction_times.pop(0)
        
        # Update phrase frequencies
        words = message.lower().split()
        for word in words:
            self.phrase_frequencies[user_id][word] += 1
        
        # Add to buffer for batch learning
        self.interaction_buffer.append({
            'user_id': user_id,
            'message': message,
            'style_features': style_features,
            'context': context,
            'timestamp': datetime.now()
        })
        
        # Trigger learning if buffer is full
        if len(self.interaction_buffer) >= 50:
            self._batch_learn_preferences()
    
    def get_personalization_params(
        self,
        user_id: str,
        intent_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get personalization parameters for response generation"""
        profile = self.get_or_create_profile(user_id)
        
        # Base parameters from profile
        params = {
            'style': profile.communication_style,
            'tone': profile.tone_preference,
            'format': profile.response_format,
            'language': profile.language,
            'target_length': profile.avg_response_length,
            'complexity': profile.vocab_complexity,
            'use_emojis': profile.emoji_preference > 0.3,
            'detail_level': profile.detail_preference
        }
        
        # Adjust based on context
        if context:
            # Urgency adjustment
            if context.get('urgency', 0) > 0.7:
                params['style'] = 'concise'
                params['target_length'] *= 0.5
            
            # Time-based adjustments
            if 'timestamp' in context:
                try:
                    ts = datetime.fromisoformat(context['timestamp'])
                    hour = ts.hour
                    
                    # Late night/early morning - be more concise
                    if hour < 6 or hour > 22:
                        params['style'] = 'concise'
                        params['tone'] = 'casual'
                except:
                    pass
        
        # Intent-specific adjustments
        if intent_type:
            if intent_type == 'error':
                params['tone'] = 'empathetic'
                params['detail_level'] = max(0.7, params['detail_level'])
            elif intent_type == 'technical_query':
                params['tone'] = 'technical'
                params['complexity'] = max(0.6, params['complexity'])
        
        # Add preferred phrases
        params['preferred_phrases'] = self._get_preferred_phrases(user_id)
        params['avoided_phrases'] = profile.avoided_phrases
        
        return params
    
    def _get_preferred_phrases(self, user_id: str) -> List[str]:
        """Get user's preferred phrases"""
        if user_id not in self.phrase_frequencies:
            return []
        
        # Get top phrases (excluding common words)
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        
        phrases = []
        for word, count in sorted(
            self.phrase_frequencies[user_id].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if word not in common_words and count > 2:
                phrases.append(word)
            
            if len(phrases) >= 10:
                break
        
        return phrases
    
    def update_satisfaction(
        self,
        user_id: str,
        satisfaction_score: float,
        response_text: Optional[str] = None,
        response_params: Optional[Dict] = None
    ):
        """Update satisfaction metrics"""
        profile = self.get_or_create_profile(user_id)
        profile.satisfaction_scores.append(satisfaction_score)
        
        # Keep last 100 scores
        if len(profile.satisfaction_scores) > 100:
            profile.satisfaction_scores.pop(0)
        
        # Learn from high/low satisfaction
        if response_params:
            key = f"{response_params.get('style', 'unknown')}_{response_params.get('tone', 'unknown')}"
            
            # Update effectiveness scores
            current = self.response_effectiveness[user_id][key]
            self.response_effectiveness[user_id][key] = (
                0.7 * current + 0.3 * satisfaction_score
            )
            
            # Adjust preferences based on satisfaction
            if satisfaction_score > 0.8:
                # This style works well
                if 'style' in response_params:
                    profile.communication_style = response_params['style']
                if 'tone' in response_params:
                    profile.tone_preference = response_params['tone']
            elif satisfaction_score < 0.3:
                # This style doesn't work
                if response_text:
                    # Extract phrases to avoid
                    words = response_text.lower().split()[:10]  # First 10 words
                    profile.avoided_phrases.extend(words)
                    profile.avoided_phrases = list(set(profile.avoided_phrases))[:20]
        
        # Add to feedback buffer
        self.feedback_buffer.append({
            'user_id': user_id,
            'satisfaction': satisfaction_score,
            'response_params': response_params,
            'timestamp': datetime.now()
        })
    
    def _batch_learn_preferences(self):
        """Batch learning from accumulated interactions"""
        if len(self.interaction_buffer) < 20:
            return
        
        logger.info("Running batch preference learning...")
        
        # Group by user
        user_interactions = defaultdict(list)
        for interaction in self.interaction_buffer:
            user_interactions[interaction['user_id']].append(interaction)
        
        # Learn patterns per user
        for user_id, interactions in user_interactions.items():
            # Cluster similar interactions
            if len(interactions) > 5:
                embeddings = [
                    self._create_message_embedding(i['message'], i.get('context'))
                    for i in interactions
                ]
                
                # Simple clustering
                n_clusters = min(3, len(interactions) // 2)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(embeddings)
                
                # Store cluster model
                self.pattern_clusters[user_id] = kmeans
                
                # Analyze each cluster
                for cluster_id in range(n_clusters):
                    cluster_interactions = [
                        interactions[i] for i in range(len(interactions))
                        if clusters[i] == cluster_id
                    ]
                    
                    # Extract common patterns
                    self._extract_cluster_patterns(user_id, cluster_id, cluster_interactions)
        
        # Clear buffer
        self.interaction_buffer.clear()
        
        # Save profiles
        self._save_profiles()
    
    def _extract_cluster_patterns(
        self,
        user_id: str,
        cluster_id: int,
        interactions: List[Dict]
    ):
        """Extract patterns from interaction cluster"""
        # Aggregate style features
        avg_features = defaultdict(float)
        for interaction in interactions:
            for key, value in interaction['style_features'].items():
                if isinstance(value, (int, float)):
                    avg_features[key] += value
        
        # Average
        for key in avg_features:
            avg_features[key] /= len(interactions)
        
        # Create pattern
        pattern = InteractionPattern(
            pattern_type=f'cluster_{cluster_id}',
            pattern_value=dict(avg_features),
            confidence=0.7,
            occurrences=len(interactions)
        )
        
        self.interaction_patterns[user_id].append(pattern)
    
    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive insights about a user"""
        profile = self.get_or_create_profile(user_id)
        
        insights = {
            'profile': {
                'communication_style': profile.communication_style,
                'tone_preference': profile.tone_preference,
                'complexity_level': profile.vocab_complexity,
                'detail_preference': profile.detail_preference,
                'emoji_usage': profile.emoji_preference
            },
            'behavior': {
                'interaction_count': profile.interaction_count,
                'avg_satisfaction': np.mean(profile.satisfaction_scores) if profile.satisfaction_scores else 0.5,
                'last_interaction': profile.last_interaction.isoformat() if profile.last_interaction else None,
                'peak_hours': self._get_peak_hours(user_id),
                'patience_level': profile.patience_level
            },
            'preferences': {
                'preferred_phrases': self._get_preferred_phrases(user_id)[:5],
                'avoided_phrases': profile.avoided_phrases[:5],
                'response_format': profile.response_format,
                'avg_expected_length': profile.avg_response_length
            },
            'patterns': self._summarize_patterns(user_id)
        }
        
        return insights
    
    def _get_peak_hours(self, user_id: str) -> List[int]:
        """Get user's peak interaction hours"""
        if user_id not in self.time_patterns:
            return []
        
        hour_counts = defaultdict(float)
        for hour, weight in self.time_patterns[user_id]:
            hour_counts[hour] += weight
        
        # Get top 3 hours
        top_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        return [hour for hour, _ in top_hours]
    
    def _summarize_patterns(self, user_id: str) -> Dict[str, Any]:
        """Summarize user patterns"""
        if user_id not in self.interaction_patterns:
            return {}
        
        pattern_summary = defaultdict(list)
        for pattern in self.interaction_patterns[user_id]:
            pattern_summary[pattern.pattern_type].append({
                'value': pattern.pattern_value,
                'confidence': pattern.confidence,
                'occurrences': pattern.occurrences
            })
        
        return dict(pattern_summary)
    
    def export_profiles(self) -> Dict[str, Any]:
        """Export all user profiles for analysis"""
        return {
            user_id: {
                'profile': profile.__dict__,
                'insights': self.get_user_insights(user_id)
            }
            for user_id, profile in self.user_profiles.items()
        }
    
    def _save_profiles(self):
        """Save user profiles to disk"""
        save_path = Path("backend/data/user_profiles.json")
        save_path.parent.mkdir(exist_ok=True)
        
        # Prepare data for JSON serialization
        data = {}
        for user_id, profile in self.user_profiles.items():
            profile_data = profile.__dict__.copy()
            
            # Convert datetime objects
            if profile_data['last_interaction']:
                profile_data['last_interaction'] = profile_data['last_interaction'].isoformat()
            
            profile_data['interaction_times'] = [
                t.isoformat() for t in profile_data['interaction_times']
            ]
            
            data[user_id] = profile_data
        
        # Save additional data
        save_data = {
            'profiles': data,
            'phrase_frequencies': dict(self.phrase_frequencies),
            'response_effectiveness': dict(self.response_effectiveness),
            'time_patterns': dict(self.time_patterns)
        }
        
        try:
            with open(save_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            logger.debug("Saved user profiles")
        except Exception as e:
            logger.error(f"Error saving profiles: {e}")
    
    def _load_profiles(self):
        """Load user profiles from disk"""
        save_path = Path("backend/data/user_profiles.json")
        
        if not save_path.exists():
            logger.info("No saved profiles found")
            return
        
        try:
            with open(save_path, 'r') as f:
                data = json.load(f)
            
            # Load profiles
            for user_id, profile_data in data.get('profiles', {}).items():
                # Convert datetime strings back
                if profile_data.get('last_interaction'):
                    profile_data['last_interaction'] = datetime.fromisoformat(
                        profile_data['last_interaction']
                    )
                
                profile_data['interaction_times'] = [
                    datetime.fromisoformat(t) for t in profile_data.get('interaction_times', [])
                ]
                
                # Create profile
                profile = UserProfile(**profile_data)
                self.user_profiles[user_id] = profile
            
            # Load additional data
            self.phrase_frequencies = defaultdict(
                lambda: defaultdict(int),
                data.get('phrase_frequencies', {})
            )
            self.response_effectiveness = defaultdict(
                lambda: defaultdict(float),
                data.get('response_effectiveness', {})
            )
            self.time_patterns = defaultdict(list, data.get('time_patterns', {}))
            
            logger.info(f"Loaded {len(self.user_profiles)} user profiles")
            
        except Exception as e:
            logger.error(f"Error loading profiles: {e}")


# Singleton instance
_personalization_engine: Optional[PersonalizationEngine] = None


def get_personalization_engine() -> PersonalizationEngine:
    """Get singleton instance of personalization engine"""
    global _personalization_engine
    if _personalization_engine is None:
        _personalization_engine = PersonalizationEngine()
    return _personalization_engine