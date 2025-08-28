#!/usr/bin/env python3
"""
Dynamic Response Composer for Vision System
Template-free, neural-based response generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import numpy as np
from pathlib import Path
import logging
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

@dataclass
class ResponseContext:
    """Context for response generation"""
    intent_type: str
    confidence: float
    user_name: Optional[str] = None
    conversation_history: List[Dict] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    emotion_state: Optional[str] = None
    time_of_day: Optional[str] = None
    previous_responses: List[str] = field(default_factory=list)
    success_rate: float = 0.0

@dataclass
class GeneratedResponse:
    """Generated response with metadata"""
    text: str
    format: str = "text"  # text, json, markdown
    confidence: float = 1.0
    style: str = "neutral"  # neutral, formal, casual, technical
    emotion_tone: Optional[str] = None
    alternatives: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ResponseStyleEncoder(nn.Module):
    """Neural network for encoding response style"""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, style_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, style_dim),
            nn.Tanh()
        )
        
        # Style-specific heads
        self.tone_head = nn.Linear(style_dim, 32)
        self.formality_head = nn.Linear(style_dim, 16)
        self.length_head = nn.Linear(style_dim, 8)
        
    def forward(self, x):
        style_features = self.encoder(x)
        tone = self.tone_head(style_features)
        formality = self.formality_head(style_features)
        length = self.length_head(style_features)
        return style_features, tone, formality, length

class NeuralResponseGenerator(nn.Module):
    """Neural network for generating responses"""
    
    def __init__(self, context_dim: int = 768, style_dim: int = 128, vocab_size: int = 10000):
        super().__init__()
        
        # Context understanding
        self.context_encoder = nn.LSTM(
            input_size=context_dim,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Style fusion
        self.style_fusion = nn.Sequential(
            nn.Linear(512 + style_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Response generation
        self.response_decoder = nn.LSTM(
            input_size=256,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Output projection
        self.output_projection = nn.Linear(512, vocab_size)
        
    def forward(self, context_embeddings, style_features):
        # Encode context
        context_out, (h, c) = self.context_encoder(context_embeddings)
        
        # Combine with style
        combined = torch.cat([context_out[:, -1, :], style_features], dim=-1)
        fused = self.style_fusion(combined)
        
        # Generate response features
        response_features, _ = self.response_decoder(fused.unsqueeze(1))
        
        # Project to vocabulary
        output = self.output_projection(response_features)
        
        return output, response_features

class DynamicResponseComposer:
    """
    Template-free response generation using neural networks
    Learns from user preferences and adapts style dynamically
    """
    
    def __init__(self):
        # Neural components
        self.style_encoder = ResponseStyleEncoder()
        self.response_generator = NeuralResponseGenerator()
        
        # Response templates (will be phased out as neural generation improves)
        self.response_patterns = self._initialize_response_patterns()
        
        # User preference tracking
        self.user_preferences = defaultdict(lambda: {
            'style': 'neutral',
            'verbosity': 'balanced',
            'technical_level': 'medium',
            'emotion_preference': 'professional'
        })
        
        # Response effectiveness tracking
        self.response_metrics = defaultdict(lambda: {
            'generated': 0,
            'successful': 0,
            'user_feedback': []
        })
        
        # Learning data
        self.training_examples = []
        self.style_examples = defaultdict(list)
        
        # Load any saved models
        self._load_models()
        
        logger.info("Dynamic Response Composer initialized")
    
    def _initialize_response_patterns(self) -> Dict[str, List[str]]:
        """Initialize base response patterns (to be learned and replaced)"""
        return {
            'vision_capability_confirmation': [
                "Yes {user_name}, I can see your screen clearly.",
                "I have visual on your display, {user_name}.",
                "Visual feed confirmed and active.",
                "I'm looking at your screen now, {user_name}.",
                "Screen access confirmed, {user_name}."
            ],
            'screen_analysis': [
                "{description}",
                "Looking at your screen, {description}",
                "I can see {description}"
            ],
            'error': [
                "I encountered an issue: {error}",
                "There was a problem: {error}",
                "Unable to complete the request: {error}"
            ]
        }
    
    async def compose_response(
        self, 
        content: str,
        context: ResponseContext,
        force_style: Optional[str] = None
    ) -> GeneratedResponse:
        """
        Compose a response using neural generation and learned patterns
        """
        # Get user preferences
        user_prefs = self._get_user_preferences(context.user_name)
        
        # Determine response style
        style = force_style or self._determine_response_style(context, user_prefs)
        
        # Generate response using neural network
        if self._should_use_neural_generation(context):
            response = await self._neural_generate_response(content, context, style)
        else:
            # Fallback to pattern-based generation
            response = self._pattern_generate_response(content, context, style)
        
        # Apply post-processing
        response = self._post_process_response(response, context, user_prefs)
        
        # Track metrics
        self._track_response_metrics(response, context)
        
        return response
    
    def _should_use_neural_generation(self, context: ResponseContext) -> bool:
        """Determine if neural generation should be used"""
        # Use neural generation if we have enough training data
        if len(self.training_examples) > 100:
            return True
        
        # Use neural for high-confidence intents
        if context.confidence > 0.8:
            return True
        
        # Fallback to patterns for now
        return False
    
    async def _neural_generate_response(
        self, 
        content: str,
        context: ResponseContext,
        style: str
    ) -> GeneratedResponse:
        """Generate response using neural network"""
        try:
            # Create embeddings from content and context
            content_embedding = self._create_content_embedding(content)
            context_embedding = self._create_context_embedding(context)
            
            # Encode style preferences
            style_features, tone, formality, length = self.style_encoder(
                torch.tensor(context_embedding, dtype=torch.float32).unsqueeze(0)
            )
            
            # Generate response
            with torch.no_grad():
                output, features = self.response_generator(
                    torch.tensor(content_embedding, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                    style_features.squeeze(0)
                )
            
            # Decode to text (simplified - in production would use proper decoding)
            response_text = self._decode_neural_output(output, content, context)
            
            # Determine format
            response_format = self._determine_format(context, style)
            
            return GeneratedResponse(
                text=response_text,
                format=response_format,
                confidence=float(torch.sigmoid(output.max()).item()),
                style=style,
                emotion_tone=self._detect_emotion_tone(features),
                alternatives=self._generate_alternatives(features, content, context)
            )
            
        except Exception as e:
            logger.error(f"Neural generation failed: {e}")
            # Fallback to pattern generation
            return self._pattern_generate_response(content, context, style)
    
    def _pattern_generate_response(
        self,
        content: str,
        context: ResponseContext,
        style: str
    ) -> GeneratedResponse:
        """Generate response using learned patterns"""
        intent = context.intent_type
        
        # Get patterns for this intent
        if intent in self.response_patterns:
            patterns = self.response_patterns[intent]
        else:
            patterns = ["{content}"]  # Default pattern
        
        # Select pattern based on style
        pattern = self._select_pattern_by_style(patterns, style)
        
        # Fill in the pattern
        response_text = pattern.format(
            content=content,
            user_name=context.user_name or "there",
            description=content,
            error=content
        )
        
        # Adjust based on user preferences
        response_text = self._apply_style_adjustments(response_text, style, context)
        
        return GeneratedResponse(
            text=response_text,
            format="text",
            confidence=context.confidence,
            style=style,
            alternatives=self._generate_pattern_alternatives(patterns, content, context)
        )
    
    def _determine_response_style(
        self, 
        context: ResponseContext,
        user_prefs: Dict[str, Any]
    ) -> str:
        """Determine appropriate response style using neural routing"""
        # Factors to consider
        factors = {
            'user_preference': user_prefs.get('style', 'neutral'),
            'time_of_day': context.time_of_day,
            'conversation_length': len(context.conversation_history),
            'confidence': context.confidence,
            'success_rate': context.success_rate,
            'emotion_state': context.emotion_state
        }
        
        # Neural style selection (simplified)
        if context.confidence < 0.5:
            return 'cautious'
        elif context.emotion_state == 'frustrated':
            return 'empathetic'
        elif factors['time_of_day'] in ['morning', 'evening']:
            return 'casual'
        elif factors['conversation_length'] > 10:
            return 'concise'
        else:
            return user_prefs.get('style', 'neutral')
    
    def _apply_style_adjustments(
        self,
        text: str,
        style: str,
        context: ResponseContext
    ) -> str:
        """Apply style-specific adjustments to text"""
        if style == 'formal':
            # Remove contractions
            text = text.replace("I'm", "I am").replace("can't", "cannot")
            # Add formal greeting if needed
            if context.user_name and not any(context.user_name in t for t in [text]):
                text = f"Sir {context.user_name}, {text}"
                
        elif style == 'casual':
            # Add casual elements
            text = text.replace("I am", "I'm").replace("cannot", "can't")
            
        elif style == 'concise':
            # Remove unnecessary words
            text = re.sub(r'\b(just|really|very|quite)\b', '', text)
            text = ' '.join(text.split())  # Clean up extra spaces
            
        elif style == 'technical':
            # Add technical details if available
            if 'confidence' in context.__dict__:
                text += f" (Confidence: {context.confidence:.1%})"
                
        elif style == 'empathetic':
            # Add understanding phrases
            if "error" in text.lower() or "problem" in text.lower():
                text = f"I understand this might be frustrating. {text}"
        
        return text.strip()
    
    def _create_content_embedding(self, content: str) -> np.ndarray:
        """Create embedding from content"""
        # Simplified embedding creation
        # In production, would use proper sentence transformers
        words = content.lower().split()
        embedding = np.zeros(768)
        
        for i, word in enumerate(words):
            # Simple hash-based embedding
            hash_val = hash(word)
            embedding[hash_val % 768] += 1.0 / (i + 1)
            
        return embedding / (len(words) + 1)
    
    def _create_context_embedding(self, context: ResponseContext) -> np.ndarray:
        """Create embedding from context"""
        # Combine various context features
        features = []
        
        # Confidence feature
        features.append(context.confidence)
        
        # Success rate feature
        features.append(context.success_rate)
        
        # Conversation length feature
        features.append(min(len(context.conversation_history) / 10, 1.0))
        
        # Time features
        time_features = {
            'morning': [1, 0, 0, 0],
            'afternoon': [0, 1, 0, 0],
            'evening': [0, 0, 1, 0],
            'night': [0, 0, 0, 1]
        }
        features.extend(time_features.get(context.time_of_day, [0, 0, 0, 0]))
        
        # Pad to 512 dimensions
        embedding = np.zeros(512)
        embedding[:len(features)] = features
        
        return embedding
    
    def _decode_neural_output(
        self,
        output: torch.Tensor,
        content: str,
        context: ResponseContext
    ) -> str:
        """Decode neural network output to text"""
        # Simplified decoding - in production would use beam search
        # For now, use content as base and modify based on neural output
        
        # Get top predictions
        probs = F.softmax(output.squeeze(), dim=-1)
        top_probs, top_indices = torch.topk(probs, 5)
        
        # Modify content based on neural suggestions
        response = content
        
        # Add context-appropriate prefix/suffix
        if context.confidence > 0.8:
            response = f"I can clearly see that {response}"
        elif context.confidence < 0.5:
            response = f"From what I can determine, {response}"
        
        return response
    
    def _detect_emotion_tone(self, features: torch.Tensor) -> Optional[str]:
        """Detect emotional tone from response features"""
        # Simplified emotion detection
        feature_mean = features.mean().item()
        
        if feature_mean > 0.5:
            return "positive"
        elif feature_mean < -0.5:
            return "concerned"
        else:
            return "neutral"
    
    def _determine_format(self, context: ResponseContext, style: str) -> str:
        """Determine response format based on context"""
        # Check user preferences
        user_format = context.user_preferences.get('format', 'text')
        
        # Override for technical style
        if style == 'technical' and context.intent_type == 'screen_analysis':
            return 'markdown'
        
        # JSON for high-detail requests
        if 'json' in str(context.conversation_history).lower():
            return 'json'
        
        return user_format
    
    def _generate_alternatives(
        self,
        features: torch.Tensor,
        content: str,
        context: ResponseContext
    ) -> List[str]:
        """Generate alternative responses"""
        alternatives = []
        
        # Generate variations based on different styles
        for style in ['formal', 'casual', 'concise']:
            if style != context.user_preferences.get('style'):
                alt = self._apply_style_adjustments(content, style, context)
                if alt != content:
                    alternatives.append(alt)
        
        return alternatives[:3]  # Limit to 3 alternatives
    
    def _generate_pattern_alternatives(
        self,
        patterns: List[str],
        content: str,
        context: ResponseContext
    ) -> List[str]:
        """Generate alternatives from patterns"""
        alternatives = []
        
        for pattern in patterns[:3]:
            if pattern != patterns[0]:  # Skip the selected pattern
                alt = pattern.format(
                    content=content,
                    user_name=context.user_name or "there",
                    description=content,
                    error=content
                )
                alternatives.append(alt)
        
        return alternatives
    
    def _select_pattern_by_style(self, patterns: List[str], style: str) -> str:
        """Select pattern based on style preference"""
        if not patterns:
            return "{content}"
        
        # Simple selection based on style
        if style == 'concise':
            # Choose shortest pattern
            return min(patterns, key=len)
        elif style == 'formal':
            # Choose pattern with user_name if available
            for pattern in patterns:
                if '{user_name}' in pattern:
                    return pattern
        
        # Default to first pattern
        return patterns[0]
    
    def _post_process_response(
        self,
        response: GeneratedResponse,
        context: ResponseContext,
        user_prefs: Dict[str, Any]
    ) -> GeneratedResponse:
        """Apply final post-processing to response"""
        # Apply format conversion if needed
        if response.format == 'json':
            response.text = self._convert_to_json(response.text, context)
        elif response.format == 'markdown':
            response.text = self._convert_to_markdown(response.text, context)
        
        # Apply length constraints
        max_length = {
            'concise': 50,
            'balanced': 150,
            'verbose': 300
        }.get(user_prefs.get('verbosity', 'balanced'), 150)
        
        if len(response.text.split()) > max_length:
            response.text = self._truncate_intelligently(response.text, max_length)
        
        # Add metadata
        response.metadata.update({
            'generated_at': datetime.now().isoformat(),
            'intent': context.intent_type,
            'style_applied': response.style,
            'user_prefs': user_prefs
        })
        
        return response
    
    def _convert_to_json(self, text: str, context: ResponseContext) -> str:
        """Convert response to JSON format"""
        data = {
            'response': text,
            'intent': context.intent_type,
            'confidence': context.confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        return json.dumps(data, indent=2)
    
    def _convert_to_markdown(self, text: str, context: ResponseContext) -> str:
        """Convert response to Markdown format"""
        # Add appropriate formatting
        if context.intent_type == 'screen_analysis':
            # Format as detailed analysis
            lines = text.split('. ')
            markdown = "## Screen Analysis\n\n"
            for line in lines:
                if line.strip():
                    markdown += f"- {line.strip()}\n"
            return markdown
        else:
            return f"**Response**: {text}"
    
    def _truncate_intelligently(self, text: str, max_words: int) -> str:
        """Truncate text intelligently at sentence boundaries"""
        words = text.split()
        if len(words) <= max_words:
            return text
        
        # Find last sentence boundary within limit
        truncated = ' '.join(words[:max_words])
        last_period = truncated.rfind('.')
        if last_period > max_words * 0.7:  # If we have at least 70% of target
            return truncated[:last_period + 1]
        else:
            return truncated + "..."
    
    def _get_user_preferences(self, user_name: Optional[str]) -> Dict[str, Any]:
        """Get user preferences"""
        if user_name:
            return self.user_preferences[user_name]
        else:
            return self.user_preferences['default']
    
    def _track_response_metrics(self, response: GeneratedResponse, context: ResponseContext):
        """Track response effectiveness metrics"""
        metric_key = f"{context.intent_type}:{response.style}"
        self.response_metrics[metric_key]['generated'] += 1
        
        # Store for potential feedback
        self.response_metrics[metric_key]['recent_response'] = {
            'text': response.text,
            'context': context.__dict__,
            'timestamp': datetime.now().isoformat()
        }
    
    def learn_from_feedback(
        self,
        response_text: str,
        was_effective: bool,
        user_feedback: Optional[str] = None,
        user_name: Optional[str] = None
    ):
        """Learn from user feedback on responses"""
        # Find the response in metrics
        for metric_key, metrics in self.response_metrics.items():
            recent = metrics.get('recent_response', {})
            if recent.get('text') == response_text:
                if was_effective:
                    metrics['successful'] += 1
                
                if user_feedback:
                    metrics['user_feedback'].append({
                        'feedback': user_feedback,
                        'effective': was_effective,
                        'timestamp': datetime.now().isoformat()
                    })
                
                # Update user preferences based on feedback
                if user_name and was_effective:
                    intent, style = metric_key.split(':')
                    self.user_preferences[user_name]['style'] = style
                
                break
        
        # Add to training examples
        self.training_examples.append({
            'response': response_text,
            'effective': was_effective,
            'feedback': user_feedback,
            'timestamp': datetime.now().isoformat()
        })
        
        # Retrain if enough examples
        if len(self.training_examples) % 50 == 0:
            self._retrain_models()
    
    def update_user_preference(
        self,
        user_name: str,
        preference_type: str,
        value: Any
    ):
        """Update user preference"""
        self.user_preferences[user_name][preference_type] = value
        logger.info(f"Updated {user_name}'s {preference_type} preference to {value}")
    
    def _retrain_models(self):
        """Retrain neural models with accumulated examples"""
        # This would implement actual training logic
        # For now, just log
        logger.info(f"Retraining models with {len(self.training_examples)} examples")
        self._save_models()
    
    def _save_models(self):
        """Save model state"""
        save_path = Path("backend/data/response_composer_models.pt")
        save_path.parent.mkdir(exist_ok=True)
        
        torch.save({
            'style_encoder': self.style_encoder.state_dict(),
            'response_generator': self.response_generator.state_dict(),
            'user_preferences': dict(self.user_preferences),
            'response_metrics': dict(self.response_metrics),
            'training_examples': self.training_examples[-1000:]  # Keep last 1000
        }, save_path)
    
    def _load_models(self):
        """Load saved model state"""
        save_path = Path("backend/data/response_composer_models.pt")
        
        if save_path.exists():
            try:
                data = torch.load(save_path)
                self.style_encoder.load_state_dict(data['style_encoder'])
                self.response_generator.load_state_dict(data['response_generator'])
                self.user_preferences.update(data['user_preferences'])
                self.response_metrics.update(data['response_metrics'])
                self.training_examples = data['training_examples']
                logger.info("Loaded saved response composer models")
            except Exception as e:
                logger.error(f"Error loading models: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get response generation metrics"""
        total_generated = sum(m['generated'] for m in self.response_metrics.values())
        total_successful = sum(m['successful'] for m in self.response_metrics.values())
        
        return {
            'total_responses_generated': total_generated,
            'success_rate': total_successful / max(1, total_generated),
            'styles_used': list(set(k.split(':')[1] for k in self.response_metrics.keys())),
            'user_count': len(self.user_preferences),
            'training_examples': len(self.training_examples)
        }

# Singleton instance
_composer_instance: Optional[DynamicResponseComposer] = None

def get_response_composer() -> DynamicResponseComposer:
    """Get singleton instance of response composer"""
    global _composer_instance
    if _composer_instance is None:
        _composer_instance = DynamicResponseComposer()
    return _composer_instance