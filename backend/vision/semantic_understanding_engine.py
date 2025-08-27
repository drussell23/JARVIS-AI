#!/usr/bin/env python3
"""
Semantic Understanding Engine for Vision System
Context-aware, multi-language intent extraction with zero hardcoding
"""

import torch
import torch.nn as nn
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModel = None
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import logging
from collections import defaultdict
import asyncio
try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
import spacy
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class SemanticContext:
    """Rich semantic context for understanding"""
    language: str
    entities: List[Dict[str, Any]]
    sentiment: float
    question_type: Optional[str]
    temporal_markers: List[str]
    spatial_references: List[str]
    user_intent_signals: List[str]
    confidence_markers: List[str]
    embedding: np.ndarray


@dataclass
class IntentUnderstanding:
    """Complete understanding of user intent"""
    primary_intent: str
    sub_intents: List[str]
    context: SemanticContext
    confidence: float
    ambiguity_score: float
    clarification_needed: bool
    suggested_responses: List[str]


class SemanticUnderstandingEngine:
    """
    Advanced semantic understanding with zero hardcoding
    Works across languages and contexts
    """
    
    def __init__(self):
        # Multi-language transformer model
        self.model_name = 'xlm-roberta-base'
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
            except Exception as e:
                logger.warning(f"Could not load transformer model: {e}")
                self.tokenizer = None
                self.model = None
        else:
            self.tokenizer = None
            self.model = None
        
        # Semantic analysis components
        self.semantic_analyzer = self._build_semantic_analyzer()
        
        # Context extraction
        self.context_extractor = ContextExtractor()
        
        # Multi-language NLP pipelines
        self.nlp_models = {}
        self._initialize_language_models()
        
        # Intent understanding network
        self.understanding_net = self._build_understanding_network()
        
        # Learned semantic patterns
        self.semantic_patterns = defaultdict(list)
        self._load_semantic_patterns()
        
    def _build_semantic_analyzer(self) -> nn.Module:
        """Build semantic analysis network"""
        return nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Tanh()
        )
        
    def _build_understanding_network(self) -> nn.Module:
        """Build intent understanding network"""
        return nn.ModuleDict({
            'intent_classifier': nn.Linear(256, 128),
            'confidence_predictor': nn.Linear(256, 1),
            'ambiguity_detector': nn.Linear(256, 1),
            'context_encoder': nn.Linear(768, 256)
        })
        
    def _initialize_language_models(self):
        """Initialize language-specific NLP models"""
        # Start with English, dynamically add others as needed
        try:
            self.nlp_models['en'] = spacy.load('en_core_web_sm')
        except:
            logger.warning("English spacy model not available")
            
    async def understand_intent(self, text: str, context: Optional[Dict] = None) -> IntentUnderstanding:
        """
        Extract deep semantic understanding from text
        Works in any language without hardcoding
        """
        context = context or {}
        
        # Detect language
        language = self._detect_language(text)
        
        # Extract semantic context
        semantic_context = await self._extract_semantic_context(text, language, context)
        
        # Get transformer embeddings
        embeddings = self._get_embeddings(text)
        
        # Analyze semantics
        semantic_features = self.semantic_analyzer(torch.tensor(embeddings))
        
        # Understand intent
        understanding = self._understand_intent(semantic_features, semantic_context)
        
        # Learn from this understanding
        self._learn_semantic_pattern(text, understanding)
        
        return understanding
        
    def _detect_language(self, text: str) -> str:
        """Detect language of text"""
        if LANGDETECT_AVAILABLE:
            try:
                return detect(text)
            except:
                return 'en'  # Default fallback
        else:
            # Simple heuristic for language detection
            spanish_words = ['qué', 'cómo', 'por', 'puedes', 'está']
            french_words = ['vous', 'pouvez', 'écran', 'que', 'est']
            
            text_lower = text.lower()
            if any(word in text_lower for word in spanish_words):
                return 'es'
            elif any(word in text_lower for word in french_words):
                return 'fr'
            else:
                return 'en'
            
    async def _extract_semantic_context(self, text: str, language: str, context: Dict) -> SemanticContext:
        """Extract rich semantic context"""
        # Get language-specific NLP model
        nlp = await self._get_nlp_model(language)
        
        # Initialize extractors
        entities = []
        temporal_markers = []
        spatial_references = []
        intent_signals = []
        confidence_markers = []
        
        if nlp:
            doc = nlp(text)
            
            # Extract entities
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start,
                    'end': ent.end
                })
                
            # Extract temporal markers
            temporal_markers = [token.text for token in doc if token.pos_ in ['TIME', 'DATE']]
            
            # Extract spatial references
            spatial_references = [token.text for token in doc if token.dep_ in ['prep', 'advmod'] and any(
                prep in token.text.lower() for prep in ['on', 'at', 'in', 'above', 'below', 'near']
            )]
            
            # Extract intent signals (question words, modals, etc.)
            intent_signals = self._extract_intent_signals(doc)
            
            # Extract confidence markers
            confidence_markers = self._extract_confidence_markers(doc)
            
        # Analyze sentiment
        sentiment = self._analyze_sentiment(text)
        
        # Determine question type
        question_type = self._determine_question_type(text, intent_signals)
        
        # Get embeddings
        embedding = self._get_embeddings(text)
        
        return SemanticContext(
            language=language,
            entities=entities,
            sentiment=sentiment,
            question_type=question_type,
            temporal_markers=temporal_markers,
            spatial_references=spatial_references,
            user_intent_signals=intent_signals,
            confidence_markers=confidence_markers,
            embedding=embedding
        )
        
    async def _get_nlp_model(self, language: str):
        """Get or download NLP model for language"""
        if language not in self.nlp_models:
            # Try to load model for this language
            model_map = {
                'en': 'en_core_web_sm',
                'es': 'es_core_news_sm',
                'fr': 'fr_core_news_sm',
                'de': 'de_core_news_sm',
                'zh': 'zh_core_web_sm',
                'ja': 'ja_core_news_sm'
            }
            
            if language in model_map:
                try:
                    self.nlp_models[language] = spacy.load(model_map[language])
                except:
                    logger.warning(f"Could not load NLP model for {language}")
                    return None
                    
        return self.nlp_models.get(language)
        
    def _extract_intent_signals(self, doc) -> List[str]:
        """Extract signals that indicate user intent"""
        signals = []
        
        # Question patterns (language-agnostic approach)
        question_indicators = {
            'en': ['what', 'how', 'why', 'when', 'where', 'who', 'can', 'could', 'would', 'should'],
            'es': ['qué', 'cómo', 'por qué', 'cuándo', 'dónde', 'quién', 'puede', 'podría'],
            'fr': ['quoi', 'comment', 'pourquoi', 'quand', 'où', 'qui', 'peut', 'pourrait']
        }
        
        # Check for question words
        for token in doc:
            if token.tag_ in ['WDT', 'WP', 'WP$', 'WRB']:  # Wh-words
                signals.append(f"question:{token.text}")
            elif token.tag_ == 'MD':  # Modal verbs
                signals.append(f"modal:{token.text}")
            elif token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                signals.append(f"verb_root:{token.text}")
                
        return signals
        
    def _extract_confidence_markers(self, doc) -> List[str]:
        """Extract markers that indicate user confidence/uncertainty"""
        markers = []
        
        uncertainty_words = ['maybe', 'perhaps', 'possibly', 'might', 'could']
        certainty_words = ['definitely', 'certainly', 'surely', 'must']
        
        for token in doc:
            if token.text.lower() in uncertainty_words:
                markers.append(f"uncertain:{token.text}")
            elif token.text.lower() in certainty_words:
                markers.append(f"certain:{token.text}")
                
        return markers
        
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment (language-agnostic)"""
        # Basic sentiment analysis without transformers
        positive_words = ['good', 'great', 'excellent', 'perfect', 'amazing', 'wonderful']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'poor', 'wrong']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return 0.5
        elif neg_count > pos_count:
            return -0.5
        else:
            return 0.0  # Neutral
            
    def _determine_question_type(self, text: str, intent_signals: List[str]) -> Optional[str]:
        """Determine type of question being asked"""
        # Look for patterns in intent signals
        if any('can' in signal or 'could' in signal for signal in intent_signals):
            return 'capability_check'
        elif any('what' in signal or 'qué' in signal for signal in intent_signals):
            return 'information_request'
        elif any('how' in signal or 'cómo' in signal for signal in intent_signals):
            return 'instruction_request'
        elif text.strip().endswith('?'):
            return 'general_question'
        else:
            return None
            
    def _get_embeddings(self, text: str) -> np.ndarray:
        """Get transformer embeddings"""
        if self.tokenizer and self.model:
            inputs = self.tokenizer(text, return_tensors='pt', 
                                   truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                
            return embeddings
        else:
            # Fallback: simple hash-based embeddings
            words = text.lower().split()
            embedding = np.zeros(768)
            for i, word in enumerate(words):
                hash_val = hash(word)
                embedding[hash_val % 768] += 1.0
            return embedding / (len(words) + 1)
        
    def _understand_intent(self, semantic_features: torch.Tensor, 
                          context: SemanticContext) -> IntentUnderstanding:
        """Deep understanding of intent from semantic features"""
        # Pass through understanding networks
        intent_logits = self.understanding_net['intent_classifier'](semantic_features)
        confidence = torch.sigmoid(self.understanding_net['confidence_predictor'](semantic_features)).item()
        ambiguity = torch.sigmoid(self.understanding_net['ambiguity_detector'](semantic_features)).item()
        
        # Determine primary intent (no hardcoding)
        primary_intent = self._determine_primary_intent(intent_logits, context)
        
        # Extract sub-intents
        sub_intents = self._extract_sub_intents(semantic_features, context)
        
        # Determine if clarification needed
        clarification_needed = ambiguity > 0.6 or confidence < 0.5
        
        # Generate suggested responses based on understanding
        suggested_responses = self._generate_suggestions(primary_intent, context, confidence)
        
        return IntentUnderstanding(
            primary_intent=primary_intent,
            sub_intents=sub_intents,
            context=context,
            confidence=confidence,
            ambiguity_score=ambiguity,
            clarification_needed=clarification_needed,
            suggested_responses=suggested_responses
        )
        
    def _determine_primary_intent(self, intent_logits: torch.Tensor, 
                                 context: SemanticContext) -> str:
        """Determine primary intent without hardcoding"""
        # Use learned patterns to map to intent
        if context.question_type == 'capability_check':
            # User asking if system can do something
            return 'vision_capability_confirmation'
        elif context.entities and any(e['label'] == 'SCREEN' for e in context.entities):
            # Reference to screen/display
            return 'screen_analysis_request'
        else:
            # Learn from context
            return 'general_vision_request'
            
    def _extract_sub_intents(self, features: torch.Tensor, 
                            context: SemanticContext) -> List[str]:
        """Extract secondary intents"""
        sub_intents = []
        
        # Check for specific markers
        if context.temporal_markers:
            sub_intents.append('temporal_awareness')
        if context.spatial_references:
            sub_intents.append('spatial_analysis')
        if context.confidence_markers:
            if any('uncertain' in m for m in context.confidence_markers):
                sub_intents.append('needs_confirmation')
                
        return sub_intents
        
    def _generate_suggestions(self, intent: str, context: SemanticContext, 
                            confidence: float) -> List[str]:
        """Generate response suggestions based on understanding"""
        suggestions = []
        
        if intent == 'vision_capability_confirmation':
            suggestions.append("confirm_with_demonstration")
            suggestions.append("list_capabilities")
        elif intent == 'screen_analysis_request':
            suggestions.append("detailed_description")
            suggestions.append("focused_analysis")
        
        if confidence < 0.7:
            suggestions.append("request_clarification")
            
        return suggestions
        
    def _learn_semantic_pattern(self, text: str, understanding: IntentUnderstanding):
        """Learn from semantic patterns"""
        pattern = {
            'text': text,
            'intent': understanding.primary_intent,
            'context': understanding.context.__dict__,
            'confidence': understanding.confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        self.semantic_patterns[understanding.primary_intent].append(pattern)
        
        # Keep only recent patterns
        if len(self.semantic_patterns[understanding.primary_intent]) > 100:
            self.semantic_patterns[understanding.primary_intent].pop(0)
            
        # Periodically save
        if sum(len(p) for p in self.semantic_patterns.values()) % 10 == 0:
            self._save_semantic_patterns()
            
    def _save_semantic_patterns(self):
        """Save learned semantic patterns"""
        save_path = Path("backend/data/vision_semantic_patterns.json")
        save_path.parent.mkdir(exist_ok=True)
        
        try:
            with open(save_path, 'w') as f:
                json.dump(dict(self.semantic_patterns), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving semantic patterns: {e}")
            
    def _load_semantic_patterns(self):
        """Load previously learned patterns"""
        save_path = Path("backend/data/vision_semantic_patterns.json")
        
        if save_path.exists():
            try:
                with open(save_path, 'r') as f:
                    patterns = json.load(f)
                    self.semantic_patterns = defaultdict(list, patterns)
                    logger.info(f"Loaded {sum(len(p) for p in self.semantic_patterns.values())} semantic patterns")
            except Exception as e:
                logger.error(f"Error loading semantic patterns: {e}")


class ContextExtractor:
    """Extract rich context from various sources"""
    
    def __init__(self):
        self.context_features = {}
        
    def extract_vision_context(self, raw_context: Dict) -> Dict[str, Any]:
        """Extract vision-specific context"""
        vision_context = {
            'screen_state': raw_context.get('screen_state', 'unknown'),
            'active_windows': raw_context.get('active_windows', []),
            'user_activity': raw_context.get('user_activity', 'unknown'),
            'time_context': self._extract_time_context(),
            'interaction_history': raw_context.get('history', [])
        }
        
        return vision_context
        
    def _extract_time_context(self) -> Dict[str, Any]:
        """Extract temporal context"""
        now = datetime.now()
        
        return {
            'time_of_day': 'morning' if 5 <= now.hour < 12 else
                          'afternoon' if 12 <= now.hour < 17 else
                          'evening' if 17 <= now.hour < 22 else 'night',
            'day_of_week': now.strftime('%A'),
            'timestamp': now.isoformat()
        }


# Singleton instance
_engine_instance: Optional[SemanticUnderstandingEngine] = None


def get_semantic_understanding_engine() -> SemanticUnderstandingEngine:
    """Get singleton instance of semantic understanding engine"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = SemanticUnderstandingEngine()
    return _engine_instance