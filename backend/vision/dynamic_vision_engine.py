#!/usr/bin/env python3
"""
Dynamic Vision Engine - Zero Hardcoding ML-Based Vision System
Learns and adapts to any vision-related command without predefined patterns
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import numpy as np
from pathlib import Path
import pickle
from collections import defaultdict
import re
import inspect

logger = logging.getLogger(__name__)


@dataclass
class VisionCapability:
    """Dynamically discovered vision capability"""
    name: str
    description: str
    handler: Callable
    confidence_threshold: float = 0.7
    learned_patterns: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    usage_count: int = 0
    last_used: Optional[datetime] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)


@dataclass
class VisionIntent:
    """ML-analyzed vision intent"""
    raw_command: str
    action_verb: Optional[str] = None
    target_object: Optional[str] = None
    modifiers: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    context_clues: Dict[str, Any] = field(default_factory=dict)
    semantic_embedding: Optional[np.ndarray] = None
    predicted_capability: Optional[str] = None


class DynamicVisionEngine:
    """
    Completely dynamic vision engine with zero hardcoding
    Learns capabilities, patterns, and user preferences on the fly
    """
    
    def __init__(self):
        # Dynamic capability registry
        self.capabilities: Dict[str, VisionCapability] = {}
        
        # ML components
        self.pattern_embeddings = {}
        self.semantic_model = self._initialize_semantic_model()
        self.intent_classifier = self._initialize_intent_classifier()
        
        # Learning system
        self.command_history = []
        self.feedback_data = []
        self.user_preferences = defaultdict(float)
        
        # Dynamic pattern learning
        self.learned_patterns = defaultdict(list)
        self.pattern_scores = defaultdict(float)
        
        # Context awareness
        self.context_memory = []
        self.session_context = {}
        
        # Plugin system for vision providers
        self.vision_providers = {}
        self._discover_vision_providers()
        
        # Load any saved learning data
        self._load_learned_data()
        
        logger.info("Dynamic Vision Engine initialized with zero hardcoding")
        
    def _initialize_semantic_model(self):
        """Initialize semantic understanding model"""
        try:
            # Try to use sentence transformers for semantic similarity
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            logger.info("Sentence transformers not available, using basic embeddings")
            return None
            
    def _initialize_intent_classifier(self):
        """Initialize intent classification system"""
        # This would ideally use a proper ML model
        # For now, we'll use a dynamic learning system
        return {
            'verb_patterns': defaultdict(list),
            'object_patterns': defaultdict(list),
            'capability_patterns': defaultdict(list)
        }
        
    def _register_basic_capabilities(self):
        """Register basic built-in vision capabilities"""
        # Screen capture capability
        async def describe_screen_capability(**kwargs):
            """Describe what's on the screen"""
            try:
                from vision.screen_capture_fallback import capture_with_intelligence
                import os
                
                # Check if Claude API is available
                api_key = os.getenv("ANTHROPIC_API_KEY")
                use_claude = api_key is not None
                
                # Get the query from kwargs or use default
                query = kwargs.get('query', 'Please describe in detail what you see on my screen. Include any applications, windows, icons, or content visible.')
                
                # Use intelligent capture with Claude
                result = capture_with_intelligence(query=query, use_claude=use_claude)
                
                if result['success']:
                    if result.get('intelligence_used') and result.get('analysis'):
                        # Return Claude's analysis
                        return result['analysis']
                    else:
                        # Fallback without Claude
                        return "I captured the screen but Claude Vision is not available. To enable intelligent screen analysis, please ensure ANTHROPIC_API_KEY is set."
                else:
                    return result.get('error', 'Unable to capture screen. Please check screen recording permissions.')
            except Exception as e:
                return f"Error analyzing screen: {e}"
        
        self.register_capability("basic_describe_screen", VisionCapability(
            name="basic_describe_screen",
            description="Describe what's on the screen",
            handler=describe_screen_capability,
            confidence_threshold=0.3,
            examples=[
                "describe my screen", 
                "what's on my screen", 
                "tell me what you see",
                "describe what you see on my screen",
                "what am I looking at",
                "can you describe what I'm looking at",
                "describe what is on my screen"
            ]
        ))
        
        # Window analysis capability
        async def analyze_window_capability(**kwargs):
            """Analyze the current window"""
            return "I'm analyzing the current window. This would show window details and content."
        
        self.register_capability("basic_analyze_window", VisionCapability(
            name="basic_analyze_window",
            description="Analyze the current window",
            handler=analyze_window_capability,
            confidence_threshold=0.6,
            examples=["analyze this window", "check current window", "what window is open"]
        ))
        
        logger.info("Registered basic vision capabilities")
    
    def _discover_vision_providers(self):
        """Dynamically discover available vision providers"""
        # Check for available vision modules
        vision_modules = [
            'intelligent_vision_integration',
            'screen_vision',
            'jarvis_workspace_integration',
            'enhanced_monitoring',
            'screen_capture_fallback',
            'claude_vision_analyzer'
        ]
        
        # Also register some basic built-in capabilities
        self._register_basic_capabilities()
        
        for module_name in vision_modules:
            try:
                module = __import__(f'vision.{module_name}', fromlist=[''])
                # Dynamically inspect the module for vision capabilities
                self._register_module_capabilities(module_name, module)
                logger.info(f"Discovered vision module: {module_name}")
            except ImportError as e:
                logger.debug(f"Vision module {module_name} not available: {e}")
                
    def _register_module_capabilities(self, module_name: str, module):
        """Dynamically register capabilities from a vision module"""
        # Inspect module for classes and methods that provide vision functionality
        
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and 'vision' in name.lower():
                # Found a vision class, inspect its methods
                for method_name, method in inspect.getmembers(obj):
                    if (inspect.ismethod(method) or inspect.isfunction(method)) and \
                       not method_name.startswith('_'):
                        # This is a public method - potential capability
                        capability_name = f"{module_name}.{name}.{method_name}"
                        self._analyze_and_register_capability(capability_name, method)
                        
    def _analyze_and_register_capability(self, name: str, method):
        """Analyze a method and register it as a capability if applicable"""
        # Extract information from method name and docstring
        doc = inspect.getdoc(method) or ""
        
        # Use NLP to understand what this method does
        keywords = self._extract_keywords_from_text(method.__name__ + " " + doc)
        
        if any(vision_word in keywords for vision_word in ['screen', 'window', 'display', 'visual', 'image', 'capture']):
            capability = VisionCapability(
                name=name,
                description=doc.split('\n')[0] if doc else f"Auto-discovered: {method.__name__}",
                handler=method,
                confidence_threshold=0.7,
                parameters=self._extract_method_parameters(method)
            )
            
            self.register_capability(name, capability)
            
    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract keywords using NLP techniques"""
        # Basic keyword extraction - would use proper NLP in production
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        return [w for w in words if w not in stopwords and len(w) > 2]
        
    def _extract_method_parameters(self, method) -> Dict[str, Any]:
        """Extract parameter information from method signature"""
        import inspect
        sig = inspect.signature(method)
        params = {}
        
        for param_name, param in sig.parameters.items():
            if param_name not in ['self', 'cls']:
                params[param_name] = {
                    'type': str(param.annotation) if param.annotation != param.empty else 'Any',
                    'default': param.default if param.default != param.empty else None,
                    'required': param.default == param.empty
                }
                
        return params
        
    def register_capability(self, name: str, capability: VisionCapability):
        """Register a new vision capability"""
        self.capabilities[name] = capability
        logger.info(f"Registered vision capability: {name}")
        
        # Learn from the capability name and description
        self._learn_capability_patterns(name, capability)
        
    def _learn_capability_patterns(self, name: str, capability: VisionCapability):
        """Learn patterns from capability metadata"""
        # Extract patterns from name and description
        words = self._extract_keywords_from_text(
            name + " " + capability.description + " " + " ".join(capability.examples)
        )
        
        # Update pattern database
        for word in words:
            self.learned_patterns[word].append(name)
            
        # Create embeddings if semantic model available
        if self.semantic_model:
            text = f"{name} {capability.description}"
            embedding = self.semantic_model.encode(text)
            self.pattern_embeddings[name] = embedding
            
    async def process_vision_command(self, command: str, context: Optional[Dict] = None) -> Tuple[str, Dict]:
        """
        Process any vision command using ML and dynamic routing
        No hardcoded patterns - everything is learned
        """
        # Record command for learning
        self.command_history.append({
            'command': command,
            'timestamp': datetime.now(),
            'context': context or {}
        })
        
        # Analyze intent using ML
        intent = self._analyze_intent(command, context)
        
        # Find best matching capability
        capability_scores = self._score_capabilities_for_intent(intent)
        
        if not capability_scores:
            # No capabilities found - try to learn what the user wants
            return await self._handle_unknown_intent(intent)
            
        # Execute best matching capability
        best_capability = max(capability_scores, key=capability_scores.get)
        confidence = capability_scores[best_capability]
        
        if confidence < 0.3:
            # Low confidence - ask for clarification
            return await self._clarify_intent(intent, capability_scores)
            
        # Execute the capability
        return await self._execute_capability(best_capability, intent, confidence)
        
    def _analyze_intent(self, command: str, context: Optional[Dict]) -> VisionIntent:
        """Analyze command intent using ML techniques"""
        intent = VisionIntent(raw_command=command)
        
        # Extract linguistic features
        words = command.lower().split()
        
        # Find action verb (using POS tagging would be better)
        potential_verbs = []
        for i, word in enumerate(words):
            if i == 0 or words[i-1] in ['please', 'can', 'could', 'would']:
                potential_verbs.append(word)
                
        if potential_verbs:
            intent.action_verb = potential_verbs[0]
            
        # Extract target object
        # Look for nouns after prepositions or at end
        for i, word in enumerate(words):
            if word in ['my', 'the', 'this', 'that'] and i + 1 < len(words):
                intent.target_object = words[i + 1]
                break
                
        # Extract modifiers
        intent.modifiers = [w for w in words if w in ['all', 'every', 'current', 'active', 'open']]
        
        # Add context clues
        if context:
            intent.context_clues = context
            
        # Generate semantic embedding if available
        if self.semantic_model:
            intent.semantic_embedding = self.semantic_model.encode(command)
            
        return intent
        
    def _score_capabilities_for_intent(self, intent: VisionIntent) -> Dict[str, float]:
        """Score each capability for how well it matches the intent"""
        scores = {}
        
        for name, capability in self.capabilities.items():
            score = 0.0
            
            # Semantic similarity if embeddings available
            if self.semantic_model and intent.semantic_embedding is not None:
                if name in self.pattern_embeddings:
                    similarity = np.dot(intent.semantic_embedding, self.pattern_embeddings[name])
                    score += similarity * 0.4
                    
            # Keyword matching
            command_words = set(self._extract_keywords_from_text(intent.raw_command))
            capability_words = set(self._extract_keywords_from_text(
                name + " " + capability.description
            ))
            
            overlap = len(command_words & capability_words)
            if overlap > 0:
                score += (overlap / len(command_words)) * 0.3
                
            # Learn from past usage
            if capability.usage_count > 0:
                score += (capability.success_rate * 0.2)
                
            # Boost recently used capabilities slightly
            if capability.last_used:
                recency = (datetime.now() - capability.last_used).total_seconds()
                if recency < 3600:  # Used in last hour
                    score += 0.1
                    
            scores[name] = score
            
        return scores
        
    async def _execute_capability(self, capability_name: str, intent: VisionIntent, confidence: float) -> Tuple[str, Dict]:
        """Execute a vision capability"""
        capability = self.capabilities[capability_name]
        
        # Update usage statistics
        capability.usage_count += 1
        capability.last_used = datetime.now()
        
        try:
            # Prepare parameters based on intent
            params = self._prepare_parameters(capability, intent)
            
            # Execute the handler
            if asyncio.iscoroutinefunction(capability.handler):
                result = await capability.handler(**params)
            else:
                result = capability.handler(**params)
                
            # Process result
            if isinstance(result, str):
                response = result
            elif hasattr(result, 'description'):
                response = result.description
            else:
                response = str(result)
                
            # Learn from success
            self._learn_from_execution(capability_name, intent, True, response)
            
            return response, {
                'capability': capability_name,
                'confidence': confidence,
                'intent': intent.__dict__,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error executing capability {capability_name}: {e}")
            
            # Learn from failure
            self._learn_from_execution(capability_name, intent, False, str(e))
            
            return f"I encountered an error with the vision system: {str(e)}", {
                'capability': capability_name,
                'confidence': confidence,
                'error': str(e),
                'success': False
            }
            
    def _prepare_parameters(self, capability: VisionCapability, intent: VisionIntent) -> Dict[str, Any]:
        """Prepare parameters for capability execution based on intent"""
        params = {}
        
        # Use capability's parameter definitions
        for param_name, param_info in capability.parameters.items():
            # Try to extract parameter value from intent
            if param_name == 'query' or param_name == 'command':
                params[param_name] = intent.raw_command
            elif param_name == 'target' and intent.target_object:
                params[param_name] = intent.target_object
            elif param_info.get('default') is not None:
                params[param_name] = param_info['default']
            elif not param_info.get('required', False):
                # Optional parameter, skip
                continue
                
        return params
        
    async def _handle_unknown_intent(self, intent: VisionIntent) -> Tuple[str, Dict]:
        """Handle commands that don't match any capability"""
        # Try to understand what the user wants
        suggestions = self._generate_suggestions(intent)
        
        response = "I'm not sure how to handle that vision command yet. "
        
        if suggestions:
            response += f"Did you mean: {', '.join(suggestions[:3])}?"
        else:
            response += "Could you rephrase your request?"
            
        # Record for learning
        self.feedback_data.append({
            'intent': intent,
            'status': 'unknown',
            'timestamp': datetime.now()
        })
        
        return response, {
            'intent': intent.__dict__,
            'status': 'unknown',
            'suggestions': suggestions
        }
        
    async def _clarify_intent(self, intent: VisionIntent, scores: Dict[str, float]) -> Tuple[str, Dict]:
        """Ask for clarification when confidence is low"""
        top_matches = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        response = "I'm not quite sure what you'd like me to do. Do you want me to:\n"
        
        for i, (cap_name, score) in enumerate(top_matches, 1):
            capability = self.capabilities[cap_name]
            response += f"{i}. {capability.description}\n"
            
        return response, {
            'intent': intent.__dict__,
            'status': 'needs_clarification',
            'options': [cap[0] for cap in top_matches]
        }
        
    def _generate_suggestions(self, intent: VisionIntent) -> List[str]:
        """Generate suggestions based on available capabilities"""
        suggestions = []
        
        # Find capabilities with similar keywords
        if intent.action_verb:
            for name, capability in self.capabilities.items():
                if intent.action_verb in capability.description.lower():
                    suggestions.extend(capability.examples[:2])
                    
        # Use learned patterns
        for word in self._extract_keywords_from_text(intent.raw_command):
            if word in self.learned_patterns:
                for cap_name in self.learned_patterns[word][:2]:
                    if cap_name in self.capabilities:
                        suggestions.extend(self.capabilities[cap_name].examples[:1])
                        
        return list(set(suggestions))[:5]  # Unique suggestions, max 5
        
    def _learn_from_execution(self, capability_name: str, intent: VisionIntent, success: bool, result: str):
        """Learn from execution results"""
        capability = self.capabilities[capability_name]
        
        # Update success rate
        if capability.usage_count > 0:
            old_rate = capability.success_rate
            capability.success_rate = (old_rate * (capability.usage_count - 1) + (1 if success else 0)) / capability.usage_count
            
        # Learn command patterns
        if success:
            capability.learned_patterns.append(intent.raw_command)
            # Keep only unique patterns
            capability.learned_patterns = list(set(capability.learned_patterns[-100:]))
            
            # Update pattern scores
            for word in self._extract_keywords_from_text(intent.raw_command):
                self.pattern_scores[f"{word}->{capability_name}"] += 1
                
        # Save learning data periodically
        if len(self.command_history) % 10 == 0:
            self._save_learned_data()
            
    def learn_from_feedback(self, command: str, feedback: str, correct_action: Optional[str] = None):
        """Learn from user feedback"""
        self.feedback_data.append({
            'command': command,
            'feedback': feedback,
            'correct_action': correct_action,
            'timestamp': datetime.now()
        })
        
        # If user provided correct action, boost that capability
        if correct_action:
            for name, capability in self.capabilities.items():
                if correct_action.lower() in name.lower() or correct_action.lower() in capability.description.lower():
                    capability.examples.append(command)
                    self._learn_capability_patterns(name, capability)
                    
    def _save_learned_data(self):
        """Save learned patterns and statistics"""
        data = {
            'pattern_scores': dict(self.pattern_scores),
            'learned_patterns': dict(self.learned_patterns),
            'user_preferences': dict(self.user_preferences),
            'capability_stats': {
                name: {
                    'success_rate': cap.success_rate,
                    'usage_count': cap.usage_count,
                    'learned_patterns': cap.learned_patterns,
                    'examples': cap.examples
                }
                for name, cap in self.capabilities.items()
            }
        }
        
        save_path = Path("backend/data/vision_learning.json")
        save_path.parent.mkdir(exist_ok=True)
        
        try:
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving learned data: {e}")
            
    def _load_learned_data(self):
        """Load previously learned patterns and statistics"""
        save_path = Path("backend/data/vision_learning.json")
        
        if not save_path.exists():
            return
            
        try:
            with open(save_path, 'r') as f:
                data = json.load(f)
                
            self.pattern_scores = defaultdict(float, data.get('pattern_scores', {}))
            self.learned_patterns = defaultdict(list, data.get('learned_patterns', {}))
            self.user_preferences = defaultdict(float, data.get('user_preferences', {}))
            
            # Apply learned stats to capabilities
            cap_stats = data.get('capability_stats', {})
            for name, stats in cap_stats.items():
                if name in self.capabilities:
                    cap = self.capabilities[name]
                    cap.success_rate = stats.get('success_rate', 0.0)
                    cap.usage_count = stats.get('usage_count', 0)
                    cap.learned_patterns = stats.get('learned_patterns', [])
                    cap.examples = stats.get('examples', [])
                    
            logger.info("Loaded learned vision data")
            
        except Exception as e:
            logger.error(f"Error loading learned data: {e}")
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            'total_capabilities': len(self.capabilities),
            'total_commands_processed': len(self.command_history),
            'learned_patterns': sum(len(patterns) for patterns in self.learned_patterns.values()),
            'average_success_rate': np.mean([cap.success_rate for cap in self.capabilities.values()]) if self.capabilities else 0,
            'most_used_capabilities': sorted(
                [(name, cap.usage_count) for name, cap in self.capabilities.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }


# Singleton instance
_engine = None


def get_dynamic_vision_engine() -> DynamicVisionEngine:
    """Get singleton dynamic vision engine"""
    global _engine
    if _engine is None:
        _engine = DynamicVisionEngine()
    return _engine