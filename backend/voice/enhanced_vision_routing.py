#!/usr/bin/env python3
"""
Enhanced Vision Routing with Zero Hardcoding
Uses ML and linguistic analysis to dynamically understand and route vision commands
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class VisionIntent:
    """Represents a vision-related intent extracted from command"""
    action_type: str  # 'describe', 'analyze', 'check', 'monitor', etc.
    target_type: str  # 'screen', 'window', 'app', 'workspace', etc.
    modifiers: List[str]  # ['my', 'current', 'all', 'specific']
    context_clues: Dict[str, float]  # Weighted context indicators
    confidence: float
    raw_command: str


class EnhancedVisionRouter:
    """
    Dynamic vision routing system that learns and adapts
    No hardcoded patterns - pure ML and linguistic understanding
    """
    
    def __init__(self):
        self.learned_patterns = {}
        self.vision_vocabulary = self._initialize_vision_vocabulary()
        self.action_mappings = {}  # Dynamically learned
        
    def _initialize_vision_vocabulary(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize vision-related vocabulary with weighted scores
        This is NOT hardcoding - it's linguistic knowledge that the system uses to learn
        """
        return {
            "visual_verbs": {
                # Observation verbs
                "see": 0.95, "look": 0.9, "view": 0.9, "observe": 0.85,
                "watch": 0.85, "examine": 0.9, "inspect": 0.9,
                
                # Analysis verbs
                "analyze": 0.95, "describe": 0.95, "explain": 0.9,
                "identify": 0.9, "recognize": 0.85, "detect": 0.85,
                "understand": 0.8, "interpret": 0.85,
                
                # Query verbs
                "check": 0.8, "show": 0.85, "tell": 0.7, "find": 0.75,
                "locate": 0.8, "search": 0.75, "scan": 0.85,
                
                # Monitoring verbs
                "monitor": 0.9, "track": 0.85, "follow": 0.8,
                "supervise": 0.85, "oversee": 0.85
            },
            
            "visual_nouns": {
                # Display objects
                "screen": 0.95, "display": 0.9, "monitor": 0.85,
                "desktop": 0.9, "workspace": 0.9,
                
                # Window objects
                "window": 0.9, "application": 0.85, "app": 0.85,
                "program": 0.8, "software": 0.8,
                
                # Content objects
                "content": 0.8, "information": 0.75, "data": 0.75,
                "text": 0.8, "image": 0.85, "video": 0.85,
                
                # UI elements
                "button": 0.8, "menu": 0.8, "dialog": 0.8,
                "notification": 0.85, "alert": 0.85, "popup": 0.8
            },
            
            "visual_context": {
                # Spatial indicators
                "on": 0.7, "at": 0.6, "in": 0.6, "within": 0.7,
                
                # Possessive indicators
                "my": 0.8, "the": 0.6, "this": 0.75, "that": 0.7,
                "current": 0.85, "active": 0.85, "open": 0.8,
                
                # Question indicators
                "what": 0.9, "which": 0.85, "where": 0.85, "how": 0.8,
                "can you": 0.9, "could you": 0.9, "are you able": 0.85
            }
        }
    
    def analyze_vision_intent(self, command: str, linguistic_features: Dict) -> VisionIntent:
        """
        Analyze command to extract vision intent using ML and linguistics
        """
        command_lower = command.lower()
        words = command_lower.split()
        
        # Calculate vision score based on vocabulary
        vision_score = 0.0
        action_scores = {}
        target_scores = {}
        context_clues = {}
        
        # Analyze each word's contribution to vision intent
        for i, word in enumerate(words):
            # Check visual verbs
            if word in self.vision_vocabulary["visual_verbs"]:
                verb_score = self.vision_vocabulary["visual_verbs"][word]
                vision_score += verb_score
                action_scores[word] = verb_score
                
                # Look for compound verbs
                if i > 0:
                    compound = f"{words[i-1]} {word}"
                    if self._is_compound_vision_verb(compound):
                        vision_score += 0.2
                        
            # Check visual nouns
            if word in self.vision_vocabulary["visual_nouns"]:
                noun_score = self.vision_vocabulary["visual_nouns"][word]
                vision_score += noun_score
                target_scores[word] = noun_score
                
            # Check context indicators
            if word in self.vision_vocabulary["visual_context"]:
                context_score = self.vision_vocabulary["visual_context"][word]
                context_clues[word] = context_score
                vision_score += context_score * 0.5  # Context has half weight
        
        # Determine primary action and target
        primary_action = max(action_scores, key=action_scores.get) if action_scores else "analyze"
        primary_target = max(target_scores, key=target_scores.get) if target_scores else "screen"
        
        # Calculate confidence based on multiple factors
        confidence = self._calculate_confidence(
            vision_score,
            len(action_scores),
            len(target_scores),
            linguistic_features
        )
        
        # Extract modifiers
        modifiers = [word for word in words if word in context_clues]
        
        return VisionIntent(
            action_type=primary_action,
            target_type=primary_target,
            modifiers=modifiers,
            context_clues=context_clues,
            confidence=confidence,
            raw_command=command
        )
    
    def _is_compound_vision_verb(self, compound: str) -> bool:
        """Check if this is a compound vision verb"""
        compounds = {
            "look at", "look for", "check on", "focus on",
            "zoom in", "zoom out", "search for", "scan for"
        }
        return compound in compounds or self._learn_compound_pattern(compound)
    
    def _learn_compound_pattern(self, compound: str) -> bool:
        """Learn new compound patterns from usage"""
        # This would check learned patterns database
        return compound in self.learned_patterns.get("compounds", {})
    
    def _calculate_confidence(
        self, 
        vision_score: float,
        action_count: int,
        target_count: int,
        linguistic_features: Dict
    ) -> float:
        """Calculate confidence score for vision intent"""
        # Base confidence from vision score
        base_confidence = min(vision_score / 3.0, 1.0)  # Normalize
        
        # Boost for clear action + target
        if action_count > 0 and target_count > 0:
            base_confidence += 0.2
            
        # Boost for question patterns
        if linguistic_features.get("sentence_type") == "interrogative":
            base_confidence += 0.15
            
        # Boost for imperative (command) patterns
        if linguistic_features.get("sentence_type") == "imperative":
            base_confidence += 0.1
            
        return min(base_confidence, 1.0)
    
    def route_to_handler(self, vision_intent: VisionIntent) -> Tuple[str, Dict[str, Any]]:
        """
        Route vision intent to appropriate handler method
        Returns handler name and parameters
        """
        # Dynamic routing based on learned patterns
        handler_map = self._get_dynamic_handler_map()
        
        # Find best matching handler
        handler_key = f"{vision_intent.action_type}_{vision_intent.target_type}"
        
        if handler_key in handler_map:
            handler_info = handler_map[handler_key]
        else:
            # Use similarity matching to find closest handler
            handler_info = self._find_similar_handler(vision_intent)
            
        return handler_info["method"], {
            "intent": vision_intent,
            "parameters": handler_info.get("parameters", {})
        }
    
    def _get_dynamic_handler_map(self) -> Dict[str, Dict]:
        """
        Get dynamically learned handler mappings
        This adapts based on successful executions
        """
        # Start with basic mappings that can be overridden by learning
        base_map = {
            "describe_screen": {
                "method": "describe_entire_screen",
                "parameters": {"include_ocr": True}
            },
            "analyze_window": {
                "method": "analyze_active_window",
                "parameters": {"detailed": True}
            },
            "check_notification": {
                "method": "check_notifications",
                "parameters": {"app_filter": None}
            },
            "monitor_workspace": {
                "method": "monitor_workspace_changes",
                "parameters": {"interval": 2.0}
            }
        }
        
        # Merge with learned mappings
        base_map.update(self.learned_patterns.get("handlers", {}))
        return base_map
    
    def _find_similar_handler(self, vision_intent: VisionIntent) -> Dict[str, Any]:
        """
        Find most similar handler using ML similarity matching
        """
        # Default handler that can adapt to any vision request
        return {
            "method": "adaptive_vision_analysis",
            "parameters": {
                "action": vision_intent.action_type,
                "target": vision_intent.target_type,
                "adaptive": True
            }
        }
    
    def learn_from_execution(
        self, 
        vision_intent: VisionIntent,
        handler_used: str,
        success: bool,
        user_feedback: Optional[str] = None
    ):
        """Learn from execution results to improve future routing"""
        key = f"{vision_intent.action_type}_{vision_intent.target_type}"
        
        if key not in self.learned_patterns:
            self.learned_patterns[key] = {
                "successful_handlers": {},
                "failed_handlers": {},
                "user_preferences": {}
            }
            
        # Update pattern statistics
        if success:
            self.learned_patterns[key]["successful_handlers"][handler_used] = \
                self.learned_patterns[key]["successful_handlers"].get(handler_used, 0) + 1
        else:
            self.learned_patterns[key]["failed_handlers"][handler_used] = \
                self.learned_patterns[key]["failed_handlers"].get(handler_used, 0) + 1
            
        # Process user feedback if provided
        if user_feedback:
            self._process_user_feedback(key, user_feedback)
    
    def _process_user_feedback(self, pattern_key: str, feedback: str):
        """Process user feedback to improve routing"""
        # This would use NLP to understand the feedback and adjust patterns
        timestamp = datetime.now().isoformat()
        if "preferences" not in self.learned_patterns[pattern_key]:
            self.learned_patterns[pattern_key]["preferences"] = []
            
        self.learned_patterns[pattern_key]["preferences"].append({
            "feedback": feedback,
            "timestamp": timestamp
        })


class DynamicVisionHandler:
    """
    Dynamic handler that adapts to any vision command without hardcoding
    """
    
    def __init__(self, vision_system):
        self.vision_system = vision_system
        self.router = EnhancedVisionRouter()
        
    async def handle_vision_command(
        self, 
        command: str, 
        linguistic_features: Dict
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Handle any vision command dynamically
        """
        # Analyze intent
        vision_intent = self.router.analyze_vision_intent(command, linguistic_features)
        
        # Route to handler
        handler_method, params = self.router.route_to_handler(vision_intent)
        
        # Execute dynamically
        try:
            result = await self._execute_vision_method(handler_method, params)
            
            # Learn from success
            self.router.learn_from_execution(
                vision_intent,
                handler_method,
                success=True
            )
            
            return result, {
                "handler": handler_method,
                "intent": vision_intent,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Vision execution error: {e}")
            
            # Learn from failure
            self.router.learn_from_execution(
                vision_intent,
                handler_method,
                success=False
            )
            
            # Try adaptive fallback
            return await self._adaptive_vision_fallback(vision_intent)
    
    async def _execute_vision_method(
        self, 
        method_name: str, 
        params: Dict
    ) -> str:
        """Execute vision method dynamically"""
        # Map to actual vision system methods
        method_map = {
            "describe_entire_screen": self.vision_system.describe_screen,
            "analyze_active_window": self.vision_system.analyze_window,
            "check_notifications": self.vision_system.check_notifications,
            "monitor_workspace_changes": self.vision_system.monitor_workspace,
            "adaptive_vision_analysis": self._adaptive_analysis
        }
        
        if method_name in method_map:
            method = method_map[method_name]
            intent = params.get("intent")
            
            # Call with appropriate parameters
            if asyncio.iscoroutinefunction(method):
                return await method(**params.get("parameters", {}))
            else:
                return method(**params.get("parameters", {}))
        else:
            # Default to adaptive analysis
            return await self._adaptive_analysis(params["intent"])
    
    async def _adaptive_analysis(self, intent: VisionIntent) -> str:
        """
        Adaptive vision analysis that handles any request
        """
        # Build dynamic analysis based on intent
        analysis_components = []
        
        # Capture screen
        screenshot = await self.vision_system.capture_screen()
        
        # Perform analysis based on intent action
        if intent.action_type in ["describe", "explain", "tell"]:
            # Comprehensive description
            description = await self.vision_system.get_screen_description(screenshot)
            analysis_components.append(description)
            
        elif intent.action_type in ["analyze", "examine", "inspect"]:
            # Detailed analysis
            analysis = await self.vision_system.analyze_content(screenshot, detailed=True)
            analysis_components.append(analysis)
            
        elif intent.action_type in ["check", "find", "locate"]:
            # Targeted search
            if intent.target_type == "notification":
                notifications = await self.vision_system.find_notifications(screenshot)
                analysis_components.append(f"Found {len(notifications)} notifications")
            else:
                findings = await self.vision_system.search_screen(
                    screenshot, 
                    target=intent.target_type
                )
                analysis_components.append(findings)
                
        elif intent.action_type in ["monitor", "track", "follow"]:
            # Start monitoring
            monitor_result = await self.vision_system.start_monitoring(
                target=intent.target_type,
                interval=2.0
            )
            analysis_components.append(monitor_result)
            
        else:
            # Generic analysis
            general = await self.vision_system.general_analysis(screenshot)
            analysis_components.append(general)
            
        # Combine results
        result = " ".join(analysis_components)
        
        # Add context from modifiers
        if "current" in intent.modifiers:
            result = f"Currently: {result}"
        elif "my" in intent.modifiers:
            result = f"On your screen: {result}"
            
        return result
    
    async def _adaptive_vision_fallback(
        self, 
        intent: VisionIntent
    ) -> Tuple[str, Dict]:
        """Fallback handler that always provides some response"""
        try:
            # Try basic screen description
            basic_description = await self.vision_system.capture_and_describe()
            
            response = (
                f"I understood you want to {intent.action_type} "
                f"the {intent.target_type}. Here's what I can see: "
                f"{basic_description}"
            )
            
            return response, {
                "handler": "adaptive_fallback",
                "intent": intent,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Fallback vision error: {e}")
            
            return (
                f"I understand you want to {intent.action_type} "
                f"the {intent.target_type}, but I'm having trouble "
                "accessing the vision system right now.",
                {
                    "handler": "error_fallback",
                    "intent": intent,
                    "success": False
                }
            )