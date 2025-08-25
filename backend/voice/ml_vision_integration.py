#!/usr/bin/env python3
"""
ML Vision Integration - Seamlessly integrates vision routing with existing handlers
Zero hardcoding approach to fixing vision command routing
"""

import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import asyncio

from .enhanced_vision_routing import EnhancedVisionRouter, DynamicVisionHandler, VisionIntent
from ..vision.enhanced_vision_system import EnhancedVisionSystem
from ..vision.intelligent_vision_integration import IntelligentVisionIntegration

logger = logging.getLogger(__name__)


class MLVisionIntegration:
    """
    Integrates ML-based vision routing with existing JARVIS command handlers
    Fixes the routing issue without any hardcoding
    """
    
    def __init__(self, anthropic_api_key: Optional[str] = None):
        self.vision_router = EnhancedVisionRouter()
        
        # Initialize vision systems
        if anthropic_api_key:
            self.vision_system = EnhancedVisionSystem(anthropic_api_key)
            self.intelligent_vision = IntelligentVisionIntegration(anthropic_api_key)
        else:
            # Fallback to basic vision
            from ..vision.screen_vision import ScreenVisionSystem
            self.vision_system = ScreenVisionSystem()
            self.intelligent_vision = None
            
        self.dynamic_handler = DynamicVisionHandler(self.vision_system)
        
        # Learning cache
        self.execution_history = []
        self.pattern_corrections = {}
        
    def should_handle_as_vision(
        self, 
        command: str,
        classification: Any,
        linguistic_features: Optional[Dict] = None
    ) -> Tuple[bool, float]:
        """
        Determine if command should be handled as vision without hardcoding
        Returns (is_vision, confidence)
        """
        # Get vision intent analysis
        vision_intent = self.vision_router.analyze_vision_intent(
            command, 
            linguistic_features or {}
        )
        
        # Check classification type
        is_classified_as_vision = (
            hasattr(classification, 'type') and 
            classification.type == 'vision'
        )
        
        # ML-based decision combining multiple signals
        vision_confidence = vision_intent.confidence
        classification_confidence = getattr(classification, 'confidence', 0.5)
        
        # Weighted decision
        if is_classified_as_vision:
            # Boost confidence if classifier agrees
            final_confidence = (vision_confidence * 0.7 + classification_confidence * 0.3)
        else:
            # Rely more on vision analysis if classifier is uncertain
            if classification_confidence < 0.6:
                final_confidence = vision_confidence * 0.9
            else:
                final_confidence = vision_confidence * 0.6
                
        # Decision threshold (adaptive based on learning)
        threshold = self._get_adaptive_threshold()
        
        return final_confidence >= threshold, final_confidence
    
    def _get_adaptive_threshold(self) -> float:
        """Get adaptive threshold based on learning history"""
        base_threshold = 0.65
        
        # Adjust based on recent performance
        if len(self.execution_history) > 10:
            recent = self.execution_history[-10:]
            success_rate = sum(1 for h in recent if h["success"]) / len(recent)
            
            if success_rate > 0.8:
                # Lower threshold if we're doing well
                return base_threshold - 0.05
            elif success_rate < 0.5:
                # Raise threshold if too many failures
                return base_threshold + 0.1
                
        return base_threshold
    
    async def enhance_system_command(
        self,
        command: str,
        original_handler,
        classification: Any
    ) -> Tuple[str, bool]:
        """
        Enhance system command handler to properly route vision commands
        Returns (response, handled)
        """
        # Check if this should be vision
        is_vision, confidence = self.should_handle_as_vision(
            command, 
            classification,
            getattr(classification, 'linguistic_features', None)
        )
        
        if is_vision and confidence > 0.7:
            # Route to vision handler
            logger.info(f"Routing to vision handler (confidence: {confidence:.2f})")
            
            try:
                response, metadata = await self.dynamic_handler.handle_vision_command(
                    command,
                    getattr(classification, 'linguistic_features', {})
                )
                
                # Record success
                self._record_execution(command, "vision", True, metadata)
                
                return response, True
                
            except Exception as e:
                logger.error(f"Vision handling error: {e}")
                # Fall through to original handler
                
        # Not vision or vision failed - use original handler
        return None, False
    
    async def fix_misrouted_command(
        self,
        command: str,
        error_message: str,
        classification: Any
    ) -> Optional[str]:
        """
        Fix commands that were misrouted to system handler
        This is called when system handler fails with 'Unknown action'
        """
        # Check if error indicates misrouting
        if "Unknown system action" in error_message or "couldn't complete that action" in error_message:
            # Extract the failed action
            import re
            match = re.search(r"Unknown system action: (\w+)", error_message)
            failed_action = match.group(1) if match else None
            
            # Check if this looks like a vision command
            is_vision, confidence = self.should_handle_as_vision(command, classification)
            
            if is_vision or (failed_action and failed_action in ["describe", "analyze", "check", "look"]):
                logger.info(f"Fixing misrouted vision command: {command}")
                
                try:
                    # Route to vision
                    response, metadata = await self.dynamic_handler.handle_vision_command(
                        command,
                        getattr(classification, 'linguistic_features', {})
                    )
                    
                    # Learn from this correction
                    self._learn_from_correction(command, "system", "vision", failed_action)
                    
                    return response
                    
                except Exception as e:
                    logger.error(f"Failed to fix misrouted command: {e}")
                    
        return None
    
    def _record_execution(
        self, 
        command: str, 
        handler_type: str,
        success: bool,
        metadata: Dict
    ):
        """Record execution for learning"""
        self.execution_history.append({
            "command": command,
            "handler": handler_type,
            "success": success,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep history manageable
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-50:]
    
    def _learn_from_correction(
        self,
        command: str,
        wrong_handler: str,
        correct_handler: str,
        failed_action: Optional[str] = None
    ):
        """Learn from routing corrections"""
        key = failed_action or command.split()[0].lower()
        
        if key not in self.pattern_corrections:
            self.pattern_corrections[key] = {
                "corrections": [],
                "patterns": {}
            }
            
        self.pattern_corrections[key]["corrections"].append({
            "command": command,
            "from": wrong_handler,
            "to": correct_handler,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update vision router's learned patterns
        if correct_handler == "vision":
            # Boost vision vocabulary for this pattern
            words = command.lower().split()
            for word in words:
                if word in self.vision_router.vision_vocabulary["visual_verbs"]:
                    # Increase weight
                    current = self.vision_router.vision_vocabulary["visual_verbs"][word]
                    self.vision_router.vision_vocabulary["visual_verbs"][word] = min(current + 0.05, 1.0)


class EnhancedCommandHandler:
    """
    Enhanced command handler that properly routes vision commands
    Drop-in replacement for fixing the routing issue
    """
    
    def __init__(self, original_handler, anthropic_api_key: Optional[str] = None):
        self.original_handler = original_handler
        self.vision_integration = MLVisionIntegration(anthropic_api_key)
        
    async def interpret_and_execute(self, command: str) -> str:
        """
        Enhanced interpret_and_execute that fixes vision routing
        """
        try:
            # First, check if this should be vision
            if hasattr(self.original_handler, 'last_classification'):
                classification = self.original_handler.last_classification
            else:
                classification = None
                
            # Try vision routing first
            response, handled = await self.vision_integration.enhance_system_command(
                command,
                self.original_handler,
                classification
            )
            
            if handled:
                return response
                
            # Fall back to original handler
            try:
                response = await self.original_handler.interpret_and_execute(command)
                return response
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if this is a misrouted vision command
                fixed_response = await self.vision_integration.fix_misrouted_command(
                    command,
                    error_msg,
                    classification
                )
                
                if fixed_response:
                    return fixed_response
                    
                # Re-raise if we couldn't fix it
                raise
                
        except Exception as e:
            logger.error(f"Enhanced command handler error: {e}")
            raise


def patch_system_handler(handler_instance, anthropic_api_key: Optional[str] = None):
    """
    Patch existing system handler to fix vision routing
    This is the main fix function
    """
    # Create enhanced handler
    enhanced = EnhancedCommandHandler(handler_instance, anthropic_api_key)
    
    # Replace the interpret_and_execute method
    original_method = handler_instance.interpret_and_execute
    handler_instance.interpret_and_execute = enhanced.interpret_and_execute
    
    # Store reference to enhanced handler
    handler_instance._vision_integration = enhanced.vision_integration
    
    # Add helper method
    handler_instance.handle_vision_command = enhanced.vision_integration.dynamic_handler.handle_vision_command
    
    logger.info("System handler patched with ML vision routing")
    
    return handler_instance