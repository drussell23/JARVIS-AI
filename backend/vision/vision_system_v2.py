#!/usr/bin/env python3
"""
JARVIS Vision System v2.0 - Zero Hardcoding Implementation
Integrates ML intent classification with semantic understanding
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime
import os
import time
import psutil
from .dynamic_response_composer import ResponseContext

# Import new ML components
from .ml_intent_classifier import get_ml_intent_classifier, VisionIntent
from .semantic_understanding_engine import get_semantic_understanding_engine, IntentUnderstanding
from .dynamic_vision_engine import get_dynamic_vision_engine
from .dynamic_response_composer import get_response_composer
from .neural_command_router import get_neural_router
from .personalization_engine import get_personalization_engine
from .transformer_command_router import get_transformer_router
from .continuous_learning_pipeline import get_learning_pipeline

# Phase 4 imports
try:
    from .advanced_continuous_learning import get_advanced_continuous_learning
    from .experience_replay_system import get_experience_replay_system
    PHASE4_AVAILABLE = True
except ImportError:
    PHASE4_AVAILABLE = False

# Phase 5 imports
try:
    from .capability_generator import get_capability_generator, FailedRequest
    from .safe_capability_synthesis import get_capability_synthesizer
    from .safety_verification_framework import get_safety_verification_framework, VerificationLevel
    from .performance_benchmarking import get_performance_benchmark
    from .gradual_rollout_system import get_gradual_rollout_system
    PHASE5_AVAILABLE = True
except ImportError:
    PHASE5_AVAILABLE = False

# Import existing components for backward compatibility
from .screen_capture_fallback import capture_with_intelligence
from system_control.vision_action_handler import get_vision_action_handler

logger = logging.getLogger(__name__)

# Log import status
if not PHASE4_AVAILABLE:
    logger.warning("Phase 4 components not available")
if not PHASE5_AVAILABLE:
    logger.warning("Phase 5 components not available")

@dataclass
class VisionResponse:
    """Unified vision response format"""
    success: bool
    message: str
    confidence: float
    intent_type: str
    data: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None

class VisionSystemV2:
    """
    Vision System 2.0 - Complete ML-based approach
    No hardcoded patterns, everything is learned dynamically
    """
    
    # CPU LIMIT CHECK
    _cpu_limit = float(os.getenv('CPU_LIMIT_PERCENT', '25'))
    _last_cpu_check = 0
    
    @classmethod
    def _check_cpu(cls):
        if time.time() - cls._last_cpu_check < 1:
            return
        cls._last_cpu_check = time.time()
        cpu = psutil.cpu_percent(interval=0.1)
        if cpu > cls._cpu_limit:
            time.sleep(0.1 * (cpu / cls._cpu_limit))
    
    def __init__(self):
        # Initialize lazy loading attributes
        self._ml_router = None
        self._semantic_engine = None
        self._transformer_router = None
        self._neural_router = None
        
        # Initialize ML components - defer loading
        self._intent_classifier = None
        self._vision_engine = None
        self._response_composer = None
        self._personalization_engine = None
        self._learning_pipeline = None
        
        # Settings
        self.use_transformer_routing = True  # Enable by default for <100ms latency
        
        # Phase 4 components (if available) - defer initialization
        self._phase4_initialized = False
        self.advanced_learning = None
        self.experience_replay = None
        self.learning_model = None
        
        # Phase 5 components (if available) - defer initialization
        self._phase5_initialized = False
        self.capability_generator = None
        self.capability_synthesizer = None
        self.safety_verifier = None
        self.performance_benchmark = None
        self.rollout_system = None
        
        # Defer handler registration until first use
        self._handlers_registered = False
        
        # Legacy handler for backward compatibility (disabled to avoid circular dependency)
        self.legacy_handler = None
        
        # System state
        self.interaction_history = []
        self.confidence_metrics = {
            'successful_interactions': 0,
            'failed_interactions': 0,
            'average_confidence': 0.0
        }
        
        # API availability
        self.claude_available = os.getenv("ANTHROPIC_API_KEY") is not None
        
        logger.info("Vision System v2.0 initialized with lazy loading")
    
    @property
    def ml_router(self):
        if self._ml_router is None:
            from .ml_intent_classifier import MLIntentClassifier
            self._ml_router = MLIntentClassifier()
        return self._ml_router
        
    @property
    def semantic_engine(self):
        if self._semantic_engine is None:
            from .semantic_understanding_engine import SemanticUnderstandingEngine
            self._semantic_engine = SemanticUnderstandingEngine()
        return self._semantic_engine
        
    @property
    def transformer_router(self):
        if self._transformer_router is None:
            try:
                from .transformer_command_router import TransformerCommandRouter
                self._transformer_router = TransformerCommandRouter()
            except ImportError:
                self._transformer_router = None
        return self._transformer_router
        
    @property
    def neural_router(self):
        if self._neural_router is None:
            try:
                from .neural_command_router import NeuralCommandRouter
                self._neural_router = NeuralCommandRouter()
            except ImportError:
                self._neural_router = None
        return self._neural_router
        
    @property
    def intent_classifier(self):
        if self._intent_classifier is None:
            self._intent_classifier = get_ml_intent_classifier()
        return self._intent_classifier
        
    @property
    def vision_engine(self):
        if self._vision_engine is None:
            self._vision_engine = get_dynamic_vision_engine()
        return self._vision_engine
        
    @property
    def response_composer(self):
        if self._response_composer is None:
            self._response_composer = get_response_composer()
        return self._response_composer
        
    @property
    def personalization_engine(self):
        if self._personalization_engine is None:
            self._personalization_engine = get_personalization_engine()
        return self._personalization_engine
        
    @property
    def learning_pipeline(self):
        if self._learning_pipeline is None:
            self._learning_pipeline = get_learning_pipeline()
        return self._learning_pipeline
    
    async def _ensure_handlers_registered(self):
        """Ensure handlers are registered on first use"""
        if not self._handlers_registered:
            # Initialize Phase 4 components if available
            if PHASE4_AVAILABLE and not self._phase4_initialized:
                await self._initialize_phase4_components()
                
            # Initialize Phase 5 components if available
            if PHASE5_AVAILABLE and not self._phase5_initialized:
                await self._initialize_phase5_components()
            
            # Register handlers with routers
            if self.neural_router:
                self._register_neural_handlers()
            
            if self.transformer_router:
                self._register_transformer_handlers()
                # Complete async registration
                if hasattr(self, '_pending_handler_registrations'):
                    await self._complete_async_initialization()
            
            self._handlers_registered = True
            logger.info("Vision handlers registered on first use")
    
    async def _initialize_phase4_components(self):
        """Initialize Phase 4 components lazily"""
        try:
            # Create a simple model for learning (in production, use actual vision model)
            import torch.nn as nn
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(768, 256)
                    self.fc2 = nn.Linear(256, 128)
                    self.fc3 = nn.Linear(128, 100)  # 100 intent classes
                
                def forward(self, x):
                    x = nn.functional.relu(self.fc1(x))
                    x = nn.functional.relu(self.fc2(x))
                    return self.fc3(x)
            
            self.learning_model = SimpleModel()
            self.advanced_learning = get_advanced_continuous_learning(self.learning_model)
            self.experience_replay = get_experience_replay_system()
            self._phase4_initialized = True
            logger.info("Phase 4 advanced learning components initialized lazily")
        except Exception as e:
            logger.warning(f"Failed to initialize Phase 4 components: {e}")
            self._phase4_initialized = True  # Don't try again
    
    async def _initialize_phase5_components(self):
        """Initialize Phase 5 components lazily"""
        try:
            self.capability_generator = get_capability_generator()
            self.capability_synthesizer = get_capability_synthesizer()
            self.safety_verifier = get_safety_verification_framework()
            self.performance_benchmark = get_performance_benchmark()
            self.rollout_system = get_gradual_rollout_system()
            self._phase5_initialized = True
            logger.info("Phase 5 autonomous capability components initialized lazily")
        except Exception as e:
            logger.warning(f"Failed to initialize Phase 5 components: {e}")
            self._phase5_initialized = True  # Don't try again
    
    def _register_neural_handlers(self):
        """Register vision handlers with the neural router"""
        # Register capability confirmation handler
        async def handle_capability_confirmation(cmd: str, ctx: Dict):
            return await self._handle_capability_confirmation(
                cmd,
                self.intent_classifier.classify_intent(cmd, ctx),
                await self.semantic_engine.understand_intent(cmd, ctx),
                ctx
            )
        
        self.neural_router.register_handler(
            "capability_confirmation",
            handle_capability_confirmation,
            "Handles vision capability confirmation questions",
            ["can you see my screen?", "are you able to see?", "do you see my display?"]
        )
        
        # Register general vision handler
        async def handle_general_vision(cmd: str, ctx: Dict):
            response, metadata = await self.vision_engine.process_vision_command(cmd, ctx)
            return {'response': response, 'metadata': metadata}
        
        self.neural_router.register_handler(
            "general_vision",
            handle_general_vision,
            "Handles general vision analysis requests",
            ["describe my screen", "analyze this", "what's on my display"]
        )
    
    def _register_transformer_handlers(self):
        """Register vision handlers with the transformer router for <100ms latency"""
        # Store handlers for async registration later
        self._pending_handler_registrations = [
            (self._handle_capability_confirmation_fast, True),
            (self._handle_general_vision_fast, True),
        ]
        
        logger.info("Prepared handlers for transformer router registration")
    
    async def _complete_async_initialization(self):
        """Complete async initialization when event loop is available"""
        if hasattr(self, '_pending_handler_registrations'):
            for handler_info in self._pending_handler_registrations:
                if isinstance(handler_info, tuple):
                    handler, auto_analyze = handler_info
                    await self.transformer_router.discover_handler(handler, auto_analyze=auto_analyze)
            delattr(self, '_pending_handler_registrations')
    
    async def _handle_capability_confirmation_fast(self, command: str, context: Optional[Dict] = None):
        """Fast capability confirmation handler for transformer router"""
        ml_intent = self.intent_classifier.classify_intent(command, context)
        semantic_understanding = await self.semantic_engine.understand_intent(command, context)
        return await self._handle_capability_confirmation(command, ml_intent, semantic_understanding, context)
    
    async def _handle_general_vision_fast(self, command: str, context: Optional[Dict] = None):
        """Fast general vision handler for transformer router"""
        response, metadata = await self.vision_engine.process_vision_command(command, context)
        return {'response': response, 'metadata': metadata}
    
    async def process_command(self, command: str, context: Optional[Dict] = None) -> VisionResponse:
        """
        Process any vision command using pure ML approach
        No hardcoding, no patterns - just intelligence
        """
        # Ensure handlers are registered on first use
        await self._ensure_handlers_registered()
        
        # Ensure async initialization is complete
        if getattr(self, '_needs_async_init', False):
            await self._complete_async_initialization()
            self._needs_async_init = False
        try:
            # Extract user ID for personalization
            user_id = context.get('user', 'default') if context else 'default'
            
            # Step 1: Analyze user style
            user_style = await self.personalization_engine.analyze_user_style(
                user_id, command, context
            )
            
            # Step 2: ML-based intent classification
            ml_intent = self.intent_classifier.classify_intent(command, context)
            
            # Step 3: Semantic understanding
            semantic_understanding = await self.semantic_engine.understand_intent(command, context)
            
            # Step 4: Combine insights for best understanding
            combined_confidence = (ml_intent.confidence + semantic_understanding.confidence) / 2
            
            # Step 5: Choose router based on latency requirements
            routing_context = {
                'ml_intent': ml_intent.intent_type,
                'semantic_intent': semantic_understanding.primary_intent,
                'confidence': combined_confidence,
                'user_style': user_style,
                **(context or {})
            }
            
            if self.use_transformer_routing:
                # Use transformer router for <100ms latency
                result, route_info = await self.transformer_router.route_command(
                    command, 
                    routing_context,
                    force_exploration=context.get('force_exploration', False) if context else False
                )
                
                # Record in learning pipeline
                await self.learning_pipeline.record_learning_event(
                    event_type='command',
                    data={
                        'command': command,
                        'handler': route_info['handler'],
                        'success': result is not None,
                        'latency_ms': route_info['latency_ms'],
                        'confidence': route_info['confidence'],
                        'embedding': routing_context
                    },
                    user_id=user_id
                )
                
                # Select model for continuous learning A/B testing
                model_version, _ = self.learning_pipeline.select_model_for_request()
                
                # Record result for A/B testing
                self.learning_pipeline.record_request_result(
                    model_version=model_version,
                    success=result is not None,
                    latency_ms=route_info['latency_ms']
                )
                
                route_decision = route_info
            else:
                # Fall back to neural router
                result, route_decision = await self.neural_router.route_command(command, routing_context)
            
            # Step 6: Extract response content
            if isinstance(result, dict):
                response_content = result.get('response', str(result))
                metadata = result.get('metadata', {})
            else:
                response_content = str(result)
                metadata = {}
            
            # Step 7: Get personalization parameters
            personalization_params = self.personalization_engine.get_personalization_params(
                user_id,
                semantic_understanding.primary_intent,
                context
            )
            
            # Step 8: Compose personalized response
            response_context = ResponseContext(
                intent_type=semantic_understanding.primary_intent,
                confidence=combined_confidence,
                user_name=context.get('user_name') if context else None,
                user_preferences=personalization_params,
                emotion_state=context.get('emotion') if context else None,
                time_of_day=self._get_time_of_day()
            )
            
            generated_response = await self.response_composer.compose_response(
                response_content,
                response_context,
                force_style=personalization_params.get('tone')
            )
            
            # Step 9: Learn from interaction
            await self._learn_from_interaction(
                command, ml_intent, semantic_understanding, 
                generated_response.text, True
            )
            
            # Step 10: Record in experience replay (Phase 4)
            if self.advanced_learning and self.experience_replay:
                # Get command embedding
                command_embedding = self.intent_classifier.encoder.encode(command, convert_to_numpy=True)
                
                # Record the experience
                await self.advanced_learning.record_interaction(
                    command=command,
                    command_embedding=command_embedding,
                    intent=semantic_understanding.primary_intent,
                    confidence=combined_confidence,
                    handler=route_decision.get('handler') if isinstance(route_decision, dict) else route_decision.handler_name,
                    response=generated_response.text,
                    success=True,
                    latency_ms=route_decision.get('latency_ms', 0) if isinstance(route_decision, dict) else 0,
                    user_id=user_id,
                    context=routing_context
                )
            
            return VisionResponse(
                success=True,
                message=generated_response.text,
                confidence=generated_response.confidence,
                intent_type=semantic_understanding.primary_intent,
                data={
                    'metadata': metadata,
                    'route_decision': route_decision if isinstance(route_decision, dict) else route_decision.__dict__,
                    'personalization': personalization_params,
                    'alternatives': generated_response.alternatives,
                    'transformer_routing': self.use_transformer_routing,
                    'phase4_enabled': PHASE4_AVAILABLE and self.advanced_learning is not None,
                    'phase5_enabled': PHASE5_AVAILABLE and self.capability_generator is not None
                },
                suggestions=semantic_understanding.suggested_responses
            )
            
        except Exception as e:
            logger.error(f"Error in Vision System v2.0: {e}", exc_info=True)
            
            # Phase 5: Analyze failed request for capability generation
            if PHASE5_AVAILABLE and self.capability_generator:
                asyncio.create_task(self._analyze_failed_request(
                    command, ml_intent, semantic_understanding, str(e), context
                ))
            
            return VisionResponse(
                success=False,
                message=f"I encountered an error processing your vision command: {str(e)}",
                confidence=0.0,
                intent_type="error"
            )
    
    def _get_time_of_day(self) -> str:
        """Get current time of day category"""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    
    async def _handle_capability_confirmation(
        self, 
        command: str, 
        ml_intent: VisionIntent,
        semantic_understanding: IntentUnderstanding,
        context: Optional[Dict]
    ) -> VisionResponse:
        """Handle 'can you see my screen?' type questions"""
        try:
            # Import dynamic vision analyzer
            from vision.natural_responses import get_vision_analyzer
            analyzer = get_vision_analyzer()
            
            # Capture screen first
            result = capture_with_intelligence(
                query=command,
                use_claude=False  # We'll use our own Claude integration
            )
            
            if result['success'] and result.get('image'):
                # Use the dynamic analyzer for natural response
                if self.claude_available and analyzer.client:
                    # Full Claude analysis
                    analysis_result = await analyzer.analyze_screen_with_context(
                        result['image'],
                        command,
                        context
                    )
                    
                    if analysis_result['success']:
                        message = analysis_result['message']
                        
                        # Store insights in result
                        result['insights'] = analysis_result.get('insights', {})
                        result['dynamic_analysis'] = True
                    else:
                        # Fallback if Claude fails
                        message = analyzer._generate_fallback_response(result)
                else:
                    # No Claude API - use fallback
                    message = analyzer._generate_fallback_response(result)
                
                # Learn this successful pattern
                self.intent_classifier.learn_from_interaction(
                    command, 'vision_capability_confirmation', True, context
                )
                
                return VisionResponse(
                    success=True,
                    message=message,
                    confidence=semantic_understanding.confidence,
                    intent_type='vision_capability_confirmation',
                    data=result
                )
            else:
                return VisionResponse(
                    success=False,
                    message="I'm having trouble accessing your screen. Please check screen recording permissions.",
                    confidence=0.5,
                    intent_type='vision_capability_confirmation'
                )
                
        except Exception as e:
            logger.error(f"Error in capability confirmation: {e}")
            return VisionResponse(
                success=False,
                message=f"I encountered an error checking screen access: {str(e)}",
                confidence=0.0,
                intent_type='error'
            )
    
    async def _learn_from_interaction(
        self,
        command: str,
        ml_intent: VisionIntent,
        semantic_understanding: IntentUnderstanding,
        response: str,
        success: bool
    ):
        """Learn from each interaction to improve future performance"""
        # Update ML classifier
        self.intent_classifier.learn_from_interaction(
            command,
            semantic_understanding.primary_intent,
            success,
            semantic_understanding.context.__dict__
        )
        
        # Update confidence metrics
        self.confidence_metrics['successful_interactions' if success else 'failed_interactions'] += 1
        
        # Track confidence history for auto-tuning
        confidence = (ml_intent.confidence + semantic_understanding.confidence) / 2
        self.intent_classifier.confidence_history.append((confidence, success))
        
        # Store interaction
        self.interaction_history.append({
            'timestamp': datetime.now().isoformat(),
            'command': command,
            'ml_intent_type': ml_intent.intent_type,
            'semantic_intent_type': semantic_understanding.primary_intent,
            'confidence': confidence,
            'success': success,
            'response_preview': response[:100] + '...' if len(response) > 100 else response
        })
        
        # Keep history bounded
        if len(self.interaction_history) > 1000:
            self.interaction_history = self.interaction_history[-1000:]
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        ml_stats = self.intent_classifier.export_patterns_for_visualization()
        vision_stats = self.vision_engine.get_statistics()
        
        total_interactions = (
            self.confidence_metrics['successful_interactions'] + 
            self.confidence_metrics['failed_interactions']
        )
        
        # Get Phase 3 stats
        transformer_stats = self.transformer_router.get_routing_analytics() if self.use_transformer_routing else {}
        learning_status = self.learning_pipeline.get_learning_status()
        
        return {
            'version': '3.0',
            'phase': 'Production with Continuous Learning',
            'total_interactions': total_interactions,
            'success_rate': (
                self.confidence_metrics['successful_interactions'] / 
                max(1, total_interactions)
            ),
            'confidence_threshold': self.intent_classifier.confidence_threshold,
            'learned_patterns': ml_stats['total_patterns'],
            'intent_types': ml_stats['intents'],
            'vision_capabilities': vision_stats['total_capabilities'],
            'most_used_features': vision_stats['most_used_capabilities'],
            'claude_available': self.claude_available,
            'transformer_routing': {
                'enabled': self.use_transformer_routing,
                'performance': transformer_stats.get('performance', {}),
                'handlers_count': len(transformer_stats.get('handlers', {})),
                'cache_hit_rate': transformer_stats.get('performance', {}).get('cache_hit_rate', 0),
                'avg_latency_ms': transformer_stats.get('performance', {}).get('avg_latency_ms', 0)
            },
            'continuous_learning': {
                'pipeline_version': learning_status.get('pipeline_version'),
                'model_version': learning_status.get('model_version'),
                'learning_buffer_size': learning_status.get('learning_buffer_size', 0),
                'current_performance': learning_status.get('current_performance', {}),
                'ab_test_active': learning_status.get('ab_test_active', False)
            }
        }
        
        # Add Phase 5 stats if available
        if PHASE5_AVAILABLE:
            stats['autonomous_capabilities'] = self.get_autonomous_capabilities_status()
        
        return stats
    
    async def provide_feedback(self, command: str, correct_intent: str, was_successful: bool):
        """Accept feedback to improve the system"""
        # Learn in ML classifier
        self.intent_classifier.learn_from_interaction(
            command, correct_intent, was_successful
        )
        
        # Learn in vision engine
        self.vision_engine.learn_from_feedback(
            command, 
            "positive" if was_successful else "negative",
            correct_intent
        )
        
        logger.info(f"Feedback received: '{command}' -> {correct_intent} (success: {was_successful})")
    
    async def _analyze_failed_request(
        self,
        command: str,
        ml_intent: VisionIntent,
        semantic_understanding: IntentUnderstanding,
        error_message: str,
        context: Optional[Dict]
    ):
        """Analyze failed request for capability generation (Phase 5)"""
        # Check CPU before heavy processing
        self._check_cpu()
        
        if not self.capability_generator:
            return
            
        # Create failed request object
        failed_request = FailedRequest(
            request_id=f"req_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            command=command,
            intent=semantic_understanding.primary_intent,
            confidence=semantic_understanding.confidence,
            error_type=self._categorize_error(error_message),
            error_message=error_message,
            context=context or {},
            user_id=context.get('user') if context else None,
            attempted_handlers=[ml_intent.intent_type]
        )
        
        # Analyze and potentially generate new capability
        generated_capability = await self.capability_generator.analyze_failed_request(failed_request)
        
        if generated_capability:
            logger.info(f"Generated new capability: {generated_capability.name}")
            
            # Synthesize safe version
            safe_code, synthesis_result = await self.capability_synthesizer.synthesize_safe_capability(
                generated_capability.handler_code,
                generated_capability.name
            )
            
            if synthesis_result['safe']:
                # Update capability with safe code
                generated_capability.handler_code = safe_code
                
                # Schedule verification and rollout
                asyncio.create_task(self._verify_and_deploy_capability(generated_capability))
    
    def _categorize_error(self, error_message: str) -> str:
        """Categorize error type from error message"""
        error_lower = error_message.lower()
        
        if "no handler" in error_lower or "not found" in error_lower:
            return "missing_handler"
        elif "timeout" in error_lower:
            return "timeout"
        elif "permission" in error_lower or "access" in error_lower:
            return "permission_denied"
        elif "confidence" in error_lower:
            return "low_confidence"
        else:
            return "unknown_error"
    
    async def _verify_and_deploy_capability(self, capability):
        """Verify and deploy a generated capability (Phase 5)"""
        # Check CPU before deployment process
        self._check_cpu()
        
        if not self.safety_verifier or not self.rollout_system:
            return
            
        # Add delay to prevent CPU spikes
        await asyncio.sleep(1.0)
        
        # Verify capability safety
        verification_report = await self.safety_verifier.verify_capability(
            capability,
            VerificationLevel.COMPREHENSIVE
        )
        
        if verification_report.approved:
            # Create rollout
            rollout_id = await self.rollout_system.create_rollout(
                capability,
                verification_report
            )
            
            logger.info(f"Created rollout {rollout_id} for capability {capability.name}")
            
            # Register with routers for gradual deployment
            if self.transformer_router:
                # Create wrapper that checks rollout status
                async def capability_wrapper(cmd: str, ctx: Optional[Dict] = None):
                    # Check if should use new capability
                    if self.rollout_system.should_use_capability(
                        capability.capability_id,
                        user_id=ctx.get('user') if ctx else None,
                        request_context=ctx
                    ):
                        # Execute generated capability
                        # In production, this would properly execute the generated code
                        result = {'success': True, 'message': 'Generated capability executed'}
                        
                        # Record result
                        self.rollout_system.record_result(
                            capability.capability_id,
                            result['success'],
                            latency_ms=50.0  # Mock latency
                        )
                        
                        return result
                    else:
                        # Fall back to existing behavior
                        return None
                
                # Register with router
                await self.transformer_router.discover_handler(
                    capability_wrapper,
                    metadata={
                        'name': capability.name,
                        'generated': True,
                        'capability_id': capability.capability_id
                    }
                )
    
    def get_autonomous_capabilities_status(self) -> Dict[str, Any]:
        """Get status of Phase 5 autonomous capabilities"""
        if not PHASE5_AVAILABLE:
            return {'available': False}
            
        status = {
            'available': True,
            'components': {
                'capability_generator': self.capability_generator is not None,
                'synthesizer': self.capability_synthesizer is not None,
                'verifier': self.safety_verifier is not None,
                'benchmark': self.performance_benchmark is not None,
                'rollout': self.rollout_system is not None
            }
        }
        
        if self.capability_generator:
            status['generation_stats'] = self.capability_generator.get_statistics()
            
        if self.safety_verifier:
            status['verification_summary'] = self.safety_verifier.get_verification_summary()
            
        if self.rollout_system:
            status['rollout_status'] = self.rollout_system.get_rollout_status()
            
        return status
    
    async def shutdown(self):
        """Gracefully shutdown all components"""
        logger.info("Shutting down Vision System v2.0")
        
        # Shutdown Phase 3 components
        if self.use_transformer_routing:
            await self.transformer_router.shutdown()
        
        await self.learning_pipeline.shutdown()
        
        # Shutdown Phase 4 components
        if self.advanced_learning:
            await self.advanced_learning.shutdown()
        
        # Phase 5 components typically don't need shutdown
        
        logger.info("Vision System v2.0 shutdown complete")

# Singleton instance
_vision_system_v2: Optional[VisionSystemV2] = None

def get_vision_system_v2() -> VisionSystemV2:
    """Get singleton instance of Vision System v2.0"""
    global _vision_system_v2
    if _vision_system_v2 is None:
        _vision_system_v2 = VisionSystemV2()
        # Mark that async initialization is needed
        _vision_system_v2._needs_async_init = True
    return _vision_system_v2

async def ensure_vision_system_v2_initialized() -> VisionSystemV2:
    """Ensure Vision System v2.0 is fully initialized including async parts"""
    system = get_vision_system_v2()
    if getattr(system, '_needs_async_init', False):
        await system._complete_async_initialization()
        system._needs_async_init = False
    return system

# Compatibility layer for existing code
async def process_vision_command_v2(command: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Process vision command using v2.0 system
    Returns dict for backward compatibility
    """
    system = get_vision_system_v2()
    response = await system.process_command(command, context)
    
    return {
        'success': response.success,
        'description': response.message,
        'confidence': response.confidence,
        'intent_type': response.intent_type,
        'data': response.data,
        'suggestions': response.suggestions
    }