#!/usr/bin/env python3
"""
Context-Aware Model Loader for JARVIS
=====================================

Intelligently loads ML models based on user context, proximity, and system state.
Optimized for 35% memory usage on 16GB systems.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import numpy as np

# Import components
from ml_memory_manager import get_ml_memory_manager, ModelPriority
from voice_unlock.ml.ml_manager import get_ml_manager
from voice_unlock.ml.quantized_models import VoiceModelQuantizer, QuantizedSVMInference

# Import enhanced logging
try:
    from ml_logging_config import ml_logger, memory_visualizer
    ENHANCED_LOGGING = True
except ImportError:
    ENHANCED_LOGGING = False

logger = logging.getLogger(__name__)


class SystemContext(Enum):
    """System context states"""
    IDLE = "idle"
    VOICE_COMMAND = "voice_command"
    AUTHENTICATION = "authentication"
    CONVERSATION = "conversation"
    SCREEN_ANALYSIS = "screen_analysis"
    CODE_GENERATION = "code_generation"
    SYSTEM_MONITOR = "system_monitor"
    MEMORY_CRITICAL = "memory_critical"


class ProximityLevel(Enum):
    """User proximity levels"""
    NEAR = "near"  # < 1m
    MEDIUM = "medium"  # 1-3m  
    FAR = "far"  # > 3m
    AWAY = "away"  # Not detected


@dataclass
class ContextState:
    """Current system context state"""
    primary_context: SystemContext
    secondary_contexts: Set[SystemContext] = field(default_factory=set)
    proximity: ProximityLevel = ProximityLevel.MEDIUM
    voice_active: bool = False
    screen_active: bool = False
    memory_pressure: float = 0.0  # 0-100%
    last_interaction: datetime = field(default_factory=datetime.now)
    interaction_frequency: float = 0.0  # interactions per minute
    

@dataclass 
class ModelLoadingPlan:
    """Plan for loading/unloading models"""
    to_load: List[Tuple[str, ModelPriority]]
    to_unload: List[str]
    to_preload: List[str]  # Load in background
    estimated_memory_mb: float
    reasoning: str


class ContextAwareModelLoader:
    """
    Advanced context-aware model loading system.
    Predicts and preloads models based on user behavior.
    """
    
    # Context to model mappings
    CONTEXT_MODELS = {
        SystemContext.IDLE: [],  # No models needed
        SystemContext.VOICE_COMMAND: ["whisper_base", "embeddings"],
        SystemContext.AUTHENTICATION: ["voice_biometric", "embeddings"],
        SystemContext.CONVERSATION: ["whisper_base", "embeddings", "sentiment"],
        SystemContext.SCREEN_ANALYSIS: ["vision_encoder", "embeddings"],
        SystemContext.CODE_GENERATION: ["embeddings"],
        SystemContext.SYSTEM_MONITOR: [],  # Minimal models
        SystemContext.MEMORY_CRITICAL: []  # Emergency - unload all
    }
    
    # Proximity-based model adjustments
    PROXIMITY_ADJUSTMENTS = {
        ProximityLevel.NEAR: {
            "preload_voice": True,
            "keep_vision": True,
            "aggressive_unload": False
        },
        ProximityLevel.MEDIUM: {
            "preload_voice": True,
            "keep_vision": False,
            "aggressive_unload": False
        },
        ProximityLevel.FAR: {
            "preload_voice": False,
            "keep_vision": False,
            "aggressive_unload": True
        },
        ProximityLevel.AWAY: {
            "preload_voice": False,
            "keep_vision": False,
            "aggressive_unload": True
        }
    }
    
    def __init__(self):
        self.ml_memory_manager = get_ml_memory_manager()
        self.voice_ml_manager = get_ml_manager()
        self.current_state = ContextState(primary_context=SystemContext.IDLE)
        self.state_history: List[Tuple[datetime, ContextState]] = []
        self.transition_patterns: Dict[str, int] = {}  # For learning patterns
        self.quantizer = VoiceModelQuantizer()
        self.quantized_models: Dict[str, Any] = {}
        
        # Background tasks
        self._monitor_task = None
        self._predictor_task = None
        self._running = False
        
    async def start(self):
        """Start the context-aware loader"""
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self._predictor_task = asyncio.create_task(self._prediction_loop())
        logger.info("Context-aware model loader started")
        
    async def stop(self):
        """Stop the loader"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
        if self._predictor_task:
            self._predictor_task.cancel()
        logger.info("Context-aware model loader stopped")
        
    async def update_context(self, 
                           primary: SystemContext,
                           secondary: Optional[Set[SystemContext]] = None,
                           proximity: Optional[ProximityLevel] = None):
        """
        Update system context and trigger model loading
        
        Args:
            primary: Primary system context
            secondary: Additional active contexts
            proximity: User proximity level
        """
        # Record state transition for pattern learning
        old_primary = self.current_state.primary_context
        transition_key = f"{old_primary.value}->{primary.value}"
        self.transition_patterns[transition_key] = self.transition_patterns.get(transition_key, 0) + 1
        
        # Enhanced logging for context change
        if ENHANCED_LOGGING:
            # Count affected models
            old_models = set(self.CONTEXT_MODELS.get(old_primary, []))
            new_models = set(self.CONTEXT_MODELS.get(primary, []))
            affected = len(old_models.symmetric_difference(new_models))
            ml_logger.context_change(old_primary.value, primary.value, affected)
        
        # Update state
        self.current_state.primary_context = primary
        if secondary:
            self.current_state.secondary_contexts = secondary
        if proximity:
            self.current_state.proximity = proximity
            
        self.current_state.last_interaction = datetime.now()
        
        # Add to history
        self.state_history.append((datetime.now(), ContextState(
            primary_context=primary,
            secondary_contexts=self.current_state.secondary_contexts.copy(),
            proximity=self.current_state.proximity,
            voice_active=self.current_state.voice_active,
            screen_active=self.current_state.screen_active,
            memory_pressure=self.current_state.memory_pressure
        )))
        
        # Trim history to last hour
        cutoff = datetime.now() - timedelta(hours=1)
        self.state_history = [(t, s) for t, s in self.state_history if t > cutoff]
        
        # Update ML memory manager context
        context_tags = [primary.value]
        context_tags.extend(ctx.value for ctx in self.current_state.secondary_contexts)
        self.ml_memory_manager.update_context(context_tags)
        
        # Execute loading plan
        plan = await self._create_loading_plan()
        await self._execute_plan(plan)
        
    async def handle_proximity_change(self, proximity: ProximityLevel, distance_m: Optional[float] = None):
        """
        Handle proximity changes for predictive loading
        
        Args:
            proximity: New proximity level
            distance_m: Actual distance in meters
        """
        self.current_state.proximity = proximity
        
        # Proximity-based preloading
        adjustments = self.PROXIMITY_ADJUSTMENTS[proximity]
        
        if adjustments["preload_voice"] and proximity == ProximityLevel.NEAR:
            # User approaching - preload voice models
            logger.info(f"User proximity {distance_m:.1f}m - preloading voice models")
            if ENHANCED_LOGGING:
                ml_logger.proximity_change(proximity.value, distance_m, "Preloading voice models")
            asyncio.create_task(self._preload_voice_models())
            
        elif adjustments["aggressive_unload"] and proximity in [ProximityLevel.FAR, ProximityLevel.AWAY]:
            # User away - unload non-essential models
            logger.info("User away - unloading non-essential models")
            if ENHANCED_LOGGING:
                ml_logger.proximity_change(proximity.value, distance_m, "Unloading non-essential models")
            await self._unload_non_essential()
            
    async def _create_loading_plan(self) -> ModelLoadingPlan:
        """Create an optimized model loading plan based on context"""
        to_load = []
        to_unload = []
        to_preload = []
        
        # Get required models for current context
        required_models = set(self.CONTEXT_MODELS.get(self.current_state.primary_context, []))
        
        # Add models for secondary contexts
        for ctx in self.current_state.secondary_contexts:
            required_models.update(self.CONTEXT_MODELS.get(ctx, []))
            
        # Adjust based on proximity
        proximity_adj = self.PROXIMITY_ADJUSTMENTS[self.current_state.proximity]
        if not proximity_adj["keep_vision"] and "vision_encoder" in required_models:
            required_models.discard("vision_encoder")
            
        # Check memory pressure
        if self.current_state.memory_pressure > 50:
            # High memory pressure - only keep critical models
            required_models = {m for m in required_models if self._is_critical(m)}
            
        # Get currently loaded models
        loaded_models = set(self.ml_memory_manager.models.keys())
        
        # Determine what to load/unload
        to_load_set = required_models - loaded_models
        to_unload_set = loaded_models - required_models
        
        # Prioritize loading
        for model in to_load_set:
            priority = self._get_model_priority(model)
            to_load.append((model, priority))
            
        to_load.sort(key=lambda x: x[1].value)  # Sort by priority
        
        # Predictive preloading
        predicted = await self._predict_next_models()
        to_preload = [m for m in predicted if m not in required_models and m not in loaded_models]
        
        # Calculate memory impact
        memory_impact = sum(
            self.ml_memory_manager.MODEL_CONFIGS[m].size_mb 
            for m, _ in to_load
        )
        
        return ModelLoadingPlan(
            to_load=to_load,
            to_unload=list(to_unload_set),
            to_preload=to_preload[:2],  # Max 2 preloads
            estimated_memory_mb=memory_impact,
            reasoning=f"Context: {self.current_state.primary_context.value}, "
                     f"Proximity: {self.current_state.proximity.value}, "
                     f"Memory: {self.current_state.memory_pressure:.0f}%"
        )
        
    async def _execute_plan(self, plan: ModelLoadingPlan):
        """Execute a model loading plan"""
        logger.info(f"Executing plan: {plan.reasoning}")
        
        if ENHANCED_LOGGING:
            # Log the plan details
            print("\n" + "=" * 60)
            print(f"📋 MODEL LOADING PLAN")
            print("=" * 60)
            print(f"Reason: {plan.reasoning}")
            if plan.to_unload:
                print(f"\n🔴 To Unload ({len(plan.to_unload)} models):")
                for model in plan.to_unload:
                    print(f"  - {model}")
            if plan.to_load:
                print(f"\n🟢 To Load ({len(plan.to_load)} models):")
                for model, priority in plan.to_load:
                    print(f"  - {model} (priority: {priority.name})")
            if plan.to_preload:
                print(f"\n🔮 To Preload ({len(plan.to_preload)} models):")
                for model in plan.to_preload:
                    print(f"  - {model}")
            print(f"\n💾 Estimated Memory Impact: {plan.estimated_memory_mb:.1f}MB")
            print("=" * 60 + "\n")
        
        # Unload first to free memory
        for model_id in plan.to_unload:
            await self.ml_memory_manager.unload_model(model_id)
            
        # Load required models
        for model_id, priority in plan.to_load:
            try:
                if model_id == "voice_biometric" and self.current_state.memory_pressure > 30:
                    # Use quantized version for voice
                    if ENHANCED_LOGGING:
                        ml_logger.quantized_load(model_id, 50.0, 6.0)  # Original 50MB -> 6MB quantized
                    await self._load_quantized_voice_model()
                else:
                    await self.ml_memory_manager.load_model(model_id)
            except Exception as e:
                logger.error(f"Failed to load {model_id}: {e}")
                
        # Preload in background
        if plan.to_preload and self.current_state.memory_pressure < 40:
            asyncio.create_task(self._preload_models(plan.to_preload))
            
    async def _predict_next_models(self) -> List[str]:
        """Predict which models will be needed next"""
        predictions = []
        
        # Analyze transition patterns
        current = self.current_state.primary_context.value
        
        # Find most likely transitions
        likely_transitions = [
            (k.split('->')[1], count)
            for k, count in self.transition_patterns.items()
            if k.startswith(f"{current}->")
        ]
        likely_transitions.sort(key=lambda x: x[1], reverse=True)
        
        # Get models for likely next contexts
        for next_context, _ in likely_transitions[:2]:
            try:
                context_enum = SystemContext(next_context)
                models = self.CONTEXT_MODELS.get(context_enum, [])
                predictions.extend(models)
            except ValueError:
                continue
                
        # Time-based predictions
        hour = datetime.now().hour
        if 6 <= hour <= 9:  # Morning - likely voice commands
            predictions.append("whisper_base")
        elif 9 <= hour <= 17:  # Work hours - likely code generation
            predictions.append("embeddings")
            
        return list(set(predictions))  # Remove duplicates
        
    async def _preload_voice_models(self):
        """Preload voice models for fast authentication"""
        try:
            # Load quantized voice model
            quantized_path = Path.home() / '.jarvis' / 'models' / 'quantized' / 'voice_auth.q8'
            if quantized_path.exists():
                model = self.quantizer.load_quantized_model(quantized_path)
                inference = self.quantizer.create_inference_engine(model)
                self.quantized_models['voice_auth'] = inference
                logger.info("Preloaded quantized voice model")
            else:
                # Fallback to regular loading
                await self.ml_memory_manager.load_model("voice_biometric")
        except Exception as e:
            logger.error(f"Failed to preload voice models: {e}")
            
    async def _load_quantized_voice_model(self):
        """Load ultra-optimized quantized voice model"""
        if 'voice_auth' in self.quantized_models:
            logger.info("Using preloaded quantized voice model")
            return self.quantized_models['voice_auth']
            
        # Load on demand
        await self._preload_voice_models()
        return self.quantized_models.get('voice_auth')
        
    async def _unload_non_essential(self):
        """Aggressively unload non-essential models"""
        essential = {"embeddings"}  # Always keep basic embeddings
        
        for model_id in list(self.ml_memory_manager.models.keys()):
            if model_id not in essential:
                await self.ml_memory_manager.unload_model(model_id)
                
    def _is_critical(self, model_id: str) -> bool:
        """Check if model is critical for current context"""
        config = self.ml_memory_manager.MODEL_CONFIGS.get(model_id)
        return config and config.priority == ModelPriority.CRITICAL
        
    def _get_model_priority(self, model_id: str) -> ModelPriority:
        """Get model priority based on context"""
        config = self.ml_memory_manager.MODEL_CONFIGS.get(model_id)
        if not config:
            return ModelPriority.LOW
            
        # Adjust priority based on context
        if self.current_state.primary_context == SystemContext.AUTHENTICATION:
            if model_id == "voice_biometric":
                return ModelPriority.CRITICAL
                
        return config.priority
        
    async def _monitor_loop(self):
        """Background monitoring loop"""
        while self._running:
            try:
                # Update memory pressure
                stats = self.ml_memory_manager.get_memory_usage()
                self.current_state.memory_pressure = stats["system_percent"]
                
                # Check for context timeout
                idle_time = (datetime.now() - self.current_state.last_interaction).total_seconds()
                if idle_time > 300 and self.current_state.primary_context != SystemContext.IDLE:
                    # 5 minutes of inactivity - switch to idle
                    await self.update_context(SystemContext.IDLE)
                    
                # Memory critical handling
                if self.current_state.memory_pressure > 70:
                    if self.current_state.primary_context != SystemContext.MEMORY_CRITICAL:
                        logger.warning(f"Memory critical: {self.current_state.memory_pressure:.0f}%")
                        if ENHANCED_LOGGING:
                            ml_logger.critical_memory(self.current_state.memory_pressure, "Switching to MEMORY_CRITICAL context")
                        await self.update_context(SystemContext.MEMORY_CRITICAL)
                        
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                
            await asyncio.sleep(10)  # Check every 10 seconds
            
    async def _prediction_loop(self):
        """Background prediction and preloading loop"""
        while self._running:
            try:
                # Calculate interaction frequency
                recent_interactions = len([
                    1 for t, _ in self.state_history 
                    if (datetime.now() - t).total_seconds() < 300  # Last 5 minutes
                ])
                self.current_state.interaction_frequency = recent_interactions / 5.0
                
                # Preload if high interaction frequency
                if self.current_state.interaction_frequency > 2.0:  # >2 per minute
                    predicted = await self._predict_next_models()
                    if predicted and self.current_state.memory_pressure < 40:
                        await self._preload_models(predicted[:1])  # Preload top prediction
                        
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                
            await asyncio.sleep(30)  # Check every 30 seconds
            
    async def _preload_models(self, model_ids: List[str]):
        """Preload models in background"""
        for model_id in model_ids:
            if self.current_state.memory_pressure > 50:
                break  # Stop if memory pressure increases
                
            try:
                await self.ml_memory_manager.load_model(model_id)
                logger.info(f"Preloaded {model_id}")
            except Exception as e:
                logger.debug(f"Preload failed for {model_id}: {e}")
                
    def get_status(self) -> Dict[str, Any]:
        """Get loader status"""
        loaded_models = list(self.ml_memory_manager.models.keys())
        
        return {
            "context": {
                "primary": self.current_state.primary_context.value,
                "secondary": [c.value for c in self.current_state.secondary_contexts],
                "proximity": self.current_state.proximity.value,
                "memory_pressure": self.current_state.memory_pressure,
                "interaction_frequency": self.current_state.interaction_frequency
            },
            "models": {
                "loaded": loaded_models,
                "quantized": list(self.quantized_models.keys()),
                "count": len(loaded_models)
            },
            "patterns": {
                "top_transitions": sorted(
                    self.transition_patterns.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            }
        }
        

# Global instance
_context_loader = None


def get_context_loader() -> ContextAwareModelLoader:
    """Get or create the global context loader"""
    global _context_loader
    if _context_loader is None:
        _context_loader = ContextAwareModelLoader()
    return _context_loader


async def initialize_context_aware_loading():
    """Initialize the context-aware loading system"""
    loader = get_context_loader()
    await loader.start()
    
    # Set initial context
    await loader.update_context(SystemContext.IDLE)
    
    logger.info("Context-aware model loading initialized")