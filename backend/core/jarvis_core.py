"""
JARVIS Core - Integrated system with Model Manager, Memory Controller, and Task Router
Built for scale and memory efficiency
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import json
from pathlib import Path

from .model_manager import ModelManager, ModelTier
from .memory_controller import MemoryController, MemoryPressure
from .task_router import TaskRouter, TaskAnalysis

logger = logging.getLogger(__name__)


class JARVISCore:
    """
    Core JARVIS system integrating all components for intelligent,
    memory-efficient operation
    """
    
    def __init__(self, 
                 models_dir: str = "models",
                 config_path: Optional[str] = None):
        """Initialize JARVIS Core with all components"""
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.model_manager = ModelManager(models_dir)
        self.memory_controller = MemoryController(
            target_percent=self.config.get("target_memory_percent", 60.0)
        )
        self.task_router = TaskRouter(self.model_manager, self.memory_controller)
        
        # Conversation context
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = self.config.get("max_history", 10)
        
        # Performance tracking
        self.session_stats = {
            "start_time": datetime.now(),
            "total_queries": 0,
            "model_switches": 0,
            "memory_optimizations": 0,
            "errors": 0
        }
        
        # Setup memory pressure callbacks
        self._setup_memory_callbacks()
        
        # Start monitoring
        asyncio.create_task(self._initialize_async())
        
    async def _initialize_async(self):
        """Async initialization tasks"""
        # Start memory monitoring
        await self.memory_controller.start_monitoring()
        
        # Ensure tiny model is loaded
        logger.info("JARVIS Core initialized and ready")
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        defaults = {
            "target_memory_percent": 60.0,
            "max_history": 10,
            "auto_optimize_memory": True,
            "predictive_loading": True,
            "quality_vs_speed": "balanced"  # "quality", "balanced", "speed"
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path) as f:
                    loaded = json.load(f)
                    defaults.update(loaded)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                
        return defaults
        
    def _setup_memory_callbacks(self):
        """Setup callbacks for memory pressure changes"""
        
        async def on_high_pressure(snapshot):
            """Handle high memory pressure"""
            logger.warning(f"High memory pressure: {snapshot.percent_used:.1f}%")
            
            # Unload advanced model if loaded
            if ModelTier.ADVANCED in self.model_manager.loaded_models:
                await self.model_manager.unload_model(ModelTier.ADVANCED)
                
        async def on_critical_pressure(snapshot):
            """Handle critical memory pressure"""
            logger.error(f"Critical memory pressure: {snapshot.percent_used:.1f}%")
            
            # Keep only tiny model
            for tier in [ModelTier.ADVANCED, ModelTier.STANDARD]:
                if tier in self.model_manager.loaded_models:
                    await self.model_manager.unload_model(tier)
                    
            # Force memory optimization
            if self.config.get("auto_optimize_memory", True):
                await self.memory_controller.optimize_memory(aggressive=True)
                self.session_stats["memory_optimizations"] += 1
                
        # Register callbacks
        self.memory_controller.register_pressure_callback(
            MemoryPressure.HIGH, on_high_pressure
        )
        self.memory_controller.register_pressure_callback(
            MemoryPressure.CRITICAL, on_critical_pressure
        )
        
    async def process_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Process a query through the intelligent routing system
        
        Args:
            query: User query
            **kwargs: Additional parameters (streaming, max_tokens, etc.)
            
        Returns:
            Response dictionary with result and metadata
        """
        start_time = datetime.now()
        self.session_stats["total_queries"] += 1
        
        try:
            # Get conversation context
            context = self._get_context()
            
            # Route to appropriate model
            model, routing_info = await self.task_router.route_task(query, context)
            
            # Track model switches
            if hasattr(self, '_last_model_tier') and self._last_model_tier != routing_info["model_tier"]:
                self.session_stats["model_switches"] += 1
            self._last_model_tier = routing_info["model_tier"]
            
            # Process with selected model
            response_text = await self._generate_response(model, query, context, **kwargs)
            
            # Update conversation history
            self._update_history(query, response_text)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Build response
            response = {
                "response": response_text,
                "metadata": {
                    "model_tier": routing_info["model_tier"],
                    "task_analysis": {
                        "type": routing_info["analysis"].task_type.value,
                        "complexity": routing_info["analysis"].complexity,
                        "confidence": routing_info["analysis"].confidence,
                        "reasoning": routing_info["analysis"].reasoning
                    },
                    "memory_state": routing_info["memory_state"],
                    "processing_time": processing_time,
                    "routing_time": routing_info["routing_time"]
                },
                "success": True
            }
            
            # Log performance
            logger.info(f"Query processed in {processing_time:.2f}s using {routing_info['model_tier']} model")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            self.session_stats["errors"] += 1
            
            return {
                "response": "I encountered an error processing your request. Please try again.",
                "error": str(e),
                "success": False
            }
            
    async def _generate_response(self, model: Any, query: str, 
                               context: List[str], **kwargs) -> str:
        """Generate response using the selected model"""
        # Build prompt with context
        prompt = self._build_prompt(query, context)
        
        # Get generation parameters
        max_tokens = kwargs.get("max_tokens", 256)
        temperature = kwargs.get("temperature", 0.7)
        
        # Generate response
        try:
            response = await asyncio.to_thread(
                model,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise
            
    def _build_prompt(self, query: str, context: List[str]) -> str:
        """Build prompt with context"""
        if not context:
            return query
            
        # Simple context injection
        context_str = "\n".join(context[-3:])  # Use last 3 exchanges
        return f"Context:\n{context_str}\n\nCurrent query: {query}"
        
    def _get_context(self) -> List[str]:
        """Get relevant conversation context"""
        context = []
        for entry in self.conversation_history[-3:]:  # Last 3 exchanges
            context.append(f"User: {entry['user']}")
            context.append(f"Assistant: {entry['assistant']}")
        return context
        
    def _update_history(self, query: str, response: str):
        """Update conversation history"""
        self.conversation_history.append({
            "user": query,
            "assistant": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim history if needed
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
            
    async def optimize_system(self) -> Dict[str, Any]:
        """Optimize system performance based on current state"""
        optimization_results = {
            "memory_optimization": None,
            "model_optimization": None,
            "suggestions": []
        }
        
        # Memory optimization
        memory_stats = self.memory_controller.get_memory_stats()
        if memory_stats["current"]["percent_used"] > 70:
            optimization_results["memory_optimization"] = await self.memory_controller.optimize_memory()
            
        # Model optimization based on workload
        recent_tasks = [
            {"type": h.get("task_type", "chat")} 
            for h in self.conversation_history[-10:]
        ]
        await self.model_manager.optimize_for_workload(recent_tasks)
        
        # Get optimization suggestions
        routing_suggestions = self.task_router.suggest_optimization()
        optimization_results["suggestions"].extend(routing_suggestions)
        
        return optimization_results
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "core": {
                "uptime": (datetime.now() - self.session_stats["start_time"]).total_seconds(),
                "total_queries": self.session_stats["total_queries"],
                "model_switches": self.session_stats["model_switches"],
                "memory_optimizations": self.session_stats["memory_optimizations"],
                "errors": self.session_stats["errors"]
            },
            "models": self.model_manager.get_model_stats(),
            "memory": self.memory_controller.get_memory_stats(),
            "routing": self.task_router.get_routing_stats(),
            "config": self.config
        }
        
    async def shutdown(self):
        """Gracefully shutdown JARVIS Core"""
        logger.info("Shutting down JARVIS Core...")
        
        # Stop memory monitoring
        await self.memory_controller.stop_monitoring()
        
        # Unload all models except tiny
        for tier in [ModelTier.ADVANCED, ModelTier.STANDARD]:
            if tier in self.model_manager.loaded_models:
                await self.model_manager.unload_model(tier)
                
        logger.info("JARVIS Core shutdown complete")
        

class JARVISAssistant:
    """High-level assistant interface for JARVIS Core"""
    
    def __init__(self, core: Optional[JARVISCore] = None):
        self.core = core or JARVISCore()
        
    async def chat(self, message: str, **kwargs) -> str:
        """Simple chat interface"""
        response = await self.core.process_query(message, **kwargs)
        return response["response"]
        
    async def chat_with_info(self, message: str, **kwargs) -> Dict[str, Any]:
        """Chat with full metadata"""
        return await self.core.process_query(message, **kwargs)
        
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return self.core.get_system_status()
        
    async def optimize(self) -> Dict[str, Any]:
        """Run system optimization"""
        return await self.core.optimize_system()