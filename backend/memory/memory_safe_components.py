"""
Memory-Safe Component Wrappers for Intelligence Integration
Provides safe loading and management of AI components with memory constraints
"""

import asyncio
import logging
from typing import Optional, Any, Dict, Callable, TypeVar, Generic
from datetime import datetime
from abc import ABC, abstractmethod
import functools
from .memory_manager import M1MemoryManager, ComponentPriority

logger = logging.getLogger(__name__)

T = TypeVar("T")

class ComponentState:
    """Track component loading state"""

    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    ERROR = "error"

class MemorySafeComponent(Generic[T], ABC):
    """
    Base class for memory-safe components
    Handles automatic loading/unloading based on memory availability
    """

    def __init__(
        self,
        component_name: str,
        memory_manager: M1MemoryManager,
        priority: ComponentPriority = ComponentPriority.MEDIUM,
        estimated_memory_mb: int = 500,
        load_timeout: float = 30.0,
    ):
        self.component_name = component_name
        self.memory_manager = memory_manager
        self.priority = priority
        self.estimated_memory_mb = estimated_memory_mb
        self.load_timeout = load_timeout

        self._component: Optional[T] = None
        self._state = ComponentState.UNLOADED
        self._load_lock = asyncio.Lock()
        self._last_error: Optional[str] = None
        self._load_count = 0
        self._error_count = 0

        # Register with memory manager
        self.memory_manager.register_component(
            component_name, priority, estimated_memory_mb
        )

    @abstractmethod
    async def _create_component(self) -> T:
        """Create the actual component instance"""
        pass

    @abstractmethod
    async def _cleanup_component(self, component: T) -> None:
        """Cleanup component resources"""
        pass

    async def get_component(self) -> Optional[T]:
        """Get the component, loading it if necessary and possible"""
        if self._state == ComponentState.LOADED and self._component:
            return self._component

        # Try to load if not loaded
        success = await self.ensure_loaded()
        return self._component if success else None

    async def ensure_loaded(self) -> bool:
        """Ensure component is loaded, return success status"""
        async with self._load_lock:
            if self._state == ComponentState.LOADED:
                return True

            if self._state == ComponentState.LOADING:
                # Wait for current loading to complete
                await asyncio.sleep(0.1)
                return self._state == ComponentState.LOADED

            # Check if we can load
            can_load, reason = await self.memory_manager.can_load_component(
                self.component_name
            )

            if not can_load:
                logger.warning(f"Cannot load {self.component_name}: {reason}")
                self._last_error = reason
                return False

            # Try to load
            return await self._load_component()

    async def _load_component(self) -> bool:
        """Load the component with memory management"""
        self._state = ComponentState.LOADING

        try:
            logger.info(f"Loading {self.component_name}...")

            # Create component with timeout
            self._component = await asyncio.wait_for(
                self._create_component(), timeout=self.load_timeout
            )

            # Register with memory manager
            success = await self.memory_manager.load_component(
                self.component_name, self._component
            )

            if success:
                self._state = ComponentState.LOADED
                self._load_count += 1
                logger.info(f"{self.component_name} loaded successfully")
                return True
            else:
                # Cleanup if registration failed
                await self._cleanup_component(self._component)
                self._component = None
                self._state = ComponentState.UNLOADED
                self._error_count += 1
                return False

        except asyncio.TimeoutError:
            logger.error(f"Timeout loading {self.component_name}")
            self._last_error = "Load timeout"
            self._state = ComponentState.ERROR
            self._error_count += 1
            return False

        except Exception as e:
            logger.error(f"Error loading {self.component_name}: {e}")
            self._last_error = str(e)
            self._state = ComponentState.ERROR
            self._error_count += 1
            return False

    async def unload(self) -> bool:
        """Unload the component to free memory"""
        async with self._load_lock:
            if self._state != ComponentState.LOADED:
                return True

            self._state = ComponentState.UNLOADING

            try:
                if self._component:
                    await self._cleanup_component(self._component)

                success = await self.memory_manager.unload_component(
                    self.component_name
                )

                self._component = None
                self._state = ComponentState.UNLOADED

                return success

            except Exception as e:
                logger.error(f"Error unloading {self.component_name}: {e}")
                self._state = ComponentState.ERROR
                return False

    def get_status(self) -> Dict[str, Any]:
        """Get component status information"""
        return {
            "name": self.component_name,
            "state": self._state,
            "priority": self.priority.name,
            "estimated_memory_mb": self.estimated_memory_mb,
            "load_count": self._load_count,
            "error_count": self._error_count,
            "last_error": self._last_error,
            "is_available": self._state == ComponentState.LOADED,
        }

def memory_safe_method(method: Callable) -> Callable:
    """
    Decorator for methods that require a loaded component
    Automatically loads component if needed and handles errors
    """

    @functools.wraps(method)
    async def wrapper(self, *args, **kwargs):
        # Ensure we have a MemorySafeComponent instance
        if not isinstance(self, MemorySafeComponent):
            raise TypeError(
                "memory_safe_method can only be used on MemorySafeComponent"
            )

        # Try to get component
        component = await self.get_component()
        if not component:
            logger.warning(f"{self.component_name} not available for {method.__name__}")
            # Return None or raise exception based on method
            if method.__name__.startswith("get_") or method.__name__.startswith("is_"):
                return None
            else:
                raise RuntimeError(f"{self.component_name} not available")

        # Call the actual method with the component
        return await method(self, component, *args, **kwargs)

    return wrapper

class MemorySafeNLPEngine(MemorySafeComponent):
    """Memory-safe wrapper for NLP Engine"""

    def __init__(self, memory_manager: M1MemoryManager):
        super().__init__(
            component_name="nlp_engine",
            memory_manager=memory_manager,
            priority=ComponentPriority.HIGH,
            estimated_memory_mb=1500,
        )

    async def _create_component(self):
        """Create NLP Engine instance"""
        from engines.nlp_engine import NLPEngine

        return NLPEngine()

    async def _cleanup_component(self, component):
        """Cleanup NLP Engine resources"""
        # NLP Engine doesn't have specific cleanup
        pass

    @memory_safe_method
    async def analyze_text(self, component, text: str):
        """Analyze text with NLP engine"""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, component.analyze, text)
            return result
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return None

    @memory_safe_method
    async def get_intent(self, component, text: str):
        """Get intent from text"""
        analysis = await self.analyze_text(component, text)
        return analysis.intent if analysis else None

    @memory_safe_method
    async def extract_entities(self, component, text: str):
        """Extract entities from text"""
        analysis = await self.analyze_text(component, text)
        return analysis.entities if analysis else []

class MemorySafeRAGEngine(MemorySafeComponent):
    """Memory-safe wrapper for RAG Engine"""

    def __init__(self, memory_manager: M1MemoryManager):
        super().__init__(
            component_name="rag_engine",
            memory_manager=memory_manager,
            priority=ComponentPriority.MEDIUM,
            estimated_memory_mb=3000,
        )

    async def _create_component(self):
        """Create RAG Engine instance"""
        from engines.rag_engine import RAGEngine

        engine = RAGEngine()
        await engine.initialize()
        return engine

    async def _cleanup_component(self, component):
        """Cleanup RAG Engine resources"""
        if hasattr(component, "cleanup"):
            await component.cleanup()

    @memory_safe_method
    async def add_knowledge(self, component, content: str, metadata: Dict = None):
        """Add knowledge to the RAG engine"""
        try:
            document = await component.add_knowledge(content, metadata or {})
            return {
                "success": True,
                "document_id": document.id,
                "chunks": len(document.chunks),
            }
        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
            return {"success": False, "error": str(e)}

    @memory_safe_method
    async def search_knowledge(self, component, query: str, k: int = 5):
        """Search the knowledge base"""
        try:
            results = await component.knowledge_base.search(query, k=k)
            return {
                "success": True,
                "results": [
                    {
                        "content": r.chunk.content,
                        "score": r.score,
                        "metadata": r.chunk.metadata,
                    }
                    for r in results
                ],
            }
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return {"success": False, "error": str(e), "results": []}

    @memory_safe_method
    async def generate_response(self, component, query: str):
        """Generate response using RAG"""
        try:
            response_data = await component.generate_response(query)
            return response_data
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I'm unable to access my knowledge base at the moment.",
            }

class MemorySafeVoiceEngine(MemorySafeComponent):
    """Memory-safe wrapper for Voice Engine"""

    def __init__(self, memory_manager: M1MemoryManager):
        super().__init__(
            component_name="voice_engine",
            memory_manager=memory_manager,
            priority=ComponentPriority.MEDIUM,
            estimated_memory_mb=2000,
        )

    async def _create_component(self):
        """Create Voice Engine instance"""
        from engines.voice_engine import VoiceEngine

        return VoiceEngine()

    async def _cleanup_component(self, component):
        """Cleanup Voice Engine resources"""
        if hasattr(component, "cleanup"):
            await component.cleanup()

    @memory_safe_method
    async def transcribe_audio(self, component, audio_data: bytes):
        """Transcribe audio to text"""
        try:
            text = await component.transcribe(audio_data)
            return {"success": True, "text": text}
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return {"success": False, "error": str(e), "text": ""}

    @memory_safe_method
    async def synthesize_speech(self, component, text: str):
        """Convert text to speech"""
        try:
            audio_data = await component.synthesize(text)
            return {"success": True, "audio": audio_data}
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            return {"success": False, "error": str(e), "audio": None}

class IntelligentComponentManager:
    """
    Manages all memory-safe components with intelligent loading strategies
    """

    def __init__(self, memory_manager: M1MemoryManager):
        self.memory_manager = memory_manager
        self.components: Dict[str, MemorySafeComponent] = {}

        # Initialize components
        self.nlp = MemorySafeNLPEngine(memory_manager)
        self.rag = MemorySafeRAGEngine(memory_manager)
        self.voice = MemorySafeVoiceEngine(memory_manager)

        self.components = {
            "nlp_engine": self.nlp,
            "rag_engine": self.rag,
            "voice_engine": self.voice,
        }

        # Usage tracking for intelligent preloading
        self.usage_stats: Dict[str, Dict[str, Any]] = {
            name: {"uses": 0, "last_used": None} for name in self.components
        }

    async def preload_essential_components(self):
        """Preload components based on priority and available memory"""
        # Sort by priority
        sorted_components = sorted(
            self.components.items(), key=lambda x: x[1].priority.value
        )

        for name, component in sorted_components:
            if component.priority == ComponentPriority.CRITICAL:
                # Always try to load critical components
                await component.ensure_loaded()
            elif component.priority == ComponentPriority.HIGH:
                # Load high priority if we have enough memory
                can_load, _ = await self.memory_manager.can_load_component(name)
                if can_load:
                    await component.ensure_loaded()

    async def get_component(self, name: str) -> Optional[MemorySafeComponent]:
        """Get a component by name, tracking usage"""
        if name in self.components:
            # Update usage stats
            self.usage_stats[name]["uses"] += 1
            self.usage_stats[name]["last_used"] = datetime.now()

            return self.components[name]
        return None

    async def optimize_loaded_components(self):
        """Optimize which components are loaded based on usage patterns"""
        # Get current memory state
        snapshot = await self.memory_manager.get_memory_snapshot()

        if snapshot.percent > 0.8:  # If memory usage is high
            # Find least recently used components
            lru_components = sorted(
                self.usage_stats.items(),
                key=lambda x: x[1]["last_used"] or datetime.min,
            )

            # Unload LRU components that aren't critical
            for name, stats in lru_components:
                component = self.components[name]
                if component.priority != ComponentPriority.CRITICAL:
                    await component.unload()

                    # Check if we've freed enough memory
                    new_snapshot = await self.memory_manager.get_memory_snapshot()
                    if new_snapshot.percent < 0.7:
                        break

    def get_status(self) -> Dict[str, Any]:
        """Get status of all components"""
        return {
            name: component.get_status() for name, component in self.components.items()
        }

    async def cleanup(self):
        """Cleanup all components"""
        for component in self.components.values():
            await component.unload()

