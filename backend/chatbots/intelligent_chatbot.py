"""
Intelligent Chatbot with Memory-Safe Component Integration
Enhances SimpleChatbot with advanced AI capabilities when memory permits
"""

import logging
from typing import List, Dict, Optional, AsyncGenerator, Any
from datetime import datetime
from dataclasses import dataclass, field
import random
import asyncio
try:
    # Try relative imports first (when used as a module)
    from .simple_chatbot import SimpleChatbot, ConversationTurn
    from ..memory.memory_safe_components import IntelligentComponentManager
    from ..memory.memory_manager import M1MemoryManager
    from ..utils.intelligent_cache import IntelligentCache
except ImportError:
    # Fall back to absolute imports (when run directly or from backend/)
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from chatbots.simple_chatbot import SimpleChatbot, ConversationTurn
    from memory.memory_safe_components import IntelligentComponentManager
    from memory.memory_manager import M1MemoryManager
    from utils.intelligent_cache import IntelligentCache

logger = logging.getLogger(__name__)


class IntelligentChatbot(SimpleChatbot):
    """
    Enhanced chatbot that intelligently uses AI components based on memory availability
    Falls back to SimpleChatbot when components aren't available
    """

    def __init__(
        self,
        memory_manager: M1MemoryManager,
        max_history_length: int = 10,
        model_name: str = "intelligent-chat",
    ):
        # Initialize base SimpleChatbot
        super().__init__(max_history_length, model_name)

        # Initialize intelligent components
        self.memory_manager = memory_manager
        self.component_manager = IntelligentComponentManager(memory_manager)

        # Initialize intelligent cache
        self._cache = IntelligentCache(memory_manager)

        # Track feature availability
        self.features_available = {
            "nlp_analysis": False,
            "rag_search": False,
            "voice_processing": False,
        }

        # Performance metrics
        self.metrics = {
            "nlp_uses": 0,
            "rag_uses": 0,
            "voice_uses": 0,
            "fallback_uses": 0,
        }

        # Initialize components in background
        asyncio.create_task(self._initialize_components())

        # Start cache optimization task
        asyncio.create_task(self._cache_optimization_loop())

    async def _initialize_components(self):
        """Initialize essential components based on memory availability"""
        try:
            await self.component_manager.preload_essential_components()
            await self._update_feature_availability()
        except Exception as e:
            logger.error(f"Error initializing components: {e}")

    async def _update_feature_availability(self):
        """Update which features are currently available"""
        status = self.component_manager.get_status()

        self.features_available["nlp_analysis"] = status.get("nlp_engine", {}).get(
            "is_available", False
        )
        self.features_available["rag_search"] = status.get("rag_engine", {}).get(
            "is_available", False
        )
        self.features_available["voice_processing"] = status.get(
            "voice_engine", {}
        ).get("is_available", False)

    async def generate_response_with_context(
        self, user_input: str, context: Optional[Dict] = None
    ) -> Dict:
        """
        Generate a response with intelligent features when available
        """
        start_time = datetime.now()

        # Try to use NLP analysis if available
        nlp_result = None
        if self.features_available["nlp_analysis"]:
            # Check cache first
            nlp_result = await self._cache.get_nlp_analysis(user_input)

            if not nlp_result:
                # Not in cache, analyze
                nlp_component = await self.component_manager.get_component("nlp_engine")
                if nlp_component:
                    nlp_result = await nlp_component.analyze_text(user_input)
                    if nlp_result:
                        # Cache the result
                        await self._cache.set_nlp_analysis(user_input, nlp_result)
                    self.metrics["nlp_uses"] += 1

        # Try to use RAG for knowledge-based queries
        rag_context = None
        if self.features_available["rag_search"] and self._is_knowledge_query(
            user_input, nlp_result
        ):
            # Check cache first
            rag_context = await self._cache.get_rag_search(user_input, k=3)

            if not rag_context:
                # Not in cache, search
                rag_component = await self.component_manager.get_component("rag_engine")
                if rag_component:
                    search_results = await rag_component.search_knowledge(
                        user_input, k=3
                    )
                    if search_results.get("success") and search_results.get("results"):
                        rag_context = search_results["results"]
                        # Cache the results
                        await self._cache.set_rag_search(user_input, rag_context, k=3)
                        self.metrics["rag_uses"] += 1

        # Generate response
        if rag_context:
            # Use RAG-enhanced response
            response = await self._generate_rag_response(
                user_input, rag_context, nlp_result
            )
        else:
            # Use standard response generation
            response = await self.generate_response(user_input)
            self.metrics["fallback_uses"] += 1

        # Enhance response quality if NLP is available
        if nlp_result and hasattr(self, "response_enhancer") and self.response_enhancer:
            try:
                enhanced = await asyncio.get_event_loop().run_in_executor(
                    None, self.response_enhancer.enhance_response, response, nlp_result
                )
                if enhanced:
                    response = enhanced
            except Exception as e:
                logger.debug(f"Response enhancement failed: {e}")

        # Calculate generation time
        generation_time = (datetime.now() - start_time).total_seconds()

        # Prepare response data
        response_data = {
            "response": response,
            "generation_time": generation_time,
            "model_used": self.model_name,
            "conversation_id": id(self),
            "turn_number": len(self.conversation_history) // 2,
            "context": context,
            "features_used": {
                "nlp_analysis": nlp_result is not None,
                "rag_search": rag_context is not None,
                "response_enhancement": nlp_result is not None,
            },
            "metrics": self.metrics.copy(),
        }

        # Add NLP analysis if available
        if nlp_result:
            response_data["nlp_analysis"] = {
                "intent": nlp_result.intent.intent.value,
                "intent_confidence": nlp_result.intent.confidence,
                "entities": [
                    {"text": e.text, "type": e.type} for e in nlp_result.entities
                ],
                "sentiment": nlp_result.sentiment,
                "is_question": nlp_result.is_question,
                "requires_action": nlp_result.requires_action,
                "topic": nlp_result.topic,
                "keywords": nlp_result.keywords[:5],
            }

        # Add RAG context if used
        if rag_context:
            response_data["knowledge_sources"] = [
                {"content": r["content"][:200] + "...", "score": r["score"]}
                for r in rag_context[:2]
            ]

        # Update feature availability periodically
        if random.random() < 0.1:  # 10% chance
            await self._update_feature_availability()

        return response_data

    def _is_knowledge_query(self, user_input: str, nlp_result: Optional[Any]) -> bool:
        """Determine if this is a knowledge-based query that would benefit from RAG"""
        # Check if it's a question
        if nlp_result and nlp_result.is_question:
            return True

        # Check for question patterns
        question_words = [
            "what",
            "when",
            "where",
            "who",
            "why",
            "how",
            "which",
            "can you explain",
        ]
        user_lower = user_input.lower()

        return (
            any(user_lower.startswith(word) for word in question_words)
            or "?" in user_input
        )

    async def _generate_rag_response(
        self, user_input: str, rag_context: List[Dict], nlp_result: Optional[Any]
    ) -> str:
        """Generate a response using RAG context"""
        # Extract relevant information from RAG results
        context_snippets = [r["content"] for r in rag_context[:2]]

        # Build a context-aware response
        if context_snippets:
            # Use the context to provide an informed answer
            context_text = " ".join(context_snippets)

            # Simple template-based response for now
            if nlp_result and nlp_result.intent.intent.value == "question":
                response = f"Based on my knowledge, {context_text}"
            else:
                response = f"I found some relevant information: {context_text}"

            # Add to history
            await self.add_to_history("assistant", response)
            return response
        else:
            # Fallback to standard response
            return await self.generate_response(user_input)

    async def add_knowledge(
        self, content: str, metadata: Optional[Dict] = None
    ) -> Dict:
        """Add knowledge to the RAG engine if available"""
        if not self.features_available["rag_search"]:
            return {
                "success": False,
                "error": "Knowledge base not available due to memory constraints",
            }

        rag_component = await self.component_manager.get_component("rag_engine")
        if rag_component:
            result = await rag_component.add_knowledge(content, metadata)
            return result
        else:
            return {"success": False, "error": "Failed to access knowledge base"}

    async def process_voice_input(self, audio_data: bytes) -> Dict:
        """Process voice input if voice engine is available"""
        if not self.features_available["voice_processing"]:
            return {
                "success": False,
                "error": "Voice processing not available due to memory constraints",
            }

        voice_component = await self.component_manager.get_component("voice_engine")
        if voice_component:
            # Transcribe audio
            transcription = await voice_component.transcribe_audio(audio_data)

            if transcription.get("success"):
                # Process the transcribed text
                text = transcription["text"]
                response_data = await self.generate_response_with_context(text)

                # Add voice-specific data
                response_data["voice_input"] = {
                    "transcribed_text": text,
                    "success": True,
                }

                self.metrics["voice_uses"] += 1
                return response_data
            else:
                return {
                    "success": False,
                    "error": f"Transcription failed: {transcription.get('error')}",
                }
        else:
            return {"success": False, "error": "Voice engine not accessible"}

    async def optimize_for_memory(self):
        """Optimize component loading based on usage patterns"""
        await self.component_manager.optimize_loaded_components()
        await self._update_feature_availability()

    async def _cache_optimization_loop(self):
        """Periodically optimize cache based on memory conditions"""
        while True:
            try:
                # Adapt cache sizes based on memory
                await self._cache.adapt_cache_sizes()

                # Clear old entries
                await self._cache.clear_old_entries()

                # Optimize loaded components
                await self.component_manager.optimize_loaded_components()

                # Update feature availability
                await self._update_feature_availability()

            except Exception as e:
                logger.error(f"Error in cache optimization: {e}")

            # Run every 5 minutes
            await asyncio.sleep(300)

    def get_capabilities(self) -> Dict[str, Any]:
        """Get current capabilities based on loaded components"""
        cache_stats = self._cache.get_stats()

        return {
            "basic_chat": True,  # Always available
            "nlp_analysis": self.features_available["nlp_analysis"],
            "knowledge_search": self.features_available["rag_search"],
            "voice_processing": self.features_available["voice_processing"],
            "memory_status": {
                "components_loaded": sum(
                    1 for v in self.features_available.values() if v
                ),
                "total_components": len(self.features_available),
            },
            "usage_metrics": self.metrics,
            "cache_stats": {
                "overall_hit_rate": cache_stats["overall"]["hit_rate"],
                "total_hits": cache_stats["overall"]["total_hits"],
                "total_misses": cache_stats["overall"]["total_misses"],
                "cache_sizes_mb": {
                    "nlp": cache_stats["by_type"]["nlp"]["stats"]["size_mb"],
                    "rag": cache_stats["by_type"]["rag"]["stats"]["size_mb"],
                    "response": cache_stats["by_type"]["response"]["stats"]["size_mb"],
                },
            },
        }

    async def cleanup(self):
        """Cleanup all components"""
        await self.component_manager.cleanup()
        await self._cache.cleanup()


# Example usage
if __name__ == "__main__":

    async def test_intelligent_chatbot():
        from memory.memory_manager import M1MemoryManager

        # Create memory manager
        memory_manager = M1MemoryManager()
        await memory_manager.start_monitoring()

        # Create intelligent chatbot
        chatbot = IntelligentChatbot(memory_manager)

        # Wait for initialization
        await asyncio.sleep(2)

        # Test various inputs
        test_inputs = [
            "Hello!",
            "What is machine learning?",
            "How are you today?",
            "Can you explain quantum computing?",
            "Thank you for your help!",
        ]

        for user_input in test_inputs:
            print(f"\nUser: {user_input}")
            response_data = await chatbot.generate_response_with_context(user_input)
            print(f"Assistant: {response_data['response']}")
            print(f"Features used: {response_data['features_used']}")

        # Show capabilities
        print(f"\nCapabilities: {chatbot.get_capabilities()}")

        # Cleanup
        await chatbot.cleanup()
        await memory_manager.cleanup()

    # Run test
    asyncio.run(test_intelligent_chatbot())
