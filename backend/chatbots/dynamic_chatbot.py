"""
Dynamic Chatbot with Seamless Mode Switching (Including LangChain)
Automatically switches between SimpleChatbot, IntelligentChatbot, and LangChainChatbot based on memory
"""

import asyncio
import logging
from typing import Dict, Optional, List, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import gc
import psutil
import os

try:
    # Try relative imports first (when used as a module)
    from .simple_chatbot import SimpleChatbot
    from .intelligent_chatbot import IntelligentChatbot
    from .langchain_chatbot import LangChainChatbot, LANGCHAIN_AVAILABLE
    from ..memory.memory_manager import M1MemoryManager, MemoryState, ComponentPriority
    from ..memory.memory_safe_components import IntelligentComponentManager
    from ..memory.intelligent_memory_optimizer import IntelligentMemoryOptimizer
except ImportError:
    # Fall back to absolute imports (when run directly or from backend/)
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from chatbots.simple_chatbot import SimpleChatbot
    from chatbots.intelligent_chatbot import IntelligentChatbot
    from chatbots.langchain_chatbot import LangChainChatbot, LANGCHAIN_AVAILABLE
    from memory.memory_manager import M1MemoryManager, MemoryState, ComponentPriority
    from memory.memory_safe_components import IntelligentComponentManager
    from memory.intelligent_memory_optimizer import IntelligentMemoryOptimizer

logger = logging.getLogger(__name__)


class ChatbotMode(Enum):
    """Operating modes for the dynamic chatbot"""

    SIMPLE = "simple"
    INTELLIGENT = "intelligent"
    LANGCHAIN = "langchain"
    TRANSITIONING = "transitioning"


class ModeThresholds:
    """Memory thresholds for mode switching"""

    # Switch to LangChain when memory usage is below this
    LANGCHAIN_THRESHOLD = 0.50  # 50% memory usage

    # Switch to Intelligent when memory usage is below this
    UPGRADE_THRESHOLD = 0.65  # 65% memory usage

    # Switch to Simple when memory usage is above this
    DOWNGRADE_THRESHOLD = 0.80  # 80% memory usage

    # Minimum time between mode switches to prevent flapping
    MODE_SWITCH_COOLDOWN = timedelta(seconds=30)

    # Memory must be stable for this duration before upgrading
    STABILITY_DURATION = timedelta(seconds=10)


class DynamicChatbot:
    """
    Self-managing chatbot that dynamically switches between Simple, Intelligent, and LangChain modes
    based on available system resources
    """

    def __init__(
        self,
        memory_manager: M1MemoryManager,
        max_history_length: int = 10,
        auto_switch: bool = True,
        preserve_context: bool = True,
        prefer_langchain: bool = True,
    ):
        self.memory_manager = memory_manager
        self.max_history_length = max_history_length
        self.auto_switch = auto_switch
        self.preserve_context = preserve_context
        # Check environment variable for LangChain preference
        env_prefer_langchain = os.getenv("PREFER_LANGCHAIN", "0") == "1"
        self.prefer_langchain = (
            prefer_langchain or env_prefer_langchain
        ) and LANGCHAIN_AVAILABLE

        # TRUE LAZY LOADING: Don't create ANYTHING until needed
        self._memory_optimizer = None
        self._current_bot = None

        # Start with minimal state
        self.current_mode = ChatbotMode.SIMPLE
        self.mode_switches = 0
        self.last_mode_switch = datetime.now()
        self.mode_switch_count = 0
        self.memory_stable_since: Optional[datetime] = None

        # Track what's actually loaded
        self._loaded_components = set()

        # Monitoring task
        self.monitor_task: Optional[asyncio.Task] = None
        self._running = False

        # Performance metrics
        self.metrics = {
            "mode_switches": 0,
            "simple_responses": 0,
            "intelligent_responses": 0,
            "langchain_responses": 0,
            "memory_cleanups": 0,
            "failed_upgrades": 0,
            "intelligent_optimizations": 0,
            "optimization_successes": 0,
        }

        # Register with memory manager
        self.memory_manager.register_component(
            "dynamic_chatbot",
            ComponentPriority.CRITICAL,
            estimated_memory_mb=50,  # Reduced since we start minimal
        )

        # Register intelligent components for future use
        asyncio.create_task(self._register_components())

        # Start monitoring if auto-switch is enabled
        if self.auto_switch:
            asyncio.create_task(self.start_monitoring())

    @property
    def current_bot(self):
        """Lazy-loaded current bot - only created when needed"""
        if self._current_bot is None:
            # Create minimal bot only when first needed
            self._current_bot = SimpleChatbot(
                max_history_length=self.max_history_length
            )
            self._loaded_components.add("simple_chatbot")
            logger.info("Lazy-loaded SimpleChatbot")
        return self._current_bot

    @current_bot.setter
    def current_bot(self, value):
        """Set the current bot"""
        self._current_bot = value

    @property
    def memory_optimizer(self):
        """Lazy-loaded memory optimizer - only created when needed"""
        if self._memory_optimizer is None:
            self._memory_optimizer = IntelligentMemoryOptimizer()
            self._loaded_components.add("memory_optimizer")
            logger.info("Lazy-loaded IntelligentMemoryOptimizer")
        return self._memory_optimizer

    async def _register_components(self):
        """Register intelligent components with memory manager"""
        try:
            await self.memory_manager.register_intelligent_components()
        except Exception as e:
            logger.warning(f"Failed to register intelligent components: {e}")

    async def start_monitoring(self):
        """Start the resource monitoring loop"""
        if self._running:
            return

        self._running = True
        logger.info("Starting dynamic chatbot resource monitoring")

        while self._running:
            try:
                await self._check_and_switch_mode()
                await asyncio.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)

    async def stop_monitoring(self):
        """Stop the resource monitoring loop"""
        self._running = False
        if self.monitor_task:
            await self.monitor_task

    async def _check_and_switch_mode(self):
        """Check memory status and switch modes if needed"""
        memory_snapshot = await self.memory_manager.get_memory_snapshot()
        current_usage = memory_snapshot.percent

        # Check if we're in cooldown period
        time_since_switch = datetime.now() - self.last_mode_switch
        if time_since_switch < ModeThresholds.MODE_SWITCH_COOLDOWN:
            return

        # Determine target mode based on memory
        target_mode = self._determine_target_mode(current_usage)

        # Check if memory is stable for upgrades
        if target_mode.value > self.current_mode.value:  # Upgrade
            if self.memory_stable_since is None:
                self.memory_stable_since = datetime.now()
            elif (
                datetime.now() - self.memory_stable_since
                > ModeThresholds.STABILITY_DURATION
            ):
                # Memory has been stable, try upgrade
                await self._switch_to_mode(target_mode)
        else:
            self.memory_stable_since = None

            # Downgrade if needed
            if target_mode.value < self.current_mode.value:
                await self._switch_to_mode(target_mode)

    def _determine_target_mode(self, memory_usage: float) -> ChatbotMode:
        """Determine the best mode based on memory usage"""
        if memory_usage > ModeThresholds.DOWNGRADE_THRESHOLD:
            return ChatbotMode.SIMPLE
        elif memory_usage > ModeThresholds.UPGRADE_THRESHOLD:
            return ChatbotMode.INTELLIGENT
        elif memory_usage > ModeThresholds.LANGCHAIN_THRESHOLD:
            return ChatbotMode.INTELLIGENT
        else:
            # Low memory usage - use best available
            if self.prefer_langchain and LANGCHAIN_AVAILABLE:
                return ChatbotMode.LANGCHAIN
            else:
                return ChatbotMode.INTELLIGENT

    async def _switch_to_mode(self, target_mode: ChatbotMode):
        """Switch to the target mode"""
        if target_mode == self.current_mode:
            return

        if target_mode == ChatbotMode.SIMPLE:
            await self._downgrade_to_simple()
        elif target_mode == ChatbotMode.INTELLIGENT:
            if self.current_mode == ChatbotMode.SIMPLE:
                await self._upgrade_to_intelligent()
            else:
                await self._downgrade_from_langchain()
        elif target_mode == ChatbotMode.LANGCHAIN:
            await self._upgrade_to_langchain()

    async def _upgrade_to_intelligent(self):
        """Upgrade from Simple to Intelligent mode"""
        logger.info("Attempting to upgrade to Intelligent chatbot mode")

        # Only load memory optimizer if we need it
        if psutil.virtual_memory().percent > 65:
            logger.info("Memory usage high, running optimization...")
            await self.memory_optimizer.optimize_for_langchain(aggressive=False)

        # Check if we can load intelligent components
        can_load, reason = await self.memory_manager.can_load_component(
            "intelligent_chatbot"
        )
        if not can_load:
            logger.warning(f"Cannot upgrade to Intelligent mode: {reason}")
            self.metrics["failed_upgrades"] += 1
            return

        try:
            # Set transitioning state
            self.current_mode = ChatbotMode.TRANSITIONING

            # Save current conversation if preserving context
            conversation_history = None
            if (
                self.preserve_context
                and self._current_bot
                and hasattr(self._current_bot, "conversation_history")
            ):
                conversation_history = list(self._current_bot.conversation_history)

            # Clean up old bot
            if self._current_bot:
                if hasattr(self._current_bot, "cleanup"):
                    await self._current_bot.cleanup()
                self._current_bot = None
                gc.collect()

            # Create new Intelligent chatbot
            new_bot = IntelligentChatbot(
                memory_manager=self.memory_manager,
                max_history_length=self.max_history_length,
            )
            self._loaded_components.add("intelligent_chatbot")

            # Restore conversation history
            if conversation_history:
                new_bot.conversation_history = conversation_history
                logger.info(f"Restored {len(conversation_history)} conversation turns")

            # Switch to new bot
            self._current_bot = new_bot
            self.current_mode = ChatbotMode.INTELLIGENT
            self.last_mode_switch = datetime.now()
            self.metrics["mode_switches"] += 1
            self.mode_switches += 1

            logger.info("Successfully upgraded to Intelligent chatbot mode")
            self._log_memory_usage()

        except Exception as e:
            logger.error(f"Failed to upgrade to Intelligent mode: {e}")
            self.current_mode = ChatbotMode.SIMPLE
            self.metrics["failed_upgrades"] += 1
            # Ensure we have a working bot
            if self._current_bot is None:
                self._current_bot = SimpleChatbot(
                    max_history_length=self.max_history_length
                )
                self._loaded_components.add("simple_chatbot")

    async def _upgrade_to_langchain(self):
        """Upgrade to LangChain mode (highest tier)"""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available, falling back to Intelligent mode")
            await self._upgrade_to_intelligent()
            return

        logger.info("Attempting to upgrade to LangChain chatbot mode")

        # Only run memory optimization if needed
        current_mem = psutil.virtual_memory().percent
        if current_mem > 55:
            logger.info(f"Memory usage at {current_mem:.1f}%, running optimization...")
            success, report = await self.memory_optimizer.optimize_for_langchain(
                aggressive=True
            )

            if not success:
                logger.warning(f"Memory optimization failed, staying in current mode")
                self.metrics["failed_upgrades"] += 1
                return
            else:
                self.metrics["optimization_successes"] += 1
                logger.info(
                    f"Memory optimization successful! Freed {report['memory_freed_mb']:.0f} MB"
                )

        # Check if we can load LangChain components
        can_load, reason = await self.memory_manager.can_load_component(
            "langchain_chatbot"
        )
        if not can_load:
            logger.warning(f"Cannot upgrade to LangChain mode: {reason}")
            self.metrics["failed_upgrades"] += 1
            return

        try:
            # Set transitioning state
            self.current_mode = ChatbotMode.TRANSITIONING

            # Save conversation history
            conversation_history = None
            if (
                self.preserve_context
                and self._current_bot
                and hasattr(self._current_bot, "conversation_history")
            ):
                conversation_history = list(self._current_bot.conversation_history)

            # Clean up old bot
            if self._current_bot:
                if hasattr(self._current_bot, "cleanup"):
                    await self._current_bot.cleanup()
                self._current_bot = None
                gc.collect()

            # Create LangChain chatbot
            new_bot = LangChainChatbot(
                memory_manager=self.memory_manager,
                max_history_length=self.max_history_length,
                use_local_models=True,
            )
            self._loaded_components.add("langchain_chatbot")

            # Wait a bit for LangChain initialization
            await asyncio.sleep(3)

            # Restore conversation history
            if conversation_history:
                new_bot.conversation_history = conversation_history
                logger.info(
                    f"Restored {len(conversation_history)} conversation turns to LangChain"
                )

            # Switch to new bot
            self._current_bot = new_bot
            self.current_mode = ChatbotMode.LANGCHAIN
            self.last_mode_switch = datetime.now()
            self.metrics["mode_switches"] += 1
            self.mode_switches += 1

            logger.info("Successfully upgraded to LangChain chatbot mode")
            self._log_memory_usage()

        except Exception as e:
            logger.error(f"Failed to upgrade to LangChain mode: {e}")
            self.metrics["failed_upgrades"] += 1
            # Try Intelligent mode as fallback
            await self._upgrade_to_intelligent()

    async def _downgrade_from_langchain(self):
        """Downgrade from LangChain to Intelligent mode"""
        logger.info("Downgrading from LangChain to Intelligent mode")

        try:
            # Set transitioning state
            self.current_mode = ChatbotMode.TRANSITIONING

            # Save conversation history
            conversation_history = None
            if self.preserve_context and hasattr(
                self.current_bot, "conversation_history"
            ):
                conversation_history = list(self.current_bot.conversation_history)

            # Clean up LangChain
            if isinstance(self.current_bot, LangChainChatbot):
                await self.current_bot.cleanup()

            # Force memory cleanup
            await self.memory_manager.optimize_memory("normal")

            # Create Intelligent chatbot
            new_bot = IntelligentChatbot(
                memory_manager=self.memory_manager,
                max_history_length=self.max_history_length,
            )

            # Restore history
            if conversation_history:
                new_bot.conversation_history = conversation_history

            # Switch
            self._current_bot = None
            gc.collect()
            self._current_bot = new_bot
            self.current_mode = ChatbotMode.INTELLIGENT
            self.last_mode_switch = datetime.now()
            self.metrics["mode_switches"] += 1

            logger.info("Successfully downgraded to Intelligent mode")

        except Exception as e:
            logger.error(f"Error during downgrade from LangChain: {e}")
            await self._downgrade_to_simple()

    def _log_memory_usage(self):
        """Log current memory usage for debugging"""
        try:
            mem = psutil.virtual_memory()
            logger.debug(
                f"Memory usage: {mem.percent:.1f}% | "
                f"Loaded components: {', '.join(self._loaded_components)}"
            )
        except:
            pass

    async def _downgrade_to_simple(self):
        """Downgrade to Simple mode to free resources"""
        logger.info("Downgrading to Simple chatbot mode to free resources")

        try:
            # Set transitioning state
            self.current_mode = ChatbotMode.TRANSITIONING

            # Save conversation history
            conversation_history = None
            if self.preserve_context and hasattr(
                self.current_bot, "conversation_history"
            ):
                conversation_history = list(self.current_bot.conversation_history)

            # Clean up current bot
            if hasattr(self.current_bot, "cleanup"):
                await self.current_bot.cleanup()

            # Aggressive memory cleanup
            await self.memory_manager.optimize_memory("aggressive")
            self.metrics["memory_cleanups"] += 1

            # Delete old bot
            self._current_bot = None
            gc.collect()

            # Create Simple chatbot
            self.current_bot = SimpleChatbot(max_history_length=self.max_history_length)

            # Restore conversation history
            if conversation_history:
                self.current_bot.conversation_history = conversation_history
                logger.info(f"Preserved {len(conversation_history)} conversation turns")

            self.current_mode = ChatbotMode.SIMPLE
            self.last_mode_switch = datetime.now()
            self.metrics["mode_switches"] += 1

            logger.info("Successfully downgraded to Simple chatbot mode")

        except Exception as e:
            logger.error(f"Error during downgrade: {e}")
            # Ensure we have a working bot
            self.current_mode = ChatbotMode.SIMPLE
            self.current_bot = SimpleChatbot(max_history_length=self.max_history_length)

    async def force_mode(self, mode: str):
        """Manually force a specific mode"""
        mode_lower = mode.lower()

        if mode_lower == "simple":
            target = ChatbotMode.SIMPLE
        elif mode_lower == "intelligent":
            target = ChatbotMode.INTELLIGENT
        elif mode_lower == "langchain":
            target = ChatbotMode.LANGCHAIN
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Use 'simple', 'intelligent', or 'langchain'"
            )

        await self._switch_to_mode(target)

    # Delegate all chatbot methods to current bot
    async def generate_response(self, user_input: str) -> str:
        """Generate a response using the current chatbot mode"""
        # Track metrics
        if self.current_mode == ChatbotMode.LANGCHAIN:
            self.metrics["langchain_responses"] += 1
        elif self.current_mode == ChatbotMode.INTELLIGENT:
            self.metrics["intelligent_responses"] += 1
        else:
            self.metrics["simple_responses"] += 1

        # Generate response (lazy loads bot if needed)
        response = await self.current_bot.generate_response(user_input)

        # Track memory usage
        self._log_memory_usage()

        # Add mode indicator if in development
        if logger.isEnabledFor(logging.DEBUG):
            response = f"[{self.current_mode.value}] {response}"

        return response

    async def generate_response_with_context(
        self, user_input: str, context: Optional[Dict] = None
    ) -> Dict:
        """Generate response with context using current mode"""
        result = await self.current_bot.generate_response_with_context(
            user_input, context
        )

        # Add dynamic chatbot metadata
        result["chatbot_mode"] = self.current_mode.value
        result["mode_switches"] = self.metrics["mode_switches"]
        result["memory_status"] = {
            "mode": self.current_mode.value,
            "can_upgrade": await self._can_upgrade(),
        }

        return result

    async def _can_upgrade(self) -> bool:
        """Check if we can upgrade to a better mode"""
        snapshot = await self.memory_manager.get_memory_snapshot()

        if self.current_mode == ChatbotMode.SIMPLE:
            return snapshot.percent < ModeThresholds.UPGRADE_THRESHOLD
        elif self.current_mode == ChatbotMode.INTELLIGENT:
            return (
                snapshot.percent < ModeThresholds.LANGCHAIN_THRESHOLD
                and LANGCHAIN_AVAILABLE
            )

        return False

    async def get_conversation_history(self) -> List[Dict]:
        """Get conversation history from current bot"""
        if hasattr(self.current_bot, "get_conversation_history"):
            return await self.current_bot.get_conversation_history()
        return []

    async def clear_history(self):
        """Clear conversation history"""
        if hasattr(self.current_bot, "clear_history"):
            await self.current_bot.clear_history()

    def set_system_prompt(self, prompt: str):
        """Set system prompt on current bot"""
        if hasattr(self.current_bot, "set_system_prompt"):
            self.current_bot.set_system_prompt(prompt)

    @property
    def model_name(self) -> str:
        """Get current model name"""
        mode_prefix = f"dynamic-{self.current_mode.value}"
        if hasattr(self.current_bot, "model_name"):
            return f"{mode_prefix}-{self.current_bot.model_name}"
        return mode_prefix

    def get_capabilities(self) -> Dict[str, Any]:
        """Get current capabilities based on mode"""
        base_capabilities = {
            "dynamic_mode": True,
            "current_mode": self.current_mode.value,
            "auto_switch": self.auto_switch,
            "mode_switches": self.metrics["mode_switches"],
            "simple_responses": self.metrics["simple_responses"],
            "intelligent_responses": self.metrics["intelligent_responses"],
            "langchain_responses": self.metrics["langchain_responses"],
            "memory_cleanups": self.metrics["memory_cleanups"],
            "langchain_available": LANGCHAIN_AVAILABLE,
            "intelligent_optimizations": self.metrics["intelligent_optimizations"],
            "optimization_successes": self.metrics["optimization_successes"],
        }

        # Get capabilities from current bot
        if hasattr(self.current_bot, "get_capabilities"):
            bot_capabilities = self.current_bot.get_capabilities()
            base_capabilities.update(bot_capabilities)

        return base_capabilities

    # Proxy methods for missing attributes
    async def generate_response_stream(self, user_input: str):
        """Stream response token by token"""
        if hasattr(self.current_bot, "generate_response_stream"):
            async for token in self.current_bot.generate_response_stream(user_input):
                yield token
        else:
            # Fallback: simulate streaming by yielding the full response
            response = await self.generate_response(user_input)
            words = response.split()
            for word in words:
                yield word + " "
                await asyncio.sleep(0.05)

    @property
    def nlp_engine(self):
        """Get NLP engine from current bot"""
        return getattr(self.current_bot, "nlp_engine", None)

    @property
    def task_planner(self):
        """Get task planner from current bot"""
        return getattr(self.current_bot, "task_planner", None)

    @property
    def automation_engine(self):
        """Get automation engine from current bot"""
        return getattr(self.current_bot, "automation_engine", None)

    @property
    def _model_loaded(self):
        """Check if model is loaded in current bot"""
        return getattr(self.current_bot, "_model_loaded", True)

    async def add_knowledge(
        self, content: str, metadata: Optional[Dict] = None
    ) -> Dict:
        """Add knowledge to the knowledge base"""
        if hasattr(self.current_bot, "add_knowledge"):
            return await self.current_bot.add_knowledge(content, metadata)
        else:
            return {
                "success": False,
                "error": f"Knowledge management not available in {self.current_mode.value} mode",
            }

    async def cleanup(self):
        """Clean up resources"""
        logger.info(
            f"Cleaning up DynamicChatbot (loaded: {', '.join(self._loaded_components)})"
        )
        await self.stop_monitoring()

        if self._current_bot and hasattr(self._current_bot, "cleanup"):
            await self._current_bot.cleanup()

        if self._memory_optimizer:
            # Memory optimizer doesn't have cleanup, but we can clear references
            self._memory_optimizer = None

        # Clear all references
        self._current_bot = None
        self._loaded_components.clear()

        # Force garbage collection
        gc.collect()

        await self.memory_manager.unload_component("dynamic_chatbot")
        logger.info("DynamicChatbot cleanup complete")


# Example usage
if __name__ == "__main__":

    async def test_dynamic_chatbot():
        from memory.memory_manager import M1MemoryManager

        # Create memory manager
        memory_manager = M1MemoryManager()
        await memory_manager.start_monitoring()

        # Create dynamic chatbot with LangChain preference
        chatbot = DynamicChatbot(
            memory_manager, auto_switch=True, prefer_langchain=True
        )

        # Test conversations
        print("Testing Dynamic Chatbot with LangChain...")

        test_queries = [
            "Hello JARVIS!",
            "What is 2 + 2?",
            "Calculate 15 * 23 + 47",
            "What's the current weather like?",
            "Tell me about artificial intelligence",
            "What's your current mode?",
        ]

        for query in test_queries:
            print(f"\nUser: {query}")
            response = await chatbot.generate_response(query)
            print(f"JARVIS: {response}")
            print(f"Mode: {chatbot.current_mode.value}")

            # Wait a bit between queries
            await asyncio.sleep(2)

        # Get final capabilities
        caps = chatbot.get_capabilities()
        print(f"\nFinal Capabilities: {caps}")

        # Cleanup
        await chatbot.cleanup()
        await memory_manager.stop_monitoring()

    asyncio.run(test_dynamic_chatbot())
