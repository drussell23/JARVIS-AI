"""
Claude API-powered Chatbot for JARVIS
Leverages Anthropic's Claude API for high-quality responses with minimal local resource usage
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

# Import centralized secret manager
try:
    from core.secret_manager import get_anthropic_key
    SECRET_MANAGER_AVAILABLE = True
except ImportError:
    try:
        from backend.core.secret_manager import get_anthropic_key
        SECRET_MANAGER_AVAILABLE = True
    except ImportError:
        SECRET_MANAGER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Check if anthropic package is available
try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic package not installed. Install with: pip install anthropic")


class ClaudeChatbot:
    """
    Cloud-based chatbot using Anthropic's Claude API with intelligent model selection

    Now supports:
    - Intelligent model selection (Claude API vs LLaMA 70B vs others)
    - Automatic fallback chains
    - RAM-aware routing
    - Cost optimization

    Perfect for M1 Macs with limited RAM as all processing happens in the cloud
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-haiku-20240307",  # Fast and cost-effective
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        use_intelligent_selection: bool = True,  # Enable intelligent model selection
    ):
        """
        Initialize Claude chatbot

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Claude model to use (haiku, sonnet, or opus)
            max_tokens: Maximum tokens in response
            temperature: Response randomness (0-1)
            system_prompt: System instructions for the assistant
            use_intelligent_selection: Use intelligent model selection (recommended)
        """
        # Get API key with fallback chain: parameter -> Secret Manager -> environment
        if api_key:
            self.api_key = api_key
        elif SECRET_MANAGER_AVAILABLE:
            self.api_key = get_anthropic_key()
        else:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key:
            logger.warning(
                "Anthropic API key not set. Intelligent model selection will prefer local models."
            )

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_intelligent_selection = use_intelligent_selection

        # Default JARVIS-style system prompt
        self.system_prompt = (
            system_prompt
            or """You are JARVIS, an intelligent AI assistant inspired by Tony Stark's AI from Iron Man.
You are helpful, witty, and highly capable. You speak with a refined, professional tone while being personable and occasionally adding subtle humor.
You excel at understanding context and providing insightful, well-structured responses."""
        )

        # Initialize client
        if ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            self.client = None

        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history_length = 10

        # Performance tracking
        self.total_tokens_used = 0
        self.api_calls_made = 0

    def is_available(self) -> bool:
        """Check if Claude API is available"""
        return ANTHROPIC_AVAILABLE and self.client is not None

    async def generate_response(self, user_input: str) -> str:
        """
        Process user input and generate response using intelligent model selection

        If intelligent selection enabled:
        1. Analyze query (intent, complexity, context)
        2. Select best model (LLaMA 70B, Claude API, or others)
        3. Execute with automatic fallback chain

        Falls back to direct Claude API if intelligent selection disabled or fails.

        Args:
            user_input: User's message

        Returns:
            AI response
        """
        # Try intelligent model selection first
        if self.use_intelligent_selection:
            try:
                return await self._generate_with_intelligent_selection(user_input)
            except Exception as e:
                logger.warning(f"Intelligent selection failed, falling back to Claude API: {e}")
                # Continue to direct Claude API below

        # Fallback: Direct Claude API
        if not self.is_available():
            return "Claude API is not available. Please install anthropic package and set API key."

        try:
            # Build messages for the API
            messages = self._build_messages(user_input)

            # Make API call
            start_time = datetime.now()

            # Use async client for better performance
            if not self.client:
                raise ValueError("Claude client not initialized")

            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=messages,
            )

            # Extract response
            ai_response = response.content[0].text

            # Track usage
            self.api_calls_made += 1
            if hasattr(response, "usage"):
                # Claude API usage structure
                if hasattr(response.usage, "input_tokens") and hasattr(
                    response.usage, "output_tokens"
                ):
                    self.total_tokens_used += (
                        response.usage.input_tokens + response.usage.output_tokens
                    )

            # Log performance
            response_time = (datetime.now() - start_time).total_seconds()
            token_info = "N/A"  # nosec B105
            if hasattr(response, "usage"):
                if hasattr(response.usage, "input_tokens") and hasattr(
                    response.usage, "output_tokens"
                ):
                    token_info = (
                        f"in:{response.usage.input_tokens}, out:{response.usage.output_tokens}"
                    )
            logger.info(
                f"Claude API call completed in {response_time:.2f}s "
                f"(model: {self.model}, tokens: {token_info})"
            )

            # Update conversation history
            self._update_history(user_input, ai_response)

            return ai_response

        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            if "credit balance is too low" in str(e):
                return (
                    "⚠️ Your Anthropic account needs credits to use Claude. "
                    "Please visit https://console.anthropic.com/settings/plans to add credits. "
                    "Claude API is pay-as-you-go and very affordable for regular use."
                )
            return f"I encountered an API error: {str(e)}. Please try again."
        except Exception as e:
            logger.error(f"Unexpected error in Claude chatbot: {e}")
            return "I encountered an unexpected error. Please try again."

    def _build_messages(self, user_input: str) -> List[Dict[str, str]]:
        """Build message list for Claude API including conversation history"""
        messages = []

        # Add conversation history (Claude format)
        for entry in self.conversation_history[-5:]:  # Last 5 exchanges for context
            messages.append({"role": "user", "content": entry["user"]})
            messages.append({"role": "assistant", "content": entry["assistant"]})

        # Add current user input
        messages.append({"role": "user", "content": user_input})

        return messages

    def _update_history(self, user_input: str, ai_response: str):
        """Update conversation history"""
        self.conversation_history.append(
            {"user": user_input, "assistant": ai_response, "timestamp": datetime.now().isoformat()}
        )

        # Trim history if needed
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length :]

    async def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")

    async def _generate_with_intelligent_selection(self, user_input: str) -> str:
        """
        Generate response using intelligent model selection

        This method:
        1. Imports the hybrid orchestrator
        2. Builds context from conversation history
        3. Calls execute_with_intelligent_model_selection()
        4. Returns the selected model's response
        """
        try:
            from backend.core.hybrid_orchestrator import HybridOrchestrator

            # Get or create orchestrator
            orchestrator = HybridOrchestrator()
            if not orchestrator.is_running:
                await orchestrator.start()

            # Build context from conversation history
            context = {
                "conversation_history": self.conversation_history[-3:],  # Last 3 exchanges
                "user_focus": "casual",  # Chatbot = casual conversation
                "system_prompt": self.system_prompt,
            }

            # Build full prompt with system instructions
            full_prompt = f"{self.system_prompt}\n\nUser: {user_input}\n\nAssistant:"

            # Execute with intelligent model selection
            start_time = datetime.now()
            result = await orchestrator.execute_with_intelligent_model_selection(
                query=full_prompt,
                intent="conversational_ai",
                required_capabilities={"conversational_ai", "chatbot_inference"},
                context=context,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            if not result.get("success"):
                raise Exception(result.get("error", "Unknown error"))

            # Extract response
            ai_response = result.get("text", "").strip()

            # Log which model was used
            model_used = result.get("model_used", "unknown")
            response_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Chatbot response generated in {response_time:.2f}s using {model_used}")

            # Update tracking
            self.api_calls_made += 1
            if model_used == "claude_api":
                # Estimate tokens for Claude API
                self.total_tokens_used += (
                    len(user_input.split()) * 1.3 + len(ai_response.split()) * 1.3
                )

            # Update conversation history
            self._update_history(user_input, ai_response)

            return ai_response

        except ImportError:
            logger.warning("Hybrid orchestrator not available, using direct Claude API")
            raise
        except Exception as e:
            logger.error(f"Error in intelligent model selection: {e}")
            raise

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        return {
            "api_calls": self.api_calls_made,
            "total_tokens": self.total_tokens_used,
            "model": self.model,
            "history_length": len(self.conversation_history),
            "intelligent_selection": self.use_intelligent_selection,
        }

    def change_model(self, model: str):
        """
        Change Claude model

        Available models:
        - claude-3-haiku-20240307: Fast and cost-effective
        - claude-3-sonnet-20240229: Balanced performance
        - claude-3-opus-20240229: Most capable
        """
        self.model = model
        logger.info(f"Changed Claude model to: {model}")

    async def stream_process(self, user_input: str):
        """
        Process user input with streaming response
        Yields response chunks as they arrive
        """
        if not self.is_available():
            yield "Claude API is not available. Please install anthropic package and set API key."
            return

        try:
            messages = self._build_messages(user_input)

            # Create streaming response
            if not self.client:
                raise ValueError("Claude client not initialized")

            stream = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=messages,
                stream=True,
            )

            full_response = ""
            async for chunk in stream:
                if chunk.type == "content_block_delta":
                    text = chunk.delta.text
                    full_response += text
                    yield text

            # Update history with complete response
            self._update_history(user_input, full_response)
            self.api_calls_made += 1

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"\nError during streaming: {str(e)}"

    async def generate_response_with_context(
        self, user_input: str, context: Optional[Dict] = None
    ) -> Dict:
        """
        Generate response with metadata (compatible with existing API)
        """
        response = await self.generate_response(user_input)

        return {
            "response": response,
            "mode": "claude",
            "model": self.model,
            "context_used": context is not None,
        }

    async def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation_history.copy()

    async def cleanup(self):
        """Cleanup resources (no-op for cloud API)"""
        logger.info("Claude chatbot cleanup (no resources to free)")

    def set_system_prompt(self, prompt: str):
        """Update system prompt"""
        self.system_prompt = prompt
        logger.info("Updated Claude system prompt")

    @property
    def model_name(self) -> str:
        """Get current model name"""
        return self.model

    def get_capabilities(self) -> Dict[str, Any]:
        """Get chatbot capabilities"""
        return {
            "streaming": True,
            "context_window": 200000,  # Claude 3 has 200k context
            "multimodal": True,  # Claude 3 supports images
            "tools": False,  # Not implemented yet
            "memory_usage": "cloud",  # No local memory usage
            "models_available": [
                "claude-3-haiku-20240307",
                "claude-3-sonnet-20240229",
                "claude-3-opus-20240229",
            ],
            "current_model": self.model,
        }

    async def generate_response_stream(self, user_input: str):
        """Alias for stream_process for compatibility"""
        async for chunk in self.stream_process(user_input):
            yield chunk

    async def add_knowledge(self, content: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Add knowledge to the system (not implemented for Claude)
        This could be implemented by storing in a vector DB for RAG
        """
        logger.warning("add_knowledge not implemented for Claude chatbot")
        return {
            "success": False,
            "message": "Knowledge base not implemented for Claude chatbot",
            "id": None,
        }
