"""
Claude API-powered Chatbot for JARVIS
Leverages Anthropic's Claude API for high-quality responses with minimal local resource usage
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

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
    Cloud-based chatbot using Anthropic's Claude API
    Perfect for M1 Macs with limited RAM as all processing happens in the cloud
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-haiku-20240307",  # Fast and cost-effective
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize Claude chatbot
        
        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Claude model to use (haiku, sonnet, or opus)
            max_tokens: Maximum tokens in response
            temperature: Response randomness (0-1)
            system_prompt: System instructions for the assistant
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter"
            )
            
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Default JARVIS-style system prompt
        self.system_prompt = system_prompt or """You are JARVIS, an intelligent AI assistant inspired by Tony Stark's AI from Iron Man. 
You are helpful, witty, and highly capable. You speak with a refined, professional tone while being personable and occasionally adding subtle humor. 
You excel at understanding context and providing insightful, well-structured responses."""
        
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
        Process user input and generate response using Claude API
        
        Args:
            user_input: User's message
            
        Returns:
            AI response
        """
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
                messages=messages
            )
            
            # Extract response
            ai_response = response.content[0].text
            
            # Track usage
            self.api_calls_made += 1
            if hasattr(response, 'usage'):
                self.total_tokens_used += response.usage.total_tokens
                
            # Log performance
            response_time = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Claude API call completed in {response_time:.2f}s "
                f"(model: {self.model}, tokens: {getattr(response.usage, 'total_tokens', 'N/A')})"
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
        self.conversation_history.append({
            "user": user_input,
            "assistant": ai_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim history if needed
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
            
    async def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")
        
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        return {
            "api_calls": self.api_calls_made,
            "total_tokens": self.total_tokens_used,
            "model": self.model,
            "history_length": len(self.conversation_history)
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
                stream=True
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
            "context_used": context is not None
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
                "claude-3-opus-20240229"
            ],
            "current_model": self.model
        }
        
    async def generate_response_stream(
        self, user_input: str
    ):
        """Alias for stream_process for compatibility"""
        async for chunk in self.stream_process(user_input):
            yield chunk
            
    async def add_knowledge(
        self, content: str, metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Add knowledge to the system (not implemented for Claude)
        This could be implemented by storing in a vector DB for RAG
        """
        logger.warning("add_knowledge not implemented for Claude chatbot")
        return {
            "success": False,
            "message": "Knowledge base not implemented for Claude chatbot",
            "id": None
        }