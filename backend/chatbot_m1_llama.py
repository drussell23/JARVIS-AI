"""
M1-Optimized Chatbot using llama.cpp
This implementation uses llama.cpp server for native M1 performance without bus errors.
"""

import requests
import json
import logging
from typing import List, Dict, Optional, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass, field
import asyncio
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


class M1Chatbot:
    """
    M1-optimized chatbot using llama.cpp server.
    This avoids all PyTorch/MPS issues and provides native performance.
    """

    def __init__(
        self,
        llama_host: str = "localhost",
        llama_port: int = 8080,
        max_history_length: int = 10,
        model_name: str = "llama.cpp",
    ):
        """
        Initialize the M1-optimized chatbot.

        Args:
            llama_host: Host where llama.cpp server is running
            llama_port: Port where llama.cpp server is running
            max_history_length: Maximum conversation history to maintain
            model_name: Name for logging purposes
        """
        self.llama_url = f"http://{llama_host}:{llama_port}"
        self.max_history_length = max_history_length
        self.model_name = model_name
        self.conversation_history: List[ConversationTurn] = []
        
        # System prompt
        self.system_prompt = """You are a helpful, friendly AI assistant. You provide clear, accurate, and thoughtful responses while maintaining a conversational tone."""
        
        # Check if llama.cpp server is running
        self._check_server_status()
        
        # Initialize NLP components if available
        try:
            from nlp_engine import NLPEngine, ConversationFlow, TaskPlanner, ResponseQualityEnhancer
            self.nlp_engine = NLPEngine()
            self.conversation_flow = ConversationFlow()
            self.task_planner = TaskPlanner()
            self.response_enhancer = ResponseQualityEnhancer()
        except Exception as e:
            logger.warning(f"NLP components not available: {e}")
            self.nlp_engine = None
            self.conversation_flow = None
            self.task_planner = None
            self.response_enhancer = None

    def _check_server_status(self):
        """Check if llama.cpp server is running"""
        try:
            response = requests.get(f"{self.llama_url}/health", timeout=2)
            if response.status_code == 200:
                logger.info("✅ llama.cpp server is running")
        except:
            logger.warning(
                f"⚠️ llama.cpp server not found at {self.llama_url}\n"
                "Please start it with: llama-server -m <model_path> --host 0.0.0.0 --port 8080"
            )

    def add_to_history(self, role: str, content: str):
        """Add a conversation turn to history."""
        turn = ConversationTurn(role=role, content=content)
        self.conversation_history.append(turn)

        # Maintain history length limit
        if len(self.conversation_history) > self.max_history_length * 2:
            self.conversation_history = self.conversation_history[-self.max_history_length * 2:]

    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history as a list of dictionaries."""
        return [
            {
                "role": turn.role,
                "content": turn.content,
                "timestamp": turn.timestamp.isoformat(),
            }
            for turn in self.conversation_history
        ]

    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []

    def _build_prompt(self, user_input: str) -> str:
        """Build the full prompt including system prompt and conversation history."""
        prompt_parts = []

        # Add system prompt
        prompt_parts.append(f"System: {self.system_prompt}\n")

        # Add conversation history
        for turn in self.conversation_history[-self.max_history_length:]:
            if turn.role == "user":
                prompt_parts.append(f"User: {turn.content}")
            else:
                prompt_parts.append(f"Assistant: {turn.content}")

        # Add current user input
        prompt_parts.append(f"User: {user_input}")
        prompt_parts.append("Assistant:")

        return "\n".join(prompt_parts)

    def generate_response(self, user_input: str) -> str:
        """
        Generate a response using llama.cpp server.
        
        Args:
            user_input: The user's message
            
        Returns:
            The model's response
        """
        # Add user input to history
        self.add_to_history("user", user_input)
        
        # Build prompt
        prompt = self._build_prompt(user_input)
        
        try:
            # Make request to llama.cpp server
            response = requests.post(
                f"{self.llama_url}/completion",
                json={
                    "prompt": prompt,
                    "n_predict": 200,  # Max tokens to generate
                    "temperature": 0.7,
                    "top_k": 40,
                    "top_p": 0.95,
                    "repeat_penalty": 1.1,
                    "stop": ["User:", "\n\n", "System:"],
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("content", "").strip()
                
                # Clean up the response
                for delimiter in ["User:", "System:", "Assistant:"]:
                    if delimiter in generated_text:
                        generated_text = generated_text.split(delimiter)[0].strip()
                
                # Add to history
                self.add_to_history("assistant", generated_text)
                
                return generated_text
            else:
                logger.error(f"llama.cpp server error: {response.status_code}")
                return "I apologize, but I'm having trouble generating a response right now."
                
        except requests.exceptions.ConnectionError:
            return "I'm sorry, but the AI service is not available. Please ensure llama.cpp server is running."
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I encountered an error while generating a response. Please try again."

    def generate_response_with_context(
        self, user_input: str, context: Optional[Dict] = None
    ) -> Dict:
        """
        Generate a response with metadata (compatible with existing API).
        
        Args:
            user_input: The user's message
            context: Optional context dictionary
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = datetime.now()
        
        # Perform NLP analysis if available
        nlp_result = None
        if self.nlp_engine:
            try:
                nlp_result = self.nlp_engine.analyze(user_input)
            except Exception as e:
                logger.warning(f"NLP analysis failed: {e}")
        
        # Generate response
        response = self.generate_response(user_input)
        
        # Calculate generation time
        generation_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "response": response,
            "generation_time": generation_time,
            "model_used": self.model_name,
            "conversation_id": id(self),
            "turn_number": len(self.conversation_history) // 2,
            "context": context,
            "nlp_analysis": {
                "intent": nlp_result.intent.intent.value if nlp_result else "unknown",
                "intent_confidence": nlp_result.intent.confidence if nlp_result else 0.0,
                "entities": [
                    {"text": e.text, "type": e.type}
                    for e in (nlp_result.entities if nlp_result else [])
                ],
                "sentiment": (
                    nlp_result.sentiment
                    if nlp_result
                    else {"positive": 0, "negative": 0, "neutral": 1}
                ),
                "is_question": nlp_result.is_question if nlp_result else False,
                "requires_action": nlp_result.requires_action if nlp_result else False,
                "topic": nlp_result.topic if nlp_result else None,
                "keywords": nlp_result.keywords[:5] if nlp_result else [],
            },
            "conversation_flow": (
                self.conversation_flow.get_context_summary()
                if self.conversation_flow
                else {}
            ),
            "task_plan": None,
        }

    async def generate_response_stream(
        self, user_input: str
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response using llama.cpp server.
        
        Args:
            user_input: The user's message
            
        Yields:
            Tokens as they are generated
        """
        # Add user input to history
        self.add_to_history("user", user_input)
        
        # Build prompt
        prompt = self._build_prompt(user_input)
        
        full_response = []
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.llama_url}/completion",
                    json={
                        "prompt": prompt,
                        "n_predict": 200,
                        "temperature": 0.7,
                        "top_k": 40,
                        "top_p": 0.95,
                        "repeat_penalty": 1.1,
                        "stop": ["User:", "\n\n", "System:"],
                        "stream": True
                    }
                ) as response:
                    async for line in response.content:
                        if line:
                            try:
                                # Parse streaming response
                                line_str = line.decode('utf-8').strip()
                                if line_str.startswith("data: "):
                                    data = json.loads(line_str[6:])
                                    token = data.get("content", "")
                                    if token:
                                        full_response.append(token)
                                        
                                        # Check for stop sequences
                                        current_text = "".join(full_response)
                                        for delimiter in ["User:", "System:", "\n\n"]:
                                            if delimiter in current_text:
                                                # Stop generation
                                                final_text = current_text.split(delimiter)[0]
                                                self.add_to_history("assistant", final_text.strip())
                                                return
                                        
                                        yield token
                            except:
                                continue
        
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield "I encountered an error while generating a response."
        
        # Add complete response to history
        complete_response = "".join(full_response).strip()
        if complete_response:
            self.add_to_history("assistant", complete_response)

    def set_system_prompt(self, prompt: str):
        """Update the system prompt."""
        self.system_prompt = prompt

    # Compatibility methods for existing API
    def _get_generation_config(self) -> Dict:
        """Get default generation config (for compatibility)"""
        return {
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.95,
        }
    
    @property
    def model(self):
        """Compatibility property"""
        return type('obj', (object,), {
            'config': type('obj', (object,), {'name_or_path': self.model_name})
        })
    
    @property
    def tokenizer(self):
        """Compatibility property"""
        return type('obj', (object,), {
            'pad_token_id': 0,
            'eos_token_id': 2
        })


# Example usage and testing
if __name__ == "__main__":
    print("Testing M1-Optimized Chatbot with llama.cpp...")
    
    # Create chatbot
    bot = M1Chatbot()
    
    # Test conversation
    test_inputs = [
        "Hello! How are you today?",
        "Can you explain what makes Python a good programming language?",
        "What did we just talk about?",
    ]
    
    for user_input in test_inputs:
        print(f"\nUser: {user_input}")
        response_data = bot.generate_response_with_context(user_input)
        print(f"Assistant: {response_data['response']}")
        print(f"(Generated in {response_data['generation_time']:.2f} seconds)")