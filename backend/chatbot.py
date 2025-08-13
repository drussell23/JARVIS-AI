from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from typing import List, Dict, Optional, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass
from threading import Thread
import asyncio


@dataclass
class ConversationTurn:
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class Chatbot:
    SUPPORTED_MODELS = {
        "dialogpt-small": "microsoft/DialoGPT-small",
        "dialogpt-medium": "microsoft/DialoGPT-medium",
        "gpt2": "gpt2",
        "gpt2-medium": "gpt2-medium",
        "bloom-560m": "bigscience/bloom-560m",
        "bloom-1b1": "bigscience/bloom-1b1",
        "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",  # Requires authentication
        "distilgpt2": "distilgpt2",  # Lighter weight option
    }
    
    def __init__(self, model_name: str = "gpt2", max_history_length: int = 10):
        """
        Initialize the chatbot with a specified model and conversation history management.
        
        Args:
            model_name: Name of the model to use (from SUPPORTED_MODELS)
            max_history_length: Maximum number of conversation turns to maintain
        """
        # Select model path
        if model_name in self.SUPPORTED_MODELS:
            model_path = self.SUPPORTED_MODELS[model_name]
        else:
            model_path = model_name  # Allow custom model paths
            
        print(f"Loading model: {model_path}")
        
        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Set pad token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Conversation management
        self.max_history_length = max_history_length
        self.conversation_history: List[ConversationTurn] = []
        
        # System prompt and personality
        self.system_prompt = """You are a helpful, friendly AI assistant. You provide clear, accurate, and thoughtful responses while maintaining a conversational tone. You are knowledgeable but admit when you're unsure about something."""
        
        # Model-specific settings
        self.model_config = self._get_model_config(model_name)
        
    def _get_model_config(self, model_name: str) -> Dict:
        """Get model-specific generation parameters."""
        configs = {
            "dialogpt": {
                "max_length": 1000,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.95,
                "do_sample": True,
            },
            "gpt2": {
                "max_length": 512,
                "temperature": 0.8,
                "top_k": 50,
                "top_p": 0.92,
                "do_sample": True,
            },
            "bloom": {
                "max_length": 512,
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.9,
                "do_sample": True,
            },
            "llama2": {
                "max_length": 512,
                "temperature": 0.6,
                "top_k": 40,
                "top_p": 0.9,
                "do_sample": True,
            }
        }
        
        # Determine config based on model name
        if "dialogpt" in model_name.lower():
            return configs["dialogpt"]
        elif "gpt2" in model_name.lower():
            return configs["gpt2"]
        elif "bloom" in model_name.lower():
            return configs["bloom"]
        elif "llama" in model_name.lower():
            return configs["llama2"]
        else:
            return configs["gpt2"]  # Default
    
    def set_system_prompt(self, prompt: str):
        """Update the system prompt/personality."""
        self.system_prompt = prompt
        
    def add_to_history(self, role: str, content: str):
        """Add a conversation turn to history."""
        turn = ConversationTurn(role=role, content=content)
        self.conversation_history.append(turn)
        
        # Maintain history length limit
        if len(self.conversation_history) > self.max_history_length * 2:
            # Keep the most recent turns
            self.conversation_history = self.conversation_history[-self.max_history_length * 2:]
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history as a list of dictionaries."""
        return [
            {
                "role": turn.role,
                "content": turn.content,
                "timestamp": turn.timestamp.isoformat()
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
        Generate a response to user input with conversation context.
        
        Args:
            user_input: The user's message
            
        Returns:
            The model's response
        """
        # Add user input to history
        self.add_to_history("user", user_input)
        
        # Build the full prompt
        prompt = self._build_prompt(user_input)
        
        # Encode the prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=150,  # Generate up to 150 new tokens
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **self.model_config
            )
        
        # Decode only the generated part
        generated_tokens = outputs[0][inputs.shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Clean up the response
        response = response.strip()
        
        # Sometimes models continue generating after a natural end
        # Stop at the first complete sentence or newline
        for delimiter in ["\n", "User:", "System:"]:
            if delimiter in response:
                response = response.split(delimiter)[0].strip()
        
        # Add response to history
        self.add_to_history("assistant", response)
        
        return response
    
    def generate_response_with_context(self, user_input: str, context: Optional[Dict] = None) -> Dict:
        """
        Generate a response with additional context information.
        
        Args:
            user_input: The user's message
            context: Optional context dictionary with additional information
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = datetime.now()
        
        # Generate the response
        response = self.generate_response(user_input)
        
        # Calculate generation time
        generation_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "response": response,
            "generation_time": generation_time,
            "model_used": self.model.config.name_or_path,
            "conversation_id": id(self),
            "turn_number": len(self.conversation_history) // 2,
            "context": context
        }
    
    async def generate_response_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response to user input.
        
        Args:
            user_input: The user's message
            
        Yields:
            Tokens as they are generated
        """
        # Add user input to history
        self.add_to_history("user", user_input)
        
        # Build the full prompt
        prompt = self._build_prompt(user_input)
        
        # Encode the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        
        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        # Generation kwargs
        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=150,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **self.model_config
        )
        
        # Start generation in a separate thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Collect the full response for history
        full_response = []
        
        # Stream tokens
        for token in streamer:
            if token:
                full_response.append(token)
                # Clean up tokens that might contain delimiters
                for delimiter in ["User:", "System:", "\n\n"]:
                    if delimiter in token:
                        token = token.split(delimiter)[0]
                        if token:
                            yield token
                        # Stop generation if we hit a delimiter
                        thread.join()
                        break
                else:
                    yield token
        
        # Wait for generation to complete
        thread.join()
        
        # Add complete response to history
        complete_response = "".join(full_response).strip()
        self.add_to_history("assistant", complete_response)


# Optional example usage
if __name__ == "__main__":
    # Test with different models
    print("Testing Enhanced Chatbot with GPT-2...")
    bot = Chatbot(model_name="gpt2")
    
    # Set a custom personality
    bot.set_system_prompt("You are a knowledgeable but casual AI assistant who enjoys helping people learn new things.")
    
    # Test conversation
    test_inputs = [
        "Hello! How are you today?",
        "Can you tell me about Python programming?",
        "What did we just talk about?"
    ]
    
    for user_input in test_inputs:
        print(f"\nUser: {user_input}")
        response_data = bot.generate_response_with_context(user_input)
        print(f"Assistant: {response_data['response']}")
        print(f"(Generated in {response_data['generation_time']:.2f} seconds)")