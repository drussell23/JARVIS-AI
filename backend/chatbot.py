from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from typing import List, Dict, Optional, AsyncGenerator, Any
from datetime import datetime
from dataclasses import dataclass, field
from threading import Thread
import asyncio
import gc
import psutil
import os
import logging

# Disable tokenizer parallelism for M1
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from nlp_engine import (
    NLPEngine,
    ConversationFlow,
    TaskPlanner,
    ResponseQualityEnhancer,
    NLPAnalysis,
)
from automation_engine import AutomationEngine
from rag_engine import RAGEngine


@dataclass
class ConversationTurn:
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


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

    def __init__(
        self,
        model_name: str = "distilgpt2",  # Default to smaller model for M1
        max_history_length: int = 10,
        device: str = "auto",  # auto, cpu, or mps
        lazy_load: bool = True,  # Lazy load models for better startup
    ):
        """
        Initialize the chatbot with M1 optimizations.

        Args:
            model_name: Name of the model to use (from SUPPORTED_MODELS)
            max_history_length: Maximum number of conversation turns to maintain
            device: Device to use ('auto', 'cpu', or 'mps')
            lazy_load: Whether to load models lazily
        """
        # M1 optimization settings
        self.device = device
        self.lazy_load = lazy_load
        self._model_loaded = False

        # Memory monitoring
        self.process = psutil.Process(os.getpid())
        # Select model path
        if model_name in self.SUPPORTED_MODELS:
            model_path = self.SUPPORTED_MODELS[model_name]
        else:
            model_path = model_name  # Allow custom model paths

        self.model_name = model_name
        self.model_path = model_path

        # Initialize model and tokenizer as None (lazy loading)
        self.model = None
        self.tokenizer = None

        # Load immediately if not lazy loading
        if not lazy_load:
            self._load_model()

        # Conversation management
        self.max_history_length = max_history_length
        self.conversation_history: List[ConversationTurn] = []

        # System prompt and personality
        self.system_prompt = """You are a helpful, friendly AI assistant. You provide clear, accurate, and thoughtful responses while maintaining a conversational tone. You are knowledgeable but admit when you're unsure about something."""

        # Model-specific settings
        self.model_config = self._get_model_config(model_name)

        # Initialize NLP components (with error handling)
        try:
            self.nlp_engine = NLPEngine()
            self.conversation_flow = ConversationFlow()
            self.task_planner = TaskPlanner()
            self.response_enhancer = ResponseQualityEnhancer()
            self.automation_engine = AutomationEngine()
            self.rag_engine = RAGEngine(base_model_name=model_name)
        except Exception as e:
            logger.warning(f"Failed to initialize some components: {e}")
            # Continue without these components
            self.nlp_engine = None
            self.conversation_flow = None
            self.task_planner = None
            self.response_enhancer = None
            self.automation_engine = None
            self.rag_engine = None

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024

    def _load_model(self):
        """Load model with M1 optimizations"""
        if self._model_loaded:
            return

        logger.info(f"Loading model: {self.model_path}")
        logger.info(f"Initial memory: {self._get_memory_usage():.2f} MB")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Determine device
            if self.device == "auto":
                if torch.backends.mps.is_available():
                    self.device = "cpu"  # Use CPU for stability on M1
                else:
                    self.device = "cpu"

            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,  # Use float32 for stability
                low_cpu_mem_usage=True,  # Optimize memory usage
            )

            # Move to device
            self.model = self.model.to(self.device)
            logger.info(f"Using device: {self.device}")

            # Set to evaluation mode
            self.model.eval()

            self._model_loaded = True
            logger.info(f"Model loaded. Memory: {self._get_memory_usage():.2f} MB")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._model_loaded = False
            raise

    def _cleanup_memory(self):
        """Clean up memory after generation"""
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _get_model_config(self, model_name: str) -> Dict:
        """Get model-specific generation parameters."""
        configs = {
            "dialogpt": {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.95,
                "do_sample": True,
            },
            "gpt2": {
                "max_new_tokens": 50,  # Reduced for M1 compatibility
                "temperature": 0.8,
                "top_k": 50,
                "top_p": 0.92,
                "do_sample": True,
            },
            "bloom": {
                "max_new_tokens": 100,
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.9,
                "do_sample": True,
            },
            "llama2": {
                "max_new_tokens": 100,
                "temperature": 0.6,
                "top_k": 40,
                "top_p": 0.9,
                "do_sample": True,
            },
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
            self.conversation_history = self.conversation_history[
                -self.max_history_length * 2 :
            ]

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
        for turn in self.conversation_history[-self.max_history_length :]:
            if turn.role == "user":
                prompt_parts.append(f"User: {turn.content}")
            else:
                prompt_parts.append(f"Assistant: {turn.content}")

        # Add current user input
        prompt_parts.append(f"User: {user_input}")
        prompt_parts.append("Assistant:")

        return "\n".join(prompt_parts)

    def _is_automation_command(self, user_input: str, nlp_result: NLPAnalysis) -> bool:
        """Check if input is an automation command"""
        automation_keywords = [
            "schedule",
            "calendar",
            "meeting",
            "appointment",
            "weather",
            "forecast",
            "temperature",
            "news",
            "stock",
            "crypto",
            "lights",
            "turn on",
            "turn off",
            "home",
            "reminder",
            "alarm",
            "timer",
        ]

        user_input_lower = user_input.lower()

        # Check keywords
        if any(keyword in user_input_lower for keyword in automation_keywords):
            return True

        # Check intents
        automation_intents = ["request_action", "task_planning"]
        if nlp_result.intent.intent.value in automation_intents:
            # Check if it's about automation topics
            if nlp_result.topic in ["technology", "task", "business"]:
                return True

        return False

    async def _handle_automation(
        self, user_input: str, nlp_result: NLPAnalysis
    ) -> Dict:
        """Handle automation commands"""
        # Build context from NLP analysis
        context = {
            "entities": [{"text": e.text, "type": e.type} for e in nlp_result.entities],
            "keywords": nlp_result.keywords,
            "intent": nlp_result.intent.intent.value,
        }

        # Process with automation engine
        if self.automation_engine:
            result = await self.automation_engine.process_command(user_input, context)
        else:
            result = {"success": False, "message": "Automation engine not available"}

        return result

    def _should_use_rag(self, user_input: str, nlp_result: NLPAnalysis) -> bool:
        """Determine if RAG should be used for this query"""
        # Use RAG for knowledge-seeking queries
        knowledge_intents = ["question", "ask_information", "help"]
        if nlp_result.intent.intent.value in knowledge_intents:
            return True

        # Use RAG if the query contains knowledge-seeking keywords
        knowledge_keywords = [
            "what is",
            "how does",
            "explain",
            "tell me about",
            "why",
            "when",
            "where",
            "who",
            "define",
            "describe",
            "help me understand",
        ]

        user_input_lower = user_input.lower()
        if any(keyword in user_input_lower for keyword in knowledge_keywords):
            return True

        # Use RAG for technical topics
        if nlp_result.topic in ["technology", "science", "education"]:
            return True

        return False

    def _build_enhanced_prompt(
        self,
        user_input: str,
        nlp_result: NLPAnalysis,
        automation_response: Optional[Dict] = None,
        rag_result: Optional[Dict] = None,
    ) -> str:
        """Build an enhanced prompt with NLP insights"""
        prompt_parts = []

        # Add system prompt with intent-specific guidance
        enhanced_system = self.system_prompt
        if nlp_result.intent.intent.value == "question":
            enhanced_system += " When answering questions, be direct and informative."
        elif nlp_result.intent.intent.value == "task_planning":
            enhanced_system += " When helping with tasks, break them down into clear, actionable steps."
        elif nlp_result.intent.intent.value == "help":
            enhanced_system += (
                " Provide clear, step-by-step guidance when users need help."
            )

        prompt_parts.append(f"System: {enhanced_system}\n")

        # Add automation context if available
        if automation_response:
            automation_context = f"Automation Result: {automation_response.get('message', 'Action completed')}"
            if automation_response.get("data"):
                automation_context += (
                    f"\nData: {str(automation_response['data'])[:200]}"
                )
            prompt_parts.append(automation_context)

        # Add RAG context if available
        if rag_result:
            rag_context = (
                f"\nRelevant Knowledge:\n{rag_result.get('context_used', '')[:1000]}"
            )
            prompt_parts.append(rag_context)

            # Add personalization from learning engine
            if rag_result.get("adapted_parameters"):
                params = rag_result["adapted_parameters"]
                if params.get("formal"):
                    enhanced_system += " Use formal language."
                if params.get("detail_level") == "high":
                    enhanced_system += " Provide detailed explanations."
                elif params.get("detail_level") == "low":
                    enhanced_system += " Keep responses concise."

        # Add context about detected entities if relevant
        if nlp_result.entities:
            entity_context = "Context: User mentioned " + ", ".join(
                [f"{e.type}: {e.text}" for e in nlp_result.entities[:3]]
            )
            prompt_parts.append(entity_context)

        # Add conversation history
        for turn in self.conversation_history[-self.max_history_length :]:
            if turn.role == "user":
                prompt_parts.append(f"User: {turn.content}")
            else:
                prompt_parts.append(f"Assistant: {turn.content}")

        # Add current user input with intent hint
        prompt_parts.append(f"User: {user_input}")
        prompt_parts.append("Assistant:")

        return "\n".join(prompt_parts)

    def _adjust_generation_config(self, nlp_result: NLPAnalysis) -> Dict:
        """Adjust generation parameters based on NLP analysis"""
        config = self.model_config.copy()

        # Adjust based on intent
        if nlp_result.intent.intent.value == "question":
            # More focused responses for questions
            config["temperature"] = max(0.5, config.get("temperature", 0.7) - 0.2)
            config["max_new_tokens"] = 200
        elif nlp_result.intent.intent.value == "task_planning":
            # Longer, more structured responses for planning
            config["max_new_tokens"] = 300
            config["temperature"] = 0.6
        elif nlp_result.intent.intent.value == "greeting":
            # Shorter, warmer responses for greetings
            config["max_new_tokens"] = 50
            config["temperature"] = 0.8

        # Adjust based on sentiment
        if nlp_result.sentiment.get("negative", 0) > 0.7:
            # More empathetic responses for negative sentiment
            config["temperature"] = min(0.9, config.get("temperature", 0.7) + 0.1)

        return config

    def generate_response(self, user_input: str) -> str:
        """
        Generate a response to user input with conversation context and NLP enhancements.

        Args:
            user_input: The user's message

        Returns:
            The model's response
        """
        # Lazy load model if needed
        if not self._model_loaded:
            self._load_model()

        # Perform NLP analysis if available
        nlp_result = None
        if self.nlp_engine:
            try:
                nlp_result = self.nlp_engine.analyze(user_input)
            except Exception as e:
                logger.warning(f"Error in NLP analysis: {e}")
                nlp_result = None

        # Check for automation commands
        automation_response = None
        if (
            nlp_result
            and self.automation_engine
            and self._is_automation_command(user_input, nlp_result)
        ):
            automation_response = asyncio.run(
                self._handle_automation(user_input, nlp_result)
            )

        # Use RAG for knowledge retrieval
        rag_result = None
        if (
            nlp_result
            and self.rag_engine
            and self._should_use_rag(user_input, nlp_result)
        ):
            rag_result = asyncio.run(
                self.rag_engine.generate_with_retrieval(
                    user_input, self.get_conversation_history()
                )
            )

        # Add user input to history
        self.add_to_history("user", user_input)

        try:
            # Build enhanced prompt based on NLP analysis
            if nlp_result:
                prompt = self._build_enhanced_prompt(
                    user_input, nlp_result, automation_response, rag_result
                )
            else:
                prompt = self._build_prompt(user_input)

            # Encode the prompt with attention mask
            if not self.tokenizer:
                raise ValueError("Tokenizer not loaded")
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, padding=True
            )

            # Adjust generation parameters based on intent
            if nlp_result:
                generation_config = self._adjust_generation_config(nlp_result)
            else:
                generation_config = self._get_generation_config()

            # Avoid passing duplicate max_new_tokens by extracting it from the config first
            max_new_tokens = generation_config.pop("max_new_tokens", 150)

            # Generate response with M1 safety measures
            if not self.model or not self.tokenizer:
                raise ValueError("Model or tokenizer not loaded")

            with torch.no_grad():
                try:
                    # Move inputs to CPU explicitly
                    input_ids = inputs["input_ids"].cpu()
                    attention_mask = inputs["attention_mask"].cpu()

                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        **generation_config,
                    )
                except RuntimeError as e:
                    logger.error(f"Generation failed with error: {e}")
                    raise

            # Decode only the generated part
            generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        except Exception as e:
            print(f"Error in model generation: {e}")
            # Fallback response
            response = f"I understand you said: '{user_input}'. How can I help you?"

        # Clean up the response
        response = response.strip()

        # Sometimes models continue generating after a natural end
        # Stop at the first complete sentence or newline
        for delimiter in ["\n", "User:", "System:"]:
            if delimiter in response:
                response = response.split(delimiter)[0].strip()

        # Enhance response quality
        try:
            if self.conversation_flow and self.response_enhancer and nlp_result:
                context = self.conversation_flow.get_context_summary()
                response = self.response_enhancer.enhance_response(
                    response, nlp_result, context
                )

                # Update conversation flow
                if nlp_result:
                    self.conversation_flow.update_flow(nlp_result, response)
        except Exception as e:
            logger.warning(f"Error in response enhancement: {e}")
            # Continue with unenhanced response

        # Add response to history
        self.add_to_history("assistant", response)

        # Clean up memory after generation
        self._cleanup_memory()
        logger.debug(f"Memory after generation: {self._get_memory_usage():.2f} MB")

        return response

    def generate_response_with_context(
        self, user_input: str, context: Optional[Dict] = None
    ) -> Dict:
        """
        Generate a response with additional context information and NLP analysis.

        Args:
            user_input: The user's message
            context: Optional context dictionary with additional information

        Returns:
            Dictionary containing response, metadata, and NLP insights
        """
        start_time = datetime.now()

        # Perform NLP analysis
        if self.nlp_engine:
            nlp_result = self.nlp_engine.analyze(user_input)
        else:
            nlp_result = None

        # Generate the response
        response = self.generate_response(user_input)

        # Calculate generation time
        generation_time = (datetime.now() - start_time).total_seconds()

        # Create task plan if requested
        task_plan = None
        if (
            nlp_result
            and nlp_result.intent.intent.value == "task_planning"
            and self.task_planner
        ):
            task_plan = self.task_planner.create_task_plan(user_input, nlp_result)

        return {
            "response": response,
            "generation_time": generation_time,
            "model_used": (
                self.model.config.name_or_path
                if self.model and hasattr(self.model, "config")
                else self.model_name
            ),
            "conversation_id": id(self),
            "turn_number": len(self.conversation_history) // 2,
            "context": context,
            "nlp_analysis": {
                "intent": nlp_result.intent.intent.value if nlp_result else "unknown",
                "intent_confidence": (
                    nlp_result.intent.confidence if nlp_result else 0.0
                ),
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
            "task_plan": task_plan,
        }

    async def generate_response_stream(
        self, user_input: str
    ) -> AsyncGenerator[str, None]:
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
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        # Generation kwargs
        model_cfg = self.model_config.copy()
        max_new_tokens = model_cfg.pop("max_new_tokens", 150)
        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **model_cfg,
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
    bot.set_system_prompt(
        "You are a knowledgeable but casual AI assistant who enjoys helping people learn new things."
    )

    # Test conversation
    test_inputs = [
        "Hello! How are you today?",
        "Can you tell me about Python programming?",
        "What did we just talk about?",
    ]

    for user_input in test_inputs:
        print(f"\nUser: {user_input}")
        response_data = bot.generate_response_with_context(user_input)
        print(f"Assistant: {response_data['response']}")
        print(f"(Generated in {response_data['generation_time']:.2f} seconds)")
