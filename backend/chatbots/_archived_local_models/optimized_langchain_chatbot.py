"""
Optimized LangChain Chatbot for M1 Macs
Uses quantized models for efficient memory usage
"""

import logging
import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Any

try:
    from langchain.llms import LlamaCpp
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationChain
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)

class OptimizedLangChainChatbot:
    """Memory-efficient LangChain chatbot using quantized models"""
    
    def __init__(self, memory_manager=None):
        self.memory_manager = memory_manager
        self.llm = None
        self.memory = None
        self.chain = None
        self.initialized = False
        
        # Auto-initialize if possible
        self.initialize()
        
    def initialize(self):
        """Initialize the quantized LLM and chain"""
        if not LANGCHAIN_AVAILABLE:
            logger.error("LangChain not available. Install with: pip install langchain langchain-community")
            return False
            
        try:
            # Find quantized model
            model_path = self._find_quantized_model()
            if not model_path:
                logger.error("No quantized model found. Run: python setup_m1_optimized_llm.py")
                return False
                
            # Initialize quantized LLM with M1 optimization
            callbacks = [StreamingStdOutCallbackHandler()] if os.getenv("STREAM_OUTPUT") == "1" else []
            
            # Load optimized config if available
            config_path = Path("jarvis_optimized_config.json")
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    model_config = config.get("model", {})
            else:
                model_config = {}
            
            self.llm = LlamaCpp(
                model_path=str(model_path),
                n_gpu_layers=1,  # Use Metal GPU
                n_ctx=model_config.get("n_ctx", 2048),      # Optimized for memory
                n_batch=model_config.get("n_batch", 256),   # Reduced batch size
                temperature=0.7,
                max_tokens=256,
                n_threads=model_config.get("n_threads", 6), # Leave CPU headroom
                use_mlock=True,
                callbacks=callbacks,
                verbose=False,
                f16_kv=True,  # Use FP16 for key-value cache
                streaming=bool(callbacks)
            )
            
            # Create conversation memory
            self.memory = ConversationBufferMemory(
                memory_key="history",
                return_messages=True,
                max_token_limit=1000  # Limit memory size
            )
            
            # Create conversation chain
            self.chain = ConversationChain(
                llm=self.llm,
                memory=self.memory,
                verbose=False
            )
            
            self.initialized = True
            logger.info(f"Initialized optimized LangChain with model: {model_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize optimized LangChain: {e}")
            return False
            
    def _find_quantized_model(self) -> Optional[Path]:
        """Find available quantized model"""
        # Check for model preference config
        config_path = Path(".jarvis_model_config.json")
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    preferred_path = Path(config.get("model_path", ""))
                    if preferred_path.exists():
                        return preferred_path
            except:
                pass
        
        # Default model search order (Phi-2 first for efficiency)
        model_paths = [
            Path("./models/phi-2.gguf"),                              # Phi-2 (2GB, efficient)
            Path.home() / ".jarvis/models/phi-2.gguf",
            Path.home() / ".jarvis/models/mistral-7b-instruct.gguf",  # Mistral (4GB, powerful)
            Path("./models/mistral-7b.gguf"),
            Path.home() / ".jarvis/models/llama2-7b.gguf",           # Llama2 (4GB)
            Path.home() / ".jarvis/models/tinyllama-1.1b.gguf",      # TinyLlama (1GB, ultra-light)
        ]
        
        # Check environment variable
        if env_path := os.getenv("QUANTIZED_MODEL_PATH"):
            model_paths.insert(0, Path(env_path))
            
        for path in model_paths:
            if path.exists():
                return path
                
        # Check for any GGUF file in models directory
        models_dir = Path.home() / ".jarvis/models"
        if models_dir.exists():
            for gguf_file in models_dir.glob("*.gguf"):
                return gguf_file
                
        return None
        
    async def chat(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Process a chat message"""
        if not self.initialized:
            if not self.initialize():
                return "Error: Failed to initialize LangChain. Please run: python setup_m1_optimized_llm.py"
                
        try:
            # Add context if provided
            if context:
                context_str = f"Context: {context}\n"
                full_message = f"{context_str}User: {message}"
            else:
                full_message = message
                
            # Generate response
            response = await asyncio.to_thread(self.chain.predict, input=full_message)
            return response
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"I encountered an error: {str(e)}"
            
    def clear_memory(self):
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()
            
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information"""
        import psutil
        
        mem = psutil.virtual_memory()
        model_info = "Not loaded"
        
        if self.llm and hasattr(self.llm, 'model_path'):
            model_path = Path(self.llm.model_path)
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                model_info = f"{model_path.name} ({size_mb:.0f}MB)"
                
        return {
            "mode": "langchain_optimized",
            "model": model_info,
            "memory_percent": mem.percent,
            "available_gb": mem.available / (1024**3),
            "initialized": self.initialized
        }
        
    async def optimize_for_device(self):
        """Optimize settings for current device"""
        import psutil
        
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        
        if self.llm:
            # Adjust parameters based on available memory
            if available_gb < 4:
                self.llm.max_tokens = 128
                self.llm.n_ctx = 512
            elif available_gb < 8:
                self.llm.max_tokens = 256
                self.llm.n_ctx = 1024
            else:
                self.llm.max_tokens = 512
                self.llm.n_ctx = 2048
                
            logger.info(f"Optimized for {available_gb:.1f}GB available memory")

# Create a compatibility wrapper
class LangChainChatbot(OptimizedLangChainChatbot):
    """Compatibility wrapper for existing code"""
    
    def __init__(self, memory_manager=None, **kwargs):
        # Redirect to optimized version
        super().__init__(memory_manager)
        
        # Handle additional initialization if needed
        if kwargs.get("use_tools"):
            logger.info("Tool usage requested but not implemented in optimized version")
            
        if kwargs.get("enable_web_search"):
            logger.info("Web search requested but not implemented in optimized version")