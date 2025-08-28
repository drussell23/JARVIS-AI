"""
Quantized LLM Wrapper for JARVIS - M1 Optimized
Provides a unified interface for using quantized models with existing JARVIS infrastructure
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class QuantizedLLMProvider:
    """Provider for quantized LLMs optimized for M1 Macs"""
    
    def __init__(self):
        self.llm = None
        self.model_path = None
        self.initialized = False
        
    def initialize(self, model_path: Optional[str] = None):
        """Initialize the quantized LLM"""
        try:
            from langchain_community.llms import LlamaCpp
            
            # Use provided path or default
            if not model_path:
                default_paths = [
                    Path.home() / ".jarvis/models/mistral-7b-instruct.gguf",
                    Path.home() / ".jarvis/models/llama2-7b.gguf",
                    Path("./models/mistral-7b.gguf")
                ]
                
                for path in default_paths:
                    if path.exists():
                        model_path = str(path)
                        break
                        
            if not model_path or not Path(model_path).exists():
                logger.error(f"No quantized model found. Run: python setup_m1_optimized_llm.py")
                return False
                
            self.model_path = model_path
            
            # Initialize with M1 optimization
            self.llm = LlamaCpp(
                model_path=model_path,
                n_gpu_layers=1,  # Use Metal GPU on M1 Macs for better performance (1 GPU layer) instead of 2
                n_ctx=2048, # Use 2048 context size for better performance on M1 Macs (2048 tokens) instead of 1024
                n_batch=512,
                temperature=0.7,
                max_tokens=512,
                n_threads=8,
                use_mlock=True,
                verbose=False,
                f16_kv=True,  # Use half precision for key/value cache
                logits_all=False,
                vocab_only=False,
                use_mmap=True,  # Memory-map the model
                streaming=False
            )
            
            self.initialized = True
            logger.info(f"Initialized quantized model: {Path(model_path).name}")
            return True
            
        except ImportError:
            logger.error("llama-cpp-python not installed. Run: pip install llama-cpp-python")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize quantized model: {e}")
            return False
            
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the quantized model"""
        if not self.initialized:
            if not self.initialize():
                return "Error: Quantized model not initialized. Run setup_m1_optimized_llm.py"
                
        try:
            # Extract relevant parameters
            max_tokens = kwargs.get('max_tokens', 256)
            temperature = kwargs.get('temperature', 0.7)
            
            # Update model parameters if needed
            if self.llm and hasattr(self.llm, 'max_tokens'):
                if max_tokens != self.llm.max_tokens:
                    self.llm.max_tokens = max_tokens
            if self.llm and hasattr(self.llm, 'temperature'):
                if temperature != self.llm.temperature:
                    self.llm.temperature = temperature
                
            # Generate response
            if self.llm:
                response = self.llm(prompt)
                return response
            else:
                return "Error: Model not initialized"
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error generating response: {str(e)}"
            
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        import psutil
        
        mem = psutil.virtual_memory()
        model_size_mb = 0
        
        if self.model_path and Path(self.model_path).exists():
            model_size_mb = Path(self.model_path).stat().st_size / (1024 * 1024)
            
        return {
            "model_loaded": self.initialized,
            "model_path": self.model_path,
            "model_size_mb": model_size_mb,
            "system_memory_percent": mem.percent,
            "available_memory_gb": mem.available / (1024**3)
        }

# Singleton instance
_quantized_provider = QuantizedLLMProvider()

def get_quantized_llm():
    """Get the singleton quantized LLM provider"""
    return _quantized_provider

def patch_transformers_import():
    """Patch transformers imports to use quantized models instead"""
    import sys
    from types import ModuleType
    
    class QuantizedModelPatcher(ModuleType):
        """Mock transformers module that redirects to quantized models"""
        
        def __init__(self):
            super().__init__('transformers')
            
        def __getattr__(self, name):
            if name == 'AutoModelForCausalLM':
                return self.AutoModelForCausalLM
            elif name == 'AutoTokenizer':
                return self.AutoTokenizer
            raise AttributeError(f"module 'transformers' has no attribute '{name}'")
        
        @staticmethod
        def AutoModelForCausalLM(*args, **kwargs):
            logger.info("Redirecting transformers model to quantized LLM")
            return get_quantized_llm()
            
        @staticmethod
        def AutoTokenizer(*args, **kwargs):
            # Return a dummy tokenizer since LlamaCpp handles tokenization
            class DummyTokenizer:
                def encode(self, text): return [1]  # Dummy
                def decode(self, ids): return ""    # Dummy
            return DummyTokenizer()
            
    # Only patch if enabled
    if os.getenv("USE_QUANTIZED_MODELS", "false").lower() == "true":
        sys.modules['transformers'] = QuantizedModelPatcher()
        logger.info("Patched transformers to use quantized models")

# Auto-patch on import if configured
if os.getenv("AUTO_PATCH_TRANSFORMERS", "false").lower() == "true":
    patch_transformers_import()