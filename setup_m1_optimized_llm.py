#!/usr/bin/env python3
"""
M1-Optimized LLM Setup Script for JARVIS
Configures quantized models that work efficiently on M1 Macs with 16GB RAM
"""

import os
import sys
import subprocess
import requests
import psutil
from pathlib import Path


class M1OptimizedLLMSetup:
    """Setup quantized LLMs optimized for M1 Macs"""
    
    MODELS = {
        "mistral-7b": {
            "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            "size_gb": 4.1,
            "name": "Mistral 7B Instruct (Q4)",
            "recommended": True
        },
        "llama2-7b": {
            "url": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
            "size_gb": 3.8,
            "name": "Llama 2 7B Chat (Q4)"
        },
        "llama2-13b": {
            "url": "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_K_M.gguf",
            "size_gb": 7.4,
            "name": "Llama 2 13B Chat (Q4)"
        },
        "codellama-7b": {
            "url": "https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q4_K_M.gguf",
            "size_gb": 3.8,
            "name": "CodeLlama 7B Instruct (Q4)"
        }
    }
    
    def __init__(self):
        self.models_dir = Path.home() / ".jarvis" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def check_memory(self):
        """Check available memory"""
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        used_percent = mem.percent
        
        print(f"\n📊 Memory Status:")
        print(f"  Available: {available_gb:.1f} GB")
        print(f"  Used: {used_percent:.1f}%")
        
        if used_percent > 80:
            print("\n⚠️  High memory usage detected!")
            print("  Run: python optimize_memory_advanced.py")
            
        return available_gb
        
    def check_dependencies(self):
        """Check and install required dependencies"""
        print("\n🔍 Checking dependencies...")
        
        # Check for llama-cpp-python
        try:
            import llama_cpp
            print("✅ llama-cpp-python installed")
        except ImportError:
            print("📦 Installing llama-cpp-python with Metal support...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "--upgrade", "--force-reinstall", "--no-cache-dir",
                "llama-cpp-python"
            ], check=True)
            
        # Check for langchain
        try:
            import langchain
            print("✅ langchain installed")
        except ImportError:
            print("📦 Installing langchain...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "langchain", "langchain-community"
            ], check=True)
            
    def download_model(self, model_key):
        """Download a quantized model"""
        model_info = self.MODELS[model_key]
        model_path = self.models_dir / f"{model_key}.gguf"
        
        if model_path.exists():
            print(f"✅ {model_info['name']} already downloaded")
            return str(model_path)
            
        print(f"\n📥 Downloading {model_info['name']}...")
        print(f"   Size: {model_info['size_gb']} GB")
        
        # Download with progress
        response = requests.get(model_info['url'], stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                progress = (downloaded / total_size) * 100 if total_size > 0 else 0
                print(f"\r   Progress: {progress:.1f}%", end='', flush=True)
                
        print(f"\n✅ Downloaded to: {model_path}")
        return str(model_path)
        
    def test_model(self, model_path):
        """Test the model with Metal acceleration"""
        print("\n🧪 Testing model with M1 GPU acceleration...")
        
        try:
            from langchain.llms import LlamaCpp
            
            # Initialize with Metal
            llm = LlamaCpp(
                model_path=model_path,
                n_gpu_layers=1,  # Enable Metal
                n_ctx=2048,
                temperature=0.7,
                max_tokens=256
            )
            
            # Test inference
            response = llm("Hello! Please introduce yourself in one sentence.")
            print(f"✅ Model response: {response}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error testing model: {e}")
            return False
            
    def create_config(self, model_path):
        """Create JARVIS configuration for quantized model"""
        config = {
            "llm": {
                "provider": "llama-cpp",
                "model_path": model_path,
                "model_type": "gguf",
                "n_gpu_layers": 1,  # M1 GPU acceleration
                "n_ctx": 2048,
                "temperature": 0.7,
                "max_tokens": 512,
                "use_mlock": True,  # Keep model in RAM
                "n_threads": 8  # Optimize for M1
            },
            "memory": {
                "target_percent": 45,
                "optimize_on_start": True,
                "use_quantized": True
            }
        }
        
        config_path = Path.home() / ".jarvis" / "llm_config.json"
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        print(f"\n✅ Created config at: {config_path}")
        return config_path
        
    def setup_jarvis_integration(self):
        """Update JARVIS to use quantized models"""
        print("\n🔧 Configuring JARVIS for quantized models...")
        
        # Create environment file
        env_path = Path(".env.llm")
        env_content = """# M1-Optimized LLM Configuration
USE_QUANTIZED_MODELS=true
MODEL_FORMAT=gguf
ENABLE_METAL=true
N_GPU_LAYERS=1
MAX_MEMORY_MB=4096
"""
        
        with open(env_path, 'w') as f:
            f.write(env_content)
            
        print(f"✅ Created {env_path}")
        
    def run_quick_fix(self):
        """Run the complete quick fix"""
        print("\n🚀 JARVIS M1-Optimized LLM Quick Fix")
        print("=" * 50)
        
        # Check memory
        available_gb = self.check_memory()
        if available_gb < 5:
            print("\n⚠️  Less than 5GB available. Running memory optimization...")
            subprocess.run([sys.executable, "optimize_memory_advanced.py", "-a"])
            
        # Install dependencies
        self.check_dependencies()
        
        # Select model based on available memory
        if available_gb > 8:
            model_key = "llama2-13b"
        else:
            model_key = "mistral-7b"
            
        print(f"\n🎯 Selected model: {self.MODELS[model_key]['name']}")
        
        # Download model
        model_path = self.download_model(model_key)
        
        # Test model
        if self.test_model(model_path):
            # Create configuration
            self.create_config(model_path)
            self.setup_jarvis_integration()
            
            print("\n✅ SUCCESS! JARVIS is now configured with quantized models!")
            print("\n📋 Next steps:")
            print("  1. Restart JARVIS: python start_system.py")
            print("  2. Enable LangChain mode")
            print("  3. Enjoy fast, memory-efficient AI!")
            
            return True
        else:
            print("\n❌ Model test failed. Please check your setup.")
            return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='M1-Optimized LLM Setup for JARVIS')
    parser.add_argument('--model', choices=list(M1OptimizedLLMSetup.MODELS.keys()),
                        help='Specific model to download')
    parser.add_argument('--test-only', action='store_true',
                        help='Only test existing model')
    parser.add_argument('--list', action='store_true',
                        help='List available models')
    
    args = parser.parse_args()
    
    setup = M1OptimizedLLMSetup()
    
    if args.list:
        print("\n📚 Available Quantized Models:")
        for key, info in setup.MODELS.items():
            recommended = " (RECOMMENDED)" if info.get('recommended') else ""
            print(f"  {key}: {info['name']} - {info['size_gb']}GB{recommended}")
        return
        
    if args.test_only:
        model_path = setup.models_dir / "mistral-7b.gguf"
        if model_path.exists():
            setup.test_model(str(model_path))
        else:
            print("❌ No model found to test")
        return
        
    if args.model:
        model_path = setup.download_model(args.model)
        setup.test_model(model_path)
        setup.create_config(model_path)
    else:
        # Run complete quick fix
        setup.run_quick_fix()


if __name__ == "__main__":
    main()