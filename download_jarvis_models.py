#!/usr/bin/env python3
"""
Download all required models for JARVIS Core
"""

import os
import subprocess
from pathlib import Path
import requests
import sys


class ModelDownloader:
    """Download models for JARVIS Core"""
    
    MODELS = {
        "tinyllama": {
            "filename": "tinyllama-1.1b.gguf",
            "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "size": "638 MB",
            "description": "TinyLlama 1.1B - Ultra lightweight for instant responses"
        },
        "phi2": {
            "filename": "phi-2.gguf",
            "url": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
            "size": "1.6 GB",
            "description": "Phi-2 - Efficient model for standard tasks"
        },
        "mistral": {
            "filename": "mistral-7b-instruct.gguf",
            "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            "size": "4.1 GB",
            "description": "Mistral 7B - Advanced model for complex tasks"
        }
    }
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
    def download_model(self, model_key):
        """Download a specific model"""
        if model_key not in self.MODELS:
            print(f"❌ Unknown model: {model_key}")
            return False
            
        model_info = self.MODELS[model_key]
        model_path = self.models_dir / model_info["filename"]
        
        if model_path.exists():
            print(f"✅ {model_info['filename']} already exists")
            return True
            
        print(f"\n📥 Downloading {model_info['description']}")
        print(f"   Size: {model_info['size']}")
        print(f"   URL: {model_info['url']}")
        
        try:
            # Use curl for better progress display
            cmd = [
                "curl", "-L", 
                "--progress-bar",
                "-o", str(model_path),
                model_info["url"]
            ]
            
            subprocess.run(cmd, check=True)
            print(f"✅ Downloaded {model_info['filename']}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to download: {e}")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
            
    def download_all(self):
        """Download all models"""
        print("📦 Downloading all JARVIS Core models")
        print("=" * 50)
        
        success_count = 0
        for key in self.MODELS:
            if self.download_model(key):
                success_count += 1
                
        print(f"\n✅ Downloaded {success_count}/{len(self.MODELS)} models")
        return success_count == len(self.MODELS)
        
    def check_models(self):
        """Check which models are present"""
        print("🔍 Checking models...")
        present = []
        missing = []
        
        for key, info in self.MODELS.items():
            path = self.models_dir / info["filename"]
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                present.append(f"✅ {info['filename']} ({size_mb:.0f}MB)")
            else:
                missing.append(f"❌ {info['filename']} - {info['description']}")
                
        print("\nPresent models:")
        for p in present:
            print(f"  {p}")
            
        if missing:
            print("\nMissing models:")
            for m in missing:
                print(f"  {m}")
                
        return len(missing) == 0


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download models for JARVIS Core")
    parser.add_argument("--check", action="store_true", help="Just check what models are present")
    parser.add_argument("--model", choices=["tinyllama", "phi2", "mistral"], 
                       help="Download specific model")
    parser.add_argument("--dir", default="models", help="Models directory (default: models)")
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(args.dir)
    
    if args.check:
        if downloader.check_models():
            print("\n✅ All models present!")
        else:
            print("\n⚠️  Some models missing. Run without --check to download.")
    elif args.model:
        if downloader.download_model(args.model):
            print("\n✅ Model downloaded successfully!")
        else:
            print("\n❌ Download failed!")
            sys.exit(1)
    else:
        # Download all
        if downloader.download_all():
            print("\n🎉 All models ready for JARVIS Core!")
        else:
            print("\n⚠️  Some downloads failed. Please retry.")
            sys.exit(1)


if __name__ == "__main__":
    main()