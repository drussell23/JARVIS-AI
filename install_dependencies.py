#!/usr/bin/env python3
"""
JARVIS Dependencies Installer
Installs all required dependencies for JARVIS to run properly
"""

import subprocess
import sys
import os


def install_package(package):
    """Install a package using pip"""
    print(f"📦 Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def main():
    print("🚀 JARVIS Dependencies Installer")
    print("=" * 50)
    
    # Core dependencies
    core_deps = [
        "psutil",  # System monitoring
        "aiohttp",  # Async HTTP
        "uvicorn",  # ASGI server
        "fastapi",  # Web framework
        "python-multipart",  # File uploads
        "websockets",  # WebSocket support
        "pydantic",  # Data validation
        "python-dotenv",  # Environment variables
    ]
    
    # AI/ML dependencies (optimized for M1)
    ml_deps = [
        "llama-cpp-python",  # Quantized models
        "langchain",  # LangChain framework
        "langchain-community",  # Community integrations
        "openai",  # OpenAI API (optional)
        "tiktoken",  # Tokenization
        "numpy",  # Numerical computing
    ]
    
    # Additional useful dependencies
    extra_deps = [
        "rich",  # Beautiful terminal output
        "requests",  # HTTP requests
        "aiofiles",  # Async file operations
        "PyYAML",  # YAML support
    ]
    
    print("\n📋 Installing core dependencies...")
    for dep in core_deps:
        try:
            install_package(dep)
            print(f"✅ {dep} installed successfully")
        except Exception as e:
            print(f"❌ Failed to install {dep}: {e}")
    
    print("\n🤖 Installing AI/ML dependencies (optimized for M1)...")
    for dep in ml_deps:
        try:
            install_package(dep)
            print(f"✅ {dep} installed successfully")
        except Exception as e:
            print(f"⚠️  Optional: {dep} - {e}")
    
    print("\n✨ Installing additional dependencies...")
    for dep in extra_deps:
        try:
            install_package(dep)
            print(f"✅ {dep} installed successfully")
        except Exception as e:
            print(f"⚠️  Optional: {dep} - {e}")
    
    print("\n" + "=" * 50)
    print("✅ Dependencies installation complete!")
    print("\n🎯 Next steps:")
    print("1. Run: python jarvis_quick_fix.py  # Set up optimized models")
    print("2. Run: python start_system.py      # Start JARVIS")
    print("=" * 50)


if __name__ == "__main__":
    main()