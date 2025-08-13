#!/usr/bin/env python3
"""
Quick dependency installer for AI-Powered Chatbot
Handles common installation issues
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n📦 {description}...")
    try:
        subprocess.check_call(command, shell=True)
        print(f"✅ {description} - Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed!")
        print(f"   Error: {e}")
        return False

def main():
    print("🚀 AI-Powered Chatbot Dependency Installer")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ is required!")
        sys.exit(1)
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
    
    # Core dependencies that often cause issues
    core_deps = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install wheel setuptools", "Installing build tools"),
        ("pip install transformers==4.36.2", "Installing Transformers library"),
        ("pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu", "Installing PyTorch (CPU)"),
        ("pip install spacy==3.7.2", "Installing spaCy"),
        ("pip install fastapi==0.109.0 uvicorn[standard]==0.27.0", "Installing FastAPI"),
    ]
    
    # Install core dependencies
    print("\n📦 Installing core dependencies...")
    for command, description in core_deps:
        if not run_command(command, description):
            print("\n⚠️  Some core dependencies failed to install.")
            print("   Try running in a virtual environment:")
            print("   python3 -m venv venv")
            print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
            print("   python install_deps.py")
    
    # Try to install all requirements
    if os.path.exists("requirements.txt"):
        print("\n📦 Installing all requirements from requirements.txt...")
        run_command("pip install -r requirements.txt", "Installing all requirements")
    
    # Download spaCy model
    print("\n📦 Downloading spaCy language model...")
    run_command("python -m spacy download en_core_web_sm", "Downloading spaCy English model")
    
    # Test imports
    print("\n🧪 Testing imports...")
    test_imports = [
        "transformers",
        "torch",
        "fastapi",
        "spacy",
        "whisper",
        "pydub",
        "gtts",
        "edge_tts"
    ]
    
    failed_imports = []
    for module in test_imports:
        try:
            __import__(module)
            print(f"✅ {module} - OK")
        except ImportError:
            print(f"⚠️  {module} - Not installed")
            failed_imports.append(module)
    
    # Summary
    print("\n" + "=" * 50)
    if not failed_imports:
        print("✅ All dependencies installed successfully!")
        print("\n🚀 You can now run: python main.py")
    else:
        print("⚠️  Some imports failed:")
        for module in failed_imports:
            print(f"   - {module}")
        print("\n💡 Tips:")
        print("   - For audio (whisper, pydub): You may need FFmpeg installed")
        print("   - For voice (pyaudio): You may need PortAudio installed")
        print("   - Run './setup.sh' for complete system setup")

if __name__ == "__main__":
    main()