#!/usr/bin/env python3
"""
Fix dependency installation issues for JARVIS
"""

import subprocess
import sys

print("🔧 JARVIS Dependency Fixer")
print("=" * 50)

# Core dependencies that work on M1
deps = [
    # Core
    "fastapi",
    "uvicorn[standard]",
    "pydantic",
    "python-multipart",
    "websockets",
    "aiohttp",
    "requests",
    "python-dotenv",
    
    # AI/ML (updated versions)
    "torch",  # Latest version
    "transformers",  # Latest version  
    "accelerate",
    "sentencepiece",
    "protobuf",
    
    # Voice
    "sounddevice",
    "soundfile", 
    "pydub",
    "gtts",
    "pyttsx3",
    "edge-tts",
    "numpy",
    "scipy",
    
    # Audio processing
    "noisereduce",
    "librosa",
    "numba",
    "audioread",
    "resampy",
    
    # Automation
    "icalendar",
    "pytz",
    "apscheduler",
    
    # RAG
    "sentence-transformers",
    "faiss-cpu",
    "chromadb",
    "tiktoken",
    "scikit-learn",
    "nltk",
    "aiofiles",
    
    # Memory management  
    "psutil",
    "objgraph",
    "pympler",
    
    # LangChain
    "langchain",
    "langchain-community",
    "duckduckgo-search",
    "numexpr",
    "wikipedia-api",
    
    # Spacy
    "spacy"
]

# Install each dependency
failed = []
for i, dep in enumerate(deps, 1):
    print(f"\n[{i}/{len(deps)}] Installing {dep}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", dep], 
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
        print(f"✅ {dep} installed")
    except:
        print(f"❌ {dep} failed")
        failed.append(dep)

# Install spacy model
print("\n📦 Installing spacy model...")
try:
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    print("✅ Spacy model installed")
except:
    print("❌ Spacy model failed")

# Skip problematic dependencies
skip_deps = [
    "openai-whisper",  # Often fails on M1
    "pyaudio",  # Requires portaudio
]

print(f"\n⚠️  Skipping optional dependencies: {', '.join(skip_deps)}")

# Summary
print("\n" + "=" * 50)
if failed:
    print(f"⚠️  Failed to install: {', '.join(failed)}")
    print("These are optional and JARVIS will still work")
else:
    print("✅ All dependencies installed successfully!")

print("\n🎯 Next step: python3 start_jarvis_fixed.py")
print("=" * 50)