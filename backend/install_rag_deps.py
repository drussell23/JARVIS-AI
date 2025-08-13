#!/usr/bin/env python3
"""
Installation helper for RAG dependencies
"""

import subprocess
import sys
import os

def install_rag_dependencies():
    """Install RAG-specific dependencies"""
    
    print("Installing RAG dependencies...")
    
    # Core RAG dependencies
    rag_deps = [
        "sentence-transformers>=2.2.2",
        "faiss-cpu>=1.7.4",  # Use faiss-gpu if you have CUDA
        "chromadb>=0.4.22",
        "tiktoken>=0.5.2",
        "scikit-learn>=1.3.2",
        "nltk>=3.8.1",
        "aiofiles>=23.2.1"
    ]
    
    # Install each dependency
    for dep in rag_deps:
        print(f"\nInstalling {dep}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✓ {dep} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {dep}: {e}")
            
    # Download NLTK data
    print("\nDownloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("✓ NLTK data downloaded successfully")
    except Exception as e:
        print(f"✗ Failed to download NLTK data: {e}")
        
    # Create necessary directories
    print("\nCreating necessary directories...")
    dirs = ["faiss_index", "chroma_db", "knowledge_base"]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✓ Created directory: {dir_name}")
        
    print("\n✅ RAG dependencies installation complete!")
    print("\nNote: If you have CUDA available, you can install faiss-gpu instead of faiss-cpu:")
    print("  pip install faiss-gpu")
    
    # Test imports
    print("\nTesting imports...")
    try:
        import sentence_transformers
        import faiss
        import chromadb
        import tiktoken
        print("✓ All imports successful!")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please check the installation and try again.")

if __name__ == "__main__":
    install_rag_dependencies()