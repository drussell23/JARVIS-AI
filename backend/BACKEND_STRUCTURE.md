# Backend Directory Structure

The backend has been organized into a clean, modular structure:

## 📁 Directory Overview

```
backend/
├── api/                    # API endpoints and routes
│   ├── voice_api.py       # Voice interaction endpoints
│   └── automation_api.py  # Task automation endpoints
│
├── chatbots/              # Chatbot implementations
│   ├── simple_chatbot.py  # Basic chatbot (fallback)
│   └── intelligent_chatbot.py # Enhanced with memory management
│
├── memory/                # Memory management system
│   ├── memory_manager.py  # Core M1-optimized memory manager
│   ├── memory_api.py      # Memory monitoring REST API
│   └── memory_safe_components.py # Safe component wrappers
│
├── engines/               # AI engines and processors
│   ├── nlp_engine.py     # NLP processing engine
│   ├── rag_engine.py     # RAG/Knowledge base engine
│   ├── voice_engine.py   # Voice processing engine
│   └── automation_engine.py # Task automation engine
│
├── models/                # ML model related files
│   ├── custom_model.py   # Custom model definitions
│   ├── training_pipeline.py # Training infrastructure
│   ├── fine_tuning.py    # Fine-tuning utilities
│   └── training_interface.py # Training API
│
├── utils/                 # Utility modules
│   ├── intelligent_cache.py # Memory-aware caching
│   ├── audio_processor.py # Audio processing utilities
│   ├── evaluation_metrics.py # Model evaluation
│   └── domain_knowledge.py # Domain-specific knowledge
│
├── static/                # Static files
│   └── demos/            # HTML demo interfaces
│
├── tests/                 # Test files
│   └── test_*.py         # Unit and integration tests
│
├── config/                # Configuration files
├── data/                  # Data storage
├── logs/                  # Application logs
├── checkpoints/           # Model checkpoints
└── main.py               # Main FastAPI application

```

## 🔑 Key Components

### Memory Management (`memory/`)
- **M1-optimized** memory management system
- Proactive component loading/unloading
- REST API for monitoring and control

### Chatbots (`chatbots/`)
- **SimpleChatbot**: Lightweight fallback implementation
- **IntelligentChatbot**: Enhanced with memory-safe AI components

### AI Engines (`engines/`)
- Modular AI components that can be loaded on-demand
- Memory-safe implementations with automatic cleanup

### APIs (`api/`)
- FastAPI routers for different functionality
- Clean separation of concerns

## 🚀 Import Examples

```python
# Main app imports
from chatbots.intelligent_chatbot import IntelligentChatbot
from memory.memory_manager import M1MemoryManager
from memory.memory_api import MemoryAPI

# API imports
from api.voice_api import VoiceAPI
from api.automation_api import AutomationAPI

# Engine imports
from engines.nlp_engine import NLPEngine
from engines.rag_engine import RAGEngine

# Utility imports
from utils.intelligent_cache import IntelligentCache
```

## 📝 Notes

- All directories contain `__init__.py` files for proper Python packaging
- The structure supports both local imports (within same package) and absolute imports
- Memory management is integrated throughout via the `memory_manager` instance
- Simple fallback mechanisms ensure the system works even under memory pressure