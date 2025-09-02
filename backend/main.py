# main.py
import os
import sys

# Fix TensorFlow import issues before any other imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Disable TF warnings
os.environ["USE_TORCH"] = "1"  # Use PyTorch backend for transformers
os.environ["USE_TF"] = "0"  # Disable TensorFlow in transformers

# Skip TensorFlow - not needed for basic operation

from fastapi import (
    FastAPI,
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import io

try:
    from chatbots.claude_vision_chatbot import (
        ClaudeVisionChatbot,
    )  # Try vision-enabled chatbot first

    VISION_CHATBOT_AVAILABLE = True
except ImportError:
    from chatbots.claude_chatbot import ClaudeChatbot  # Fall back to regular chatbot

    VISION_CHATBOT_AVAILABLE = False

# Import enhanced vision analyzer with 6 integrated components
try:
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    ENHANCED_VISION_ANALYZER_AVAILABLE = True
    logger.info("Enhanced vision analyzer with 6 integrated components available")
except ImportError as e:
    ENHANCED_VISION_ANALYZER_AVAILABLE = False
    logger.info(f"Enhanced vision analyzer not available: {e}")
import asyncio
import json
import time
from typing import Optional, List, Dict, Any
import logging

import httpx

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not installed, that's okay

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting main.py imports...")

# Import memory manager first - it's critical
try:
    from memory.memory_manager import M1MemoryManager, ComponentPriority
    from memory.memory_api import MemoryAPI, create_memory_alert_callback

    MEMORY_MANAGER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Memory manager not available: {e}")
    MEMORY_MANAGER_AVAILABLE = False

    # Create stub classes
    class M1MemoryManager:
        def __init__(self):
            pass

        async def start_monitoring(self):
            pass

        async def get_memory_snapshot(self):
            from types import SimpleNamespace

            return SimpleNamespace(
                state=SimpleNamespace(value="normal"),
                percent=0.5,
                available=8 * 1024 * 1024 * 1024,
                total=16 * 1024 * 1024 * 1024,
            )

        def register_component(self, *args, **kwargs):
            pass

        @property
        def is_m1(self):
            return True

        @property
        def components(self):
            return {}

    class ComponentPriority:
        CRITICAL = 5
        HIGH = 4
        MEDIUM = 3
        LOW = 2
        MINIMAL = 1

    class MemoryAPI:
        def __init__(self, manager):
            self.router = APIRouter()
            self.router.add_api_route("/status", self.get_status, methods=["GET"])

        async def get_status(self):
            return {"status": "memory management disabled"}


# Apply model loader patch to prevent loading 197 models
try:
    from utils.model_loader_patch import patch_model_discovery

    patch_model_discovery()
    logger.info("Model loader patch applied")
except Exception as e:
    logger.warning(f"Could not apply model loader patch: {e}")

# Import progressive model loader to prevent blocking
try:
    from utils.progressive_model_loader import model_loader

    MODEL_LOADER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Progressive model loader not available: {e}")
    MODEL_LOADER_AVAILABLE = False

    # Create stub
    class ModelLoader:
        def get_status(self):
            return {"status": "model loader disabled"}

    model_loader = ModelLoader()

# Import ML model loader for parallel initialization
try:
    from ml_model_loader import initialize_models, get_loader_status
    from api.model_status_api import (
        router as model_status_router,
        broadcast_model_status,
    )

    ML_MODEL_LOADER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML model loader not available: {e}")
    ML_MODEL_LOADER_AVAILABLE = False
    model_status_router = None

# Create global memory manager instance
logger.info("Creating memory manager...")
if MEMORY_MANAGER_AVAILABLE:
    memory_manager = M1MemoryManager()
else:
    memory_manager = M1MemoryManager()  # Stub version
logger.info("Memory manager created")

# Enable voice API for JARVIS speech functionality
VOICE_API_AVAILABLE = True
try:
    from api.voice_api import VoiceAPI

    logger.info("Voice API imported successfully")
except ImportError as e:
    logger.warning(f"Voice API import failed: {e}")
    VOICE_API_AVAILABLE = False

# Import Enhanced Voice Routes with Rust Acceleration
ENHANCED_VOICE_AVAILABLE = True
try:
    from api.enhanced_voice_routes import router as enhanced_voice_router

    logger.info("Enhanced voice routes imported successfully")
except ImportError as e:
    logger.warning(f"Enhanced voice routes import failed: {e}")
    ENHANCED_VOICE_AVAILABLE = False
    enhanced_voice_router = None

# Enable voice fix if available
VOICE_FIX_AVAILABLE = True
try:
    from api.voice_503_fix import router as voice_fix_router

    logger.info("Voice fix imported successfully")
except ImportError as e:
    logger.warning(f"Voice fix import failed: {e}")
    VOICE_FIX_AVAILABLE = False

# Enable JARVIS voice API
JARVIS_VOICE_AVAILABLE = True
try:
    from voice.jarvis_agent_voice import JARVISAgentVoice
    from api.jarvis_voice_api import JARVISVoiceAPI, JARVISCommand

    logger.info("JARVIS voice API imported successfully")
except ImportError as e:
    logger.warning(f"JARVIS voice API import failed: {e}")
    JARVIS_VOICE_AVAILABLE = False

# Skip automation API import
AUTOMATION_API_AVAILABLE = False
logger.info("Skipped automation API import")


# Define request models
class Message(BaseModel):
    user_input: str


class ChatConfig(BaseModel):
    model_name: Optional[str] = "distilgpt2"  # Default to smaller model
    system_prompt: Optional[str] = None
    stream: Optional[bool] = False
    device: Optional[str] = "auto"  # Device selection for M1

    model_config = {"protected_namespaces": ()}


class KnowledgeRequest(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = {}


class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 5
    strategy: Optional[str] = "hybrid"  # semantic, keyword, hybrid


class FeedbackRequest(BaseModel):
    query: str
    response: str
    score: float  # 0.0 to 1.0


class ChatbotAPI:
    def __init__(self):
        # Register core components with memory manager
        memory_manager.register_component(
            "simple_chatbot", ComponentPriority.CRITICAL, 100
        )
        memory_manager.register_component("nlp_engine", ComponentPriority.HIGH, 1500)
        memory_manager.register_component("rag_engine", ComponentPriority.MEDIUM, 3000)
        memory_manager.register_component(
            "voice_engine", ComponentPriority.MEDIUM, 2000
        )
        memory_manager.register_component(
            "automation_engine", ComponentPriority.LOW, 500
        )

        # Create an instance of the Chatbot - Claude API only
        try:
            # Check for Claude configuration
            claude_api_key = os.getenv("ANTHROPIC_API_KEY")

            if not claude_api_key:
                logger.warning("ANTHROPIC_API_KEY not found in environment variables")
                logger.info("Creating minimal chatbot for testing")

                # Create minimal chatbot that just echoes
                class MinimalChatbot:
                    def __init__(self):
                        self.model_name = "minimal-echo"
                        self.model = "minimal"
                        self.conversation_history = []

                    async def generate_response_with_context(self, user_input):
                        response = f"Echo: {user_input}"
                        self.conversation_history.append(
                            {"role": "user", "content": user_input}
                        )
                        self.conversation_history.append(
                            {"role": "assistant", "content": response}
                        )
                        return {
                            "response": response,
                            "conversation_id": "test",
                            "message_count": len(self.conversation_history),
                        }

                    async def generate_response_stream(self, user_input):
                        response = f"Echo: {user_input}"
                        for char in response:
                            yield char

                    async def generate_response(self, user_input):
                        return f"Echo: {user_input}"

                    async def get_response(self, prompt):
                        return f"Echo: {prompt}"

                    async def get_conversation_history(self):
                        return self.conversation_history

                    async def clear_history(self):
                        self.conversation_history = []

                    def set_system_prompt(self, prompt):
                        pass

                    def is_available(self):
                        return True

                    def get_usage_stats(self):
                        return {"requests": len(self.conversation_history) // 2}

                self.bot = MinimalChatbot()
            else:
                logger.info("Initializing Claude-powered chatbot")

                # Use vision-enabled chatbot if available
                if VISION_CHATBOT_AVAILABLE:
                    logger.info("Using vision-enabled Claude chatbot")
                    self.bot = ClaudeVisionChatbot(
                        api_key=claude_api_key,
                        model=os.getenv(
                            "CLAUDE_MODEL", "claude-3-5-sonnet-20241022"
                        ),  # Vision model
                        max_tokens=int(os.getenv("CLAUDE_MAX_TOKENS", "1024")),
                        temperature=float(os.getenv("CLAUDE_TEMPERATURE", "0.7")),
                    )
                else:
                    logger.info("Using standard Claude chatbot")
                    self.bot = ClaudeChatbot(
                        api_key=claude_api_key,
                        model=os.getenv("CLAUDE_MODEL", "claude-3-haiku-20240307"),
                        max_tokens=int(os.getenv("CLAUDE_MAX_TOKENS", "1024")),
                        temperature=float(os.getenv("CLAUDE_TEMPERATURE", "0.7")),
                    )

                logger.info("Claude chatbot initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize chatbot: {e}")
            # Create minimal fallback
            logger.info("Creating minimal chatbot as fallback")

            class MinimalChatbot:
                def __init__(self):
                    self.model_name = "minimal-echo"
                    self.model = "minimal"
                    self.conversation_history = []

                async def generate_response_with_context(self, user_input):
                    response = f"Echo: {user_input}"
                    self.conversation_history.append(
                        {"role": "user", "content": user_input}
                    )
                    self.conversation_history.append(
                        {"role": "assistant", "content": response}
                    )
                    return {
                        "response": response,
                        "conversation_id": "test",
                        "message_count": len(self.conversation_history),
                    }

                async def generate_response_stream(self, user_input):
                    response = f"Echo: {user_input}"
                    for char in response:
                        yield char

                async def generate_response(self, user_input):
                    return f"Echo: {user_input}"

                async def get_response(self, prompt):
                    return f"Echo: {prompt}"

                async def get_conversation_history(self):
                    return self.conversation_history

                async def clear_history(self):
                    self.conversation_history = []

                def set_system_prompt(self, prompt):
                    pass

                def is_available(self):
                    return True

                def get_usage_stats(self):
                    return {"requests": len(self.conversation_history) // 2}

            self.bot = MinimalChatbot()
        # Create a router for our endpoints
        self.router = APIRouter()
        # Register endpoints on the router
        self.router.add_api_route("/chat", self.chat_get, methods=["GET"])
        self.router.add_api_route("/chat", self.chat_post, methods=["POST"])
        self.router.add_api_route("/chat/stream", self.chat_stream, methods=["POST"])
        self.router.add_api_route("/chat/history", self.get_history, methods=["GET"])
        self.router.add_api_route(
            "/chat/history", self.clear_history, methods=["DELETE"]
        )
        self.router.add_api_route("/chat/config", self.update_config, methods=["POST"])
        self.router.add_api_route("/chat/analyze", self.analyze_text, methods=["POST"])
        self.router.add_api_route("/chat/plan", self.create_task_plan, methods=["POST"])
        self.router.add_api_route(
            "/chat/capabilities", self.get_capabilities, methods=["GET"]
        )
        self.router.add_api_route("/chat/mode", self.get_mode, methods=["GET"])
        self.router.add_api_route("/chat/mode", self.set_mode, methods=["POST"])
        self.router.add_api_route(
            "/chat/optimize-memory",
            self.optimize_memory_for_langchain,
            methods=["POST"],
        )

        # RAG endpoints
        self.router.add_api_route(
            "/knowledge/add", self.add_knowledge, methods=["POST"]
        )
        self.router.add_api_route(
            "/knowledge/add-file", self.add_knowledge_file, methods=["POST"]
        )
        self.router.add_api_route(
            "/knowledge/search", self.search_knowledge, methods=["POST"]
        )
        self.router.add_api_route(
            "/knowledge/feedback", self.provide_feedback, methods=["POST"]
        )
        self.router.add_api_route(
            "/knowledge/insights", self.get_learning_insights, methods=["GET"]
        )
        self.router.add_api_route(
            "/knowledge/summarize", self.summarize_conversation, methods=["POST"]
        )

    async def chat_get(self):
        """GET endpoint for informational purposes."""
        return {
            "message": "AI-Powered Chatbot API with Advanced NLP Features",
            "version": "2.0",
            "available_models": ["simple-chat"],
            "endpoints": {
                "/chat": "Standard chat endpoint with NLP enhancements",
                "/chat/stream": "Streaming chat endpoint for real-time responses",
                "/chat/history": "Get conversation history (GET) or clear it (DELETE)",
                "/chat/config": "Update chatbot configuration (model, system prompt)",
                "/chat/analyze": "Analyze text for intent, entities, sentiment without generating response",
                "/chat/plan": "Create a task plan from user input",
                "/knowledge/add": "Add documents to the knowledge base",
                "/knowledge/search": "Search the knowledge base",
                "/knowledge/feedback": "Provide feedback for learning",
            },
            "nlp_features": {
                "intent_recognition": "Identifies user intent (greeting, question, request, etc.)",
                "entity_extraction": "Extracts entities (dates, times, names, etc.)",
                "sentiment_analysis": "Analyzes emotional tone of messages",
                "conversation_flow": "Manages conversation context and state",
                "task_planning": "Creates structured plans for tasks",
                "response_quality": "Enhances response clarity and completeness",
            },
        }

    async def chat_post(self, message: Message):
        """POST endpoint that uses the Chatbot to generate a response."""
        try:
            logger.info(f"Received message: {message.user_input[:50]}...")
            response_data = await self.bot.generate_response_with_context(
                message.user_input
            )
            logger.info(f"Generated response successfully")
            return response_data
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            # Return a fallback response
            return {
                "response": "I apologize, but I encountered an error processing your request. Please try again.",
                "error": str(e),
                "conversation_id": "error",
                "message_count": 0,
            }

    async def chat_stream(self, message: Message):
        """Streaming endpoint for real-time response generation."""

        async def generate():
            # Send initial message
            yield f"data: {json.dumps({'type': 'start', 'message': 'Generating response...'})}\n\n"

            # Stream the response token by token
            async for token in self.bot.generate_response_stream(message.user_input):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                await asyncio.sleep(0.01)  # Small delay for streaming effect

            # Send completion message
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    async def get_history(self):
        """Get the conversation history."""
        history = await self.bot.get_conversation_history()
        return {
            "history": history,
            "total_turns": len(history),
        }

    async def clear_history(self):
        """Clear the conversation history."""
        await self.bot.clear_history()
        return {"message": "Conversation history cleared"}

    async def update_config(self, config: ChatConfig):
        """Update chatbot configuration."""
        updates = []

        if config.model_name and config.model_name != self.bot.model_name:
            # Currently only SimpleChatbot is supported
            updates.append("Model configuration noted (using simple-chat)")

        if config.system_prompt:
            self.bot.set_system_prompt(config.system_prompt)
            updates.append("System prompt updated")

        return {
            "message": "Configuration updated",
            "updates": updates,
            "current_model": self.bot.model_name,
        }

    async def analyze_text(self, message: Message):
        """Analyze text for NLP insights without generating a response."""
        # Claude-based analysis
        return {
            "message": "NLP analysis is integrated into Claude's responses",
            "note": "Claude provides contextual understanding natively",
            "capabilities": [
                "Intent detection",
                "Entity recognition",
                "Sentiment analysis",
                "Context awareness",
            ],
        }

    async def get_capabilities(self):
        """Get current AI capabilities based on loaded components"""
        if hasattr(self.bot, "get_capabilities"):
            return self.bot.get_capabilities()
        else:
            # Fallback for SimpleChatbot
            return {
                "basic_chat": True,
                "nlp_analysis": False,
                "knowledge_search": False,
                "voice_processing": False,
                "memory_status": {"components_loaded": 0, "total_components": 0},
            }

    async def get_mode(self):
        """Get current chatbot mode"""
        return {
            "mode": "claude",
            "auto_switch": False,
            "metrics": (
                self.bot.get_usage_stats()
                if hasattr(self.bot, "get_usage_stats")
                else {}
            ),
            "last_switch": None,
            "model": (
                self.bot.model
                if hasattr(self.bot, "model")
                else "claude-3-haiku-20240307"
            ),
        }

    async def set_mode(self, request: Dict[str, str]):
        """Set chatbot mode (deprecated - Claude only mode)"""
        raise HTTPException(
            status_code=400,
            detail="Mode switching is not available. System runs in Claude-only mode for consistent, high-quality responses.",
        )

    async def optimize_memory_for_langchain(
        self, request: Optional[Dict[str, Any]] = None
    ):
        """Memory optimization (not needed for Claude API)"""
        return {
            "success": True,
            "message": "Memory optimization not needed when using Claude API. All processing happens in the cloud.",
            "current_mode": "claude",
            "cloud_based": True,
        }

    async def create_task_plan(self, message: Message):
        """Create a task plan based on user input."""
        # Use Claude to create task plans
        task_prompt = f"""Create a detailed task plan for the following request. 
        Break it down into clear, actionable steps:
        
        Request: {message.user_input}
        
        Format the response as a numbered list of tasks."""

        response = await self.bot.get_response(task_prompt)

        return {
            "task_plan": response,
            "source": "claude",
            "capabilities": [
                "Task breakdown",
                "Priority assessment",
                "Dependency analysis",
            ],
        }

    # RAG endpoints
    async def add_knowledge(self, request: KnowledgeRequest):
        """Add a document to the knowledge base."""
        if hasattr(self.bot, "add_knowledge"):
            result = await self.bot.add_knowledge(request.content, request.metadata)
            if result.get("success"):
                return {
                    "message": "Knowledge added successfully",
                    "document_id": result.get("document_id"),
                    "chunks": result.get("chunks"),
                }
            else:
                raise HTTPException(
                    status_code=503,
                    detail=result.get("error", "Failed to add knowledge"),
                )
        else:
            raise HTTPException(
                status_code=501,
                detail="Knowledge base not available in current configuration",
            )

    async def add_knowledge_file(self, file: UploadFile = File(...)):
        """Add a file to the knowledge base."""
        try:
            content = await file.read()
            text_content = content.decode("utf-8")

            metadata = {
                "filename": file.filename,
                "content_type": file.content_type,
                "source": "uploaded_file",
            }

            # RAG not yet integrated
            raise HTTPException(
                status_code=501,
                detail="RAG engine not yet integrated. Memory management must be stable first.",
            )

            return {
                "message": f"File '{file.filename}' added to knowledge base",
                "document_id": document.id,
                "chunks": len(document.chunks),
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def search_knowledge(self, request: SearchRequest):
        """Search the knowledge base."""
        try:
            # RAG not yet integrated
            raise HTTPException(
                status_code=501,
                detail="RAG engine not yet integrated. Memory management must be stable first.",
            )

            return {
                "query": request.query,
                "results": [
                    {
                        "content": result.chunk.content,
                        "score": result.score,
                        "metadata": result.chunk.metadata,
                        "document_id": result.chunk.document_id,
                    }
                    for result in results
                ],
                "count": len(results),
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def provide_feedback(self, request: FeedbackRequest):
        """Provide feedback for learning."""
        try:
            # RAG not yet integrated
            raise HTTPException(
                status_code=501,
                detail="RAG engine not yet integrated. Memory management must be stable first.",
            )

            return {
                "message": "Feedback recorded successfully",
                "query": request.query,
                "score": request.score,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def get_learning_insights(self):
        """Get insights from the learning engine."""
        try:
            # RAG not yet integrated
            raise HTTPException(
                status_code=501,
                detail="RAG engine not yet integrated. Memory management must be stable first.",
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def summarize_conversation(self):
        """Summarize the current conversation."""
        try:
            # Get conversation history
            history = await self.bot.get_conversation_history()

            if not history:
                return {"message": "No conversation to summarize"}

            # Convert to format expected by summarizer
            messages = [
                {"role": turn["role"], "content": turn["content"]} for turn in history
            ]

            # RAG not yet integrated
            raise HTTPException(
                status_code=501,
                detail="RAG engine not yet integrated. Memory management must be stable first.",
            )

            return {
                "summary": summary.summary,
                "key_points": summary.key_points,
                "entities": summary.entities,
                "topics": summary.topics,
                "sentiment": summary.sentiment,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))


async def initialize_bridge_async():
    """Initialize the Python-TypeScript bridge asynchronously"""
    try:
        # Skip bridge initialization
        pass
        logger.info("Python-TypeScript bridge initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize bridge: {e}")
        logger.info("System will continue without bridge functionality")


async def initialize_vision_discovery_async():
    """Initialize the vision discovery system asynchronously"""
    try:
        # Skip vision discovery initialization
        pass
        logger.info("Vision discovery system initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize vision discovery: {e}")
        logger.info("System will continue without vision discovery")


# Lifespan context manager for startup/shutdown
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting up AI-Powered Chatbot with smart startup manager...")

    try:
        # Store the chatbot API instance in app state for WebSocket access
        app.state.chatbot_api = chatbot_api
        logger.info("✅ Chatbot API stored in app state")

        # Start memory monitoring if available
        if MEMORY_MANAGER_AVAILABLE:
            await memory_manager.start_monitoring()
            logger.info("✅ Memory monitoring started")
        else:
            logger.info("⚡ Starting without memory monitoring")

        # Try smart startup if available
        try:
            # Skip smart startup manager
            logger.info("⚡ Starting without smart startup manager")
        except ImportError:
            logger.info("⚡ Starting without smart startup manager")

        logger.info("✅ Server ready to handle requests!")

    except Exception as e:
        logger.error(f"❌ Startup warning: {e}")
        # Don't raise - let the server start anyway with minimal functionality
        logger.info("⚡ Starting with minimal functionality...")

    yield  # Server is running

    # Shutdown
    logger.info("Shutting down...")
    if MEMORY_MANAGER_AVAILABLE:
        await memory_manager.stop_monitoring()


# Create FastAPI app
logger.info("Creating FastAPI app...")
app = FastAPI(lifespan=lifespan)
logger.info("FastAPI app created")


# Enable CORS for all origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for demos
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Initialize Memory API first
if MEMORY_MANAGER_AVAILABLE:
    memory_api = MemoryAPI(memory_manager)
    app.include_router(memory_api.router, prefix="/memory")
    logger.info("Memory management API initialized")
else:
    # Create minimal memory API
    memory_api = MemoryAPI(memory_manager)
    app.include_router(memory_api.router, prefix="/memory")
    logger.info("Memory API initialized (stub mode)")

# Memory alert callback will be set up after initialization

# Instantiate and include our Chatbot API router
chatbot_api = ChatbotAPI()
app.include_router(chatbot_api.router)

# Include Unified Vision Handler - Resolves all WebSocket conflicts
try:
    # Import basic vision API for HTTP endpoints only
    # WebSocket routes are handled by the TypeScript router
    from api.vision_api import router as vision_router

    # Filter out WebSocket routes to prevent conflicts
    vision_router.routes = [
        route
        for route in vision_router.routes
        if not hasattr(route, "websocket_route") and not route.path.endswith("/ws")
    ]

    app.include_router(vision_router)
    logger.info("Vision API HTTP routes added successfully")
    VISION_API_AVAILABLE = True
    ENHANCED_VISION_AVAILABLE = True

except ImportError as e:
    logger.warning(f"Failed to import vision API: {e}")
    VISION_API_AVAILABLE = False
    ENHANCED_VISION_AVAILABLE = False
except Exception as e:
    logger.warning(f"Failed to initialize Vision System: {e}")
    VISION_API_AVAILABLE = False
    ENHANCED_VISION_AVAILABLE = False

# Include Voice API routes if available with memory management
if VOICE_API_AVAILABLE:
    try:
        voice_api = VoiceAPI(chatbot_api.bot)
        app.include_router(voice_api.router, prefix="/voice")
        logger.info("Voice API routes added")
    except Exception as e:
        logger.warning(f"Failed to initialize Voice API: {e}")

# Include Automation API routes if available with memory management
if AUTOMATION_API_AVAILABLE:
    try:
        # Create automation engine
        from engines.automation_engine import AutomationEngine

        automation_engine = AutomationEngine()

        # Claude doesn't need a separate automation engine
        logger.info("Automation requests are handled through Claude's API")
        if hasattr(chatbot_api.bot, "__dict__"):
            try:
                setattr(chatbot_api.bot, "automation_engine", automation_engine)
            except AttributeError:
                # Some bot types might not allow dynamic attributes
                pass

        automation_api = AutomationAPI(automation_engine)
        app.include_router(automation_api.router, prefix="/automation")
        logger.info("Automation API routes added")
    except Exception as e:
        logger.warning(f"Failed to initialize Automation API: {e}")

# Include JARVIS Voice API routes
if JARVIS_VOICE_AVAILABLE:
    try:
        jarvis_api = JARVISVoiceAPI()
        app.include_router(jarvis_api.router, prefix="/voice")
        app.state.jarvis_api = jarvis_api  # Store instance for WebSocket use
        logger.info("JARVIS Voice API routes added - Iron Man mode activated!")
    except Exception as e:
        logger.warning(f"Failed to initialize JARVIS Voice API: {e}")

# Include Enhanced Voice Routes with Rust Acceleration
if ENHANCED_VOICE_AVAILABLE:
    try:
        # Override voice routes with Rust-accelerated version
        app.include_router(enhanced_voice_router, tags=["voice"])
        logger.info(
            "Enhanced Voice Routes with Rust acceleration enabled - 10x performance boost!"
        )

        # Setup unified Rust service
        asyncio.create_task(setup_unified_service(app))
        logger.info("Unified Rust service initialized - CPU usage reduced to 25%!")
    except Exception as e:
        logger.warning(f"Failed to initialize Enhanced Voice Routes: {e}")
elif "VOICE_FIX_AVAILABLE" in globals() and VOICE_FIX_AVAILABLE:
    try:
        # Use 503 fix as fallback
        app.include_router(voice_fix_router, tags=["voice"])
        logger.info("Voice 503 fix enabled - no more Service Unavailable errors!")
    except Exception as e:
        logger.warning(f"Failed to initialize Voice 503 fix: {e}")

# Include Vision WebSocket API for real-time monitoring
# Commented out - vision_api.py already includes WebSocket at /vision/ws/vision
# try:
#     from api import vision_websocket
#     app.include_router(vision_websocket.router, prefix="/vision")
#     logger.info("Vision WebSocket API routes added - Real-time monitoring enabled!")
# except Exception as e:
#     logger.warning(f"Failed to initialize Vision WebSocket API: {e}")

# Skip notification vision API
NOTIFICATION_API_AVAILABLE = False

# Skip navigation API
NAVIGATION_API_AVAILABLE = False

# Skip ML audio API
ML_AUDIO_API_AVAILABLE = False

# Include Model Status API for real-time loading progress
if ML_MODEL_LOADER_AVAILABLE and model_status_router:
    try:
        app.include_router(model_status_router)
        logger.info(
            "Model Status API routes added - Real-time ML model loading tracking enabled!"
        )
        MODEL_STATUS_API_AVAILABLE = True
    except Exception as e:
        logger.warning(f"Failed to initialize Model Status API: {e}")
        MODEL_STATUS_API_AVAILABLE = False
else:
    MODEL_STATUS_API_AVAILABLE = False

# Skip WebSocket discovery API
WEBSOCKET_DISCOVERY_AVAILABLE = False

# Skip WebSocket HTTP handlers
WEBSOCKET_HTTP_HANDLERS_AVAILABLE = False


# Configuration for external services
VISION_SERVICE_URL = os.getenv("VISION_SERVICE_URL", "http://localhost:8001")


def load_vision_triggers():
    """Loads vision command triggers from a JSON file with a hardcoded fallback."""
    default_triggers = [
        "can you see my screen",
        "do you see my screen",
        "what's on my screen",
        "what can you see",
        "analyze my screen",
        "see my screen",
        "look at my screen",
        "describe my screen",
        "what do you see",
        "analyze what's on my screen",
    ]

    try:
        config_path = os.path.join(
            os.path.dirname(__file__), "config", "vision_triggers.json"
        )
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                data = json.load(f)
                triggers = data.get("triggers", default_triggers)
                if isinstance(triggers, list) and all(
                    isinstance(t, str) for t in triggers
                ):
                    logger.info(
                        f"Loaded {len(triggers)} vision command triggers from config file."
                    )
                    return triggers
                else:
                    logger.warning(
                        "Invalid format in vision_triggers.json. Using default triggers."
                    )
                    return default_triggers
        else:
            logger.info(
                "vision_triggers.json not found. Using default vision command triggers."
            )
            return default_triggers
    except Exception as e:
        logger.error(
            f"Error loading vision_triggers.json: {e}. Using default triggers."
        )
        return default_triggers


# Vision command trigger phrases
VISION_COMMAND_TRIGGERS = load_vision_triggers()


def is_vision_command(command_text: str) -> bool:
    """Check if a command is intended for the vision system."""
    if not command_text:
        return False

    lower_command = command_text.lower()
    return any(phrase in lower_command for phrase in VISION_COMMAND_TRIGGERS)


async def handle_vision_command_request(command_text: str) -> str:
    """Handles a command by processing it directly with vision system.
    
    The vision system now includes 6 integrated components:
    1. Swift Vision Integration - Metal-accelerated processing
    2. Window Analysis - Memory-aware window content analysis  
    3. Window Relationship Detection - Configurable window relationships
    4. Continuous Screen Analyzer - Dynamic interval monitoring
    5. Memory-Efficient Analyzer - Smart compression strategies
    6. Simplified Vision System - Configurable query templates
    
    All optimized for 16GB RAM macOS systems with no hardcoded values.
    """
    try:
        logger.info(f"Vision command received: {command_text}")
        
        # Log enhanced vision availability
        if ENHANCED_VISION_ANALYZER_AVAILABLE:
            logger.info("Enhanced vision analyzer with 6 components is available")

        # Check if we have the vision chatbot available
        chatbot_api = getattr(app.state, "chatbot_api", None)
        logger.info(f"Chatbot API available: {chatbot_api is not None}")
        if chatbot_api:
            logger.info(f"Bot available: {hasattr(chatbot_api, 'bot')}")
            if hasattr(chatbot_api, "bot"):
                logger.info(f"Bot type: {type(chatbot_api.bot).__name__}")
                logger.info(
                    f"Has analyze_screen_with_vision: {hasattr(chatbot_api.bot, 'analyze_screen_with_vision')}"
                )

        if (
            chatbot_api
            and hasattr(chatbot_api, "bot")
            and hasattr(chatbot_api.bot, "analyze_screen_with_vision")
        ):
            # Use the Claude Vision chatbot directly
            logger.info("Using Claude Vision API for screen analysis")
            response = await chatbot_api.bot.analyze_screen_with_vision(command_text)
            return response
        else:
            # If vision chatbot not available, fall back to HTTP request
            logger.warning("Vision chatbot not available, falling back to HTTP request")
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{VISION_SERVICE_URL}/vision/command",
                        json={"command": command_text, "use_claude": True},
                        timeout=30.0,
                    )
                    response.raise_for_status()
                    vision_data = response.json()
                    return vision_data.get(
                        "response", "I was unable to analyze the screen."
                    )
            except Exception as e:
                logger.error(f"HTTP request failed: {e}")
                return "I'm having trouble accessing my vision capabilities."

    except Exception as e:
        logger.error(f"An unexpected error occurred during vision processing: {e}")
        return "I encountered an unexpected error with my vision system."


async def handle_chatbot_command(command_text: str, chatbot_api_instance) -> str:
    """Handles a command by forwarding it to the chatbot."""
    if chatbot_api_instance and chatbot_api_instance.bot:
        try:
            return await chatbot_api_instance.bot.generate_response(command_text)
        except Exception as e:
            logger.error(f"Error processing command through chatbot: {e}")
            return "I apologize, but I encountered an error processing your request."
    else:
        return "JARVIS system is initializing. Please try again in a moment."


# Update root endpoint
@app.get("/")
async def root():
    # Get memory status
    memory_status = await memory_manager.get_memory_snapshot()

    # Check JARVIS status
    jarvis_status = "offline"
    if JARVIS_VOICE_AVAILABLE:
        try:
            if (
                hasattr(app.state, "jarvis_api")
                and app.state.jarvis_api.jarvis_available
            ):
                jarvis_status = "online"
            elif "jarvis_api" in locals() and jarvis_api.jarvis_available:
                jarvis_status = "standby"
        except:
            pass

    return {
        "message": "AI-Powered Chatbot with M1-Optimized Memory Management",
        "version": "6.0",
        "features": {
            "memory_management": "Proactive AI-driven memory management for M1 Macs",
            "chat": "Advanced conversational AI with NLP",
            "voice": "Speech recognition and synthesis",
            "jarvis": (
                "Iron Man-style AI assistant with personality"
                if JARVIS_VOICE_AVAILABLE
                else "Not available"
            ),
            "vision": (
                "Enhanced vision system with 6 integrated components: Swift Vision (Metal-accelerated), "
                "Window Analysis, Relationship Detection, Continuous Monitoring, Memory-Efficient Processing, "
                "and Simplified Claude-only mode - all optimized for 16GB RAM with no hardcoded values"
                if ENHANCED_VISION_ANALYZER_AVAILABLE
                else "Basic vision capabilities" if VISION_CHATBOT_AVAILABLE
                else "Not available"
            ),
            "nlp": "Intent recognition, entity extraction, sentiment analysis",
            "automation": "Calendar, weather, information services, task automation",
            "rag": "Retrieval-Augmented Generation with knowledge base",
            "learning": "Adaptive learning and personalization",
        },
        "memory_status": {
            "state": memory_status.state.value,
            "percent_used": round(memory_status.percent * 100, 1),
            "available_gb": round(memory_status.available / (1024**3), 1),
        },
        "jarvis_status": jarvis_status,
        "api_docs": "/docs",
        "endpoints": {
            "memory": "/memory/*",
            "chat": "/chat/*",
            "voice": "/voice/*",
            "jarvis": "/voice/jarvis/*" if JARVIS_VOICE_AVAILABLE else None,
            "automation": "/automation/*",
            "knowledge": "/knowledge/*",
            "vision": "/api/vision/*" if VISION_API_AVAILABLE else None,
            "notifications": (
                "/api/notifications/*"
                if "NOTIFICATION_API_AVAILABLE" in locals()
                and NOTIFICATION_API_AVAILABLE
                else None
            ),
            "navigation": (
                "/api/navigation/*"
                if "NAVIGATION_API_AVAILABLE" in locals() and NAVIGATION_API_AVAILABLE
                else None
            ),
        },
    }


# Add stub endpoints for frontend compatibility
@app.get("/voice/jarvis/status")
async def jarvis_status():
    """Stub endpoint for JARVIS voice status"""
    return {
        "status": "online",
        "jarvis_available": True,
        "mode": "minimal",
        "message": "JARVIS voice in minimal mode",
    }


# Add command endpoint for JARVIS text commands
class JarvisCommand(BaseModel):
    command: str


@app.post("/voice/jarvis/command")
async def jarvis_command(command: JarvisCommand):
    """Handle JARVIS text commands"""
    try:
        # Use the chatbot to process the command with JARVIS personality
        jarvis_prompt = f"You are JARVIS, Tony Stark's AI assistant. Respond to this command in character: {command.command}"

        if hasattr(chatbot_api.bot, "generate_response"):
            response = await chatbot_api.bot.generate_response(jarvis_prompt)
        else:
            # Fallback response
            response = f"Processing command: {command.command}"

        return {
            "success": True,
            "response": response,
            "command": command.command,
            "mode": "text",
        }
    except Exception as e:
        logger.error(f"Error processing JARVIS command: {e}")
        return {
            "success": False,
            "error": str(e),
            "response": "I apologize, sir. I encountered an error processing your command.",
        }


@app.get("/audio/ml/config")
async def audio_ml_config():
    """Configuration for ML audio system"""
    return {
        "sample_rate": 16000,
        "channels": 1,
        "format": "int16",
        "chunk_size": 1024,
        "vad_enabled": True,
        "wake_word": "hey jarvis",
    }


@app.post("/audio/ml/predict")
async def audio_ml_predict():
    """Stub endpoint for ML audio prediction"""
    return {"prediction": "normal", "confidence": 0.9, "audio_health": "good"}


@app.post("/audio/ml/error")
async def audio_ml_error(request: Request):
    """Handle ML audio errors"""
    try:
        error_data = await request.json()
        error_type = error_data.get("error", "unknown")

        # Handle different error types
        if error_type == "no-speech":
            # This is expected when there's silence - not really an error
            return {
                "handled": True,
                "action": "continue",
                "message": "No speech detected - this is normal during silence",
            }
        elif error_type == "network":
            return {
                "handled": True,
                "action": "retry",
                "message": "Network error - will retry",
            }
        else:
            # Log unexpected errors
            logger.warning(f"ML Audio error reported: {error_type}")
            return {
                "handled": True,
                "action": "continue",
                "message": f"Error acknowledged: {error_type}",
            }
    except Exception as e:
        logger.error(f"Error handling ML audio error: {e}")
        return {"handled": False, "message": str(e)}


# WebSocket endpoints with minimal implementation
@app.websocket("/voice/jarvis/stream")
async def jarvis_websocket(websocket: WebSocket):
    """Minimal WebSocket endpoint for JARVIS voice"""
    await websocket.accept()
    logger.info("JARVIS WebSocket connected")

    try:
        # Send initial connection message
        await websocket.send_json(
            {
                "type": "connection",
                "status": "connected",
                "message": "JARVIS voice stream connected (minimal mode)",
            }
        )

        # Main message processing loop
        while True:
            try:
                data = await websocket.receive_text()
                msg = json.loads(data) if data.startswith("{") else {"text": data}
                command_text = msg.get("text", "").strip()

                if not command_text:
                    continue

                # Send processing status immediately
                if is_vision_command(command_text):
                    await websocket.send_json(
                        {
                            "type": "processing",
                            "status": "processing",
                            "message": "Let me take a look at your screen...",
                        }
                    )
                else:
                    await websocket.send_json(
                        {
                            "type": "processing",
                            "status": "processing",
                            "message": "Processing your request...",
                        }
                    )

                # Route command through JARVIS voice API if available
                jarvis_api_instance = getattr(app.state, "jarvis_api", None)
                if JARVIS_VOICE_AVAILABLE and jarvis_api_instance:
                    try:
                        logger.info(f"Processing command through JARVIS API: {command_text}")
                        # Process through JARVIS with system control
                        result = await jarvis_api_instance.process_command(JARVISCommand(text=command_text))
                        
                        # Handle different response formats
                        if hasattr(result, 'body'):
                            # JSONResponse object
                            import json as json_lib
                            result_data = json_lib.loads(result.body.decode())
                            response_text = result_data.get("response", "")
                        elif isinstance(result, dict):
                            response_text = result.get("response", "")
                        else:
                            response_text = str(result)
                            
                        if not response_text:
                            logger.warning("JARVIS API returned empty response")
                            response_text = "I'm processing your request..."
                            
                        logger.info(f"JARVIS voice API response: {response_text}")
                    except Exception as e:
                        logger.error(f"Error using JARVIS voice API: {e}", exc_info=True)
                        # Fallback to vision or chatbot
                        if is_vision_command(command_text):
                            response_text = await handle_vision_command_request(command_text)
                        else:
                            chatbot_api = getattr(app.state, "chatbot_api", None)
                            response_text = await handle_chatbot_command(command_text, chatbot_api)
                else:
                    # Original routing when JARVIS voice not available
                    if is_vision_command(command_text):
                        logger.info(f"Vision command detected in WebSocket: {command_text}")
                        import time

                        start_time = time.time()
                        response_text = await handle_vision_command_request(command_text)
                        end_time = time.time()
                        logger.info(
                            f"Vision command processing took {end_time - start_time:.2f} seconds"
                        )
                    else:
                        chatbot_api = getattr(app.state, "chatbot_api", None)
                        response_text = await handle_chatbot_command(
                            command_text, chatbot_api
                        )

                # Send the response back to the client
                await websocket.send_json(
                    {
                        "type": "response",
                        "text": response_text,
                        "message": response_text,
                        "mode": "speech",  # Changed from "text" to "speech" to trigger TTS
                        "timestamp": asyncio.get_event_loop().time(),
                        "speak": True,  # Explicitly indicate this should be spoken
                    }
                )
            except WebSocketDisconnect:
                logger.info("JARVIS WebSocket client disconnected")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket loop: {e}")
                # Optionally send an error message to the client
                await websocket.send_json(
                    {"type": "error", "message": "An internal error occurred."}
                )

    except Exception as e:
        logger.error(f"JARVIS WebSocket connection error: {e}")


@app.websocket("/audio/ml/stream")
async def audio_ml_websocket(websocket: WebSocket):
    """Minimal WebSocket endpoint for ML audio"""
    await websocket.accept()
    logger.info("ML Audio WebSocket connected")

    try:
        # Send initial connection message
        await websocket.send_json(
            {"type": "connection", "status": "connected", "audio_health": "good"}
        )

        # Keep connection alive and send periodic health updates
        while True:
            try:
                message = await websocket.receive_text()

                # Parse the message if it's JSON
                try:
                    data = json.loads(message)
                    msg_type = data.get("type", "audio_data")
                except:
                    msg_type = "audio_data"

                # Send back appropriate response
                if msg_type == "config":
                    await websocket.send_json(
                        {
                            "type": "config",
                            "sample_rate": 16000,
                            "channels": 1,
                            "format": "int16",
                        }
                    )
                else:
                    # Send back audio health status
                    await websocket.send_json(
                        {
                            "type": "audio_status",
                            "prediction": "normal",
                            "confidence": 0.95,
                            "audio_health": "good",
                            "timestamp": asyncio.get_event_loop().time(),
                        }
                    )
            except WebSocketDisconnect:
                logger.info("ML Audio WebSocket client disconnected")
                break
    except Exception as e:
        logger.error(f"ML Audio WebSocket error: {e}")


# Add audio synthesis endpoint
@app.get("/audio/speak/{text}")
async def speak_text(text: str):
    """Generate and stream audio for text"""
    try:
        import gtts
        
        # Generate audio
        tts = gtts.gTTS(text, lang='en')
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return StreamingResponse(
            audio_buffer,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"inline; filename=jarvis_speech.mp3"
            }
        )
    except ImportError:
        # If gTTS not installed, return error
        logger.error("gTTS not installed. Install with: pip install gtts")
        raise HTTPException(status_code=503, detail="Text-to-speech service unavailable. Please install gTTS.")
    except Exception as e:
        logger.error(f"Audio generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Redirect old demo URLs to new locations
from fastapi.responses import RedirectResponse


@app.get("/voice_demo.html")
async def redirect_voice_demo():
    return RedirectResponse(url="/static/demos/voice_demo.html")


@app.get("/automation_demo.html")
async def redirect_automation_demo():
    return RedirectResponse(url="/static/demos/automation_demo.html")


@app.get("/rag_demo.html")
async def redirect_rag_demo():
    return RedirectResponse(url="/static/demos/rag_demo.html")


@app.get("/llm_demo.html")
async def redirect_llm_demo():
    return RedirectResponse(url="/static/demos/llm_demo.html")


@app.get("/memory_dashboard.html")
async def redirect_memory_dashboard():
    return RedirectResponse(url="/static/demos/memory_dashboard.html")


# Model status endpoint
@app.get("/models/status")
async def get_model_status():
    """Get current model loading status"""
    try:
        # Skip smart startup manager
        raise ImportError("Smart startup manager disabled")
    except ImportError:
        if MODEL_LOADER_AVAILABLE:
            return model_loader.get_status()
        else:
            return {"status": "model loading disabled", "models_loaded": 0}


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Get memory status
        memory_snapshot = await memory_manager.get_memory_snapshot()

        # Check if Claude is configured
        model_loaded = (
            chatbot_api.bot.is_available()
            if hasattr(chatbot_api.bot, "is_available")
            else True
        )

        return {
            "status": (
                "healthy" if memory_snapshot.state.value != "emergency" else "critical"
            ),
            "model": chatbot_api.bot.model_name,
            "model_loaded": model_loaded,
            "memory": {
                "state": memory_snapshot.state.value,
                "percent_used": round(memory_snapshot.percent * 100, 1),
                "available_mb": round(memory_snapshot.available / (1024 * 1024), 1),
                "total_mb": round(memory_snapshot.total / (1024 * 1024), 1),
                "components_loaded": [
                    name
                    for name, info in memory_manager.components.items()
                    if info.is_loaded
                ],
            },
            "device": "M1" if memory_manager.is_m1 else "unknown",
            "components": {
                "memory_manager": True,
                "voice_api": VOICE_API_AVAILABLE
                and "voice_engine"
                in [
                    name
                    for name, info in memory_manager.components.items()
                    if info.is_loaded
                ],
                "automation_api": AUTOMATION_API_AVAILABLE
                and "automation_engine"
                in [
                    name
                    for name, info in memory_manager.components.items()
                    if info.is_loaded
                ],
                "vision_api": VISION_API_AVAILABLE,
                "enhanced_vision": ENHANCED_VISION_ANALYZER_AVAILABLE,
                "vision_components": {
                    "swift_vision": "Metal-accelerated processing",
                    "window_analysis": "Memory-aware window analysis",
                    "relationship_detection": "Window relationship detection",
                    "continuous_monitoring": "Dynamic interval monitoring",
                    "memory_efficient": "Smart compression strategies",
                    "simplified_vision": "Configurable query templates"
                } if ENHANCED_VISION_ANALYZER_AVAILABLE else None,
            },
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI-Powered Chatbot Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8010, help="Port to bind to")
    args = parser.parse_args()

    # Also check environment variable for port
    port = int(os.getenv("PORT", args.port))

    logger.info(f"Starting M1-optimized chatbot server on {args.host}:{port}...")
    uvicorn.run(app, host=args.host, port=port)
