# main.py
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from simple_chatbot import SimpleChatbot  # Import the SimpleChatbot class
import asyncio
import json
from typing import Optional, List, Dict, Any
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import optional components with error handling
try:
    from voice_api import VoiceAPI

    VOICE_API_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Voice API not available: {e}")
    VOICE_API_AVAILABLE = False

try:
    from automation_api import AutomationAPI

    AUTOMATION_API_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Automation API not available: {e}")
    AUTOMATION_API_AVAILABLE = False


# Define request models
class Message(BaseModel):
    user_input: str


class ChatConfig(BaseModel):
    model_name: Optional[str] = "distilgpt2"  # Default to smaller model
    system_prompt: Optional[str] = None
    stream: Optional[bool] = False
    device: Optional[str] = "auto"  # Device selection for M1


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
        # Create an instance of the Chatbot with M1 optimizations
        try:
            # Detect if running on M1 Mac
            import platform

            is_m1_mac = platform.system() == "Darwin" and platform.machine() == "arm64"

            if is_m1_mac:
                # Prefer llama.cpp on M1; allow forcing via env to avoid PyTorch/MPS crashes
                try:
                    import requests

                    force_llama = os.getenv("FORCE_LLAMA", "0") == "1"
                    # Use 127.0.0.1 and a slightly longer timeout to avoid race on startup
                    llama_health_url = "http://127.0.0.1:8080/health"
                    response = requests.get(llama_health_url, timeout=3)
                    if response.status_code == 200:
                        logger.info("Using Simple Chatbot for immediate responses")
                        from simple_chatbot import SimpleChatbot

                        self.bot = SimpleChatbot()
                    else:
                        raise RuntimeError("llama.cpp server not responding")
                except Exception as e:
                    if os.getenv("FORCE_LLAMA", "0") == "1":
                        # Do not fall back if explicitly forced; surface the error clearly
                        logger.error(
                            f"FORCE_LLAMA=1 but llama.cpp health check failed: {e}"
                        )
                        raise
                    logger.warning(
                        "llama.cpp server not available, using SimpleChatbot"
                    )
                    self.bot = SimpleChatbot()
            else:
                # Non-M1 systems use SimpleChatbot too
                self.bot = SimpleChatbot()

            logger.info("Chatbot initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize chatbot: {e}")
            raise
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

        if (
            config.model_name
            and config.model_name != self.bot.model.config.name_or_path
        ):
            # Currently only SimpleChatbot is supported
            updates.append("Model configuration noted (using simple-chat)")

        if config.system_prompt:
            self.bot.set_system_prompt(config.system_prompt)
            updates.append("System prompt updated")

        return {
            "message": "Configuration updated",
            "updates": updates,
            "current_model": self.bot.model.config.name_or_path,
        }

    async def analyze_text(self, message: Message):
        """Analyze text for NLP insights without generating a response."""
        if not self.bot.nlp_engine:
            return {"error": "NLP engine not available"}

        # Run NLP analysis in thread pool
        nlp_result = await asyncio.get_event_loop().run_in_executor(
            None, self.bot.nlp_engine.analyze, message.user_input
        )

        return {
            "intent": {
                "primary": nlp_result.intent.intent.value,
                "confidence": nlp_result.intent.confidence,
                "sub_intents": [
                    {"intent": i.value, "confidence": c}
                    for i, c in (nlp_result.intent.sub_intents or [])
                ],
            },
            "entities": [
                {"text": e.text, "type": e.type, "start": e.start, "end": e.end}
                for e in nlp_result.entities
            ],
            "sentiment": nlp_result.sentiment,
            "is_question": nlp_result.is_question,
            "requires_action": nlp_result.requires_action,
            "topic": nlp_result.topic,
            "keywords": nlp_result.keywords,
        }

    async def create_task_plan(self, message: Message):
        """Create a task plan based on user input."""
        if not self.bot.nlp_engine or not self.bot.task_planner:
            return {"error": "Task planning not available"}

        # Run analysis and planning in thread pool
        nlp_result = await asyncio.get_event_loop().run_in_executor(
            None, self.bot.nlp_engine.analyze, message.user_input
        )
        task_plan = await asyncio.get_event_loop().run_in_executor(
            None, self.bot.task_planner.create_task_plan, message.user_input, nlp_result
        )

        return {
            "task_plan": task_plan,
            "nlp_insights": {
                "intent": nlp_result.intent.intent.value,
                "entities": [
                    {"text": e.text, "type": e.type} for e in nlp_result.entities
                ],
                "keywords": nlp_result.keywords[:5],
            },
        }

    # RAG endpoints
    async def add_knowledge(self, request: KnowledgeRequest):
        """Add a document to the knowledge base."""
        try:
            document = await self.bot.rag_engine.add_knowledge(
                request.content, request.metadata
            )
            return {
                "message": "Knowledge added successfully",
                "document_id": document.id,
                "chunks": len(document.chunks),
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

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

            document = await self.bot.rag_engine.add_knowledge(text_content, metadata)

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
            results = await self.bot.rag_engine.knowledge_base.search(
                request.query, k=request.k, strategy=request.strategy
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
            await self.bot.rag_engine.provide_feedback(
                request.query, request.response, request.score
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
            insights = self.bot.rag_engine.get_learning_insights()
            return insights
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def summarize_conversation(self):
        """Summarize the current conversation."""
        try:
            # Get conversation history
            history = self.bot.get_conversation_history()

            if not history:
                return {"message": "No conversation to summarize"}

            # Convert to format expected by summarizer
            messages = [
                {"role": turn["role"], "content": turn["content"]} for turn in history
            ]

            summary = await self.bot.rag_engine.summarize_conversation(messages)

            return {
                "summary": summary.summary,
                "key_points": summary.key_points,
                "entities": summary.entities,
                "topics": summary.topics,
                "sentiment": summary.sentiment,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))


# Create FastAPI app
app = FastAPI()

# Enable CORS for all origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate and include our Chatbot API router
chatbot_api = ChatbotAPI()
app.include_router(chatbot_api.router)

# Include Voice API routes if available
if VOICE_API_AVAILABLE:
    try:
        voice_api = VoiceAPI(chatbot_api.bot)
        app.include_router(voice_api.router, prefix="/voice")
        logger.info("Voice API routes added")
    except Exception as e:
        logger.warning(f"Failed to initialize Voice API: {e}")

# Include Automation API routes if available
if AUTOMATION_API_AVAILABLE and hasattr(chatbot_api.bot, "automation_engine"):
    try:
        automation_api = AutomationAPI(chatbot_api.bot.automation_engine)
        app.include_router(automation_api.router, prefix="/automation")
        logger.info("Automation API routes added")
    except Exception as e:
        logger.warning(f"Failed to initialize Automation API: {e}")


# Update root endpoint
@app.get("/")
async def root():
    return {
        "message": "AI-Powered Chatbot with Voice, NLP, Automation & RAG",
        "version": "5.0",
        "features": {
            "chat": "Advanced conversational AI with NLP",
            "voice": "Speech recognition and synthesis",
            "nlp": "Intent recognition, entity extraction, sentiment analysis",
            "automation": "Calendar, weather, information services, task automation",
            "rag": "Retrieval-Augmented Generation with knowledge base",
            "learning": "Adaptive learning and personalization",
        },
        "api_docs": "/docs",
        "endpoints": {
            "chat": "/chat/*",
            "voice": "/voice/*",
            "automation": "/automation/*",
            "knowledge": "/knowledge/*",
        },
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check if model is loaded
        model_loaded = (
            chatbot_api.bot._model_loaded
            if hasattr(chatbot_api.bot, "_model_loaded")
            else False
        )
        memory_usage = (
            chatbot_api.bot._get_memory_usage()
            if hasattr(chatbot_api.bot, "_get_memory_usage")
            else 0
        )

        return {
            "status": "healthy",
            "model": chatbot_api.bot.model_name,
            "model_loaded": model_loaded,
            "memory_usage_mb": memory_usage,
            "device": getattr(chatbot_api.bot, "device", "unknown"),
            "components": {
                "voice_api": VOICE_API_AVAILABLE,
                "automation_api": AUTOMATION_API_AVAILABLE,
            },
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting M1-optimized chatbot server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
