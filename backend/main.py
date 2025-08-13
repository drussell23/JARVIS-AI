# main.py
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from chatbot import Chatbot  # Import the Chatbot class
import asyncio
import json
from typing import Optional
from voice_api import VoiceAPI

# Define request models
class Message(BaseModel):
    user_input: str
    
class ChatConfig(BaseModel):
    model_name: Optional[str] = "gpt2"
    system_prompt: Optional[str] = None
    stream: Optional[bool] = False

class ChatbotAPI:
    def __init__(self):
        # Create an instance of the Chatbot
        self.bot = Chatbot()
        # Create a router for our endpoints
        self.router = APIRouter()
        # Register endpoints on the router
        self.router.add_api_route("/chat", self.chat_get, methods=["GET"])
        self.router.add_api_route("/chat", self.chat_post, methods=["POST"])
        self.router.add_api_route("/chat/stream", self.chat_stream, methods=["POST"])
        self.router.add_api_route("/chat/history", self.get_history, methods=["GET"])
        self.router.add_api_route("/chat/history", self.clear_history, methods=["DELETE"])
        self.router.add_api_route("/chat/config", self.update_config, methods=["POST"])
        self.router.add_api_route("/chat/analyze", self.analyze_text, methods=["POST"])
        self.router.add_api_route("/chat/plan", self.create_task_plan, methods=["POST"])
    
    async def chat_get(self):
        """GET endpoint for informational purposes."""
        return {
            "message": "AI-Powered Chatbot API with Advanced NLP Features",
            "version": "2.0",
            "available_models": list(Chatbot.SUPPORTED_MODELS.keys()),
            "endpoints": {
                "/chat": "Standard chat endpoint with NLP enhancements",
                "/chat/stream": "Streaming chat endpoint for real-time responses",
                "/chat/history": "Get conversation history (GET) or clear it (DELETE)",
                "/chat/config": "Update chatbot configuration (model, system prompt)",
                "/chat/analyze": "Analyze text for intent, entities, sentiment without generating response",
                "/chat/plan": "Create a task plan from user input"
            },
            "nlp_features": {
                "intent_recognition": "Identifies user intent (greeting, question, request, etc.)",
                "entity_extraction": "Extracts entities (dates, times, names, etc.)",
                "sentiment_analysis": "Analyzes emotional tone of messages",
                "conversation_flow": "Manages conversation context and state",
                "task_planning": "Creates structured plans for tasks",
                "response_quality": "Enhances response clarity and completeness"
            }
        }
    
    async def chat_post(self, message: Message):
        """POST endpoint that uses the Chatbot to generate a response."""
        response_data = self.bot.generate_response_with_context(message.user_input)
        return response_data
    
    async def chat_stream(self, message: Message):
        """Streaming endpoint for real-time response generation."""
        async def generate():
            # Send initial message
            yield f"data: {json.dumps({'type': 'start', 'message': 'Generating response...'})}\n\n"
            
            # Generate the response
            response = await asyncio.to_thread(
                self.bot.generate_response_stream, 
                message.user_input
            )
            
            # Stream the response token by token
            async for token in response:
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
            }
        )
    
    async def get_history(self):
        """Get the conversation history."""
        return {
            "history": self.bot.get_conversation_history(),
            "total_turns": len(self.bot.conversation_history)
        }
    
    async def clear_history(self):
        """Clear the conversation history."""
        self.bot.clear_history()
        return {"message": "Conversation history cleared"}
    
    async def update_config(self, config: ChatConfig):
        """Update chatbot configuration."""
        updates = []
        
        if config.model_name and config.model_name != self.bot.model.config.name_or_path:
            # Reinitialize with new model
            self.bot = Chatbot(model_name=config.model_name)
            updates.append(f"Model changed to {config.model_name}")
        
        if config.system_prompt:
            self.bot.set_system_prompt(config.system_prompt)
            updates.append("System prompt updated")
        
        return {
            "message": "Configuration updated",
            "updates": updates,
            "current_model": self.bot.model.config.name_or_path
        }
    
    async def analyze_text(self, message: Message):
        """Analyze text for NLP insights without generating a response."""
        nlp_result = self.bot.nlp_engine.analyze(message.user_input)
        
        return {
            "intent": {
                "primary": nlp_result.intent.intent.value,
                "confidence": nlp_result.intent.confidence,
                "sub_intents": [
                    {"intent": i.value, "confidence": c} 
                    for i, c in (nlp_result.intent.sub_intents or [])
                ]
            },
            "entities": [
                {"text": e.text, "type": e.type, "start": e.start, "end": e.end}
                for e in nlp_result.entities
            ],
            "sentiment": nlp_result.sentiment,
            "is_question": nlp_result.is_question,
            "requires_action": nlp_result.requires_action,
            "topic": nlp_result.topic,
            "keywords": nlp_result.keywords
        }
    
    async def create_task_plan(self, message: Message):
        """Create a task plan based on user input."""
        nlp_result = self.bot.nlp_engine.analyze(message.user_input)
        task_plan = self.bot.task_planner.create_task_plan(message.user_input, nlp_result)
        
        return {
            "task_plan": task_plan,
            "nlp_insights": {
                "intent": nlp_result.intent.intent.value,
                "entities": [{"text": e.text, "type": e.type} for e in nlp_result.entities],
                "keywords": nlp_result.keywords[:5]
            }
        }

# Create FastAPI app
app = FastAPI()

# Enable CORS so your React frontend (running on localhost:3000) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or ["*"] to allow all origins
    allow_credentials=True,
    allow_methods=["*"],   # allow POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)

# Instantiate and include our Chatbot API router
chatbot_api = ChatbotAPI()
app.include_router(chatbot_api.router)

# Include Voice API routes
voice_api = VoiceAPI(chatbot_api.bot)
app.include_router(voice_api.router, prefix="/voice")

# Update root endpoint
@app.get("/")
async def root():
    return {
        "message": "AI-Powered Chatbot with Voice & NLP Capabilities",
        "version": "3.0",
        "features": {
            "chat": "Advanced conversational AI with NLP",
            "voice": "Speech recognition and synthesis",
            "nlp": "Intent recognition, entity extraction, sentiment analysis"
        },
        "api_docs": "/docs",
        "endpoints": {
            "chat": "/chat/*",
            "voice": "/voice/*"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)