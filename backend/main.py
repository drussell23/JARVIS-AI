# main.py
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from chatbot import Chatbot  # Import the Chatbot class
import asyncio
import json
from typing import Optional

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
    
    async def chat_get(self):
        """GET endpoint for informational purposes."""
        return {
            "message": "This endpoint accepts POST requests with a JSON payload: {'user_input': 'your message'}.",
            "available_models": list(Chatbot.SUPPORTED_MODELS.keys()),
            "endpoints": {
                "/chat": "Standard chat endpoint",
                "/chat/stream": "Streaming chat endpoint",
                "/chat/history": "Get conversation history",
                "/chat/config": "Update chatbot configuration"
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)