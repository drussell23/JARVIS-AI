#!/usr/bin/env python3
"""
Simple test API to verify vision command routing works
"""

import asyncio
import os
import sys
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

# Load environment
load_dotenv(override=True)

# Clear module cache
for module in list(sys.modules.keys()):
    if 'unified' in module or 'vision' in module:
        del sys.modules[module]

app = FastAPI()

@app.post("/test_classification")
async def test_classification(request: dict):
    """Test command classification"""
    from api.unified_command_processor import UnifiedCommandProcessor
    
    command = request.get("text", "")
    processor = UnifiedCommandProcessor()
    
    command_type, confidence = await processor._classify_command(command)
    vision_score = processor._calculate_vision_score(command.lower().split(), command.lower())
    
    return {
        "command": command,
        "command_type": command_type.value,
        "confidence": confidence,
        "vision_score": vision_score,
        "is_vision": command_type.value == "vision"
    }

@app.post("/test_vision")
async def test_vision(request: dict):
    """Test vision handler directly"""
    from api.vision_command_handler import VisionCommandHandler
    
    command = request.get("text", "")
    handler = VisionCommandHandler()
    
    result = await handler.analyze_screen(command)
    
    return {
        "success": result.get("handled", False),
        "response": result.get("response", ""),
        "response_length": len(result.get("response", ""))
    }

if __name__ == "__main__":
    print("Starting test API server on port 8888...")
    uvicorn.run(app, host="0.0.0.0", port=8888)