"""
HTTP handlers for WebSocket messages
Provides HTTP endpoints for the TypeScript WebSocket router
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import logging
import importlib
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["websocket"])

class WebSocketHandlerRequest(BaseModel):
    """Request model for WebSocket handler calls"""
    module: str
    function: str
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}
    context: Dict[str, Any] = {}

class WebSocketHandlerResponse(BaseModel):
    """Response model for WebSocket handler calls"""
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    traceback: Optional[str] = None

# Cache for imported handlers
_handler_cache: Dict[str, Any] = {}

async def call_handler(request: WebSocketHandlerRequest) -> WebSocketHandlerResponse:
    """Call a WebSocket handler function"""
    try:
        # Get handler from cache or import
        cache_key = f"{request.module}.{request.function}"
        
        if cache_key not in _handler_cache:
            # Dynamic import
            module = importlib.import_module(request.module)
            handler = getattr(module, request.function)
            _handler_cache[cache_key] = handler
        else:
            handler = _handler_cache[cache_key]
        
        # Call handler
        if asyncio.iscoroutinefunction(handler):
            result = await handler(*request.args, **request.kwargs)
        else:
            result = handler(*request.args, **request.kwargs)
        
        return WebSocketHandlerResponse(
            success=True,
            result=result
        )
        
    except Exception as e:
        logger.error(f"Handler error: {e}", exc_info=True)
        import traceback
        return WebSocketHandlerResponse(
            success=False,
            error=str(e),
            traceback=traceback.format_exc()
        )

@router.post("/general/handler", response_model=WebSocketHandlerResponse)
async def handle_general_websocket(request: WebSocketHandlerRequest):
    """Handle general WebSocket messages"""
    # Override module if not specified
    if not request.module or request.module == "backend.api.general_websocket_handler":
        request.module = "backend.api.general_websocket_handler"
    
    return await call_handler(request)

@router.post("/vision/handler", response_model=WebSocketHandlerResponse)
async def handle_vision_websocket(request: WebSocketHandlerRequest):
    """Handle vision WebSocket messages"""
    # Override module if not specified
    if not request.module or request.module == "backend.api.unified_vision_handler":
        request.module = "backend.api.unified_vision_handler"
    
    return await call_handler(request)

@router.post("/voice/handler", response_model=WebSocketHandlerResponse)
async def handle_voice_websocket(request: WebSocketHandlerRequest):
    """Handle voice WebSocket messages"""
    # Override module if not specified
    if not request.module or request.module == "backend.api.voice_websocket_handler":
        request.module = "backend.api.voice_websocket_handler"
    
    return await call_handler(request)

@router.get("/endpoints")
async def discover_endpoints():
    """Discover available WebSocket endpoints"""
    try:
        from ..main import app
        
        # Get all WebSocket routes
        ws_routes = []
        
        for route in app.routes:
            if hasattr(route, 'path') and '/ws' in route.path:
                ws_routes.append({
                    'path': route.path,
                    'methods': list(route.methods) if hasattr(route, 'methods') else [],
                    'name': route.name if hasattr(route, 'name') else None,
                    'tags': getattr(route, 'tags', [])
                })
        
        return {
            'endpoints': ws_routes,
            'http_handlers': [
                '/ws/general/handler',
                '/ws/vision/handler', 
                '/ws/voice/handler'
            ],
            'modules': {
                'general': 'backend.api.general_websocket_handler',
                'vision': 'backend.api.unified_vision_handler',
                'voice': 'backend.api.voice_websocket_handler'
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to discover endpoints: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def execute_batch(operations: List[WebSocketHandlerRequest]):
    """Execute multiple handler calls in batch"""
    results = []
    
    # Execute operations concurrently
    tasks = [call_handler(op) for op in operations]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    for response in responses:
        if isinstance(response, Exception):
            results.append(WebSocketHandlerResponse(
                success=False,
                error=str(response)
            ))
        else:
            results.append(response)
    
    return {'results': results}

# Specific handlers for common operations
@router.post("/health/check")
async def websocket_health_check():
    """Health check for WebSocket handlers"""
    return {
        'status': 'healthy',
        'handlers': {
            'general': 'backend.api.general_websocket_handler' in _handler_cache,
            'vision': 'backend.api.unified_vision_handler' in _handler_cache,
            'voice': 'backend.api.voice_websocket_handler' in _handler_cache
        },
        'cache_size': len(_handler_cache)
    }