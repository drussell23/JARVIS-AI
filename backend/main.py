#!/usr/bin/env python3
"""
Optimized main.py with Parallel Initialization Support
Reduces startup time through concurrent imports and lazy loading
"""

import os
import sys
import asyncio
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

# Configure logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if we're in optimized mode - default to True for faster startup
OPTIMIZE_STARTUP = os.getenv('OPTIMIZE_STARTUP', 'true').lower() == 'true'
PARALLEL_IMPORTS = os.getenv('BACKEND_PARALLEL_IMPORTS', 'true').lower() == 'true'
LAZY_LOAD_MODELS = os.getenv('BACKEND_LAZY_LOAD_MODELS', 'true').lower() == 'true'

if OPTIMIZE_STARTUP:
    logger.info("🚀 Running in OPTIMIZED startup mode")
    logger.info(f"  Parallel imports: {PARALLEL_IMPORTS}")
    logger.info(f"  Lazy load models: {LAZY_LOAD_MODELS}")

# Fix TensorFlow import issues before any other imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"

# FastAPI and core imports (always needed)
from fastapi import FastAPI, APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Global component storage
components = {}
import_times = {}

async def parallel_import_components():
    """Import all components in parallel for faster startup"""
    start_time = time.time()
    logger.info("⚡ Starting parallel component imports...")
    
    import concurrent.futures
    from functools import partial
    
    # Define import tasks
    import_tasks = {
        'chatbots': import_chatbots,
        'vision': import_vision_system,
        'memory': import_memory_system,
        'voice': import_voice_system,
        'ml_models': import_ml_models,
        'monitoring': import_monitoring
    }
    
    # Use thread pool for imports
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all import tasks
        futures = {
            name: executor.submit(func)
            for name, func in import_tasks.items()
        }
        
        # Wait for completion
        for name, future in futures.items():
            try:
                result = future.result(timeout=10)
                components[name] = result
                logger.info(f"  ✅ {name} loaded")
            except Exception as e:
                logger.warning(f"  ⚠️ {name} failed: {e}")
                components[name] = None
    
    elapsed = time.time() - start_time
    logger.info(f"⚡ Parallel imports completed in {elapsed:.1f}s")

def import_chatbots():
    """Import chatbot components"""
    chatbots = {}
    
    try:
        from chatbots.claude_vision_chatbot import ClaudeVisionChatbot
        chatbots['vision'] = ClaudeVisionChatbot
        chatbots['vision_available'] = True
    except ImportError:
        try:
            from chatbots.claude_chatbot import ClaudeChatbot
            chatbots['claude'] = ClaudeChatbot
            chatbots['vision_available'] = False
        except ImportError:
            pass
    
    return chatbots

def import_vision_system():
    """Import vision components"""
    vision = {}
    
    try:
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
        from vision.video_stream_capture import VideoStreamCapture, MACOS_CAPTURE_AVAILABLE
        
        vision['analyzer'] = ClaudeVisionAnalyzer
        vision['video_capture'] = VideoStreamCapture
        vision['macos_available'] = MACOS_CAPTURE_AVAILABLE
        vision['available'] = True
    except ImportError:
        vision['available'] = False
    
    # Check purple indicator separately
    try:
        from vision.simple_purple_indicator import SimplePurpleIndicator
        vision['purple_indicator'] = True
    except ImportError:
        vision['purple_indicator'] = False
    
    return vision

def import_memory_system():
    """Import memory management"""
    memory = {}
    
    try:
        from memory.memory_manager import M1MemoryManager, ComponentPriority
        from memory.memory_api import MemoryAPI, create_memory_alert_callback
        
        memory['manager_class'] = M1MemoryManager
        memory['priority'] = ComponentPriority
        memory['api'] = MemoryAPI
        memory['available'] = True
    except ImportError:
        memory['available'] = False
        
        # Create stubs
        class M1MemoryManager:
            async def start_monitoring(self): pass
            async def stop_monitoring(self): pass
            async def get_memory_snapshot(self):
                from types import SimpleNamespace
                return SimpleNamespace(
                    state=SimpleNamespace(value="normal"),
                    percent=0.5,
                    available=8 * 1024 * 1024 * 1024,
                    total=16 * 1024 * 1024 * 1024,
                )
            def register_component(self, *args, **kwargs): pass
        
        memory['manager_class'] = M1MemoryManager
    
    return memory

def import_voice_system():
    """Import voice components"""
    voice = {}
    
    try:
        from api.voice_api import VoiceAPI
        voice['api'] = VoiceAPI
        voice['available'] = True
    except ImportError:
        voice['available'] = False
    
    try:
        from api.enhanced_voice_routes import router as enhanced_voice_router
        voice['enhanced_router'] = enhanced_voice_router
        voice['enhanced_available'] = True
    except ImportError:
        voice['enhanced_available'] = False
    
    try:
        from api.jarvis_voice_api import router as jarvis_voice_router
        voice['jarvis_router'] = jarvis_voice_router
        voice['jarvis_available'] = True
    except ImportError:
        voice['jarvis_available'] = False
    
    return voice

def import_ml_models():
    """Import ML models (lazy load if enabled)"""
    ml = {}
    
    if LAZY_LOAD_MODELS:
        logger.info("  📦 ML models will be loaded on demand")
        ml['lazy_loaded'] = True
        return ml
    
    try:
        from ml_model_loader import initialize_models, get_loader_status
        from api.model_status_api import router as model_status_router
        
        ml['initialize_models'] = initialize_models
        ml['get_status'] = get_loader_status
        ml['status_router'] = model_status_router
        ml['available'] = True
    except ImportError:
        ml['available'] = False
    
    return ml

def import_monitoring():
    """Import monitoring components"""
    monitoring = {}
    
    try:
        from api.monitoring_api import router as monitoring_router
        monitoring['router'] = monitoring_router
        monitoring['available'] = True
    except ImportError:
        monitoring['available'] = False
    
    return monitoring

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Optimized lifespan handler with parallel initialization"""
    logger.info("🚀 Starting optimized JARVIS backend...")
    start_time = time.time()
    
    # Run parallel imports if enabled
    if OPTIMIZE_STARTUP and PARALLEL_IMPORTS:
        await parallel_import_components()
    else:
        # Sequential imports (legacy mode)
        logger.info("Running sequential imports (legacy mode)")
        components['chatbots'] = import_chatbots()
        components['vision'] = import_vision_system()
        components['memory'] = import_memory_system()
        components['voice'] = import_voice_system()
        components['ml_models'] = import_ml_models()
        components['monitoring'] = import_monitoring()
    
    # Initialize memory manager
    memory_class = components.get('memory', {}).get('manager_class')
    if memory_class:
        app.state.memory_manager = memory_class()
        await app.state.memory_manager.start_monitoring()
        logger.info("✅ Memory manager initialized")
    
    # Initialize vision analyzer if available
    vision = components.get('vision', {})
    if vision.get('available'):
        analyzer_class = vision.get('analyzer')
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if analyzer_class and api_key:
            app.state.vision_analyzer = analyzer_class(api_key)
            logger.info("✅ Vision analyzer initialized")
            
            # Set app state in JARVIS factory for dependency injection
            try:
                from api.jarvis_factory import set_app_state
                set_app_state(app.state)
                logger.info("✅ App state set in JARVIS factory")
            except ImportError:
                logger.warning("⚠️ JARVIS factory not available for dependency injection")
        elif analyzer_class:
            logger.warning("⚠️ Vision analyzer available but no ANTHROPIC_API_KEY set")
    
    # Initialize ML models if not lazy loading
    ml = components.get('ml_models', {})
    if ml.get('available') and not LAZY_LOAD_MODELS:
        init_func = ml.get('initialize_models')
        if init_func:
            asyncio.create_task(init_func())
            logger.info("✅ ML models initialization started")
    
    elapsed = time.time() - start_time
    logger.info(f"✨ Optimized startup completed in {elapsed:.1f}s")
    
    # Mount routers during startup
    mount_routers()
    
    # Log final status
    logger.info("\n" + "=" * 60)
    logger.info("🤖 JARVIS Backend (Optimized) Ready!")
    logger.info(f"📊 Components loaded: {sum(1 for c in components.values() if c)}/{len(components)}")
    logger.info(f"🚀 Mode: {'Optimized' if OPTIMIZE_STARTUP else 'Legacy'}")
    logger.info("=" * 60 + "\n")
    
    yield
    
    # Cleanup
    if hasattr(app.state, 'memory_manager'):
        await app.state.memory_manager.stop_monitoring()

# Create FastAPI app
logger.info("Creating optimized FastAPI app...")
app = FastAPI(
    title="JARVIS Backend (Optimized)",
    version="12.8-parallel",
    lifespan=lifespan
)

# Configure CORS
origins = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:3001').split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Quick health check endpoint"""
    return {
        "status": "healthy",
        "mode": "optimized" if OPTIMIZE_STARTUP else "legacy",
        "parallel_imports": PARALLEL_IMPORTS,
        "lazy_models": LAZY_LOAD_MODELS,
        "components": {
            name: bool(comp) for name, comp in components.items() if comp is not None
        }
    }

# Mount routers based on available components
def mount_routers():
    """Mount API routers based on loaded components"""
    
    # Memory API
    memory = components.get('memory', {})
    if memory.get('available') and hasattr(app.state, 'memory_manager'):
        memory_api_class = memory.get('api')
        if memory_api_class:
            memory_api = memory_api_class(app.state.memory_manager)
            app.include_router(memory_api.router, prefix="/memory", tags=["memory"])
            logger.info("✅ Memory API mounted")
    
    # Voice API
    voice = components.get('voice', {})
    if voice and voice.get('jarvis_available'):
        app.include_router(voice['jarvis_router'], prefix="/voice/jarvis", tags=["jarvis"])
        logger.info("✅ JARVIS Voice API mounted")
    
    if voice and voice.get('enhanced_available'):
        app.include_router(voice['enhanced_router'], prefix="/voice/enhanced", tags=["voice"])
        logger.info("✅ Enhanced Voice API mounted")
    
    # ML Model Status API
    ml = components.get('ml_models', {})
    if ml and ml.get('status_router'):
        app.include_router(ml['status_router'], prefix="/models", tags=["models"])
        logger.info("✅ Model Status API mounted")
    
    # Monitoring API
    monitoring = components.get('monitoring', {})
    if monitoring and monitoring.get('router'):
        app.include_router(monitoring['router'], prefix="/monitoring", tags=["monitoring"])
        logger.info("✅ Monitoring API mounted")

# Note: Startup tasks are now handled in the lifespan handler above

# Basic test endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "JARVIS Backend (Optimized) is running",
        "version": "12.8-parallel",
        "components": {
            name: bool(comp) for name, comp in components.items() if comp is not None
        }
    }

# Add more endpoints based on loaded components...
# (The rest of your API endpoints would go here)

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='JARVIS Backend Server')
    parser.add_argument('--port', type=int, default=int(os.getenv('BACKEND_PORT', '8000')),
                        help='Port to run the server on')
    args = parser.parse_args()
    
    # Use optimized settings if enabled
    if OPTIMIZE_STARTUP:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=args.port,
            log_level="info",
            access_log=False,  # Disable access logs for performance
            loop="uvloop" if sys.platform != "win32" else "asyncio"  # Use uvloop on Unix
        )
    else:
        uvicorn.run(app, host="0.0.0.0", port=args.port)