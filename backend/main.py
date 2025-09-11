#!/usr/bin/env python3
"""
JARVIS AI Backend - Optimized Main Entry Point

This backend loads 6 critical components that power the JARVIS AI system:

1. CHATBOTS (Claude Vision Chatbot)
   - Powers conversational AI with Claude 3.5 Sonnet
   - Enables screen analysis and vision capabilities
   - Handles all natural language understanding and generation
   
2. VISION (Screen Capture & Analysis)
   - Real-time screen monitoring with Swift-based capture
   - Video streaming at 30 FPS with purple recording indicator
   - Computer vision analysis for understanding screen content
   - Integration Architecture: 9-stage processing pipeline with dynamic memory management
   - Bloom Filter Network: Hierarchical duplicate detection system
   - Predictive Engine: State-based prediction with Markov chains
   - Semantic Cache LSH: Intelligent caching with locality-sensitive hashing
   - Quadtree Spatial Intelligence: Optimized region-based processing
   - VSMS Core: Visual State Management System with scene understanding
   
3. MEMORY (M1 Mac Optimized Memory Manager)
   - Prevents memory leaks and manages resources efficiently
   - Critical for long-running sessions and video processing
   - Provides memory pressure alerts and automatic cleanup
   - Integration with Orchestrator for dynamic component allocation
   
4. VOICE (JARVIS Voice Interface)
   - Voice activation with "Hey JARVIS" wake word
   - Text-to-speech with multiple voice options
   - Real-time voice command processing
   
5. ML_MODELS (Machine Learning Models)
   - Sentiment analysis and NLP capabilities
   - Lazy-loaded to optimize startup time
   - Powers intelligent text understanding
   
6. MONITORING (System Health & Metrics)
   - Tracks API performance and resource usage
   - Provides health checks and status endpoints
   - Essential for production stability
   - Integration metrics tracking for all vision components

All 6 components must load successfully for full JARVIS functionality.
The system uses parallel imports to reduce startup time from ~20s to ~7-9s.

Enhanced Vision Features (v12.9.2):
- Integration Orchestrator with 1.2GB memory budget
- 4 operating modes: Normal, Pressure, Critical, Emergency
- Cross-language optimization: Python, Rust, Swift
- Intelligent component coordination based on system resources
"""

import os
import sys
import asyncio
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

# Configure logging early with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enable debug logging for specific modules
logging.getLogger("api.jarvis_voice_api").setLevel(logging.DEBUG)
logging.getLogger("api.jarvis_factory").setLevel(logging.DEBUG)
logging.getLogger("chatbots.claude_vision_chatbot").setLevel(logging.DEBUG)
# Add weather-specific debug logging
logging.getLogger("voice.jarvis_agent_voice").setLevel(logging.DEBUG)
logging.getLogger("workflows.weather_app_vision_unified").setLevel(logging.DEBUG)
logging.getLogger("system_control.unified_vision_weather").setLevel(logging.DEBUG)
logging.getLogger("api.voice_websocket_handler").setLevel(logging.DEBUG)

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
        from api.jarvis_voice_api import jarvis_api, router as jarvis_voice_router
        voice['jarvis_router'] = jarvis_voice_router
        voice['jarvis_api'] = jarvis_api
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
            
            # Set vision analyzer in vision websocket manager
            try:
                from api.vision_websocket import set_vision_analyzer
                set_vision_analyzer(app.state.vision_analyzer)
                logger.info("✅ Vision analyzer set in vision websocket manager")
            except ImportError as e:
                logger.warning(f"⚠️ Could not set vision analyzer in websocket: {e}")
            
            # Set app state in JARVIS factory for dependency injection
            try:
                from api.jarvis_factory import set_app_state
                set_app_state(app.state)
                logger.info("✅ App state set in JARVIS factory")
            except ImportError:
                logger.warning("⚠️ JARVIS factory not available for dependency injection")
            
            # Initialize weather system with vision
            try:
                from system_control.weather_system_config import initialize_weather_system
                from system_control.macos_controller import MacOSController
                
                controller = MacOSController()
                weather_bridge = initialize_weather_system(app.state.vision_analyzer, controller)
                app.state.weather_system = weather_bridge
                logger.info("✅ Weather system initialized with vision")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize weather system: {e}")
                
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
    
    # Log final status with component details
    logger.info("\n" + "=" * 60)
    logger.info("🤖 JARVIS Backend (Optimized) Ready!")
    
    # Count and display loaded components
    loaded_count = sum(1 for c in components.values() if c)
    logger.info(f"📊 Components loaded: {loaded_count}/{len(components)}")
    
    # Show status of each component
    component_status = [
        ("✅" if components.get('chatbots') else "❌", "CHATBOTS    - AI conversation & vision analysis"),
        ("✅" if components.get('vision') else "❌", "VISION      - Screen capture & real-time monitoring"),
        ("✅" if components.get('memory') else "❌", "MEMORY      - Resource management & optimization"),
        ("✅" if components.get('voice') else "❌", "VOICE       - Voice activation & speech synthesis"),
        ("✅" if components.get('ml_models') else "❌", "ML_MODELS   - NLP & sentiment analysis"),
        ("✅" if components.get('monitoring') else "❌", "MONITORING  - System health & metrics")
    ]
    
    for status, desc in component_status:
        logger.info(f"   {status} {desc}")
    
    logger.info(f"🚀 Mode: {'Optimized' if OPTIMIZE_STARTUP else 'Legacy'}")
    
    if loaded_count == 6:
        logger.info("✨ All systems operational - JARVIS is fully functional!")
    else:
        logger.warning(f"⚠️  Only {loaded_count}/6 components loaded - some features may be limited")
    
    logger.info("=" * 60 + "\n")
    
    yield
    
    # Cleanup
    if hasattr(app.state, 'memory_manager'):
        await app.state.memory_manager.stop_monitoring()

# Apply vision monitoring fix
try:
    import api.direct_vision_fix
    logger.info("Vision monitoring fix applied")
except Exception as e:
    logger.warning(f"Could not apply vision fix: {e}")

# Create FastAPI app
logger.info("Creating optimized FastAPI app...")
app = FastAPI(
    title="JARVIS Backend (Optimized)",
    version="12.9.2-integration",
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
    vision_details = {}
    
    # Check vision component status
    if hasattr(app.state, 'vision_analyzer'):
        try:
            # Check orchestrator status
            orchestrator = await app.state.vision_analyzer.get_orchestrator()
            if orchestrator:
                status = await orchestrator.get_system_status()
                vision_details['orchestrator'] = {
                    'enabled': True,
                    'mode': status['system_mode'],
                    'memory_usage_mb': status['memory_usage_mb'],
                    'active_components': sum(1 for v in status['components'].values() if v == 'active')
                }
            else:
                vision_details['orchestrator'] = {'enabled': False}
        except:
            vision_details['orchestrator'] = {'enabled': False}
    
    return {
        "status": "healthy",
        "mode": "optimized" if OPTIMIZE_STARTUP else "legacy",
        "parallel_imports": PARALLEL_IMPORTS,
        "lazy_models": LAZY_LOAD_MODELS,
        "components": {
            name: bool(comp) for name, comp in components.items() if comp is not None
        },
        "vision_enhanced": vision_details
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
    
    # Unified WebSocket API - replaces individual WebSocket endpoints
    try:
        from api.unified_websocket import router as unified_ws_router
        app.include_router(unified_ws_router, tags=["websocket"])
        logger.info("✅ Unified WebSocket API mounted at /ws")
    except ImportError as e:
        logger.warning(f"Could not import unified WebSocket router: {e}")
        
        # Fallback to individual WebSocket APIs if unified not available
        try:
            from api.vision_websocket import router as vision_ws_router
            app.include_router(vision_ws_router, prefix="/vision", tags=["vision"])
            logger.info("✅ Vision WebSocket API mounted (fallback)")
        except ImportError as e:
            logger.warning(f"Could not import vision WebSocket router: {e}")
        
        # ML Audio API
        try:
            from api.ml_audio_api import router as ml_audio_router
            app.include_router(ml_audio_router, tags=["ML Audio"])
            logger.info("✅ ML Audio API mounted (fallback)")
        except ImportError as e:
            logger.warning(f"Could not import ML Audio router: {e}")
    
    # Network Recovery API (kept separate as it's not WebSocket)
    try:
        from api.network_recovery_api import router as network_recovery_router
        app.include_router(network_recovery_router, tags=["Network Recovery"])
        logger.info("✅ Network Recovery API mounted")
    except ImportError as e:
        logger.warning(f"Could not import Network Recovery router: {e}")

# Note: Startup tasks are now handled in the lifespan handler above

# Basic test endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "JARVIS Backend (Optimized) is running",
        "version": "12.9.2-integration",
        "components": {
            name: bool(comp) for name, comp in components.items() if comp is not None
        }
    }

# Note: Main WebSocket endpoint is now handled by unified_websocket router at /ws
# This provides a single endpoint for all WebSocket communication

# Audio endpoints for frontend compatibility
@app.post("/audio/speak")
async def audio_speak_post(request: dict):
    """Forward audio speak requests to JARVIS voice API"""
    from fastapi import HTTPException
    voice = components.get('voice', {})
    jarvis_api = voice.get('jarvis_api')
    
    if not jarvis_api:
        raise HTTPException(status_code=503, detail="JARVIS voice not available")
    
    # Forward to JARVIS speak endpoint
    return await jarvis_api.speak(request)

@app.get("/audio/speak/{text}")
async def audio_speak_get(text: str):
    """GET endpoint for audio speak (frontend fallback)"""
    return await audio_speak_post({"text": text})

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