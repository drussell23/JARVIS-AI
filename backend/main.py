#!/usr/bin/env python3
"""
JARVIS AI Backend - Optimized Main Entry Point

This backend loads 7 critical components that power the JARVIS AI system:

1. CHATBOTS (Claude Vision Chatbot)
   - Powers conversational AI with Claude 3.5 Sonnet
   - Enables screen analysis and vision capabilities
   - Handles all natural language understanding and generation

2. VISION (Screen Capture & Multi-Space Analysis)
   - Real-time screen monitoring with Swift-based capture
   - Video streaming at 30 FPS with purple recording indicator
   - Multi-Space Desktop Vision: Monitors all macOS desktop spaces
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

7. VOICE UNLOCK (Biometric Mac Authentication) - NEW!
   - Voice-based biometric authentication for macOS
   - Secure voiceprint enrollment and storage
   - Anti-spoofing protection with liveness detection
   - Screensaver and system integration
   - Adaptive authentication with continuous learning

8. WAKE WORD (Hands-free Activation) - NEW!
   - "Hey JARVIS" wake word detection
   - Always-listening mode with zero button clicks
   - Multi-engine detection (Porcupine, Vosk, WebRTC)
   - Adaptive sensitivity and anti-spoofing
   - Customizable wake words and responses

All 8 components must load successfully for full JARVIS functionality.
The system uses parallel imports to reduce startup time from ~20s to ~7-9s.

Enhanced Vision Features (v13.3.1):
- Integration Orchestrator with 1.2GB memory budget
- 4 operating modes: Normal, Pressure, Critical, Emergency
- Cross-language optimization: Python, Rust, Swift
- Intelligent component coordination based on system resources
- Proactive Vision Intelligence System with real-time monitoring
- Multi-Space Desktop Vision: Sees across all macOS desktop spaces
- Smart Space Queries: "Where is Cursor IDE?", "What's on Desktop 2?"
- Debugging Assistant: Auto-detects code errors and syntax issues
- Research Helper: Monitors multi-tab research workflows
- Workflow Optimizer: Identifies repetitive patterns and suggests optimizations
- Privacy Protection: Auto-pauses during sensitive content (passwords, banking)
- Natural Voice Communication: Speaks suggestions and warnings naturally

Browser Automation Features (v13.4.0):
- Natural Language Browser Control: "Open Safari and go to Google"
- Chained Commands: "Open a new tab and search for weather"
- Dynamic Browser Discovery: Controls any browser without hardcoding
- Smart Context: Remembers which browser you're using between commands
- Type & Search: "Type python tutorials and press enter"
- Tab Management: "Open another tab", "Open a new tab in Chrome"
- Cross-Browser Support: Safari, Chrome, Firefox, and others
- AppleScript Integration: Native macOS browser control
"""

import os
import sys
import asyncio
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

# Enable enhanced ML model logging
try:
    from enable_ml_logging import configure_ml_logging

    ml_logger_instance, memory_visualizer = configure_ml_logging()
    logger = logging.getLogger(__name__)
    logger.info("🚀 Enhanced ML model logging enabled")
    ML_LOGGING_ENABLED = True
except ImportError:
    # Configure logging early with more detail
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.warning("Enhanced ML logging not available - using standard logging")
    ML_LOGGING_ENABLED = False

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
OPTIMIZE_STARTUP = os.getenv("OPTIMIZE_STARTUP", "true").lower() == "true"
PARALLEL_IMPORTS = os.getenv("BACKEND_PARALLEL_IMPORTS", "true").lower() == "true"
LAZY_LOAD_MODELS = os.getenv("BACKEND_LAZY_LOAD_MODELS", "true").lower() == "true"

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
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load environment variables (force override of system env vars)
try:
    from dotenv import load_dotenv

    load_dotenv(override=True)  # Force .env to override existing environment variables
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
        "chatbots": import_chatbots,
        "vision": import_vision_system,
        "memory": import_memory_system,
        "voice": import_voice_system,
        "ml_models": import_ml_models,
        "monitoring": import_monitoring,
        "voice_unlock": import_voice_unlock,
        "wake_word": import_wake_word,
    }

    # Use thread pool for imports
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all import tasks
        futures = {name: executor.submit(func) for name, func in import_tasks.items()}

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

        chatbots["vision"] = ClaudeVisionChatbot
        chatbots["vision_available"] = True
    except ImportError:
        try:
            from chatbots.claude_chatbot import ClaudeChatbot

            chatbots["claude"] = ClaudeChatbot
            chatbots["vision_available"] = False
        except ImportError:
            pass

    return chatbots


def import_vision_system():
    """Import vision components"""
    vision = {}

    try:
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
        from vision.video_stream_capture import (
            VideoStreamCapture,
            MACOS_CAPTURE_AVAILABLE,
        )

        vision["analyzer"] = ClaudeVisionAnalyzer
        vision["video_capture"] = VideoStreamCapture
        vision["macos_available"] = MACOS_CAPTURE_AVAILABLE
        vision["available"] = True
    except ImportError:
        vision["available"] = False

    # Check purple indicator separately
    try:
        from vision.simple_purple_indicator import SimplePurpleIndicator

        vision["purple_indicator"] = True
    except ImportError:
        vision["purple_indicator"] = False

    return vision


def import_memory_system():
    """Import memory management"""
    memory = {}

    try:
        from memory.memory_manager import M1MemoryManager, ComponentPriority
        from memory.memory_api import MemoryAPI, create_memory_alert_callback

        memory["manager_class"] = M1MemoryManager
        memory["priority"] = ComponentPriority
        memory["api"] = MemoryAPI
        memory["available"] = True
    except ImportError:
        memory["available"] = False

        # Create stubs
        class M1MemoryManager:
            async def start_monitoring(self):
                pass

            async def stop_monitoring(self):
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

        memory["manager_class"] = M1MemoryManager

    return memory


def import_voice_system():
    """Import voice components"""
    voice = {}

    try:
        from api.voice_api import VoiceAPI

        voice["api"] = VoiceAPI
        voice["available"] = True
    except ImportError:
        voice["available"] = False

    try:
        from api.enhanced_voice_routes import router as enhanced_voice_router

        voice["enhanced_router"] = enhanced_voice_router
        voice["enhanced_available"] = True
    except ImportError:
        voice["enhanced_available"] = False

    try:
        from api.jarvis_voice_api import jarvis_api, router as jarvis_voice_router

        voice["jarvis_router"] = jarvis_voice_router
        voice["jarvis_api"] = jarvis_api
        voice["jarvis_available"] = True
    except ImportError:
        voice["jarvis_available"] = False

    return voice


def import_ml_models():
    """Import ML models (lazy load if enabled)"""
    ml = {}

    if LAZY_LOAD_MODELS:
        logger.info("  📦 ML models will be loaded on demand")
        ml["lazy_loaded"] = True
        return ml

    try:
        from ml_model_loader import initialize_models, get_loader_status
        from api.model_status_api import router as model_status_router

        ml["initialize_models"] = initialize_models
        ml["get_status"] = get_loader_status
        ml["status_router"] = model_status_router
        ml["available"] = True
    except ImportError:
        ml["available"] = False

    return ml


def import_monitoring():
    """Import monitoring components"""
    monitoring = {}

    try:
        from api.monitoring_api import router as monitoring_router

        monitoring["router"] = monitoring_router
        monitoring["available"] = True
    except ImportError:
        monitoring["available"] = False

    return monitoring


def import_voice_unlock():
    """Import voice unlock components"""
    voice_unlock = {}

    try:
        from api.voice_unlock_api import (
            router as voice_unlock_router,
            initialize_voice_unlock,
        )

        voice_unlock["router"] = voice_unlock_router
        voice_unlock["initialize"] = initialize_voice_unlock
        voice_unlock["available"] = True

        # Try to initialize immediately
        if initialize_voice_unlock():
            voice_unlock["initialized"] = True
            logger.info("  ✅ Voice Unlock API initialized")
        else:
            voice_unlock["initialized"] = False
            logger.warning("  ⚠️  Voice Unlock API initialization failed")

        # Also import the startup integration for WebSocket server
        try:
            from voice_unlock.startup_integration import (
                initialize_voice_unlock_system,
                shutdown_voice_unlock_system,
                voice_unlock_startup,
            )

            voice_unlock["startup_integration"] = True
            voice_unlock["initialize_system"] = initialize_voice_unlock_system
            voice_unlock["shutdown_system"] = shutdown_voice_unlock_system
            voice_unlock["startup_manager"] = voice_unlock_startup
        except ImportError:
            logger.warning("  ⚠️  Voice Unlock startup integration not available")
            voice_unlock["startup_integration"] = False

    except ImportError as e:
        logger.warning(f"  ⚠️  Voice Unlock not available: {e}")
        voice_unlock["available"] = False
        voice_unlock["initialized"] = False

    return voice_unlock


def import_wake_word():
    """Import wake word detection components"""
    wake_word = {}

    try:
        from api.wake_word_api import (
            router as wake_word_router,
            initialize_wake_word,
            wake_service,
        )

        wake_word["router"] = wake_word_router
        wake_word["initialize"] = initialize_wake_word
        wake_word["service"] = wake_service
        wake_word["available"] = True

        # Try to initialize immediately
        if initialize_wake_word():
            wake_word["initialized"] = True
            logger.info("  ✅ Wake Word detection initialized")
        else:
            wake_word["initialized"] = False
            logger.warning("  ⚠️  Wake Word initialization failed")

    except ImportError as e:
        logger.warning(f"  ⚠️  Wake Word not available: {e}")
        wake_word["available"] = False
        wake_word["initialized"] = False

    return wake_word


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Optimized lifespan handler with parallel initialization"""
    logger.info("🚀 Starting optimized JARVIS backend...")
    start_time = time.time()

    # CRITICAL: Check for code changes and clean up old instances FIRST
    try:
        from process_cleanup_manager import (
            ensure_fresh_jarvis_instance,
            cleanup_system_for_jarvis,
        )

        logger.info("🔍 Checking for code changes and old instances...")
        if not ensure_fresh_jarvis_instance():
            logger.error("❌ Another JARVIS instance is already running on this port!")
            logger.error("   Please stop the other instance or use a different port.")
            raise RuntimeError("Port conflict - another JARVIS instance is running")

        # Run full system cleanup
        logger.info("🧹 Running system cleanup before startup...")
        cleanup_report = await cleanup_system_for_jarvis(dry_run=False)

        if cleanup_report.get("code_changes_cleanup"):
            logger.info(
                f"✅ Cleaned {len(cleanup_report['code_changes_cleanup'])} old instances due to code changes"
            )

    except ImportError:
        logger.warning(
            "⚠️  Process cleanup manager not available - old instances may still be running"
        )
    except Exception as e:
        logger.error(f"Error during process cleanup: {e}")
        # Continue startup anyway

    # Run parallel imports if enabled
    if OPTIMIZE_STARTUP and PARALLEL_IMPORTS:
        await parallel_import_components()
    else:
        # Sequential imports (legacy mode)
        logger.info("Running sequential imports (legacy mode)")
        components["chatbots"] = import_chatbots()
        components["vision"] = import_vision_system()
        components["memory"] = import_memory_system()
        components["voice"] = import_voice_system()
        components["ml_models"] = import_ml_models()
        components["monitoring"] = import_monitoring()
        components["voice_unlock"] = import_voice_unlock()
        components["wake_word"] = import_wake_word()

    # Initialize memory manager
    memory_class = components.get("memory", {}).get("manager_class")
    if memory_class:
        app.state.memory_manager = memory_class()
        await app.state.memory_manager.start_monitoring()
        logger.info("✅ Memory manager initialized")

    # Discover running services (if dynamic CORS is available)
    try:
        from api.dynamic_cors_handler import AutoPortDiscovery

        services = await AutoPortDiscovery.discover_services()
        if services:
            logger.info(f"🔍 Discovered services: {services}")
            config = AutoPortDiscovery.get_recommended_config(services)
            logger.info(f"📝 Recommended config: {config}")
    except Exception as e:
        logger.debug(f"Service discovery skipped: {e}")

    # Initialize Rust acceleration for vision system with self-healing
    try:
        from vision.rust_startup_integration import (
            initialize_rust_acceleration,
            get_rust_status,
        )
        from vision.rust_self_healer import get_self_healer
        from vision.dynamic_component_loader import get_component_loader

        # Start self-healing and dynamic component loader
        logger.info("🔧 Initializing self-healing system...")
        loader = get_component_loader()
        await loader.start()  # This also starts the self-healer

        # Initialize Rust acceleration
        rust_config = await initialize_rust_acceleration()

        if rust_config.get("available"):
            app.state.rust_acceleration = rust_config
            logger.info("🦀 Rust acceleration initialized:")

            # Log performance boosts
            boosts = rust_config.get("performance_boost", {})
            if boosts:
                for component, boost in boosts.items():
                    if boost > 1.0:
                        logger.info(f"   • {component}: {boost:.1f}x faster")

            # Log memory savings
            mem_savings = rust_config.get("memory_savings", {})
            if mem_savings.get("enabled"):
                logger.info(f"   • Memory pool: {mem_savings['rust_pool_mb']}MB")
                logger.info(
                    f"   • Estimated savings: {mem_savings['estimated_savings_percent']}%"
                )
        else:
            logger.info("🦀 Rust acceleration not available (Python fallback active)")
            logger.debug(f"   Reason: {rust_config.get('fallback_reason', 'Unknown')}")

    except Exception as e:
        logger.warning(f"⚠️ Could not initialize Rust acceleration: {e}")
        app.state.rust_acceleration = {"available": False}

    # Initialize vision analyzer if available
    vision = components.get("vision", {})
    if vision.get("available"):
        analyzer_class = vision.get("analyzer")
        api_key = os.getenv("ANTHROPIC_API_KEY")
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
                logger.warning(
                    "⚠️ JARVIS factory not available for dependency injection"
                )

            # Initialize proactive monitoring components
            try:
                # Set JARVIS API in vision command handler for voice integration
                from api.vision_command_handler import vision_command_handler

                voice = components.get("voice", {})
                if voice.get("jarvis_api"):
                    vision_command_handler.jarvis_api = voice["jarvis_api"]
                    logger.info(
                        "✅ JARVIS voice API connected to pure vision command handler"
                    )

                # Initialize pure intelligence with API key
                if api_key:
                    await vision_command_handler.initialize_intelligence(api_key)
                    logger.info("✅ Pure vision intelligence initialized")

                # Log proactive monitoring configuration
                proactive_config = app.state.vision_analyzer.get_proactive_config()
                if proactive_config["proactive_enabled"]:
                    logger.info(
                        "✅ Proactive Vision Intelligence System initialized with:"
                    )
                    logger.info(
                        f"   - Confidence threshold: {proactive_config['confidence_threshold']}"
                    )
                    logger.info(
                        f"   - Voice announcements: {'enabled' if proactive_config['voice_enabled'] else 'disabled'}"
                    )
                    logger.info("   - Debugging Assistant: Auto-detects code errors")
                    logger.info("   - Research Helper: Monitors multi-tab workflows")
                    logger.info(
                        "   - Workflow Optimizer: Identifies repetitive patterns"
                    )
                    logger.info(
                        "   - Privacy Protection: Auto-pauses for sensitive content"
                    )
                    logger.info(
                        "   - Say 'Start monitoring my screen' to activate intelligent assistance"
                    )
                else:
                    logger.info("⚠️ Proactive monitoring disabled in configuration")
            except Exception as e:
                logger.warning(
                    f"⚠️ Could not initialize proactive monitoring components: {e}"
                )

            # Initialize weather system with vision
            try:
                from system_control.weather_system_config import (
                    initialize_weather_system,
                )
                from system_control.macos_controller import MacOSController

                controller = MacOSController()
                weather_bridge = initialize_weather_system(
                    app.state.vision_analyzer, controller
                )
                app.state.weather_system = weather_bridge
                logger.info("✅ Weather system initialized with vision")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize weather system: {e}")

            # Initialize vision status manager
            try:
                from vision.vision_status_integration import (
                    initialize_vision_status,
                    setup_vision_status_callbacks,
                )

                # Initialize after WebSocket is mounted
                async def setup_vision_status():
                    await asyncio.sleep(0.5)  # Give WebSocket time to initialize
                    success = await initialize_vision_status(app)
                    if success:
                        setup_vision_status_callbacks(app)
                        logger.info(
                            "✅ Vision status manager initialized and connected"
                        )

                asyncio.create_task(setup_vision_status())
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize vision status manager: {e}")

        elif analyzer_class:
            logger.warning("⚠️ Vision analyzer available but no ANTHROPIC_API_KEY set")

    # Initialize ML models if not lazy loading
    ml = components.get("ml_models", {})
    if ml.get("available") and not LAZY_LOAD_MODELS:
        init_func = ml.get("initialize_models")
        if init_func:
            if ML_LOGGING_ENABLED:
                logger.info("\n" + "=" * 60)
                logger.info("🤖 INITIALIZING ML MODELS WITH SMART LAZY LOADING")
                logger.info("=" * 60)
                logger.info("Target: <35% memory usage (5.6GB of 16GB)")
                logger.info("Strategy: Load models one-at-a-time, only when needed")
                logger.info("Watch the console for real-time loading details...\n")
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
        (
            "✅" if components.get("chatbots") else "❌",
            "CHATBOTS    - AI conversation & vision analysis",
        ),
        (
            "✅" if components.get("vision") else "❌",
            "VISION      - Screen capture & real-time monitoring",
        ),
        (
            "✅" if components.get("memory") else "❌",
            "MEMORY      - Resource management & optimization",
        ),
        (
            "✅" if components.get("voice") else "❌",
            "VOICE       - Voice activation & speech synthesis",
        ),
        (
            "✅" if components.get("ml_models") else "❌",
            "ML_MODELS   - NLP & sentiment analysis",
        ),
        (
            "✅" if components.get("monitoring") else "❌",
            "MONITORING  - System health & metrics",
        ),
        (
            "✅" if components.get("voice_unlock") else "❌",
            "VOICE_UNLOCK - Biometric Mac authentication",
        ),
        (
            "✅" if components.get("wake_word") else "❌",
            "WAKE_WORD   - Hands-free 'Hey JARVIS' activation",
        ),
    ]

    for status, desc in component_status:
        logger.info(f"   {status} {desc}")

    logger.info(f"🚀 Mode: {'Optimized' if OPTIMIZE_STARTUP else 'Legacy'}")

    if loaded_count == 8:
        logger.info("✨ All systems operational - JARVIS is fully functional!")
    else:
        logger.warning(
            f"⚠️  Only {loaded_count}/8 components loaded - some features may be limited"
        )

    logger.info("=" * 60 + "\n")

    # Initialize Voice Unlock system components (WebSocket server)
    voice_unlock = components.get("voice_unlock", {})
    if (
        voice_unlock
        and voice_unlock.get("startup_integration")
        and voice_unlock.get("initialize_system")
    ):
        try:
            logger.info("🔐 Starting Voice Unlock system components...")
            init_system = voice_unlock["initialize_system"]
            success = await init_system()
            if success:
                app.state.voice_unlock_system = voice_unlock["startup_manager"]
                logger.info("✅ Voice Unlock system started")
                logger.info("   Say 'Hey JARVIS, unlock my mac' when screen is locked")
            else:
                logger.warning("⚠️ Voice Unlock system failed to start")
        except Exception as e:
            logger.error(f"Failed to start Voice Unlock system: {e}")

    # Initialize wake word service after all components are loaded
    wake_word = components.get("wake_word", {})
    if wake_word.get("service") and wake_word.get("initialized"):
        # Define activation callback that sends to WebSocket clients
        async def wake_word_activation_callback(data):
            """Handle wake word activation"""
            logger.info(f"Wake word activated: {data}")
            # This will be sent through WebSocket to connected clients
            # The frontend will handle the actual response

        try:
            # Start the wake word service with callback
            wake_service = wake_word["service"]
            if wake_service:
                success = await wake_service.start(wake_word_activation_callback)
                if success:
                    app.state.wake_service = wake_service
                    logger.info(
                        "🎤 Wake word detection service started - Say 'Hey JARVIS'!"
                    )
                else:
                    logger.warning("⚠️ Wake word service failed to start")
        except Exception as e:
            logger.error(f"Failed to start wake word service: {e}")

    # Register with autonomous systems
    try:
        from core.autonomous_orchestrator import get_orchestrator
        from core.zero_config_mesh import get_mesh

        orchestrator = get_orchestrator()
        mesh = get_mesh()

        # Start autonomous systems
        await orchestrator.start()
        await mesh.start()

        # Register backend service
        backend_port = int(os.getenv("BACKEND_PORT", "8000"))
        await orchestrator.register_service("jarvis_backend", backend_port, "http")
        await mesh.join({
            "name": "jarvis_backend",
            "port": backend_port,
            "protocol": "http",
            "type": "backend",
            "endpoints": {
                "health": "/health",
                "vision": "/vision",
                "voice": "/voice",
                "chat": "/chat"
            }
        })

        app.state.orchestrator = orchestrator
        app.state.mesh = mesh

        logger.info("✅ Registered with autonomous orchestrator and mesh network")
    except Exception as e:
        logger.warning(f"⚠️ Could not register with autonomous systems: {e}")

    yield

    # Cleanup
    logger.info("🛑 Shutting down JARVIS backend...")

    # Stop autonomous systems
    if hasattr(app.state, "orchestrator"):
        try:
            await app.state.orchestrator.stop()
            logger.info("✅ Autonomous orchestrator stopped")
        except Exception as e:
            logger.error(f"Failed to stop orchestrator: {e}")

    if hasattr(app.state, "mesh"):
        try:
            await app.state.mesh.stop()
            logger.info("✅ Mesh network stopped")
        except Exception as e:
            logger.error(f"Failed to stop mesh: {e}")

    # Save current code state for next startup
    try:
        from process_cleanup_manager import ProcessCleanupManager

        manager = ProcessCleanupManager()
        manager._save_code_state()
        logger.info("✅ Code state saved for next startup")
    except Exception as e:
        logger.error(f"Failed to save code state: {e}")

    # Stop Voice Unlock system
    if hasattr(app.state, "voice_unlock_system") and voice_unlock.get(
        "shutdown_system"
    ):
        try:
            shutdown_system = voice_unlock["shutdown_system"]
            await shutdown_system()
            logger.info("✅ Voice Unlock system stopped")
        except Exception as e:
            logger.error(f"Failed to stop Voice Unlock system: {e}")

    # Stop dynamic component loader and self-healer
    try:
        from vision.dynamic_component_loader import get_component_loader

        loader = get_component_loader()
        await loader.stop()
        logger.info("✅ Self-healing system stopped")
    except Exception as e:
        logger.error(f"Error stopping self-healing: {e}")

    if hasattr(app.state, "memory_manager"):
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
    version="13.4.0-browser-automation",
    lifespan=lifespan,
)

# Configure Dynamic CORS
try:
    from api.dynamic_cors_handler import DynamicCORSMiddleware, AutoPortDiscovery

    # Add dynamic CORS middleware
    class DynamicCORSWrapper:
        def __init__(self, app):
            self.cors_handler = DynamicCORSMiddleware(app)

        async def __call__(self, scope, receive, send):
            if scope["type"] == "http":
                # Create request object from scope
                from starlette.requests import Request

                request = Request(scope, receive)

                async def call_next(request):
                    # Create response by calling the app
                    async def receive_wrapper():
                        return await receive()

                    async def send_wrapper(message):
                        pass

                    # Execute the app and capture response
                    await self.cors_handler.app(scope, receive, send)

                # Let the middleware handle it
                await self.cors_handler(request, call_next)
            else:
                # Non-HTTP, pass through
                await self.cors_handler.app(scope, receive, send)

    # For now, use standard CORS with dynamic configuration
    origins = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:3001,http://localhost:8000,http://localhost:8010",
    ).split(",")
    backend_port = os.getenv("BACKEND_PORT", "8000")
    if backend_port == "8010":
        origins.extend(["http://localhost:8010", "ws://localhost:8010"])

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins
        + ["http://127.0.0.1:3000", "http://127.0.0.1:8000", "http://127.0.0.1:8010"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    logger.info("✅ CORS configured with dynamic origins")

except Exception as e:
    # Fallback to static CORS if dynamic handler not available
    logger.warning(f"Dynamic CORS handler error: {e}, using static configuration")
    origins = os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:3001"
    ).split(",")
    backend_port = os.getenv("BACKEND_PORT", "8000")
    if backend_port == "8010":
        origins.extend(["http://localhost:8010", "ws://localhost:8010"])

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
    ml_audio_details = {}
    vision_status = {}

    # Check vision status manager
    if hasattr(app.state, "vision_status_manager"):
        vision_status = app.state.vision_status_manager.get_status()

    # Check vision component status
    if hasattr(app.state, "vision_analyzer"):
        try:
            # Check orchestrator status
            orchestrator = await app.state.vision_analyzer.get_orchestrator()
            if orchestrator:
                status = await orchestrator.get_system_status()
                vision_details["orchestrator"] = {
                    "enabled": True,
                    "mode": status["system_mode"],
                    "memory_usage_mb": status["memory_usage_mb"],
                    "active_components": sum(
                        1 for v in status["components"].values() if v == "active"
                    ),
                }
            else:
                vision_details["orchestrator"] = {"enabled": False}
        except:
            vision_details["orchestrator"] = {"enabled": False}

    # Check ML audio system status
    if hasattr(app.state, "ml_audio_state"):
        ml_state = app.state.ml_audio_state
        ml_audio_details = {
            "enabled": True,
            "active_streams": len(ml_state.active_streams),
            "total_processed": ml_state.total_processed,
            "uptime_hours": round(ml_state.get_uptime(), 2),
            "capabilities": ml_state.system_capabilities,
            "performance": ml_state.get_performance_metrics(),
            "quality_insights": ml_state.get_quality_insights(),
        }

    # Check Rust acceleration status
    rust_details = {}
    if hasattr(app.state, "rust_acceleration"):
        rust_config = app.state.rust_acceleration
        if rust_config.get("available"):
            rust_details = {
                "enabled": True,
                "components": rust_config.get("components", {}),
                "performance_boost": rust_config.get("performance_boost", {}),
                "memory_savings": rust_config.get("memory_savings", {}),
            }
        else:
            rust_details = {"enabled": False}

    # Check self-healing status
    self_healing_details = {}
    try:
        from vision.rust_self_healer import get_self_healer

        healer = get_self_healer()
        health_report = healer.get_health_report()
        self_healing_details = {
            "enabled": health_report.get("running", False),
            "fix_attempts": health_report.get("total_fix_attempts", 0),
            "success_rate": health_report.get("success_rate", 0.0),
            "last_successful_build": health_report.get("last_successful_build"),
        }
    except:
        self_healing_details = {"enabled": False}

    # Check voice unlock status
    voice_unlock_details = {}
    if hasattr(app.state, "voice_unlock") and app.state.voice_unlock.get("initialized"):
        try:
            from voice_unlock.services.mac_unlock_service import MacUnlockService

            # Get basic status without initializing service
            voice_unlock_details = {
                "enabled": True,
                "initialized": True,
                "api_available": True,
            }
        except:
            voice_unlock_details = {"enabled": False, "initialized": False}
    else:
        voice_unlock_details = {"enabled": False, "initialized": False}

    return {
        "status": "healthy",
        "mode": "optimized" if OPTIMIZE_STARTUP else "legacy",
        "parallel_imports": PARALLEL_IMPORTS,
        "lazy_models": LAZY_LOAD_MODELS,
        "components": {
            name: bool(comp) for name, comp in components.items() if comp is not None
        },
        "vision_status": vision_status,
        "vision_enhanced": vision_details,
        "ml_audio_system": ml_audio_details,
        "rust_acceleration": rust_details,
        "self_healing": self_healing_details,
        "voice_unlock": voice_unlock_details,
    }


@app.get("/autonomous/status")
async def autonomous_status():
    """Get autonomous orchestrator and mesh network status"""
    orchestrator_status = None
    mesh_status = None

    if hasattr(app.state, "orchestrator"):
        try:
            orchestrator_status = app.state.orchestrator.get_status()
        except Exception as e:
            orchestrator_status = {"error": str(e)}

    if hasattr(app.state, "mesh"):
        try:
            mesh_status = app.state.mesh.get_status()
        except Exception as e:
            mesh_status = {"error": str(e)}

    return {
        "autonomous_enabled": orchestrator_status is not None or mesh_status is not None,
        "orchestrator": orchestrator_status,
        "mesh": mesh_status
    }


@app.get("/autonomous/services")
async def autonomous_services():
    """Get list of all discovered services"""
    if not hasattr(app.state, "orchestrator"):
        return {"error": "Orchestrator not available"}

    try:
        return app.state.orchestrator.get_frontend_config()
    except Exception as e:
        return {"error": str(e)}


# Mount routers based on available components
def mount_routers():
    """Mount API routers based on loaded components"""
    import os  # Ensure os is available in this scope

    # Memory API
    memory = components.get("memory", {})
    if memory.get("available") and hasattr(app.state, "memory_manager"):
        memory_api_class = memory.get("api")
        if memory_api_class:
            memory_api = memory_api_class(app.state.memory_manager)
            app.include_router(memory_api.router, prefix="/memory", tags=["memory"])
            logger.info("✅ Memory API mounted")

    # Voice API
    voice = components.get("voice", {})
    if voice and voice.get("jarvis_available"):
        app.include_router(
            voice["jarvis_router"], prefix="/voice/jarvis", tags=["jarvis"]
        )
        logger.info("✅ JARVIS Voice API mounted")

    if voice and voice.get("enhanced_available"):
        app.include_router(
            voice["enhanced_router"], prefix="/voice/enhanced", tags=["voice"]
        )
        logger.info("✅ Enhanced Voice API mounted")

    # ML Model Status API
    ml = components.get("ml_models", {})
    if ml and ml.get("status_router"):
        app.include_router(ml["status_router"], prefix="/models", tags=["models"])
        logger.info("✅ Model Status API mounted")

    # Monitoring API
    monitoring = components.get("monitoring", {})
    if monitoring and monitoring.get("router"):
        app.include_router(
            monitoring["router"], prefix="/monitoring", tags=["monitoring"]
        )
        logger.info("✅ Monitoring API mounted")

    # Voice Unlock API
    voice_unlock = components.get("voice_unlock", {})
    if voice_unlock and voice_unlock.get("router"):
        app.include_router(voice_unlock["router"], tags=["voice_unlock"])
        logger.info("✅ Voice Unlock API mounted")
        if voice_unlock.get("initialized"):
            app.state.voice_unlock = voice_unlock
            logger.info("✅ Voice Unlock service ready")

    # Wake Word API
    wake_word = components.get("wake_word", {})
    if wake_word and wake_word.get("router"):
        app.include_router(wake_word["router"], tags=["wake_word"])
        logger.info("✅ Wake Word API mounted")
        if wake_word.get("initialized"):
            app.state.wake_word = wake_word
            logger.info("✅ Wake Word detection ready")
        else:
            logger.warning("⚠️ Wake Word API mounted but not initialized")

    # Rust API (if Rust components are available)
    if hasattr(app.state, "rust_acceleration") and app.state.rust_acceleration.get(
        "available"
    ):
        try:
            from api.rust_api import router as rust_router

            app.include_router(rust_router, prefix="/rust", tags=["rust"])
            logger.info("✅ Rust acceleration API mounted")
        except ImportError:
            logger.debug("Rust API not available")

    # Self-healing API
    try:
        from api.self_healing_api import router as self_healing_router

        app.include_router(
            self_healing_router, prefix="/self-healing", tags=["self-healing"]
        )
        logger.info("✅ Self-healing API mounted")
    except ImportError:
        logger.debug("Self-healing API not available")

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

    # Vision WebSocket endpoint at /vision/ws/vision
    try:
        from api.vision_ws_endpoint import (
            router as vision_ws_endpoint_router,
            set_vision_analyzer,
        )

        app.include_router(vision_ws_endpoint_router, tags=["vision"])

        # Set vision analyzer if available
        vision = components.get("vision", {})
        if vision and vision.get("analyzer"):
            set_vision_analyzer(vision["analyzer"])

        logger.info("✅ Vision WebSocket endpoint mounted at /vision/ws/vision")
    except ImportError as e:
        logger.warning(f"Could not import vision WebSocket endpoint: {e}")

    # ML Audio API (with built-in fallback) - Always mount regardless of WebSocket status
    try:
        from api.ml_audio_api import router as ml_audio_router

        app.include_router(ml_audio_router, tags=["ML Audio"])
        logger.info("✅ ML Audio API mounted")
    except ImportError as e:
        logger.error(f"Could not import ML Audio router: {e}")

    # Network Recovery API (kept separate as it's not WebSocket)
    try:
        from api.network_recovery_api import router as network_recovery_router

        app.include_router(network_recovery_router, tags=["Network Recovery"])
        logger.info("✅ Network Recovery API mounted")
    except ImportError as e:
        logger.warning(f"Could not import Network Recovery router: {e}")

    # ML Audio functionality is now included in the unified ml_audio_api.py

    # Auto Configuration API (for dynamic client configuration)
    try:
        from api.auto_config_endpoint import router as auto_config_router

        app.include_router(auto_config_router, tags=["Auto Configuration"])
        logger.info(
            "✅ Auto Configuration API mounted - clients can auto-discover settings"
        )
    except ImportError as e:
        logger.warning(f"Could not import Auto Config router: {e}")

    # Autonomous Service API (for zero-configuration mode)
    try:
        # Check if we should use memory-optimized version
        use_memory_optimized = (
            os.getenv("MEMORY_OPTIMIZED_MODE", "true").lower() == "true"
        )

        if use_memory_optimized:
            # Import memory-optimized orchestrator
            from backend.core.memory_optimized_orchestrator import (
                get_memory_optimized_orchestrator,
            )

            orchestrator = get_memory_optimized_orchestrator(
                memory_limit_mb=400
            )  # 400MB limit for orchestrator

            # Start it in background
            async def start_orchestrator():
                await orchestrator.start()
                logger.info("✅ Memory-optimized orchestrator started (400MB limit)")

            asyncio.create_task(start_orchestrator())
            logger.info("🚀 Using memory-optimized autonomous orchestrator")

        # Always mount the API router
        from api.autonomous_service_api import router as autonomous_router

        app.include_router(autonomous_router)
        logger.info("✅ Autonomous Service API mounted")
        logger.info("🤖 Zero-configuration mode enabled - services will auto-discover")

    except ImportError as e:
        logger.warning(f"Autonomous Service API not available: {e}")

    # Mount static files for auto-config script
    try:
        import os

        static_dir = os.path.join(os.path.dirname(__file__), "static")
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        logger.info(
            "✅ Static files mounted - auto-config script available at /static/jarvis-auto-config.js"
        )
    except Exception as e:
        logger.warning(f"Could not mount static files: {e}")


# Note: Startup tasks are now handled in the lifespan handler above


# Simple command endpoint for testing
@app.post("/api/command")
async def process_command(request: dict):
    """Simple command endpoint for testing"""
    command = request.get("command", "")

    # Use unified command processor if available
    try:
        from api.unified_command_processor import UnifiedCommandProcessor

        # Use enhanced Context Intelligence for screen lock/unlock
        USE_ENHANCED_CONTEXT = True

        if USE_ENHANCED_CONTEXT:
            try:
                from api.simple_context_handler_enhanced import (
                    wrap_with_enhanced_context,
                )

                processor = UnifiedCommandProcessor()
                context_handler = wrap_with_enhanced_context(processor)
                result = await context_handler.process_with_context(command)
            except ImportError as e:
                logger.error(f"Enhanced context handler not available: {e}")
                # Fallback to simple context
                try:
                    from api.simple_context_handler import wrap_with_simple_context

                    processor = UnifiedCommandProcessor()
                    context_handler = wrap_with_simple_context(processor)
                    result = await context_handler.process_with_context(command)
                except ImportError:
                    processor = UnifiedCommandProcessor()
                    result = await processor.process_command(command)
        else:
            # Use standard processor
            processor = UnifiedCommandProcessor()
            result = await processor.process_command(command)
        return result
    except Exception as e:
        logger.error(f"Command processing error: {e}")
        return {"error": str(e), "command": command}


# Basic test endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "JARVIS Backend (Optimized) is running",
        "version": "13.4.0-browser-automation",
        "proactive_vision_enabled": hasattr(app.state, "vision_analyzer"),
        "components": {
            name: bool(comp) for name, comp in components.items() if comp is not None
        },
    }


# Note: Main WebSocket endpoint is now handled by unified_websocket router at /ws
# This provides a single endpoint for all WebSocket communication


# ML Audio WebSocket compatibility endpoint
@app.websocket("/audio/ml/stream")
async def ml_audio_websocket_compat(websocket: WebSocket):
    """ML Audio WebSocket endpoint for backward compatibility with enhanced features"""
    await websocket.accept()
    logger.info(
        "ML Audio WebSocket connection (legacy endpoint) - providing enhanced compatibility"
    )

    try:
        # Import unified handler and datetime
        from api.unified_websocket import ws_manager, connection_capabilities
        from datetime import datetime
        import json

        # Get client info
        client_host = websocket.client.host if websocket.client else "unknown"
        client_id = f"ml_audio_{client_host}_{datetime.now().timestamp()}"

        # Send enhanced welcome message with system capabilities
        ml_state = getattr(app.state, "ml_audio_state", None)
        welcome_msg = {
            "type": "connection_established",
            "client_id": client_id,
            "server_time": datetime.now().isoformat(),
            "capabilities": ml_state.system_capabilities if ml_state else {},
            "recommended_config": (
                ml_state.get_client_recommendations(client_id, "")
                if ml_state
                else {"chunk_size": 512, "sample_rate": 16000, "format": "base64"}
            ),
            "migration_notice": {
                "message": "This endpoint provides full compatibility. For best performance, consider using /ws",
                "new_endpoint": "/ws",
                "benefits": [
                    "unified_interface",
                    "better_performance",
                    "more_features",
                ],
            },
        }
        await websocket.send_json(welcome_msg)

        # Add to unified connections with ML audio context
        ws_manager.connections[client_id] = websocket
        connection_capabilities[client_id] = {"ml_audio_stream", "legacy_client"}

        # Track stream if ML state available
        if ml_state:
            ml_state.active_streams[client_id] = {
                "started_at": datetime.now(),
                "processed_chunks": 0,
                "total_bytes": 0,
                "quality_scores": [],
                "websocket": True,
            }

        while True:
            # Receive message
            data = await websocket.receive_json()

            # Convert to unified format
            unified_msg = {
                "type": "ml_audio_stream",
                "audio_data": data.get("audio_data", data.get("data", "")),
                "sample_rate": data.get("sample_rate", 16000),
                "format": data.get("format", "base64"),
            }

            # Handle through unified manager
            response = await ws_manager.handle_message(client_id, unified_msg)

            # Send response
            await websocket.send_json(response)

    except WebSocketDisconnect:
        logger.info("ML Audio WebSocket disconnected (legacy)")
        if client_id in ws_manager.connections:
            ws_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"ML Audio WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass


# Audio endpoints for frontend compatibility
@app.post("/audio/speak")
async def audio_speak_post(request: dict):
    """Forward audio speak requests to JARVIS voice API"""
    from fastapi import HTTPException

    voice = components.get("voice", {})
    jarvis_api = voice.get("jarvis_api")

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
    parser = argparse.ArgumentParser(description="JARVIS Backend Server")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("BACKEND_PORT", "8000")),
        help="Port to run the server on",
    )
    args = parser.parse_args()

    # Use optimized settings if enabled
    if OPTIMIZE_STARTUP:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=args.port,
            log_level="info",
            access_log=False,  # Disable access logs for performance
            loop=(
                "uvloop" if sys.platform != "win32" else "asyncio"
            ),  # Use uvloop on Unix
        )
    else:
        uvicorn.run(app, host="0.0.0.0", port=args.port)
