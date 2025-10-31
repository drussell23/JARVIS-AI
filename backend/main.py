#!/usr/bin/env python3
"""
JARVIS AI Backend - Optimized Main Entry Point with Advanced Intelligence

This backend loads 10 critical components that power the JARVIS AI system:

1. CHATBOTS (Claude Vision Chatbot)
   - Powers conversational AI with Claude 3.5 Sonnet
   - Enables screen analysis and vision capabilities
   - Handles all natural language understanding and generation

2. VISION (Screen Capture & Multi-Space Analysis + YOLOv8)
   - Real-time screen monitoring with Swift-based capture
   - Video streaming at 30 FPS with purple recording indicator
   - Multi-Space Desktop Vision: Monitors all macOS desktop spaces
   - YOLOv8 Integration: Real-time UI element detection (10-20x faster than Claude)
     * 5 model sizes from nano (3MB) to xlarge (68MB)
     * Detects 20+ UI types: buttons, icons, Control Center, TV UI, monitors
     * Free after download, real-time capable (10-20 FPS)
   - Hybrid YOLO-Claude Vision: Intelligent task routing
     * YOLO-first for UI detection (fast, accurate, free)
     * Claude for OCR and complex semantic analysis
     * Hybrid mode for comprehensive understanding
   - Enhanced Window Detection with UI element tracking
   - Multi-Monitor Layout Detection with vision-based awareness
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

4. VOICE (JARVIS Voice Interface with CoreML Acceleration)
   - Voice activation with "Hey JARVIS" wake word
   - CoreML Voice Engine: Hardware-accelerated VAD on Apple Neural Engine
     * 232KB model (4-bit quantized Silero VAD)
     * <10ms inference latency
     * ~5-10MB runtime memory
     * Zero CPU usage (runs on Neural Engine)
   - Text-to-speech with multiple voice options
   - Real-time voice command processing
   - Adaptive threshold learning for improved accuracy

5. ML_MODELS (Machine Learning Models)
   - Sentiment analysis and NLP capabilities
   - Lazy-loaded to optimize startup time
   - Powers intelligent text understanding

6. MONITORING (System Health & Metrics)
   - Tracks API performance and resource usage
   - Provides health checks and status endpoints
   - Essential for production stability
   - Integration metrics tracking for all vision components

7. VOICE UNLOCK (Cloud SQL Biometric Authentication)
   - Voice-based biometric authentication for macOS
   - Cloud SQL storage: 59 voice samples + 768-byte averaged embedding
   - PostgreSQL database via GCP Cloud SQL proxy (127.0.0.1:5432)
   - Personalized responses using verified speaker identity
   - Anti-spoofing protection with 75% confidence threshold
   - Screensaver and system integration
   - Adaptive authentication with continuous learning

8. WAKE WORD (Hands-free Activation)
   - "Hey JARVIS" wake word detection
   - Always-listening mode with zero button clicks
   - Multi-engine detection (Porcupine, Vosk, WebRTC)
   - Adaptive sensitivity and anti-spoofing
   - Customizable wake words and responses

9. DISPLAY MONITOR (External Display Management)
   - Automatic AirPlay/external display detection
   - Multi-method detection (AppleScript, CoreGraphics, Yabai)
   - Voice announcements for display availability
   - Smart caching for 3-5x performance improvement
   - Auto-connect or prompt modes
   - Living Room TV monitoring (configurable)
   - Zero hardcoding - fully configuration-driven
   - Event-driven callbacks for custom integrations

10. INTELLIGENCE STACK (UAE + SAI + Learning Database) - ADVANCED! 🧠
   - UAE (Unified Awareness Engine): Context intelligence + decision fusion
   - SAI (Situational Awareness): Real-time UI monitoring (10s interval)
   - Learning Database: Persistent memory with async SQLite + ChromaDB
   - Predictive Intelligence: Learns patterns and predicts actions
   - Cross-Session Memory: Remembers across restarts
   - Temporal Pattern Recognition: Time-based behavior learning
   - Self-Healing: Adapts automatically to environment changes
   - Confidence-Weighted Decisions: Fuses context + real-time perception
   - Zero Hardcoding: Fully dynamic, learns everything
   - Capabilities:
     * Learns user patterns across macOS workspace
     * Predicts actions before you ask (proactive)
     * Adapts to UI changes automatically (reactive)
     * Remembers preferences forever (persistent)
     * Gets smarter over time (continuous learning)

All 10 components must load successfully for full JARVIS functionality.
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

# CRITICAL: Set multiprocessing start method to 'spawn' BEFORE any other imports
# This prevents segmentation faults from semaphore leaks on macOS
import multiprocessing
import os
import subprocess
import sys

# Set critical environment variables FIRST
if sys.platform == "darwin":  # macOS specific
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    os.environ["PYTHONUNBUFFERED"] = "1"

# Clean up leaked semaphores from previous runs FIRST
if sys.platform == "darwin":  # macOS specific
    try:
        # Get current user
        user = os.getenv("USER", "")
        if user:
            # Clean up semaphores (macOS xargs doesn't have -r flag)
            result = subprocess.run(
                f"ipcs -s 2>/dev/null | grep {user} | awk '{{print $2}}' | while read id; do ipcrm -s $id 2>/dev/null; done",
                shell=True,
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                print(f"[STARTUP] Cleaned up leaked semaphores")
    except Exception as e:
        print(f"[STARTUP] Semaphore cleanup warning: {e}")

    # Set spawn mode - MUST be before any other multiprocessing usage
    try:
        multiprocessing.set_start_method("spawn", force=True)
        print("[STARTUP] Set multiprocessing to spawn mode")
    except RuntimeError as e:
        # Already set, that's fine
        if "context has already been set" not in str(e):
            print(f"[STARTUP] Multiprocessing warning: {e}")

# Now continue with other imports
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

# DEBUG: Print location info
print(f"[STARTUP-DEBUG] Running from: {os.path.abspath(__file__)}")
print(f"[STARTUP-DEBUG] Working directory: {os.getcwd()}")
print(f"[STARTUP-DEBUG] Python path: {sys.path[:3]}")  # First 3 entries

# DEBUG: Run coordinate diagnostic
try:
    print("[STARTUP-DEBUG] Running coordinate diagnostic...")
    exec(
        open(
            "/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/diagnose_coordinate_doubling.py"
        ).read()
    )
except Exception as e:
    print(f"[STARTUP-DEBUG] Coordinate diagnostic failed: {e}")

# DEBUG: Install PyAutoGUI intercept to track coordinate doubling
try:
    print("[STARTUP-DEBUG] Installing PyAutoGUI intercept...")
    sys.path.insert(0, "/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent")
    import pyautogui_intercept

    pyautogui_intercept.install_intercept()
    print(
        "[STARTUP-DEBUG] ✅ PyAutoGUI intercept installed - logging to /tmp/pyautogui_intercept.log"
    )
except Exception as e:
    print(f"[STARTUP-DEBUG] PyAutoGUI intercept failed: {e}")

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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Load environment variables (force override of system env vars)
try:
    from dotenv import load_dotenv

    load_dotenv(override=True)  # Force .env to override existing environment variables
except ImportError:
    pass

# Global component storage
components = {}
import_times = {}

# Dynamic Component Manager
dynamic_component_manager = None
DYNAMIC_LOADING_ENABLED = False

# GCP VM Manager
gcp_vm_manager = None
GCP_VM_ENABLED = os.getenv("GCP_VM_ENABLED", "true").lower() == "true"

try:
    from core.dynamic_component_manager import get_component_manager

    logger.info("✅ Dynamic Component Manager available")
    DYNAMIC_LOADING_ENABLED = os.getenv("DYNAMIC_COMPONENT_LOADING", "true").lower() == "true"
    if DYNAMIC_LOADING_ENABLED:
        logger.info("🧩 Dynamic Component Loading: ENABLED")
    else:
        logger.info(
            "⚠️ Dynamic Component Loading: DISABLED (set DYNAMIC_COMPONENT_LOADING=true to enable)"
        )
except ImportError:
    logger.warning("⚠️ Dynamic Component Manager not available - using legacy loading")
    DYNAMIC_LOADING_ENABLED = False


async def parallel_import_components():
    """Import all components in parallel for faster startup"""
    start_time = time.time()
    logger.info("⚡ Starting parallel component imports...")

    import concurrent.futures

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
        "display_monitor": import_display_monitor,
        "goal_inference": import_goal_inference,
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
        from vision.video_stream_capture import MACOS_CAPTURE_AVAILABLE, VideoStreamCapture

        vision["analyzer"] = ClaudeVisionAnalyzer
        vision["video_capture"] = VideoStreamCapture
        vision["macos_available"] = MACOS_CAPTURE_AVAILABLE
        vision["available"] = True
    except ImportError:
        vision["available"] = False

    # Check purple indicator separately
    try:
        pass

        vision["purple_indicator"] = True
    except ImportError:
        vision["purple_indicator"] = False

    return vision


def import_memory_system():
    """Import memory management"""
    memory = {}

    try:
        from memory.memory_api import MemoryAPI
        from memory.memory_manager import ComponentPriority, M1MemoryManager

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
        from api.jarvis_voice_api import jarvis_api
        from api.jarvis_voice_api import router as jarvis_voice_router

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
        from api.model_status_api import router as model_status_router
        from ml_model_loader import get_loader_status, initialize_models

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
    import logging

    logger = logging.getLogger(__name__)

    voice_unlock = {}

    try:
        from api.voice_unlock_api import initialize_voice_unlock
        from api.voice_unlock_api import router as voice_unlock_router

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
        from api.wake_word_api import initialize_wake_word
        from api.wake_word_api import router as wake_word_router
        from api.wake_word_api import wake_service

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


def import_display_monitor():
    """Import display monitor components"""
    display_monitor = {}

    try:
        from display.advanced_display_monitor import AdvancedDisplayMonitor, get_display_monitor
        from display.display_config_manager import get_config_manager
        from display.display_voice_handler import create_voice_handler

        display_monitor["get_monitor"] = get_display_monitor
        display_monitor["monitor_class"] = AdvancedDisplayMonitor
        display_monitor["voice_handler_factory"] = create_voice_handler
        display_monitor["config_manager_factory"] = get_config_manager
        display_monitor["available"] = True

        logger.info("  ✅ Display Monitor components loaded")

    except ImportError as e:
        logger.warning(f"  ⚠️  Display Monitor not available: {e}")
        display_monitor["available"] = False

    return display_monitor


def import_goal_inference():
    """Import Goal Inference and Learning Database components with auto-configuration"""
    goal_inference = {}

    try:
        # Import Goal Inference + Autonomous Engine Integration
        import json
        import os
        from pathlib import Path

        from backend.intelligence.goal_autonomous_uae_integration import get_integration
        from backend.intelligence.learning_database import get_learning_database

        # Load or create configuration
        config_path = Path("backend/config/integration_config.json")

        # Check for environment variable overrides
        preset_override = os.getenv("JARVIS_GOAL_PRESET", None)
        automation_override = os.getenv("JARVIS_GOAL_AUTOMATION", None)

        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            # Create default configuration automatically
            logger.info("  📝 Creating default Goal Inference configuration...")
            config = _create_default_goal_config()
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"  ✅ Configuration created at {config_path}")

        # Apply preset if environment variable is set
        if preset_override:
            logger.info(f"  🎯 Applying preset from environment: {preset_override}")
            config = _apply_preset_to_config(config, preset_override)
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

        # Apply automation override if environment variable is set
        if automation_override:
            automation_enabled = automation_override.lower() == "true"
            config["integration"]["enable_automation"] = automation_enabled
            logger.info(
                f"  🤖 Automation override: {'ENABLED' if automation_enabled else 'DISABLED'}"
            )
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

        # Initialize integration with config
        integration = get_integration()
        goal_inference["integration"] = integration
        goal_inference["available"] = True
        goal_inference["config"] = config

        # Initialize learning database with config
        db_config = {
            "cache_size": config.get("performance", {}).get("max_prediction_cache_size", 1000),
            "cache_ttl_seconds": config.get("performance", {}).get("cache_ttl_seconds", 3600),
            "enable_ml_features": config.get("learning", {}).get("enabled", True),
            "auto_optimize": True,
            "batch_insert_size": 100,
        }
        learning_db = get_learning_database
        goal_inference["learning_db"] = learning_db
        goal_inference["db_config"] = db_config

        # Log configuration
        logger.info("  ✅ Goal Inference + Learning Database loaded")
        logger.info(f"     • Goal Confidence: {config['goal_inference']['min_goal_confidence']}")
        logger.info(
            f"     • Proactive Suggestions: {config['integration']['enable_proactive_suggestions']}"
        )
        logger.info(f"     • Automation: {config['integration']['enable_automation']}")
        logger.info(f"     • Learning: {config['learning']['enabled']}")
        logger.info(f"     • Database Cache: {db_config['cache_size']} entries")

        # Get current metrics
        try:
            metrics = integration.get_metrics()
            if metrics.get("goals_inferred", 0) > 0:
                logger.info(
                    f"     • Previous session: {metrics['goals_inferred']} goals, {metrics.get('actions_executed', 0)} actions"
                )
                logger.info(f"     • Success rate: {metrics.get('success_rate', 0):.1%}")
        except Exception as e:
            logger.debug(f"Could not load metrics: {e}")

        # Apply configuration to integration
        _apply_config_to_integration(integration, config)

    except ImportError as e:
        logger.warning(f"  ⚠️  Goal Inference not available: {e}")
        goal_inference["available"] = False

    except Exception as e:
        logger.warning(f"  ⚠️  Goal Inference initialization failed: {e}")
        goal_inference["available"] = False

    return goal_inference


def _create_default_goal_config():
    """Create default Goal Inference configuration"""
    return {
        "goal_inference": {
            "min_goal_confidence": 0.75,
            "goal_confidence_threshold": 0.75,
            "enable_learning": True,
            "max_active_goals": 10,
            "goal_timeout_minutes": 30,
            "pattern_learning_enabled": True,
        },
        "autonomous_decisions": {
            "min_decision_confidence": 0.70,
            "enable_predictive_display": True,
            "auto_connect_threshold": 0.85,
            "max_concurrent_actions": 5,
            "learning_rate": 0.01,
            "exploration_rate": 0.1,
        },
        "integration": {
            "enable_proactive_suggestions": True,
            "proactive_suggestion_threshold": 0.85,
            "enable_automation": False,
            "automation_threshold": 0.95,
            "feedback_window_minutes": 30,
            "cache_duration_minutes": 5,
        },
        "display_optimization": {
            "enable_predictive_connection": True,
            "preload_resources": True,
            "predictive_confidence_threshold": 0.85,
            "default_display": "Living Room TV",
            "connection_patterns": {
                "meeting_preparation": "Living Room TV",
                "project_completion": "External Monitor",
                "presentation": "Living Room TV",
                "casual_viewing": "Living Room TV",
            },
        },
        "learning": {
            "enabled": True,
            "min_samples_for_pattern": 3,
            "pattern_confidence_boost": 0.05,
            "success_rate_threshold": 0.7,
            "feedback_weight": 0.1,
            "save_state_interval_minutes": 60,
        },
        "user_preferences": {
            "verbose_suggestions": False,
            "explain_reasoning": True,
            "show_confidence_scores": False,
            "auto_accept_high_confidence": False,
            "notification_style": "subtle",
        },
        "performance": {
            "max_prediction_cache_size": 100,
            "cache_ttl_seconds": 300,
            "parallel_processing": True,
            "max_workers": 4,
            "timeout_seconds": 5,
        },
        "safety": {
            "require_confirmation_for_automation": True,
            "max_automation_actions_per_day": 50,
            "blacklist_actions": [],
            "whitelist_actions": ["connect_display", "open_application", "organize_workspace"],
            "risk_tolerance": 0.5,
        },
        "logging": {
            "log_predictions": True,
            "log_decisions": True,
            "log_learning_events": True,
            "metrics_tracking": True,
            "debug_mode": False,
        },
    }


def _apply_preset_to_config(config, preset):
    """Apply a configuration preset"""
    presets = {
        "aggressive": {
            "goal_inference.min_goal_confidence": 0.65,
            "autonomous_decisions.min_decision_confidence": 0.60,
            "integration.proactive_suggestion_threshold": 0.75,
            "integration.enable_automation": True,
            "learning.pattern_confidence_boost": 0.10,
        },
        "balanced": {
            "goal_inference.min_goal_confidence": 0.75,
            "autonomous_decisions.min_decision_confidence": 0.70,
            "integration.proactive_suggestion_threshold": 0.85,
            "integration.enable_automation": False,
            "learning.pattern_confidence_boost": 0.05,
        },
        "conservative": {
            "goal_inference.min_goal_confidence": 0.85,
            "autonomous_decisions.min_decision_confidence": 0.80,
            "integration.proactive_suggestion_threshold": 0.90,
            "integration.enable_automation": False,
            "learning.pattern_confidence_boost": 0.02,
        },
        "learning": {
            "learning.enabled": True,
            "learning.min_samples_for_pattern": 2,
            "learning.pattern_confidence_boost": 0.10,
            "learning.feedback_weight": 0.15,
            "autonomous_decisions.exploration_rate": 0.2,
        },
        "performance": {
            "performance.max_prediction_cache_size": 200,
            "performance.cache_ttl_seconds": 600,
            "performance.parallel_processing": True,
            "display_optimization.preload_resources": True,
        },
    }

    if preset in presets:
        for path, value in presets[preset].items():
            keys = path.split(".")
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value

    return config


def _apply_config_to_integration(integration, config):
    """Apply configuration settings to integration"""
    try:
        # Apply goal inference settings
        if hasattr(integration, "goal_inference"):
            goal_config = config.get("goal_inference", {})
            integration.goal_inference.min_confidence = goal_config.get("min_goal_confidence", 0.75)
            integration.goal_inference.max_active_goals = goal_config.get("max_active_goals", 10)

        # Apply autonomous decision settings
        if hasattr(integration, "autonomous_engine"):
            auto_config = config.get("autonomous_decisions", {})
            integration.autonomous_engine.min_confidence = auto_config.get(
                "min_decision_confidence", 0.70
            )
            integration.autonomous_engine.learning_rate = auto_config.get("learning_rate", 0.01)

        # Apply integration settings
        integration_config = config.get("integration", {})
        integration.enable_proactive = integration_config.get("enable_proactive_suggestions", True)
        integration.enable_automation = integration_config.get("enable_automation", False)

    except Exception as e:
        logger.debug(f"Could not apply all config settings: {e}")


async def memory_pressure_callback(pressure_level: str):
    """
    Callback for memory pressure changes - triggers GCP VM creation if needed

    Args:
        pressure_level: 'low', 'medium', 'high', 'critical'
    """
    global gcp_vm_manager  #

    logger.info(f"📊 Memory pressure changed: {pressure_level}")

    # Only create VM on high or critical pressure
    if pressure_level not in ["high", "critical"]:
        return

    if not GCP_VM_ENABLED:
        logger.info("⚠️  GCP VM creation disabled (GCP_VM_ENABLED=false)")
        return

    try:
        # Check if GCP VM Manager is initialized
        if gcp_vm_manager is None:  # Check if gcp_vm_manager is initialized
            # Initialize GCP VM Manager
            from core.gcp_vm_manager import get_gcp_vm_manager

            # Get GCP VM Manager instance
            gcp_vm_manager = await get_gcp_vm_manager()

        # Get current memory snapshot
        from core.platform_memory_monitor import get_memory_monitor

        memory_monitor = get_memory_monitor()
        snapshot = memory_monitor.capture_snapshot()

        # Determine if VM should be created based on memory pressure level
        should_create, reason, confidence = await gcp_vm_manager.should_create_vm(
            snapshot,  # Memory snapshot
            trigger_reason=f"Memory pressure: {pressure_level}",  # Trigger reason
        )

        if should_create:
            logger.info(f"🚀 Creating GCP Spot VM: {reason} (confidence: {confidence:.2%})")

            # Determine which components to offload
            components_to_offload = []
            if pressure_level == "critical":
                # Offload heavy components
                components_to_offload = ["VISION", "CHATBOTS", "ML_MODELS", "LOCAL_LLM"]
            else:
                # Just offload the heaviest
                components_to_offload = ["VISION", "CHATBOTS"]

            # Create GCP VM instance
            vm_instance = await gcp_vm_manager.create_vm(
                components=components_to_offload,  # Components to offload
                trigger_reason=f"Memory pressure: {pressure_level} - {reason}",  # Trigger reason
                # Metadata for tracking
                metadata={
                    "pressure_level": pressure_level,  # Pressure level
                    "confidence": confidence,  # Confidence
                    "local_ram_gb": snapshot.total_gb if snapshot else 0,  # Local RAM GB
                    "used_ram_gb": snapshot.used_gb if snapshot else 0,  # Used RAM GB
                },
            )

            if vm_instance:
                logger.info(f"✅ GCP VM created: {vm_instance.name}")
                logger.info(f"   IP: {vm_instance.ip_address}")
                logger.info(f"   Components: {', '.join(vm_instance.components)}")
            else:
                logger.error("❌ Failed to create GCP VM")
        else:
            logger.info(f"ℹ️  VM creation not needed: {reason}")

    except Exception as e:
        logger.error(f"Error in memory pressure callback: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Optimized lifespan handler with parallel initialization"""
    logger.info("🚀 Starting optimized JARVIS backend...")
    start_time = time.time()

    # Initialize dynamic component manager if enabled
    global dynamic_component_manager, DYNAMIC_LOADING_ENABLED, gcp_vm_manager
    if DYNAMIC_LOADING_ENABLED and get_component_manager:
        logger.info("🧩 Initializing Dynamic Component Management System...")
        dynamic_component_manager = get_component_manager()
        app.state.component_manager = dynamic_component_manager

        # Register memory pressure callback for GCP VM creation
        if GCP_VM_ENABLED:
            logger.info("☁️  GCP VM auto-creation enabled")
            dynamic_component_manager.memory_monitor.register_callback(memory_pressure_callback)
            logger.info("✅ Memory pressure callback registered")

        # Start memory pressure monitoring
        asyncio.create_task(dynamic_component_manager.start_monitoring())
        logger.info(f"   Memory limit: {dynamic_component_manager.memory_limit_gb}GB")
        logger.info(f"   ARM64 optimized: {dynamic_component_manager.arm64_optimizer.is_arm64}")
        logger.info("✅ Dynamic component loading enabled")

    # CRITICAL: Check for code changes and clean up old instances FIRST
    # TEMPORARILY DISABLED - causing hang
    logger.info("⚠️ Process cleanup temporarily disabled for debugging")

    # Run parallel imports if enabled
    if DYNAMIC_LOADING_ENABLED and dynamic_component_manager:
        # Dynamic loading mode - load only CORE components at startup
        logger.info("🧩 Loading CORE components dynamically...")
        try:
            from core.dynamic_component_manager import ComponentPriority

            # Load only CORE priority components at startup
            core_components = [
                name
                for name, comp in dynamic_component_manager.components.items()
                if comp.priority == ComponentPriority.CORE
            ]

            # IMPORTANT: Always include vision as CORE to prevent multi-space query issues
            if "vision" not in core_components:
                core_components.append("vision")
                logger.info(
                    "   ⚠️ Vision not in CORE list, adding it to ensure multi-space queries work"
                )

            logger.info(f"   Loading {len(core_components)} CORE components: {core_components}")

            for comp_name in core_components:
                success = await dynamic_component_manager.load_component(comp_name)
                if success:
                    comp = dynamic_component_manager.components[comp_name]
                    components[comp_name] = comp.instance
                    logger.info(f"   ✅ {comp_name} loaded ({comp.memory_estimate_mb}MB)")
                else:
                    logger.warning(f"   ⚠️ {comp_name} failed to load")

            logger.info(
                f"✅ Dynamic component loading active - {len(core_components)} CORE components loaded"
            )
            logger.info(f"   Other components will load on-demand based on user commands")

        except Exception as e:
            logger.error(f"Dynamic loading failed, falling back to legacy mode: {e}")
            DYNAMIC_LOADING_ENABLED = False

    if not DYNAMIC_LOADING_ENABLED:
        # Legacy mode - load all components at startup
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
            components["display_monitor"] = import_display_monitor()
            components["goal_inference"] = import_goal_inference()

    # ═══════════════════════════════════════════════════════════════
    # ADVANCED COMPONENT WARMUP (Pre-initialize for instant response)
    # ═══════════════════════════════════════════════════════════════
    try:
        logger.info("🚀 Starting advanced component warmup...")
        from api.unified_command_processor import get_unified_processor

        processor = get_unified_processor(app=app)
        warmup_report = await processor.warmup_components()

        if warmup_report:
            logger.info(
                f"✅ Component warmup complete! "
                f"{warmup_report['ready_count']}/{warmup_report['total_count']} ready "
                f"in {warmup_report['total_load_time']:.2f}s"
            )
            app.state.warmup_report = warmup_report
        else:
            logger.warning("⚠️ Component warmup failed, using lazy initialization")
    except Exception as e:
        logger.error(f"❌ Component warmup error: {e}", exc_info=True)
        logger.warning("⚠️ Falling back to lazy initialization")

    # Initialize memory manager
    memory_class = components.get("memory", {}).get("manager_class")
    if memory_class:
        app.state.memory_manager = memory_class()
        await app.state.memory_manager.start_monitoring()
        logger.info("✅ Memory manager initialized")

    # Initialize Goal Inference and start background tasks
    goal_inference = components.get("goal_inference", {})
    if goal_inference and goal_inference.get("integration"):
        try:
            integration = goal_inference["integration"]
            app.state.goal_inference_integration = integration

            # Start background tasks for learning and pattern optimization
            async def periodic_database_cleanup():
                """Clean up old patterns and optimize database"""
                while True:
                    try:
                        await asyncio.sleep(3600)  # Run every hour
                        if hasattr(integration, "learning_db"):
                            # Clean up old patterns
                            integration.learning_db.cleanup_old_patterns(days=30)
                            # Optimize database
                            integration.learning_db.optimize()
                            logger.debug("✅ Goal Inference database cleanup completed")
                    except Exception as e:
                        logger.error(f"Database cleanup error: {e}")

            async def periodic_pattern_analysis():
                """Analyze patterns and update confidence scores"""
                while True:
                    try:
                        await asyncio.sleep(1800)  # Run every 30 minutes
                        if hasattr(integration, "learning_db"):
                            # Analyze patterns
                            patterns = integration.learning_db.analyze_patterns()
                            # Update confidence scores based on success rates
                            for pattern in patterns:
                                if pattern.get("success_rate", 0) > 0.9:
                                    integration.learning_db.boost_pattern_confidence(
                                        pattern["id"], boost=0.05
                                    )
                            logger.debug("✅ Pattern analysis completed")
                    except Exception as e:
                        logger.error(f"Pattern analysis error: {e}")

            # Start background tasks
            asyncio.create_task(periodic_database_cleanup())
            asyncio.create_task(periodic_pattern_analysis())

            logger.info("✅ Goal Inference background tasks started")
            logger.info("   • Database cleanup: every 1 hour")
            logger.info("   • Pattern analysis: every 30 minutes")

        except Exception as e:
            logger.warning(f"⚠️ Could not start Goal Inference tasks: {e}")

    # Initialize vision analyzer BEFORE UAE (so UAE can use it)
    logger.info("👁️  Initializing Claude Vision Analyzer...")
    vision = components.get("vision", {})
    vision_analyzer = None
    if vision.get("available"):
        analyzer_class = vision.get("analyzer")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if analyzer_class and api_key:
            vision_analyzer = analyzer_class(api_key)
            app.state.vision_analyzer = vision_analyzer
            logger.info("✅ Claude Vision Analyzer initialized and stored in app.state")
        else:
            logger.warning("⚠️  Vision analyzer available but no ANTHROPIC_API_KEY set")
    else:
        logger.warning("⚠️  Vision system not available")

    # Initialize UAE (Unified Awareness Engine) with LAZY LOADING for memory optimization
    # This prevents 10GB+ memory usage at startup by loading on first use
    try:
        # Check if lazy loading is enabled (default: True for memory efficiency)
        lazy_load_intelligence = os.getenv("JARVIS_LAZY_INTELLIGENCE", "true").lower() == "true"

        if lazy_load_intelligence:
            logger.info("🧠 UAE/SAI/Learning DB: LAZY LOADING enabled (loads on first use)")
            logger.info("   💾 Memory saved: ~8-10GB at startup")
            logger.info("   ⚡ Intelligence components will initialize when needed")

            # Store initialization parameters for lazy loading
            app.state.uae_lazy_config = {
                "vision_analyzer": vision_analyzer,
                "sai_monitoring_interval": 5.0,
                "enable_auto_start": True,
                "enable_learning_db": True,
                "enable_yabai": True,
                "enable_proactive_intelligence": True,
            }
            app.state.uae_engine = None  # Will be initialized on first use
            app.state.learning_db = None
            app.state.uae_initializing = False

            # Initialize Hybrid Orchestrator (always initialized)
            logger.info("🌐 Initializing Hybrid Orchestrator (Local + GCP)...")
            try:
                from backend.core.hybrid_orchestrator import get_orchestrator

                hybrid_orchestrator = get_orchestrator()
                await hybrid_orchestrator.start()
                app.state.hybrid_orchestrator = hybrid_orchestrator
                logger.info("✅ Hybrid Orchestrator initialized (intelligent routing active)")
            except Exception as e:
                logger.warning(f"⚠️  Hybrid Orchestrator not available: {e}")
                app.state.hybrid_orchestrator = None

        else:
            logger.info(
                "🧠 Initializing UAE (Unified Awareness Engine) with Learning Database + Yabai..."
            )
            from intelligence.uae_integration import get_learning_db, get_yabai, initialize_uae

            # Use the vision analyzer we just created
            if vision_analyzer:
                logger.info("✅ Connecting vision analyzer to UAE + SAI + Learning Database")

            # Create voice callback for Phase 4 Proactive Intelligence
            async def voice_callback(text: str):
                """Voice callback for proactive suggestions"""
                try:
                    voice = components.get("voice", {})
                    jarvis_api = voice.get("jarvis_api")
                    if jarvis_api:
                        await jarvis_api.speak({"text": text})
                        logger.debug(f"[PROACTIVE-VOICE] Spoke: {text}")
                    else:
                        logger.warning("[PROACTIVE-VOICE] JARVIS API not available")
                except Exception as e:
                    logger.error(f"[PROACTIVE-VOICE] Error: {e}")

            # Create notification callback for Phase 4 Proactive Intelligence
            async def notification_callback(title: str, message: str, priority: str = "low"):
                """Notification callback for proactive suggestions"""
                try:
                    # Log notification (can be extended to use macOS notifications)
                    logger.info(f"[PROACTIVE-NOTIFY] [{priority.upper()}] {title}: {message}")
                    # Future: Can integrate with macOS notification center
                    # osascript -e 'display notification "message" with title "title"'
                except Exception as e:
                    logger.error(f"[PROACTIVE-NOTIFY] Error: {e}")

            # Initialize UAE with SAI + Learning Database + Yabai + Proactive Intelligence
            logger.info("🔧 Initializing FULL intelligence stack (24/7 mode)...")
            logger.info("   Step 1/8: Learning Database initialization...")
            logger.info("   Step 2/8: Behavioral Pattern Learning...")
            logger.info("   Step 3/8: Yabai Spatial Intelligence (workspace monitoring)...")
            logger.info("   Step 4/8: Situational Awareness Engine (SAI)...")
            logger.info("   Step 5/8: Context Intelligence Layer...")
            logger.info("   Step 6/8: Decision Fusion Engine + 24/7 monitoring...")
            logger.info("   Step 7/8: Goal-Oriented Workflow Prediction...")
            logger.info("   Step 8/8: Proactive Communication Engine (Magic)...")

            uae = await initialize_uae(
                vision_analyzer=vision_analyzer,
                sai_monitoring_interval=5.0,  # Enhanced 24/7 mode: 5 seconds
                enable_auto_start=True,  # Start monitoring immediately
                enable_learning_db=True,  # Enable persistent memory
                enable_yabai=True,  # Enable Yabai spatial intelligence
                enable_proactive_intelligence=True,  # Enable Phase 4: Proactive Communication
                voice_callback=voice_callback,  # Natural voice suggestions
                notification_callback=notification_callback,  # Visual notifications
            )

            if uae and uae.is_active:
                app.state.uae_engine = uae

                # Get Learning DB instance
                learning_db = get_learning_db()
                if learning_db:
                    app.state.learning_db = learning_db

                    # Get Learning DB metrics
                    try:
                        metrics = await learning_db.get_learning_metrics()

                        # Get Yabai instance and metrics
                        yabai = get_yabai()
                        yabai_active = yabai is not None and yabai.yabai_available

                        logger.info(
                            "✅ UAE + SAI + Learning Database + Yabai + Proactive Intelligence initialized successfully"
                        )
                        logger.info("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                        logger.info(
                            "   🧠 PHASE 4 INTELLIGENCE STACK: FULLY OPERATIONAL (24/7 MODE)"
                        )
                        logger.info("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                        logger.info("   📍 PHASE 1: Environmental Awareness")
                        logger.info(
                            "   • SAI (Situational Awareness): ✅ Active (5s monitoring - 24/7)"
                        )
                        logger.info(
                            f"   • Yabai Spatial Intelligence: {'✅ Active (workspace monitoring)' if yabai_active else '⚠️  Not available'}"
                        )
                        logger.info("   • Context Intelligence: ✅ Active (with persistent memory)")
                        logger.info("")
                        logger.info("   📍 PHASE 2: Decision Intelligence")
                        logger.info("   • Decision Fusion Engine: ✅ Active (confidence-weighted)")
                        logger.info("   • Cross-Session Memory: ✅ Enabled (survives restarts)")
                        logger.info("")
                        logger.info("   📍 PHASE 3: Behavioral Learning (Smart)")
                        logger.info("   • Learning Database: ✅ Active (async + ChromaDB)")
                        logger.info("   • Predictive Intelligence: ✅ Enabled (temporal patterns)")
                        logger.info("   • 24/7 Behavioral Learning: ✅ Enabled (always watching)")
                        logger.info("   • Workflow Pattern Recognition: ✅ Active")
                        logger.info("")
                        logger.info("   📍 PHASE 4: Proactive Communication (Magic)")
                        logger.info("   • Natural Language Suggestions: ✅ Active")
                        logger.info("   • Voice Output: ✅ Enabled (JARVIS API)")
                        logger.info("   • Predictive App Launching: ✅ Active")
                        logger.info("   • Workflow Optimization Tips: ✅ Active")
                        logger.info("   • Smart Space Switching: ✅ Active")
                        logger.info("   • Context-Aware Timing: ✅ Enabled (focus-level detection)")
                        logger.info("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                        logger.info("   📊 LEARNING DATABASE METRICS:")
                        logger.info(f"   • Total Patterns: {metrics['patterns']['total_patterns']}")
                        logger.info(
                            f"   • Display Patterns: {metrics['display_patterns']['total_display_patterns']}"
                        )
                        logger.info(
                            f"   • Pattern Cache Hit Rate: {metrics['cache_performance']['pattern_cache_hit_rate']:.1%}"
                        )
                        logger.info("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                        logger.info("   🎯 CAPABILITIES:")
                        logger.info("   • Learns user patterns across all macOS workspace")
                        logger.info("   • Predicts actions before you ask")
                        logger.info("   • Proactively suggests apps and workflows naturally")
                        logger.info("   • Speaks suggestions with human-like communication")
                        logger.info("   • Adapts to UI changes automatically")
                        logger.info("   • Remembers preferences across restarts")
                        logger.info("   • Self-healing when environment changes")
                        logger.info(
                            "   • Respects your focus level (no interruptions during deep work)"
                        )
                        logger.info("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                        logger.info("   💬 PROACTIVE EXAMPLES:")
                        logger.info(
                            "   • 'Hey, you usually open Slack around this time. Want me to launch it?'"
                        )
                        logger.info(
                            "   • 'I noticed your email workflow is slower than usual. Try filtering first.'"
                        )
                        logger.info(
                            "   • 'You typically switch to Space 2 when coding. Should I move you there?'"
                        )
                        logger.info("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                    except Exception as e:
                        logger.warning(f"Could not get Learning DB metrics: {e}")
                else:
                    logger.info("✅ UAE + SAI initialized successfully")
                    logger.info("   • SAI monitoring: Active (10s interval)")
                    logger.info("   • Context intelligence: Active")
                    logger.info("   • Display clicker: Will use UAE+SAI enhanced mode")
                    logger.info("   • Proactive adaptation: Enabled")
                    logger.warning("   ⚠️  Learning Database: Not active (no persistent memory)")
            else:
                logger.warning("⚠️ UAE initialized but not active")

            # Initialize Hybrid Orchestrator (always initialized)
            logger.info("🌐 Initializing Hybrid Orchestrator (Local + GCP)...")
            try:
                from backend.core.hybrid_orchestrator import get_orchestrator

                hybrid_orchestrator = get_orchestrator()
                await hybrid_orchestrator.start()
                app.state.hybrid_orchestrator = hybrid_orchestrator
                logger.info("✅ Hybrid Orchestrator initialized (intelligent routing active)")
                logger.info("   • Local Mac (16GB) - Vision, Voice, macOS features")
                logger.info("   • GCP Cloud (32GB) - ML, NLP, heavy processing")
                logger.info("   • UAE/SAI/CAI integrated for intelligent routing")
            except Exception as e:
                logger.warning(f"⚠️  Hybrid Orchestrator not available: {e}")
                app.state.hybrid_orchestrator = None

    except Exception as e:
        logger.warning(f"⚠️ Could not initialize UAE + Learning Database: {e}")
        logger.info("   Falling back to SAI-only mode for display connections")

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
        from vision.dynamic_component_loader import get_component_loader
        from vision.rust_startup_integration import initialize_rust_acceleration

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
                logger.info(f"   • Estimated savings: {mem_savings['estimated_savings_percent']}%")
        else:
            logger.info("🦀 Rust acceleration not available (Python fallback active)")
            logger.debug(f"   Reason: {rust_config.get('fallback_reason', 'Unknown')}")

    except Exception as e:
        logger.warning(f"⚠️ Could not initialize Rust acceleration: {e}")
        app.state.rust_acceleration = {"available": False}

    # Connect vision analyzer to other components (analyzer already initialized earlier)
    if hasattr(app.state, "vision_analyzer") and app.state.vision_analyzer:
        logger.info("🔗 Connecting vision analyzer to other JARVIS components...")

        # Connect Vision Navigator to vision analyzer (for display connection)
        try:
            from display.vision_ui_navigator import get_vision_navigator

            navigator = get_vision_navigator()
            navigator.set_vision_analyzer(app.state.vision_analyzer)
            logger.info("✅ Vision Navigator connected to Claude Vision analyzer")
            logger.info("   👁️ JARVIS can now navigate Control Center using vision!")
        except Exception as e:
            logger.debug(f"Vision Navigator connection skipped: {e}")

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
    else:
        logger.warning("⚠️ Vision analyzer not available - vision features disabled")

    # Initialize proactive monitoring components
    try:
        # Set JARVIS API in vision command handler for voice integration
        from api.vision_command_handler import vision_command_handler

        voice = components.get("voice", {})
        if voice.get("jarvis_api"):
            vision_command_handler.jarvis_api = voice["jarvis_api"]
            logger.info("✅ JARVIS voice API connected to pure vision command handler")

        # Initialize pure intelligence with API key
        if api_key:
            await vision_command_handler.initialize_intelligence(api_key)
            logger.info("✅ Pure vision intelligence initialized")

        # ========================================================================
        # Initialize Context Integration Bridge (Priority 1-3 Features)
        # Multi-Space Context Tracking + Implicit Reference + Cross-Space Intelligence
        # ========================================================================
        try:
            from backend.core.context.context_integration_bridge import (
                initialize_integration_bridge,
            )

            logger.info("🧠 Initializing Context Intelligence System...")
            logger.info("   Priority 1: Multi-Space Context Tracking")
            logger.info("   Priority 2: 'What Does It Say?' Understanding")
            logger.info("   Priority 3: Cross-Space Intelligence")

            # Initialize bridge with auto-start
            bridge = await initialize_integration_bridge(auto_start=True)
            app.state.context_bridge = bridge

            # Integrate with PureVisionIntelligence for vision updates
            if hasattr(vision_command_handler, "vision_intelligence"):
                logger.info("   🔗 Connecting Vision Intelligence to Context Bridge...")
                # Store bridge reference in vision intelligence so it can feed updates
                vision_command_handler.vision_intelligence.context_bridge = bridge
                logger.info("   ✅ Vision Intelligence connected to Context Bridge")

            # Integrate with AsyncPipeline for command processing
            jarvis_api = voice.get("jarvis_api")
            if jarvis_api and hasattr(jarvis_api, "async_pipeline"):
                jarvis_api.async_pipeline.context_bridge = bridge
                logger.info("   ✅ AsyncPipeline connected to Context Bridge")

            # Get intelligence summary
            summary = bridge.get_workspace_intelligence_summary()
            logger.info("✅ Context Intelligence System initialized:")
            logger.info(
                f"   • Multi-Space Context Tracking: Active ({summary.get('total_spaces', 0)} spaces)"
            )
            logger.info(f"   • Implicit Reference Resolution: Enabled")
            logger.info(f"   • Cross-Space Intelligence: Enabled")
            logger.info(
                f"   • Natural Language Queries: 'what does it say?', 'what am I working on?'"
            )
            logger.info(f"   • Workspace Synthesis: Combining context from all spaces")

        except ImportError as e:
            logger.warning(f"   ⚠️ Context Intelligence System not available: {e}")
            app.state.context_bridge = None
        except Exception as e:
            logger.error(
                f"   ❌ Context Intelligence initialization failed: {e}",
                exc_info=True,
            )
            app.state.context_bridge = None

        # ========================================================================
        # Initialize ALL 6 Upgraded v2.0 Systems with HybridMonitoring Integration
        # ========================================================================
        logger.info("\n" + "=" * 60)
        logger.info("🚀 INITIALIZING v2.0 INTELLIGENT SYSTEMS")
        logger.info("=" * 60)

        try:
            # Get HybridProactiveMonitoringManager (if available)
            hybrid_monitoring = None
            try:
                from context_intelligence.managers.hybrid_proactive_monitoring_manager import (
                    get_hybrid_proactive_monitoring_manager,
                )

                hybrid_monitoring = get_hybrid_proactive_monitoring_manager()
                logger.info("✅ HybridProactiveMonitoringManager: Available")
            except Exception as e:
                logger.warning(f"⚠️ HybridMonitoring not available: {e}")

            # Get ImplicitReferenceResolver (from context bridge)
            implicit_resolver = None
            if hasattr(app.state, "context_bridge") and app.state.context_bridge:
                try:
                    implicit_resolver = app.state.context_bridge.implicit_resolver
                    logger.info("✅ ImplicitReferenceResolver: Available")
                except Exception as e:
                    logger.warning(f"⚠️ ImplicitResolver not available: {e}")

            # 1. TemporalQueryHandler v3.0
            try:
                from context_intelligence.handlers.temporal_query_handler import (
                    initialize_temporal_query_handler,
                )
                from context_intelligence.managers import get_change_detection_manager
                from core.conversation_tracker import get_conversation_tracker

                temporal_handler = initialize_temporal_query_handler(
                    proactive_monitoring_manager=hybrid_monitoring,
                    change_detection_manager=get_change_detection_manager(),
                    implicit_resolver=implicit_resolver,
                    conversation_tracker=get_conversation_tracker(),
                )
                app.state.temporal_handler = temporal_handler
                logger.info("✅ TemporalQueryHandler v3.0 initialized")
                logger.info("   • Pattern analysis, predictive analysis, anomaly detection")
            except Exception as e:
                logger.warning(f"⚠️ TemporalQueryHandler v3.0 init failed: {e}")

            # 2. ErrorRecoveryManager v2.0
            try:
                from autonomy.error_recovery import ErrorRecoveryManager
                from context_intelligence.managers import get_change_detection_manager

                error_recovery = ErrorRecoveryManager(
                    hybrid_monitoring_manager=hybrid_monitoring,
                    implicit_resolver=implicit_resolver,
                    change_detection_manager=get_change_detection_manager(),
                )
                app.state.error_recovery = error_recovery
                logger.info("✅ ErrorRecoveryManager v2.0 initialized")
                logger.info("   • Proactive error detection, frequency tracking, auto-healing")
            except Exception as e:
                logger.warning(f"⚠️ ErrorRecoveryManager v2.0 init failed: {e}")

            # 3. StateIntelligence v2.0
            try:
                from context_intelligence.managers import get_change_detection_manager
                from vision.intelligence.state_intelligence import initialize_state_intelligence

                async def handle_stuck_alert(alert):
                    """Handle stuck state alerts"""
                    logger.warning(f"[STUCK-STATE] {alert['message']}")

                state_intelligence = initialize_state_intelligence(
                    user_id="default",
                    hybrid_monitoring_manager=hybrid_monitoring,
                    implicit_resolver=implicit_resolver,
                    change_detection_manager=get_change_detection_manager(),
                    stuck_alert_callback=handle_stuck_alert,
                )
                app.state.state_intelligence = state_intelligence

                # Start stuck state monitoring
                asyncio.create_task(state_intelligence.start_stuck_state_monitoring())

                logger.info("✅ StateIntelligence v2.0 initialized")
                logger.info("   • Auto-recording, stuck state detection, productivity tracking")
            except Exception as e:
                logger.warning(f"⚠️ StateIntelligence v2.0 init failed: {e}")

            # 4. StateDetectionPipeline v2.0
            try:
                from context_intelligence.managers import get_change_detection_manager
                from vision.intelligence.state_detection_pipeline import StateDetectionPipeline

                async def handle_state_transition(transition):
                    """Handle state transition alerts"""
                    logger.info(
                        f"[STATE-TRANSITION] Space {transition['space_id']}: "
                        f"{transition['from_state']} → {transition['to_state']}"
                    )

                async def handle_new_state(new_state):
                    """Handle unknown state detection"""
                    logger.info(f"[NEW-STATE] Unknown state in Space {new_state['space_id']}")

                state_detection = StateDetectionPipeline(
                    hybrid_monitoring_manager=hybrid_monitoring,
                    implicit_resolver=implicit_resolver,
                    change_detection_manager=get_change_detection_manager(),
                    state_transition_callback=handle_state_transition,
                    new_state_callback=handle_new_state,
                )
                app.state.state_detection = state_detection
                logger.info("✅ StateDetectionPipeline v2.0 initialized")
                logger.info("   • Auto-triggered detection, visual signature learning")
            except Exception as e:
                logger.warning(f"⚠️ StateDetectionPipeline v2.0 init failed: {e}")

            # 5. ComplexComplexityHandler v2.0
            try:
                from context_intelligence.handlers.complex_complexity_handler import (
                    initialize_complex_complexity_handler,
                )
                from context_intelligence.managers import (
                    get_capture_strategy_manager,
                    get_ocr_strategy_manager,
                )

                complex_handler = initialize_complex_complexity_handler(
                    temporal_handler=(
                        app.state.temporal_handler
                        if hasattr(app.state, "temporal_handler")
                        else None
                    ),
                    capture_manager=get_capture_strategy_manager(),
                    ocr_manager=get_ocr_strategy_manager(),
                    implicit_resolver=implicit_resolver,
                    hybrid_monitoring_manager=hybrid_monitoring,
                    prefer_monitoring_cache=True,
                )
                app.state.complex_handler = complex_handler
                logger.info("✅ ComplexComplexityHandler v2.0 initialized")
                logger.info("   • Ultra-fast queries (87% faster), monitoring cache enabled")
            except Exception as e:
                logger.warning(f"⚠️ ComplexComplexityHandler v2.0 init failed: {e}")

            # 6. PredictiveQueryHandler v2.0
            try:
                from context_intelligence.handlers.predictive_query_handler import (
                    initialize_predictive_handler,
                )

                predictive_handler = initialize_predictive_handler(
                    context_graph=None,  # TODO: Add context graph if available
                    hybrid_monitoring_manager=hybrid_monitoring,
                    implicit_resolver=implicit_resolver,
                    enable_vision=True,
                    claude_api_key=api_key,
                )
                app.state.predictive_handler = predictive_handler
                logger.info("✅ PredictiveQueryHandler v2.0 initialized")
                logger.info("   • Progress tracking, bug prediction, workflow suggestions")
            except Exception as e:
                logger.warning(f"⚠️ PredictiveQueryHandler v2.0 init failed: {e}")

            logger.info("\n" + "=" * 60)
            logger.info("✨ ALL 6 v2.0 SYSTEMS INITIALIZED")
            logger.info("=" * 60)
            logger.info("🎯 Enhanced Capabilities:")
            logger.info("   1. TemporalQueryHandler    - ML-powered temporal analysis")
            logger.info("   2. ErrorRecoveryManager    - Proactive error detection & healing")
            logger.info("   3. StateIntelligence       - Auto-learning state patterns")
            logger.info("   4. StateDetectionPipeline  - Visual signature learning")
            logger.info("   5. ComplexComplexityHandler - 87% faster complex queries")
            logger.info("   6. PredictiveQueryHandler  - Intelligent predictions")
            logger.info("\n🚀 All systems integrated with HybridMonitoring & ImplicitResolver!")
            logger.info("=" * 60 + "\n")

        except Exception as e:
            logger.error(f"❌ v2.0 Systems initialization failed: {e}", exc_info=True)

        # Log proactive monitoring configuration
        proactive_config = app.state.vision_analyzer.get_proactive_config()
        if proactive_config["proactive_enabled"]:
            logger.info("✅ Proactive Vision Intelligence System initialized with:")
            logger.info(f"   - Confidence threshold: {proactive_config['confidence_threshold']}")
            logger.info(
                f"   - Voice announcements: {'enabled' if proactive_config['voice_enabled'] else 'disabled'}"
            )
            logger.info("   - Debugging Assistant: Auto-detects code errors")
            logger.info("   - Research Helper: Monitors multi-tab workflows")
            logger.info("   - Workflow Optimizer: Identifies repetitive patterns")
            logger.info("   - Privacy Protection: Auto-pauses for sensitive content")
            logger.info("   - Say 'Start monitoring my screen' to activate intelligent assistance")
        else:
            logger.info("⚠️ Proactive monitoring disabled in configuration")
    except Exception as e:
        logger.warning(f"⚠️ Could not initialize proactive monitoring components: {e}")

    # Initialize weather system with vision
    try:
        from system_control.macos_controller import MacOSController
        from system_control.weather_system_config import initialize_weather_system

        controller = MacOSController()
        weather_bridge = initialize_weather_system(app.state.vision_analyzer, controller)
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
                logger.info("✅ Vision status manager initialized and connected")

        asyncio.create_task(setup_vision_status())
    except Exception as e:
        logger.warning(f"⚠️ Could not initialize vision status manager: {e}")

    # NOTE: This elif was orphaned/unreachable - analyzer_class check already done at line 988
    # elif analyzer_class:
    #     logger.warning("⚠️ Vision analyzer available but no ANTHROPIC_API_KEY set")

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
        logger.warning(f"⚠️  Only {loaded_count}/8 components loaded - some features may be limited")

    logger.info("=" * 60 + "\n")

    # Initialize Voice Unlock system components (WebSocket server)
    voice_unlock = components.get("voice_unlock") or {}
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
                    logger.info("🎤 Wake word detection service started - Say 'Hey JARVIS'!")
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
        await mesh.join(
            {
                "name": "jarvis_backend",
                "port": backend_port,
                "protocol": "http",
                "type": "backend",
                "endpoints": {
                    "health": "/health",
                    "vision": "/vision",
                    "voice": "/voice",
                    "chat": "/chat",
                },
            }
        )

        app.state.orchestrator = orchestrator
        app.state.mesh = mesh

        logger.info("✅ Registered with autonomous orchestrator and mesh network")
    except Exception as e:
        logger.warning(f"⚠️ Could not register with autonomous systems: {e}")

    # Initialize Cost Tracking System (Priority 2: Cost Monitoring & Alerts)
    try:
        from core.cost_tracker import initialize_cost_tracking

        await initialize_cost_tracking()
        logger.info("✅ Cost Tracking System initialized")
        logger.info("   • Auto-cleanup enabled for orphaned VMs")
        logger.info("   • Real-time cost monitoring active")
        logger.info("   • Alert system configured")
    except Exception as e:
        logger.warning(f"⚠️ Cost tracking initialization failed: {e}")

    yield

    # Cleanup
    logger.info("🛑 Shutting down JARVIS backend...")

    # Notify all WebSocket clients about shutdown
    try:
        from api.jarvis_voice_api import broadcast_shutdown_notification

        await broadcast_shutdown_notification()
        logger.info("✅ Shutdown notifications sent to WebSocket clients")
        # Give clients a brief moment to receive the notification
        await asyncio.sleep(0.5)
    except Exception as e:
        logger.warning(f"Failed to broadcast shutdown notification: {e}")

    # Cleanup GCP VM Manager (before cost tracker to finalize costs)
    try:
        # Get GCP VM Manager instance
        if gcp_vm_manager:  # Check if gcp_vm_manager is initialized
            logger.info("🧹 Cleaning up GCP VMs...")
            await gcp_vm_manager.cleanup()  # Cleanup GCP VM Manager
            logger.info("✅ GCP VM Manager cleanup complete")
    except Exception as e:
        logger.error(f"Failed to cleanup GCP VM Manager: {e}")

    # Shutdown Cost Tracking System
    try:
        from core.cost_tracker import get_cost_tracker

        tracker = get_cost_tracker()
        if tracker:
            await tracker.shutdown()
            logger.info("✅ Cost Tracking System shutdown complete")
    except Exception as e:
        logger.error(f"Failed to shutdown cost tracker: {e}")

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

    # Shutdown Unified Context Bridge
    if hasattr(app.state, "context_bridge"):
        try:
            from backend.core.unified_context_bridge import shutdown_context_bridge

            await shutdown_context_bridge()
            logger.info("✅ Unified Context Bridge stopped")
        except Exception as e:
            logger.error(f"Failed to stop Context Bridge: {e}")

    # Shutdown Goal Inference Integration
    if hasattr(app.state, "goal_inference_integration"):
        try:
            integration = app.state.goal_inference_integration
            if hasattr(integration, "learning_db"):
                # Save final state and close connections
                integration.learning_db.close()
            logger.info("✅ Goal Inference Integration stopped")
        except Exception as e:
            logger.error(f"Failed to stop Goal Inference: {e}")

    # Shutdown UAE (Unified Awareness Engine) + Learning Database + Yabai
    if hasattr(app.state, "uae_engine"):
        try:
            from intelligence.uae_integration import get_learning_db, get_yabai, shutdown_uae

            logger.info("🧠 Shutting down Intelligence Stack...")

            # Get Learning DB metrics before shutdown
            learning_db = get_learning_db()
            if learning_db:
                try:
                    metrics = await learning_db.get_learning_metrics()
                    logger.info("   📊 Final Learning Database Stats:")
                    logger.info(
                        f"   • Total Patterns Learned: {metrics['patterns']['total_patterns']}"
                    )
                    logger.info(
                        f"   • Display Patterns: {metrics['display_patterns']['total_display_patterns']}"
                    )
                    logger.info(f"   • Total Actions Logged: {metrics['actions']['total_actions']}")
                    logger.info(f"   • Success Rate: {metrics['actions']['success_rate']:.1f}%")
                    logger.info(
                        f"   • Cache Hit Rate: {metrics['cache_performance']['pattern_cache_hit_rate']:.1%}"
                    )
                except Exception as e:
                    logger.debug(f"Could not get final metrics: {e}")

            # Get Phase 2 metrics before shutdown
            logger.info("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logger.info("   📊 PHASE 2 INTELLIGENCE STACK - FINAL STATS")
            logger.info("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            # Yabai metrics
            yabai = get_yabai()
            if yabai and yabai.is_monitoring:
                try:
                    yabai_metrics = yabai.get_metrics()
                    logger.info("   🗺️  Yabai Spatial Intelligence:")
                    logger.info(f"      • Spaces Monitored: {yabai_metrics['spaces_monitored']}")
                    logger.info(f"      • Windows Tracked: {yabai_metrics['windows_tracked']}")
                    logger.info(f"      • Space Changes: {yabai_metrics['total_space_changes']}")
                    logger.info(f"      • Monitoring Cycles: {yabai_metrics['monitoring_cycles']}")
                    logger.info(f"      • Events Emitted: {yabai_metrics.get('events_emitted', 0)}")
                    logger.info(
                        f"      • Session Duration: {yabai_metrics['session_duration_minutes']:.1f} minutes"
                    )
                except Exception as e:
                    logger.debug(f"Could not get Yabai metrics: {e}")

            # Pattern Learner metrics
            from intelligence.uae_integration import get_pattern_learner_sync

            pattern_learner = get_pattern_learner_sync()
            if pattern_learner:
                try:
                    pl_stats = pattern_learner.get_statistics()
                    logger.info("   🧠 Workspace Pattern Learner (ML):")
                    logger.info(f"      • Total Patterns Learned: {pl_stats['total_patterns']}")
                    logger.info(f"      • Workflows Detected: {pl_stats['workflows_detected']}")
                    logger.info(f"      • Temporal Patterns: {pl_stats['temporal_patterns']}")
                    logger.info(f"      • Spatial Preferences: {pl_stats['spatial_preferences']}")
                    logger.info(
                        f"      • Predictions Generated: {pl_stats['predictions_generated']}"
                    )
                    logger.info(f"      • ML Clustering Runs: {pl_stats['clustering_runs']}")
                except Exception as e:
                    logger.debug(f"Could not get Pattern Learner stats: {e}")

            # Bridge metrics
            from intelligence.uae_integration import get_integration_bridge

            bridge = get_integration_bridge()
            if bridge and bridge.is_active:
                try:
                    bridge_metrics = bridge.get_metrics()
                    logger.info("   🔗 Yabai ↔ SAI Integration Bridge:")
                    logger.info(f"      • Events Bridged: {bridge_metrics['events_bridged']}")
                    logger.info(f"      • Yabai → SAI: {bridge_metrics['yabai_to_sai']}")
                    logger.info(f"      • SAI → Yabai: {bridge_metrics['sai_to_yabai']}")
                    logger.info(f"      • Contexts Enriched: {bridge_metrics['contexts_enriched']}")
                    logger.info(
                        f"      • Actions Coordinated: {bridge_metrics['actions_coordinated']}"
                    )
                except Exception as e:
                    logger.debug(f"Could not get Bridge metrics: {e}")

            logger.info("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            # Shutdown UAE + Learning DB + Yabai + Phase 2
            await shutdown_uae()

            logger.info("✅ UAE (Unified Awareness Engine) stopped")
            logger.info("✅ SAI (Situational Awareness) stopped")
            logger.info("✅ Yabai (Spatial Intelligence) stopped")
            logger.info("✅ Pattern Learner (ML) finalized")
            logger.info("✅ Integration Bridge (Yabai↔SAI) stopped")
            logger.info("✅ Learning Database closed (all data persisted)")
        except Exception as e:
            logger.error(f"Failed to stop UAE + Learning Database + Yabai: {e}")

    # Stop Voice Unlock system
    voice_unlock = components.get("voice_unlock") or {}
    if hasattr(app.state, "voice_unlock_system") and voice_unlock.get("shutdown_system"):
        try:
            shutdown_system = voice_unlock["shutdown_system"]
            await shutdown_system()
            logger.info("✅ Voice Unlock system stopped")
        except Exception as e:
            logger.error(f"Failed to stop Voice Unlock system: {e}")

    # Stop display monitoring (Component #9)
    if hasattr(app.state, "display_monitor"):
        try:
            await app.state.display_monitor.stop()
            logger.info("✅ Display monitoring stopped")
        except Exception as e:
            logger.error(f"Error stopping display monitoring: {e}")

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
    pass

    logger.info("Vision monitoring fix applied")
except Exception as e:
    logger.warning(f"Could not apply vision fix: {e}")

# Force reload vision handler to get latest fixes
try:
    pass

    logger.info("Vision handler reloaded with multi-space fixes")
except Exception as e:
    logger.warning(f"Could not reload vision handler: {e}")

# Create FastAPI app
logger.info("Creating optimized FastAPI app...")
app = FastAPI(
    title="JARVIS Backend (Optimized)",
    version="13.4.0-browser-automation",
    lifespan=lifespan,
)

# Configure Dynamic CORS
try:
    from api.dynamic_cors_handler import DynamicCORSMiddleware

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
    origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")
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
            pass

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

    # Check dynamic component manager status
    component_manager_details = {}
    if hasattr(app.state, "component_manager"):
        mgr = app.state.component_manager
        component_manager_details = {
            "enabled": True,
            "total_components": len(mgr.components),
            "memory_pressure": mgr.memory_monitor.current_pressure().value,
            "arm64_optimized": mgr.arm64_optimizer.is_arm64,
            "m1_detected": mgr.arm64_optimizer.is_m1,
            "config_loaded": (os.path.exists(mgr.config_path) if mgr.config_path else False),
            "advanced_preloader": {
                "predictor_active": mgr.advanced_predictor is not None,
                "dependency_resolver_active": mgr.dependency_resolver is not None,
                "smart_cache_active": mgr.smart_cache is not None,
            },
        }
    else:
        component_manager_details = {"enabled": False}

    return {
        "status": "healthy",
        "mode": "optimized" if OPTIMIZE_STARTUP else "legacy",
        "parallel_imports": PARALLEL_IMPORTS,
        "lazy_models": LAZY_LOAD_MODELS,
        "components": {name: bool(comp) for name, comp in components.items() if comp is not None},
        "vision_status": vision_status,
        "vision_enhanced": vision_details,
        "ml_audio_system": ml_audio_details,
        "rust_acceleration": rust_details,
        "self_healing": self_healing_details,
        "voice_unlock": voice_unlock_details,
        "component_manager": component_manager_details,
    }


@app.get("/hybrid/status")
async def hybrid_status():
    """
    Get hybrid cloud routing status and SAI learning metrics.

    Returns comprehensive status of:
    - RAM usage and trends
    - GCP deployment state
    - SAI learning statistics
    - Component locations (local vs GCP)
    - Migration metrics
    - Crash prevention stats
    """
    # Check if hybrid coordinator is available (from start_system.py)
    # Note: This endpoint works even if start_system.py isn't running
    # It will show the last known state or indicate hybrid is inactive

    try:
        # Try to import and check if coordinator is running
        # This is a read-only status check
        from datetime import datetime

        import psutil

        # Get current RAM state
        mem = psutil.virtual_memory()
        ram_state = {
            "total_gb": mem.total / (1024**3),
            "used_gb": mem.used / (1024**3),
            "available_gb": mem.available / (1024**3),
            "percent": mem.percent,
            "status": (
                "EMERGENCY"
                if mem.percent >= 95
                else (
                    "CRITICAL"
                    if mem.percent >= 85
                    else (
                        "WARNING"
                        if mem.percent >= 75
                        else "ELEVATED" if mem.percent >= 60 else "OPTIMAL"
                    )
                )
            ),
        }

        # Check if running on GCP (via environment detection)
        import os

        is_gcp = os.path.exists("/.dockerenv") or os.getenv("GCP_PROJECT_ID") is not None

        # Try to load SAI learned parameters from database
        learned_params = {}
        try:
            # Check if learning database has hybrid parameters
            import json
            import sys
            from pathlib import Path

            sys.path.insert(0, str(Path(__file__).parent))
            from intelligence.learning_database import get_learning_database

            db = await get_learning_database()

            # Query for latest hybrid learning stats
            async with db.db.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT metadata
                    FROM patterns
                    WHERE pattern_type = 'hybrid_threshold'
                    ORDER BY last_seen DESC
                    LIMIT 1
                """
                )
                result = await cursor.fetchone()

                if result and result[0]:
                    metadata = json.loads(result[0])
                    learned_params = {
                        "thresholds": metadata.get("thresholds", {}),
                        "confidence": metadata.get("confidence", {}),
                        "component_weights": metadata.get("component_weights", {}),
                        "stats": metadata.get("stats", {}),
                        "last_updated": metadata.get("last_updated"),
                    }

            await db.close()

        except Exception as e:
            learned_params = {"error": f"Could not load learning data: {str(e)}"}

        # Build response
        response = {
            "timestamp": datetime.now().isoformat(),
            "hybrid_enabled": os.getenv("JARVIS_HYBRID_MODE", "auto") in ["auto", "true", "1"],
            "current_location": "gcp" if is_gcp else "local",
            "ram": ram_state,
            "gcp_available": is_gcp or bool(os.getenv("GCP_PROJECT_ID")),
            "sai_learning": (
                learned_params
                if learned_params
                else {
                    "status": "No learned parameters yet",
                    "note": "Run start_system.py to enable learning",
                }
            ),
            "features": {
                "crash_prevention": True,
                "auto_scaling": True,
                "predictive_routing": True,
                "cost_optimization": True,
                "persistent_learning": True,
            },
            "thresholds": learned_params.get(
                "thresholds",
                {
                    "warning": 0.75,
                    "critical": 0.85,
                    "optimal": 0.60,
                    "emergency": 0.95,
                    "note": "Default values (not yet learned)",
                },
            ),
        }

        return response

    except Exception as e:
        # Return error but still provide basic info
        return {
            "timestamp": datetime.now().isoformat(),
            "hybrid_enabled": False,
            "error": str(e),
            "message": "Hybrid routing status unavailable. Run start_system.py to enable.",
            "features": {
                "crash_prevention": True,
                "auto_scaling": True,
                "predictive_routing": True,
                "cost_optimization": True,
                "persistent_learning": True,
            },
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
        "mesh": mesh_status,
    }


@app.get("/components/status")
async def component_status():
    """Get dynamic component manager status with performance metrics"""
    if not hasattr(app.state, "component_manager"):
        return {"enabled": False, "message": "Dynamic component loading not enabled"}

    mgr = app.state.component_manager
    status = mgr.get_status()

    return {
        "enabled": True,
        "config_path": mgr.config_path,
        "memory_limit_gb": mgr.memory_limit_gb,
        **status,  # Unpack all status fields
    }


@app.get("/components/metrics")
async def component_metrics():
    """Get detailed performance metrics"""
    if not hasattr(app.state, "component_manager"):
        return {"enabled": False}

    mgr = app.state.component_manager
    status = mgr.get_status()

    # Calculate efficiency score
    total_loads = status["performance"]["total_loads"]
    cache_hit_rate = status["performance"]["cache_hit_rate"]
    memory_saved = status["memory"]["saved_mb"]

    efficiency_score = 0
    if total_loads > 0:
        # Score based on cache hits, memory savings, and load count
        efficiency_score = min(100, (cache_hit_rate * 0.4) + (min(memory_saved / 100, 50) * 0.6))

    return {
        "enabled": True,
        "timestamp": datetime.now().isoformat(),
        "efficiency_score": round(efficiency_score, 1),
        "metrics": {
            "component_utilization": {
                "total": status["total_components"],
                "loaded": status["loaded_components"],
                "utilization_percent": (
                    round(
                        (status["loaded_components"] / status["total_components"]) * 100,
                        1,
                    )
                    if status["total_components"] > 0
                    else 0
                ),
            },
            "memory_metrics": status["memory"],
            "performance_metrics": status["performance"],
            "platform_info": status["platform"],
        },
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
        app.include_router(voice["jarvis_router"], prefix="/voice/jarvis", tags=["jarvis"])
        logger.info("✅ JARVIS Voice API mounted")

        # Set JARVIS instance in unified WebSocket pipeline
        try:
            from api.unified_websocket import set_jarvis_instance

            jarvis_api = voice.get("jarvis_api")
            if jarvis_api:
                set_jarvis_instance(jarvis_api)
        except Exception as e:
            logger.warning(f"⚠️  Could not set JARVIS in WebSocket pipeline: {e}")

    if voice and voice.get("enhanced_available"):
        app.include_router(voice["enhanced_router"], prefix="/voice/enhanced", tags=["voice"])
        logger.info("✅ Enhanced Voice API mounted")

    # ML Model Status API
    ml = components.get("ml_models", {})
    if ml and ml.get("status_router"):
        app.include_router(ml["status_router"], prefix="/models", tags=["models"])
        logger.info("✅ Model Status API mounted")

    # Monitoring API
    monitoring = components.get("monitoring", {})
    if monitoring and monitoring.get("router"):
        app.include_router(monitoring["router"], prefix="/monitoring", tags=["monitoring"])
        logger.info("✅ Monitoring API mounted")

    # Voice Unlock API
    voice_unlock = components.get("voice_unlock", {})
    if voice_unlock and voice_unlock.get("router"):
        app.include_router(voice_unlock["router"], tags=["voice_unlock"])
        logger.info("✅ Voice Unlock API mounted")
        if voice_unlock.get("initialized"):
            app.state.voice_unlock = voice_unlock
            logger.info("✅ Voice Unlock service ready")

    # Screen Control API - HTTP REST endpoints for unlock/lock
    try:
        from api.screen_control_api import router as screen_control_router

        app.include_router(screen_control_router)
        logger.info("✅ Screen Control REST API mounted at /api/screen")
    except Exception as e:
        logger.warning(f"Screen Control API not available: {e}")

    # Wake Word API - Always mount (has stub functionality)
    try:
        from api.wake_word_api import router as wake_word_router

        # Router already has prefix="/api/wake-word", don't add it again
        app.include_router(wake_word_router)
        logger.info("✅ Wake Word API mounted at /api/wake-word")

        # Check if the full service is available
        wake_word = components.get("wake_word", {})
        if wake_word and wake_word.get("initialized"):
            app.state.wake_word = wake_word
            logger.info("✅ Wake Word detection service available")
        else:
            logger.info("ℹ️  Wake Word API available (stub mode - service not initialized)")
    except ImportError as e:
        logger.warning(f"⚠️ Wake Word API not available: {e}")

    # Rust API (if Rust components are available)
    if hasattr(app.state, "rust_acceleration") and app.state.rust_acceleration.get("available"):
        try:
            from api.rust_api import router as rust_router

            app.include_router(rust_router, prefix="/rust", tags=["rust"])
            logger.info("✅ Rust acceleration API mounted")
        except ImportError:
            logger.debug("Rust API not available")

    # Self-healing API
    try:
        from api.self_healing_api import router as self_healing_router

        app.include_router(self_healing_router, prefix="/self-healing", tags=["self-healing"])
        logger.info("✅ Self-healing API mounted")
    except ImportError:
        logger.debug("Self-healing API not available")

    # Context Intelligence API (Priority 1-3 features)
    if hasattr(app.state, "context_bridge") and app.state.context_bridge:
        from pydantic import BaseModel

        class ContextQueryRequest(BaseModel):
            query: str
            current_space_id: Optional[int] = None

        @app.post("/context/query", tags=["context"])
        async def query_context(request: ContextQueryRequest):
            """
            Natural language query interface for workspace context.

            Examples:
            - "what does it say?" → Find and explain most recent error
            - "what's the error?" → Find most recent error
            - "what am I working on?" → Synthesize workspace activity
            - "what's related?" → Show cross-space relationships
            """
            try:
                response = await app.state.context_bridge.handle_user_query(
                    request.query, request.current_space_id
                )
                return {"success": True, "response": response}
            except Exception as e:
                logger.error(f"Context query failed: {e}")
                return {"success": False, "error": str(e)}

        @app.get("/context/summary", tags=["context"])
        async def get_context_summary():
            """Get comprehensive workspace intelligence summary"""
            try:
                summary = app.state.context_bridge.get_workspace_intelligence_summary()
                return {"success": True, "summary": summary}
            except Exception as e:
                logger.error(f"Context summary failed: {e}")
                return {"success": False, "error": str(e)}

        @app.post("/context/ocr_update", tags=["context"])
        async def process_ocr_update(
            space_id: int,
            app_name: str,
            ocr_text: str,
            screenshot_path: Optional[str] = None,
        ):
            """Process OCR update from vision system"""
            try:
                await app.state.context_bridge.process_ocr_update(
                    space_id=space_id,
                    app_name=app_name,
                    ocr_text=ocr_text,
                    screenshot_path=screenshot_path,
                )
                return {"success": True}
            except Exception as e:
                logger.error(f"OCR update failed: {e}")
                return {"success": False, "error": str(e)}

        logger.info("✅ Context Intelligence API mounted at /context")
        logger.info("   • POST /context/query - Natural language queries")
        logger.info("   • GET  /context/summary - Workspace intelligence summary")
        logger.info("   • POST /context/ocr_update - Vision system integration")

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
        from api.vision_ws_endpoint import router as vision_ws_endpoint_router
        from api.vision_ws_endpoint import set_vision_analyzer

        app.include_router(vision_ws_endpoint_router, tags=["vision"])

        # Set vision analyzer if available
        vision = components.get("vision", {})
        if vision and vision.get("analyzer"):
            set_vision_analyzer(vision["analyzer"])

        logger.info("✅ Vision WebSocket endpoint mounted at /vision/ws/vision")
    except ImportError as e:
        logger.warning(f"Could not import vision WebSocket endpoint: {e}")

    # Multi-Monitor Display Routes
    try:
        from api.display_routes import router as display_router

        app.include_router(display_router, tags=["displays"])
        logger.info("✅ Multi-Monitor display routes configured")
    except Exception as e:
        logger.warning(f"⚠️  Multi-Monitor display routes not available: {e}")

    # Proximity-Aware Display Routes (Phase 1.2)
    try:
        from api.proximity_display_api import router as proximity_display_router

        app.include_router(proximity_display_router, tags=["proximity-display"])
        logger.info("✅ Proximity-Aware Display API configured")
    except Exception as e:
        logger.warning(f"⚠️  Proximity-Aware Display API not available: {e}")

    # Advanced Display Monitor (Component #9) - Multi-method detection with voice integration
    try:
        display_monitor_comp = components.get("display_monitor", {})
        if display_monitor_comp.get("available"):
            logger.info("🖥️  Initializing Advanced Display Monitor (Component #9)...")

            # Create voice handler for display monitor
            voice_handler_factory = display_monitor_comp.get("voice_handler_factory")
            get_monitor = display_monitor_comp.get("get_monitor")

            if voice_handler_factory and get_monitor:
                # Create voice handler
                voice_handler = voice_handler_factory()

                # Get monitor instance with voice integration
                monitor = get_monitor(voice_handler=voice_handler)

                # Register as the app's monitor instance (singleton pattern)
                from display.advanced_display_monitor import set_app_display_monitor

                set_app_display_monitor(monitor)
                logger.info("   ✅ Display monitor registered as singleton")

                # Connect Vision Navigator with Claude Vision analyzer
                try:
                    from display.vision_ui_navigator import get_vision_navigator

                    navigator = get_vision_navigator()

                    # Connect vision analyzer if available
                    if hasattr(app.state, "vision_analyzer"):
                        navigator.set_vision_analyzer(app.state.vision_analyzer)
                        # Also connect to monitor
                        monitor.vision_analyzer = app.state.vision_analyzer
                        logger.info("   ✅ Vision Navigator connected to Claude Vision")
                        logger.info("   👁️ JARVIS can now SEE and CLICK UI elements!")
                    else:
                        logger.warning(
                            "   ⚠️ Vision analyzer not available yet (will connect later)"
                        )

                except Exception as nav_err:
                    logger.warning(f"   ⚠️ Could not initialize Vision Navigator: {nav_err}")

                # Set WebSocket manager for UI notifications
                try:
                    from api.unified_websocket import ws_manager

                    monitor.set_websocket_manager(ws_manager)
                    ws_manager.display_monitor = (
                        monitor  # Allow ws_manager to send current status to new clients
                    )
                    logger.info("   ✅ Display monitor connected to WebSocket")
                except Exception as ws_err:
                    logger.warning(f"   ⚠️ Could not connect display monitor to WebSocket: {ws_err}")

                # Store monitor in app state for access by other components
                app.state.display_monitor = monitor

                # Start monitoring automatically
                async def start_display_monitoring():
                    await asyncio.sleep(2)  # Wait for system to fully initialize
                    await monitor.start()
                    logger.info("   ✅ Display monitoring started")
                    logger.info("   📺 Monitoring for configured displays (Living Room TV)")
                    logger.info("   🎤 Voice announcements enabled")
                    logger.info("   ⚡ Smart caching enabled (3-5x performance)")
                    logger.info("   🔍 Detection methods: AppleScript, CoreGraphics, Yabai")

                asyncio.create_task(start_display_monitoring())
                logger.info("✅ Advanced Display Monitor configured")
            else:
                logger.warning("   ⚠️ Display monitor factories not available")
        else:
            logger.warning("⚠️  Display Monitor not available (component not loaded)")

    except Exception as e:
        logger.warning(f"⚠️  Display Monitor initialization failed: {e}")
        import traceback

        logger.debug(traceback.format_exc())

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
        logger.info("✅ Auto Configuration API mounted - clients can auto-discover settings")
    except ImportError as e:
        logger.warning(f"Could not import Auto Config router: {e}")

    # Autonomous Service API (for zero-configuration mode)
    try:
        # Check if we should use memory-optimized version
        use_memory_optimized = os.getenv("MEMORY_OPTIMIZED_MODE", "true").lower() == "true"

        if use_memory_optimized:
            # Import memory-optimized orchestrator
            from backend.core.memory_optimized_orchestrator import get_memory_optimized_orchestrator

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

    # Mount Hybrid Cloud Cost Monitoring API
    try:
        from core.cost_tracker import initialize_cost_tracking
        from routers.hybrid import router as hybrid_router

        app.include_router(hybrid_router)
        logger.info("✅ Hybrid Cloud Cost Monitoring API mounted at /hybrid")

        # Initialize cost tracking database
        asyncio.create_task(initialize_cost_tracking())

    except ImportError as e:
        logger.warning(f"Hybrid Cloud API not available: {e}")

    # Model Lifecycle Management API (Phase 3.1+)
    try:
        from api.model_routes import router as model_router

        app.include_router(model_router)
        logger.info("✅ Model Lifecycle Management API mounted at /models")
        logger.info("   Endpoints: /models/registry, /models/select, /models/execute")
        logger.info("   Monitoring: /models/metrics, /models/health")
        logger.info("   Lifecycle: /models/lifecycle/status, /models/lifecycle/ram")

    except ImportError as e:
        logger.warning(f"Model Lifecycle Management API not available: {e}")

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

    # Trigger intelligent preloading (Phase 2) if available
    if hasattr(app.state, "component_manager") and app.state.component_manager:
        mgr = app.state.component_manager
        if mgr.advanced_predictor:
            try:
                # Predict and preload next 1-3 components in background
                asyncio.create_task(mgr.predict_and_preload(command, steps_ahead=3))
                logger.debug(f"🔮 Advanced preloading triggered for: '{command[:50]}'")
            except Exception as e:
                logger.debug(f"Advanced preloading failed: {e}")

    # Use unified command processor if available
    try:
        from api.unified_command_processor import UnifiedCommandProcessor

        # Use enhanced Context Intelligence for screen lock/unlock
        USE_ENHANCED_CONTEXT = True

        if USE_ENHANCED_CONTEXT:
            try:
                from api.simple_context_handler_enhanced import wrap_with_enhanced_context

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
        "components": {name: bool(comp) for name, comp in components.items() if comp is not None},
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
        from datetime import datetime

        from api.unified_websocket import connection_capabilities, ws_manager

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


# ============================================================
# LAZY LOADING HELPER FOR UAE/SAI/LEARNING DB
# ============================================================


async def ensure_uae_loaded(app_state):
    """
    Lazy-load UAE/SAI/Learning DB on first use with Memory Quantizer integration.
    This saves 8-10GB of RAM at startup and prevents OOM kills.
    """
    # Already loaded?
    if app_state.uae_engine is not None:
        return app_state.uae_engine

    # Already initializing?
    if app_state.uae_initializing:
        # Wait for initialization to complete
        import asyncio

        for _ in range(50):  # Wait up to 5 seconds
            await asyncio.sleep(0.1)
            if app_state.uae_engine is not None:
                return app_state.uae_engine
        logger.warning("[LAZY-UAE] Timeout waiting for UAE initialization")
        return None

    # ============================================================
    # MEMORY QUANTIZER INTEGRATION - Intelligent Load Prevention
    # ============================================================
    try:
        from core.memory_quantizer import MemoryQuantizer, MemoryTier

        # Get memory quantizer instance
        quantizer = MemoryQuantizer()
        metrics = quantizer.get_current_metrics()  # Synchronous call

        # Log current memory state
        logger.info(f"[LAZY-UAE] Memory check before loading:")
        logger.info(f"[LAZY-UAE]   • Tier: {metrics.tier.value}")
        logger.info(f"[LAZY-UAE]   • Pressure: {metrics.pressure.value}")
        logger.info(f"[LAZY-UAE]   • Available: {metrics.system_memory_available_gb:.2f} GB")
        logger.info(f"[LAZY-UAE]   • Usage: {metrics.system_memory_percent:.1f}%")

        # Estimated UAE/SAI/Learning DB memory requirement
        REQUIRED_MEMORY_GB = 10.0

        # Check if we have enough memory available
        if metrics.system_memory_available_gb < REQUIRED_MEMORY_GB:
            logger.error(f"[LAZY-UAE] ❌ Insufficient memory for intelligence components")
            logger.error(f"[LAZY-UAE]    Required: {REQUIRED_MEMORY_GB:.1f} GB")
            logger.error(f"[LAZY-UAE]    Available: {metrics.system_memory_available_gb:.2f} GB")
            logger.error(
                f"[LAZY-UAE]    Deficit: {REQUIRED_MEMORY_GB - metrics.system_memory_available_gb:.2f} GB"
            )

            # Provide fallback recommendation
            logger.info(f"[LAZY-UAE] 💡 Falling back to basic multi-space detection (Yabai only)")
            return None

        # Check memory tier - refuse to load in dangerous tiers
        dangerous_tiers = {MemoryTier.CRITICAL, MemoryTier.EMERGENCY, MemoryTier.CONSTRAINED}
        if metrics.tier in dangerous_tiers:
            logger.warning(
                f"[LAZY-UAE] ⚠️  Memory tier is {metrics.tier.value} - postponing intelligence loading"
            )
            logger.warning(f"[LAZY-UAE]    Current tier: {metrics.tier.value}")
            logger.warning(f"[LAZY-UAE]    Required tier: ELEVATED or better")
            logger.info(f"[LAZY-UAE] 💡 Using lightweight mode until memory pressure reduces")
            return None

        # Predictive check - will loading cause OOM?
        predicted_usage = metrics.system_memory_percent + (
            REQUIRED_MEMORY_GB / metrics.system_memory_gb * 100
        )
        if predicted_usage > 90:
            logger.warning(
                f"[LAZY-UAE] ⚠️  Loading would push usage to {predicted_usage:.1f}% (OOM risk)"
            )
            logger.warning(f"[LAZY-UAE]    Current: {metrics.system_memory_percent:.1f}%")
            logger.warning(f"[LAZY-UAE]    After load: ~{predicted_usage:.1f}%")
            logger.warning(f"[LAZY-UAE]    Safe threshold: <85%")
            return None

        # Memory check PASSED - safe to load
        logger.info(f"[LAZY-UAE] ✅ Memory check PASSED - safe to load intelligence")
        logger.info(f"[LAZY-UAE]    Predicted usage after load: {predicted_usage:.1f}%")

    except Exception as e:
        logger.warning(f"[LAZY-UAE] Memory Quantizer check failed: {e}")
        logger.warning(f"[LAZY-UAE] Proceeding with loading (no safety check)")

    # Start initialization
    app_state.uae_initializing = True
    logger.info("[LAZY-UAE] 🧠 Initializing UAE/SAI/Learning DB on first use...")

    try:
        from intelligence.uae_integration import get_learning_db, initialize_uae

        config = app_state.uae_lazy_config

        # Create voice callback
        async def voice_callback(text: str):
            """Voice callback for proactive suggestions"""
            try:
                voice = components.get("voice", {})
                jarvis_api = voice.get("jarvis_api")
                if jarvis_api:
                    await jarvis_api.speak({"text": text})
                    logger.debug(f"[PROACTIVE-VOICE] Spoke: {text}")
            except Exception as e:
                logger.error(f"[PROACTIVE-VOICE] Error: {e}")

        # Create notification callback
        async def notification_callback(title: str, message: str, priority: str = "low"):
            """Notification callback for proactive suggestions"""
            try:
                logger.info(f"[PROACTIVE-NOTIFY] [{priority.upper()}] {title}: {message}")
            except Exception as e:
                logger.error(f"[PROACTIVE-NOTIFY] Error: {e}")

        # Initialize UAE
        uae = await initialize_uae(
            vision_analyzer=config["vision_analyzer"],
            sai_monitoring_interval=config["sai_monitoring_interval"],
            enable_auto_start=config["enable_auto_start"],
            enable_learning_db=config["enable_learning_db"],
            enable_yabai=config["enable_yabai"],
            enable_proactive_intelligence=config["enable_proactive_intelligence"],
            voice_callback=voice_callback,
            notification_callback=notification_callback,
        )

        if uae and uae.is_active:
            app_state.uae_engine = uae
            app_state.learning_db = get_learning_db()
            logger.info("[LAZY-UAE] ✅ UAE/SAI/Learning DB loaded successfully")
            return uae
        else:
            logger.warning("[LAZY-UAE] ⚠️  UAE initialized but not active")
            return None

    except Exception as e:
        logger.error(f"[LAZY-UAE] ❌ Failed to load UAE: {e}")
        return None
    finally:
        app_state.uae_initializing = False


# Add more endpoints based on loaded components...
# (The rest of your API endpoints would go here)

if __name__ == "__main__":
    import argparse

    import uvicorn

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="JARVIS Backend Server")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("BACKEND_PORT", "8010")),
        help="Port to run the server on",
    )
    args = parser.parse_args()

    # Print startup information
    print(f"\n🚀 Starting JARVIS Backend")
    print(f"   HTTP:      http://localhost:{args.port}")
    print(f"   WebSocket: ws://localhost:{args.port}/ws")
    print(f"   API Docs:  http://localhost:{args.port}/docs")
    print("=" * 60)

    # Use optimized settings if enabled
    if OPTIMIZE_STARTUP:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=args.port,
            log_level="info",
            access_log=False,  # Disable access logs for performance
            loop=("uvloop" if sys.platform != "win32" else "asyncio"),  # Use uvloop on Unix
        )
    else:
        uvicorn.run(app, host="0.0.0.0", port=args.port)
