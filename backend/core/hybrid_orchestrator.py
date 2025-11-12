"""
JARVIS Hybrid Orchestrator - UAE/SAI/CAI Integrated
Main entry point for hybrid local/cloud architecture
Coordinates between local Mac and GCP backends with intelligent routing

Integrated Intelligence Systems:
- UAE (Unified Awareness Engine): Real-time context aggregation
- SAI (Self-Aware Intelligence): Self-healing and optimization
- CAI (Context Awareness Intelligence): Intent prediction
- learning_database: Persistent memory and pattern learning

This module provides the main orchestration layer for JARVIS, handling:
- Intelligent request routing between local and cloud backends
- Integration with multiple AI systems (UAE, SAI, CAI)
- Model lifecycle management and selection
- Cost optimization and resource management
- Health monitoring and automatic failover
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from backend.core.hybrid_backend_client import HybridBackendClient
from backend.core.hybrid_router import HybridRouter, RouteDecision, RoutingContext

logger = logging.getLogger(__name__)

# UAE/SAI/CAI Integration (lazy loaded)
_uae_engine = None
_sai_system = None
_cai_system = None
_learning_db = None

# Phase 3.1: Local LLM Integration (lazy loaded)
_llm_inference = None

# Phase 3.1+: Intelligent Model Management (lazy loaded)
_model_registry = None
_lifecycle_manager = None
_model_selector = None


def _get_uae():
    """Lazy load UAE (Unified Awareness Engine).
    
    Returns:
        UnifiedAwarenessEngine or None: UAE instance if available, None otherwise
    """
    global _uae_engine
    if _uae_engine is None:
        try:
            from intelligence.unified_awareness_engine import UnifiedAwarenessEngine

            _uae_engine = UnifiedAwarenessEngine()
            logger.info("✅ UAE loaded")
        except Exception as e:
            logger.warning(f"UAE not available: {e}")
    return _uae_engine


def _get_sai():
    """Lazy load SAI (Self-Aware Intelligence).
    
    Returns:
        SelfAwareIntelligence or None: SAI instance if available, None otherwise
    """
    global _sai_system
    if _sai_system is None:
        try:
            from intelligence.self_aware_intelligence import SelfAwareIntelligence

            _sai_system = SelfAwareIntelligence()
            logger.info("✅ SAI loaded")
        except Exception as e:
            logger.warning(f"SAI not available: {e}")
    return _sai_system


def _get_cai():
    """Lazy load CAI (Context Awareness Intelligence).
    
    Returns:
        ContextAwarenessIntelligence or None: CAI instance if available, None otherwise
    """
    global _cai_system
    if _cai_system is None:
        try:
            # CAI might be part of UAE or separate module
            from intelligence.context_awareness_intelligence import ContextAwarenessIntelligence

            _cai_system = ContextAwarenessIntelligence()
            logger.info("✅ CAI loaded")
        except Exception as e:
            logger.warning(f"CAI not available: {e}")
    return _cai_system


async def _get_learning_db():
    """Lazy load learning database for persistent memory.
    
    Returns:
        LearningDatabase or None: Learning database instance if available, None otherwise
    """
    global _learning_db
    if _learning_db is None:
        try:
            from intelligence.learning_database import get_learning_database

            _learning_db = await get_learning_database()
            logger.info("✅ learning_database loaded")
        except Exception as e:
            logger.warning(f"learning_database not available: {e}")
    return _learning_db


def _get_llm():
    """Lazy load Local LLM (Phase 3.1).
    
    Returns:
        LocalLLMInference or None: LLM inference instance if available, None otherwise
    """
    global _llm_inference
    if _llm_inference is None:
        try:
            from backend.intelligence.local_llm_inference import get_llm_inference

            _llm_inference = get_llm_inference()
            logger.info("✅ Local LLM (LLaMA 3.1 70B) ready for lazy loading")
        except Exception as e:
            logger.warning(f"Local LLM not available: {e}")
    return _llm_inference


def _get_model_registry():
    """Lazy load Model Registry (Phase 3.1+).
    
    Returns:
        ModelRegistry or None: Model registry instance if available, None otherwise
    """
    global _model_registry
    if _model_registry is None:
        try:
            from backend.intelligence.model_registry import get_model_registry

            _model_registry = get_model_registry()
            logger.info("✅ Model Registry initialized")
        except Exception as e:
            logger.warning(f"Model Registry not available: {e}")
    return _model_registry


def _get_lifecycle_manager():
    """Lazy load Model Lifecycle Manager (Phase 3.1+).
    
    Returns:
        ModelLifecycleManager or None: Lifecycle manager instance if available, None otherwise
    """
    global _lifecycle_manager
    if _lifecycle_manager is None:
        try:
            from backend.intelligence.model_lifecycle_manager import get_lifecycle_manager

            _lifecycle_manager = get_lifecycle_manager()
            logger.info("✅ Model Lifecycle Manager initialized")
        except Exception as e:
            logger.warning(f"Model Lifecycle Manager not available: {e}")
    return _lifecycle_manager


def _get_model_selector():
    """Lazy load Intelligent Model Selector (Phase 3.1+).
    
    Returns:
        IntelligentModelSelector or None: Model selector instance if available, None otherwise
    """
    global _model_selector
    if _model_selector is None:
        try:
            from backend.intelligence.model_selector import get_model_selector

            _model_selector = get_model_selector()
            logger.info("✅ Intelligent Model Selector initialized")
        except Exception as e:
            logger.warning(f"Model Selector not available: {e}")
    return _model_selector


class HybridOrchestrator:
    """Main orchestrator for JARVIS hybrid architecture.
    
    This class coordinates between local Mac and GCP backends, providing:
    - Intelligent request routing based on context and capabilities
    - Integration with UAE/SAI/CAI intelligence systems
    - Automatic failover and load balancing
    - Health monitoring and performance analytics
    - Cost optimization through idle time tracking
    - Model lifecycle management and selection
    
    Attributes:
        config_path: Path to hybrid configuration file
        client: HybridBackendClient for backend communication
        router: HybridRouter for intelligent routing decisions
        is_running: Whether the orchestrator is currently running
        request_count: Total number of requests processed
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the HybridOrchestrator.
        
        Args:
            config_path: Path to configuration file. Defaults to backend/core/hybrid_config.yaml
        """
        self.config_path = config_path or "backend/core/hybrid_config.yaml"

        # Initialize components
        self.client = HybridBackendClient(self.config_path)
        self.router = HybridRouter(self.client.config)

        # State
        self.is_running = False
        self.request_count = 0

        logger.info("🎭 HybridOrchestrator initialized")

    async def start(self) -> None:
        """Start the orchestrator and all its components.
        
        Raises:
            RuntimeError: If orchestrator is already running
        """
        if self.is_running:
            logger.warning("Orchestrator already running")
            return

        logger.info("🚀 Starting HybridOrchestrator...")
        await self.client.start()

        # Start Model Lifecycle Manager (Phase 3.1+)
        lifecycle_manager = _get_lifecycle_manager()
        if lifecycle_manager:
            await lifecycle_manager.start()
            logger.info("✅ Model Lifecycle Manager started")

        # Register backend capabilities from discovered services
        self._register_backend_capabilities()

        self.is_running = True
        logger.info("✅ HybridOrchestrator started")

    async def stop(self) -> None:
        """Stop the orchestrator and clean up resources."""
        if not self.is_running:
            return

        logger.info("🛑 Stopping HybridOrchestrator...")
        await self.client.stop()
        self.is_running = False
        logger.info("✅ HybridOrchestrator stopped")

    async def execute_command(
        self,
        command: str,
        command_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute command with intelligent routing and UAE/SAI/CAI integration.

        Args:
            command: The command to execute
            command_type: Type of command (query, action, etc.)
            metadata: Additional metadata for routing

        Returns:
            Dict containing:
                - success: Whether execution succeeded
                - result: Command execution result
                - routing: Routing decision metadata
                - intelligence: Context from UAE/SAI/CAI systems
                - error: Error message if execution failed

        Raises:
            Exception: If command execution fails and self-healing is unsuccessful
        """
        if not self.is_running:
            await self.start()

        self.request_count += 1

        # Create routing context
        context = RoutingContext(command=command, command_type=command_type, metadata=metadata)

        # Route the request
        decision, backend_name, route_metadata = self.router.route(context)
        rule = self._get_rule(route_metadata["rule"])

        logger.info(
            f"📨 Request #{self.request_count}: '{command[:50]}...' "
            f"→ {decision.value} (rule: {route_metadata['rule']}, "
            f"confidence: {route_metadata['confidence']:.2f})"
        )

        # Enrich with intelligence systems
        intelligence_context = await self._gather_intelligence_context(command, rule)

        # Merge with existing metadata
        enhanced_metadata = {**(metadata or {}), **intelligence_context}

        # Determine capability based on decision
        capability = self._get_capability_from_decision(decision, command)

        # Execute request
        try:
            result = await self.client.execute(
                path="/api/command",
                method="POST",
                data={
                    "command": command,
                    "command_type": command_type,
                    "metadata": enhanced_metadata,
                    "intelligence_context": intelligence_context,
                },
                capability=capability,
            )

            # Add routing and intelligence metadata to response
            result["routing"] = {
                "decision": decision.value,
                "backend": backend_name,
                **route_metadata,
            }
            result["intelligence"] = intelligence_context

            # ============== PHASE 2.5: Record Backend Activity ==============
            # Record activity for GCP idle time tracking
            if backend_name:
                self.router.record_backend_activity(backend_name)
                logger.debug(f"📝 Recorded activity for '{backend_name}'")
            # ================================================================

            # Learn from execution (SAI)
            if rule and rule.get("use_sai"):
                await self._sai_learn_from_execution(command, result)

            # Store in learning database
            if rule and rule.get("use_learning_db"):
                await self._store_in_learning_db(command, result)

            return result

        except Exception as e:
            logger.error(f"Command execution failed: {e}")

            # SAI self-healing attempt
            if self._should_attempt_self_heal():
                logger.info("🔧 SAI attempting self-heal...")
                heal_result = await self._sai_self_heal(e, command)
                if heal_result.get("success"):
                    return heal_result

            return {
                "success": False,
                "error": str(e),
                "routing": {"decision": decision.value, "backend": backend_name, **route_metadata},
                "intelligence": intelligence_context,
            }

    def _get_rule(self, rule_name: str) -> Optional[Dict]:
        """Get routing rule by name.
        
        Args:
            rule_name: Name of the routing rule to retrieve
            
        Returns:
            Dict containing rule configuration or None if not found
        """
        rules = self.client.config["hybrid"]["routing"].get("rules", [])
        for rule in rules:
            if rule.get("name") == rule_name:
                return rule
        return None

    async def _gather_intelligence_context(
        self, command: str, rule: Optional[Dict]
    ) -> Dict[str, Any]:
        """Gather context from UAE/SAI/CAI/learning_database systems.

        Args:
            command: The command being executed
            rule: Routing rule configuration

        Returns:
            Dict containing enriched context from intelligence systems:
                - uae: Unified Awareness Engine context
                - cai: Context Awareness Intelligence predictions
                - learning_db: Historical patterns and preferences
                - llm: Local LLM availability status
        """
        context = {}

        if not rule:
            return context

        # UAE: Unified Awareness Engine
        if rule.get("use_uae"):
            uae = _get_uae()
            if uae:
                try:
                    uae_context = await asyncio.to_thread(uae.get_current_context)
                    context["uae"] = {
                        "screen_state": uae_context.get("screen_locked", False),
                        "active_apps": uae_context.get("active_apps", []),
                        "current_space": uae_context.get("current_space"),
                        "network_status": uae_context.get("network_connected", True),
                    }
                    logger.debug(f"🧠 UAE context gathered")
                except Exception as e:
                    logger.warning(f"UAE context failed: {e}")

        # CAI: Context Awareness Intelligence
        if rule.get("use_cai"):
            cai = _get_cai()
            if cai:
                try:
                    intent = await asyncio.to_thread(cai.predict_intent, command)
                    context["cai"] = {
                        "predicted_intent": intent.get("intent"),
                        "confidence": intent.get("confidence", 0.0),
                        "suggested_action": intent.get("suggestion"),
                    }
                    logger.debug(f"🎯 CAI intent: {intent.get('intent')}")
                except Exception as e:
                    logger.warning(f"CAI prediction failed: {e}")

        # learning_database: Historical patterns
        if rule.get("use_learning_db"):
            learning_db = await _get_learning_db()
            if learning_db:
                try:
                    similar_patterns = await learning_db.find_similar_patterns(command)
                    context["learning_db"] = {
                        "similar_commands": [p.get("command") for p in similar_patterns[:3]],
                        "success_rate": (
                            sum(p.get("success", 0) for p in similar_patterns)
                            / len(similar_patterns)
                            if similar_patterns
                            else 0.0
                        ),
                        "learned_preferences": (
                            similar_patterns[0].get("metadata") if similar_patterns else {}
                        ),
                    }
                    logger.debug(f"📚 learning_db: Found {len(similar_patterns)} similar patterns")
                except Exception as e:
                    logger.warning(f"learning_db query failed: {e}")

        # ============== PHASE 3.1: Local LLM Integration ==============
        # LLM: Local Language Model (LLaMA 3.1 70B on GCP)
        if rule.get("use_llm"):
            llm = _get_llm()
            if llm:
                try:
                    # Check if LLM is available and started
                    if not llm.is_running:
                        await llm.start()

                    # Get LLM status for routing decisions
                    llm_status = llm.get_status()
                    context["llm"] = {
                        "available": llm_status["model_state"] == "loaded",
                        "model_name": llm_status.get("model_name"),
                        "health": llm_status.get("health", {}),
                        "avg_inference_time": llm_status["health"].get("avg_inference_time", 0),
                    }
                    logger.debug(
                        f"🤖 LLM context: {llm_status['model_state']} "
                        f"({llm_status['health'].get('success_rate', 0):.1%} success rate)"
                    )
                except Exception as e:
                    logger.warning(f"LLM context failed: {e}")
        # ===============================================================

        return context

    async def _sai_learn_from_execution(self, command: str, result: Dict) -> None:
        """Allow SAI to learn from command execution results.
        
        Args:
            command: The executed command
            result: Execution result containing success status and metadata
        """
        sai = _get_sai()
        if sai:
            try:
                await asyncio.to_thread(
                    sai.learn_from_execution,
                    command=command,
                    success=result.get("success", False),
                    response_time=result.get("response_time", 0),
                    metadata=result.get("routing", {}),
                )
                logger.debug("🤖 SAI learned from execution")
            except Exception as e:
                logger.warning(f"SAI learning failed: {e}")

    async def _store_in_learning_db(self, command: str, result: Dict) -> None:
        """Store execution results in learning database for future reference.
        
        Args:
            command: The executed command
            result: Execution result to store
        """
        learning_db = await _get_learning_db()
        if learning_db:
            try:
                await learning_db.store_interaction(
                    command=command, result=result, timestamp=asyncio.get_event_loop().time()
                )
                logger.debug("💾 Stored in learning_database")
            except Exception as e:
                logger.warning(f"learning_db storage failed: {e}")

    def _should_attempt_self_heal(self) -> bool:
        """Check if SAI should attempt self-healing based on configuration.
        
        Returns:
            bool: True if self-healing should be attempted
        """
        config = self.client.config.get("hybrid", {}).get("intelligence", {}).get("sai", {})
        return config.get("enabled", False) and config.get("self_healing", False)

    async def _sai_self_heal(self, error: Exception, command: str) -> Dict[str, Any]:
        """Attempt SAI self-healing from execution error.
        
        Args:
            error: The exception that occurred
            command: The command that failed
            
        Returns:
            Dict containing heal result and potentially retried command result
        """
        sai = _get_sai()
        if sai:
            try:
                heal_result = await asyncio.to_thread(
                    sai.attempt_self_heal, error=str(error), context={"command": command}
                )
                if heal_result.get("healed"):
                    logger.info(f"✅ SAI self-heal successful: {heal_result.get('action')}")
                    # Retry command after heal
                    return await self.execute_command(command)
                else:
                    logger.warning(f"⚠️  SAI self-heal unsuccessful")
            except Exception as e:
                logger.error(f"SAI self-heal failed: {e}")

        return {"success": False, "error": str(error)}

    # ============== PHASE 3.1: LLM Helper Methods ==============

    async def execute_with_intelligent_model_selection(
        self,
        query: str,
        intent: Optional[str] = None,
        required_capabilities: Optional[set] = None,
        context: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute a query with intelligent model selection.

        The model selector will:
        1. Analyze the query (CAI integration)
        2. Consider context and focus level (UAE integration)
        3. Check RAM availability (SAI integration)
        4. Score all capable models
        5. Select the best option
        6. Load model if needed (lifecycle manager)
        7. Execute the query

        Args:
            query: User's query text
            intent: Pre-classified intent (optional)
            required_capabilities: Required capabilities (optional)
            context: Additional context from UAE/SAI/CAI
            **kwargs: Additional parameters for model

        Returns:
            Dict with result and metadata:
                - success: Whether execution succeeded
                - text: Generated response text
                - model_used: Name of the model that was used
                - fallback_used: Whether a fallback model was used
                - error: Error message if execution failed
        """
        model_selector = _get_model_selector()
        if not model_selector:
            logger.warning("Model selector not available, falling back to direct LLM")
            return await self.execute_llm_inference(query, **kwargs)

        lifecycle_manager = _get_lifecycle_manager()
        if not lifecycle_manager:
            logger.warning("Lifecycle manager not available")
            return await self.execute_llm_inference(query, **kwargs)

        try:
            # Select best model with fallback chain
            primary_model, fallbacks = await model_selector.select_with_fallback(
                query=query,
                intent=intent,
                required_capabilities=required_capabilities,
                context=context,
            )

            if not primary_model:
                logger.error("No suitable model found for query")
                return {"success": False, "error": "No suitable model found", "query": query}

            # Try primary model
            try:
                result = await self._execute_with_model(
                    primary_model, query, lifecycle_manager, **kwargs
                )
                if result["success"]:
                    result["model_used"] = primary_model.name
                    return result
            except Exception as e:
                logger.warning(f"Primary model {primary_model.name} failed: {e}")

            # Try fallbacks
            for fallback_model in fallbacks:
                try:
                    logger.info(f"Trying fallback model: {fallback_model.name}")
                    result = await self._execute_with_model(
                        fallback_model, query, lifecycle_manager, **kwargs
                    )
                    if result["success"]:
                        result["model_used"] = fallback_model.name
                        result["fallback_used"] = True
                        return result
                except Exception as e:
                    logger.warning(f"Fallback model {fallback_model.name} failed: {e}")
                    continue

            return {"success": False, "error": "All models failed", "query": query}

        except Exception as e:
            logger.error(f"Error in intelligent model selection: {e}")
            return {"success": False, "error": str(e), "query": query}

    async def _execute_with_model(
        self, model_def, query: str, lifecycle_manager, **kwargs
    ) -> Dict[str, Any]:
        """Execute query with a specific model.
        
        Args:
            model_def: Model definition containing name, type, and capabilities
            query: Query to execute
            lifecycle_manager: Model lifecycle manager instance
            **kwargs: Additional parameters for model execution
            
        Returns:
            Dict containing execution result
        """
        # Load model if needed
        model_instance = await lifecycle_manager.get_model(
            model_def.name, required_by="orchestrator"
        )

        if not model_instance:
            return {"success": False, "error": f"Failed to load {model_def.name}"}

        # Execute based on model type
        if model_def.model_type == "llm":
            # LLM inference
            if model_def.name == "llama_70b":
                result = await self.execute_llm_inference(query, **kwargs)
            elif model_def.name == "claude_api":
                # TODO: Add Claude API execution
                result = {
                    "success": True,
                    "text": f"[Claude API would process: {query}]",
                    "model": "claude_api",
                }
            else:
                result = {"success": False, "error": f"Unknown LLM: {model_def.name}"}

        elif model_def.model_type == "vision":
            # Vision model execution
            # Check if this is a YOLO model
            if model_def.name.startswith("yolov8"):
                try:
                    from backend.vision.yolo_vision_detector import get_yolo_detector

                    yolo_detector = get_yolo_detector()

                    # Extract image from query if it's multimodal content
                    image_data = kwargs.get("image_data")
                    if not image_data and isinstance(query, list):
                        # Extract image from multimodal content
                        for content in query:
                            if isinstance(content, dict) and content.get("type") == "image":
                                image_data = content.get("source", {}).get("data")
                                break

                    if not image_data:
                        return {
                            "success": False,
                            "error": "No image data provided for YOLO detection",
                        }

                    # Decode base64 image if needed
                    if isinstance(image_data, str):
                        import base64
                        from io import BytesIO

                        from PIL import Image

                        image_bytes = base64.b64decode(image_data)
                        image = Image.open(BytesIO(image_bytes))
                    else:
                        image = image_data

                    # Perform detection
                    detection_result = await yolo_detector.detect_ui_elements(image)

                    # Format result
                    detections_list = [
                        {
                            "class": det.class_name,
                            "confidence": det.confidence,
                            "bbox": {
                                "x": det.bbox.x,
                                "y": det.bbox.y,
                                "width": det.bbox.width,
                                "height": det.bbox.height,
                            },
                        }
                        for det in detection_result.detections
                    ]

                    result = {
                        "success": True,
                        "text": f"Detected {len(detections_list)} objects",
                        "detections": detections_list,
                        "model": model_def.name,
                    }

                except Exception as e:
                    logger.error(f"YOLO detection failed: {e}")
                    result = {"success": False, "error": f"YOLO detection failed: {e}"}
            else:
                # Other vision models (Claude Vision)
                result = {
                    "success": True,
                    "text": f"[Vision model {model_def.name} would process the query]",
                    "model": model_def.name,
                }

        elif model_def.model_type == "embedding":
            # Semantic search
            result = {
                "success": True,
                "text": f"[Semantic search would process: {query}]",
                "model": model_def.name,
            }

        else:
            result = {"success": False, "error": f"Unknown model type: {model_def.model_type}"}

        return result

    async def execute_llm_inference(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute LLM inference with LLaMA 3.1 70B.

        Args:
            prompt: Text prompt for generation
            max_tokens: Max tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            **kwargs: Additional generation parameters

        Returns:
            Dict containing:
                - success: Whether generation succeeded
                - text: Generated text response
                - model: Model name used
                - backend: Backend that processed the request
                - error: Error message if generation failed
        """
        llm = _get_llm()
        if not llm:
            return {
                "success": False,
                "error": "Local LLM not available",
                "text": "",
            }

        try:
            # Start LLM if not running
            if not llm.is_running:
                await llm.start()

            # Generate text
            generated_text = await llm.generate(
                prompt, max_tokens=max_tokens, temperature=temperature, **kwargs
            )

            return {
                "success": True,
                "text": generated_text,
                "model": "llama-3.1-70b",
                "backend": "gcp",
            }

        except Exception as e:
            logger.error(f"LLM inference failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
            }

    async def classify_intent_with_llm(self, command: str) -> Dict[str, Any]:
        """Use LLM to classify user intent.
        
        Args:
            command: User command to classify
            
        Returns:
            Dict containing:
                - success: Whether classification succeeded
                - intent: Classified intent
                - confidence: Confidence score (0.0-1.0)
                - entities: Extracted entities
                - action: Suggested action
                - error: Error message if classification failed
        """
        prompt = f"""Classify the intent of this command. Respond with JSON format:
{{"intent": "...", "confidence": 0.0-1.0, "entities": [...], "action": "..."}}

Command: "{command}"

Classification:"""

        result = await self.execute_llm_inference(prompt, max_tokens=100, temperature=0.3)

        if result["success"]:
            try:
                import json

                # Parse JSON from response
                text = result["text"].strip()
                # Extract JSON if wrapped in other text
                if "{" in text:
                    json_start = text.index("{")
                    json_end = text.rindex("}") + 1
                    text = text[json_start:json_end]

                classification = json.loads(text)
                return {
                    "success": True,
                    "intent": classification.get("intent"),
                    "confidence": classification.get("confidence", 0.8),
                    "entities": classification.get("entities", []),
                    "action": classification.get("action"),
                }
            except Exception as e:
                logger.warning(f"Failed to parse LLM classification: {e}")
                return {"success": False, "error": str(e)}

        return result


# ============================================================================
# GLOBAL ORCHESTRATOR INSTANCE
# ============================================================================

_global_orchestrator: Optional[HybridOrchestrator] = None


def get_orchestrator() -> HybridOrchestrator:
    """Get or create the global Hybrid Orchestrator instance.
    
    This function provides a singleton pattern for accessing the hybrid
    orchestrator throughout the application. The orchestrator coordinates
    between local and cloud backends with intelligent routing.
    
    Returns:
        HybridOrchestrator: The global orchestrator instance
        
    Example:
        >>> from backend.core.hybrid_orchestrator import get_orchestrator
        >>> orchestrator = get_orchestrator()
        >>> result = await orchestrator.execute_command("open safari")
    """
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = HybridOrchestrator()
        logger.info("✅ Global HybridOrchestrator instance created")
    return _global_orchestrator


async def get_orchestrator_async() -> HybridOrchestrator:
    """Get or create and start the global Hybrid Orchestrator instance.
    
    Similar to get_orchestrator() but ensures the orchestrator is started
    before returning. Useful for async contexts where you want to guarantee
    the orchestrator is ready to use.
    
    Returns:
        HybridOrchestrator: The global orchestrator instance (started)
        
    Example:
        >>> orchestrator = await get_orchestrator_async()
        >>> result = await orchestrator.execute_command("search for AI news")
    """
    orchestrator = get_orchestrator()
    if not orchestrator.is_running:
        await orchestrator.start()
    return orchestrator
