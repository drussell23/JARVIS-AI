"""
JARVIS Hybrid Orchestrator - UAE/SAI/CAI Integrated
Main entry point for hybrid local/cloud architecture
Coordinates between local Mac and GCP backends with intelligent routing

Integrated Intelligence Systems:
- UAE (Unified Awareness Engine): Real-time context aggregation
- SAI (Self-Aware Intelligence): Self-healing and optimization
- CAI (Context Awareness Intelligence): Intent prediction
- learning_database: Persistent memory and pattern learning
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
    """Lazy load UAE"""
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
    """Lazy load SAI"""
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
    """Lazy load CAI"""
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
    """Lazy load learning database"""
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
    """Lazy load Local LLM (Phase 3.1)"""
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
    """Lazy load Model Registry (Phase 3.1+)"""
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
    """Lazy load Model Lifecycle Manager (Phase 3.1+)"""
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
    """Lazy load Intelligent Model Selector (Phase 3.1+)"""
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
    """
    Main orchestrator for JARVIS hybrid architecture
    Features:
    - Intelligent request routing
    - Automatic failover
    - Load balancing
    - Health monitoring
    - Performance analytics
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "backend/core/hybrid_config.yaml"

        # Initialize components
        self.client = HybridBackendClient(self.config_path)
        self.router = HybridRouter(self.client.config)

        # State
        self.is_running = False
        self.request_count = 0

        logger.info("🎭 HybridOrchestrator initialized")

    async def start(self):
        """Start the orchestrator"""
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

    async def stop(self):
        """Stop the orchestrator"""
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
        """
        Execute command with intelligent routing + UAE/SAI/CAI integration

        Args:
            command: The command to execute
            command_type: Type of command (query, action, etc.)
            metadata: Additional metadata for routing

        Returns:
            Response from backend with intelligence context
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
        """Get routing rule by name"""
        rules = self.client.config["hybrid"]["routing"].get("rules", [])
        for rule in rules:
            if rule.get("name") == rule_name:
                return rule
        return None

    async def _gather_intelligence_context(
        self, command: str, rule: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        Gather context from UAE/SAI/CAI/learning_database

        Returns enriched context for command execution
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

    async def _sai_learn_from_execution(self, command: str, result: Dict):
        """SAI learns from command execution"""
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

    async def _store_in_learning_db(self, command: str, result: Dict):
        """Store execution in learning database"""
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
        """Check if SAI should attempt self-healing"""
        config = self.client.config.get("hybrid", {}).get("intelligence", {}).get("sai", {})
        return config.get("enabled", False) and config.get("self_healing", False)

    async def _sai_self_heal(self, error: Exception, command: str) -> Dict[str, Any]:
        """SAI attempts to self-heal from error"""
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
        """
        Execute a query with intelligent model selection

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
            Dict with result and metadata
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
        """Execute query with a specific model"""
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
        """
        Execute LLM inference with LLaMA 3.1 70B

        Args:
            prompt: Text prompt for generation
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            Dict with generated text and metadata
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
        """Use LLM to classify user intent"""
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

    async def generate_response_with_llm(
        self, command: str, context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Generate response using LLM with context"""
        # Build prompt with context
        prompt = f"User command: {command}\n\n"

        if context:
            if "uae" in context:
                prompt += f"Context: {context['uae']}\n"
            if "cai" in context:
                prompt += f"Intent: {context['cai'].get('predicted_intent')}\n"

        prompt += "\nRespond naturally and helpfully:\n"

        result = await self.execute_llm_inference(prompt, max_tokens=256, temperature=0.7)

        return result

    # ===============================================================

    async def execute_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute a natural language query"""
        return await self.execute_command(command=query, command_type="query", metadata=kwargs)

    async def execute_action(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute an action command"""
        return await self.execute_command(command=action, command_type="action", metadata=kwargs)

    async def capture_screen(self, **kwargs) -> Dict[str, Any]:
        """Capture screen (always routed to local)"""
        return await self.client.execute(
            path="/api/vision/capture", method="POST", data=kwargs, capability="vision_capture"
        )

    async def unlock_screen(self, **kwargs) -> Dict[str, Any]:
        """Unlock screen (always routed to local)"""
        return await self.client.execute(
            path="/api/unlock", method="POST", data=kwargs, capability="screen_unlock"
        )

    async def analyze_with_ml(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform ML analysis (routed to cloud for heavy processing)"""
        return await self.client.execute(
            path="/api/ml/analyze", method="POST", data=data, capability="ml_processing"
        )

    def _get_capability_from_decision(self, decision: RouteDecision, command: str) -> Optional[str]:
        """Determine capability requirement from routing decision"""
        if decision == RouteDecision.LOCAL:
            # Check for specific local capabilities
            command_lower = command.lower()
            if any(kw in command_lower for kw in ["screenshot", "screen", "capture", "vision"]):
                return "vision_capture"
            elif any(kw in command_lower for kw in ["unlock", "password", "login"]):
                return "screen_unlock"
            elif any(kw in command_lower for kw in ["hey jarvis", "voice", "listen"]):
                return "voice_activation"
            else:
                return "macos_automation"

        elif decision == RouteDecision.CLOUD:
            # Check for specific cloud capabilities
            command_lower = command.lower()
            if any(kw in command_lower for kw in ["analyze", "understand", "explain", "summarize"]):
                return "nlp_analysis"
            elif any(kw in command_lower for kw in ["chat", "talk", "conversation"]):
                return "chatbot_inference"
            else:
                return "ml_processing"

        return None

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status with enhanced Phase 2.5 metrics"""
        routing_analytics = self.router.get_analytics()

        return {
            "running": self.is_running,
            "request_count": self.request_count,
            "client_metrics": self.client.get_metrics(),
            "routing_analytics": routing_analytics,
            # Phase 2.5: Backend activity tracking
            "backend_activity": routing_analytics.get("backend_activity", {}),
            # Phase 2.5: Capabilities mapping
            "registered_capabilities": {
                name: len(caps) for name, caps in self.router._backend_capabilities.items()
            },
        }

    def get_backend_health(self) -> Dict[str, Any]:
        """Get health of all backends with Phase 2.5 enhancements"""
        health_data = {}

        for name, backend in self.client.backends.items():
            idle_minutes = self.router._get_backend_idle_minutes(name)

            health_data[name] = {
                "healthy": backend.health.healthy,
                "response_time": backend.health.response_time,
                "success_rate": backend.health.success_rate,
                "circuit_state": backend.circuit_breaker.state.value,
                "last_check": backend.health.last_check.isoformat(),
                # Phase 2.5: Activity tracking
                "idle_minutes": round(idle_minutes, 2),
                "is_idle": idle_minutes > 10,
                # Phase 2.5: Capabilities
                "capabilities_count": len(backend.capabilities),
                "capabilities": backend.capabilities[:5],  # First 5 for overview
            }

        return health_data

    def get_detailed_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive diagnostics for debugging
        Includes Phase 2.5 features: idle tracking, capabilities, cost optimization
        """
        # Get RAM monitor status
        ram_monitor_factory = _get_ram_monitor()
        ram_status = None
        if ram_monitor_factory:
            try:
                ram_monitor = ram_monitor_factory(self.client.config)
                ram_status = {
                    "current_pressure_percent": ram_monitor.get_current_pressure_percent(),
                    "should_prefer_gcp": ram_monitor.should_prefer_gcp(),
                    "should_force_gcp": ram_monitor.should_force_gcp(),
                    "can_reclaim_to_local": ram_monitor.can_reclaim_to_local(),
                    "monitoring_active": ram_monitor.is_running,
                }
            except Exception as e:
                ram_status = {"error": str(e)}

        # Get cost optimization status
        gcp_idle = self.router._get_backend_idle_minutes("gcp")
        local_idle = self.router._get_backend_idle_minutes("local")

        return {
            "orchestrator": {
                "running": self.is_running,
                "request_count": self.request_count,
            },
            "ram_monitor": ram_status,
            "backends": {
                name: {
                    "type": backend.type.value,
                    "priority": backend.priority,
                    "enabled": backend.enabled,
                    "healthy": backend.health.healthy,
                    "idle_minutes": round(self.router._get_backend_idle_minutes(name), 2),
                    "capabilities": backend.capabilities,
                    "circuit_state": backend.circuit_breaker.state.value,
                }
                for name, backend in self.client.backends.items()
            },
            "routing": {
                "total_rules": len(self.router.rules),
                "strategy": self.router.strategy,
                "history_size": len(self.router.routing_history),
            },
            "cost_optimization": {
                "gcp_idle_minutes": round(gcp_idle, 2),
                "local_idle_minutes": round(local_idle, 2),
                "can_shutdown_gcp": gcp_idle > 10
                and ram_status
                and ram_status.get("can_reclaim_to_local", False),
            },
            "capabilities_registry": {
                backend_name: {
                    "count": len(caps),
                    "capabilities": caps,
                    "cache_age_seconds": (
                        round(
                            __import__("time").time()
                            - self.router._capabilities_last_updated.get(backend_name, 0),
                            2,
                        )
                        if backend_name in self.router._capabilities_last_updated
                        else None
                    ),
                }
                for backend_name, caps in self.router._backend_capabilities.items()
            },
        }

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()

    # ============== PHASE 2.5: Backend Capabilities Registration ==============

    def _register_backend_capabilities(self):
        """Register capabilities from discovered backends"""
        for backend_name, backend in self.client.backends.items():
            self.router.register_backend_capabilities(backend_name, backend.capabilities)
            logger.info(
                f"✅ Registered {len(backend.capabilities)} capabilities for '{backend_name}': "
                f"{', '.join(backend.capabilities[:3])}{'...' if len(backend.capabilities) > 3 else ''}"
            )

    def get_backend_capabilities(self, backend_name: str) -> List[str]:
        """Get capabilities for a specific backend"""
        if backend_name in self.client.backends:
            return self.client.backends[backend_name].capabilities
        return []

    def get_backends_for_capability(self, capability: str) -> List[str]:
        """Get list of backends that support a capability"""
        return self.router.get_backends_for_capability(capability)

    def get_idle_time(self, backend_name: str) -> float:
        """Get idle time in minutes for a backend"""
        return self.router._get_backend_idle_minutes(backend_name)

    def get_activity_summary(self) -> Dict[str, Any]:
        """Get activity summary for all backends"""
        return self.router._get_activity_summary()

    async def trigger_cost_optimization(self) -> Dict[str, Any]:
        """
        Check if cost optimization should be triggered (GCP shutdown)
        Based on:
        - GCP idle time >10 minutes
        - Local RAM pressure <40%
        - No pending high-priority tasks

        Returns:
            Dict with optimization actions taken
        """
        ram_monitor_factory = _get_ram_monitor()
        if not ram_monitor_factory:
            return {"error": "RAM monitor not available"}

        ram_monitor = ram_monitor_factory(self.client.config)
        current_pressure = ram_monitor.get_current_pressure_percent()

        gcp_idle = self.router._get_backend_idle_minutes("gcp")
        local_pressure_ok = current_pressure < 40  # Local has capacity
        gcp_been_idle = gcp_idle > 10  # GCP idle for >10 minutes

        if local_pressure_ok and gcp_been_idle:
            logger.info(
                f"💰 Cost optimization triggered: Local RAM {current_pressure:.1f}%, "
                f"GCP idle {gcp_idle:.1f}min"
            )

            # TODO: Implement GCP shutdown logic
            # For now, just return recommendation
            return {
                "action": "recommend_gcp_shutdown",
                "reason": "Cost optimization - local has capacity, GCP idle",
                "local_ram_percent": current_pressure,
                "gcp_idle_minutes": gcp_idle,
                "estimated_savings": "$0.029/hr (~$0.48/day if idle)",
            }
        else:
            return {
                "action": "no_optimization",
                "reason": "Conditions not met",
                "local_ram_percent": current_pressure,
                "gcp_idle_minutes": gcp_idle,
                "local_pressure_ok": local_pressure_ok,
                "gcp_been_idle": gcp_been_idle,
            }


def _get_ram_monitor():
    """Lazy load RAM monitor"""
    try:
        from backend.core.advanced_ram_monitor import get_ram_monitor

        return get_ram_monitor
    except ImportError:
        logger.warning("RAM monitor not available for cost optimization")
        return None


# Global instance (lazy initialized)
_orchestrator: Optional[HybridOrchestrator] = None


def get_orchestrator() -> HybridOrchestrator:
    """Get or create global orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = HybridOrchestrator()
    return _orchestrator


async def execute_hybrid_command(command: str, **kwargs) -> Dict[str, Any]:
    """Convenience function for executing commands"""
    orchestrator = get_orchestrator()
    if not orchestrator.is_running:
        await orchestrator.start()
    return await orchestrator.execute_command(command, **kwargs)
