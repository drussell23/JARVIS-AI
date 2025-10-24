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
from typing import Dict, Any, Optional
from pathlib import Path

from backend.core.hybrid_backend_client import HybridBackendClient
from backend.core.hybrid_router import HybridRouter, RoutingContext, RouteDecision

logger = logging.getLogger(__name__)

# UAE/SAI/CAI Integration (lazy loaded)
_uae_engine = None
_sai_system = None
_cai_system = None
_learning_db = None


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
        metadata: Optional[Dict[str, Any]] = None
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
        context = RoutingContext(
            command=command,
            command_type=command_type,
            metadata=metadata
        )

        # Route the request
        decision, backend_name, route_metadata = self.router.route(context)
        rule = self._get_rule(route_metadata['rule'])

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
                    "intelligence_context": intelligence_context
                },
                capability=capability
            )

            # Add routing and intelligence metadata to response
            result['routing'] = {
                'decision': decision.value,
                'backend': backend_name,
                **route_metadata
            }
            result['intelligence'] = intelligence_context

            # Learn from execution (SAI)
            if rule and rule.get('use_sai'):
                await self._sai_learn_from_execution(command, result)

            # Store in learning database
            if rule and rule.get('use_learning_db'):
                await self._store_in_learning_db(command, result)

            return result

        except Exception as e:
            logger.error(f"Command execution failed: {e}")

            # SAI self-healing attempt
            if self._should_attempt_self_heal():
                logger.info("🔧 SAI attempting self-heal...")
                heal_result = await self._sai_self_heal(e, command)
                if heal_result.get('success'):
                    return heal_result

            return {
                'success': False,
                'error': str(e),
                'routing': {
                    'decision': decision.value,
                    'backend': backend_name,
                    **route_metadata
                },
                'intelligence': intelligence_context
            }

    def _get_rule(self, rule_name: str) -> Optional[Dict]:
        """Get routing rule by name"""
        rules = self.client.config['hybrid']['routing'].get('rules', [])
        for rule in rules:
            if rule.get('name') == rule_name:
                return rule
        return None

    async def _gather_intelligence_context(self, command: str, rule: Optional[Dict]) -> Dict[str, Any]:
        """
        Gather context from UAE/SAI/CAI/learning_database

        Returns enriched context for command execution
        """
        context = {}

        if not rule:
            return context

        # UAE: Unified Awareness Engine
        if rule.get('use_uae'):
            uae = _get_uae()
            if uae:
                try:
                    uae_context = await asyncio.to_thread(uae.get_current_context)
                    context['uae'] = {
                        'screen_state': uae_context.get('screen_locked', False),
                        'active_apps': uae_context.get('active_apps', []),
                        'current_space': uae_context.get('current_space'),
                        'network_status': uae_context.get('network_connected', True),
                    }
                    logger.debug(f"🧠 UAE context gathered")
                except Exception as e:
                    logger.warning(f"UAE context failed: {e}")

        # CAI: Context Awareness Intelligence
        if rule.get('use_cai'):
            cai = _get_cai()
            if cai:
                try:
                    intent = await asyncio.to_thread(cai.predict_intent, command)
                    context['cai'] = {
                        'predicted_intent': intent.get('intent'),
                        'confidence': intent.get('confidence', 0.0),
                        'suggested_action': intent.get('suggestion'),
                    }
                    logger.debug(f"🎯 CAI intent: {intent.get('intent')}")
                except Exception as e:
                    logger.warning(f"CAI prediction failed: {e}")

        # learning_database: Historical patterns
        if rule.get('use_learning_db'):
            learning_db = await _get_learning_db()
            if learning_db:
                try:
                    similar_patterns = await learning_db.find_similar_patterns(command)
                    context['learning_db'] = {
                        'similar_commands': [p.get('command') for p in similar_patterns[:3]],
                        'success_rate': sum(p.get('success', 0) for p in similar_patterns) / len(similar_patterns) if similar_patterns else 0.0,
                        'learned_preferences': similar_patterns[0].get('metadata') if similar_patterns else {}
                    }
                    logger.debug(f"📚 learning_db: Found {len(similar_patterns)} similar patterns")
                except Exception as e:
                    logger.warning(f"learning_db query failed: {e}")

        return context

    async def _sai_learn_from_execution(self, command: str, result: Dict):
        """SAI learns from command execution"""
        sai = _get_sai()
        if sai:
            try:
                await asyncio.to_thread(
                    sai.learn_from_execution,
                    command=command,
                    success=result.get('success', False),
                    response_time=result.get('response_time', 0),
                    metadata=result.get('routing', {})
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
                    command=command,
                    result=result,
                    timestamp=asyncio.get_event_loop().time()
                )
                logger.debug("💾 Stored in learning_database")
            except Exception as e:
                logger.warning(f"learning_db storage failed: {e}")

    def _should_attempt_self_heal(self) -> bool:
        """Check if SAI should attempt self-healing"""
        config = self.client.config.get('hybrid', {}).get('intelligence', {}).get('sai', {})
        return config.get('enabled', False) and config.get('self_healing', False)

    async def _sai_self_heal(self, error: Exception, command: str) -> Dict[str, Any]:
        """SAI attempts to self-heal from error"""
        sai = _get_sai()
        if sai:
            try:
                heal_result = await asyncio.to_thread(
                    sai.attempt_self_heal,
                    error=str(error),
                    context={'command': command}
                )
                if heal_result.get('healed'):
                    logger.info(f"✅ SAI self-heal successful: {heal_result.get('action')}")
                    # Retry command after heal
                    return await self.execute_command(command)
                else:
                    logger.warning(f"⚠️  SAI self-heal unsuccessful")
            except Exception as e:
                logger.error(f"SAI self-heal failed: {e}")

        return {'success': False, 'error': str(error)}

    async def execute_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute a natural language query"""
        return await self.execute_command(
            command=query,
            command_type="query",
            metadata=kwargs
        )

    async def execute_action(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute an action command"""
        return await self.execute_command(
            command=action,
            command_type="action",
            metadata=kwargs
        )

    async def capture_screen(self, **kwargs) -> Dict[str, Any]:
        """Capture screen (always routed to local)"""
        return await self.client.execute(
            path="/api/vision/capture",
            method="POST",
            data=kwargs,
            capability="vision_capture"
        )

    async def unlock_screen(self, **kwargs) -> Dict[str, Any]:
        """Unlock screen (always routed to local)"""
        return await self.client.execute(
            path="/api/unlock",
            method="POST",
            data=kwargs,
            capability="screen_unlock"
        )

    async def analyze_with_ml(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform ML analysis (routed to cloud for heavy processing)"""
        return await self.client.execute(
            path="/api/ml/analyze",
            method="POST",
            data=data,
            capability="ml_processing"
        )

    def _get_capability_from_decision(self, decision: RouteDecision, command: str) -> Optional[str]:
        """Determine capability requirement from routing decision"""
        if decision == RouteDecision.LOCAL:
            # Check for specific local capabilities
            command_lower = command.lower()
            if any(kw in command_lower for kw in ['screenshot', 'screen', 'capture', 'vision']):
                return "vision_capture"
            elif any(kw in command_lower for kw in ['unlock', 'password', 'login']):
                return "screen_unlock"
            elif any(kw in command_lower for kw in ['hey jarvis', 'voice', 'listen']):
                return "voice_activation"
            else:
                return "macos_automation"

        elif decision == RouteDecision.CLOUD:
            # Check for specific cloud capabilities
            command_lower = command.lower()
            if any(kw in command_lower for kw in ['analyze', 'understand', 'explain', 'summarize']):
                return "nlp_analysis"
            elif any(kw in command_lower for kw in ['chat', 'talk', 'conversation']):
                return "chatbot_inference"
            else:
                return "ml_processing"

        return None

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            'running': self.is_running,
            'request_count': self.request_count,
            'client_metrics': self.client.get_metrics(),
            'routing_analytics': self.router.get_analytics()
        }

    def get_backend_health(self) -> Dict[str, Any]:
        """Get health of all backends"""
        return {
            name: {
                'healthy': backend.health.healthy,
                'response_time': backend.health.response_time,
                'success_rate': backend.health.success_rate,
                'circuit_state': backend.circuit_breaker.state.value,
                'last_check': backend.health.last_check.isoformat()
            }
            for name, backend in self.client.backends.items()
        }

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()


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
