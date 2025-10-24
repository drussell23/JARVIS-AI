"""
JARVIS Hybrid Orchestrator
Main entry point for hybrid local/cloud architecture
Coordinates between local Mac and GCP backends with intelligent routing
"""
import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from backend.core.hybrid_backend_client import HybridBackendClient
from backend.core.hybrid_router import HybridRouter, RoutingContext, RouteDecision

logger = logging.getLogger(__name__)


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
        Execute command with intelligent routing

        Args:
            command: The command to execute
            command_type: Type of command (query, action, etc.)
            metadata: Additional metadata for routing

        Returns:
            Response from backend
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

        logger.info(
            f"📨 Request #{self.request_count}: '{command[:50]}...' "
            f"→ {decision.value} (rule: {route_metadata['rule']}, "
            f"confidence: {route_metadata['confidence']:.2f})"
        )

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
                    "metadata": metadata
                },
                capability=capability
            )

            # Add routing metadata to response
            result['routing'] = {
                'decision': decision.value,
                'backend': backend_name,
                **route_metadata
            }

            return result

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'routing': {
                    'decision': decision.value,
                    'backend': backend_name,
                    **route_metadata
                }
            }

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
