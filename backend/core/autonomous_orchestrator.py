#!/usr/bin/env python3
"""
Autonomous Orchestrator for JARVIS
Self-discovering, self-healing service orchestration
"""

import asyncio
import logging
from typing import Dict, Optional, Any
from datetime import datetime
import aiohttp
from collections import defaultdict

logger = logging.getLogger(__name__)


class ServiceInfo:
    """Information about a discovered service"""

    def __init__(self, name: str, port: int, protocol: str = "http"):
        self.name = name
        self.port = port
        self.protocol = protocol
        self.health_score = 1.0
        self.last_check = None
        self.url = f"{protocol}://localhost:{port}"


class AutonomousOrchestrator:
    """
    Autonomous service orchestrator that discovers and manages services
    """

    def __init__(self):
        self.services: Dict[str, ServiceInfo] = {}
        self._running = False
        self._health_check_task = None
        self.health_history = defaultdict(list)

    async def start(self):
        """Start the orchestrator"""
        self._running = True
        logger.info("✅ Autonomous orchestrator started")

        # Start health check background task
        self._health_check_task = asyncio.create_task(self._periodic_health_check())

    async def stop(self):
        """Stop the orchestrator"""
        self._running = False

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        logger.info("Autonomous orchestrator stopped")

    async def discover_service(self, name: str, port: int, check_health: bool = True) -> Optional[Dict[str, Any]]:
        """
        Discover a service at the given port

        Args:
            name: Service name
            port: Port to check
            check_health: Whether to verify service is healthy

        Returns:
            Service info dict if found and healthy, None otherwise
        """
        if check_health:
            is_healthy = await self._check_service_health(port)
            if not is_healthy:
                logger.warning(f"Service {name} on port {port} is not healthy")
                return None

        service_info = {
            "name": name,
            "protocol": "http",
            "port": port,
            "url": f"http://localhost:{port}"
        }

        return service_info

    async def register_service(self, name: str, port: int, protocol: str = "http") -> bool:
        """
        Register a service with the orchestrator

        Args:
            name: Service name
            port: Service port
            protocol: Service protocol (http, ws, etc.)

        Returns:
            True if registration successful
        """
        service = ServiceInfo(name, port, protocol)
        self.services[name] = service

        logger.info(f"✅ Registered service: {name} on {protocol}://localhost:{port}")

        # Perform initial health check
        is_healthy = await self._check_service_health(port)
        service.health_score = 1.0 if is_healthy else 0.0
        service.last_check = datetime.now()

        return True

    def get_service(self, name: str) -> Optional[ServiceInfo]:
        """Get a registered service by name"""
        return self.services.get(name)

    async def _check_service_health(self, port: int) -> bool:
        """Check if a service on the given port is healthy"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://localhost:{port}/health",
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.debug(f"Health check failed for port {port}: {e}")
            return False

    async def _periodic_health_check(self):
        """Periodically check health of all registered services"""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                for name, service in self.services.items():
                    is_healthy = await self._check_service_health(service.port)

                    # Update health score (exponential moving average)
                    current_health = 1.0 if is_healthy else 0.0
                    service.health_score = 0.7 * service.health_score + 0.3 * current_health
                    service.last_check = datetime.now()

                    # Track health history
                    self.health_history[name].append({
                        "timestamp": datetime.now(),
                        "healthy": is_healthy,
                        "score": service.health_score
                    })

                    # Keep only last 100 checks
                    if len(self.health_history[name]) > 100:
                        self.health_history[name] = self.health_history[name][-100:]

                    if service.health_score < 0.5:
                        logger.warning(f"Service {name} health degraded: {service.health_score:.2f}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic health check: {e}")

    def get_frontend_config(self) -> Dict[str, Any]:
        """Get configuration for frontend"""
        return {
            "backend": {
                "url": "http://localhost:8000",
                "wsUrl": "ws://localhost:8000",
                "endpoints": {
                    "health": "/health",
                    "ml_audio_config": "/audio/ml/config",
                    "ml_audio_stream": "/audio/ml/stream",
                    "jarvis_status": "/voice/jarvis/status",
                    "jarvis_activate": "/voice/jarvis/activate",
                    "wake_word_status": "/api/wake-word/status",
                    "vision_websocket": "/vision/ws/vision"
                }
            },
            "services": {
                name: {
                    "url": service.url,
                    "health_score": service.health_score,
                    "last_check": service.last_check.isoformat() if service.last_check else None
                }
                for name, service in self.services.items()
            }
        }

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            "running": self._running,
            "services": {
                name: {
                    "port": service.port,
                    "protocol": service.protocol,
                    "health_score": service.health_score,
                    "last_check": service.last_check.isoformat() if service.last_check else None
                }
                for name, service in self.services.items()
            }
        }


# Global orchestrator instance
_orchestrator: Optional[AutonomousOrchestrator] = None


def get_orchestrator() -> AutonomousOrchestrator:
    """Get or create the global orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AutonomousOrchestrator()
    return _orchestrator
