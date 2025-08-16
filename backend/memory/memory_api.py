"""
Memory Management API for AI-Powered Chatbot
Provides REST endpoints for memory monitoring and control
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime

from .memory_manager import (
    M1MemoryManager,
    ComponentPriority,
    MemoryState,
    MemorySnapshot,
)


class ComponentRegistration(BaseModel):
    """Request model for component registration"""

    name: str
    priority: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    estimated_memory_mb: int


class LoadComponentRequest(BaseModel):
    """Request model for loading a component"""

    name: str


class MemoryStateResponse(BaseModel):
    """Response model for memory state"""

    state: str
    percent_used: float
    available_mb: float
    used_mb: float
    total_mb: float
    timestamp: str


class ComponentStatusResponse(BaseModel):
    """Response model for component status"""

    name: str
    priority: str
    is_loaded: bool
    estimated_mb: float
    actual_mb: Optional[float]
    last_used: Optional[str]
    can_load: bool
    load_reason: Optional[str]


class MemoryReportResponse(BaseModel):
    """Response model for full memory report"""

    current_state: Dict[str, Any]
    components: List[Dict[str, Any]]
    analysis: Dict[str, Any]
    thresholds: Dict[str, float]


class MemoryAPI:
    """API for memory management functionality"""

    def __init__(self, memory_manager: M1MemoryManager):
        self.memory_manager = memory_manager
        self.router = APIRouter()

        # Register routes
        self._register_routes()

        # Memory monitoring will be started on first request or via startup event

    def _register_routes(self):
        """Register API routes"""
        self.router.add_api_route(
            "/status",
            self.get_memory_status,
            methods=["GET"],
            response_model=MemoryStateResponse,
        )
        self.router.add_api_route(
            "/report",
            self.get_memory_report,
            methods=["GET"],
            response_model=MemoryReportResponse,
        )
        self.router.add_api_route(
            "/components",
            self.list_components,
            methods=["GET"],
            response_model=List[ComponentStatusResponse],
        )
        self.router.add_api_route(
            "/components/register", self.register_component, methods=["POST"]
        )
        self.router.add_api_route(
            "/components/{component_name}/load", self.load_component, methods=["POST"]
        )
        self.router.add_api_route(
            "/components/{component_name}/unload",
            self.unload_component,
            methods=["POST"],
        )
        self.router.add_api_route(
            "/components/{component_name}/status",
            self.get_component_status,
            methods=["GET"],
            response_model=ComponentStatusResponse,
        )
        self.router.add_api_route("/optimize", self.optimize_memory, methods=["POST"])
        self.router.add_api_route(
            "/emergency-cleanup", self.emergency_cleanup, methods=["POST"]
        )

    async def _start_monitoring(self):
        """Start memory monitoring on initialization"""
        await self.memory_manager.start_monitoring()

    async def get_memory_status(self) -> MemoryStateResponse:
        """Get current memory status"""
        snapshot = await self.memory_manager.get_memory_snapshot()

        return MemoryStateResponse(
            state=snapshot.state.value,
            percent_used=snapshot.percent * 100,
            available_mb=snapshot.available / 1024 / 1024,
            used_mb=snapshot.used / 1024 / 1024,
            total_mb=snapshot.total / 1024 / 1024,
            timestamp=snapshot.timestamp.isoformat(),
        )

    async def get_memory_report(self) -> MemoryReportResponse:
        """Get detailed memory report"""
        report = await self.memory_manager.get_memory_report()
        return MemoryReportResponse(**report)

    async def list_components(self) -> List[ComponentStatusResponse]:
        """List all registered components with status"""
        components = []

        for name, info in self.memory_manager.components.items():
            can_load, reason = await self.memory_manager.can_load_component(name)

            components.append(
                ComponentStatusResponse(
                    name=name,
                    priority=info.priority.name,
                    is_loaded=info.is_loaded,
                    estimated_mb=info.estimated_memory / 1024 / 1024,
                    actual_mb=(
                        info.actual_memory / 1024 / 1024 if info.actual_memory else None
                    ),
                    last_used=info.last_used.isoformat() if info.last_used else None,
                    can_load=can_load,
                    load_reason=reason,
                )
            )

        return components

    async def register_component(
        self, request: ComponentRegistration
    ) -> Dict[str, str]:
        """Register a new component for memory management"""
        try:
            priority = ComponentPriority[request.priority]
            self.memory_manager.register_component(
                request.name, priority, request.estimated_memory_mb
            )
            return {
                "status": "success",
                "message": f"Component {request.name} registered",
            }
        except KeyError:
            raise HTTPException(400, f"Invalid priority: {request.priority}")
        except Exception as e:
            raise HTTPException(500, str(e))

    async def load_component(self, component_name: str) -> Dict[str, Any]:
        """Load a component (actual loading logic would be in main.py)"""
        if component_name not in self.memory_manager.components:
            raise HTTPException(404, f"Component {component_name} not found")

        can_load, reason = await self.memory_manager.can_load_component(component_name)

        return {
            "component": component_name,
            "can_load": can_load,
            "reason": reason,
            "message": "Component loading must be initiated from main application",
        }

    async def unload_component(self, component_name: str) -> Dict[str, str]:
        """Unload a component"""
        if component_name not in self.memory_manager.components:
            raise HTTPException(404, f"Component {component_name} not found")

        success = await self.memory_manager.unload_component(component_name)

        if success:
            return {
                "status": "success",
                "message": f"Component {component_name} unloaded",
            }
        else:
            return {"status": "failed", "message": f"Failed to unload {component_name}"}

    async def get_component_status(
        self, component_name: str
    ) -> ComponentStatusResponse:
        """Get status of a specific component"""
        if component_name not in self.memory_manager.components:
            raise HTTPException(404, f"Component {component_name} not found")

        info = self.memory_manager.components[component_name]
        can_load, reason = await self.memory_manager.can_load_component(component_name)

        return ComponentStatusResponse(
            name=component_name,
            priority=info.priority.name,
            is_loaded=info.is_loaded,
            estimated_mb=info.estimated_memory / 1024 / 1024,
            actual_mb=info.actual_memory / 1024 / 1024 if info.actual_memory else None,
            last_used=info.last_used.isoformat() if info.last_used else None,
            can_load=can_load,
            load_reason=reason,
        )

    async def optimize_memory(self) -> Dict[str, Any]:
        """Manually trigger memory optimization"""
        snapshot_before = await self.memory_manager.get_memory_snapshot()

        # Force optimization
        await self.memory_manager._optimize_memory()

        # Wait a moment for GC to complete
        await asyncio.sleep(0.5)

        snapshot_after = await self.memory_manager.get_memory_snapshot()

        freed_mb = (snapshot_before.used - snapshot_after.used) / 1024 / 1024

        return {
            "status": "completed",
            "freed_mb": max(0, freed_mb),
            "before_percent": snapshot_before.percent * 100,
            "after_percent": snapshot_after.percent * 100,
        }

    async def emergency_cleanup(self) -> Dict[str, Any]:
        """Trigger emergency cleanup"""
        # Get list of loaded components before cleanup
        loaded_before = [
            name
            for name, info in self.memory_manager.components.items()
            if info.is_loaded
        ]

        # Perform emergency cleanup
        await self.memory_manager._emergency_shutdown()

        # Get list after cleanup
        loaded_after = [
            name
            for name, info in self.memory_manager.components.items()
            if info.is_loaded
        ]

        unloaded = list(set(loaded_before) - set(loaded_after))

        return {
            "status": "completed",
            "unloaded_components": unloaded,
            "remaining_components": loaded_after,
        }


# Webhook for memory alerts (can be used by frontend)
class MemoryAlert(BaseModel):
    """Memory alert notification"""

    timestamp: str
    state: str
    percent_used: float
    message: str
    severity: str  # "info", "warning", "error", "critical"


async def create_memory_alert_callback(websocket_manager=None):
    """Create a callback for memory state changes"""

    async def memory_alert_callback(snapshot: MemorySnapshot):
        alert = MemoryAlert(
            timestamp=snapshot.timestamp.isoformat(),
            state=snapshot.state.value,
            percent_used=snapshot.percent * 100,
            message=f"Memory {snapshot.state.value}: {snapshot.percent * 100:.1f}% used",
            severity=(
                "info"
                if snapshot.state == MemoryState.HEALTHY
                else (
                    "warning"
                    if snapshot.state == MemoryState.WARNING
                    else (
                        "error"
                        if snapshot.state == MemoryState.CRITICAL
                        else "critical"
                    )
                )
            ),
        )

        # Log the alert
        import logging

        logger = logging.getLogger(__name__)
        if alert.severity != "info":
            logger.warning(f"Memory alert: {alert.message}")

        # Send to websocket if available
        if websocket_manager:
            await websocket_manager.broadcast(alert.dict())

    return memory_alert_callback

