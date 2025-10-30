"""
Memory Management API for AI-Powered Chatbot

This module provides REST endpoints for memory monitoring and control in an AI-powered
chatbot system. It includes comprehensive memory management capabilities, intelligent
optimization, and real-time monitoring with alerts.

The API supports:
- Real-time memory status monitoring
- Component registration and lifecycle management
- Intelligent memory optimization for different modes (e.g., LangChain)
- Emergency cleanup procedures
- Memory alerts and notifications

Example:
    >>> from memory_api import MemoryAPI
    >>> from memory_manager import M1MemoryManager
    >>> 
    >>> memory_manager = M1MemoryManager()
    >>> api = MemoryAPI(memory_manager)
    >>> # API routes are now available via api.router
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
from .intelligent_memory_optimizer import (
    IntelligentMemoryOptimizer,
    MemoryOptimizationAPI
)

class ComponentRegistration(BaseModel):
    """Request model for component registration.
    
    Attributes:
        name: Unique identifier for the component
        priority: Priority level ("CRITICAL", "HIGH", "MEDIUM", "LOW")
        estimated_memory_mb: Estimated memory usage in megabytes
    """

    name: str
    priority: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    estimated_memory_mb: int

class LoadComponentRequest(BaseModel):
    """Request model for loading a component.
    
    Attributes:
        name: Name of the component to load
    """

    name: str

class MemoryStateResponse(BaseModel):
    """Response model for memory state information.
    
    Attributes:
        state: Current memory state (HEALTHY, WARNING, CRITICAL, EMERGENCY)
        percent_used: Percentage of memory currently in use
        available_mb: Available memory in megabytes
        used_mb: Used memory in megabytes
        total_mb: Total system memory in megabytes
        timestamp: ISO timestamp of when the snapshot was taken
    """

    state: str
    percent_used: float
    available_mb: float
    used_mb: float
    total_mb: float
    timestamp: str

class ComponentStatusResponse(BaseModel):
    """Response model for component status information.
    
    Attributes:
        name: Component name
        priority: Component priority level
        is_loaded: Whether the component is currently loaded
        estimated_mb: Estimated memory usage in megabytes
        actual_mb: Actual memory usage in megabytes (if loaded)
        last_used: ISO timestamp of last usage
        can_load: Whether the component can be loaded given current memory state
        load_reason: Explanation of why component can/cannot be loaded
    """

    name: str
    priority: str
    is_loaded: bool
    estimated_mb: float
    actual_mb: Optional[float]
    last_used: Optional[str]
    can_load: bool
    load_reason: Optional[str]

class MemoryReportResponse(BaseModel):
    """Response model for comprehensive memory report.
    
    Attributes:
        current_state: Current memory state information
        components: List of all registered components with their status
        analysis: Memory usage analysis and recommendations
        thresholds: Memory threshold configuration
    """

    current_state: Dict[str, Any]
    components: List[Dict[str, Any]]
    analysis: Dict[str, Any]
    thresholds: Dict[str, float]

class OptimizeLangChainRequest(BaseModel):
    """Request model for optimizing memory for LangChain operations.
    
    Attributes:
        force: Force optimization even if memory is already below threshold
    """
    
    force: bool = False  # Force optimization even if already below threshold

class OptimizationReportResponse(BaseModel):
    """Response model for memory optimization report.
    
    Attributes:
        success: Whether optimization was successful
        initial_percent: Memory usage percentage before optimization
        final_percent: Memory usage percentage after optimization
        memory_freed_mb: Amount of memory freed in megabytes
        actions_taken: List of optimization actions performed
        target_percent: Target memory usage percentage
        message: Human-readable optimization result message
    """
    
    success: bool
    initial_percent: float
    final_percent: float
    memory_freed_mb: float
    actions_taken: List[Dict[str, Any]]
    target_percent: float
    message: str

class MemoryAPI:
    """API for memory management functionality.
    
    Provides REST endpoints for monitoring and controlling memory usage in an
    AI-powered chatbot system. Includes intelligent optimization capabilities
    and real-time monitoring.
    
    Attributes:
        memory_manager: The M1MemoryManager instance for core memory operations
        router: FastAPI router containing all API endpoints
        intelligent_optimizer: Intelligent memory optimization engine
        optimization_api: API for memory optimization suggestions
    
    Example:
        >>> memory_manager = M1MemoryManager()
        >>> api = MemoryAPI(memory_manager)
        >>> app.include_router(api.router, prefix="/memory")
    """

    def __init__(self, memory_manager: M1MemoryManager):
        """Initialize the Memory API.
        
        Args:
            memory_manager: The M1MemoryManager instance to use for memory operations
        """
        self.memory_manager = memory_manager
        self.router = APIRouter()
        
        # Initialize intelligent memory optimizer
        self.intelligent_optimizer = IntelligentMemoryOptimizer()
        self.optimization_api = MemoryOptimizationAPI()

        # Register routes
        self._register_routes()

        # Memory monitoring will be started on first request or via startup event

    def _register_routes(self) -> None:
        """Register all API routes with the FastAPI router.
        
        Sets up all endpoints for memory management including status monitoring,
        component management, and optimization operations.
        """
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
        
        # Intelligent optimization routes
        self.router.add_api_route(
            "/optimize/langchain",
            self.optimize_for_langchain,
            methods=["POST"],
            response_model=OptimizationReportResponse
        )
        self.router.add_api_route(
            "/optimize/suggestions",
            self.get_optimization_suggestions,
            methods=["GET"]
        )

    async def _start_monitoring(self) -> None:
        """Start memory monitoring on initialization.
        
        Initializes the memory monitoring system to begin tracking memory usage
        and component states.
        """
        await self.memory_manager.start_monitoring()

    async def get_memory_status(self) -> MemoryStateResponse:
        """Get current memory status.
        
        Returns:
            MemoryStateResponse: Current memory state including usage percentages,
                available memory, and timestamp
        
        Example:
            >>> status = await api.get_memory_status()
            >>> print(f"Memory usage: {status.percent_used}%")
        """
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
        """Get detailed memory report.
        
        Returns:
            MemoryReportResponse: Comprehensive memory report including current state,
                component information, analysis, and threshold configuration
        
        Example:
            >>> report = await api.get_memory_report()
            >>> print(f"Components loaded: {len(report.components)}")
        """
        report = await self.memory_manager.get_memory_report()
        return MemoryReportResponse(**report)

    async def list_components(self) -> List[ComponentStatusResponse]:
        """List all registered components with their current status.
        
        Returns:
            List[ComponentStatusResponse]: List of all components with their
                priority, load status, memory usage, and availability
        
        Example:
            >>> components = await api.list_components()
            >>> loaded = [c for c in components if c.is_loaded]
        """
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
        """Register a new component for memory management.
        
        Args:
            request: Component registration details including name, priority,
                and estimated memory usage
        
        Returns:
            Dict[str, str]: Success/failure status and message
        
        Raises:
            HTTPException: If priority is invalid (400) or registration fails (500)
        
        Example:
            >>> request = ComponentRegistration(
            ...     name="llm_model",
            ...     priority="HIGH",
            ...     estimated_memory_mb=2048
            ... )
            >>> result = await api.register_component(request)
        """
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
        """Check if a component can be loaded and provide guidance.
        
        Note: This endpoint only checks if loading is possible. Actual loading
        must be initiated from the main application.
        
        Args:
            component_name: Name of the component to check for loading
        
        Returns:
            Dict[str, Any]: Loading feasibility information including whether
                the component can be loaded and the reason
        
        Raises:
            HTTPException: If component is not found (404)
        
        Example:
            >>> result = await api.load_component("llm_model")
            >>> if result["can_load"]:
            ...     # Proceed with loading in main application
        """
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
        """Unload a component from memory.
        
        Args:
            component_name: Name of the component to unload
        
        Returns:
            Dict[str, str]: Success/failure status and message
        
        Raises:
            HTTPException: If component is not found (404)
        
        Example:
            >>> result = await api.unload_component("embedding_model")
            >>> print(result["message"])
        """
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
        """Get detailed status of a specific component.
        
        Args:
            component_name: Name of the component to query
        
        Returns:
            ComponentStatusResponse: Detailed component status including
                priority, load state, memory usage, and availability
        
        Raises:
            HTTPException: If component is not found (404)
        
        Example:
            >>> status = await api.get_component_status("llm_model")
            >>> print(f"Component loaded: {status.is_loaded}")
        """
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
        """Manually trigger memory optimization.
        
        Forces garbage collection and component unloading to free up memory.
        
        Returns:
            Dict[str, Any]: Optimization results including memory freed and
                before/after usage percentages
        
        Example:
            >>> result = await api.optimize_memory()
            >>> print(f"Freed {result['freed_mb']:.1f} MB")
        """
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
        """Trigger emergency memory cleanup.
        
        Performs aggressive memory cleanup by unloading non-critical components
        and forcing garbage collection. Used when memory usage reaches critical levels.
        
        Returns:
            Dict[str, Any]: Cleanup results including lists of unloaded and
                remaining components
        
        Example:
            >>> result = await api.emergency_cleanup()
            >>> print(f"Unloaded: {result['unloaded_components']}")
        """
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
    
    async def optimize_for_langchain(self, request: OptimizeLangChainRequest) -> OptimizationReportResponse:
        """Intelligently optimize memory for LangChain operations.
        
        Performs targeted optimization to ensure sufficient memory is available
        for LangChain operations while preserving critical components.
        
        Args:
            request: Optimization request with force flag
        
        Returns:
            OptimizationReportResponse: Detailed optimization report including
                memory freed, actions taken, and success status
        
        Example:
            >>> request = OptimizeLangChainRequest(force=True)
            >>> result = await api.optimize_for_langchain(request)
            >>> print(f"Optimization success: {result.success}")
        """
        # Check current memory
        snapshot = await self.memory_manager.get_memory_snapshot()
        
        # If already below threshold and not forcing, return success
        if snapshot.percent <= 0.5 and not request.force:
            return OptimizationReportResponse(
                success=True,
                initial_percent=snapshot.percent * 100,
                final_percent=snapshot.percent * 100,
                memory_freed_mb=0,
                actions_taken=[],
                target_percent=50,
                message="Memory already optimized for LangChain mode"
            )
        
        # Run intelligent optimization
        success, report = await self.intelligent_optimizer.optimize_for_langchain()
        
        return OptimizationReportResponse(
            success=report["success"],
            initial_percent=report["initial_percent"],
            final_percent=report["final_percent"],
            memory_freed_mb=report["memory_freed_mb"],
            actions_taken=report["actions_taken"],
            target_percent=report["target_percent"],
            message=report.get("message", "Memory optimization completed" if success else "Could not free enough memory")
        )
    
    async def get_optimization_suggestions(self) -> Dict[str, Any]:
        """Get memory optimization suggestions.
        
        Analyzes current memory usage and provides intelligent suggestions
        for optimization without actually performing any changes.
        
        Returns:
            Dict[str, Any]: Optimization suggestions including recommended
                actions and potential memory savings
        
        Example:
            >>> suggestions = await api.get_optimization_suggestions()
            >>> for action in suggestions.get("recommendations", []):
            ...     print(f"Suggestion: {action['description']}")
        """
        result = await self.optimization_api.get_suggestions()
        return result

# Webhook for memory alerts (can be used by frontend)
class MemoryAlert(BaseModel):
    """Memory alert notification model.
    
    Used for real-time memory state change notifications that can be sent
    to frontend clients via WebSocket or other notification mechanisms.
    
    Attributes:
        timestamp: ISO timestamp when the alert was generated
        state: Current memory state (HEALTHY, WARNING, CRITICAL, EMERGENCY)
        percent_used: Current memory usage percentage
        message: Human-readable alert message
        severity: Alert severity level ("info", "warning", "error", "critical")
    """

    timestamp: str
    state: str
    percent_used: float
    message: str
    severity: str  # "info", "warning", "error", "critical"

async def create_memory_alert_callback(websocket_manager=None):
    """Create a callback function for memory state change alerts.
    
    Creates an async callback that can be registered with the memory manager
    to receive notifications when memory state changes occur.
    
    Args:
        websocket_manager: Optional WebSocket manager for broadcasting alerts
            to connected clients
    
    Returns:
        Callable: Async callback function that processes memory snapshots
            and generates alerts
    
    Example:
        >>> callback = await create_memory_alert_callback(ws_manager)
        >>> memory_manager.register_callback(callback)
    """

    async def memory_alert_callback(snapshot: MemorySnapshot):
        """Process memory snapshot and generate alert.
        
        Args:
            snapshot: Memory snapshot containing current state information
        """
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

        # Log the alert - only log critical alerts to avoid spam
        import logging

        logger = logging.getLogger(__name__)
        if alert.severity == "critical":
            logger.warning(f"Memory alert: {alert.message}")

        # Send to websocket if available
        if websocket_manager:
            await websocket_manager.broadcast(alert.dict())

    return memory_alert_callback