"""
Model Management API Routes - Comprehensive Endpoints
====================================================

REST API endpoints for:
- Model status and registry inspection
- Intelligent model selection
- Lifecycle management controls
- Performance metrics and monitoring
- Cost tracking and optimization insights

All endpoints are async and return JSON.
"""

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/models", tags=["models"])


# ============== Request/Response Models ==============


class ModelSelectionRequest(BaseModel):
    """Request for intelligent model selection"""

    query: str = Field(..., description="User query text")
    intent: Optional[str] = Field(None, description="Pre-classified intent (optional)")
    required_capabilities: Optional[List[str]] = Field(
        None, description="Required capabilities (optional)"
    )
    context: Optional[Dict] = Field(None, description="Additional context from UAE/SAI/CAI")
    max_fallbacks: int = Field(3, description="Maximum number of fallback models")


class ModelExecutionRequest(BaseModel):
    """Request to execute query with intelligent model selection"""

    query: str = Field(..., description="Query to execute")
    intent: Optional[str] = None
    required_capabilities: Optional[List[str]] = None
    context: Optional[Dict] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class ModelLoadRequest(BaseModel):
    """Request to manually load a model"""

    model_name: str = Field(..., description="Name of model to load")
    priority: int = Field(50, description="Loading priority")
    timeout_seconds: float = Field(120.0, description="Max wait time")


class ModelUnloadRequest(BaseModel):
    """Request to manually unload a model"""

    model_name: str = Field(..., description="Name of model to unload")
    force: bool = Field(False, description="Force unload even if protected")


# ============== Model Registry Endpoints ==============


@router.get("/registry")
async def get_model_registry():
    """
    Get complete model registry with all registered models

    Returns detailed information about:
    - All available models
    - Capabilities per model
    - Resource requirements
    - Current states
    - Usage statistics
    """
    try:
        from backend.intelligence.model_registry import get_model_registry

        registry = get_model_registry()
        status = registry.get_status()

        return {
            "success": True,
            "total_models": status["total_models"],
            "loaded_models": status["loaded_models"],
            "total_ram_gb": status["total_ram_gb"],
            "models": status["models"],
            "registry_info": {
                "capabilities_indexed": len(registry.capability_index),
                "backend_constraints": registry.backend_constraints,
            },
        }
    except Exception as e:
        logger.error(f"Error getting model registry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/registry/{model_name}")
async def get_model_details(model_name: str):
    """
    Get detailed information about a specific model

    Args:
        model_name: Name of the model (e.g., "llama_70b", "yolov8m", "claude_api")

    Returns:
        Comprehensive model definition including capabilities, resources, performance
    """
    try:
        from backend.intelligence.model_registry import get_model_registry

        registry = get_model_registry()
        model = registry.get_model(model_name)

        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

        return {
            "success": True,
            "model": {
                "name": model.name,
                "display_name": model.display_name,
                "model_type": model.model_type,
                "capabilities": list(model.capabilities),
                "deployment": model.deployment.value,
                "backend_preference": model.backend_preference,
                "resources": {
                    "ram_gb": model.resources.ram_gb,
                    "disk_gb": model.resources.disk_gb,
                    "vram_gb": model.resources.vram_gb,
                    "min_ram_backend_gb": model.resources.min_ram_backend_gb,
                    "cost_per_query_usd": model.resources.cost_per_query_usd,
                    "disk_storage_cost_per_month": model.resources.disk_storage_cost_per_month,
                },
                "performance": {
                    "latency_ms": model.performance.latency_ms,
                    "quality_score": model.performance.quality_score,
                    "throughput_qps": model.performance.throughput_qps,
                    "load_from_cache_seconds": model.performance.load_from_cache_seconds,
                    "load_from_archive_seconds": model.performance.load_from_archive_seconds,
                },
                "state": {
                    "current_state": model.current_state.value,
                    "last_used_timestamp": model.last_used_timestamp,
                    "total_queries": model.total_queries,
                    "total_cost_usd": model.total_cost_usd,
                },
                "config": {
                    "lazy_load": model.lazy_load,
                    "keep_loaded": model.keep_loaded,
                    "priority": model.priority,
                },
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model details for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capabilities")
async def get_capabilities():
    """
    Get all available capabilities and which models support them

    Returns:
        Dictionary mapping capabilities to list of model names
    """
    try:
        from backend.intelligence.model_registry import get_model_registry

        registry = get_model_registry()

        capabilities_map = {}
        for capability, model_names in registry.capability_index.items():
            capabilities_map[capability] = list(model_names)

        return {
            "success": True,
            "total_capabilities": len(capabilities_map),
            "capabilities": capabilities_map,
        }
    except Exception as e:
        logger.error(f"Error getting capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capabilities/{capability}")
async def get_models_for_capability(capability: str):
    """
    Get all models that support a specific capability

    Args:
        capability: Capability name (e.g., "nlp_analysis", "vision", "code_explanation")

    Returns:
        List of models that support this capability
    """
    try:
        from backend.intelligence.model_registry import get_model_registry

        registry = get_model_registry()
        models = registry.get_models_for_capability(capability)

        if not models:
            return {
                "success": True,
                "capability": capability,
                "models": [],
                "message": f"No models found for capability '{capability}'",
            }

        return {
            "success": True,
            "capability": capability,
            "total_models": len(models),
            "models": [
                {
                    "name": m.name,
                    "display_name": m.display_name,
                    "model_type": m.model_type,
                    "state": m.current_state.value,
                    "quality_score": m.performance.quality_score,
                    "latency_ms": m.performance.latency_ms,
                    "cost_per_query": m.resources.cost_per_query_usd,
                }
                for m in models
            ],
        }
    except Exception as e:
        logger.error(f"Error getting models for capability {capability}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Lifecycle Management Endpoints ==============


@router.get("/lifecycle/status")
async def get_lifecycle_status():
    """
    Get comprehensive lifecycle manager status

    Returns:
        - Running status
        - RAM usage (local + GCP)
        - Loaded/cached/archived model counts
        - Statistics (loads, unloads, evictions)
        - Configuration policies
    """
    try:
        from backend.intelligence.model_lifecycle_manager import get_lifecycle_manager

        manager = get_lifecycle_manager()
        status = manager.get_status()

        return {"success": True, **status}
    except Exception as e:
        logger.error(f"Error getting lifecycle status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/lifecycle/load")
async def load_model(request: ModelLoadRequest):
    """
    Manually load a model into RAM

    This will:
    1. Check RAM availability
    2. Evict other models if needed (LRU)
    3. Load model from disk cache or cloud storage
    4. Return when model is ready
    """
    try:
        from backend.intelligence.model_lifecycle_manager import get_lifecycle_manager

        manager = get_lifecycle_manager()

        model_instance = await manager.get_model(
            model_name=request.model_name,
            required_by="api_request",
            priority=request.priority,
            timeout=request.timeout_seconds,
        )

        if model_instance:
            return {
                "success": True,
                "message": f"Model '{request.model_name}' loaded successfully",
                "model_name": request.model_name,
            }
        else:
            raise HTTPException(
                status_code=500, detail=f"Failed to load model '{request.model_name}'"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model {request.model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/lifecycle/unload")
async def unload_model(request: ModelUnloadRequest):
    """
    Manually unload a model from RAM

    This will:
    1. Check if model is loaded
    2. Stop any active inference
    3. Unload model to disk cache
    4. Free RAM
    """
    try:
        from backend.intelligence.model_lifecycle_manager import get_lifecycle_manager

        manager = get_lifecycle_manager()

        await manager.unload_model(model_name=request.model_name, force=request.force)

        return {
            "success": True,
            "message": f"Model '{request.model_name}' unloaded successfully",
            "model_name": request.model_name,
        }
    except Exception as e:
        logger.error(f"Error unloading model {request.model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/lifecycle/ram")
async def get_ram_status(backend: str = Query("gcp", description="Backend name (local or gcp)")):
    """
    Get detailed RAM status for a specific backend

    Args:
        backend: "local" (16GB macOS) or "gcp" (32GB Spot VM)

    Returns:
        RAM usage, available space, loaded models
    """
    try:
        from backend.intelligence.model_lifecycle_manager import get_lifecycle_manager

        manager = get_lifecycle_manager()
        ram_status = await manager._get_ram_status(backend)

        return {
            "success": True,
            "backend": ram_status.backend,
            "total_gb": ram_status.total_gb,
            "used_gb": round(ram_status.used_gb, 2),
            "available_gb": round(ram_status.available_gb, 2),
            "percent_used": round(ram_status.percent_used, 1),
            "models_loaded": ram_status.models_loaded,
            "model_ram_gb": round(ram_status.model_ram_gb, 2),
            "system_ram_gb": round(ram_status.used_gb - ram_status.model_ram_gb, 2),
        }
    except Exception as e:
        logger.error(f"Error getting RAM status for {backend}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Intelligent Model Selection Endpoints ==============


@router.post("/select")
async def select_model(request: ModelSelectionRequest):
    """
    Intelligent model selection based on query and context

    The selector will:
    1. Analyze query intent (CAI integration)
    2. Consider user context and focus level (UAE integration)
    3. Check RAM availability (SAI integration)
    4. Score all capable models
    5. Return best model + fallback chain

    Does NOT execute the query - just returns the selection.
    """
    try:
        from backend.intelligence.model_selector import get_model_selector

        selector = get_model_selector()

        primary_model, fallbacks = await selector.select_with_fallback(
            query=request.query,
            intent=request.intent,
            required_capabilities=(
                set(request.required_capabilities) if request.required_capabilities else None
            ),
            context=request.context,
            max_fallbacks=request.max_fallbacks,
        )

        if not primary_model:
            raise HTTPException(status_code=404, detail="No suitable model found for query")

        return {
            "success": True,
            "query": request.query[:100],
            "primary_model": {
                "name": primary_model.name,
                "display_name": primary_model.display_name,
                "model_type": primary_model.model_type,
                "state": primary_model.current_state.value,
                "estimated_latency_seconds": primary_model.get_load_time_estimate(),
                "cost_per_query": primary_model.resources.cost_per_query_usd,
                "quality_score": primary_model.performance.quality_score,
            },
            "fallbacks": [
                {
                    "name": m.name,
                    "display_name": m.display_name,
                    "state": m.current_state.value,
                }
                for m in fallbacks
            ],
            "total_options": 1 + len(fallbacks),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error selecting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute")
async def execute_with_model_selection(request: ModelExecutionRequest):
    """
    Execute query with intelligent model selection

    This is the full end-to-end flow:
    1. Select best model (with fallback chain)
    2. Load model if needed
    3. Execute query
    4. Return result + metadata

    This is integrated with the hybrid orchestrator.
    """
    try:
        from backend.core.hybrid_orchestrator import HybridOrchestrator

        # Get or create orchestrator instance
        orchestrator = HybridOrchestrator()
        if not orchestrator.is_running:
            await orchestrator.start()

        result = await orchestrator.execute_with_intelligent_model_selection(
            query=request.query,
            intent=request.intent,
            required_capabilities=(
                set(request.required_capabilities) if request.required_capabilities else None
            ),
            context=request.context,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        return result
    except Exception as e:
        logger.error(f"Error executing with model selection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/select/recommend/{intent}")
async def get_model_recommendations(
    intent: str, max_recommendations: int = Query(3, description="Maximum recommendations")
):
    """
    Get model recommendations for a specific intent

    Useful for debugging and understanding which models are best for different tasks.

    Args:
        intent: Intent type (e.g., "vision_analysis", "code_explanation")
        max_recommendations: Max number of recommendations to return

    Returns:
        Scored list of recommended models with reasoning
    """
    try:
        from backend.intelligence.model_selector import get_model_selector

        selector = get_model_selector()

        recommendations = await selector.get_model_recommendations(
            intent=intent, max_recommendations=max_recommendations
        )

        return {
            "success": True,
            "intent": intent,
            "total_recommendations": len(recommendations),
            "recommendations": recommendations,
        }
    except Exception as e:
        logger.error(f"Error getting recommendations for {intent}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Monitoring & Metrics Endpoints ==============


@router.get("/metrics")
async def get_model_metrics():
    """
    Get comprehensive model performance metrics

    Returns:
        - Per-model query counts
        - Per-model costs
        - Cache hit rates
        - Load/unload statistics
        - RAM pressure events
    """
    try:
        from backend.intelligence.model_lifecycle_manager import get_lifecycle_manager
        from backend.intelligence.model_registry import get_model_registry

        registry = get_model_registry()
        manager = get_lifecycle_manager()

        # Get per-model stats
        model_metrics = {}
        for name, model in registry.models.items():
            model_metrics[name] = {
                "total_queries": model.total_queries,
                "total_cost_usd": round(model.total_cost_usd, 2),
                "current_state": model.current_state.value,
                "last_used_timestamp": model.last_used_timestamp,
                "avg_cost_per_query": (
                    round(model.total_cost_usd / model.total_queries, 4)
                    if model.total_queries > 0
                    else 0
                ),
            }

        # Get lifecycle stats
        lifecycle_stats = manager.stats

        return {
            "success": True,
            "model_metrics": model_metrics,
            "lifecycle_stats": lifecycle_stats,
            "cache_hit_rate": (
                round(
                    lifecycle_stats["cache_hits"]
                    / (lifecycle_stats["cache_hits"] + lifecycle_stats["cache_misses"])
                    * 100,
                    2,
                )
                if (lifecycle_stats["cache_hits"] + lifecycle_stats["cache_misses"]) > 0
                else 0
            ),
        }
    except Exception as e:
        logger.error(f"Error getting model metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/cost")
async def get_cost_metrics():
    """
    Get cost tracking and optimization insights

    Returns:
        - Total costs per model
        - Storage costs
        - Potential savings from optimizations
        - Cost breakdown by model type
    """
    try:
        from backend.intelligence.model_registry import get_model_registry

        registry = get_model_registry()

        # Calculate costs
        total_api_cost = 0.0
        total_storage_cost = 0.0
        cost_by_type = {}

        for name, model in registry.models.items():
            # API costs
            total_api_cost += model.total_cost_usd

            # Storage costs (monthly)
            total_storage_cost += model.resources.disk_storage_cost_per_month

            # By type
            if model.model_type not in cost_by_type:
                cost_by_type[model.model_type] = {
                    "api_cost": 0.0,
                    "storage_cost": 0.0,
                    "query_count": 0,
                }
            cost_by_type[model.model_type]["api_cost"] += model.total_cost_usd
            cost_by_type[model.model_type][
                "storage_cost"
            ] += model.resources.disk_storage_cost_per_month
            cost_by_type[model.model_type]["query_count"] += model.total_queries

        return {
            "success": True,
            "total_api_cost_usd": round(total_api_cost, 2),
            "total_storage_cost_monthly_usd": round(total_storage_cost, 2),
            "cost_by_model_type": {
                k: {
                    "api_cost_usd": round(v["api_cost"], 2),
                    "storage_cost_monthly_usd": round(v["storage_cost"], 2),
                    "query_count": v["query_count"],
                }
                for k, v in cost_by_type.items()
            },
        }
    except Exception as e:
        logger.error(f"Error getting cost metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def model_system_health():
    """
    Health check for model management system

    Returns:
        - System health status
        - RAM status
        - Model states
        - Any warnings or errors
    """
    try:
        from backend.intelligence.model_lifecycle_manager import get_lifecycle_manager
        from backend.intelligence.model_registry import get_model_registry

        registry = get_model_registry()
        manager = get_lifecycle_manager()

        # Check for issues
        warnings = []
        errors = []

        # Check RAM pressure
        status = manager.get_status()
        if status["ram"]["gcp"]["percent_used"] > 90:
            warnings.append("GCP RAM usage >90%")
        if status["ram"]["local"]["percent_used"] > 90:
            warnings.append("Local RAM usage >90%")

        # Check for error states
        error_models = [
            name for name, model in registry.models.items() if model.current_state.value == "error"
        ]
        if error_models:
            errors.append(f"Models in error state: {', '.join(error_models)}")

        # Overall health
        health_status = "healthy"
        if errors:
            health_status = "unhealthy"
        elif warnings:
            health_status = "degraded"

        return {
            "success": True,
            "status": health_status,
            "lifecycle_running": status["running"],
            "total_models": registry.get_status()["total_models"],
            "loaded_models": registry.get_status()["loaded_models"],
            "warnings": warnings,
            "errors": errors,
            "ram_status": status["ram"],
        }
    except Exception as e:
        logger.error(f"Error checking model system health: {e}")
        return {
            "success": False,
            "status": "unhealthy",
            "error": str(e),
        }
