#!/usr/bin/env python3
"""
Hybrid Cloud Intelligence API Router

Provides endpoints for hybrid cloud cost tracking, metrics, and management.
"""

import logging
from datetime import datetime

from core.cost_tracker import get_cost_tracker, initialize_cost_tracking
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/hybrid", tags=["Hybrid Cloud"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================


class CostSummaryResponse(BaseModel):
    """Cost summary response model"""

    period: str
    period_start: str
    period_end: str
    total_vms_created: int
    total_runtime_hours: float
    total_estimated_cost: float
    orphaned_vms_count: int
    orphaned_vms_cost: float
    average_vm_lifetime_hours: float
    cost_savings_vs_regular: float
    savings_percentage: float


class RoutingMetricsResponse(BaseModel):
    """Routing metrics response model"""

    period: str
    total_requests: int
    local_requests: int
    gcp_requests: int
    gcp_routing_ratio: float
    average_local_ram_percent: float


class OrphanedVMReport(BaseModel):
    """Orphaned VM report model"""

    total_orphaned_vms: int
    total_orphaned_cost: float
    orphaned_vms: list


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.get("/health")
async def hybrid_health():
    """Health check for hybrid cloud router"""
    return {
        "status": "healthy",
        "service": "hybrid-cloud-intelligence",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/cost", response_model=CostSummaryResponse)
async def get_cost_summary(
    period: str = Query(
        "all", description="Time period: day, week, month, or all", regex="^(day|week|month|all)$"
    )
):
    """
    Get cost summary for specified period.

    Returns:
        - Total VMs created
        - Total runtime hours
        - Total estimated cost
        - Orphaned VMs count and cost
        - Average VM lifetime
        - Cost savings vs regular VMs
    """
    try:
        tracker = get_cost_tracker()
        summary = await tracker.get_cost_summary(period)

        if not summary:
            raise HTTPException(status_code=404, detail="No cost data available")

        return CostSummaryResponse(**summary)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cost summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cost/daily")
async def get_daily_cost():
    """Get cost summary for the last 24 hours"""
    try:
        tracker = get_cost_tracker()
        return await tracker.get_cost_summary("day")
    except Exception as e:
        logger.error(f"Failed to get daily cost: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cost/weekly")
async def get_weekly_cost():
    """Get cost summary for the last 7 days"""
    try:
        tracker = get_cost_tracker()
        return await tracker.get_cost_summary("week")
    except Exception as e:
        logger.error(f"Failed to get weekly cost: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cost/monthly")
async def get_monthly_cost():
    """Get cost summary for the last 30 days"""
    try:
        tracker = get_cost_tracker()
        return await tracker.get_cost_summary("month")
    except Exception as e:
        logger.error(f"Failed to get monthly cost: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/routing", response_model=RoutingMetricsResponse)
async def get_routing_metrics(
    period: str = Query(
        "day", description="Time period: day, week, month, or all", regex="^(day|week|month|all)$"
    )
):
    """
    Get routing performance metrics.

    Returns:
        - Total routing decisions
        - Local vs GCP request counts
        - GCP routing ratio
        - Average local RAM usage
    """
    try:
        tracker = get_cost_tracker()
        metrics = await tracker.get_routing_metrics(period)

        return RoutingMetricsResponse(**metrics)

    except Exception as e:
        logger.error(f"Failed to get routing metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orphaned-vms", response_model=OrphanedVMReport)
async def get_orphaned_vms():
    """
    Get report of all orphaned VMs (VMs that weren't properly cleaned up).

    Returns:
        - Total orphaned VMs count
        - Total cost of orphaned VMs
        - List of orphaned VM details
    """
    try:
        tracker = get_cost_tracker()
        report = await tracker.get_orphaned_vms_report()

        return OrphanedVMReport(**report)

    except Exception as e:
        logger.error(f"Failed to get orphaned VMs report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_hybrid_status():
    """
    Get comprehensive hybrid cloud status including costs and performance.

    Returns combined view of:
        - Current cost summary (all time)
        - Today's routing metrics
        - Orphaned VMs report
    """
    try:
        tracker = get_cost_tracker()

        # Get all data in parallel
        cost_summary = await tracker.get_cost_summary("all")
        daily_cost = await tracker.get_cost_summary("day")
        routing_metrics = await tracker.get_routing_metrics("day")
        orphaned_report = await tracker.get_orphaned_vms_report()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "cost": {"all_time": cost_summary, "today": daily_cost},
            "routing": routing_metrics,
            "orphaned_vms": orphaned_report,
            "health": "healthy",
        }

    except Exception as e:
        logger.error(f"Failed to get hybrid status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/initialize")
async def initialize_tracking():
    """Initialize cost tracking database (idempotent)"""
    try:
        await initialize_cost_tracking()
        return {
            "status": "initialized",
            "message": "Cost tracking database initialized successfully",
        }
    except Exception as e:
        logger.error(f"Failed to initialize cost tracking: {e}")
        raise HTTPException(status_code=500, detail=str(e))
