#!/usr/bin/env python3
"""
Cost Tracking System for JARVIS Hybrid Cloud Intelligence

Tracks GCP Spot VM costs, runtime hours, and cost optimization metrics.
Stores data in learning database for historical analysis and alerts.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# COST CONFIGURATION
# ============================================================================

# GCP e2-highmem-4 Spot VM pricing (us-central1)
SPOT_VM_HOURLY_COST = 0.029  # $0.029/hour for e2-highmem-4 Spot
REGULAR_VM_HOURLY_COST = 0.120  # $0.120/hour for e2-highmem-4 Regular (for comparison)

# Alert thresholds
COST_ALERT_THRESHOLDS = {
    "daily": 1.00,  # Alert if daily cost exceeds $1.00
    "weekly": 5.00,  # Alert if weekly cost exceeds $5.00
    "monthly": 20.00,  # Alert if monthly cost exceeds $20.00
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "max_vm_lifetime_hours": 2.5,  # Alert if VM runs > 2.5 hours (approaching 3hr limit)
    "max_local_ram_percent": 85,  # Alert if local RAM exceeds 85%
    "min_gcp_routing_ratio": 0.1,  # Alert if less than 10% of heavy workloads go to GCP
}


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class VMSession:
    """Represents a single GCP VM session"""

    instance_id: str
    created_at: datetime
    deleted_at: Optional[datetime] = None
    runtime_hours: float = 0.0
    estimated_cost: float = 0.0
    components: List[str] = None
    trigger_reason: str = ""  # e.g., "HIGH_RAM", "PROACTIVE", "MANUAL"
    is_orphaned: bool = False

    def __post_init__(self):
        if self.components is None:
            self.components = []

    def calculate_cost(self):
        """Calculate cost based on runtime"""
        if self.deleted_at:
            runtime = (self.deleted_at - self.created_at).total_seconds() / 3600
        else:
            runtime = (datetime.utcnow() - self.created_at).total_seconds() / 3600

        self.runtime_hours = runtime
        self.estimated_cost = runtime * SPOT_VM_HOURLY_COST
        return self.estimated_cost


@dataclass
class CostMetrics:
    """Cost and performance metrics"""

    total_vms_created: int = 0
    total_runtime_hours: float = 0.0
    total_estimated_cost: float = 0.0
    orphaned_vms_count: int = 0
    orphaned_vms_cost: float = 0.0
    average_vm_lifetime_hours: float = 0.0
    local_requests: int = 0
    gcp_requests: int = 0
    gcp_routing_ratio: float = 0.0
    cost_savings_vs_regular: float = 0.0
    period_start: datetime = None
    period_end: datetime = None

    def __post_init__(self):
        if self.period_start is None:
            self.period_start = datetime.utcnow()
        if self.period_end is None:
            self.period_end = datetime.utcnow()


# ============================================================================
# COST TRACKER CLASS
# ============================================================================


class CostTracker:
    """
    Tracks GCP Spot VM costs and performance metrics.

    Features:
    - Records VM creation/deletion events
    - Calculates runtime hours and estimated costs
    - Tracks orphaned VMs (cleanup failures)
    - Monitors local vs GCP routing ratios
    - Generates cost alerts when thresholds exceeded
    - Stores historical data in learning database
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize cost tracker.

        Args:
            db_path: Path to SQLite database (optional, uses learning_database if None)
        """
        self.db_path = db_path or Path.home() / ".jarvis" / "learning" / "cost_tracking.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory cache for active sessions
        self.active_sessions: Dict[str, VMSession] = {}

        # Cost metrics
        self.metrics = CostMetrics()

        logger.info(f"💰 CostTracker initialized: {self.db_path}")

    async def initialize_database(self):
        """Create database tables if they don't exist"""
        try:
            import aiosqlite

            async with aiosqlite.connect(self.db_path) as db:
                # VM Sessions table
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS vm_sessions (
                        instance_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        deleted_at TEXT,
                        runtime_hours REAL DEFAULT 0.0,
                        estimated_cost REAL DEFAULT 0.0,
                        components TEXT,
                        trigger_reason TEXT,
                        is_orphaned INTEGER DEFAULT 0
                    )
                """
                )

                # Cost metrics table (daily aggregates)
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cost_metrics (
                        date TEXT PRIMARY KEY,
                        total_vms_created INTEGER DEFAULT 0,
                        total_runtime_hours REAL DEFAULT 0.0,
                        total_estimated_cost REAL DEFAULT 0.0,
                        orphaned_vms_count INTEGER DEFAULT 0,
                        orphaned_vms_cost REAL DEFAULT 0.0,
                        average_vm_lifetime_hours REAL DEFAULT 0.0,
                        local_requests INTEGER DEFAULT 0,
                        gcp_requests INTEGER DEFAULT 0,
                        gcp_routing_ratio REAL DEFAULT 0.0,
                        cost_savings_vs_regular REAL DEFAULT 0.0
                    )
                """
                )

                # Routing metrics table (for performance tracking)
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS routing_metrics (
                        timestamp TEXT PRIMARY KEY,
                        local_ram_percent REAL,
                        routing_decision TEXT,
                        components_shifted TEXT,
                        local_request INTEGER DEFAULT 1
                    )
                """
                )

                await db.commit()
                logger.info("✅ Cost tracking database initialized")

        except Exception as e:
            logger.error(f"Failed to initialize cost tracking database: {e}")

    async def record_vm_created(
        self, instance_id: str, components: List[str], trigger_reason: str = "HIGH_RAM"
    ):
        """
        Record VM creation event.

        Args:
            instance_id: GCP instance ID
            components: List of components being shifted to GCP
            trigger_reason: Why VM was created (HIGH_RAM, PROACTIVE, etc.)
        """
        session = VMSession(
            instance_id=instance_id,
            created_at=datetime.utcnow(),
            components=components,
            trigger_reason=trigger_reason,
        )

        self.active_sessions[instance_id] = session
        self.metrics.total_vms_created += 1

        # Store in database
        try:
            import aiosqlite

            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT OR REPLACE INTO vm_sessions
                    (instance_id, created_at, components, trigger_reason)
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        instance_id,
                        session.created_at.isoformat(),
                        json.dumps(components),
                        trigger_reason,
                    ),
                )
                await db.commit()

            logger.info(f"💰 VM created: {instance_id} (trigger: {trigger_reason})")

        except Exception as e:
            logger.error(f"Failed to record VM creation: {e}")

    async def record_vm_deleted(self, instance_id: str, was_orphaned: bool = False):
        """
        Record VM deletion event and calculate final cost.

        Args:
            instance_id: GCP instance ID
            was_orphaned: Whether this VM was orphaned (found by cleanup script)
        """
        session = self.active_sessions.get(instance_id)

        if not session:
            # VM might have been created in previous session
            logger.warning(f"VM {instance_id} not found in active sessions")
            session = VMSession(
                instance_id=instance_id,
                created_at=datetime.utcnow() - timedelta(hours=1),  # Estimate
                is_orphaned=was_orphaned,
            )

        session.deleted_at = datetime.utcnow()
        session.is_orphaned = was_orphaned
        cost = session.calculate_cost()

        # Update metrics
        self.metrics.total_runtime_hours += session.runtime_hours
        self.metrics.total_estimated_cost += cost

        if was_orphaned:
            self.metrics.orphaned_vms_count += 1
            self.metrics.orphaned_vms_cost += cost

        # Calculate average VM lifetime
        if self.metrics.total_vms_created > 0:
            self.metrics.average_vm_lifetime_hours = (
                self.metrics.total_runtime_hours / self.metrics.total_vms_created
            )

        # Store in database
        try:
            import aiosqlite

            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    UPDATE vm_sessions
                    SET deleted_at = ?, runtime_hours = ?, estimated_cost = ?, is_orphaned = ?
                    WHERE instance_id = ?
                """,
                    (
                        session.deleted_at.isoformat(),
                        session.runtime_hours,
                        session.estimated_cost,
                        1 if was_orphaned else 0,
                        instance_id,
                    ),
                )
                await db.commit()

            logger.info(
                f"💰 VM deleted: {instance_id} "
                f"(runtime: {session.runtime_hours:.2f}h, cost: ${cost:.4f}, "
                f"orphaned: {was_orphaned})"
            )

        except Exception as e:
            logger.error(f"Failed to record VM deletion: {e}")

        # Remove from active sessions
        if instance_id in self.active_sessions:
            del self.active_sessions[instance_id]

        # Check for cost alerts
        await self._check_cost_alerts()

    async def record_routing_decision(
        self,
        local_ram_percent: float,
        decision: str,
        components: Optional[List[str]] = None,
        routed_to_gcp: bool = False,
    ):
        """
        Record workload routing decision for performance tracking.

        Args:
            local_ram_percent: Current local RAM usage percentage
            decision: Routing decision (LOCAL, GCP, etc.)
            components: Components involved
            routed_to_gcp: Whether workload was routed to GCP
        """
        if routed_to_gcp:
            self.metrics.gcp_requests += 1
        else:
            self.metrics.local_requests += 1

        # Update routing ratio
        total_requests = self.metrics.local_requests + self.metrics.gcp_requests
        if total_requests > 0:
            self.metrics.gcp_routing_ratio = self.metrics.gcp_requests / total_requests

        # Store in database
        try:
            import aiosqlite

            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT INTO routing_metrics
                    (timestamp, local_ram_percent, routing_decision, components_shifted, local_request)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        datetime.utcnow().isoformat(),
                        local_ram_percent,
                        decision,
                        json.dumps(components or []),
                        0 if routed_to_gcp else 1,
                    ),
                )
                await db.commit()

        except Exception as e:
            logger.error(f"Failed to record routing decision: {e}")

    async def get_cost_summary(self, period: str = "all") -> Dict[str, Any]:
        """
        Get cost summary for specified period.

        Args:
            period: "day", "week", "month", or "all"

        Returns:
            Dictionary with cost metrics
        """
        now = datetime.utcnow()
        period_start = {
            "day": now - timedelta(days=1),
            "week": now - timedelta(weeks=1),
            "month": now - timedelta(days=30),
            "all": datetime(2000, 1, 1),  # Beginning of time
        }.get(period, datetime(2000, 1, 1))

        try:
            import aiosqlite

            async with aiosqlite.connect(self.db_path) as db:
                # Get VM sessions in period
                async with db.execute(
                    """
                    SELECT
                        COUNT(*) as total_vms,
                        SUM(runtime_hours) as total_hours,
                        SUM(estimated_cost) as total_cost,
                        SUM(CASE WHEN is_orphaned = 1 THEN 1 ELSE 0 END) as orphaned_vms,
                        SUM(CASE WHEN is_orphaned = 1 THEN estimated_cost ELSE 0 END) as orphaned_cost,
                        AVG(runtime_hours) as avg_lifetime
                    FROM vm_sessions
                    WHERE created_at >= ?
                """,
                    (period_start.isoformat(),),
                ) as cursor:
                    row = await cursor.fetchone()

                    if row:
                        total_cost = row[2] or 0.0
                        total_hours = row[1] or 0.0

                        # Calculate savings vs regular VMs
                        regular_cost = total_hours * REGULAR_VM_HOURLY_COST
                        savings = regular_cost - total_cost

                        return {
                            "period": period,
                            "period_start": period_start.isoformat(),
                            "period_end": now.isoformat(),
                            "total_vms_created": row[0] or 0,
                            "total_runtime_hours": round(total_hours, 2),
                            "total_estimated_cost": round(total_cost, 4),
                            "orphaned_vms_count": row[3] or 0,
                            "orphaned_vms_cost": round(row[4] or 0.0, 4),
                            "average_vm_lifetime_hours": round(row[5] or 0.0, 2),
                            "cost_savings_vs_regular": round(savings, 4),
                            "savings_percentage": round(
                                (savings / regular_cost * 100) if regular_cost > 0 else 0, 1
                            ),
                        }

            return {}

        except Exception as e:
            logger.error(f"Failed to get cost summary: {e}")
            return {}

    async def get_routing_metrics(self, period: str = "day") -> Dict[str, Any]:
        """
        Get routing performance metrics.

        Args:
            period: "day", "week", "month", or "all"

        Returns:
            Dictionary with routing metrics
        """
        now = datetime.utcnow()
        period_start = {
            "day": now - timedelta(days=1),
            "week": now - timedelta(weeks=1),
            "month": now - timedelta(days=30),
            "all": datetime(2000, 1, 1),
        }.get(period, now - timedelta(days=1))

        try:
            import aiosqlite

            async with aiosqlite.connect(self.db_path) as db:
                # Get routing metrics
                async with db.execute(
                    """
                    SELECT
                        COUNT(*) as total_requests,
                        SUM(local_request) as local_requests,
                        SUM(CASE WHEN local_request = 0 THEN 1 ELSE 0 END) as gcp_requests,
                        AVG(local_ram_percent) as avg_ram
                    FROM routing_metrics
                    WHERE timestamp >= ?
                """,
                    (period_start.isoformat(),),
                ) as cursor:
                    row = await cursor.fetchone()

                    if row and row[0]:
                        total = row[0]
                        local = row[1] or 0
                        gcp = row[2] or 0

                        return {
                            "period": period,
                            "total_requests": total,
                            "local_requests": local,
                            "gcp_requests": gcp,
                            "gcp_routing_ratio": round(gcp / total if total > 0 else 0, 3),
                            "average_local_ram_percent": round(row[3] or 0, 1),
                        }

            return {
                "period": period,
                "total_requests": 0,
                "local_requests": 0,
                "gcp_requests": 0,
                "gcp_routing_ratio": 0.0,
                "average_local_ram_percent": 0.0,
            }

        except Exception as e:
            logger.error(f"Failed to get routing metrics: {e}")
            return {}

    async def _check_cost_alerts(self):
        """Check if cost thresholds are exceeded and log alerts"""
        # Check daily cost
        day_summary = await self.get_cost_summary("day")
        daily_cost = day_summary.get("total_estimated_cost", 0)

        if daily_cost > COST_ALERT_THRESHOLDS["daily"]:
            logger.warning(
                f"💰 COST ALERT: Daily cost ${daily_cost:.2f} exceeds "
                f"threshold ${COST_ALERT_THRESHOLDS['daily']:.2f}"
            )

        # Check weekly cost
        week_summary = await self.get_cost_summary("week")
        weekly_cost = week_summary.get("total_estimated_cost", 0)

        if weekly_cost > COST_ALERT_THRESHOLDS["weekly"]:
            logger.warning(
                f"💰 COST ALERT: Weekly cost ${weekly_cost:.2f} exceeds "
                f"threshold ${COST_ALERT_THRESHOLDS['weekly']:.2f}"
            )

        # Check monthly cost
        month_summary = await self.get_cost_summary("month")
        monthly_cost = month_summary.get("total_estimated_cost", 0)

        if monthly_cost > COST_ALERT_THRESHOLDS["monthly"]:
            logger.warning(
                f"💰 COST ALERT: Monthly cost ${monthly_cost:.2f} exceeds "
                f"threshold ${COST_ALERT_THRESHOLDS['monthly']:.2f}"
            )

        # Check average VM lifetime
        avg_lifetime = self.metrics.average_vm_lifetime_hours
        if avg_lifetime > PERFORMANCE_THRESHOLDS["max_vm_lifetime_hours"]:
            logger.warning(
                f"⚠️  PERFORMANCE ALERT: Average VM lifetime {avg_lifetime:.2f}h "
                f"exceeds threshold {PERFORMANCE_THRESHOLDS['max_vm_lifetime_hours']}h "
                f"(approaching 3hr max)"
            )

    async def get_orphaned_vms_report(self) -> Dict[str, Any]:
        """Get report of all orphaned VMs"""
        try:
            import aiosqlite

            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    """
                    SELECT instance_id, created_at, deleted_at, runtime_hours, estimated_cost
                    FROM vm_sessions
                    WHERE is_orphaned = 1
                    ORDER BY created_at DESC
                    LIMIT 50
                """
                ) as cursor:
                    rows = await cursor.fetchall()

                    orphaned_vms = []
                    for row in rows:
                        orphaned_vms.append(
                            {
                                "instance_id": row[0],
                                "created_at": row[1],
                                "deleted_at": row[2],
                                "runtime_hours": round(row[3], 2),
                                "estimated_cost": round(row[4], 4),
                            }
                        )

                    return {
                        "total_orphaned_vms": len(orphaned_vms),
                        "total_orphaned_cost": sum(vm["estimated_cost"] for vm in orphaned_vms),
                        "orphaned_vms": orphaned_vms,
                    }

        except Exception as e:
            logger.error(f"Failed to get orphaned VMs report: {e}")
            return {"total_orphaned_vms": 0, "total_orphaned_cost": 0, "orphaned_vms": []}


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_cost_tracker_instance: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get or create global cost tracker instance"""
    global _cost_tracker_instance

    if _cost_tracker_instance is None:
        _cost_tracker_instance = CostTracker()

    return _cost_tracker_instance


async def initialize_cost_tracking():
    """Initialize cost tracking system"""
    tracker = get_cost_tracker()
    await tracker.initialize_database()
    logger.info("💰 Cost tracking system initialized")
