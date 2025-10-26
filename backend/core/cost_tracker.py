#!/usr/bin/env python3
"""
Advanced Cost Tracking System for JARVIS Hybrid Cloud Intelligence

Fully async, dynamic, configuration-driven cost tracking with no hardcoding.
Tracks GCP VM costs, runtime hours, and cost optimization metrics.
Stores data in learning database for historical analysis and alerts.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND TYPES
# ============================================================================


class AlertLevel(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class TriggerReason(Enum):
    """VM creation trigger reasons"""

    HIGH_RAM = "HIGH_RAM"
    PROACTIVE = "PROACTIVE"
    MANUAL = "MANUAL"
    EMERGENCY = "EMERGENCY"
    SCHEDULED = "SCHEDULED"


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================


@dataclass
class CostTrackerConfig:
    """Dynamic configuration for cost tracking - no hardcoding"""

    # Database configuration
    db_path: Optional[Path] = None
    db_connection_pool_size: int = 5
    db_timeout: int = 30

    # GCP configuration (loaded from env)
    gcp_project_id: str = field(default_factory=lambda: os.getenv("GCP_PROJECT_ID", ""))
    gcp_region: str = field(default_factory=lambda: os.getenv("GCP_REGION", "us-central1"))
    gcp_zone: str = field(default_factory=lambda: os.getenv("GCP_ZONE", "us-central1-a"))

    # VM pricing (dynamically loadable from GCP API or config)
    spot_vm_hourly_cost: float = field(
        default_factory=lambda: float(os.getenv("SPOT_VM_HOURLY_COST", "0.029"))
    )
    regular_vm_hourly_cost: float = field(
        default_factory=lambda: float(os.getenv("REGULAR_VM_HOURLY_COST", "0.120"))
    )
    vm_instance_type: str = field(default_factory=lambda: os.getenv("GCP_VM_TYPE", "e2-highmem-4"))

    # Alert thresholds (configurable via env)
    alert_threshold_daily: float = field(
        default_factory=lambda: float(os.getenv("COST_ALERT_DAILY", "1.00"))
    )
    alert_threshold_weekly: float = field(
        default_factory=lambda: float(os.getenv("COST_ALERT_WEEKLY", "5.00"))
    )
    alert_threshold_monthly: float = field(
        default_factory=lambda: float(os.getenv("COST_ALERT_MONTHLY", "20.00"))
    )

    # Performance thresholds
    max_vm_lifetime_hours: float = field(
        default_factory=lambda: float(os.getenv("MAX_VM_LIFETIME_HOURS", "2.5"))
    )
    max_local_ram_percent: float = field(
        default_factory=lambda: float(os.getenv("MAX_LOCAL_RAM_PERCENT", "85"))
    )
    min_gcp_routing_ratio: float = field(
        default_factory=lambda: float(os.getenv("MIN_GCP_ROUTING_RATIO", "0.1"))
    )

    # Cleanup configuration
    orphaned_vm_max_age_hours: int = field(
        default_factory=lambda: int(os.getenv("ORPHANED_VM_MAX_AGE_HOURS", "6"))
    )
    cleanup_check_interval_hours: int = field(
        default_factory=lambda: int(os.getenv("CLEANUP_CHECK_INTERVAL_HOURS", "6"))
    )

    # Alert notification configuration
    alert_email: Optional[str] = field(default_factory=lambda: os.getenv("JARVIS_ALERT_EMAIL"))
    enable_desktop_notifications: bool = field(
        default_factory=lambda: os.getenv("ENABLE_DESKTOP_NOTIFICATIONS", "true").lower() == "true"
    )
    enable_email_alerts: bool = field(
        default_factory=lambda: os.getenv("ENABLE_EMAIL_ALERTS", "false").lower() == "true"
    )

    # Advanced features
    enable_auto_cleanup: bool = field(
        default_factory=lambda: os.getenv("ENABLE_AUTO_CLEANUP", "true").lower() == "true"
    )
    enable_cost_forecasting: bool = field(
        default_factory=lambda: os.getenv("ENABLE_COST_FORECASTING", "true").lower() == "true"
    )

    def __post_init__(self):
        """Initialize paths and validate configuration"""
        if self.db_path is None:
            self.db_path = Path.home() / ".jarvis" / "learning" / "cost_tracking.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class VMSession:
    """Represents a single GCP VM session with enhanced tracking"""

    instance_id: str
    created_at: datetime
    deleted_at: Optional[datetime] = None
    runtime_hours: float = 0.0
    estimated_cost: float = 0.0
    actual_cost: Optional[float] = None  # From GCP billing API if available
    components: List[str] = field(default_factory=list)
    trigger_reason: str = TriggerReason.HIGH_RAM.value
    is_orphaned: bool = False
    vm_type: str = "e2-highmem-4"
    region: str = "us-central1"
    zone: str = "us-central1-a"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_cost(self, hourly_rate: float) -> float:
        """Calculate cost based on runtime and hourly rate"""
        if self.deleted_at:
            runtime = (self.deleted_at - self.created_at).total_seconds() / 3600
        else:
            runtime = (datetime.utcnow() - self.created_at).total_seconds() / 3600

        self.runtime_hours = runtime
        self.estimated_cost = runtime * hourly_rate
        return self.estimated_cost


@dataclass
class CostMetrics:
    """Enhanced cost and performance metrics"""

    total_vms_created: int = 0
    total_runtime_hours: float = 0.0
    total_estimated_cost: float = 0.0
    total_actual_cost: float = 0.0  # From GCP billing API
    orphaned_vms_count: int = 0
    orphaned_vms_cost: float = 0.0
    average_vm_lifetime_hours: float = 0.0
    median_vm_lifetime_hours: float = 0.0
    local_requests: int = 0
    gcp_requests: int = 0
    gcp_routing_ratio: float = 0.0
    cost_savings_vs_regular: float = 0.0
    cost_efficiency_score: float = 0.0  # 0-100 score
    period_start: datetime = field(default_factory=datetime.utcnow)
    period_end: datetime = field(default_factory=datetime.utcnow)


# ============================================================================
# ADVANCED COST TRACKER CLASS
# ============================================================================


class CostTracker:
    """
    Advanced async cost tracking system with dynamic configuration.

    Features:
    - Fully async database operations
    - Dynamic configuration via environment variables
    - No hardcoded values
    - Alert callbacks and notifications
    - Auto-cleanup of orphaned VMs
    - Cost forecasting
    - GCP billing API integration (optional)
    - WebSocket event streaming
    """

    def __init__(self, config: Optional[CostTrackerConfig] = None):
        """
        Initialize advanced cost tracker.

        Args:
            config: Optional configuration object
        """
        self.config = config or CostTrackerConfig()
        self.active_sessions: Dict[str, VMSession] = {}
        self.metrics = CostMetrics()
        self._alert_callbacks: List[Callable] = []
        self._cleanup_task: Optional[asyncio.Task] = None
        self._db_lock = asyncio.Lock()

        logger.info(f"💰 Advanced CostTracker initialized")
        logger.info(f"   DB: {self.config.db_path}")
        logger.info(f"   VM Type: {self.config.vm_instance_type}")
        logger.info(f"   Region: {self.config.gcp_region}")
        logger.info(f"   Spot Rate: ${self.config.spot_vm_hourly_cost:.4f}/hr")

    async def initialize(self):
        """Initialize cost tracking system"""
        await self.initialize_database()

        if self.config.enable_auto_cleanup:
            self._cleanup_task = asyncio.create_task(self._auto_cleanup_loop())
            logger.info(
                f"✅ Auto-cleanup enabled (every {self.config.cleanup_check_interval_hours}h)"
            )

    async def shutdown(self):
        """Gracefully shutdown cost tracker"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("💰 CostTracker shutdown complete")

    async def initialize_database(self):
        """Create database tables with enhanced schema"""
        try:
            import aiosqlite

            async with self._db_lock:
                async with aiosqlite.connect(
                    self.config.db_path, timeout=self.config.db_timeout
                ) as db:
                    # Enhanced VM Sessions table
                    await db.execute(
                        """
                        CREATE TABLE IF NOT EXISTS vm_sessions (
                            instance_id TEXT PRIMARY KEY,
                            created_at TEXT NOT NULL,
                            deleted_at TEXT,
                            runtime_hours REAL DEFAULT 0.0,
                            estimated_cost REAL DEFAULT 0.0,
                            actual_cost REAL,
                            components TEXT,
                            trigger_reason TEXT,
                            is_orphaned INTEGER DEFAULT 0,
                            vm_type TEXT,
                            region TEXT,
                            zone TEXT,
                            metadata TEXT
                        )
                    """
                    )

                    # Cost configuration history table
                    await db.execute(
                        """
                        CREATE TABLE IF NOT EXISTS cost_config_history (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT NOT NULL,
                            config_key TEXT NOT NULL,
                            old_value TEXT,
                            new_value TEXT,
                            reason TEXT
                        )
                    """
                    )

                    # Enhanced routing metrics table
                    await db.execute(
                        """
                        CREATE TABLE IF NOT EXISTS routing_metrics (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT NOT NULL,
                            local_ram_percent REAL,
                            routing_decision TEXT,
                            components_shifted TEXT,
                            local_request INTEGER DEFAULT 1,
                            decision_latency_ms REAL,
                            success INTEGER DEFAULT 1
                        )
                    """
                    )

                    # Alerts history table
                    await db.execute(
                        """
                        CREATE TABLE IF NOT EXISTS alert_history (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT NOT NULL,
                            alert_level TEXT NOT NULL,
                            alert_type TEXT NOT NULL,
                            message TEXT NOT NULL,
                            details TEXT,
                            acknowledged INTEGER DEFAULT 0
                        )
                    """
                    )

                    # Cost forecasts table
                    await db.execute(
                        """
                        CREATE TABLE IF NOT EXISTS cost_forecasts (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            generated_at TEXT NOT NULL,
                            forecast_period TEXT NOT NULL,
                            forecast_start TEXT NOT NULL,
                            forecast_end TEXT NOT NULL,
                            predicted_cost REAL NOT NULL,
                            confidence_score REAL,
                            actual_cost REAL,
                            accuracy_score REAL
                        )
                    """
                    )

                    await db.commit()

                    # Run migrations for existing databases
                    await self._migrate_database(db)

                    logger.info("✅ Advanced cost tracking database initialized")

        except Exception as e:
            logger.error(f"Failed to initialize cost tracking database: {e}")
            raise

    async def _migrate_database(self, db):
        """Migrate existing database schema to add missing columns"""
        try:
            # Check if actual_cost column exists in vm_sessions
            cursor = await db.execute("PRAGMA table_info(vm_sessions)")
            columns = await cursor.fetchall()
            column_names = [col[1] for col in columns]

            if "actual_cost" not in column_names:
                logger.info("🔧 Migrating database: adding actual_cost column")
                await db.execute("ALTER TABLE vm_sessions ADD COLUMN actual_cost REAL")
                await db.commit()
                logger.info("✅ Migration complete: actual_cost column added")
        except Exception as e:
            logger.warning(f"Database migration check failed (non-critical): {e}")

    async def record_vm_created(
        self,
        instance_id: str,
        components: List[str],
        trigger_reason: str = TriggerReason.HIGH_RAM.value,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record VM creation with enhanced tracking"""
        session = VMSession(
            instance_id=instance_id,
            created_at=datetime.utcnow(),
            components=components,
            trigger_reason=trigger_reason,
            vm_type=self.config.vm_instance_type,
            region=self.config.gcp_region,
            zone=self.config.gcp_zone,
            metadata=metadata or {},
        )

        self.active_sessions[instance_id] = session
        self.metrics.total_vms_created += 1

        try:
            import aiosqlite

            async with self._db_lock:
                async with aiosqlite.connect(self.config.db_path) as db:
                    await db.execute(
                        """
                        INSERT OR REPLACE INTO vm_sessions
                        (instance_id, created_at, components, trigger_reason, vm_type, region, zone, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            instance_id,
                            session.created_at.isoformat(),
                            json.dumps(components),
                            trigger_reason,
                            session.vm_type,
                            session.region,
                            session.zone,
                            json.dumps(metadata or {}),
                        ),
                    )
                    await db.commit()

            logger.info(
                f"💰 VM created: {instance_id} (trigger: {trigger_reason}, type: {session.vm_type})"
            )

            # Trigger alert callbacks
            await self._notify_event("vm_created", {"instance_id": instance_id, "session": session})

        except Exception as e:
            logger.error(f"Failed to record VM creation: {e}")

    async def record_vm_deleted(
        self, instance_id: str, was_orphaned: bool = False, actual_cost: Optional[float] = None
    ):
        """Record VM deletion with cost calculation"""
        session = self.active_sessions.get(instance_id)

        if not session:
            logger.warning(f"VM {instance_id} not found in active sessions - querying database")
            session = await self._load_session_from_db(instance_id)

        if not session:
            logger.error(f"VM {instance_id} not found anywhere - creating placeholder")
            session = VMSession(
                instance_id=instance_id,
                created_at=datetime.utcnow() - timedelta(hours=1),
                is_orphaned=was_orphaned,
                vm_type=self.config.vm_instance_type,
                region=self.config.gcp_region,
                zone=self.config.gcp_zone,
            )

        session.deleted_at = datetime.utcnow()
        session.is_orphaned = was_orphaned
        session.actual_cost = actual_cost
        cost = session.calculate_cost(self.config.spot_vm_hourly_cost)

        # Update metrics
        self.metrics.total_runtime_hours += session.runtime_hours
        self.metrics.total_estimated_cost += cost
        if actual_cost:
            self.metrics.total_actual_cost += actual_cost

        if was_orphaned:
            self.metrics.orphaned_vms_count += 1
            self.metrics.orphaned_vms_cost += cost

        # Calculate average VM lifetime
        if self.metrics.total_vms_created > 0:
            self.metrics.average_vm_lifetime_hours = (
                self.metrics.total_runtime_hours / self.metrics.total_vms_created
            )

        try:
            import aiosqlite

            async with self._db_lock:
                async with aiosqlite.connect(self.config.db_path) as db:
                    await db.execute(
                        """
                        UPDATE vm_sessions
                        SET deleted_at = ?, runtime_hours = ?, estimated_cost = ?,
                            actual_cost = ?, is_orphaned = ?
                        WHERE instance_id = ?
                    """,
                        (
                            session.deleted_at.isoformat(),
                            session.runtime_hours,
                            session.estimated_cost,
                            actual_cost,
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

            # Remove from active sessions
            if instance_id in self.active_sessions:
                del self.active_sessions[instance_id]

            # Check for cost alerts
            await self._check_cost_alerts()

            # Trigger alert callbacks
            await self._notify_event("vm_deleted", {"instance_id": instance_id, "session": session})

        except Exception as e:
            logger.error(f"Failed to record VM deletion: {e}")

    async def _load_session_from_db(self, instance_id: str) -> Optional[VMSession]:
        """Load VM session from database"""
        try:
            import aiosqlite

            async with self._db_lock:
                async with aiosqlite.connect(self.config.db_path) as db:
                    async with db.execute(
                        """
                        SELECT created_at, components, trigger_reason, vm_type, region, zone, metadata
                        FROM vm_sessions
                        WHERE instance_id = ?
                    """,
                        (instance_id,),
                    ) as cursor:
                        row = await cursor.fetchone()
                        if row:
                            return VMSession(
                                instance_id=instance_id,
                                created_at=datetime.fromisoformat(row[0]),
                                components=json.loads(row[1]) if row[1] else [],
                                trigger_reason=row[2] or TriggerReason.HIGH_RAM.value,
                                vm_type=row[3] or self.config.vm_instance_type,
                                region=row[4] or self.config.gcp_region,
                                zone=row[5] or self.config.gcp_zone,
                                metadata=json.loads(row[6]) if row[6] else {},
                            )
        except Exception as e:
            logger.error(f"Failed to load session from DB: {e}")

        return None

    async def cleanup_orphaned_vms(self) -> Dict[str, Any]:
        """
        Automatically cleanup orphaned VMs from GCP.
        Integrated version of cleanup script.
        """
        results = {
            "checked_at": datetime.utcnow().isoformat(),
            "orphaned_vms_found": 0,
            "orphaned_vms_deleted": 0,
            "errors": [],
            "vms": [],
        }

        try:
            # List all jarvis VMs from GCP
            cmd = [
                "gcloud",
                "compute",
                "instances",
                "list",
                f"--project={self.config.gcp_project_id}",
                "--filter=name~'jarvis-auto-.*'",
                "--format=json",
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = f"gcloud list failed: {stderr.decode()}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
                return results

            vms = json.loads(stdout.decode()) if stdout else []

            for vm in vms:
                instance_id = vm.get("name")
                creation_time = vm.get("creationTimestamp")

                if not instance_id or not creation_time:
                    continue

                # Parse creation time
                created_at = datetime.fromisoformat(creation_time.replace("Z", "+00:00"))
                age_hours = (datetime.now(created_at.tzinfo) - created_at).total_seconds() / 3600

                results["vms"].append(
                    {
                        "instance_id": instance_id,
                        "age_hours": round(age_hours, 2),
                        "status": vm.get("status"),
                    }
                )

                if age_hours >= self.config.orphaned_vm_max_age_hours:
                    results["orphaned_vms_found"] += 1

                    logger.warning(f"🗑️  Orphaned VM found: {instance_id} (age: {age_hours:.1f}h)")

                    # Delete the VM
                    delete_cmd = [
                        "gcloud",
                        "compute",
                        "instances",
                        "delete",
                        instance_id,
                        f"--project={self.config.gcp_project_id}",
                        f"--zone={self.config.gcp_zone}",
                        "--quiet",
                    ]

                    delete_process = await asyncio.create_subprocess_exec(
                        *delete_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )

                    await delete_process.communicate()

                    if delete_process.returncode == 0:
                        results["orphaned_vms_deleted"] += 1

                        # Record deletion in database
                        await self.record_vm_deleted(instance_id, was_orphaned=True)

                        logger.info(f"✅ Deleted orphaned VM: {instance_id}")

                        # Send notification
                        if self.config.enable_desktop_notifications:
                            await self._send_desktop_notification(
                                f"Deleted orphaned VM: {instance_id}"
                            )
                    else:
                        error_msg = f"Failed to delete {instance_id}"
                        results["errors"].append(error_msg)
                        logger.error(error_msg)

            # Log summary
            logger.info(
                f"🧹 Cleanup complete: {results['orphaned_vms_deleted']}/{results['orphaned_vms_found']} orphaned VMs deleted"
            )

            return results

        except Exception as e:
            error_msg = f"Cleanup failed: {e}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            return results

    async def _auto_cleanup_loop(self):
        """Background task for automatic cleanup"""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_check_interval_hours * 3600)
                logger.info("🔄 Running scheduled orphaned VM cleanup...")
                results = await self.cleanup_orphaned_vms()

                if results["orphaned_vms_deleted"] > 0:
                    await self._log_alert(
                        AlertLevel.WARNING,
                        "orphaned_vms_cleanup",
                        f"Cleaned up {results['orphaned_vms_deleted']} orphaned VMs",
                        results,
                    )

            except asyncio.CancelledError:
                logger.info("Auto-cleanup loop cancelled")
                break
            except Exception as e:
                logger.error(f"Auto-cleanup loop error: {e}")

    async def record_routing_decision(
        self,
        local_ram_percent: float,
        decision: str,
        components: Optional[List[str]] = None,
        routed_to_gcp: bool = False,
        decision_latency_ms: Optional[float] = None,
        success: bool = True,
    ):
        """Record routing decision with enhanced metrics"""
        if routed_to_gcp:
            self.metrics.gcp_requests += 1
        else:
            self.metrics.local_requests += 1

        total_requests = self.metrics.local_requests + self.metrics.gcp_requests
        if total_requests > 0:
            self.metrics.gcp_routing_ratio = self.metrics.gcp_requests / total_requests

        try:
            import aiosqlite

            async with self._db_lock:
                async with aiosqlite.connect(self.config.db_path) as db:
                    await db.execute(
                        """
                        INSERT INTO routing_metrics
                        (timestamp, local_ram_percent, routing_decision, components_shifted,
                         local_request, decision_latency_ms, success)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            datetime.utcnow().isoformat(),
                            local_ram_percent,
                            decision,
                            json.dumps(components or []),
                            0 if routed_to_gcp else 1,
                            decision_latency_ms,
                            1 if success else 0,
                        ),
                    )
                    await db.commit()

        except Exception as e:
            logger.error(f"Failed to record routing decision: {e}")

    async def get_cost_summary(self, period: str = "all") -> Dict[str, Any]:
        """Get cost summary with enhanced metrics"""
        now = datetime.utcnow()
        period_start = {
            "day": now - timedelta(days=1),
            "week": now - timedelta(weeks=1),
            "month": now - timedelta(days=30),
            "all": datetime(2000, 1, 1),
        }.get(period, datetime(2000, 1, 1))

        try:
            import aiosqlite

            async with self._db_lock:
                async with aiosqlite.connect(self.config.db_path) as db:
                    async with db.execute(
                        """
                        SELECT
                            COUNT(*) as total_vms,
                            SUM(runtime_hours) as total_hours,
                            SUM(estimated_cost) as total_cost,
                            SUM(actual_cost) as total_actual,
                            SUM(CASE WHEN is_orphaned = 1 THEN 1 ELSE 0 END) as orphaned_vms,
                            SUM(CASE WHEN is_orphaned = 1 THEN estimated_cost ELSE 0 END) as orphaned_cost,
                            AVG(runtime_hours) as avg_lifetime,
                            MAX(runtime_hours) as max_lifetime,
                            MIN(runtime_hours) as min_lifetime
                        FROM vm_sessions
                        WHERE created_at >= ?
                    """,
                        (period_start.isoformat(),),
                    ) as cursor:
                        row = await cursor.fetchone()

                        if row:
                            total_cost = row[2] or 0.0
                            total_hours = row[1] or 0.0
                            total_actual = row[3] or 0.0

                            regular_cost = total_hours * self.config.regular_vm_hourly_cost
                            savings = regular_cost - total_cost

                            # Calculate cost efficiency score
                            if total_actual > 0:
                                efficiency = ((total_actual - total_cost) / total_actual) * 100
                            else:
                                efficiency = 0.0

                            return {
                                "period": period,
                                "period_start": period_start.isoformat(),
                                "period_end": now.isoformat(),
                                "total_vms_created": row[0] or 0,
                                "total_runtime_hours": round(total_hours, 2),
                                "total_estimated_cost": round(total_cost, 4),
                                "total_actual_cost": round(total_actual, 4),
                                "orphaned_vms_count": row[4] or 0,
                                "orphaned_vms_cost": round(row[5] or 0.0, 4),
                                "average_vm_lifetime_hours": round(row[6] or 0.0, 2),
                                "max_vm_lifetime_hours": round(row[7] or 0.0, 2),
                                "min_vm_lifetime_hours": round(row[8] or 0.0, 2),
                                "cost_savings_vs_regular": round(savings, 4),
                                "savings_percentage": round(
                                    (savings / regular_cost * 100) if regular_cost > 0 else 0, 1
                                ),
                                "cost_efficiency_score": round(efficiency, 2),
                                "vm_type": self.config.vm_instance_type,
                                "spot_rate": self.config.spot_vm_hourly_cost,
                                "regular_rate": self.config.regular_vm_hourly_cost,
                            }

            return {}

        except Exception as e:
            logger.error(f"Failed to get cost summary: {e}")
            return {}

    async def get_routing_metrics(self, period: str = "day") -> Dict[str, Any]:
        """Get routing metrics with enhanced analytics"""
        now = datetime.utcnow()
        period_start = {
            "day": now - timedelta(days=1),
            "week": now - timedelta(weeks=1),
            "month": now - timedelta(days=30),
            "all": datetime(2000, 1, 1),
        }.get(period, now - timedelta(days=1))

        try:
            import aiosqlite

            async with self._db_lock:
                async with aiosqlite.connect(self.config.db_path) as db:
                    async with db.execute(
                        """
                        SELECT
                            COUNT(*) as total_requests,
                            SUM(local_request) as local_requests,
                            SUM(CASE WHEN local_request = 0 THEN 1 ELSE 0 END) as gcp_requests,
                            AVG(local_ram_percent) as avg_ram,
                            MAX(local_ram_percent) as max_ram,
                            AVG(decision_latency_ms) as avg_latency,
                            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_decisions
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
                            successful = row[6] or 0

                            return {
                                "period": period,
                                "total_requests": total,
                                "local_requests": local,
                                "gcp_requests": gcp,
                                "successful_decisions": successful,
                                "success_rate": round(
                                    (successful / total * 100) if total > 0 else 0, 2
                                ),
                                "gcp_routing_ratio": round(gcp / total if total > 0 else 0, 3),
                                "average_local_ram_percent": round(row[3] or 0, 1),
                                "max_local_ram_percent": round(row[4] or 0, 1),
                                "average_decision_latency_ms": round(row[5] or 0, 2),
                            }

            return {
                "period": period,
                "total_requests": 0,
                "local_requests": 0,
                "gcp_requests": 0,
                "successful_decisions": 0,
                "success_rate": 0.0,
                "gcp_routing_ratio": 0.0,
                "average_local_ram_percent": 0.0,
                "max_local_ram_percent": 0.0,
                "average_decision_latency_ms": 0.0,
            }

        except Exception as e:
            logger.error(f"Failed to get routing metrics: {e}")
            return {}

    async def _check_cost_alerts(self):
        """Check cost thresholds and trigger alerts"""
        checks = [
            ("day", self.config.alert_threshold_daily, "daily"),
            ("week", self.config.alert_threshold_weekly, "weekly"),
            ("month", self.config.alert_threshold_monthly, "monthly"),
        ]

        for period, threshold, label in checks:
            summary = await self.get_cost_summary(period)
            actual_cost = summary.get("total_estimated_cost", 0)

            if actual_cost > threshold:
                await self._log_alert(
                    AlertLevel.WARNING,
                    f"cost_threshold_{label}",
                    f"{label.capitalize()} cost ${actual_cost:.2f} exceeds threshold ${threshold:.2f}",
                    summary,
                )

        # Check performance thresholds
        avg_lifetime = self.metrics.average_vm_lifetime_hours
        if avg_lifetime > self.config.max_vm_lifetime_hours:
            await self._log_alert(
                AlertLevel.WARNING,
                "vm_lifetime_threshold",
                f"Average VM lifetime {avg_lifetime:.2f}h exceeds threshold {self.config.max_vm_lifetime_hours}h",
                {"average_vm_lifetime_hours": avg_lifetime},
            )

    async def _log_alert(
        self, level: AlertLevel, alert_type: str, message: str, details: Optional[Dict] = None
    ):
        """Log alert to database and trigger notifications"""
        try:
            import aiosqlite

            async with self._db_lock:
                async with aiosqlite.connect(self.config.db_path) as db:
                    await db.execute(
                        """
                        INSERT INTO alert_history (timestamp, alert_level, alert_type, message, details)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        (
                            datetime.utcnow().isoformat(),
                            level.value,
                            alert_type,
                            message,
                            json.dumps(details or {}),
                        ),
                    )
                    await db.commit()

            # Log to console
            log_func = {
                AlertLevel.INFO: logger.info,
                AlertLevel.WARNING: logger.warning,
                AlertLevel.CRITICAL: logger.critical,
                AlertLevel.EMERGENCY: logger.critical,
            }.get(level, logger.warning)

            log_func(f"💰 ALERT [{level.value.upper()}]: {message}")

            # Send notifications
            if self.config.enable_desktop_notifications:
                await self._send_desktop_notification(f"{level.value.upper()}: {message}")

            # Send email alerts for medium/high severity
            if level in [AlertLevel.MEDIUM, AlertLevel.HIGH]:
                await self._send_email_alert(
                    subject=f"{level.value.upper()} Alert: {alert_type}", message=message
                )

            # Trigger callbacks
            await self._notify_event(
                "alert", {"level": level.value, "type": alert_type, "message": message}
            )

        except Exception as e:
            logger.error(f"Failed to log alert: {e}")

    async def _send_desktop_notification(self, message: str):
        """Send desktop notification (macOS/Linux)"""
        try:
            if os.path.exists("/usr/bin/osascript"):  # macOS
                cmd = [
                    "osascript",
                    "-e",
                    f'display notification "{message}" with title "JARVIS Cost Alert" sound name "Purr"',
                ]
                await asyncio.create_subprocess_exec(*cmd)
        except Exception as e:
            logger.debug(f"Desktop notification failed: {e}")

    async def _send_email_alert(self, subject: str, message: str):
        """Send email alert notification (async)"""
        if not self.config.enable_email_alerts or not self.config.alert_email:
            return

        try:
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText

            import aiosmtplib

            # Email configuration from environment
            smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
            smtp_port = int(os.getenv("SMTP_PORT", "587"))
            smtp_user = os.getenv("SMTP_USER", self.config.alert_email)
            smtp_password = os.getenv("SMTP_PASSWORD", "")

            if not smtp_password:
                logger.debug("SMTP password not configured - skipping email")
                return

            # Create email message
            msg = MIMEMultipart("alternative")
            msg["From"] = smtp_user
            msg["To"] = self.config.alert_email
            msg["Subject"] = f"🤖 JARVIS: {subject}"

            # HTML email content
            html = f"""
            <html>
              <head></head>
              <body style="font-family: Arial, sans-serif; padding: 20px;">
                <h2 style="color: #2196F3;">🤖 JARVIS Cost Alert</h2>
                <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px;">
                  <h3>{subject}</h3>
                  <p style="white-space: pre-wrap;">{message}</p>
                </div>
                <hr style="margin: 20px 0;">
                <p style="color: #666; font-size: 12px;">
                  This is an automated alert from JARVIS Hybrid Cloud Intelligence.<br>
                  <a href="http://localhost:8010/hybrid/status">View Cost Dashboard</a>
                </p>
              </body>
            </html>
            """

            # Plain text fallback
            text = f"JARVIS Cost Alert\n\n{subject}\n\n{message}"

            msg.attach(MIMEText(text, "plain"))
            msg.attach(MIMEText(html, "html"))

            # Send email asynchronously
            await aiosmtplib.send(
                msg,
                hostname=smtp_server,
                port=smtp_port,
                username=smtp_user,
                password=smtp_password,
                start_tls=True,
            )

            logger.info(f"📧 Email alert sent to {self.config.alert_email}")

        except ImportError:
            logger.warning("aiosmtplib not installed - install via: pip install aiosmtplib")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    async def get_orphaned_vms_report(self) -> Dict[str, Any]:
        """Get comprehensive orphaned VMs report"""
        try:
            import aiosqlite

            async with self._db_lock:
                async with aiosqlite.connect(self.config.db_path) as db:
                    async with db.execute(
                        """
                        SELECT instance_id, created_at, deleted_at, runtime_hours,
                               estimated_cost, vm_type, region, zone
                        FROM vm_sessions
                        WHERE is_orphaned = 1
                        ORDER BY created_at DESC
                        LIMIT 100
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
                                    "vm_type": row[5],
                                    "region": row[6],
                                    "zone": row[7],
                                }
                            )

                        return {
                            "total_orphaned_vms": len(orphaned_vms),
                            "total_orphaned_cost": sum(vm["estimated_cost"] for vm in orphaned_vms),
                            "orphaned_vms": orphaned_vms,
                            "max_age_hours": self.config.orphaned_vm_max_age_hours,
                        }

        except Exception as e:
            logger.error(f"Failed to get orphaned VMs report: {e}")
            return {
                "total_orphaned_vms": 0,
                "total_orphaned_cost": 0,
                "orphaned_vms": [],
                "max_age_hours": self.config.orphaned_vm_max_age_hours,
            }

    def register_alert_callback(self, callback: Callable):
        """Register callback for alert events"""
        self._alert_callbacks.append(callback)

    async def _notify_event(self, event_type: str, data: Dict[str, Any]):
        """Notify registered callbacks of events"""
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, data)
                else:
                    callback(event_type, data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_cost_tracker_instance: Optional[CostTracker] = None


def get_cost_tracker(config: Optional[CostTrackerConfig] = None) -> CostTracker:
    """Get or create global cost tracker instance"""
    global _cost_tracker_instance

    if _cost_tracker_instance is None:
        _cost_tracker_instance = CostTracker(config)

    return _cost_tracker_instance


async def initialize_cost_tracking(config: Optional[CostTrackerConfig] = None):
    """Initialize cost tracking system"""
    tracker = get_cost_tracker(config)
    await tracker.initialize()
    logger.info("💰 Advanced cost tracking system initialized")
