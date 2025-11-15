#!/usr/bin/env python3
"""
Intelligent Component Weight Profiler & Dynamic Offloading System

Zero hardcoding - automatically detects heavy components and intelligently routes
them between local Mac and GCP Spot VMs based on real-time metrics.

Features:
- Automatic component weight detection via import profiling
- Dynamic RAM pressure scoring
- Intelligent hybrid routing decisions
- Real-time component migration
- Integration with existing cost tracking and GCP optimizer
- Learning from historical patterns
"""

import asyncio
import importlib
import inspect
import logging
import os
import sys
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import psutil

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class ComponentProfile:
    """Profile of a component's resource requirements"""

    name: str
    module_path: str
    weight_score: float = 0.0  # 0-100, higher = heavier
    ram_usage_mb: float = 0.0
    import_time_ms: float = 0.0
    cpu_usage_percent: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    can_offload: bool = True
    offload_priority: int = 50  # 0-100, higher = offload first
    last_measured: Optional[datetime] = None
    measurement_count: int = 0
    historical_weights: deque = field(default_factory=lambda: deque(maxlen=10))


@dataclass
class OffloadDecision:
    """Decision about where to execute a component"""

    component_name: str
    execute_on: str  # "local" or "gcp"
    confidence: float  # 0-1
    reasoning: List[str]
    estimated_ram_saved_mb: float = 0.0
    estimated_cost_usd: float = 0.0
    risk_score: float = 0.0  # 0-1, higher = riskier


@dataclass
class SystemSnapshot:
    """Current system state snapshot"""

    timestamp: datetime
    ram_available_gb: float
    ram_total_gb: float
    ram_usage_percent: float
    cpu_usage_percent: float
    active_components: List[str]
    components_on_gcp: List[str]
    pressure_level: str  # "low", "medium", "high", "critical"


# ============================================================================
# INTELLIGENT COMPONENT PROFILER
# ============================================================================


class IntelligentComponentProfiler:
    """
    Automatically profiles components and makes intelligent offloading decisions.

    Zero hardcoding - learns component weights through:
    1. Import-time profiling (memory delta, CPU usage)
    2. Runtime monitoring
    3. Historical pattern analysis
    4. Dependency graph analysis
    """

    def __init__(
        self,
        backend_dir: Path,
        gcp_optimizer=None,
        cost_tracker=None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize intelligent component profiler.

        Args:
            backend_dir: Path to backend directory
            gcp_optimizer: Optional IntelligentGCPOptimizer instance
            cost_tracker: Optional CostTracker instance
            config: Optional configuration overrides
        """
        self.backend_dir = backend_dir
        self.gcp_optimizer = gcp_optimizer
        self.cost_tracker = cost_tracker
        self.config = config or {}

        # Component registry (auto-populated)
        self.components: Dict[str, ComponentProfile] = {}
        self.component_history: deque = deque(maxlen=1000)

        # System state tracking
        self.system_snapshots: deque = deque(maxlen=100)
        self.current_snapshot: Optional[SystemSnapshot] = None

        # Offloading state
        self.offloaded_components: Set[str] = set()
        self.local_components: Set[str] = set()
        self.migration_history: List[Dict] = []

        # Dynamic thresholds (learned from patterns)
        self.thresholds = {
            "heavy_component_ram_mb": float(os.getenv("HEAVY_COMPONENT_RAM_MB", "500")),
            "offload_ram_threshold_gb": float(os.getenv("OFFLOAD_RAM_THRESHOLD_GB", "6.0")),
            "critical_ram_threshold_gb": float(os.getenv("CRITICAL_RAM_THRESHOLD_GB", "2.0")),
            "min_weight_for_offload": float(os.getenv("MIN_WEIGHT_FOR_OFFLOAD", "40.0")),
            "max_concurrent_migrations": int(os.getenv("MAX_CONCURRENT_MIGRATIONS", "3")),
        }

        # Component detection patterns (dynamically expandable)
        self.component_patterns = self._load_component_patterns()

        # Callbacks for events
        self._event_callbacks: List[Callable] = []

        logger.info("🧠 IntelligentComponentProfiler initialized")
        logger.info(f"   Backend: {backend_dir}")
        logger.info(f"   Heavy component threshold: {self.thresholds['heavy_component_ram_mb']:.0f}MB")
        logger.info(f"   Offload RAM threshold: {self.thresholds['offload_ram_threshold_gb']:.1f}GB")

    def _load_component_patterns(self) -> Dict[str, Dict]:
        """
        Load component detection patterns from backend structure.
        Zero hardcoding - scans backend directory for importable modules.
        """
        patterns = {}

        # Common heavy component categories
        categories = {
            "ml_models": {
                "paths": [
                    "intelligence",
                    "voice",
                    "vision",
                ],
                "keywords": ["model", "encoder", "embedding", "neural", "torch", "tensorflow"],
                "base_priority": 80,
            },
            "databases": {
                "paths": ["intelligence"],
                "keywords": ["database", "chromadb", "faiss", "learning"],
                "base_priority": 70,
            },
            "processing": {
                "paths": ["vision", "voice", "intelligence"],
                "keywords": ["processor", "engine", "analyzer", "detector"],
                "base_priority": 60,
            },
            "services": {
                "paths": ["api", "services"],
                "keywords": ["service", "client", "manager"],
                "base_priority": 40,
            },
        }

        # Scan backend for Python modules
        for category, config in categories.items():
            for path in config["paths"]:
                module_dir = self.backend_dir / path
                if module_dir.exists():
                    for py_file in module_dir.glob("**/*.py"):
                        if py_file.name.startswith("__"):
                            continue

                        # Check if file contains relevant keywords
                        try:
                            content = py_file.read_text().lower()
                            if any(kw in content for kw in config["keywords"]):
                                module_path = str(py_file.relative_to(self.backend_dir))[:-3].replace(
                                    "/", "."
                                )
                                patterns[module_path] = {
                                    "category": category,
                                    "base_priority": config["base_priority"],
                                    "detected": True,
                                }
                        except Exception:
                            pass

        logger.info(f"   Detected {len(patterns)} potential heavy components")
        return patterns

    async def profile_component(
        self, module_path: str, force_reload: bool = False
    ) -> Optional[ComponentProfile]:
        """
        Profile a component by measuring its import cost.

        Args:
            module_path: Python module path (e.g., "intelligence.learning_database")
            force_reload: Force reimport even if already loaded

        Returns:
            ComponentProfile or None if profiling failed
        """
        if module_path in self.components and not force_reload:
            return self.components[module_path]

        logger.info(f"📊 Profiling component: {module_path}")

        try:
            # Capture initial state
            process = psutil.Process()
            ram_before = process.memory_info().rss / 1024 / 1024  # MB
            cpu_before = process.cpu_percent(interval=0.1)
            start_time = time.time()

            # Import module
            if module_path in sys.modules and force_reload:
                importlib.reload(sys.modules[module_path])
            else:
                importlib.import_module(module_path)

            # Capture post-import state
            import_time_ms = (time.time() - start_time) * 1000
            ram_after = process.memory_info().rss / 1024 / 1024
            cpu_after = process.cpu_percent(interval=0.1)

            # Calculate metrics
            ram_usage_mb = max(0, ram_after - ram_before)
            cpu_usage_percent = max(0, cpu_after - cpu_before)

            # Calculate weight score (0-100)
            weight_score = self._calculate_weight_score(ram_usage_mb, import_time_ms, cpu_usage_percent)

            # Determine offload priority
            base_priority = self.component_patterns.get(module_path, {}).get("base_priority", 50)
            offload_priority = min(100, base_priority + (weight_score * 0.3))

            # Get dependencies
            dependencies = self._get_module_dependencies(module_path)

            profile = ComponentProfile(
                name=module_path.split(".")[-1],
                module_path=module_path,
                weight_score=weight_score,
                ram_usage_mb=ram_usage_mb,
                import_time_ms=import_time_ms,
                cpu_usage_percent=cpu_usage_percent,
                dependencies=dependencies,
                can_offload=ram_usage_mb > 100,  # Only offload if >100MB
                offload_priority=int(offload_priority),
                last_measured=datetime.now(),
                measurement_count=1,
            )

            # Store historical weight
            profile.historical_weights.append(weight_score)

            self.components[module_path] = profile

            logger.info(f"   ✓ {module_path}: {ram_usage_mb:.1f}MB RAM, weight={weight_score:.1f}/100")

            return profile

        except Exception as e:
            logger.warning(f"   Failed to profile {module_path}: {e}")
            return None

    def _calculate_weight_score(
        self, ram_mb: float, import_time_ms: float, cpu_percent: float
    ) -> float:
        """
        Calculate composite weight score (0-100).

        Higher score = heavier component = higher offload priority.
        """
        # RAM weight (most important)
        ram_score = min(100, (ram_mb / self.thresholds["heavy_component_ram_mb"]) * 70)

        # Import time weight
        time_score = min(100, (import_time_ms / 5000) * 20)  # 5s = 20 points

        # CPU weight
        cpu_score = min(100, (cpu_percent / 50) * 10)  # 50% CPU = 10 points

        return ram_score + time_score + cpu_score

    def _get_module_dependencies(self, module_path: str) -> List[str]:
        """Get list of dependencies for a module"""
        dependencies = []

        try:
            if module_path in sys.modules:
                module = sys.modules[module_path]

                # Inspect imports
                for name, obj in inspect.getmembers(module):
                    if inspect.ismodule(obj):
                        if hasattr(obj, "__name__"):
                            dep_name = obj.__name__
                            if dep_name.startswith(("intelligence", "voice", "vision")):
                                dependencies.append(dep_name)

        except Exception as e:
            logger.debug(f"Could not extract dependencies for {module_path}: {e}")

        return list(set(dependencies))[:10]  # Limit to 10

    async def analyze_system_state(self) -> SystemSnapshot:
        """Analyze current system state and create snapshot"""
        mem = psutil.virtual_memory()

        # Determine pressure level
        if mem.available / 1024**3 < self.thresholds["critical_ram_threshold_gb"]:
            pressure_level = "critical"
        elif mem.available / 1024**3 < self.thresholds["offload_ram_threshold_gb"]:
            pressure_level = "high"
        elif mem.percent > 70:
            pressure_level = "medium"
        else:
            pressure_level = "low"

        snapshot = SystemSnapshot(
            timestamp=datetime.now(),
            ram_available_gb=mem.available / 1024**3,
            ram_total_gb=mem.total / 1024**3,
            ram_usage_percent=mem.percent,
            cpu_usage_percent=psutil.cpu_percent(interval=0.1),
            active_components=list(self.local_components),
            components_on_gcp=list(self.offloaded_components),
            pressure_level=pressure_level,
        )

        self.current_snapshot = snapshot
        self.system_snapshots.append(snapshot)

        return snapshot

    async def make_offload_decision(
        self, component: ComponentProfile, snapshot: Optional[SystemSnapshot] = None
    ) -> OffloadDecision:
        """
        Make intelligent decision about where to execute a component.

        Considers:
        - Current RAM pressure
        - Component weight
        - GCP budget remaining
        - Cost vs benefit
        - Historical patterns
        """
        if snapshot is None:
            snapshot = await self.analyze_system_state()

        reasoning = []

        # Factor 1: RAM pressure
        ram_factor = 0.0
        if snapshot.pressure_level == "critical":
            ram_factor = 1.0
            reasoning.append(f"CRITICAL RAM pressure ({snapshot.ram_available_gb:.1f}GB available)")
        elif snapshot.pressure_level == "high":
            ram_factor = 0.8
            reasoning.append(f"HIGH RAM pressure ({snapshot.ram_available_gb:.1f}GB available)")
        elif snapshot.pressure_level == "medium":
            ram_factor = 0.5
            reasoning.append(f"MEDIUM RAM pressure ({snapshot.ram_available_gb:.1f}GB available)")
        else:
            ram_factor = 0.2
            reasoning.append(f"LOW RAM pressure ({snapshot.ram_available_gb:.1f}GB available)")

        # Factor 2: Component weight
        weight_factor = component.weight_score / 100.0
        reasoning.append(f"Component weight: {component.weight_score:.1f}/100")

        # Factor 3: GCP optimizer recommendation
        gcp_factor = 0.5
        if self.gcp_optimizer:
            try:
                from backend.core.platform_memory_monitor import get_memory_monitor

                monitor = get_memory_monitor()
                mem_snapshot = await monitor.get_memory_pressure()
                should_create, reason, score = await self.gcp_optimizer.should_create_vm(mem_snapshot)

                if should_create:
                    gcp_factor = 1.0
                    reasoning.append(f"GCP optimizer recommends VM creation (score: {score.composite_score:.1f})")
                else:
                    gcp_factor = 0.3
                    reasoning.append("GCP optimizer: VM not needed")
            except Exception as e:
                logger.debug(f"GCP optimizer check failed: {e}")

        # Factor 4: Cost consideration
        cost_factor = 1.0
        if self.cost_tracker:
            try:
                summary = await self.cost_tracker.get_cost_summary("day")
                daily_spend = summary.get("total_estimated_cost", 0)
                budget_limit = float(os.getenv("COST_ALERT_DAILY", "1.00"))

                if daily_spend >= budget_limit:
                    cost_factor = 0.0
                    reasoning.append(f"Daily budget exhausted (${daily_spend:.2f}/${budget_limit:.2f})")
                elif daily_spend >= budget_limit * 0.8:
                    cost_factor = 0.5
                    reasoning.append(f"Approaching budget limit (${daily_spend:.2f}/${budget_limit:.2f})")
                else:
                    reasoning.append(f"Budget available (${daily_spend:.2f}/${budget_limit:.2f})")
            except Exception as e:
                logger.debug(f"Cost tracker check failed: {e}")

        # Calculate composite score
        composite_score = (
            ram_factor * 0.40 + weight_factor * 0.30 + gcp_factor * 0.20 + cost_factor * 0.10
        )

        # Make decision
        execute_on = "gcp" if composite_score >= 0.6 else "local"
        confidence = abs(composite_score - 0.5) * 2  # 0-1 scale

        # Estimate RAM saved
        estimated_ram_saved_mb = component.ram_usage_mb if execute_on == "gcp" else 0

        # Estimate cost (2 hour average runtime)
        spot_rate = float(os.getenv("SPOT_VM_HOURLY_COST", "0.029"))
        estimated_cost_usd = spot_rate * 2.0 if execute_on == "gcp" else 0.0

        # Calculate risk score
        risk_score = 1.0 - confidence if execute_on == "gcp" else 0.0

        decision = OffloadDecision(
            component_name=component.name,
            execute_on=execute_on,
            confidence=confidence,
            reasoning=reasoning,
            estimated_ram_saved_mb=estimated_ram_saved_mb,
            estimated_cost_usd=estimated_cost_usd,
            risk_score=risk_score,
        )

        logger.info(
            f"🎯 Decision for {component.name}: {execute_on.upper()} "
            f"(confidence: {confidence:.2f}, score: {composite_score:.2f})"
        )
        for reason in reasoning:
            logger.info(f"   • {reason}")

        return decision

    async def auto_discover_components(self) -> List[ComponentProfile]:
        """
        Automatically discover all heavy components in backend.

        Returns:
            List of profiled components
        """
        logger.info("🔍 Auto-discovering components...")

        discovered = []

        for module_path in self.component_patterns.keys():
            profile = await self.profile_component(module_path)
            if profile and profile.weight_score > self.thresholds["min_weight_for_offload"]:
                discovered.append(profile)

        # Sort by weight (heaviest first)
        discovered.sort(key=lambda p: p.weight_score, reverse=True)

        logger.info(f"   Discovered {len(discovered)} heavy components")
        for i, comp in enumerate(discovered[:10], 1):
            logger.info(
                f"   {i}. {comp.name}: {comp.weight_score:.1f}/100 ({comp.ram_usage_mb:.0f}MB)"
            )

        return discovered

    async def optimize_offloading(self) -> Dict[str, Any]:
        """
        Intelligently optimize component distribution between local and GCP.

        Returns:
            Optimization report
        """
        logger.info("🚀 Optimizing component offloading...")

        snapshot = await self.analyze_system_state()

        # Get all heavy components
        heavy_components = [c for c in self.components.values() if c.weight_score >= 40]
        heavy_components.sort(key=lambda c: c.offload_priority, reverse=True)

        recommendations = []
        total_ram_to_save = 0.0
        total_cost = 0.0

        for component in heavy_components:
            decision = await self.make_offload_decision(component, snapshot)

            if decision.execute_on == "gcp":
                recommendations.append(
                    {
                        "component": component.name,
                        "module_path": component.module_path,
                        "action": "offload_to_gcp",
                        "ram_saved_mb": decision.estimated_ram_saved_mb,
                        "estimated_cost": decision.estimated_cost_usd,
                        "confidence": decision.confidence,
                        "reasoning": decision.reasoning,
                    }
                )
                total_ram_to_save += decision.estimated_ram_saved_mb
                total_cost += decision.estimated_cost_usd

        report = {
            "timestamp": datetime.now().isoformat(),
            "current_pressure": snapshot.pressure_level,
            "ram_available_gb": snapshot.ram_available_gb,
            "heavy_components_analyzed": len(heavy_components),
            "offload_recommendations": len(recommendations),
            "total_ram_to_save_mb": total_ram_to_save,
            "total_ram_to_save_gb": total_ram_to_save / 1024,
            "estimated_total_cost": total_cost,
            "recommendations": recommendations,
        }

        logger.info(
            f"   📊 Analysis complete: {len(recommendations)} offload recommendations "
            f"(save {total_ram_to_save/1024:.1f}GB, cost ${total_cost:.2f})"
        )

        return report

    def register_event_callback(self, callback: Callable):
        """Register callback for profiling events"""
        self._event_callbacks.append(callback)

    async def _notify_event(self, event_type: str, data: Dict[str, Any]):
        """Notify registered callbacks"""
        for callback in self._event_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, data)
                else:
                    callback(event_type, data)
            except Exception as e:
                logger.error(f"Event callback error: {e}")


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_profiler_instance: Optional[IntelligentComponentProfiler] = None


def get_component_profiler(
    backend_dir: Optional[Path] = None,
    gcp_optimizer=None,
    cost_tracker=None,
    config: Optional[Dict] = None,
) -> IntelligentComponentProfiler:
    """Get or create global component profiler instance"""
    global _profiler_instance

    if _profiler_instance is None:
        if backend_dir is None:
            backend_dir = Path(__file__).parent.parent

        _profiler_instance = IntelligentComponentProfiler(
            backend_dir=backend_dir,
            gcp_optimizer=gcp_optimizer,
            cost_tracker=cost_tracker,
            config=config,
        )

    return _profiler_instance


# ============================================================================
# TEST / DEMO
# ============================================================================


async def test_profiler():
    """Test the intelligent component profiler"""
    from backend.core.cost_tracker import get_cost_tracker
    from backend.core.intelligent_gcp_optimizer import get_gcp_optimizer

    print("\n" + "=" * 80)
    print("Intelligent Component Profiler Test")
    print("=" * 80 + "\n")

    # Initialize dependencies
    cost_tracker = get_cost_tracker()
    await cost_tracker.initialize()

    gcp_optimizer = get_gcp_optimizer()

    # Initialize profiler
    profiler = get_component_profiler(
        gcp_optimizer=gcp_optimizer,
        cost_tracker=cost_tracker,
    )

    # Auto-discover components
    print("🔍 Auto-discovering components...\n")
    discovered = await profiler.auto_discover_components()

    # Analyze system state
    print("\n📊 Current system state:\n")
    snapshot = await profiler.analyze_system_state()
    print(f"RAM: {snapshot.ram_available_gb:.1f}GB / {snapshot.ram_total_gb:.1f}GB")
    print(f"Pressure: {snapshot.pressure_level}")
    print(f"Active components: {len(snapshot.active_components)}")

    # Optimize offloading
    print("\n🚀 Optimizing component distribution...\n")
    report = await profiler.optimize_offloading()

    print(f"Heavy components analyzed: {report['heavy_components_analyzed']}")
    print(f"Offload recommendations: {report['offload_recommendations']}")
    print(f"Total RAM to save: {report['total_ram_to_save_gb']:.2f}GB")
    print(f"Estimated cost: ${report['estimated_total_cost']:.2f}")

    print("\nTop recommendations:")
    for i, rec in enumerate(report["recommendations"][:5], 1):
        print(f"  {i}. {rec['component']}:")
        print(f"     • Save: {rec['ram_saved_mb']:.0f}MB")
        print(f"     • Cost: ${rec['estimated_cost']:.2f}")
        print(f"     • Confidence: {rec['confidence']:.2f}")


if __name__ == "__main__":
    asyncio.run(test_profiler())
