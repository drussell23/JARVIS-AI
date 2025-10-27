"""
Intelligent Request Router for JARVIS Hybrid Architecture
Automatically routes requests to optimal backend based on:
- Task capabilities
- Resource requirements
- Backend health and availability
- Historical performance
- Real-time load
- RAM-aware routing with memory pressure detection
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Import RAM monitor (lazy to avoid circular imports)
_ram_monitor = None


def _dummy_monitor(config=None):
    """Dummy monitor when RAM monitor unavailable"""
    return None


def _get_ram_monitor():
    """Lazy load RAM monitor"""
    global _ram_monitor
    if _ram_monitor is None:
        try:
            from backend.core.advanced_ram_monitor import get_ram_monitor

            _ram_monitor = get_ram_monitor
        except ImportError:
            logger.warning("RAM monitor not available")
            _ram_monitor = _dummy_monitor
    return _ram_monitor


class RouteDecision(Enum):
    """Routing decision"""

    LOCAL = "local"
    CLOUD = "cloud"
    AUTO = "auto"
    NONE = "none"


@dataclass
class RoutingContext:
    """Context for routing decision"""

    command: str
    command_type: Optional[str] = None
    memory_required: Optional[str] = None
    keywords: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class HybridRouter:
    """
    Intelligent router that decides where to execute requests
    Zero hardcoding - all rules from configuration
    """

    def __init__(self, config: Dict):
        self.config = config
        self.routing_config = config["hybrid"]["routing"]
        self.rules = self.routing_config.get("rules", [])
        self.strategy = self.routing_config.get("strategy", "capability_based")

        # Compile regex patterns for performance
        self._compiled_patterns = {}
        self._compile_patterns()

        # Performance tracking
        self.routing_history = []
        self.max_history = 1000

        logger.info(f"🎯 HybridRouter initialized with {len(self.rules)} rules")

    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency"""
        for rule in self.rules:
            if "match" in rule and "pattern" in rule["match"]:
                pattern = rule["match"]["pattern"]
                self._compiled_patterns[rule["name"]] = re.compile(pattern, re.IGNORECASE)

    def route(self, context: RoutingContext) -> Tuple[RouteDecision, Optional[str], Dict[str, Any]]:
        """
        Determine optimal backend for request

        Returns:
            (decision, backend_name, metadata)
        """
        # Extract keywords from command if not provided
        if context.keywords is None:
            context.keywords = self._extract_keywords(context.command)

        # Apply routing rules in order
        for rule in self.rules:
            if self._matches_rule(rule, context):
                decision = self._make_decision(rule, context)
                metadata = {
                    "rule": rule["name"],
                    "strategy": self.strategy,
                    "confidence": self._calculate_confidence(rule, context),
                }

                # Track routing decision
                self._track_decision(context, decision, metadata)

                logger.debug(
                    f"Routed '{context.command}' via rule '{rule['name']}' → {decision.value}"
                )
                return decision, rule.get("route_to"), metadata

        # No rule matched - use default strategy
        decision = RouteDecision.AUTO
        metadata = {"rule": "default", "strategy": self.strategy, "confidence": 0.5}
        return decision, None, metadata

    def _matches_rule(self, rule: Dict, context: RoutingContext) -> bool:
        """Check if context matches routing rule"""
        match_config = rule.get("match", {})

        # Check 'all' wildcard
        if match_config.get("all"):
            return True

        # Check capabilities
        if "capabilities" in match_config:
            match_config["capabilities"]
            # This will be matched against backend capabilities later

        # ============== NEW: RAM-Aware Routing (Phase 2) ==============
        # Check memory pressure conditions
        if "memory_pressure" in match_config:
            required_pressure = match_config["memory_pressure"]
            current_pressure = self._get_current_memory_pressure()

            if not self._matches_memory_pressure(current_pressure, required_pressure):
                return False

        # Check GCP idle time (for cost optimization)
        if "gcp_idle_minutes" in match_config:
            match_config["gcp_idle_minutes"]
            # TODO: Track GCP idle time
            # For now, skip this check
        # ===============================================================

        # Check memory requirements
        if "memory_required" in match_config:
            required_mem = match_config["memory_required"]
            if context.memory_required:
                if not self._matches_memory_requirement(context.memory_required, required_mem):
                    return False
            else:
                # Estimate memory from command
                estimated_mem = self._estimate_memory(context.command)
                if not self._matches_memory_requirement(estimated_mem, required_mem):
                    return False

        # Check keywords
        if "keywords" in match_config and context.keywords:
            required_keywords = match_config["keywords"]
            command_lower = context.command.lower()
            if not any(kw.lower() in command_lower for kw in required_keywords):
                return False

        # Check command type
        if "command_type" in match_config:
            if context.command_type != match_config["command_type"]:
                return False

        # Check regex pattern
        if rule["name"] in self._compiled_patterns:
            pattern = self._compiled_patterns[rule["name"]]
            if not pattern.search(context.command):
                return False

        # Check custom metadata
        if "metadata" in match_config and context.metadata:
            for key, value in match_config["metadata"].items():
                if context.metadata.get(key) != value:
                    return False

        return True

    def _get_current_memory_pressure(self) -> float:
        """Get current memory pressure as percentage"""
        try:
            ram_monitor_factory = _get_ram_monitor()
            if ram_monitor_factory:
                ram_monitor = ram_monitor_factory(self.config)
                return ram_monitor.get_current_pressure_percent()
        except Exception as e:
            logger.debug(f"Could not get RAM pressure: {e}")

        return 0.0  # Safe default

    def _matches_memory_pressure(self, current: float, required: str) -> bool:
        """Check if memory pressure condition is met"""
        # Parse conditions like ">70", "<40", ">=85"
        if required.startswith(">="):
            threshold = float(required[2:])
            return current >= threshold
        elif required.startswith("<="):
            threshold = float(required[2:])
            return current <= threshold
        elif required.startswith(">"):
            threshold = float(required[1:])
            return current > threshold
        elif required.startswith("<"):
            threshold = float(required[1:])
            return current < threshold
        else:
            # Exact match
            try:
                threshold = float(required)
                return abs(current - threshold) < 5  # Within 5% tolerance
            except ValueError:
                return False

    def _matches_memory_requirement(self, actual: str, required: str) -> bool:
        """Check if memory requirement is met"""
        # Parse memory strings like "8GB", ">8GB", "<4GB"
        actual_gb = self._parse_memory(actual)
        required_gb = self._parse_memory(required)

        if required.startswith(">"):
            return actual_gb > required_gb
        elif required.startswith("<"):
            return actual_gb < required_gb
        elif required.startswith(">="):
            return actual_gb >= required_gb
        elif required.startswith("<="):
            return actual_gb <= required_gb
        else:
            return actual_gb >= required_gb

    def _parse_memory(self, mem_str: str) -> float:
        """Parse memory string to GB value"""
        # Remove comparison operators
        mem_str = re.sub(r"^[><=]+", "", mem_str).strip()

        # Extract number and unit
        match = re.match(r"([\d.]+)\s*([KMGT]?B)?", mem_str, re.IGNORECASE)
        if not match:
            return 0.0

        value = float(match.group(1))
        unit = (match.group(2) or "GB").upper()

        # Convert to GB
        multipliers = {"B": 1 / (1024**3), "KB": 1 / (1024**2), "MB": 1 / 1024, "GB": 1, "TB": 1024}

        return value * multipliers.get(unit, 1)

    def _estimate_memory(self, command: str) -> str:
        """Estimate memory requirements from command"""
        command_lower = command.lower()

        # ML/AI tasks
        ml_keywords = ["train", "model", "neural", "deep learning", "llm", "gpt", "analyze large"]
        if any(kw in command_lower for kw in ml_keywords):
            return ">8GB"

        # Vision/image processing
        vision_keywords = ["image", "video", "screenshot", "vision", "ocr"]
        if any(kw in command_lower for kw in vision_keywords):
            return ">2GB"

        # Light tasks
        return "<1GB"

    def _extract_keywords(self, command: str) -> List[str]:
        """Extract meaningful keywords from command"""
        # Remove common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
        }

        words = command.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords

    def _make_decision(self, rule: Dict, context: RoutingContext) -> RouteDecision:
        """Make routing decision based on rule"""
        route_to = rule.get("route_to", "auto")

        if route_to == "auto":
            return RouteDecision.AUTO
        elif route_to in ["local", "gcp", "cloud"]:
            return RouteDecision.LOCAL if route_to == "local" else RouteDecision.CLOUD
        elif route_to == "none":
            return RouteDecision.NONE
        else:
            return RouteDecision.AUTO

    def _calculate_confidence(self, rule: Dict, context: RoutingContext) -> float:
        """Calculate confidence score for routing decision"""
        confidence = 0.5  # Base confidence

        # Higher confidence for specific rules
        match_config = rule.get("match", {})

        if "command_type" in match_config:
            confidence += 0.2

        if "capabilities" in match_config:
            confidence += 0.15

        if "keywords" in match_config:
            # More keywords matched = higher confidence
            required_keywords = match_config["keywords"]
            command_lower = context.command.lower()
            matched = sum(1 for kw in required_keywords if kw.lower() in command_lower)
            confidence += 0.15 * (matched / len(required_keywords))

        if "memory_required" in match_config:
            confidence += 0.1

        return min(confidence, 1.0)

    def _track_decision(self, context: RoutingContext, decision: RouteDecision, metadata: Dict):
        """Track routing decision for analytics"""
        self.routing_history.append(
            {"command": context.command, "decision": decision.value, "metadata": metadata}
        )

        # Limit history size
        if len(self.routing_history) > self.max_history:
            self.routing_history = self.routing_history[-self.max_history :]

    def get_analytics(self) -> Dict[str, Any]:
        """Get routing analytics"""
        if not self.routing_history:
            return {"total": 0}

        total = len(self.routing_history)
        local_count = sum(1 for h in self.routing_history if h["decision"] == "local")
        cloud_count = sum(1 for h in self.routing_history if h["decision"] == "cloud")
        auto_count = sum(1 for h in self.routing_history if h["decision"] == "auto")

        # Rule usage stats
        rule_usage = {}
        for entry in self.routing_history:
            rule = entry["metadata"].get("rule", "unknown")
            rule_usage[rule] = rule_usage.get(rule, 0) + 1

        return {
            "total": total,
            "local": local_count,
            "cloud": cloud_count,
            "auto": auto_count,
            "local_pct": (local_count / total) * 100 if total > 0 else 0,
            "cloud_pct": (cloud_count / total) * 100 if total > 0 else 0,
            "rule_usage": rule_usage,
            "avg_confidence": (
                sum(h["metadata"].get("confidence", 0) for h in self.routing_history) / total
                if total > 0
                else 0
            ),
        }
