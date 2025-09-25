#!/usr/bin/env python3
"""
Intelligent Process Cleanup Manager for JARVIS
Dynamically identifies and cleans up stuck/hanging processes without hardcoding
Uses Swift performance monitoring for minimal overhead
"""

import psutil
import os
import sys
import time
import signal
import asyncio
from typing import List, Dict, Tuple, Optional, Set, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

# Check for Swift availability
try:
    from core.performance_bridge import swift_library

    SWIFT_AVAILABLE = True
except (ImportError, OSError):
    SWIFT_AVAILABLE = False

logger = logging.getLogger(__name__)


class ProcessCleanupManager:
    """Manages cleanup of stuck or zombie processes"""

    def __init__(self):
        """Initialize the cleanup manager"""
        self.swift_monitor = None  # Initialize as None
        
        # Default configuration - Aggressive memory target of 35%
        self.config = {
            'check_interval': 5.0,  # seconds
            'process_timeout': 30.0,  # seconds
            'stuck_process_time': 7200.0,  # 2 hours - consider process stuck
            'memory_threshold': 0.35,  # 35% memory usage target
            'memory_threshold_warning': 0.50,  # 50% warning threshold
            'memory_threshold_critical': 0.70,  # 70% critical threshold
            'cpu_threshold': 0.9,  # 90% CPU usage
            'cpu_threshold_system': 80.0,  # 80% total system CPU usage threshold
            'cpu_threshold_single': 50.0,  # 50% CPU for single process
            'enable_cleanup': True,
            'aggressive_cleanup': True,  # Enable aggressive memory management
            # JARVIS-specific patterns
            'jarvis_patterns': [
                'jarvis', 'main.py', 'jarvis_backend', 'jarvis_voice',
                'voice_unlock', 'websocket_server', 'jarvis-ai-agent',
                'unified_command_processor', 'resource_manager'
            ],
            'jarvis_port_patterns': [8000, 8001, 8080, 8765, 5000],  # Common JARVIS ports
            'system_critical': [
                'kernel_task', 'WindowServer', 'loginwindow', 'launchd',
                'systemd', 'init', 'Finder', 'Dock', 'SystemUIServer'
            ]
        }
        
        # Learning patterns
        self.problem_patterns = {}
        self.cleanup_history = []
        
        # Load history if exists
        self._load_cleanup_history()

    def _get_swift_monitor(self):
        """Lazy load the Swift monitor"""
        if self.swift_monitor is None and SWIFT_AVAILABLE:
            from core.swift_system_monitor import get_swift_system_monitor

            self.swift_monitor = get_swift_system_monitor()
        return self.swift_monitor

    def get_system_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the system state using Swift if available"""
        monitor = self._get_swift_monitor()
        if monitor:
            return monitor.get_system_snapshot()
        else:
            # Fallback to psutil
            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_available_mb": psutil.virtual_memory().available
                // (1024 * 1024),
                "timestamp": datetime.now(),
            }

    def analyze_system_state(self) -> Dict[str, Any]:
        """Analyze system state for cleanup"""
        snapshot = self.get_system_snapshot()

        # Use Swift monitor for process details if available
        monitor = self._get_swift_monitor()
        if monitor:
            snapshot["all_processes"] = monitor.get_all_processes()

        state = {
            "cpu_percent": snapshot["cpu_percent"],
            "memory_percent": snapshot["memory_percent"],
            "memory_available_mb": snapshot["memory_available_mb"],
            "timestamp": snapshot["timestamp"],
        }

        # Find problematic processes
        state["high_cpu_processes"] = self._find_high_cpu_processes()
        state["stuck_processes"] = self._find_stuck_processes()
        state["zombie_processes"] = self._find_zombie_processes()
        state["jarvis_processes"] = self._find_jarvis_processes()

        return state

    def _find_high_cpu_processes(self) -> List[Dict]:
        """Find processes using excessive CPU"""
        high_cpu = []

        for proc in psutil.process_iter(
            ["pid", "name", "cpu_percent", "memory_percent"]
        ):
            try:
                # Get CPU usage over a short interval
                cpu = proc.cpu_percent(interval=0.1)

                if cpu > self.config["cpu_threshold_single"]:
                    proc_info = {
                        "pid": proc.pid,
                        "name": proc.name(),
                        "cpu_percent": cpu,
                        "memory_percent": proc.memory_percent(),
                        "cmdline": " ".join(proc.cmdline()[:5]),  # First 5 args
                        "create_time": datetime.fromtimestamp(proc.create_time()),
                        "is_jarvis": self._is_jarvis_process(proc),
                    }
                    high_cpu.append(proc_info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return sorted(high_cpu, key=lambda x: x["cpu_percent"], reverse=True)

    def _find_stuck_processes(self) -> List[Dict]:
        """Find processes that appear to be stuck or hanging"""
        stuck = []
        current_time = time.time()

        for proc in psutil.process_iter(["pid", "name", "create_time", "status"]):
            try:
                # Check if process is in uninterruptible sleep or zombie state
                if proc.status() in [psutil.STATUS_DISK_SLEEP, psutil.STATUS_ZOMBIE]:
                    stuck.append(
                        {
                            "pid": proc.pid,
                            "name": proc.name(),
                            "status": proc.status(),
                            "age_seconds": current_time - proc.create_time(),
                        }
                    )

                # Check for old JARVIS processes that might be stuck
                if self._is_jarvis_process(proc):
                    age = current_time - proc.create_time()
                    if age > self.config["stuck_process_time"]:
                        # Check if it's actually doing something
                        cpu_usage = proc.cpu_percent(interval=1.0)
                        if cpu_usage < 0.1:  # Less than 0.1% CPU - probably stuck
                            stuck.append(
                                {
                                    "pid": proc.pid,
                                    "name": proc.name(),
                                    "status": "likely_stuck",
                                    "age_seconds": age,
                                    "cpu_percent": cpu_usage,
                                }
                            )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return stuck

    def _find_zombie_processes(self) -> List[Dict]:
        """Find zombie processes"""
        zombies = []

        for proc in psutil.process_iter(["pid", "name", "status", "ppid"]):
            try:
                if proc.status() == psutil.STATUS_ZOMBIE:
                    zombies.append(
                        {"pid": proc.pid, "name": proc.name(), "ppid": proc.ppid()}
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return zombies

    def _find_jarvis_processes(self) -> List[Dict]:
        """Find all JARVIS-related processes"""
        jarvis_procs = []

        for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
            try:
                if self._is_jarvis_process(proc):
                    jarvis_procs.append(
                        {
                            "pid": proc.pid,
                            "name": proc.name(),
                            "cmdline": " ".join(proc.cmdline()[:3]),
                            "age_seconds": time.time() - proc.create_time(),
                            "cpu_percent": proc.cpu_percent(interval=0.1),
                            "memory_mb": proc.memory_info().rss // (1024 * 1024),
                        }
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return jarvis_procs

    def _is_jarvis_process(self, proc: psutil.Process) -> bool:
        """Intelligently determine if a process is JARVIS-related"""
        try:
            # Check process name
            proc_name = proc.name().lower()

            # Check command line
            cmdline = " ".join(proc.cmdline()).lower()

            # Dynamic pattern matching
            for pattern in self.config["jarvis_patterns"]:
                if pattern.lower() in proc_name or pattern.lower() in cmdline:
                    return True

            # Check if it's using JARVIS ports
            for conn in proc.connections():
                if conn.laddr.port in self.config["jarvis_port_patterns"]:
                    return True

            return False
        except:
            return False

    def calculate_cleanup_priority(self, proc_info: Dict) -> float:
        """Calculate cleanup priority score (higher = more likely to clean)"""
        score = 0.0

        # CPU usage factor
        if proc_info.get("cpu_percent", 0) > self.config["cpu_threshold_single"]:
            score += proc_info["cpu_percent"] / 100.0 * 0.4

        # Memory usage factor
        if proc_info.get("memory_percent", 0) > 20:
            score += proc_info["memory_percent"] / 100.0 * 0.2

        # Age factor (older = higher priority)
        age_hours = proc_info.get("age_seconds", 0) / 3600
        if age_hours > 1:
            score += min(age_hours / 24, 1.0) * 0.2

        # Stuck/zombie factor
        if proc_info.get("status") in ["zombie", "likely_stuck"]:
            score += 0.5

        # JARVIS process factor (be more aggressive with our own processes)
        if proc_info.get("is_jarvis", False):
            score += 0.3

        # Learn from patterns
        proc_name = proc_info.get("name", "")
        if proc_name in self.problem_patterns:
            score += self.problem_patterns[proc_name] * 0.2

        return min(score, 1.0)

    async def smart_cleanup(self, dry_run: bool = False) -> Dict[str, any]:
        """Perform intelligent cleanup of problematic processes"""
        logger.info("🧹 Starting intelligent process cleanup...")

        # Analyze system state
        state = self.analyze_system_state()

        cleanup_report = {
            "timestamp": datetime.now(),
            "system_state": {
                "cpu_percent": state["cpu_percent"],
                "memory_percent": state["memory_percent"],
            },
            "actions": [],
            "freed_resources": {"cpu_percent": 0, "memory_mb": 0},
        }

        # Build list of cleanup candidates
        candidates = []

        # Add high CPU processes
        for proc in state["high_cpu_processes"]:
            proc["reason"] = "high_cpu"
            proc["priority"] = self.calculate_cleanup_priority(proc)
            candidates.append(proc)

        # Add stuck processes
        for proc in state["stuck_processes"]:
            proc["reason"] = "stuck"
            proc["priority"] = self.calculate_cleanup_priority(proc)
            candidates.append(proc)

        # Add zombies
        for proc in state["zombie_processes"]:
            proc["reason"] = "zombie"
            proc["priority"] = 1.0  # Always clean zombies
            candidates.append(proc)

        # Sort by priority
        candidates.sort(key=lambda x: x["priority"], reverse=True)

        # Cleanup high-priority processes
        for candidate in candidates:
            if candidate["priority"] < 0.3:
                continue  # Skip low priority

            # Skip protected processes
            if candidate["name"] in self.config["system_critical"]:
                continue

            action = {
                "pid": candidate["pid"],
                "name": candidate["name"],
                "reason": candidate["reason"],
                "priority": candidate["priority"],
                "action": "none",
                "success": False,
            }

            if not dry_run:
                try:
                    proc = psutil.Process(candidate["pid"])

                    # Try graceful termination first
                    logger.info(
                        f"Terminating {candidate['name']} (PID: {candidate['pid']}, Reason: {candidate['reason']})"
                    )
                    proc.terminate()

                    # Wait for graceful shutdown
                    try:
                        proc.wait(timeout=5)
                        action["action"] = "terminated"
                        action["success"] = True
                    except psutil.TimeoutExpired:
                        # Force kill if needed
                        logger.warning(
                            f"Force killing {candidate['name']} (PID: {candidate['pid']})"
                        )
                        proc.kill()
                        action["action"] = "killed"
                        action["success"] = True

                    # Estimate freed resources
                    cleanup_report["freed_resources"]["cpu_percent"] += candidate.get(
                        "cpu_percent", 0
                    )
                    cleanup_report["freed_resources"]["memory_mb"] += candidate.get(
                        "memory_mb", 0
                    )

                    # Learn from this
                    self._update_problem_patterns(candidate["name"], True)

                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    action["error"] = str(e)
                    logger.error(f"Failed to cleanup {candidate['name']}: {e}")
            else:
                action["action"] = "would_terminate"

            cleanup_report["actions"].append(action)

        # Clean up orphaned JARVIS ports
        if not dry_run:
            self._cleanup_orphaned_ports()

        # Save cleanup history for learning
        self.cleanup_history.append(cleanup_report)
        self._save_cleanup_history()

        # Log summary
        logger.info(
            f"Cleanup complete: {len(cleanup_report['actions'])} processes handled"
        )
        logger.info(
            f"Freed approximately {cleanup_report['freed_resources']['cpu_percent']:.1f}% CPU, "
            f"{cleanup_report['freed_resources']['memory_mb']}MB memory"
        )

        return cleanup_report

    def _cleanup_orphaned_ports(self):
        """Clean up ports that might be stuck from previous JARVIS runs"""
        for port in self.config["jarvis_port_patterns"]:
            try:
                # Find process using this port
                for conn in psutil.net_connections():
                    if conn.laddr.port == port and conn.status == "LISTEN":
                        try:
                            proc = psutil.Process(conn.pid)
                            # Only kill if it's been running for a while and is idle
                            if time.time() - proc.create_time() > 300:  # 5 minutes
                                if proc.cpu_percent(interval=0.1) < 0.1:
                                    logger.info(
                                        f"Cleaning up orphaned port {port} (PID: {proc.pid})"
                                    )
                                    proc.terminate()
                        except:
                            pass
            except:
                pass

    def _update_problem_patterns(self, process_name: str, was_problematic: bool):
        """Learn from cleanup actions"""
        if process_name not in self.problem_patterns:
            self.problem_patterns[process_name] = 0.0

        # Exponential moving average
        alpha = 0.3
        self.problem_patterns[process_name] = (
            alpha * (1.0 if was_problematic else 0.0)
            + (1 - alpha) * self.problem_patterns[process_name]
        )

    def _save_cleanup_history(self):
        """Save cleanup history for learning"""
        history_file = Path.home() / ".jarvis" / "cleanup_history.json"
        history_file.parent.mkdir(exist_ok=True)

        # Keep only last 100 entries
        recent_history = self.cleanup_history[-100:]

        try:
            with open(history_file, "w") as f:
                json.dump(recent_history, f, default=str, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cleanup history: {e}")

    def get_cleanup_recommendations(self) -> List[str]:
        """Get recommendations based on current system state"""
        state = self.analyze_system_state()
        recommendations = []

        if state["cpu_percent"] > self.config["cpu_threshold_system"]:
            recommendations.append(
                f"System CPU is high ({state['cpu_percent']:.1f}%). Consider closing unnecessary applications."
            )

        # Convert memory_percent to 0-1 range for comparison
        memory_percent_normalized = state["memory_percent"] / 100.0
        
        if memory_percent_normalized > self.config.get("memory_threshold_critical", 0.70):
            recommendations.append(
                f"⚠️ CRITICAL: Memory usage is very high ({state['memory_percent']:.1f}%). "
                f"Immediate action required!"
            )
        elif memory_percent_normalized > self.config.get("memory_threshold_warning", 0.50):
            recommendations.append(
                f"⚡ WARNING: Memory usage is elevated ({state['memory_percent']:.1f}%). "
                f"Target is {self.config['memory_threshold'] * 100:.0f}%."
            )
        elif memory_percent_normalized > self.config["memory_threshold"]:
            recommendations.append(
                f"Memory usage ({state['memory_percent']:.1f}%) exceeds target of "
                f"{self.config['memory_threshold'] * 100:.0f}%. Consider optimization."
            )

        if len(state["zombie_processes"]) > 0:
            recommendations.append(
                f"Found {len(state['zombie_processes'])} zombie processes that should be cleaned."
            )

        if len(state["stuck_processes"]) > 0:
            recommendations.append(
                f"Found {len(state['stuck_processes'])} potentially stuck processes."
            )

        old_jarvis = [p for p in state["jarvis_processes"] if p["age_seconds"] > 3600]
        if old_jarvis:
            recommendations.append(
                f"Found {len(old_jarvis)} old JARVIS processes that may be stuck."
            )

        return recommendations
    
    def _load_cleanup_history(self):
        """Load cleanup history from disk"""
        history_file = Path.home() / ".jarvis" / "cleanup_history.json"
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    self.cleanup_history = json.load(f)
                    # Rebuild problem patterns from history
                    for entry in self.cleanup_history[-50:]:  # Last 50 entries
                        for action in entry.get("actions", []):
                            if action.get("success"):
                                self._update_problem_patterns(
                                    action.get("name", ""), True
                                )
            except Exception as e:
                logger.error(f"Failed to load cleanup history: {e}")
    
    def cleanup_old_jarvis_processes(self, max_age_hours: float = 12.0) -> List[Dict]:
        """
        Specifically clean up old JARVIS processes that have been running too long
        
        Args:
            max_age_hours: Maximum age in hours before considering a JARVIS process stale
            
        Returns:
            List of cleaned up processes
        """
        cleaned = []
        current_time = time.time()
        
        for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
            try:
                if self._is_jarvis_process(proc):
                    age_hours = (current_time - proc.create_time()) / 3600
                    
                    if age_hours > max_age_hours:
                        # Check if it's the current main process
                        cmdline = " ".join(proc.cmdline())
                        if "main.py" in cmdline:
                            # This is likely an old JARVIS main process
                            logger.warning(
                                f"Found stale JARVIS process (PID: {proc.pid}, "
                                f"Age: {age_hours:.1f} hours)"
                            )
                            
                            try:
                                proc.terminate()
                                proc.wait(timeout=5)
                                cleaned.append({
                                    "pid": proc.pid,
                                    "name": proc.name(),
                                    "age_hours": age_hours,
                                    "status": "terminated"
                                })
                                logger.info(f"Terminated old JARVIS process {proc.pid}")
                            except psutil.TimeoutExpired:
                                proc.kill()
                                cleaned.append({
                                    "pid": proc.pid,
                                    "name": proc.name(),
                                    "age_hours": age_hours,
                                    "status": "killed"
                                })
                                logger.warning(f"Force killed old JARVIS process {proc.pid}")
                            except Exception as e:
                                logger.error(f"Failed to clean up PID {proc.pid}: {e}")
                                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return cleaned
    
    def get_jarvis_process_age(self) -> Optional[float]:
        """Get the age of the main JARVIS process in hours"""
        for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
            try:
                if self._is_jarvis_process(proc):
                    cmdline = " ".join(proc.cmdline())
                    if "main.py" in cmdline:
                        age_hours = (time.time() - proc.create_time()) / 3600
                        return age_hours
            except:
                continue
        return None


# Convenience functions for integration
async def cleanup_system_for_jarvis(dry_run: bool = False) -> Dict[str, any]:
    """Main entry point for cleaning up system before JARVIS starts"""
    manager = ProcessCleanupManager()
    return await manager.smart_cleanup(dry_run=dry_run)


def get_system_recommendations() -> List[str]:
    """Get recommendations for system optimization"""
    manager = ProcessCleanupManager()
    return manager.get_cleanup_recommendations()


if __name__ == "__main__":
    # Test the cleanup manager
    import asyncio

    async def test():
        print("🔍 Analyzing system state...")

        manager = ProcessCleanupManager()
        state = manager.analyze_system_state()

        print(f"\n📊 System State:")
        print(f"  CPU: {state['cpu_percent']:.1f}%")
        print(f"  Memory: {state['memory_percent']:.1f}%")
        print(f"  High CPU processes: {len(state['high_cpu_processes'])}")
        print(f"  Stuck processes: {len(state['stuck_processes'])}")
        print(f"  Zombie processes: {len(state['zombie_processes'])}")
        print(f"  JARVIS processes: {len(state['jarvis_processes'])}")

        print("\n💡 Recommendations:")
        for rec in manager.get_cleanup_recommendations():
            print(f"  • {rec}")

        print("\n🧹 Performing dry run cleanup...")
        report = await manager.smart_cleanup(dry_run=True)

        print(f"\nWould clean {len(report['actions'])} processes:")
        for action in report["actions"][:5]:  # Show first 5
            print(f"  • {action['name']} (PID: {action['pid']}) - {action['reason']}")

        if len(report["actions"]) > 5:
            print(f"  ... and {len(report['actions']) - 5} more")

    asyncio.run(test())
