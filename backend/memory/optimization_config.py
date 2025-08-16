"""
Memory Optimization Configuration
Customize which apps to target and how aggressively to free memory
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field


@dataclass
class AppProfile:
    """Profile for a specific application"""
    name: str
    patterns: List[str]  # Patterns to match process names
    priority: int  # 1-10, higher = more likely to close
    can_close: bool = True
    can_suspend: bool = True
    graceful_close: bool = True  # Try graceful close before force
    min_memory_percent: float = 1.0  # Only target if using this much memory


class OptimizationConfig:
    """Configuration for memory optimization"""
    
    def __init__(self):
        # Target memory percentage for LangChain
        self.target_memory_percent = 45
        
        # Apps to close first (high priority)
        self.high_priority_apps = [
            AppProfile(
                name="Cursor",
                patterns=["cursor", "cursor helper"],
                priority=9,
                graceful_close=True,
                min_memory_percent=5.0
            ),
            AppProfile(
                name="Visual Studio Code",
                patterns=["code", "code helper", "vscode"],
                priority=9,
                graceful_close=True,
                min_memory_percent=5.0
            ),
            AppProfile(
                name="IntelliJ IDEA",
                patterns=["intellij", "idea"],
                priority=9,
                graceful_close=True,
                min_memory_percent=5.0
            ),
            AppProfile(
                name="WhatsApp",
                patterns=["whatsapp"],
                priority=8,
                graceful_close=False,
                min_memory_percent=1.0
            ),
            AppProfile(
                name="Slack",
                patterns=["slack"],
                priority=7,
                graceful_close=True,
                min_memory_percent=2.0
            ),
            AppProfile(
                name="Discord",
                patterns=["discord"],
                priority=7,
                graceful_close=True,
                min_memory_percent=2.0
            ),
            AppProfile(
                name="Docker Desktop",
                patterns=["docker", "docker desktop"],
                priority=8,
                graceful_close=True,
                min_memory_percent=3.0
            ),
        ]
        
        # Apps to suspend (medium priority)
        self.suspendable_apps = [
            AppProfile(
                name="Spotify",
                patterns=["spotify"],
                priority=5,
                can_close=False,
                can_suspend=True,
                min_memory_percent=1.0
            ),
            AppProfile(
                name="Music",
                patterns=["music"],
                priority=5,
                can_close=False,
                can_suspend=True,
                min_memory_percent=1.0
            ),
            AppProfile(
                name="Messages",
                patterns=["messages"],
                priority=4,
                can_close=False,
                can_suspend=True,
                min_memory_percent=0.5
            ),
        ]
        
        # Browser handling
        self.browser_config = {
            "max_tabs": 3,  # Keep only this many tabs
            "close_if_memory_percent": 10.0,  # Close browser if using > 10%
            "browsers": ["chrome", "safari", "firefox", "edge", "brave"]
        }
        
        # Helper processes to kill
        self.helper_patterns = [
            "helper", "renderer", "gpu process", "utility",
            "crashpad", "reportcrash", "mdworker", "mds_stores",
            "photoanalysisd", "cloudphotod", "bird",
            "commerce", "akd", "tccd", "nsurlsessiond"
        ]
        
        # Protected processes (never kill)
        self.protected_patterns = [
            "kernel", "launchd", "systemd", "init",
            "windowserver", "loginwindow", "finder",
            "dock", "systemuiserver", "python", "node",
            "uvicorn", "fastapi"  # Don't kill our own servers!
        ]
    
    def get_app_profile(self, process_name: str):
        """Get profile for a process by name"""
        process_lower = process_name.lower()
        
        # Check high priority apps
        for app in self.high_priority_apps:
            for pattern in app.patterns:
                if pattern.lower() in process_lower:
                    return app
        
        # Check suspendable apps
        for app in self.suspendable_apps:
            for pattern in app.patterns:
                if pattern.lower() in process_lower:
                    return app
        
        return None
    
    def is_protected(self, process_name: str) -> bool:
        """Check if process is protected"""
        process_lower = process_name.lower()
        for pattern in self.protected_patterns:
            if pattern in process_lower:
                return True
        return False
    
    def is_helper(self, process_name: str) -> bool:
        """Check if process is a helper that can be killed"""
        process_lower = process_name.lower()
        for pattern in self.helper_patterns:
            if pattern in process_lower:
                return not self.is_protected(process_name)
        return False
    
    def should_close_for_langchain(self, process_name: str, memory_percent: float) -> bool:
        """Determine if process should be closed to enable LangChain"""
        # Don't close if protected
        if self.is_protected(process_name):
            return False
        
        # Check app profile
        profile = self.get_app_profile(process_name)
        if profile:
            return (
                profile.can_close and 
                memory_percent >= profile.min_memory_percent and
                profile.priority >= 7  # Only high priority for LangChain
            )
        
        # Check if it's a browser using lots of memory
        process_lower = process_name.lower()
        for browser in self.browser_config["browsers"]:
            if browser in process_lower and memory_percent > self.browser_config["close_if_memory_percent"]:
                return True
        
        return False


# Global config instance
optimization_config = OptimizationConfig()