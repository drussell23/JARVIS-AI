"""
Memory Optimization Configuration

This module provides configuration classes and settings for memory optimization,
specifically designed to customize which applications to target and how
aggressively to free memory for LangChain operations.

The module defines application profiles with different priorities and behaviors,
browser handling configurations, and protection rules for critical system processes.

Example:
    >>> from memory.optimization_config import optimization_config
    >>> profile = optimization_config.get_app_profile("cursor")
    >>> if profile:
    ...     print(f"Priority: {profile.priority}")
    Priority: 9
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field

@dataclass
class AppProfile:
    """Profile configuration for a specific application.
    
    Defines how an application should be handled during memory optimization,
    including closure priority, suspension capabilities, and memory thresholds.
    
    Attributes:
        name: Human-readable name of the application
        patterns: List of strings to match against process names
        priority: Priority level from 1-10, where higher values indicate
                 higher likelihood of being closed/suspended
        can_close: Whether the application can be closed during optimization
        can_suspend: Whether the application can be suspended instead of closed
        graceful_close: Whether to attempt graceful closure before force termination
        min_memory_percent: Minimum memory usage percentage to target this app
    
    Example:
        >>> profile = AppProfile(
        ...     name="Test App",
        ...     patterns=["test", "testapp"],
        ...     priority=5,
        ...     min_memory_percent=2.0
        ... )
        >>> profile.can_close
        True
    """
    name: str
    patterns: List[str]  # Patterns to match process names
    priority: int  # 1-10, higher = more likely to close
    can_close: bool = True
    can_suspend: bool = True
    graceful_close: bool = True  # Try graceful close before force
    min_memory_percent: float = 1.0  # Only target if using this much memory

class OptimizationConfig:
    """Configuration manager for memory optimization strategies.
    
    Manages application profiles, memory thresholds, and optimization rules
    for freeing memory to enable LangChain operations. Provides methods to
    determine which processes should be closed, suspended, or protected.
    
    Attributes:
        target_memory_percent: Target memory usage percentage for LangChain
        aggressive_mode_threshold: Memory threshold to trigger aggressive optimization
        high_priority_apps: List of applications with high closure priority
        suspendable_apps: List of applications that can be suspended
        browser_config: Configuration for browser tab and memory management
        helper_patterns: Process name patterns for helper processes
        protected_patterns: Process name patterns for protected system processes
    
    Example:
        >>> config = OptimizationConfig()
        >>> config.target_memory_percent
        45
        >>> config.is_protected("kernel")
        True
    """
    
    def __init__(self) -> None:
        """Initialize optimization configuration with default settings.
        
        Sets up application profiles, memory thresholds, and process patterns
        for memory optimization operations.
        """
        # Target memory percentage for LangChain
        self.target_memory_percent = 45
        
        # Aggressive mode settings
        self.aggressive_mode_threshold = 55  # Use aggressive mode if above this
        
        # Apps to close first (high priority)
        self.high_priority_apps = [
            AppProfile(
                name="Cursor",
                patterns=["cursor", "cursor helper"],
                priority=9,
                graceful_close=True,
                min_memory_percent=2.0
            ),
            AppProfile(
                name="Visual Studio Code",
                patterns=["code", "code helper", "vscode"],
                priority=9,
                graceful_close=True,
                min_memory_percent=3.0
            ),
            AppProfile(
                name="IntelliJ IDEA",
                patterns=["intellij", "idea"],
                priority=9,
                graceful_close=True,
                min_memory_percent=3.0
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
            "close_if_memory_percent": 5.0,  # Close browser if using > 5%
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
    
    def get_app_profile(self, process_name: str) -> Optional[AppProfile]:
        """Get application profile for a process by name.
        
        Searches through high priority and suspendable applications to find
        a matching profile based on process name patterns.
        
        Args:
            process_name: Name of the process to look up
            
        Returns:
            AppProfile object if a match is found, None otherwise
            
        Example:
            >>> config = OptimizationConfig()
            >>> profile = config.get_app_profile("cursor")
            >>> profile.name if profile else None
            'Cursor'
        """
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
        """Check if a process is protected from termination.
        
        Protected processes include critical system processes and the current
        application's own processes that should never be killed.
        
        Args:
            process_name: Name of the process to check
            
        Returns:
            True if the process is protected, False otherwise
            
        Example:
            >>> config = OptimizationConfig()
            >>> config.is_protected("kernel")
            True
            >>> config.is_protected("notepad")
            False
        """
        process_lower = process_name.lower()
        for pattern in self.protected_patterns:
            if pattern in process_lower:
                return True
        return False
    
    def is_helper(self, process_name: str) -> bool:
        """Check if a process is a helper process that can be terminated.
        
        Helper processes are typically background processes that support
        main applications and can be safely killed to free memory.
        
        Args:
            process_name: Name of the process to check
            
        Returns:
            True if the process is a killable helper, False otherwise
            
        Example:
            >>> config = OptimizationConfig()
            >>> config.is_helper("chrome helper")
            True
            >>> config.is_helper("kernel helper")  # Protected
            False
        """
        process_lower = process_name.lower()
        for pattern in self.helper_patterns:
            if pattern in process_lower:
                return not self.is_protected(process_name)
        return False
    
    def should_close_for_langchain(self, process_name: str, memory_percent: float) -> bool:
        """Determine if a process should be closed to enable LangChain operations.
        
        Evaluates whether a process should be terminated based on its profile,
        memory usage, and protection status to free memory for LangChain.
        
        Args:
            process_name: Name of the process to evaluate
            memory_percent: Current memory usage percentage of the process
            
        Returns:
            True if the process should be closed, False otherwise
            
        Example:
            >>> config = OptimizationConfig()
            >>> config.should_close_for_langchain("cursor", 3.0)
            True
            >>> config.should_close_for_langchain("kernel", 10.0)
            False
        """
        # Don't close if protected
        if self.is_protected(process_name):
            return False
        
        # Check app profile
        profile = self.get_app_profile(process_name)
        if profile:
            return (
                profile.can_close and 
                memory_percent >= profile.min_memory_percent * 0.5 and  # Lower threshold
                profile.priority >= 6  # Lower priority threshold
            )
        
        # Check if it's a browser using lots of memory
        process_lower = process_name.lower()
        for browser in self.browser_config["browsers"]:
            if browser in process_lower and memory_percent > self.browser_config["close_if_memory_percent"]:
                return True
        
        return False

# Global config instance
optimization_config = OptimizationConfig()