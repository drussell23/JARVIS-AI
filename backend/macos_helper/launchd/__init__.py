"""
JARVIS macOS Helper - LaunchAgent Configuration

Provides LaunchAgent/LaunchDaemon setup for running the macOS helper
as a background service that starts on login.

Features:
- LaunchAgent plist generation
- Installation/uninstallation scripts
- Service management (start, stop, status)
- Log rotation configuration
- Automatic restart on failure

Apple Compliance:
- Uses standard launchd mechanisms
- User-level LaunchAgent (not root daemon)
- Respects user preferences

Usage:
    # Generate and install LaunchAgent
    python -m macos_helper.launchd install

    # Uninstall
    python -m macos_helper.launchd uninstall

    # Check status
    python -m macos_helper.launchd status
"""

from .service_manager import (
    LaunchAgentManager,
    ServiceStatus,
    generate_plist,
    install_service,
    uninstall_service,
    get_service_status,
    start_service,
    stop_service,
    restart_service,
)

__all__ = [
    "LaunchAgentManager",
    "ServiceStatus",
    "generate_plist",
    "install_service",
    "uninstall_service",
    "get_service_status",
    "start_service",
    "stop_service",
    "restart_service",
]
