#!/usr/bin/env python3
"""
Global Startup Announcement Coordinator
========================================

Prevents multiple overlapping JARVIS startup announcements by coordinating
all systems (Voice API, Display Monitor, WebSocket, etc.) to speak only ONCE.

Problem:
- jarvis_voice_api.py announces: "Good morning, Sir. JARVIS systems initialized..."
- advanced_display_monitor.py announces: "JARVIS online. Display detected..."
- Multiple voices overlap and confuse the user

Solution:
- Single global lock prevents duplicate announcements
- First system to request announcement wins
- All other systems are silently ignored
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class StartupAnnouncementCoordinator:
    """
    Global singleton coordinator for startup announcements.

    Usage:
        coordinator = get_startup_coordinator()
        announced = await coordinator.announce_if_first("jarvis_voice_api")
        if announced:
            # This system made the announcement
        else:
            # Another system already announced, skip
    """

    def __init__(self):
        self._announced = False
        self._lock = asyncio.Lock()
        self._first_announcer = None
        self._announcement_time = None

    async def announce_if_first(self, system_name: str, message: Optional[str] = None) -> bool:
        """
        Attempt to make startup announcement. Returns True if this system should announce.

        Args:
            system_name: Name of the system requesting announcement (e.g., "jarvis_voice_api")
            message: Optional custom message (otherwise auto-generated)

        Returns:
            True if this system should announce, False if another system already did
        """
        async with self._lock:
            if self._announced:
                logger.info(
                    f"[STARTUP COORDINATOR] {system_name} requested announcement, "
                    f"but {self._first_announcer} already announced at {self._announcement_time}"
                )
                return False

            # This system wins - mark as announced
            self._announced = True
            self._first_announcer = system_name
            self._announcement_time = datetime.now().isoformat()

            logger.info(
                f"[STARTUP COORDINATOR] ✅ {system_name} will make the startup announcement"
            )
            return True

    def has_announced(self) -> bool:
        """Check if startup has been announced"""
        return self._announced

    def reset(self):
        """Reset coordinator (for testing or manual restart)"""
        self._announced = False
        self._first_announcer = None
        self._announcement_time = None
        logger.info("[STARTUP COORDINATOR] Reset - ready for new announcement")

    @staticmethod
    def generate_greeting() -> str:
        """Generate time-aware startup greeting"""
        hour = datetime.now().hour

        if 5 <= hour < 12:
            return "Good morning, Sir. JARVIS systems initialized and ready for your command."
        elif 12 <= hour < 17:
            return "Good afternoon, Sir. JARVIS at your disposal."
        elif 17 <= hour < 22:
            return "Good evening, Sir. JARVIS at your service."
        else:
            return "Welcome back, Sir. JARVIS systems online despite the late hour."


# Global singleton instance
_coordinator_instance: Optional[StartupAnnouncementCoordinator] = None


def get_startup_coordinator() -> StartupAnnouncementCoordinator:
    """Get global startup announcement coordinator singleton"""
    global _coordinator_instance

    if _coordinator_instance is None:
        _coordinator_instance = StartupAnnouncementCoordinator()
        logger.info("[STARTUP COORDINATOR] Created global coordinator instance")

    return _coordinator_instance


# Convenience function for simple usage
async def should_announce_startup(system_name: str) -> bool:
    """
    Simple helper - returns True if this system should announce startup.

    Usage:
        if await should_announce_startup("my_system"):
            await speak("JARVIS online")
    """
    coordinator = get_startup_coordinator()
    return await coordinator.announce_if_first(system_name)
