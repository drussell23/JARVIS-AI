#!/usr/bin/env python3
"""
Proactive Monitoring Handler for JARVIS
Enables real-time monitoring and reporting of workspace changes
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class ProactiveMonitoringHandler:
    """Handles proactive monitoring commands and workspace change reporting"""

    def __init__(self, vision_command_handler):
        self.vision_handler = vision_command_handler
        self.monitoring_active = False
        self.report_changes = False
        self.last_workspace_state = None
        self.monitoring_task = None

    async def handle_monitoring_request(self, command: str) -> Dict[str, Any]:
        """Handle various monitoring requests"""
        command_lower = command.lower()

        # IMPORTANT: Exclude lock/unlock screen commands - not monitoring
        if ("lock" in command_lower and "screen" in command_lower) or (
            "unlock" in command_lower and "screen" in command_lower
        ):
            return {
                "handled": False,
                "reason": "Lock/unlock screen commands are not monitoring commands",
            }

        # Check for reporting activation
        if any(
            phrase in command_lower
            for phrase in [
                "report any changes",
                "report activities",
                "tell me what changes",
                "notify me of changes",
                "watch for changes",
                "monitor changes",
            ]
        ):
            return await self.enable_change_reporting()

        # IMPORTANT: Skip desktop space queries - they should be handled by vision pipeline
        if any(
            phrase in command_lower
            for phrase in [
                "desktop space",
                "desktop spaces",
                "across my desktop",
                "across desktop",
            ]
        ):
            return {
                "handled": False,
                "reason": "Desktop space queries should be handled by vision pipeline",
            }

        # Check for workspace insights (but NOT desktop space queries)
        if any(
            phrase in command_lower
            for phrase in [
                "workspace insights",
                "workspace activity",
                "show me changes",
            ]
        ):
            # Double-check it's not a desktop space query
            if not any(
                phrase in command_lower
                for phrase in ["desktop space", "desktop spaces", "across my desktop"]
            ):
                return await self.get_workspace_insights()

        return {"handled": False, "reason": "Not a monitoring command"}

    async def enable_change_reporting(self) -> Dict[str, Any]:
        """Enable proactive change reporting"""
        self.report_changes = True

        # Start monitoring if not already active
        if not self.monitoring_active and self.vision_handler.intelligence:
            # Enable multi-space monitoring
            if hasattr(
                self.vision_handler.intelligence, "start_multi_space_monitoring"
            ):
                success = (
                    await self.vision_handler.intelligence.start_multi_space_monitoring()
                )
                if success:
                    self.monitoring_active = True
                    # Start the monitoring loop
                    self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        response = """I'll start monitoring your workspace for changes across all desktop spaces. I'll notify you when I detect:
        
• Applications opening or closing
• Window focus changes
• New desktop spaces
• Significant activity patterns

The purple indicator confirms I'm actively watching. What specific changes are most important to you?"""

        return {
            "handled": True,
            "response": response,
            "monitoring_active": True,
            "report_changes": True,
        }

    async def _monitoring_loop(self):
        """Main monitoring loop that detects and reports changes"""
        logger.info("Starting proactive monitoring loop")

        while self.monitoring_active and self.report_changes:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds

                if not self.vision_handler.intelligence:
                    continue

                # Get current workspace state
                current_state = await self._get_workspace_state()

                if self.last_workspace_state:
                    # Detect changes
                    changes = self._detect_changes(
                        self.last_workspace_state, current_state
                    )

                    if changes:
                        # Report significant changes
                        await self._report_changes(changes)

                self.last_workspace_state = current_state

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)

    async def _get_workspace_state(self) -> Dict[str, Any]:
        """Get current workspace state across all spaces"""
        try:
            # Use the multi-space detector
            if hasattr(self.vision_handler.intelligence, "multi_space_detector"):
                window_data = (
                    self.vision_handler.intelligence.multi_space_detector.get_all_windows_across_spaces()
                )

                return {
                    "timestamp": datetime.now(),
                    "spaces": window_data.get("spaces", []),
                    "windows": window_data.get("windows", []),
                    "current_space": window_data.get("current_space", {}),
                    "window_count": len(window_data.get("windows", [])),
                    "space_count": len(window_data.get("spaces", [])),
                }
        except Exception as e:
            logger.error(f"Error getting workspace state: {e}")
            return {}

    def _detect_changes(
        self, old_state: Dict[str, Any], new_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect significant changes between states"""
        changes = []

        # Check for new/closed windows
        old_windows = {w.get("window_id"): w for w in old_state.get("windows", [])}
        new_windows = {w.get("window_id"): w for w in new_state.get("windows", [])}

        # New windows
        for wid, window in new_windows.items():
            if wid not in old_windows:
                changes.append(
                    {
                        "type": "window_opened",
                        "app": window.get("app_name", "Unknown"),
                        "title": window.get("window_title", "Untitled"),
                        "space": window.get("space_id", 1),
                    }
                )

        # Closed windows
        for wid, window in old_windows.items():
            if wid not in new_windows:
                changes.append(
                    {
                        "type": "window_closed",
                        "app": window.get("app_name", "Unknown"),
                        "title": window.get("window_title", "Untitled"),
                        "space": window.get("space_id", 1),
                    }
                )

        # Space changes
        if old_state.get("current_space", {}).get("id") != new_state.get(
            "current_space", {}
        ).get("id"):
            changes.append(
                {
                    "type": "space_changed",
                    "from": old_state.get("current_space", {}).get("id", 1),
                    "to": new_state.get("current_space", {}).get("id", 1),
                }
            )

        return changes

    async def _report_changes(self, changes: List[Dict[str, Any]]):
        """Report changes through JARVIS voice"""
        for change in changes:
            message = self._format_change_message(change)
            if message and self.vision_handler.jarvis_api:
                try:
                    await self.vision_handler.jarvis_api.speak_proactive(message)
                except Exception as e:
                    logger.error(f"Failed to report change: {e}")

    def _format_change_message(self, change: Dict[str, Any]) -> str:
        """Format change into natural language"""
        change_type = change.get("type")

        if change_type == "window_opened":
            return (
                f"I notice you've opened {change['app']} on Desktop {change['space']}"
            )
        elif change_type == "window_closed":
            return f"{change['app']} has been closed on Desktop {change['space']}"
        elif change_type == "space_changed":
            return f"You've switched from Desktop {change['from']} to Desktop {change['to']}"

        return None

    async def get_workspace_insights(self) -> Dict[str, Any]:
        """Get current workspace insights"""
        if not self.vision_handler.intelligence:
            return {
                "handled": True,
                "response": "I need to initialize my vision systems first. Please try again in a moment.",
            }

        insights = await self.vision_handler.intelligence.get_workspace_insights()

        return {
            "handled": True,
            "response": insights,
            "monitoring_active": self.monitoring_active,
        }

    async def stop_monitoring(self):
        """Stop proactive monitoring"""
        self.monitoring_active = False
        self.report_changes = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            self.monitoring_task = None

        if self.vision_handler.intelligence:
            await self.vision_handler.intelligence.stop_multi_space_monitoring()


# Global instance
_monitoring_handler = None


def get_monitoring_handler(vision_command_handler) -> ProactiveMonitoringHandler:
    """Get or create monitoring handler"""
    global _monitoring_handler
    if _monitoring_handler is None:
        _monitoring_handler = ProactiveMonitoringHandler(vision_command_handler)
    return _monitoring_handler
