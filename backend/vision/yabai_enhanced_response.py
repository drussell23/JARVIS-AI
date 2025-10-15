"""
Enhanced response generation for Yabai-based multi-space intelligence
Provides detailed, accurate descriptions of Mission Control spaces
"""

import logging
from typing import Dict, List, Any, Optional
from .yabai_space_detector import get_yabai_detector

logger = logging.getLogger(__name__)


class YabaiEnhancedResponse:
    """Generate intelligent responses based on Yabai space data"""

    def __init__(self):
        self.yabai_detector = get_yabai_detector()

    def generate_workspace_overview(self, query: str) -> str:
        """Generate a comprehensive workspace overview using Yabai data"""

        if not self.yabai_detector.is_available():
            return "I'm unable to detect your Mission Control spaces. Please ensure Yabai is running."

        # Get workspace summary
        summary = self.yabai_detector.get_workspace_summary()

        if summary["total_spaces"] == 0:
            return "I couldn't detect any Mission Control spaces. Yabai may need to be restarted."

        # Build detailed response
        response = []

        # Opening summary
        response.append(
            f"Sir, you have **{summary['total_spaces']} Mission Control spaces** active"
        )

        if summary["total_windows"] > 0:
            response.append(
                f"with **{summary['total_windows']} windows** across **{summary['total_applications']} applications**."
            )
        else:
            response.append("with no windows currently open.")

        # Current space info
        if summary["current_space"]:
            current = summary["current_space"]
            response.append(f"\n\nYou're currently on **Space {current['space_id']}**")

            if current["window_count"] > 0:
                apps = current["applications"]
                if len(apps) == 1:
                    response.append(f"working in **{apps[0]}**")
                    # Add window title for context
                    if current["windows"] and current["windows"][0]["title"]:
                        title = current["windows"][0]["title"][:60]
                        if len(current["windows"][0]["title"]) > 60:
                            title += "..."
                        response.append(f"on *{title}*.")
                    else:
                        response.append(".")
                else:
                    response.append(f"with **{', '.join(apps[:2])}**")
                    if len(apps) > 2:
                        response.append(f"and {len(apps)-2} other apps.")
                    else:
                        response.append(".")
            else:
                response.append("which is currently empty.")

        # Detailed space breakdown
        response.append("\n\n**Detailed Space Breakdown:**")

        for space in summary["spaces"]:
            space_desc = f"\n\n**Space {space['space_id']}**"

            # Add indicators
            indicators = []
            if space["is_current"]:
                indicators.append("📍 Current")
            if space["is_fullscreen"]:
                indicators.append("🔳 Fullscreen")
            if space["is_visible"]:
                indicators.append("👁 Visible")

            if indicators:
                space_desc += f" [{', '.join(indicators)}]"

            space_desc += ":"

            if space["window_count"] == 0:
                space_desc += " Empty workspace"
            else:
                # List applications and key windows
                apps = space["applications"]
                windows = space["windows"]

                if len(apps) == 1:
                    space_desc += f"\n- **{apps[0]}**"
                    if windows and windows[0]["title"]:
                        title = windows[0]["title"]
                        # Shorten long titles
                        if len(title) > 80:
                            title = title[:77] + "..."
                        space_desc += f"\n  - *{title}*"
                else:
                    space_desc += f"\n- **Applications**: {', '.join(apps[:3])}"
                    if len(apps) > 3:
                        space_desc += f" and {len(apps)-3} more"

                    # Show first couple window titles
                    for i, window in enumerate(windows[:2]):
                        if window["title"]:
                            title = window["title"]
                            if len(title) > 60:
                                title = title[:57] + "..."
                            space_desc += f"\n  - {window['app']}: *{title}*"

                    if len(windows) > 2:
                        space_desc += f"\n  - ...and {len(windows)-2} more windows"

            response.append(space_desc)

        # Activity summary
        response.append("\n\n**Activity Summary:**")

        # Group spaces by activity type
        dev_spaces = []
        browser_spaces = []
        comm_spaces = []
        other_spaces = []
        empty_spaces = []

        for space in summary["spaces"]:
            if space["window_count"] == 0:
                empty_spaces.append(space["space_id"])
            elif any(
                app in ["Cursor", "Code", "Terminal", "Xcode"]
                for app in space["applications"]
            ):
                dev_spaces.append(space["space_id"])
            elif any(
                app in ["Google Chrome", "Safari", "Firefox"]
                for app in space["applications"]
            ):
                browser_spaces.append(space["space_id"])
            elif any(
                app in ["WhatsApp", "Slack", "Discord", "Messages"]
                for app in space["applications"]
            ):
                comm_spaces.append(space["space_id"])
            else:
                other_spaces.append(space["space_id"])

        if dev_spaces:
            response.append(
                f"\n- 💻 **Development** on spaces: {', '.join(map(str, dev_spaces))}"
            )
        if browser_spaces:
            response.append(
                f"\n- 🌐 **Browsing** on spaces: {', '.join(map(str, browser_spaces))}"
            )
        if comm_spaces:
            response.append(
                f"\n- 💬 **Communication** on spaces: {', '.join(map(str, comm_spaces))}"
            )
        if other_spaces:
            response.append(
                f"\n- 📂 **Other tasks** on spaces: {', '.join(map(str, other_spaces))}"
            )
        if empty_spaces:
            response.append(
                f"\n- ⬜ **Empty** spaces: {', '.join(map(str, empty_spaces))}"
            )

        # Workspace organization insight
        response.append("\n\n**Workspace Organization:**")
        if summary["total_spaces"] >= 7:
            response.append("\nYou're using a comprehensive multi-space setup, ")
            if dev_spaces and browser_spaces:
                response.append(
                    "with clear separation between development and browsing activities. "
                )
            if empty_spaces:
                response.append(
                    f"You have {len(empty_spaces)} empty spaces available for new workflows."
                )
            else:
                response.append("All spaces are actively being used.")

        return "".join(response)

    def should_use_yabai_response(self, query: str) -> bool:
        """Determine if we should use Yabai-enhanced response"""

        # Keywords that indicate multi-space queries
        multi_space_keywords = [
            "spaces",
            "desktop",
            "mission control",
            "workspace",
            "across",
            "all my",
            "what's happening",
            "overview",
            "summary",
            "everything",
            "multiple",
        ]

        query_lower = query.lower()
        return any(keyword in query_lower for keyword in multi_space_keywords)


# Global instance
yabai_response = YabaiEnhancedResponse()


def get_yabai_response() -> YabaiEnhancedResponse:
    """Get the global Yabai response generator"""
    return yabai_response
