"""
Objective-C bridge for Mission Control space detection
Provides access to all Mission Control spaces without visible switching
"""

import ctypes
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ObjCSpaceDetector:
    """Objective-C bridge for Mission Control space detection"""

    def __init__(self):
        self.lib = None
        self._load_library()

    def _load_library(self):
        """Load the Objective-C dynamic library"""
        # DISABLED: This library causes segfaults (exit code -11)
        # The library has memory management issues that cause crashes
        # when processing desktop space queries
        logger.warning(
            "[OBJC_SPACE_DETECTOR] Objective-C library disabled due to segfault issues. "
            "Using Python fallback for space detection."
        )
        self.lib = None
        return
        
        # Original code kept for reference but not executed
        try:
            # Get the path to the compiled library
            lib_path = Path(__file__).parent / "libspace_detection.dylib"

            if not lib_path.exists():
                logger.error(f"Objective-C library not found at {lib_path}")
                return

            # Load the library
            self.lib = ctypes.CDLL(str(lib_path))

            # Define function signatures
            self.lib.enumerate_spaces_json.restype = ctypes.c_char_p
            self.lib.enumerate_spaces_json.argtypes = []

            self.lib.get_space_info_json.restype = ctypes.c_char_p
            self.lib.get_space_info_json.argtypes = [ctypes.c_int]

            self.lib.get_space_count.restype = ctypes.c_int
            self.lib.get_space_count.argtypes = []

            self.lib.get_current_space.restype = ctypes.c_int
            self.lib.get_current_space.argtypes = []

            self.lib.space_exists.restype = ctypes.c_bool
            self.lib.space_exists.argtypes = [ctypes.c_int]

            logger.info("[OBJC_SPACE_DETECTOR] Objective-C library loaded successfully")

        except Exception as e:
            logger.error(f"[OBJC_SPACE_DETECTOR] Failed to load library: {e}")
            self.lib = None

    def is_available(self) -> bool:
        """Check if the Objective-C detector is available"""
        return self.lib is not None

    def enumerate_all_spaces(self) -> List[Dict[str, Any]]:
        """Enumerate all Mission Control spaces"""
        if not self.is_available():
            logger.warning(
                "[OBJC_SPACE_DETECTOR] Library not available, falling back to Python detection"
            )
            return []

        try:
            # Call the Objective-C function
            result_ptr = self.lib.enumerate_spaces_json()

            if not result_ptr:
                logger.error(
                    "[OBJC_SPACE_DETECTOR] Failed to get spaces from Objective-C"
                )
                return []

            # Convert C string to Python string
            result_str = ctypes.string_at(result_ptr).decode("utf-8")

            # Parse JSON
            spaces = json.loads(result_str)

            logger.info(
                f"[OBJC_SPACE_DETECTOR] Detected {len(spaces)} spaces via Objective-C"
            )

            return spaces

        except Exception as e:
            logger.error(f"[OBJC_SPACE_DETECTOR] Error enumerating spaces: {e}")
            return []

    def get_space_info(self, space_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific space"""
        if not self.is_available():
            return None

        try:
            # Call the Objective-C function
            result_ptr = self.lib.get_space_info_json(space_id)

            if not result_ptr:
                logger.error(
                    f"[OBJC_SPACE_DETECTOR] Failed to get info for space {space_id}"
                )
                return None

            # Convert C string to Python string
            result_str = ctypes.string_at(result_ptr).decode("utf-8")

            # Parse JSON
            space_info = json.loads(result_str)

            return space_info

        except Exception as e:
            logger.error(
                f"[OBJC_SPACE_DETECTOR] Error getting space info for {space_id}: {e}"
            )
            return None

    def get_space_count(self) -> int:
        """Get the total number of spaces"""
        if not self.is_available():
            return 0

        try:
            return self.lib.get_space_count()
        except Exception as e:
            logger.error(f"[OBJC_SPACE_DETECTOR] Error getting space count: {e}")
            return 0

    def get_current_space(self) -> int:
        """Get the current active space"""
        if not self.is_available():
            return 1

        try:
            return self.lib.get_current_space()
        except Exception as e:
            logger.error(f"[OBJC_SPACE_DETECTOR] Error getting current space: {e}")
            return 1

    def space_exists(self, space_id: int) -> bool:
        """Check if a space exists"""
        if not self.is_available():
            return False

        try:
            return self.lib.space_exists(space_id)
        except Exception as e:
            logger.error(
                f"[OBJC_SPACE_DETECTOR] Error checking if space {space_id} exists: {e}"
            )
            return False

    def get_windows_for_space(self, space_id: int) -> List[Dict[str, Any]]:
        """Get all windows in a specific space"""
        space_info = self.get_space_info(space_id)
        if space_info:
            return space_info.get("windows", [])
        return []

    def get_space_summary(self, space_id: int) -> Dict[str, Any]:
        """Get a summary of a space"""
        space_info = self.get_space_info(space_id)
        if not space_info:
            return {
                "space_id": space_id,
                "space_name": f"Desktop {space_id}",
                "is_current": False,
                "window_count": 0,
                "applications": [],
                "primary_activity": "Empty",
            }

        # Extract applications from windows
        applications = []
        for window in space_info.get("windows", []):
            app_name = window.get("app_name", "Unknown")
            if app_name not in applications:
                applications.append(app_name)

        return {
            "space_id": space_id,
            "space_name": space_info.get("space_name", f"Desktop {space_id}"),
            "is_current": space_info.get("is_current", False),
            "window_count": space_info.get("window_count", 0),
            "applications": applications,
            "primary_activity": space_info.get("primary_activity", "Empty"),
        }


# Global instance
objc_space_detector = ObjCSpaceDetector()


def get_objc_space_detector() -> ObjCSpaceDetector:
    """Get the global Objective-C space detector instance"""
    return objc_space_detector
