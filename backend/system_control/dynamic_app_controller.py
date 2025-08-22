#!/usr/bin/env python3
"""
Dynamic App Controller for JARVIS
Uses vision system to detect and control any application without hardcoding
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
import subprocess
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class DynamicAppController:
    """Dynamically detects and controls applications using vision and system APIs"""
    
    def __init__(self):
        self.detected_apps_cache = {}
        self.last_vision_scan = None
        
    def get_all_running_apps(self) -> List[Dict[str, str]]:
        """Get all running applications with detailed info"""
        try:
            # Use AppleScript to get comprehensive app info
            script = '''
            tell application "System Events"
                set appList to {}
                repeat with proc in (every process)
                    try
                        set procName to name of proc
                        set procPath to (POSIX path of (file of proc))
                        set procVisible to visible of proc
                        set procPID to unix id of proc
                        set end of appList to procName & "|" & procPath & "|" & procVisible & "|" & procPID
                    end try
                end repeat
                return appList
            end tell
            '''
            
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                apps = []
                for line in result.stdout.strip().split(", "):
                    if "|" in line:
                        parts = line.split("|")
                        if len(parts) >= 4:
                            apps.append({
                                "name": parts[0].strip(),
                                "path": parts[1].strip(),
                                "visible": parts[2].strip() == "true",
                                "pid": parts[3].strip()
                            })
                return apps
            
        except Exception as e:
            logger.error(f"Error getting running apps: {e}")
            
        return []
    
    def find_app_by_fuzzy_name(self, search_name: str) -> Optional[Dict[str, str]]:
        """Find app using fuzzy matching"""
        search_lower = search_name.lower().strip()
        running_apps = self.get_all_running_apps()
        
        # First try exact match
        for app in running_apps:
            if app["name"].lower() == search_lower:
                return app
        
        # Try without spaces
        search_no_space = search_lower.replace(" ", "")
        for app in running_apps:
            if app["name"].lower().replace(" ", "") == search_no_space:
                return app
        
        # Try partial match
        for app in running_apps:
            if search_lower in app["name"].lower() or search_no_space in app["name"].lower().replace(" ", ""):
                return app
        
        # Try common variations
        variations = [
            search_name,
            search_name.capitalize(),
            search_name.title(),
            search_name.upper(),
            search_name.replace(" ", ""),
            search_name.replace("-", " "),
            search_name.replace("_", " ")
        ]
        
        for variation in variations:
            for app in running_apps:
                if variation in app["name"]:
                    return app
        
        return None
    
    def close_app_by_exact_name(self, app_name: str) -> Tuple[bool, str]:
        """Close app using exact process name"""
        try:
            # Try graceful quit first
            script = f'''
            tell application "System Events"
                set appProcs to every process whose name is "{app_name}"
                repeat with appProc in appProcs
                    tell appProc
                        if exists then
                            try
                                quit
                                return "Closed {app_name}"
                            on error
                                -- Try clicking quit menu
                                try
                                    set frontmost to true
                                    tell application "System Events"
                                        keystroke "q" using command down
                                    end tell
                                    return "Closed {app_name} using keyboard shortcut"
                                on error
                                    return "Failed to close {app_name}"
                                end try
                            end try
                        end if
                    end tell
                end repeat
                return "Application {app_name} not found"
            end tell
            '''
            
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                return True, result.stdout.strip()
            
            # If AppleScript fails, try pkill as last resort
            pkill_result = subprocess.run(
                ["pkill", "-x", app_name],
                capture_output=True,
                text=True
            )
            
            if pkill_result.returncode == 0:
                return True, f"Force closed {app_name}"
            
            return False, f"Failed to close {app_name}"
            
        except Exception as e:
            logger.error(f"Error closing app {app_name}: {e}")
            return False, str(e)
    
    def open_app_by_name(self, app_name: str) -> Tuple[bool, str]:
        """Open app by name, searching installed applications"""
        try:
            # First try direct open
            result = subprocess.run(
                ["open", "-a", app_name],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                return True, f"Opened {app_name}"
            
            # Search in Applications folders
            app_paths = [
                f"/Applications/{app_name}.app",
                f"~/Applications/{app_name}.app",
                f"/System/Applications/{app_name}.app"
            ]
            
            for path in app_paths:
                expanded_path = os.path.expanduser(path)
                if os.path.exists(expanded_path):
                    subprocess.run(["open", expanded_path])
                    return True, f"Opened {app_name}"
            
            # Try to find by partial match
            for app_dir in ["/Applications", os.path.expanduser("~/Applications"), "/System/Applications"]:
                if os.path.exists(app_dir):
                    for app in os.listdir(app_dir):
                        if app.endswith('.app') and app_name.lower() in app.lower():
                            subprocess.run(["open", os.path.join(app_dir, app)])
                            return True, f"Opened {app[:-4]}"
            
            return False, f"Could not find application: {app_name}"
            
        except Exception as e:
            logger.error(f"Error opening app {app_name}: {e}")
            return False, str(e)
    
    async def close_app_intelligently(self, search_name: str) -> Tuple[bool, str]:
        """Intelligently close app using fuzzy matching"""
        # Find the app
        app_info = self.find_app_by_fuzzy_name(search_name)
        
        if not app_info:
            return False, f"Could not find running application matching '{search_name}'"
        
        # Close using exact name
        success, message = self.close_app_by_exact_name(app_info["name"])
        
        if success:
            return True, f"{app_info['name']} has been closed, Sir."
        else:
            return False, f"Failed to close {app_info['name']}: {message}"
    
    async def open_app_intelligently(self, search_name: str) -> Tuple[bool, str]:
        """Intelligently open app using fuzzy matching"""
        # First check if already running
        app_info = self.find_app_by_fuzzy_name(search_name)
        
        if app_info and app_info["visible"]:
            # App is already running, just bring to front
            script = f'tell application "{app_info["name"]}" to activate'
            subprocess.run(["osascript", "-e", script])
            return True, f"{app_info['name']} is already running and now active"
        
        # Try to open the app
        success, message = self.open_app_by_name(search_name)
        return success, message
    
    def update_from_vision(self, vision_data: Dict[str, Any]):
        """Update detected apps from vision system"""
        if "applications" in vision_data:
            self.detected_apps_cache = {
                app: True for app in vision_data["applications"]
            }
            self.last_vision_scan = vision_data.get("timestamp")
    
    def get_app_suggestions(self, partial_name: str) -> List[str]:
        """Get app name suggestions"""
        suggestions = []
        running_apps = self.get_all_running_apps()
        
        partial_lower = partial_name.lower()
        
        for app in running_apps:
            if partial_lower in app["name"].lower():
                suggestions.append(app["name"])
        
        return suggestions[:5]  # Return top 5 suggestions


# Singleton instance
_dynamic_controller = None

def get_dynamic_app_controller() -> DynamicAppController:
    """Get or create the dynamic app controller instance"""
    global _dynamic_controller
    if _dynamic_controller is None:
        _dynamic_controller = DynamicAppController()
    return _dynamic_controller