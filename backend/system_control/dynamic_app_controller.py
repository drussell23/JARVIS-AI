#!/usr/bin/env python3
"""
Dynamic App Controller for JARVIS
Dynamically discovers and controls all macOS applications without hardcoding
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
import subprocess
import re
from pathlib import Path
import json
import plistlib
from difflib import get_close_matches

logger = logging.getLogger(__name__)

class DynamicAppController:
    """Dynamically detects and controls applications using system APIs"""
    
    def __init__(self):
        self.detected_apps_cache = {}
        self.installed_apps_cache = {}
        self.app_bundle_ids = {}
        self.last_vision_scan = None
        self.last_scan_time = None
        self._scan_installed_applications()
    
    def _scan_installed_applications(self):
        """Scan system for all installed applications"""
        logger.info("Scanning for installed applications...")
        self.installed_apps_cache = {}
        self.app_bundle_ids = {}
        
        # Directories to scan for applications
        app_directories = [
            "/Applications",
            "/System/Applications",
            "/System/Applications/Utilities",
            os.path.expanduser("~/Applications"),
            "/Applications/Utilities",
            # Common third-party app locations
            "/Applications/Setapp",
            os.path.expanduser("~/Applications/Setapp"),
        ]
        
        for directory in app_directories:
            if os.path.exists(directory):
                self._scan_directory(directory)
        
        # Also get apps using system profiler for complete list
        self._scan_with_system_profiler()
        
        logger.info(f"Found {len(self.installed_apps_cache)} installed applications")
    
    def _scan_directory(self, directory: str):
        """Scan a directory for .app bundles"""
        try:
            for item in os.listdir(directory):
                if item.endswith('.app'):
                    app_path = os.path.join(directory, item)
                    app_name = item[:-4]  # Remove .app extension
                    
                    # Get bundle info if available
                    info_plist = os.path.join(app_path, "Contents", "Info.plist")
                    if os.path.exists(info_plist):
                        try:
                            with open(info_plist, 'rb') as f:
                                plist = plistlib.load(f)
                                bundle_id = plist.get('CFBundleIdentifier', '')
                                display_name = plist.get('CFBundleDisplayName', app_name)
                                
                                self.app_bundle_ids[app_name.lower()] = bundle_id
                                # Store with various name formats
                                self.installed_apps_cache[display_name.lower()] = {
                                    'name': display_name,
                                    'path': app_path,
                                    'bundle_id': bundle_id,
                                    'actual_name': app_name
                                }
                        except Exception as e:
                            logger.debug(f"Could not read plist for {app_name}: {e}")
                    
                    # Store basic info
                    self.installed_apps_cache[app_name.lower()] = {
                        'name': app_name,
                        'path': app_path,
                        'bundle_id': '',
                        'actual_name': app_name
                    }
                    
                    # Also store without spaces
                    name_no_space = app_name.replace(" ", "").lower()
                    self.installed_apps_cache[name_no_space] = self.installed_apps_cache[app_name.lower()]
                    
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
    
    def _scan_with_system_profiler(self):
        """Use system_profiler to get comprehensive app list"""
        try:
            # This command gets all applications known to the system
            result = subprocess.run(
                ["system_profiler", "SPApplicationsDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                apps = data.get('SPApplicationsDataType', [])
                
                for app in apps:
                    app_name = app.get('_name', '')
                    if app_name:
                        self.installed_apps_cache[app_name.lower()] = {
                            'name': app_name,
                            'path': app.get('path', ''),
                            'bundle_id': app.get('bundle_identifier', ''),
                            'version': app.get('version', ''),
                            'actual_name': app_name
                        }
                        
                        # Store without spaces too
                        name_no_space = app_name.replace(" ", "").lower()
                        self.installed_apps_cache[name_no_space] = self.installed_apps_cache[app_name.lower()]
                        
        except Exception as e:
            logger.debug(f"Could not use system_profiler: {e}")
        
    def is_app_running(self, app_name: str) -> bool:
        """Check if an app is currently running"""
        running_apps = self.get_all_running_apps()
        app_name_lower = app_name.lower()
        
        for app in running_apps:
            if app_name_lower in app['name'].lower() or app['name'].lower() in app_name_lower:
                return True
        
        # Also check using the find_app_by_name to handle variations
        app_info = self.find_app_by_name(app_name)
        if app_info:
            actual_name = app_info.get('actual_name', app_info.get('name', ''))
            for app in running_apps:
                if actual_name.lower() == app['name'].lower():
                    return True
        
        return False
    
    def get_all_running_apps(self) -> List[Dict[str, str]]:
        """Get all running applications with detailed info"""
        try:
            # Use a simpler, faster AppleScript
            script = '''
            tell application "System Events"
                get name of (every process whose background only is false)
            end tell
            '''
            
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=3
            )
            
            if result.returncode == 0:
                apps = []
                app_names = result.stdout.strip().split(", ")
                for app_name in app_names:
                    if app_name:
                        apps.append({
                            "name": app_name.strip(),
                            "path": "",
                            "visible": True,
                            "pid": ""
                        })
                return apps
            
        except Exception as e:
            logger.error(f"Error getting running apps: {e}")
            
        return []
    
    def find_app_by_name(self, search_name: str) -> Optional[Dict[str, str]]:
        """Find app using intelligent matching"""
        search_lower = search_name.lower().strip()
        
        # Direct match in installed apps
        if search_lower in self.installed_apps_cache:
            return self.installed_apps_cache[search_lower]
        
        # Try without spaces
        search_no_space = search_lower.replace(" ", "")
        if search_no_space in self.installed_apps_cache:
            return self.installed_apps_cache[search_no_space]
        
        # Handle special cases and common variations first
        variations = self._generate_name_variations(search_name)
        for variation in variations:
            if variation.lower() in self.installed_apps_cache:
                return self.installed_apps_cache[variation.lower()]
        
        # Try exact word boundary matching to avoid "spotify" matching "spotlight"
        for app_key, app_info in self.installed_apps_cache.items():
            # Check if search term is a complete word in the app name
            app_words = app_key.split()
            search_words = search_lower.split()
            
            # Check if all search words are in app words
            if all(word in app_words for word in search_words):
                return app_info
            
            # Check if search is exact match for start of app name
            if app_key.startswith(search_lower + " ") or app_key == search_lower:
                return app_info
        
        # Try fuzzy matching
        all_names = list(self.installed_apps_cache.keys())
        matches = get_close_matches(search_lower, all_names, n=3, cutoff=0.7)
        
        if matches:
            # Prefer matches that start with the search term
            for match in matches:
                if match.startswith(search_lower):
                    return self.installed_apps_cache[match]
            return self.installed_apps_cache[matches[0]]
        
        # Try partial matching as last resort
        for app_key, app_info in self.installed_apps_cache.items():
            if search_lower in app_key or search_no_space in app_key:
                return app_info
        
        return None
    
    def find_app_by_fuzzy_name(self, search_name: str) -> Optional[Dict[str, str]]:
        """Legacy method - redirects to find_app_by_name"""
        return self.find_app_by_name(search_name)
    
    def _generate_name_variations(self, name: str) -> List[str]:
        """Generate common name variations"""
        variations = [name]
        
        # Handle common patterns
        if name.lower() == "code":
            variations.extend(["Visual Studio Code", "VSCode"])
        elif name.lower() in ["discord", "discord app"]:
            variations.append("Discord")
        elif name.lower() == "whatsapp":
            variations.extend(["WhatsApp", "WhatsApp Desktop"])
        elif name.lower() == "chrome":
            variations.append("Google Chrome")
        elif name.lower() == "zoom":
            variations.extend(["zoom.us", "Zoom"])
        elif name.lower() == "spotify":
            variations.append("Spotify")
        
        # Add variations with/without spaces
        if " " in name:
            variations.append(name.replace(" ", ""))
        else:
            # Try to add spaces at capital letters
            spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
            variations.append(spaced)
        
        return variations
    
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
                timeout=2
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
        """Open app by name using enhanced app discovery"""
        app_info = self.find_app_by_name(app_name)
        
        if not app_info:
            # Try refreshing cache once
            self._scan_installed_applications()
            app_info = self.find_app_by_name(app_name)
            
            if not app_info:
                suggestions = self.get_app_suggestions(app_name)
                if suggestions:
                    return False, f"Could not find '{app_name}'. Did you mean: {', '.join(suggestions[:3])}?"
                return False, f"Could not find application: {app_name}"
        
        try:
            # Check if already running
            script = f'''
            tell application "System Events"
                set appList to name of every application process
                return appList contains "{app_info['actual_name']}"
            end tell
            '''
            
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True
            )
            
            is_running = result.stdout.strip() == "true"
            
            # Open or activate the app
            if app_info.get('path'):
                subprocess.run(["open", app_info['path']], check=True)
            else:
                subprocess.run(["open", "-a", app_info['actual_name']], check=True)
            
            if is_running:
                return True, f"Switching to {app_info['name']}"
            else:
                return True, f"Opening {app_info['name']}"
                
        except Exception as e:
            logger.error(f"Error opening {app_name}: {e}")
            return False, f"Failed to open {app_info['name']}: {str(e)}"
    
    async def close_app_intelligently(self, search_name: str) -> Tuple[bool, str]:
        """Intelligently close app using enhanced discovery"""
        # First check if the app is running
        if not self.is_app_running(search_name):
            app_info = self.find_app_by_name(search_name)
            app_display_name = app_info['name'] if app_info else search_name
            return True, f"{app_display_name} is already closed"
        
        # Find the app in installed apps
        app_info = self.find_app_by_name(search_name)
        
        if not app_info:
            # Check running apps
            running_apps = self.get_all_running_apps()
            for running_app in running_apps:
                if search_name.lower() in running_app['name'].lower():
                    app_info = {'name': running_app['name'], 'actual_name': running_app['name']}
                    break
            
            if not app_info:
                return False, f"Could not find application: {search_name}"
        
        try:
            # Try graceful quit
            script = f'''
            tell application "{app_info['actual_name']}" to quit
            '''
            
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                return True, f"Closed {app_info['name']}"
            
            # Try with System Events
            script = f'''
            tell application "System Events"
                if exists process "{app_info['actual_name']}" then
                    tell process "{app_info['actual_name']}"
                        set frontmost to true
                        keystroke "q" using command down
                    end tell
                    return "Closed using keyboard shortcut"
                else
                    return "Application not running"
                end if
            end tell
            '''
            
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True
            )
            
            if "not running" in result.stdout:
                return True, f"{app_info['name']} was not running"
            
            return True, f"Closed {app_info['name']}"
            
        except Exception as e:
            logger.error(f"Error closing {search_name}: {e}")
            return False, f"Failed to close {app_info['name']}: {str(e)}"
    
    async def open_app_intelligently(self, search_name: str) -> Tuple[bool, str]:
        """Intelligently open app using enhanced discovery"""
        # First check if the app is already running
        if self.is_app_running(search_name):
            app_info = self.find_app_by_name(search_name)
            app_display_name = app_info['name'] if app_info else search_name
            return True, f"{app_display_name} is already open"
        
        # If not running, open it
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
        """Get app name suggestions for partial matches"""
        partial_lower = partial_name.lower()
        suggestions = []
        
        # Get all app names that contain the partial string
        for key, app_info in self.installed_apps_cache.items():
            if partial_lower in key or partial_lower in app_info['name'].lower():
                if app_info['name'] not in suggestions:
                    suggestions.append(app_info['name'])
        
        # Sort by similarity
        return sorted(suggestions, key=lambda x: len(x))[:5]
    
    def refresh_app_list(self):
        """Manually refresh the installed apps list"""
        self._scan_installed_applications()
        return len(self.installed_apps_cache)

# Singleton instance
_dynamic_controller = None

def get_dynamic_app_controller() -> DynamicAppController:
    """Get or create the dynamic app controller instance"""
    global _dynamic_controller
    if _dynamic_controller is None:
        _dynamic_controller = DynamicAppController()
    return _dynamic_controller