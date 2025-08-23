#!/usr/bin/env python3
"""
macOS System Controller for JARVIS AI Agent
Provides voice-activated control of macOS environment through natural language commands
"""

import os
import subprocess
import json
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import psutil
import asyncio
from enum import Enum
import re

logger = logging.getLogger(__name__)


class CommandCategory(Enum):
    """Categories of system commands"""
    APPLICATION = "application"
    FILE = "file"
    SYSTEM = "system"
    WEB = "web"
    WORKFLOW = "workflow"
    DANGEROUS = "dangerous"


class SafetyLevel(Enum):
    """Safety levels for commands"""
    SAFE = "safe"
    CAUTION = "caution"
    DANGEROUS = "dangerous"
    FORBIDDEN = "forbidden"


class MacOSController:
    """Controls macOS system operations with safety checks"""
    
    def __init__(self):
        self.home_dir = Path.home()
        self.safe_directories = [
            self.home_dir / "Desktop",
            self.home_dir / "Documents",
            self.home_dir / "Downloads",
            self.home_dir / "Pictures",
            self.home_dir / "Music",
            self.home_dir / "Movies"
        ]
        
        # Blocked applications for safety
        self.blocked_apps = {
            "System Preferences", "System Settings", "Activity Monitor",
            "Terminal", "Console", "Disk Utility", "Keychain Access"
        }
        
        # Common application mappings
        self.app_aliases = {
            "chrome": "Google Chrome",
            "firefox": "Firefox",
            "safari": "Safari",
            "whatsapp": "WhatsApp",
            "whatsapp desktop": "WhatsApp",
            "spotify": "Spotify",
            "slack": "Slack",
            "zoom": "zoom.us",
            "vscode": "Visual Studio Code",
            "code": "Visual Studio Code",
            "mail": "Mail",
            "calendar": "Calendar",
            "finder": "Finder",
            "messages": "Messages",
            "notes": "Notes",
            "preview": "Preview",
            "terminal": "Terminal",
            "music": "Music",
            "photos": "Photos",
            "pages": "Pages",
            "numbers": "Numbers",
            "keynote": "Keynote"
        }
        
    def execute_applescript(self, script: str) -> Tuple[bool, str]:
        """Execute AppleScript and return result"""
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return True, result.stdout.strip()
            else:
                return False, result.stderr.strip()
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)
            
    def execute_shell(self, command: str, safe_mode: bool = True) -> Tuple[bool, str]:
        """Execute shell command with safety checks"""
        if safe_mode:
            # Block dangerous commands
            dangerous_patterns = [
                r'rm\s+-rf', r'sudo', r'dd\s+', r'mkfs', r'format',
                r'>\s*/dev/', r'chmod\s+777', r'pkill', r'killall'
            ]
            for pattern in dangerous_patterns:
                if re.search(pattern, command, re.IGNORECASE):
                    return False, f"Blocked dangerous command pattern: {pattern}"
                    
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0, result.stdout or result.stderr
        except Exception as e:
            return False, str(e)
            
    # Application Control Methods
    
    def open_application(self, app_name: str) -> Tuple[bool, str]:
        """Open an application"""
        # Resolve aliases
        app_name = self.app_aliases.get(app_name.lower(), app_name)
        
        # Check if blocked
        if app_name in self.blocked_apps:
            return False, f"Opening {app_name} is blocked for safety"
            
        script = f'tell application "{app_name}" to activate'
        success, message = self.execute_applescript(script)
        
        if success:
            return True, f"Opened {app_name}"
        else:
            # Try alternative method
            success, message = self.execute_shell(f"open -a '{app_name}'")
            if success:
                return True, f"Opened {app_name}"
            return False, f"Failed to open {app_name}: {message}"
            
    def close_application(self, app_name: str) -> Tuple[bool, str]:
        """Close an application gracefully"""
        app_name = self.app_aliases.get(app_name.lower(), app_name)
        
        # First try the standard quit command
        script = f'tell application "{app_name}" to quit'
        success, message = self.execute_applescript(script)
        
        if success:
            return True, f"Closed {app_name}"
        
        # If that fails, try using System Events
        script = f'''
        tell application "System Events"
            if exists process "{app_name}" then
                tell process "{app_name}"
                    set frontmost to true
                    keystroke "q" using command down
                end tell
                return "Closed using keyboard shortcut"
            else
                return "Application not running"
            end if
        end tell
        '''
        success, message = self.execute_applescript(script)
        
        if success:
            return True, f"Closed {app_name}"
        
        # Final attempt: Force quit if necessary
        # But only for non-system apps
        if app_name not in ["Finder", "System Preferences", "System Settings"]:
            success, message = self.execute_shell(f"pkill -x '{app_name}'", safe_mode=False)
            if success:
                return True, f"Force closed {app_name}"
        
        return False, f"Failed to close {app_name}: {message}"
        
    def switch_to_application(self, app_name: str) -> Tuple[bool, str]:
        """Switch to an already open application"""
        app_name = self.app_aliases.get(app_name.lower(), app_name)
        
        script = f'''
        tell application "System Events"
            set frontmost of process "{app_name}" to true
        end tell
        '''
        success, message = self.execute_applescript(script)
        
        if success:
            return True, f"Switched to {app_name}"
        return False, f"Failed to switch to {app_name}: {message}"
        
    def list_open_applications(self) -> List[str]:
        """Get list of currently open applications"""
        script = '''
        tell application "System Events"
            get name of (every process whose background only is false)
        end tell
        '''
        success, output = self.execute_applescript(script)
        
        if success:
            apps = output.split(", ")
            return [app.strip() for app in apps if app.strip()]
        return []
        
    def minimize_all_windows(self) -> Tuple[bool, str]:
        """Minimize all windows"""
        script = '''
        tell application "System Events"
            set visible of every process to false
        end tell
        '''
        return self.execute_applescript(script)
        
    def activate_mission_control(self) -> Tuple[bool, str]:
        """Activate Mission Control"""
        script = '''
        tell application "Mission Control" to launch
        '''
        return self.execute_applescript(script)
        
    # File Operations
    
    def is_safe_path(self, path: Path) -> bool:
        """Check if a path is in a safe directory"""
        path = path.resolve()
        return any(path.is_relative_to(safe_dir) for safe_dir in self.safe_directories)
        
    def open_file(self, file_path: str) -> Tuple[bool, str]:
        """Open a file with its default application"""
        path = Path(file_path).expanduser()
        
        if not path.exists():
            return False, f"File not found: {file_path}"
            
        if not self.is_safe_path(path):
            return False, f"Access to {file_path} is restricted for safety"
            
        success, message = self.execute_shell(f"open '{path}'")
        if success:
            return True, f"Opened {path.name}"
        return False, f"Failed to open file: {message}"
        
    def create_file(self, file_path: str, content: str = "") -> Tuple[bool, str]:
        """Create a new file"""
        path = Path(file_path).expanduser()
        
        if not self.is_safe_path(path):
            return False, f"Cannot create file in restricted directory"
            
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return True, f"Created file: {path.name}"
        except Exception as e:
            return False, f"Failed to create file: {str(e)}"
            
    def delete_file(self, file_path: str, confirm: bool = True) -> Tuple[bool, str]:
        """Delete a file (requires confirmation)"""
        path = Path(file_path).expanduser()
        
        if not path.exists():
            return False, f"File not found: {file_path}"
            
        if not self.is_safe_path(path):
            return False, f"Cannot delete file in restricted directory"
            
        if confirm:
            # In real implementation, this would trigger a confirmation dialog
            return False, "File deletion requires user confirmation"
            
        try:
            path.unlink()
            return True, f"Deleted file: {path.name}"
        except Exception as e:
            return False, f"Failed to delete file: {str(e)}"
            
    def search_files(self, query: str, directory: Optional[str] = None) -> List[str]:
        """Search for files using Spotlight"""
        if directory:
            path = Path(directory).expanduser()
            if not self.is_safe_path(path):
                return []
            search_cmd = f"mdfind -onlyin '{path}' '{query}'"
        else:
            search_cmd = f"mdfind '{query}'"
            
        success, output = self.execute_shell(search_cmd)
        
        if success:
            files = output.strip().split('\n')
            # Filter to only safe paths
            return [f for f in files if f and self.is_safe_path(Path(f))]
        return []
        
    # System Settings Control
    
    def set_volume(self, level: int) -> Tuple[bool, str]:
        """Set system volume (0-100)"""
        level = max(0, min(100, level))
        script = f"set volume output volume {level}"
        success, _ = self.execute_applescript(script)
        
        if success:
            return True, f"Volume set to {level}%"
        return False, "Failed to set volume"
        
    def mute_volume(self, mute: bool = True) -> Tuple[bool, str]:
        """Mute or unmute system volume"""
        script = f"set volume output muted {str(mute).lower()}"
        success, _ = self.execute_applescript(script)
        
        if success:
            state = "muted" if mute else "unmuted"
            return True, f"Volume {state}"
        return False, "Failed to change mute state"
        
    def adjust_brightness(self, level: float) -> Tuple[bool, str]:
        """Adjust screen brightness (0.0-1.0)"""
        # This requires additional setup with brightness control tools
        return False, "Brightness control requires additional setup"
        
    def toggle_wifi(self, enable: bool) -> Tuple[bool, str]:
        """Toggle WiFi on/off"""
        action = "on" if enable else "off"
        success, message = self.execute_shell(f"networksetup -setairportpower airport {action}")
        
        if success:
            return True, f"WiFi turned {action}"
        return False, f"Failed to toggle WiFi: {message}"
        
    def take_screenshot(self, save_path: Optional[str] = None) -> Tuple[bool, str]:
        """Take a screenshot"""
        if save_path:
            path = Path(save_path).expanduser()
            if not self.is_safe_path(path.parent):
                return False, "Cannot save screenshot to restricted directory"
            cmd = f"screencapture '{path}'"
        else:
            # Save to desktop with timestamp
            from datetime import datetime
            filename = f"Screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            path = self.home_dir / "Desktop" / filename
            cmd = f"screencapture '{path}'"
            
        success, message = self.execute_shell(cmd)
        
        if success:
            return True, f"Screenshot saved to {path.name}"
        return False, f"Failed to take screenshot: {message}"
        
    def sleep_display(self) -> Tuple[bool, str]:
        """Put display to sleep"""
        success, message = self.execute_shell("pmset displaysleepnow")
        
        if success:
            return True, "Display sleeping"
        return False, f"Failed to sleep display: {message}"
        
    # Web Integration
    
    def open_url(self, url: str, browser: Optional[str] = None) -> Tuple[bool, str]:
        """Open URL in browser"""
        if browser:
            browser = self.app_aliases.get(browser.lower(), browser)
            cmd = f"open -a '{browser}' '{url}'"
        else:
            cmd = f"open '{url}'"
            
        success, message = self.execute_shell(cmd)
        
        if success:
            return True, f"Opened {url}"
        return False, f"Failed to open URL: {message}"
        
    def web_search(self, query: str, engine: str = "google") -> Tuple[bool, str]:
        """Perform web search"""
        engines = {
            "google": f"https://www.google.com/search?q={query}",
            "bing": f"https://www.bing.com/search?q={query}",
            "duckduckgo": f"https://duckduckgo.com/?q={query}"
        }
        
        url = engines.get(engine.lower(), engines["google"])
        return self.open_url(url)
        
    # Complex Workflows
    
    async def execute_workflow(self, workflow_name: str) -> Tuple[bool, str]:
        """Execute predefined workflows"""
        workflows = {
            "morning_routine": [
                ("open_application", "Mail"),
                ("open_application", "Calendar"),
                ("web_search", "weather today"),
                ("open_url", "https://news.google.com")
            ],
            "development_setup": [
                ("open_application", "Visual Studio Code"),
                ("open_application", "Terminal"),
                ("open_application", "Docker"),
                ("open_url", "http://localhost:3000")
            ],
            "meeting_prep": [
                ("set_volume", 50),
                ("close_application", "Spotify"),
                ("minimize_all_windows", None),
                ("open_application", "zoom.us")
            ]
        }
        
        if workflow_name not in workflows:
            return False, f"Unknown workflow: {workflow_name}"
            
        results = []
        for action, param in workflows[workflow_name]:
            method = getattr(self, action)
            if param is not None:
                success, message = method(param)
            else:
                success, message = method()
            results.append(f"{action}: {message}")
            
            if not success:
                return False, f"Workflow failed at {action}: {message}"
                
            # Small delay between actions
            await asyncio.sleep(0.5)
            
        return True, f"Completed {workflow_name} workflow"
        
    # Utility Methods
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        info = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "battery": None
        }
        
        if hasattr(psutil, "sensors_battery"):
            battery = psutil.sensors_battery()
            if battery:
                info["battery"] = {
                    "percent": battery.percent,
                    "charging": battery.power_plugged
                }
                
        return info
        
    def validate_command(self, command: str, category: CommandCategory) -> SafetyLevel:
        """Validate command safety level"""
        if category == CommandCategory.DANGEROUS:
            return SafetyLevel.DANGEROUS
            
        # Check for dangerous patterns
        dangerous_patterns = [
            r'delete.*system', r'remove.*all', r'format', r'shutdown',
            r'restart', r'sudo', r'admin', r'root'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return SafetyLevel.DANGEROUS
                
        if category in [CommandCategory.APPLICATION, CommandCategory.WEB]:
            return SafetyLevel.SAFE
            
        return SafetyLevel.CAUTION
    
    def find_installed_application(self, partial_name: str) -> Optional[str]:
        """Find installed application by partial name"""
        # Common application directories
        app_dirs = [
            "/Applications",
            "~/Applications",
            "/System/Applications"
        ]
        
        partial_lower = partial_name.lower()
        
        for app_dir in app_dirs:
            expanded_dir = os.path.expanduser(app_dir)
            if not os.path.exists(expanded_dir):
                continue
                
            for app in os.listdir(expanded_dir):
                if app.endswith('.app'):
                    app_name = app[:-4]  # Remove .app extension
                    if partial_lower in app_name.lower():
                        return app_name
        
        return None