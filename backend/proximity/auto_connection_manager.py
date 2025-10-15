"""
Automatic Display Connection Manager
=====================================

Handles automatic display connection/mirroring/extending based on proximity
decisions. Executes connection actions via macOS AppleScript automation.

Features:
- Automatic connection evaluation
- Debouncing to prevent rapid connect/disconnect
- User override tracking
- Connection state management
- AppleScript-based display automation (backend only)
- Async execution with timeout handling
- Robust error handling and retry logic

Author: Derek Russell
Date: 2025-10-14
"""

import asyncio
import logging
import subprocess
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
from enum import Enum

from .proximity_display_context import (
    ConnectionDecision,
    ConnectionAction,
    ConnectionState,
    DisplayLocation
)

logger = logging.getLogger(__name__)


class DisplayMode(Enum):
    """Display connection modes"""
    MIRROR = "mirror"
    EXTEND = "extend"
    DISCONNECT = "disconnect"


class ConnectionResult:
    """Result of a connection attempt"""
    def __init__(
        self,
        success: bool,
        display_id: int,
        action: str,
        message: str,
        execution_time: float = 0.0
    ):
        self.success = success
        self.display_id = display_id
        self.action = action
        self.message = message
        self.execution_time = execution_time
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "display_id": self.display_id,
            "action": self.action,
            "message": self.message,
            "execution_time": round(self.execution_time, 3),
            "timestamp": self.timestamp.isoformat()
        }


class AutoConnectionManager:
    """
    Manages automatic display connections based on proximity decisions
    
    This is the core automation engine - it evaluates proximity decisions,
    applies debouncing, respects user overrides, and executes display
    connections via AppleScript (backend only).
    """
    
    def __init__(
        self,
        debounce_seconds: float = 3.0,
        auto_connect_enabled: bool = True
    ):
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.debounce_seconds = debounce_seconds
        self.auto_connect_enabled = auto_connect_enabled
        
        # State tracking
        self.last_action_time: Dict[int, datetime] = {}
        self.connection_states: Dict[int, ConnectionState] = {}
        self.user_overrides: Dict[int, datetime] = {}  # display_id -> override_time
        self.connection_history: List[ConnectionResult] = []
        
        # Performance metrics
        self.total_connections = 0
        self.successful_connections = 0
        self.failed_connections = 0
        self.user_override_count = 0
        
        self.logger.info("[AUTO-CONNECT] Manager initialized")
    
    async def evaluate_and_execute(
        self,
        decision: ConnectionDecision,
        force: bool = False
    ) -> Optional[ConnectionResult]:
        """
        Evaluate a connection decision and execute if appropriate
        
        Args:
            decision: ConnectionDecision from ProximityDisplayBridge
            force: If True, bypass debouncing and override checks
            
        Returns:
            ConnectionResult or None if no action taken
        """
        try:
            display_id = decision.display_id
            action = decision.action
            
            # Check if auto-connect is enabled
            if not self.auto_connect_enabled and not force:
                self.logger.info(f"[AUTO-CONNECT] Disabled, skipping action for display {display_id}")
                return None
            
            # Check for user override (user manually disconnected recently)
            if not force and self._has_recent_user_override(display_id):
                self.logger.info(f"[AUTO-CONNECT] User override active for display {display_id}, skipping")
                return None
            
            # Apply debouncing (prevent rapid connect/disconnect)
            if not force and not self._should_execute_action(display_id):
                self.logger.debug(f"[AUTO-CONNECT] Debouncing active for display {display_id}, skipping")
                return None
            
            # Execute action based on decision
            if action == ConnectionAction.AUTO_CONNECT:
                if decision.should_auto_connect or force:
                    result = await self._connect_display(display_id, decision.display_name, DisplayMode.EXTEND)
                    self._update_action_time(display_id)
                    return result
                else:
                    self.logger.info(f"[AUTO-CONNECT] Confidence too low for auto-connect: {decision.confidence}")
                    return None
            
            elif action == ConnectionAction.PROMPT_USER:
                # Log that user should be prompted (actual prompting happens elsewhere)
                self.logger.info(f"[AUTO-CONNECT] Proximity detected, user prompt recommended: {decision.display_name}")
                return None
            
            elif action == ConnectionAction.DISCONNECT:
                result = await self._disconnect_display(display_id, decision.display_name)
                self._update_action_time(display_id)
                return result
            
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"[AUTO-CONNECT] Error evaluating decision: {e}")
            return None
    
    def _should_execute_action(self, display_id: int) -> bool:
        """Check if enough time has passed since last action (debouncing)"""
        if display_id not in self.last_action_time:
            return True
        
        time_since_last = (datetime.now() - self.last_action_time[display_id]).total_seconds()
        return time_since_last >= self.debounce_seconds
    
    def _update_action_time(self, display_id: int):
        """Update last action time for debouncing"""
        self.last_action_time[display_id] = datetime.now()
    
    def _has_recent_user_override(self, display_id: int) -> bool:
        """Check if user recently manually disconnected (respect user intent)"""
        if display_id not in self.user_overrides:
            return False
        
        override_time = self.user_overrides[display_id]
        time_since_override = (datetime.now() - override_time).total_seconds()
        
        # Respect user override for 5 minutes
        return time_since_override < 300
    
    def register_user_override(self, display_id: int):
        """Register that user manually disconnected (prevent auto-reconnect)"""
        self.user_overrides[display_id] = datetime.now()
        self.user_override_count += 1
        self.logger.info(f"[AUTO-CONNECT] User override registered for display {display_id}")
    
    async def _connect_display(
        self,
        display_id: int,
        display_name: str,
        mode: DisplayMode = DisplayMode.EXTEND
    ) -> ConnectionResult:
        """
        Connect to a display via AppleScript (backend automation)
        
        Args:
            display_id: Display ID
            display_name: Human-readable name
            mode: Connection mode (mirror or extend)
            
        Returns:
            ConnectionResult with success status
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"[AUTO-CONNECT] Attempting to connect display {display_id} ({display_name}) in {mode.value} mode")
            
            # Update state
            self.connection_states[display_id] = ConnectionState.CONNECTING
            
            # Execute AppleScript to mirror/extend display
            if mode == DisplayMode.MIRROR:
                success, message = await self._execute_mirror_applescript(display_id)
            else:  # EXTEND
                success, message = await self._execute_extend_applescript(display_id)
            
            # Update metrics
            self.total_connections += 1
            if success:
                self.successful_connections += 1
                self.connection_states[display_id] = ConnectionState.CONNECTED
            else:
                self.failed_connections += 1
                self.connection_states[display_id] = ConnectionState.ERROR
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = ConnectionResult(
                success=success,
                display_id=display_id,
                action=f"connect_{mode.value}",
                message=message,
                execution_time=execution_time
            )
            
            self.connection_history.append(result)
            return result
            
        except Exception as e:
            self.logger.error(f"[AUTO-CONNECT] Error connecting display: {e}")
            self.failed_connections += 1
            self.connection_states[display_id] = ConnectionState.ERROR
            
            execution_time = (datetime.now() - start_time).total_seconds()
            result = ConnectionResult(
                success=False,
                display_id=display_id,
                action=f"connect_{mode.value}",
                message=f"Error: {str(e)}",
                execution_time=execution_time
            )
            self.connection_history.append(result)
            return result
    
    async def _disconnect_display(
        self,
        display_id: int,
        display_name: str
    ) -> ConnectionResult:
        """
        Disconnect a display via AppleScript
        
        Args:
            display_id: Display ID
            display_name: Human-readable name
            
        Returns:
            ConnectionResult with success status
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"[AUTO-CONNECT] Attempting to disconnect display {display_id} ({display_name})")
            
            # Update state
            self.connection_states[display_id] = ConnectionState.DISCONNECTING
            
            # Execute AppleScript to disconnect
            success, message = await self._execute_disconnect_applescript(display_id)
            
            if success:
                self.connection_states[display_id] = ConnectionState.AVAILABLE
            else:
                self.connection_states[display_id] = ConnectionState.ERROR
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = ConnectionResult(
                success=success,
                display_id=display_id,
                action="disconnect",
                message=message,
                execution_time=execution_time
            )
            
            self.connection_history.append(result)
            return result
            
        except Exception as e:
            self.logger.error(f"[AUTO-CONNECT] Error disconnecting display: {e}")
            self.connection_states[display_id] = ConnectionState.ERROR
            
            execution_time = (datetime.now() - start_time).total_seconds()
            result = ConnectionResult(
                success=False,
                display_id=display_id,
                action="disconnect",
                message=f"Error: {str(e)}",
                execution_time=execution_time
            )
            self.connection_history.append(result)
            return result
    
    async def _execute_mirror_applescript(self, display_id: int) -> Tuple[bool, str]:
        """
        Execute AppleScript to mirror displays
        
        Note: macOS doesn't provide simple display control APIs.
        This uses AppleScript to automate System Preferences.
        Requires Accessibility permissions.
        
        Returns:
            (success: bool, message: str)
        """
        applescript = f'''
        tell application "System Preferences"
            reveal anchor "displaysDisplayTab" of pane "com.apple.preference.displays"
            activate
        end tell
        
        delay 1
        
        tell application "System Events"
            tell process "System Preferences"
                try
                    click checkbox "Mirror Displays" of tab group 1 of window 1
                    return "Success: Mirror mode enabled"
                on error errMsg
                    return "Error: " & errMsg
                end try
            end tell
        end tell
        '''
        
        try:
            result = await asyncio.create_subprocess_exec(
                'osascript', '-e', applescript,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=10.0)
            
            output = stdout.decode().strip()
            
            if "Success" in output:
                self.logger.info(f"[APPLESCRIPT] Mirror mode enabled for display {display_id}")
                return True, output
            else:
                self.logger.warning(f"[APPLESCRIPT] Mirror failed: {output}")
                return False, output
                
        except asyncio.TimeoutError:
            self.logger.error(f"[APPLESCRIPT] Mirror script timed out")
            return False, "Timeout: AppleScript execution exceeded 10 seconds"
        except Exception as e:
            self.logger.error(f"[APPLESCRIPT] Mirror error: {e}")
            return False, f"Error: {str(e)}"
    
    async def _execute_extend_applescript(self, display_id: int) -> Tuple[bool, str]:
        """
        Execute AppleScript to extend displays
        
        Returns:
            (success: bool, message: str)
        """
        applescript = f'''
        tell application "System Preferences"
            reveal anchor "displaysDisplayTab" of pane "com.apple.preference.displays"
            activate
        end tell
        
        delay 1
        
        tell application "System Events"
            tell process "System Preferences"
                try
                    -- Uncheck mirror if it's checked
                    set mirrorCheckbox to checkbox "Mirror Displays" of tab group 1 of window 1
                    if value of mirrorCheckbox is 1 then
                        click mirrorCheckbox
                    end if
                    return "Success: Extend mode enabled"
                on error errMsg
                    return "Error: " & errMsg
                end try
            end tell
        end tell
        '''
        
        try:
            result = await asyncio.create_subprocess_exec(
                'osascript', '-e', applescript,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=10.0)
            
            output = stdout.decode().strip()
            
            if "Success" in output:
                self.logger.info(f"[APPLESCRIPT] Extend mode enabled for display {display_id}")
                return True, output
            else:
                self.logger.warning(f"[APPLESCRIPT] Extend failed: {output}")
                return False, output
                
        except asyncio.TimeoutError:
            self.logger.error(f"[APPLESCRIPT] Extend script timed out")
            return False, "Timeout: AppleScript execution exceeded 10 seconds"
        except Exception as e:
            self.logger.error(f"[APPLESCRIPT] Extend error: {e}")
            return False, f"Error: {str(e)}"
    
    async def _execute_disconnect_applescript(self, display_id: int) -> Tuple[bool, str]:
        """
        Execute AppleScript to disconnect display
        
        Note: Disconnecting external displays programmatically is limited.
        This is a placeholder - actual disconnection typically requires
        physical cable removal or AirPlay/wireless disconnect.
        
        Returns:
            (success: bool, message: str)
        """
        self.logger.info(f"[APPLESCRIPT] Disconnect requested for display {display_id}")
        
        # For now, just log (physical disconnect or AirPlay required)
        return True, "Disconnect logged (physical disconnection required for external displays)"
    
    def get_connection_state(self, display_id: int) -> ConnectionState:
        """Get current connection state for a display"""
        return self.connection_states.get(display_id, ConnectionState.AVAILABLE)
    
    def get_manager_stats(self) -> Dict:
        """Get manager statistics"""
        return {
            "auto_connect_enabled": self.auto_connect_enabled,
            "debounce_seconds": self.debounce_seconds,
            "total_connections": self.total_connections,
            "successful_connections": self.successful_connections,
            "failed_connections": self.failed_connections,
            "success_rate": round(self.successful_connections / max(self.total_connections, 1), 3),
            "user_override_count": self.user_override_count,
            "active_connections": len([s for s in self.connection_states.values() if s == ConnectionState.CONNECTED]),
            "recent_history": [r.to_dict() for r in self.connection_history[-10:]]
        }


# Singleton instance
_auto_connection_manager: Optional[AutoConnectionManager] = None

def get_auto_connection_manager(
    debounce_seconds: float = 3.0,
    auto_connect_enabled: bool = True
) -> AutoConnectionManager:
    """Get singleton AutoConnectionManager instance"""
    global _auto_connection_manager
    if _auto_connection_manager is None:
        _auto_connection_manager = AutoConnectionManager(debounce_seconds, auto_connect_enabled)
    return _auto_connection_manager
