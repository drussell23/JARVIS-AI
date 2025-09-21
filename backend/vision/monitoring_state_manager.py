"""
Monitoring State Manager - Centralized monitoring state management
Part of Screen Monitoring Activation & macOS Purple Indicator System
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime
from enum import Enum
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class MonitoringState(Enum):
    """Monitoring system states"""
    INACTIVE = "inactive"
    ACTIVATING = "activating"
    ACTIVE = "active"
    DEACTIVATING = "deactivating"
    ERROR = "error"


class MonitoringCapability(Enum):
    """Available monitoring capabilities"""
    BASIC_SCREEN = "basic_screen"
    MULTI_SPACE = "multi_space"
    PROACTIVE = "proactive"
    ACTIVITY_REPORTING = "activity_reporting"
    MACOS_INDICATOR = "macos_indicator"


class MonitoringStateManager:
    """
    Centralized management of monitoring state across all components
    Ensures consistency between vision system, macOS indicator, and UI
    """
    
    def __init__(self):
        self.current_state = MonitoringState.INACTIVE
        self.active_capabilities: List[MonitoringCapability] = []
        self.state_history = []
        self.state_callbacks = []
        self.last_state_change = None
        self.activation_start_time = None
        self.error_details = None
        self.persistence_file = Path.home() / ".jarvis" / "monitoring_state.json"
        
        # Component status tracking
        self.component_status = {
            'vision_intelligence': False,
            'macos_indicator': False,
            'multi_space': False,
            'proactive_monitor': False,
            'websocket_connected': False
        }
        
        # Load persisted state
        self._load_state()
    
    def add_state_callback(self, callback: Callable[[MonitoringState, Dict[str, Any]], None]):
        """Add a callback to be notified of state changes"""
        self.state_callbacks.append(callback)
    
    async def transition_to(self, new_state: MonitoringState, details: Optional[Dict[str, Any]] = None):
        """
        Transition to a new monitoring state
        
        Args:
            new_state: The target state
            details: Additional details about the transition
        """
        old_state = self.current_state
        
        # Validate transition
        if not self._is_valid_transition(old_state, new_state):
            logger.warning(f"Invalid state transition: {old_state} -> {new_state}")
            return
        
        # Update state
        self.current_state = new_state
        self.last_state_change = datetime.now()
        
        # Track activation timing
        if new_state == MonitoringState.ACTIVATING:
            self.activation_start_time = datetime.now()
        elif new_state == MonitoringState.ACTIVE and self.activation_start_time:
            activation_time = (datetime.now() - self.activation_start_time).total_seconds()
            logger.info(f"[STATE] Monitoring activation took {activation_time:.2f} seconds")
        
        # Handle error state
        if new_state == MonitoringState.ERROR:
            self.error_details = details
        else:
            self.error_details = None
        
        # Record in history
        self.state_history.append({
            'from': old_state.value,
            'to': new_state.value,
            'timestamp': self.last_state_change.isoformat(),
            'details': details
        })
        
        # Keep history size reasonable
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-50:]
        
        # Persist state
        self._save_state()
        
        # Notify callbacks
        await self._notify_state_change(new_state, {
            'previous_state': old_state,
            'details': details,
            'timestamp': self.last_state_change
        })
        
        logger.info(f"[STATE] Transitioned from {old_state.value} to {new_state.value}")
    
    def update_component_status(self, component: str, active: bool):
        """Update the status of a monitoring component"""
        if component in self.component_status:
            self.component_status[component] = active
            logger.debug(f"[STATE] Component {component} is now {'active' if active else 'inactive'}")
            
            # Check if all critical components are active
            if self.current_state == MonitoringState.ACTIVATING:
                if self._are_critical_components_active():
                    asyncio.create_task(self.transition_to(MonitoringState.ACTIVE))
    
    def add_capability(self, capability: MonitoringCapability):
        """Add an active monitoring capability"""
        if capability not in self.active_capabilities:
            self.active_capabilities.append(capability)
            logger.info(f"[STATE] Added capability: {capability.value}")
    
    def remove_capability(self, capability: MonitoringCapability):
        """Remove a monitoring capability"""
        if capability in self.active_capabilities:
            self.active_capabilities.remove(capability)
            logger.info(f"[STATE] Removed capability: {capability.value}")
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get comprehensive monitoring state information"""
        info = {
            'current_state': self.current_state.value,
            'is_active': self.current_state == MonitoringState.ACTIVE,
            'is_transitioning': self.current_state in [MonitoringState.ACTIVATING, MonitoringState.DEACTIVATING],
            'active_capabilities': [cap.value for cap in self.active_capabilities],
            'component_status': self.component_status.copy(),
            'last_state_change': self.last_state_change.isoformat() if self.last_state_change else None,
            'error_details': self.error_details
        }
        
        # Add timing info
        if self.activation_start_time and self.current_state == MonitoringState.ACTIVATING:
            info['activation_duration'] = (datetime.now() - self.activation_start_time).total_seconds()
        
        return info
    
    def is_monitoring_active(self) -> bool:
        """Check if monitoring is currently active"""
        return self.current_state == MonitoringState.ACTIVE
    
    def can_start_monitoring(self) -> Tuple[bool, Optional[str]]:
        """Check if monitoring can be started"""
        if self.current_state == MonitoringState.ACTIVE:
            return False, "Monitoring is already active"
        elif self.current_state == MonitoringState.ACTIVATING:
            return False, "Monitoring is currently activating"
        elif self.current_state == MonitoringState.DEACTIVATING:
            return False, "Monitoring is currently deactivating"
        else:
            return True, None
    
    def can_stop_monitoring(self) -> Tuple[bool, Optional[str]]:
        """Check if monitoring can be stopped"""
        if self.current_state == MonitoringState.INACTIVE:
            return False, "Monitoring is not active"
        elif self.current_state == MonitoringState.DEACTIVATING:
            return False, "Monitoring is already deactivating"
        elif self.current_state == MonitoringState.ACTIVATING:
            return False, "Monitoring is still activating"
        else:
            return True, None
    
    async def _notify_state_change(self, new_state: MonitoringState, context: Dict[str, Any]):
        """Notify all callbacks of state change"""
        for callback in self.state_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(new_state, context)
                else:
                    callback(new_state, context)
            except Exception as e:
                logger.error(f"Error in state callback: {e}")
    
    def _is_valid_transition(self, from_state: MonitoringState, to_state: MonitoringState) -> bool:
        """Validate state transitions"""
        valid_transitions = {
            MonitoringState.INACTIVE: [MonitoringState.ACTIVATING, MonitoringState.ERROR],
            MonitoringState.ACTIVATING: [MonitoringState.ACTIVE, MonitoringState.ERROR, MonitoringState.INACTIVE],
            MonitoringState.ACTIVE: [MonitoringState.DEACTIVATING, MonitoringState.ERROR],
            MonitoringState.DEACTIVATING: [MonitoringState.INACTIVE, MonitoringState.ERROR],
            MonitoringState.ERROR: [MonitoringState.INACTIVE, MonitoringState.ACTIVATING]
        }
        
        return to_state in valid_transitions.get(from_state, [])
    
    def _are_critical_components_active(self) -> bool:
        """Check if critical components are active for monitoring"""
        # At minimum, we need vision intelligence and macOS indicator
        return (self.component_status.get('vision_intelligence', False) and 
                self.component_status.get('macos_indicator', False))
    
    def _save_state(self):
        """Persist state to disk"""
        try:
            self.persistence_file.parent.mkdir(parents=True, exist_ok=True)
            
            state_data = {
                'current_state': self.current_state.value,
                'active_capabilities': [cap.value for cap in self.active_capabilities],
                'last_state_change': self.last_state_change.isoformat() if self.last_state_change else None,
                'component_status': self.component_status
            }
            
            with open(self.persistence_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save monitoring state: {e}")
    
    def _load_state(self):
        """Load persisted state from disk"""
        try:
            if self.persistence_file.exists():
                with open(self.persistence_file, 'r') as f:
                    state_data = json.load(f)
                
                # Only restore state if it was active (don't auto-restart monitoring)
                if state_data.get('current_state') == MonitoringState.ACTIVE.value:
                    # Set to inactive - monitoring must be explicitly restarted
                    self.current_state = MonitoringState.INACTIVE
                    logger.info("[STATE] Previous monitoring session detected, set to inactive")
                
                # Restore capabilities
                for cap_value in state_data.get('active_capabilities', []):
                    try:
                        capability = MonitoringCapability(cap_value)
                        self.active_capabilities.append(capability)
                    except ValueError:
                        pass
                
                logger.info("[STATE] Loaded monitoring state from disk")
                
        except Exception as e:
            logger.error(f"Failed to load monitoring state: {e}")


# Global instance
_state_manager = None


def get_state_manager():
    """Get or create the global state manager"""
    global _state_manager
    if _state_manager is None:
        _state_manager = MonitoringStateManager()
    return _state_manager