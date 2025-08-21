#!/usr/bin/env python3
"""
System States and Transitions for JARVIS Autonomous System
Manages the overall state machine for autonomous operations
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import json

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """Overall system states"""
    INITIALIZING = auto()
    IDLE = auto()
    MONITORING = auto()
    PROCESSING = auto()
    DECIDING = auto()
    EXECUTING = auto()
    ERROR_RECOVERY = auto()
    PAUSED = auto()
    SHUTDOWN = auto()


class ComponentState(Enum):
    """Individual component states"""
    NOT_INITIALIZED = auto()
    READY = auto()
    ACTIVE = auto()
    BUSY = auto()
    ERROR = auto()
    OFFLINE = auto()


class TransitionReason(Enum):
    """Reasons for state transitions"""
    USER_REQUEST = "user_request"
    AUTOMATIC = "automatic"
    ERROR = "error"
    RECOVERY = "recovery"
    TIMEOUT = "timeout"
    COMPLETION = "completion"
    EXTERNAL_TRIGGER = "external_trigger"


@dataclass
class StateTransition:
    """Record of a state transition"""
    from_state: SystemState
    to_state: SystemState
    reason: TransitionReason
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class ComponentStatus:
    """Status of a system component"""
    name: str
    state: ComponentState
    last_update: datetime = field(default_factory=datetime.now)
    health_score: float = 1.0  # 0.0 to 1.0
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_healthy(self) -> bool:
        return self.state in [ComponentState.READY, ComponentState.ACTIVE] and self.health_score > 0.7
        
    @property
    def needs_attention(self) -> bool:
        return self.state == ComponentState.ERROR or self.health_score < 0.5 or self.error_count > 5


class SystemStateManager:
    """Manages system states and transitions"""
    
    def __init__(self):
        self.current_state = SystemState.INITIALIZING
        self.previous_state = None
        self.state_history: List[StateTransition] = []
        self.components: Dict[str, ComponentStatus] = {}
        
        # State transition rules
        self.valid_transitions = {
            SystemState.INITIALIZING: {
                SystemState.IDLE,
                SystemState.ERROR_RECOVERY
            },
            SystemState.IDLE: {
                SystemState.MONITORING,
                SystemState.PAUSED,
                SystemState.SHUTDOWN
            },
            SystemState.MONITORING: {
                SystemState.PROCESSING,
                SystemState.IDLE,
                SystemState.PAUSED,
                SystemState.ERROR_RECOVERY
            },
            SystemState.PROCESSING: {
                SystemState.DECIDING,
                SystemState.MONITORING,
                SystemState.ERROR_RECOVERY
            },
            SystemState.DECIDING: {
                SystemState.EXECUTING,
                SystemState.MONITORING,
                SystemState.ERROR_RECOVERY
            },
            SystemState.EXECUTING: {
                SystemState.MONITORING,
                SystemState.ERROR_RECOVERY,
                SystemState.PROCESSING
            },
            SystemState.ERROR_RECOVERY: {
                SystemState.IDLE,
                SystemState.MONITORING,
                SystemState.SHUTDOWN
            },
            SystemState.PAUSED: {
                SystemState.MONITORING,
                SystemState.IDLE,
                SystemState.SHUTDOWN
            },
            SystemState.SHUTDOWN: set()  # Terminal state
        }
        
        # State callbacks
        self.state_callbacks: Dict[SystemState, List[Callable]] = {
            state: [] for state in SystemState
        }
        self.transition_callbacks: List[Callable] = []
        
        # State timeouts
        self.state_timeouts = {
            SystemState.INITIALIZING: 30,  # seconds
            SystemState.PROCESSING: 10,
            SystemState.DECIDING: 5,
            SystemState.EXECUTING: 60,
            SystemState.ERROR_RECOVERY: 120
        }
        
        # Timeout tracking
        self.state_entered_time = datetime.now()
        self.timeout_task = None
        
    def register_component(self, name: str, initial_state: ComponentState = ComponentState.NOT_INITIALIZED):
        """Register a system component"""
        self.components[name] = ComponentStatus(
            name=name,
            state=initial_state
        )
        logger.info(f"Registered component: {name}")
        
    def update_component_state(self, name: str, state: ComponentState, 
                             health_score: Optional[float] = None,
                             metadata: Optional[Dict[str, Any]] = None):
        """Update a component's state"""
        if name not in self.components:
            logger.warning(f"Unknown component: {name}")
            return
            
        component = self.components[name]
        component.state = state
        component.last_update = datetime.now()
        
        if health_score is not None:
            component.health_score = max(0.0, min(1.0, health_score))
            
        if metadata:
            component.metadata.update(metadata)
            
        # Track errors
        if state == ComponentState.ERROR:
            component.error_count += 1
            
        logger.debug(f"Component {name} state updated to {state.name}")
        
        # Check if component issues should trigger system state change
        self._check_component_health()
        
    def _check_component_health(self):
        """Check overall component health and trigger state changes if needed"""
        unhealthy_components = [
            c for c in self.components.values() 
            if c.needs_attention
        ]
        
        # If critical components are unhealthy, trigger error recovery
        critical_components = ['vision_pipeline', 'decision_engine', 'action_queue']
        critical_unhealthy = [
            c for c in unhealthy_components 
            if c.name in critical_components
        ]
        
        if critical_unhealthy and self.current_state not in [
            SystemState.ERROR_RECOVERY, 
            SystemState.SHUTDOWN
        ]:
            logger.warning(f"Critical components unhealthy: {[c.name for c in critical_unhealthy]}")
            asyncio.create_task(
                self.transition_to(SystemState.ERROR_RECOVERY, TransitionReason.ERROR)
            )
            
    async def transition_to(self, new_state: SystemState, 
                          reason: TransitionReason,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Transition to a new state"""
        if new_state == self.current_state:
            logger.debug(f"Already in state {new_state.name}")
            return True
            
        # Check if transition is valid
        if new_state not in self.valid_transitions.get(self.current_state, set()):
            logger.error(
                f"Invalid transition: {self.current_state.name} -> {new_state.name}"
            )
            return False
            
        # Create transition record
        transition = StateTransition(
            from_state=self.current_state,
            to_state=new_state,
            reason=reason,
            metadata=metadata or {}
        )
        
        # Execute transition
        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_entered_time = datetime.now()
        self.state_history.append(transition)
        
        # Cancel previous timeout
        if self.timeout_task:
            self.timeout_task.cancel()
            
        # Set new timeout if applicable
        if new_state in self.state_timeouts:
            self.timeout_task = asyncio.create_task(
                self._handle_state_timeout(new_state)
            )
            
        logger.info(
            f"State transition: {transition.from_state.name} -> {transition.to_state.name} "
            f"(reason: {reason.value})"
        )
        
        # Execute callbacks
        await self._execute_state_callbacks(new_state)
        await self._execute_transition_callbacks(transition)
        
        return True
        
    async def _handle_state_timeout(self, state: SystemState):
        """Handle state timeout"""
        timeout = self.state_timeouts[state]
        await asyncio.sleep(timeout)
        
        # Check if still in the same state
        if self.current_state == state:
            logger.warning(f"State {state.name} timed out after {timeout}s")
            
            # Determine recovery action
            if state == SystemState.INITIALIZING:
                await self.transition_to(SystemState.ERROR_RECOVERY, TransitionReason.TIMEOUT)
            elif state in [SystemState.PROCESSING, SystemState.DECIDING]:
                await self.transition_to(SystemState.MONITORING, TransitionReason.TIMEOUT)
            elif state == SystemState.EXECUTING:
                await self.transition_to(SystemState.ERROR_RECOVERY, TransitionReason.TIMEOUT)
                
    async def _execute_state_callbacks(self, state: SystemState):
        """Execute callbacks for entering a state"""
        for callback in self.state_callbacks.get(state, []):
            try:
                await callback(state)
            except Exception as e:
                logger.error(f"State callback error: {e}")
                
    async def _execute_transition_callbacks(self, transition: StateTransition):
        """Execute callbacks for state transitions"""
        for callback in self.transition_callbacks:
            try:
                await callback(transition)
            except Exception as e:
                logger.error(f"Transition callback error: {e}")
                
    def add_state_callback(self, state: SystemState, callback: Callable):
        """Add callback for entering a specific state"""
        self.state_callbacks[state].append(callback)
        
    def add_transition_callback(self, callback: Callable):
        """Add callback for any state transition"""
        self.transition_callbacks.append(callback)
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'current_state': self.current_state.name,
            'previous_state': self.previous_state.name if self.previous_state else None,
            'state_duration': (datetime.now() - self.state_entered_time).total_seconds(),
            'components': {
                name: {
                    'state': comp.state.name,
                    'health': comp.health_score,
                    'errors': comp.error_count,
                    'last_update': comp.last_update.isoformat()
                }
                for name, comp in self.components.items()
            },
            'healthy_components': sum(1 for c in self.components.values() if c.is_healthy),
            'unhealthy_components': sum(1 for c in self.components.values() if c.needs_attention),
            'recent_transitions': [
                {
                    'from': t.from_state.name,
                    'to': t.to_state.name,
                    'reason': t.reason.value,
                    'timestamp': t.timestamp.isoformat()
                }
                for t in self.state_history[-5:]  # Last 5 transitions
            ]
        }
        
    def can_transition_to(self, state: SystemState) -> bool:
        """Check if transition to given state is valid"""
        return state in self.valid_transitions.get(self.current_state, set())
        
    def get_available_transitions(self) -> Set[SystemState]:
        """Get all valid transitions from current state"""
        return self.valid_transitions.get(self.current_state, set())
        
    async def initialize_system(self) -> bool:
        """Initialize the system"""
        logger.info("Initializing system...")
        
        # Register core components
        core_components = [
            'vision_pipeline',
            'decision_engine',
            'action_queue',
            'websocket_manager',
            'behavior_manager'
        ]
        
        for component in core_components:
            self.register_component(component)
            
        # Simulate initialization
        await asyncio.sleep(1)
        
        # Update component states
        for component in core_components:
            self.update_component_state(
                component, 
                ComponentState.READY,
                health_score=1.0
            )
            
        # Transition to idle
        success = await self.transition_to(
            SystemState.IDLE,
            TransitionReason.COMPLETION
        )
        
        return success
        
    async def start_monitoring(self) -> bool:
        """Start system monitoring"""
        if self.current_state != SystemState.IDLE:
            logger.warning(f"Cannot start monitoring from state {self.current_state.name}")
            return False
            
        return await self.transition_to(
            SystemState.MONITORING,
            TransitionReason.USER_REQUEST
        )
        
    async def pause_system(self) -> bool:
        """Pause the system"""
        if self.current_state in [SystemState.SHUTDOWN, SystemState.INITIALIZING]:
            return False
            
        return await self.transition_to(
            SystemState.PAUSED,
            TransitionReason.USER_REQUEST
        )
        
    async def resume_system(self) -> bool:
        """Resume from pause"""
        if self.current_state != SystemState.PAUSED:
            return False
            
        # Return to monitoring
        return await self.transition_to(
            SystemState.MONITORING,
            TransitionReason.USER_REQUEST
        )
        
    async def shutdown_system(self) -> bool:
        """Shutdown the system"""
        return await self.transition_to(
            SystemState.SHUTDOWN,
            TransitionReason.USER_REQUEST
        )


# Global state manager instance
state_manager = SystemStateManager()


async def test_state_system():
    """Test the state management system"""
    print("🔄 Testing System State Manager")
    print("=" * 50)
    
    manager = SystemStateManager()
    
    # Add callbacks
    async def state_callback(state):
        print(f"   Entered state: {state.name}")
        
    async def transition_callback(transition):
        print(f"   Transition: {transition.from_state.name} -> {transition.to_state.name}")
        
    manager.add_state_callback(SystemState.MONITORING, state_callback)
    manager.add_transition_callback(transition_callback)
    
    # Initialize system
    print("\n🚀 Initializing system...")
    await manager.initialize_system()
    
    # Get status
    status = manager.get_system_status()
    print(f"\n📊 System Status:")
    print(f"   Current State: {status['current_state']}")
    print(f"   Components: {status['healthy_components']} healthy, {status['unhealthy_components']} unhealthy")
    
    # Test transitions
    print("\n🔄 Testing state transitions...")
    
    # Start monitoring
    success = await manager.start_monitoring()
    print(f"   Start monitoring: {'✓' if success else '✗'}")
    
    # Simulate processing
    await manager.transition_to(SystemState.PROCESSING, TransitionReason.AUTOMATIC)
    await asyncio.sleep(0.5)
    
    await manager.transition_to(SystemState.DECIDING, TransitionReason.COMPLETION)
    await asyncio.sleep(0.5)
    
    await manager.transition_to(SystemState.EXECUTING, TransitionReason.COMPLETION)
    
    # Simulate component error
    print("\n⚠️ Simulating component error...")
    manager.update_component_state(
        'vision_pipeline',
        ComponentState.ERROR,
        health_score=0.2
    )
    
    # Wait for error recovery
    await asyncio.sleep(1)
    
    # Final status
    final_status = manager.get_system_status()
    print(f"\n📊 Final Status:")
    print(f"   Current State: {final_status['current_state']}")
    print(f"   State History: {len(manager.state_history)} transitions")
    
    print("\n✅ State system test complete!")


if __name__ == "__main__":
    asyncio.run(test_state_system())