#!/usr/bin/env python3
"""
Autonomous Action Queue System for JARVIS

This module provides a priority-based queue system for managing autonomous actions
with safety controls, retry logic, and comprehensive monitoring. The queue processes
actions based on dynamic priority scoring that considers factors like urgency,
confidence, age, and category importance.

The system supports:
- Priority-based action execution
- Concurrent action processing with limits
- Automatic retry with exponential backoff
- Permission management integration
- Real-time queue monitoring and statistics
- Configurable safety controls

Example:
    >>> from autonomy.action_queue import action_queue
    >>> action = AutonomousAction(...)
    >>> await action_queue.add_action(action)
    >>> state = action_queue.get_queue_state()
"""

import asyncio
import heapq
import logging
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict

from .autonomous_decision_engine import AutonomousAction, ActionPriority, ActionCategory
from .action_executor import ActionExecutor, ExecutionResult, ExecutionStatus
from .permission_manager import PermissionManager

logger = logging.getLogger(__name__)

class QueueStatus(Enum):
    """Status enumeration for the action queue.
    
    Attributes:
        IDLE: Queue is empty and not processing
        PROCESSING: Queue is actively processing actions
        PAUSED: Queue processing is temporarily stopped
        OVERLOADED: Queue has reached capacity limits
    """
    IDLE = "idle"
    PROCESSING = "processing"
    PAUSED = "paused"
    OVERLOADED = "overloaded"

@dataclass(order=True)
class QueuedAction:
    """Represents an action in the queue with associated metadata.
    
    This class wraps an AutonomousAction with queue-specific information
    including priority scoring, timing data, and retry tracking.
    
    Attributes:
        priority_score: Dynamic priority score (lower = higher priority)
        action: The autonomous action to execute
        queued_at: Timestamp when action was added to queue
        attempts: Number of execution attempts made
        last_attempt: Timestamp of most recent execution attempt
    """
    priority_score: float = field(compare=True)
    action: AutonomousAction = field(compare=False)
    queued_at: datetime = field(default_factory=datetime.now, compare=False)
    attempts: int = field(default=0, compare=False)
    last_attempt: Optional[datetime] = field(default=None, compare=False)
    
    @property
    def age_seconds(self) -> float:
        """Calculate how long the action has been queued.
        
        Returns:
            float: Age of the queued action in seconds
        """
        return (datetime.now() - self.queued_at).total_seconds()
        
    def calculate_priority_score(self) -> float:
        """Calculate dynamic priority score based on multiple factors.
        
        The priority score considers:
        - Base priority level (1-5, lower is higher priority)
        - Action confidence (higher confidence = higher priority)
        - Queue age (older actions get priority boost)
        - Category importance (security > communication > maintenance)
        
        Returns:
            float: Calculated priority score (lower = higher priority)
        """
        # Base priority (1-5, lower is higher priority)
        base_score = self.action.priority.value
        
        # Adjust for confidence (higher confidence = higher priority)
        confidence_boost = (1 - self.action.confidence) * 2
        
        # Adjust for age (older actions get priority boost)
        age_boost = min(self.age_seconds / 300, 1.0)  # Max boost after 5 minutes
        
        # Adjust for category importance
        category_weights = {
            ActionCategory.SECURITY: -2,      # High priority
            ActionCategory.COMMUNICATION: -1,  
            ActionCategory.CALENDAR: -1,
            ActionCategory.NOTIFICATION: 0,
            ActionCategory.WORKFLOW: 0,
            ActionCategory.ORGANIZATION: 1,    # Lower priority
            ActionCategory.MAINTENANCE: 2
        }
        category_adjustment = category_weights.get(self.action.category, 0)
        
        # Calculate final score (lower is higher priority)
        final_score = base_score + confidence_boost + category_adjustment - age_boost
        
        return final_score

class ActionQueueManager:
    """Manages the priority queue of autonomous actions with safety controls.
    
    This class provides a comprehensive queue management system that handles
    action prioritization, concurrent execution, retry logic, and monitoring.
    It integrates with the permission system for safety and provides detailed
    statistics and callbacks for monitoring.
    
    Attributes:
        executor: ActionExecutor instance for running actions
        permission_manager: PermissionManager for safety controls
        action_queue: Priority queue of actions (min heap)
        max_queue_size: Maximum number of queued actions
        max_concurrent_actions: Maximum concurrent executions
        max_retries: Maximum retry attempts per action
        retry_delay: Delay between retry attempts in seconds
        status: Current queue status
        active_executions: Number of currently executing actions
        processing_task: Background processing task
        stats: Execution statistics dictionary
        status_callbacks: List of status change callbacks
        execution_callbacks: List of execution result callbacks
    """
    
    def __init__(self, executor: Optional[ActionExecutor] = None):
        """Initialize the action queue manager.
        
        Args:
            executor: Optional ActionExecutor instance. Creates default if None.
        """
        self.executor = executor or ActionExecutor()
        self.permission_manager = PermissionManager()
        
        # Priority queue (min heap - lower scores have higher priority)
        self.action_queue: List[QueuedAction] = []
        
        # Queue configuration
        self.max_queue_size = 100
        self.max_concurrent_actions = 3
        self.max_retries = 3
        self.retry_delay = 5.0  # seconds
        
        # Queue state
        self.status = QueueStatus.IDLE
        self.active_executions = 0
        self.processing_task = None
        
        # Statistics
        self.stats = {
            'total_queued': 0,
            'total_executed': 0,
            'total_failed': 0,
            'total_cancelled': 0,
            'execution_times': [],
            'category_counts': defaultdict(int)
        }
        
        # Callbacks
        self.status_callbacks: List[Callable] = []
        self.execution_callbacks: List[Callable] = []
        
    async def add_action(self, action: AutonomousAction) -> bool:
        """Add an action to the priority queue.
        
        Creates a QueuedAction wrapper, calculates priority score, and adds
        to the queue. Automatically starts processing if queue was idle.
        
        Args:
            action: The autonomous action to queue
            
        Returns:
            bool: True if action was successfully queued, False if queue full
            
        Raises:
            Exception: If priority calculation or queue operation fails
        """
        if len(self.action_queue) >= self.max_queue_size:
            logger.warning(f"Queue full, rejecting action: {action.action_type}")
            self.status = QueueStatus.OVERLOADED
            return False
            
        # Create queued action
        queued = QueuedAction(
            priority_score=0,  # Will be calculated
            action=action
        )
        
        # Calculate priority score
        queued.priority_score = queued.calculate_priority_score()
        
        # Add to queue
        heapq.heappush(self.action_queue, queued)
        
        # Update stats
        self.stats['total_queued'] += 1
        self.stats['category_counts'][action.category] += 1
        
        logger.info(f"Queued action: {action.action_type} (priority: {queued.priority_score:.2f})")
        
        # Start processing if not already running
        if self.status == QueueStatus.IDLE:
            await self.start_processing()
            
        return True
        
    async def add_actions(self, actions: List[AutonomousAction]) -> int:
        """Add multiple actions to the queue in priority order.
        
        Sorts actions by priority and confidence before adding to ensure
        optimal queue ordering.
        
        Args:
            actions: List of autonomous actions to queue
            
        Returns:
            int: Number of actions successfully added
        """
        added = 0
        
        # Sort by priority before adding
        sorted_actions = sorted(
            actions, 
            key=lambda a: (a.priority.value, -a.confidence)
        )
        
        for action in sorted_actions:
            if await self.add_action(action):
                added += 1
                
        return added
        
    async def start_processing(self):
        """Start the queue processing loop.
        
        Creates a background task to continuously process queued actions.
        Does nothing if processing is already active.
        """
        if self.processing_task and not self.processing_task.done():
            return
            
        self.status = QueueStatus.PROCESSING
        self.processing_task = asyncio.create_task(self._process_queue())
        logger.info("Started queue processing")
        
    async def stop_processing(self):
        """Stop the queue processing loop.
        
        Sets status to paused and cancels the processing task.
        Active executions will complete normally.
        """
        self.status = QueueStatus.PAUSED
        if self.processing_task:
            self.processing_task.cancel()
        logger.info("Stopped queue processing")
        
    async def _process_queue(self):
        """Main queue processing loop.
        
        Continuously processes actions from the queue while respecting
        concurrency limits and handling errors gracefully.
        
        Raises:
            Exception: Logs and continues processing on individual errors
        """
        while self.status == QueueStatus.PROCESSING:
            try:
                # Check if we can process more actions
                if self.active_executions >= self.max_concurrent_actions:
                    await asyncio.sleep(0.5)
                    continue
                    
                # Get next action from queue
                if not self.action_queue:
                    self.status = QueueStatus.IDLE
                    break
                    
                queued_action = heapq.heappop(self.action_queue)
                
                # Check if action is too old
                if queued_action.age_seconds > 600:  # 10 minutes
                    logger.warning(f"Action too old, discarding: {queued_action.action.action_type}")
                    self.stats['total_cancelled'] += 1
                    continue
                    
                # Execute action
                asyncio.create_task(self._execute_action(queued_action))
                
                # Small delay between actions
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                await asyncio.sleep(1)
                
    async def _execute_action(self, queued_action: QueuedAction):
        """Execute a single action with retry logic and permission checking.
        
        Handles the complete execution lifecycle including permission requests,
        execution, result processing, and retry logic for failed actions.
        
        Args:
            queued_action: The queued action to execute
            
        Raises:
            Exception: Logs execution errors and updates failure statistics
        """
        self.active_executions += 1
        action = queued_action.action
        
        try:
            # Check permission if required
            if action.requires_permission:
                permission = await self._request_permission(action)
                if not permission:
                    logger.info(f"Permission denied for: {action.action_type}")
                    self.stats['total_cancelled'] += 1
                    return
                    
            # Execute action
            logger.info(f"Executing: {action.action_type} (attempt {queued_action.attempts + 1})")
            
            result = await self.executor.execute_action(action)
            
            # Handle result
            if result.status == ExecutionStatus.SUCCESS:
                self.stats['total_executed'] += 1
                if result.execution_time:
                    self.stats['execution_times'].append(result.execution_time)
                    
                # Notify callbacks
                await self._notify_execution(result)
                
            elif result.status == ExecutionStatus.FAILED:
                # Check if we should retry
                queued_action.attempts += 1
                queued_action.last_attempt = datetime.now()
                
                if queued_action.attempts < self.max_retries:
                    # Re-queue for retry
                    await asyncio.sleep(self.retry_delay)
                    queued_action.priority_score = queued_action.calculate_priority_score()
                    heapq.heappush(self.action_queue, queued_action)
                    logger.info(f"Re-queued for retry: {action.action_type}")
                else:
                    self.stats['total_failed'] += 1
                    logger.error(f"Action failed after {queued_action.attempts} attempts: {action.action_type}")
                    
        except Exception as e:
            logger.error(f"Action execution error: {e}")
            self.stats['total_failed'] += 1
            
        finally:
            self.active_executions -= 1
            
    async def _request_permission(self, action: AutonomousAction) -> bool:
        """Request permission for an action from the permission manager.
        
        Creates a permission request with current queue context and waits
        for approval within the timeout period.
        
        Args:
            action: The action requiring permission
            
        Returns:
            bool: True if permission granted, False otherwise
        """
        request = PermissionRequest(
            action=action,
            context={
                'queue_size': len(self.action_queue),
                'active_executions': self.active_executions
            },
            timeout=30  # 30 second timeout
        )
        
        return await self.permission_manager.request_permission(request)
        
    async def _notify_execution(self, result: ExecutionResult):
        """Notify registered callbacks about action execution results.
        
        Calls all registered execution callbacks with the result, logging
        any callback errors without interrupting other callbacks.
        
        Args:
            result: The execution result to broadcast
        """
        for callback in self.execution_callbacks:
            try:
                await callback(result)
            except Exception as e:
                logger.error(f"Callback error: {e}")
                
    def get_queue_state(self) -> Dict[str, Any]:
        """Get comprehensive current queue state and statistics.
        
        Provides detailed information about queue status, pending actions,
        execution statistics, and performance metrics.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - status: Current queue status
                - queue_length: Number of pending actions
                - active_executions: Number of currently executing actions
                - queue_preview: List of top 10 pending actions with details
                - stats: Comprehensive execution statistics
        """
        # Get queue preview
        queue_preview = []
        temp_queue = list(self.action_queue)
        temp_queue.sort()
        
        for item in temp_queue[:10]:  # Top 10 items
            queue_preview.append({
                'type': item.action.action_type,
                'target': item.action.target,
                'priority': item.action.priority.name,
                'score': item.priority_score,
                'age': item.age_seconds,
                'attempts': item.attempts
            })
            
        # Calculate average execution time
        avg_execution_time = 0
        if self.stats['execution_times']:
            avg_execution_time = sum(self.stats['execution_times']) / len(self.stats['execution_times'])
            
        return {
            'status': self.status.value,
            'queue_length': len(self.action_queue),
            'active_executions': self.active_executions,
            'queue_preview': queue_preview,
            'stats': {
                'total_queued': self.stats['total_queued'],
                'total_executed': self.stats['total_executed'],
                'total_failed': self.stats['total_failed'],
                'total_cancelled': self.stats['total_cancelled'],
                'avg_execution_time': avg_execution_time,
                'category_distribution': dict(self.stats['category_counts'])
            }
        }
        
    def clear_queue(self):
        """Clear all pending actions from the queue.
        
        Removes all queued actions and updates cancellation statistics.
        Does not affect currently executing actions.
        """
        cleared = len(self.action_queue)
        self.action_queue.clear()
        self.stats['total_cancelled'] += cleared
        logger.info(f"Cleared {cleared} actions from queue")
        
    def pause(self):
        """Pause queue processing.
        
        Sets status to paused, stopping new action processing.
        Currently executing actions will complete normally.
        """
        self.status = QueueStatus.PAUSED
        logger.info("Queue paused")
        
    def resume(self):
        """Resume queue processing if currently paused.
        
        Restarts the processing loop if the queue was paused.
        Does nothing if already processing or idle.
        """
        if self.status == QueueStatus.PAUSED:
            self.status = QueueStatus.PROCESSING
            asyncio.create_task(self._process_queue())
            logger.info("Queue resumed")
            
    def add_status_callback(self, callback: Callable):
        """Add a callback for queue status changes.
        
        Args:
            callback: Callable to invoke on status changes
        """
        self.status_callbacks.append(callback)
        
    def add_execution_callback(self, callback: Callable):
        """Add a callback for action execution results.
        
        Args:
            callback: Callable to invoke with ExecutionResult objects
        """
        self.execution_callbacks.append(callback)

# Global queue instance
action_queue = ActionQueueManager()

async def test_action_queue():
    """Test the action queue system with sample actions.
    
    Creates test actions of different priorities and categories,
    adds them to the queue, and monitors execution progress.
    Demonstrates queue functionality and statistics collection.
    
    Example:
        >>> await test_action_queue()
        🤖 Action Queue Test
        ==================================================
        ✅ Added 3 actions to queue
        ...
    """
    from .autonomous_decision_engine import AutonomousAction, ActionPriority, ActionCategory
    
    # Create test actions
    test_actions = [
        AutonomousAction(
            action_type="handle_urgent_message",
            target="Slack",
            params={"urgency": "high"},
            priority=ActionPriority.HIGH,
            confidence=0.9,
            category=ActionCategory.COMMUNICATION,
            reasoning="Urgent message detected"
        ),
        AutonomousAction(
            action_type="security_check",
            target="System",
            params={"check_type": "password"},
            priority=ActionPriority.CRITICAL,
            confidence=0.95,
            category=ActionCategory.SECURITY,
            reasoning="Security verification needed"
        ),
        AutonomousAction(
            action_type="organize_windows",
            target="Desktop",
            params={"layout": "tiled"},
            priority=ActionPriority.LOW,
            confidence=0.7,
            category=ActionCategory.ORGANIZATION,
            reasoning="Desktop cleanup suggested"
        )
    ]
    
    print("🤖 Action Queue Test")
    print("=" * 50)
    
    # Add actions to queue
    added = await action_queue.add_actions(test_actions)
    print(f"\n✅ Added {added} actions to queue")
    
    # Get queue state
    state = action_queue.get_queue_state()
    print(f"\n📊 Queue State:")
    print(f"   Status: {state['status']}")
    print(f"   Queue Length: {state['queue_length']}")
    print(f"   Active Executions: {state['active_executions']}")
    
    print(f"\n📋 Queue Preview:")
    for i, item in enumerate(state['queue_preview'], 1):
        print(f"   {i}. {item['type']} ({item['priority']}) - Score: {item['score']:.2f}")
    
    # Let it process for a bit
    await asyncio.sleep(2)
    
    # Final state
    final_state = action_queue.get_queue_state()
    print(f"\n📈 Final Stats:")
    for key, value in final_state['stats'].items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    asyncio.run(test_action_queue())