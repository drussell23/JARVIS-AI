#!/usr/bin/env python3
"""
Permission Manager for JARVIS Autonomous System.

This module manages and learns autonomous action permissions through user feedback
and pattern recognition. It provides intelligent permission checking, learning from
user decisions, and automated rule suggestions based on historical patterns.

The system tracks permission decisions, analyzes patterns in user behavior, and
gradually automates routine decisions while maintaining security for sensitive actions.

Example:
    >>> from autonomy.permission_manager import PermissionManager
    >>> manager = PermissionManager()
    >>> decision, confidence, reason = manager.check_permission(action)
    >>> if decision is None:
    ...     # Ask user for permission
    ...     user_approved = ask_user(action)
    ...     manager.record_decision(action, user_approved)
"""

import json
import logging
from typing import Dict, Tuple, Optional, List, Any
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

from .autonomous_decision_engine import AutonomousAction, ActionCategory, ActionPriority

logger = logging.getLogger(__name__)

class PermissionRecord:
    """Record of permission decisions for learning and pattern analysis.
    
    Tracks approval/denial counts, decision history, and contextual information
    to enable intelligent permission automation over time.
    
    Attributes:
        approved_count (int): Number of times this action was approved
        denied_count (int): Number of times this action was denied
        last_decision (Optional[bool]): Most recent permission decision
        last_decision_time (Optional[datetime]): Timestamp of last decision
        context_history (List[Dict]): Historical context data for decisions
    """
    
    def __init__(self):
        """Initialize a new permission record with zero counts."""
        self.approved_count = 0
        self.denied_count = 0
        self.last_decision = None
        self.last_decision_time = None
        self.context_history = []
        
    @property
    def total_decisions(self) -> int:
        """Get total number of decisions made for this action.
        
        Returns:
            int: Sum of approved and denied decisions
        """
        return self.approved_count + self.denied_count
    
    @property
    def approval_rate(self) -> float:
        """Calculate approval rate for this action.
        
        Returns:
            float: Ratio of approvals to total decisions (0.0-1.0).
                  Returns 0.5 for actions with no decision history.
        """
        if self.total_decisions == 0:
            return 0.5  # Default to 50% for unknown actions
        return self.approved_count / self.total_decisions
    
    def record_decision(self, approved: bool, context: Optional[Dict] = None) -> None:
        """Record a permission decision with optional context.
        
        Args:
            approved (bool): Whether the action was approved
            context (Optional[Dict]): Contextual information about the decision
                including time, priority, reasoning, etc.
        """
        if approved:
            self.approved_count += 1
        else:
            self.denied_count += 1
        
        self.last_decision = approved
        self.last_decision_time = datetime.now()
        
        if context:
            self.context_history.append({
                'timestamp': datetime.now().isoformat(),
                'approved': approved,
                'context': context
            })
            # Keep only last 50 contexts to prevent unbounded growth
            if len(self.context_history) > 50:
                self.context_history = self.context_history[-50:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary for JSON serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the permission record
        """
        return {
            'approved_count': self.approved_count,
            'denied_count': self.denied_count,
            'last_decision': self.last_decision,
            'last_decision_time': self.last_decision_time.isoformat() if self.last_decision_time else None,
            'context_history': self.context_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PermissionRecord':
        """Create PermissionRecord from dictionary data.
        
        Args:
            data (Dict[str, Any]): Dictionary containing record data
            
        Returns:
            PermissionRecord: Reconstructed permission record
        """
        record = cls()
        record.approved_count = data.get('approved_count', 0)
        record.denied_count = data.get('denied_count', 0)
        record.last_decision = data.get('last_decision')
        if data.get('last_decision_time'):
            record.last_decision_time = datetime.fromisoformat(data['last_decision_time'])
        record.context_history = data.get('context_history', [])
        return record

class PermissionManager:
    """Manages and learns autonomous action permissions.
    
    This class provides intelligent permission checking by learning from user
    decisions and applying pattern-based rules. It can automatically approve
    or deny actions based on historical patterns while maintaining security
    for sensitive operations.
    
    Attributes:
        permissions_file (Path): Path to persistent permissions storage
        permissions (Dict[str, PermissionRecord]): Permission records by action key
        auto_approve_threshold (float): Approval rate threshold for auto-approval
        auto_deny_threshold (float): Approval rate threshold for auto-denial
        learning_threshold (int): Minimum decisions before auto-decisions
        category_rules (Dict): Category-specific permission rules
        quiet_hours (Dict): Time-based restriction configuration
        pattern_permissions (Dict): Pattern-based permission rules
    """
    
    def __init__(self):
        """Initialize the permission manager with default settings."""
        self.permissions_file = Path("backend/data/autonomous_permissions.json")
        self.permissions = self._load_permissions()
        
        # Confidence thresholds for automated decisions
        self.auto_approve_threshold = 0.90  # 90% approval rate for auto-approval
        self.auto_deny_threshold = 0.10     # 10% approval rate for auto-denial
        self.learning_threshold = 5         # Minimum decisions before auto-decisions
        
        # Category-specific rules for different action types
        self.category_rules = {
            ActionCategory.SECURITY: {
                'always_ask': True,  # Always ask for security actions
                'min_confidence': 0.95
            },
            ActionCategory.COMMUNICATION: {
                'always_ask': False,
                'min_confidence': 0.7
            },
            ActionCategory.NOTIFICATION: {
                'always_ask': False,
                'min_confidence': 0.6
            },
            ActionCategory.MAINTENANCE: {
                'always_ask': False,
                'min_confidence': 0.5
            }
        }
        
        # Time-based rules for quiet hours
        self.quiet_hours = {
            'start': 22,  # 10 PM
            'end': 8      # 8 AM
        }
        
        # Pattern-based permissions for common scenarios
        self.pattern_permissions = self._load_pattern_permissions()
    
    def _load_permissions(self) -> Dict[str, PermissionRecord]:
        """Load permissions from persistent storage.
        
        Returns:
            Dict[str, PermissionRecord]: Loaded permission records
        """
        if self.permissions_file.exists():
            try:
                with open(self.permissions_file, 'r') as f:
                    data = json.load(f)
                    return {
                        key: PermissionRecord.from_dict(record)
                        for key, record in data.items()
                    }
            except Exception as e:
                logger.error(f"Failed to load permissions: {e}")
        
        return {}
    
    def _save_permissions(self) -> None:
        """Save permissions to persistent storage.
        
        Raises:
            Exception: If file operations fail
        """
        self.permissions_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            data = {
                key: record.to_dict()
                for key, record in self.permissions.items()
            }
            with open(self.permissions_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save permissions: {e}")
    
    def _load_pattern_permissions(self) -> Dict[str, Dict[str, Any]]:
        """Load pattern-based permission rules.
        
        Returns:
            Dict[str, Dict[str, Any]]: Pattern rules with regex patterns,
                default permissions, and confidence adjustments
        """
        return {
            'routine_messages': {
                'pattern': r'automated|notification|reminder|digest',
                'default_permission': True,
                'confidence_boost': 0.1
            },
            'urgent_items': {
                'pattern': r'urgent|critical|emergency|asap',
                'default_permission': None,  # Always ask
                'confidence_boost': -0.2  # Reduce confidence for manual review
            },
            'bulk_actions': {
                'pattern': r'all|bulk|mass|batch',
                'default_permission': False,  # Default deny
                'confidence_boost': -0.3
            }
        }
    
    def check_permission(self, action: AutonomousAction) -> Tuple[Optional[bool], float, str]:
        """Check if action is permitted based on learned patterns.
        
        Analyzes the action against historical decisions, category rules,
        time-based restrictions, and pattern matching to determine if
        the action should be automatically approved, denied, or require
        user input.
        
        Args:
            action (AutonomousAction): The action to check permissions for
            
        Returns:
            Tuple[Optional[bool], float, str]: A tuple containing:
                - Permission decision (True/False for auto-decision, None for user input)
                - Confidence score (0.0-1.0)
                - Reason string explaining the decision
                
        Example:
            >>> decision, confidence, reason = manager.check_permission(action)
            >>> if decision is None:
            ...     # Need to ask user
            ...     user_decision = ask_user(action, reason)
        """
        # Always ask for security actions
        category_rule = self.category_rules.get(action.category, {})
        if category_rule.get('always_ask', False):
            return None, 0.0, "Security actions always require explicit permission"
        
        # Check quiet hours for non-critical actions
        if self._is_quiet_hours() and action.priority not in [ActionPriority.CRITICAL, ActionPriority.HIGH]:
            return False, 0.9, "Action blocked during quiet hours"
        
        # Build permission key for historical lookup
        permission_key = self._build_permission_key(action)
        
        # Check if we have history for this action
        if permission_key in self.permissions:
            record = self.permissions[permission_key]
            
            # Need minimum decisions before auto-deciding
            if record.total_decisions >= self.learning_threshold:
                approval_rate = record.approval_rate
                
                # Auto-approve if consistently approved
                if approval_rate >= self.auto_approve_threshold:
                    return True, approval_rate, f"Auto-approved based on {record.approved_count}/{record.total_decisions} past approvals"
                
                # Auto-deny if consistently denied
                elif approval_rate <= self.auto_deny_threshold:
                    return False, 1 - approval_rate, f"Auto-denied based on {record.denied_count}/{record.total_decisions} past denials"
        
        # Check pattern-based permissions
        pattern_decision = self._check_pattern_permissions(action)
        if pattern_decision[0] is not None:
            return pattern_decision
        
        # Check confidence threshold for category
        min_confidence = category_rule.get('min_confidence', 0.7)
        if action.confidence < min_confidence:
            return None, action.confidence, f"Confidence {action.confidence:.1%} below threshold {min_confidence:.1%}"
        
        # For new actions with high confidence, suggest approval
        if action.confidence >= 0.85:
            return None, action.confidence, "High confidence action - suggesting approval"
        
        # Default: ask user
        return None, action.confidence, "Requires user decision"
    
    def _build_permission_key(self, action: AutonomousAction) -> str:
        """Build a unique key for permission tracking.
        
        Args:
            action (AutonomousAction): Action to build key for
            
        Returns:
            str: Unique permission key combining category, type, and target
        """
        # Include category for better grouping
        return f"{action.category.value}:{action.action_type}:{action.target}"
    
    def _is_quiet_hours(self) -> bool:
        """Check if current time is in quiet hours.
        
        Returns:
            bool: True if current time falls within configured quiet hours
        """
        current_hour = datetime.now().hour
        
        if self.quiet_hours['start'] > self.quiet_hours['end']:
            # Quiet hours span midnight (e.g., 22:00 to 08:00)
            return current_hour >= self.quiet_hours['start'] or current_hour < self.quiet_hours['end']
        else:
            # Normal hours (e.g., 01:00 to 05:00)
            return self.quiet_hours['start'] <= current_hour < self.quiet_hours['end']
    
    def _check_pattern_permissions(self, action: AutonomousAction) -> Tuple[Optional[bool], float, str]:
        """Check pattern-based permission rules using regex matching.
        
        Args:
            action (AutonomousAction): Action to check against patterns
            
        Returns:
            Tuple[Optional[bool], float, str]: Permission decision, confidence, and reason
        """
        import re
        
        # Combine action details for pattern matching
        action_text = f"{action.action_type} {action.target} {action.reasoning}".lower()
        
        for pattern_name, rule in self.pattern_permissions.items():
            if re.search(rule['pattern'], action_text):
                if rule['default_permission'] is not None:
                    confidence = action.confidence + rule.get('confidence_boost', 0)
                    confidence = max(0, min(1, confidence))  # Clamp to [0, 1]
                    
                    return (
                        rule['default_permission'],
                        confidence,
                        f"Pattern '{pattern_name}' matched"
                    )
        
        return None, action.confidence, "No pattern match"
    
    def record_decision(self, action: AutonomousAction, approved: bool, 
                       user_reason: Optional[str] = None) -> None:
        """Record user's permission decision for learning.
        
        Stores the decision with contextual information to improve future
        permission checking through pattern recognition and statistical analysis.
        
        Args:
            action (AutonomousAction): The action that was decided upon
            approved (bool): Whether the user approved the action
            user_reason (Optional[str]): Optional reason provided by user
        """
        permission_key = self._build_permission_key(action)
        
        # Get or create permission record
        if permission_key not in self.permissions:
            self.permissions[permission_key] = PermissionRecord()
        
        record = self.permissions[permission_key]
        
        # Build context for learning
        context = {
            'action_type': action.action_type,
            'priority': action.priority.name,
            'confidence': action.confidence,
            'time_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'reasoning': action.reasoning,
            'user_reason': user_reason
        }
        
        # Record the decision
        record.record_decision(approved, context)
        
        # Learn from patterns in the decision
        self._learn_from_decision(action, approved, context)
        
        # Save updated permissions
        self._save_permissions()
        
        logger.info(f"Recorded {'approval' if approved else 'denial'} for {permission_key}")
    
    def _learn_from_decision(self, action: AutonomousAction, approved: bool, context: Dict[str, Any]) -> None:
        """Learn patterns from user decisions to improve future automation.
        
        Args:
            action (AutonomousAction): The action that was decided upon
            approved (bool): Whether the action was approved
            context (Dict[str, Any]): Contextual information about the decision
        """
        # Time-based learning
        if not approved and context['time_of_day'] >= 22:
            # User denies actions late at night - adjust quiet hours
            logger.info("Learning: User prefers fewer actions late at night")
        
        # Priority-based learning
        if action.priority == ActionPriority.LOW and approved:
            # User approves low priority actions - maybe increase automation
            logger.info("Learning: User approves low priority actions")
        
        # Category-based learning
        if action.category == ActionCategory.NOTIFICATION and not approved:
            # User denies notification actions - maybe increase threshold
            logger.info("Learning: User prefers fewer notification actions")
    
    def get_permission_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about permissions and decisions.
        
        Returns:
            Dict[str, Any]: Statistics including:
                - total_decisions: Total number of decisions made
                - unique_actions: Number of unique action types
                - category_stats: Approval/denial counts by category
                - auto_approval_candidates: Actions eligible for auto-approval
                - auto_denial_candidates: Actions eligible for auto-denial
        """
        stats = {
            'total_decisions': sum(r.total_decisions for r in self.permissions.values()),
            'unique_actions': len(self.permissions),
            'category_stats': defaultdict(lambda: {'approved': 0, 'denied': 0}),
            'auto_approval_candidates': [],
            'auto_denial_candidates': []
        }
        
        for key, record in self.permissions.items():
            # Parse category from key
            category = key.split(':')[0]
            stats['category_stats'][category]['approved'] += record.approved_count
            stats['category_stats'][category]['denied'] += record.denied_count
            
            # Find automation candidates
            if record.total_decisions >= self.learning_threshold:
                if record.approval_rate >= self.auto_approve_threshold:
                    stats['auto_approval_candidates'].append({
                        'action': key,
                        'approval_rate': record.approval_rate,
                        'decisions': record.total_decisions
                    })
                elif record.approval_rate <= self.auto_deny_threshold:
                    stats['auto_denial_candidates'].append({
                        'action': key,
                        'denial_rate': 1 - record.approval_rate,
                        'decisions': record.total_decisions
                    })
        
        return dict(stats)
    
    def suggest_automation_rules(self) -> List[Dict[str, Any]]:
        """Suggest new automation rules based on historical patterns.
        
        Analyzes decision history to identify patterns in user behavior
        and suggests automation rules that could reduce manual decisions.
        
        Returns:
            List[Dict[str, Any]]: List of automation suggestions, each containing:
                - type: Type of rule (time_rule, category_rule, etc.)
                - description: Human-readable description
                - confidence: Confidence in the suggestion (0.0-1.0)
                - sample_size: Number of decisions the suggestion is based on
        """
        suggestions = []
        
        # Analyze permission history for patterns
        time_patterns = defaultdict(lambda: {'approved': 0, 'denied': 0})
        category_patterns = defaultdict(lambda: {'approved': 0, 'denied': 0})
        
        for record in self.permissions.values():
            for context in record.context_history:
                hour = context['context'].get('time_of_day', 0)
                category = context['context'].get('category', 'unknown')
                
                if context['approved']:
                    time_patterns[hour]['approved'] += 1
                    category_patterns[category]['approved'] += 1
                else:
                    time_patterns[hour]['denied'] += 1
                    category_patterns[category]['denied'] += 1
        
        # Suggest time-based rules
        for hour, stats in time_patterns.items():
            total = stats['approved'] + stats['denied']
            if total >= 10:  # Enough data
                denial_rate = stats['denied'] / total
                if denial_rate > 0.8:
                    suggestions.append({
                        'type': 'time_rule',
                        'description': f"Block most actions at {hour}:00",
                        'confidence': denial_rate,
                        'sample_size': total
                    })
        
        # Suggest category-based rules
        for category, stats in category_patterns.items():
            total = stats['approved'] + stats['denied']
            if total >= 20:  # Enough data
                approval_rate = stats['approved'] / total
                if approval_rate > 0.9:
                    suggestions.append({
                        'type': 'category_rule',
                        'description': f"Auto-approve {category} actions",
                        'confidence': approval_rate,
                        'sample_size': total
                    })
        
        return suggestions

def test_permission_manager() -> None:
    """Test the permission manager functionality.
    
    Creates a test action, checks permissions, simulates user decisions,
    and displays statistics to verify the system is working correctly.
    """
    manager = PermissionManager()
    
    # Create test action
    from .autonomous_decision_engine import AutonomousAction, ActionCategory, ActionPriority
    
    test_action = AutonomousAction(
        action_type='handle_notifications',
        target='Discord',
        params={'count': 5},
        priority=ActionPriority.MEDIUM,
        confidence=0.8,
        category=ActionCategory.NOTIFICATION,
        reasoning="5 unread messages detected"
    )
    
    # Check permission
    decision, confidence, reason = manager.check_permission(test_action)
    
    print("🔐 Permission Manager Test:")
    print("=" * 50)
    print(f"Action: {test_action.action_type} on {test_action.target}")
    print(f"Decision: {decision}")
    print(f"Confidence: {confidence:.1%}")
    print(f"Reason: {reason}")
    
    # Simulate learning
    if decision is None:
        print("\n🤔 Simulating user approval...")
        manager.record_decision(test_action, True, "Looks good")
        
        # Check again
        decision2, confidence2, reason2 = manager.check_permission(test_action)
        print(f"\nAfter learning:")
        print(f"Decision: {decision2}")
        print(f"Confidence: {confidence2:.1%}")
        print(f"Reason: {reason2}")
    
    # Show stats
    stats = manager.get_permission_stats()
    print(f"\n📊 Permission Statistics:")
    print(f"Total decisions: {stats['total_decisions']}")
    print(f"Unique actions: {stats['unique_actions']}")
    
    # Show suggestions
    suggestions = manager.suggest_automation_rules()
    if suggestions:
        print(f"\n💡 Automation Suggestions:")
        for suggestion in suggestions:
            print(f"- {suggestion['description']} (confidence: {suggestion['confidence']:.1%})")

if __name__ == "__main__":
    test_permission_manager()