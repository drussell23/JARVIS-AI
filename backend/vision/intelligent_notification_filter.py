#!/usr/bin/env python3
"""
Intelligent Notification Filtering System
Ensures only relevant, important notifications reach the user
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import re

logger = logging.getLogger(__name__)


@dataclass
class NotificationContext:
    """Context for making notification decisions"""
    user_activity: str
    focus_level: float  # 0-1, higher means more focused
    recent_notifications: List[Dict]
    time_of_day: str
    workflow_phase: str
    interaction_history: List[Dict]


class NotificationFilter:
    """
    Intelligent filtering system that learns from user behavior
    and context to deliver only valuable notifications
    """
    
    def __init__(self):
        # Filtering configuration
        self.config = {
            'base_importance_threshold': 0.6,
            'focus_multiplier': 1.5,  # Increase threshold when user is focused
            'cooldown_periods': {
                'high': 30,    # seconds
                'medium': 60,
                'low': 300
            },
            'burst_protection': {
                'window_seconds': 60,
                'max_notifications': 3
            },
            'quiet_hours': [],  # e.g., [(22, 7)] for 10pm-7am
            'contextual_rules_enabled': True
        }
        
        # State tracking
        self.notification_history = deque(maxlen=1000)
        self.category_cooldowns = defaultdict(lambda: datetime.min)
        self.similar_content_cache = {}  # hash -> last_notified
        self.user_responses = defaultdict(list)  # track engagement
        
        # Learning state
        self.importance_adjustments = defaultdict(float)
        self.ignored_patterns = set()
        self.valued_patterns = set()
        self.context_preferences = defaultdict(dict)
        
    def should_notify(self, 
                     change: 'ScreenChange', 
                     context: NotificationContext) -> Tuple[bool, str]:
        """
        Determine if a notification should be sent
        
        Returns:
            (should_notify, reason)
        """
        # Calculate importance score
        importance_score = self._calculate_importance_score(change, context)
        
        # Check all filters
        filters = [
            ('importance', self._check_importance_threshold(importance_score, context)),
            ('cooldown', self._check_cooldown(change)),
            ('burst', self._check_burst_protection()),
            ('duplicate', self._check_duplicate(change)),
            ('context', self._check_context_appropriateness(change, context)),
            ('quiet_hours', self._check_quiet_hours()),
            ('user_preference', self._check_user_preferences(change))
        ]
        
        # Apply filters
        for filter_name, (passed, reason) in filters:
            if not passed:
                logger.debug(f"Notification filtered by {filter_name}: {reason}")
                return False, reason
                
        # All filters passed
        return True, "All checks passed"
        
    def _calculate_importance_score(self, 
                                   change: 'ScreenChange', 
                                   context: NotificationContext) -> float:
        """
        Calculate dynamic importance score based on multiple factors
        
        Score = (Base Priority × Confidence × Context Relevance × Temporal Factor × User Preference)
        """
        # Base priority scores
        priority_scores = {
            'high': 0.9,
            'medium': 0.7,
            'low': 0.5
        }
        
        base_score = priority_scores.get(change.importance.value, 0.5)
        
        # Apply confidence
        score = base_score * change.confidence
        
        # Context relevance multiplier
        context_relevance = self._calculate_context_relevance(change, context)
        score *= context_relevance
        
        # Temporal factor (urgency)
        temporal_factor = self._calculate_temporal_factor(change)
        score *= temporal_factor
        
        # User preference adjustment
        preference_adjustment = self._get_preference_adjustment(change)
        score *= (1.0 + preference_adjustment)
        
        # Focus level adjustment (higher threshold when focused)
        if context.focus_level > 0.7:
            score *= 0.8  # Make it harder to interrupt
            
        logger.debug(f"Importance calculation: base={base_score}, "
                    f"confidence={change.confidence}, "
                    f"relevance={context_relevance}, "
                    f"temporal={temporal_factor}, "
                    f"final={score}")
                    
        return score
        
    def _calculate_context_relevance(self, 
                                    change: 'ScreenChange', 
                                    context: NotificationContext) -> float:
        """Calculate how relevant the change is to current context"""
        relevance = 1.0
        
        # Check if change relates to current activity
        if context.user_activity:
            activity_lower = context.user_activity.lower()
            change_lower = change.description.lower()
            
            # Coding context
            if 'coding' in activity_lower or 'programming' in activity_lower:
                if any(term in change_lower for term in ['error', 'warning', 'failed', 'exception']):
                    relevance = 1.5  # More relevant
                elif any(term in change_lower for term in ['email', 'social', 'news']):
                    relevance = 0.5  # Less relevant
                    
            # Writing context
            elif 'writing' in activity_lower or 'document' in activity_lower:
                if any(term in change_lower for term in ['save', 'spell', 'grammar']):
                    relevance = 1.3
                elif any(term in change_lower for term in ['code', 'terminal', 'debug']):
                    relevance = 0.6
                    
        # Workflow phase adjustments
        if context.workflow_phase == 'deep_work':
            relevance *= 0.7  # Less interruptions during deep work
        elif context.workflow_phase == 'break':
            relevance *= 1.2  # More open to notifications during breaks
            
        return relevance
        
    def _calculate_temporal_factor(self, change: 'ScreenChange') -> float:
        """Calculate urgency based on content"""
        # Keywords indicating time sensitivity
        urgent_keywords = [
            'urgent', 'immediate', 'now', 'asap', 'deadline',
            'expir', 'timeout', 'critical', 'emergency'
        ]
        
        change_lower = change.description.lower()
        if any(keyword in change_lower for keyword in urgent_keywords):
            return 1.3
            
        # Updates are moderately time-sensitive
        if change.category.value == 'update':
            return 1.1
            
        # Errors are usually important
        if change.category.value == 'error':
            return 1.2
            
        return 1.0
        
    def _get_preference_adjustment(self, change: 'ScreenChange') -> float:
        """Get learned preference adjustment"""
        # Check if this type of change has been consistently ignored
        pattern_key = f"{change.category.value}:{change.location}"
        
        if pattern_key in self.ignored_patterns:
            return -0.5  # Reduce importance
            
        if pattern_key in self.valued_patterns:
            return 0.3  # Increase importance
            
        # Check category-level adjustments
        category_adjustment = self.importance_adjustments.get(change.category.value, 0.0)
        
        return category_adjustment
        
    def _check_importance_threshold(self, 
                                   score: float, 
                                   context: NotificationContext) -> Tuple[bool, str]:
        """Check if importance score meets threshold"""
        threshold = self.config['base_importance_threshold']
        
        # Adjust threshold based on focus level
        if context.focus_level > 0.7:
            threshold *= self.config['focus_multiplier']
            
        passed = score >= threshold
        reason = f"Score {score:.2f} {'meets' if passed else 'below'} threshold {threshold:.2f}"
        
        return passed, reason
        
    def _check_cooldown(self, change: 'ScreenChange') -> Tuple[bool, str]:
        """Check category-based cooldown"""
        category = change.category.value
        priority = change.importance.value
        
        last_notified = self.category_cooldowns[category]
        cooldown_seconds = self.config['cooldown_periods'].get(priority, 60)
        
        if datetime.now() - last_notified < timedelta(seconds=cooldown_seconds):
            remaining = cooldown_seconds - (datetime.now() - last_notified).seconds
            return False, f"Category {category} in cooldown for {remaining}s"
            
        return True, "No cooldown active"
        
    def _check_burst_protection(self) -> Tuple[bool, str]:
        """Prevent notification bursts"""
        window = self.config['burst_protection']['window_seconds']
        max_notifications = self.config['burst_protection']['max_notifications']
        
        # Count recent notifications
        cutoff_time = datetime.now() - timedelta(seconds=window)
        recent_count = sum(1 for n in self.notification_history 
                          if n['timestamp'] > cutoff_time)
        
        if recent_count >= max_notifications:
            return False, f"Burst protection: {recent_count} notifications in {window}s"
            
        return True, "Within burst limits"
        
    def _check_duplicate(self, change: 'ScreenChange') -> Tuple[bool, str]:
        """Check for duplicate notifications"""
        # Create content hash
        content_hash = hash(f"{change.category.value}:{change.description}")
        
        if content_hash in self.similar_content_cache:
            last_notified = self.similar_content_cache[content_hash]
            if datetime.now() - last_notified < timedelta(minutes=5):
                return False, "Similar notification sent recently"
                
        return True, "No duplicate found"
        
    def _check_context_appropriateness(self, 
                                      change: 'ScreenChange', 
                                      context: NotificationContext) -> Tuple[bool, str]:
        """Check if notification is appropriate for current context"""
        if not self.config['contextual_rules_enabled']:
            return True, "Contextual rules disabled"
            
        # Don't interrupt deep focus unless critical
        if context.focus_level > 0.8 and change.importance.value != 'high':
            return False, "User in deep focus"
            
        # Check workflow-specific rules
        if context.workflow_phase == 'meeting':
            # Only urgent notifications during meetings
            if change.importance.value != 'high':
                return False, "User in meeting"
                
        return True, "Context appropriate"
        
    def _check_quiet_hours(self) -> Tuple[bool, str]:
        """Check if current time is in quiet hours"""
        if not self.config['quiet_hours']:
            return True, "No quiet hours configured"
            
        current_hour = datetime.now().hour
        
        for start_hour, end_hour in self.config['quiet_hours']:
            if start_hour <= current_hour < end_hour:
                return False, f"Quiet hours ({start_hour}-{end_hour})"
                
        return True, "Not in quiet hours"
        
    def _check_user_preferences(self, change: 'ScreenChange') -> Tuple[bool, str]:
        """Check learned user preferences"""
        # Check if user has consistently ignored this type
        ignore_count = sum(1 for response in self.user_responses[change.category.value]
                          if response == 'ignored')
        
        if ignore_count > 5:
            return False, "User consistently ignores this type"
            
        return True, "No negative preference pattern"
        
    def record_notification(self, change: 'ScreenChange', context: NotificationContext):
        """Record that a notification was sent"""
        self.notification_history.append({
            'change': change,
            'context': context,
            'timestamp': datetime.now()
        })
        
        # Update cooldowns
        self.category_cooldowns[change.category.value] = datetime.now()
        
        # Update duplicate cache
        content_hash = hash(f"{change.category.value}:{change.description}")
        self.similar_content_cache[content_hash] = datetime.now()
        
    def record_user_response(self, notification_id: str, response: str):
        """Record user response to notification for learning"""
        # Find the notification
        for notif in self.notification_history:
            if id(notif) == notification_id:  # Simple ID system
                category = notif['change'].category.value
                self.user_responses[category].append(response)
                
                # Update learning state
                if response == 'ignored':
                    self.importance_adjustments[category] -= 0.1
                elif response == 'engaged':
                    self.importance_adjustments[category] += 0.1
                    
                # Update pattern sets
                pattern_key = f"{category}:{notif['change'].location}"
                if response == 'ignored' and self.user_responses[category].count('ignored') > 3:
                    self.ignored_patterns.add(pattern_key)
                elif response == 'engaged' and self.user_responses[category].count('engaged') > 2:
                    self.valued_patterns.add(pattern_key)
                    
                break
                
    def get_filter_stats(self) -> Dict[str, Any]:
        """Get filtering statistics"""
        return {
            'total_notifications': len(self.notification_history),
            'category_cooldowns': dict(self.category_cooldowns),
            'importance_adjustments': dict(self.importance_adjustments),
            'ignored_patterns': len(self.ignored_patterns),
            'valued_patterns': len(self.valued_patterns),
            'recent_burst_count': sum(1 for n in self.notification_history 
                                    if n['timestamp'] > datetime.now() - timedelta(minutes=1))
        }
        
    def update_config(self, config_updates: Dict[str, Any]):
        """Update filter configuration"""
        self.config.update(config_updates)
        logger.info(f"Filter configuration updated: {config_updates}")
        
    def reset_learning(self):
        """Reset learned patterns"""
        self.importance_adjustments.clear()
        self.ignored_patterns.clear()
        self.valued_patterns.clear()
        self.user_responses.clear()
        logger.info("Filter learning state reset")