"""
Monitoring Command Classifier - Distinguishes between vision queries and monitoring control commands
Part of Screen Monitoring Activation & macOS Purple Indicator System
"""

import re
from typing import Dict, Any, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CommandType(Enum):
    """Types of commands JARVIS can receive"""
    MONITORING_CONTROL = "monitoring_control"  # Start/stop monitoring
    VISION_QUERY = "vision_query"             # Questions about what's on screen
    MONITORING_STATUS = "monitoring_status"    # Queries about monitoring state
    AMBIGUOUS = "ambiguous"                    # Unclear intent
    OTHER = "other"                            # Not vision/monitoring related


class MonitoringAction(Enum):
    """Specific monitoring control actions"""
    START = "start"
    STOP = "stop"
    STATUS = "status"
    NONE = "none"


class MonitoringCommandClassifier:
    """
    Intelligent command classifier that distinguishes between:
    - Monitoring control commands (start/stop monitoring)
    - Vision queries (what do you see?)
    - Status queries (is monitoring active?)
    """
    
    def __init__(self):
        # Monitoring control patterns
        self.monitoring_start_patterns = [
            r'\b(start|begin|enable|activate|turn on|initiate)\b.*\b(monitoring|monitor|screen recording|watching)\b',
            r'\bmonitor\s+(my\s+)?(screen|desktop|display)\b',
            r'\b(watch|observe|track)\s+(my\s+)?(screen|desktop|activity)\b',
            r'\benable\s+screen\s+monitoring\b',
            r'\bactivate\s+monitoring\b',
            r'\bstart\s+watching\b',
            r'\bscreen\s+monitoring\s+on\b',
        ]
        
        self.monitoring_stop_patterns = [
            r'\b(stop|end|disable|deactivate|turn off|cease)\b.*\b(monitoring|monitor|watching)\b',
            r'\bstop\s+watching\b',
            r'\bmonitoring\s+off\b',
            r'\bdisable\s+screen\s+recording\b',
            r'\bdeactivate\s+monitoring\b',
        ]
        
        # Vision query patterns
        self.vision_query_patterns = [
            r'\bwhat\s+(do\s+you\s+)?see\b',
            r'\bwhat(\'s|\s+is)\s+on\s+(my\s+)?(screen|display)\b',
            r'\bdescribe\s+(what|the)\b',
            r'\banalyze\s+(this|the|my)\s+screen\b',
            r'\btell\s+me\s+what\b',
            r'\bcan\s+you\s+see\b',
            r'\bwhat\s+are\s+you\s+looking\s+at\b',
        ]
        
        # Status query patterns
        self.status_query_patterns = [
            r'\bis\s+monitoring\s+(active|on|enabled)\b',
            r'\bare\s+you\s+monitoring\b',
            r'\bmonitoring\s+status\b',
            r'\bis\s+screen\s+recording\s+on\b',
            r'\bcheck\s+monitoring\b',
        ]
        
    def classify_command(self, command: str) -> Tuple[CommandType, MonitoringAction, float]:
        """
        Classify a command and return its type, action, and confidence
        
        Returns:
            Tuple of (CommandType, MonitoringAction, confidence_score)
        """
        command_lower = command.lower().strip()
        
        # IMPORTANT: Exclude lock/unlock screen commands - these are system commands
        if 'lock' in command_lower and 'screen' in command_lower:
            return CommandType.OTHER, MonitoringAction.NONE, 0.0
        if 'unlock' in command_lower and 'screen' in command_lower:
            return CommandType.OTHER, MonitoringAction.NONE, 0.0
        
        # Check for monitoring control commands first (highest priority)
        for pattern in self.monitoring_start_patterns:
            if re.search(pattern, command_lower):
                logger.info(f"Classified as MONITORING START: '{command}'")
                return CommandType.MONITORING_CONTROL, MonitoringAction.START, 0.95
                
        for pattern in self.monitoring_stop_patterns:
            if re.search(pattern, command_lower):
                logger.info(f"Classified as MONITORING STOP: '{command}'")
                return CommandType.MONITORING_CONTROL, MonitoringAction.STOP, 0.95
        
        # Check for status queries
        for pattern in self.status_query_patterns:
            if re.search(pattern, command_lower):
                logger.info(f"Classified as MONITORING STATUS: '{command}'")
                return CommandType.MONITORING_STATUS, MonitoringAction.STATUS, 0.90
        
        # Check for vision queries
        for pattern in self.vision_query_patterns:
            if re.search(pattern, command_lower):
                logger.info(f"Classified as VISION QUERY: '{command}'")
                return CommandType.VISION_QUERY, MonitoringAction.NONE, 0.85
        
        # Heuristic checks for ambiguous cases
        monitoring_keywords = ['monitor', 'monitoring', 'watch', 'screen', 'recording']
        vision_keywords = ['see', 'look', 'view', 'show', 'describe', 'what']
        
        monitoring_score = sum(1 for keyword in monitoring_keywords if keyword in command_lower)
        vision_score = sum(1 for keyword in vision_keywords if keyword in command_lower)
        
        if monitoring_score > vision_score:
            logger.info(f"Heuristic classified as MONITORING: '{command}'")
            return CommandType.MONITORING_CONTROL, MonitoringAction.START, 0.60
        elif vision_score > monitoring_score:
            logger.info(f"Heuristic classified as VISION: '{command}'")
            return CommandType.VISION_QUERY, MonitoringAction.NONE, 0.60
        else:
            logger.info(f"Classified as AMBIGUOUS: '{command}'")
            return CommandType.AMBIGUOUS, MonitoringAction.NONE, 0.30
    
    def get_command_context(self, command: str, current_monitoring_state: bool) -> Dict[str, Any]:
        """
        Get additional context for command processing
        
        Args:
            command: The command text
            current_monitoring_state: Whether monitoring is currently active
            
        Returns:
            Dictionary with command context information
        """
        cmd_type, action, confidence = self.classify_command(command)
        
        context = {
            'command': command,
            'type': cmd_type,
            'action': action,
            'confidence': confidence,
            'current_monitoring_state': current_monitoring_state,
            'requires_macos_integration': cmd_type == CommandType.MONITORING_CONTROL,
            'requires_vision_analysis': cmd_type == CommandType.VISION_QUERY,
            'is_state_query': cmd_type == CommandType.MONITORING_STATUS,
        }
        
        # Add appropriate response hints
        if cmd_type == CommandType.MONITORING_CONTROL:
            if action == MonitoringAction.START:
                if current_monitoring_state:
                    context['response_hint'] = 'already_monitoring'
                else:
                    context['response_hint'] = 'start_monitoring'
            elif action == MonitoringAction.STOP:
                if current_monitoring_state:
                    context['response_hint'] = 'stop_monitoring'
                else:
                    context['response_hint'] = 'not_monitoring'
        
        return context


# Global instance
_classifier = None


def get_monitoring_classifier():
    """Get or create the global classifier instance"""
    global _classifier
    if _classifier is None:
        _classifier = MonitoringCommandClassifier()
    return _classifier


def classify_monitoring_command(command: str, current_state: bool = False) -> Dict[str, Any]:
    """
    Convenience function to classify a command
    
    Args:
        command: The command text
        current_state: Current monitoring state
        
    Returns:
        Command context dictionary
    """
    classifier = get_monitoring_classifier()
    return classifier.get_command_context(command, current_state)