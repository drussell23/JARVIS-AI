#!/usr/bin/env python3
"""
Activity Reporting Command Patterns for JARVIS
Maps natural language requests to proactive monitoring actions
"""

ACTIVITY_REPORTING_PATTERNS = [
    # Direct reporting requests
    "report any changes",
    "report changes",
    "report activities",
    "report any specific changes",
    "tell me what changes",
    "tell me about changes",
    "notify me of changes",
    "let me know what changes",
    "inform me of changes",
    
    # Monitoring requests
    "watch for changes",
    "monitor changes",
    "monitor activities", 
    "track changes",
    "observe changes",
    "keep an eye on",
    "watch my workspace",
    "monitor my workspace",
    
    # Activity queries
    "what's happening",
    "what changed",
    "any changes",
    "show me changes",
    "workspace activity",
    "desktop activity",
    
    # Confirmation responses
    "yes report",
    "yes notify",
    "yes tell me",
    "yes monitor",
    "yes watch",
    
    # Workspace insights
    "workspace insights",
    "give me insights",
    "analyze my workspace",
    "workspace summary"
]

def is_activity_reporting_command(command: str) -> bool:
    """Check if command is requesting activity reporting"""
    command_lower = command.lower()
    return any(pattern in command_lower for pattern in ACTIVITY_REPORTING_PATTERNS)