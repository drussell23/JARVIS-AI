#!/usr/bin/env python3
"""
UAE Natural Communication Configuration
========================================

Contextual response templates and configuration for UAE natural communication.

This module provides:
- Response templates for different scenarios
- Context-aware message selection
- Personality settings
- Custom communication patterns

Author: Derek J. Russell
Date: October 2025
Version: 1.0.0
"""

from typing import Dict, List, Any
from enum import Enum


class ResponseStyle(Enum):
    """Communication style options"""
    PROFESSIONAL = "professional"  # Formal, technical
    CASUAL = "casual"              # Friendly, conversational
    TECHNICAL = "technical"        # Detailed, technical terms
    MINIMAL = "minimal"            # Brief, to the point


# ============================================================================
# Response Templates by Scenario
# ============================================================================

DECISION_START_TEMPLATES = {
    ResponseStyle.PROFESSIONAL: [
        "Analyzing location for {element}...",
        "Searching for {element}...",
        "Locating {element} position..."
    ],
    ResponseStyle.CASUAL: [
        "Looking for {element}...",
        "Finding {element}...",
        "Let me find {element}..."
    ],
    ResponseStyle.TECHNICAL: [
        "Initiating position detection for {element}...",
        "Running multi-layer awareness scan for {element}...",
        "Executing UAE decision pipeline for {element}..."
    ],
    ResponseStyle.MINIMAL: [
        "Finding {element}...",
        "Locating {element}..."
    ]
}

DECISION_MADE_TEMPLATES = {
    # Context source (learned from history)
    'context': {
        ResponseStyle.PROFESSIONAL: [
            "Found {element} using learned position. Confidence: {confidence}%",
            "Using known location from previous interactions. {confidence}% confident",
            "Retrieved from memory - {confidence}% confidence"
        ],
        ResponseStyle.CASUAL: [
            "Got it! I remember where that is. {confidence}% sure",
            "Found it right where I expected! {confidence}% confidence",
            "I know this one - {confidence}% confident"
        ],
        ResponseStyle.TECHNICAL: [
            "Context Intelligence layer matched. Confidence: {confidence}%",
            "Historical pattern detected at known coordinates. {confidence}% certainty",
            "CI layer decision: position retrieved from knowledge base ({confidence}%)"
        ],
        ResponseStyle.MINIMAL: [
            "Found (known location)",
            "Located - {confidence}%"
        ]
    },

    # Situational source (detected in real-time)
    'situation': {
        ResponseStyle.PROFESSIONAL: [
            "Found {element} through real-time detection. Confidence: {confidence}%",
            "Located using visual analysis. {confidence}% confident",
            "Detected position visually - {confidence}% confidence"
        ],
        ResponseStyle.CASUAL: [
            "Found it by looking around! {confidence}% sure",
            "Spotted it visually - {confidence}% confidence",
            "Just detected it - {confidence}% confident"
        ],
        ResponseStyle.TECHNICAL: [
            "Situational Awareness layer detection complete. Confidence: {confidence}%",
            "Real-time vision analysis successful. {confidence}% certainty",
            "SAI layer decision: position detected via vision system ({confidence}%)"
        ],
        ResponseStyle.MINIMAL: [
            "Found (visual detection)",
            "Detected - {confidence}%"
        ]
    },

    # Fused decision (both context and situation)
    'fused': {
        ResponseStyle.PROFESSIONAL: [
            "Position confirmed using both memory and vision. Confidence: {confidence}%",
            "Verified location through dual-layer analysis. {confidence}% confident",
            "Cross-validated position - {confidence}% confidence"
        ],
        ResponseStyle.CASUAL: [
            "Double-checked it - looks good! {confidence}% sure",
            "Confirmed with both memory and vision. {confidence}% confidence",
            "Verified it's in the right spot - {confidence}% confident"
        ],
        ResponseStyle.TECHNICAL: [
            "Unified decision: CI + SAI fusion complete. Confidence: {confidence}%",
            "Dual-layer awareness integration successful. {confidence}% certainty",
            "UAE fusion: context-situation alignment verified ({confidence}%)"
        ],
        ResponseStyle.MINIMAL: [
            "Found (verified)",
            "Confirmed - {confidence}%"
        ]
    },

    # Position changed (adaptation)
    'position_changed': {
        ResponseStyle.PROFESSIONAL: [
            "Position has changed. Updating memory with new location.",
            "{element} has moved. Recording new position.",
            "Detected position shift - updating knowledge base."
        ],
        ResponseStyle.CASUAL: [
            "Oh, it moved! Updating my memory.",
            "Different spot than last time - noted!",
            "It shifted a bit - I'll remember the new spot."
        ],
        ResponseStyle.TECHNICAL: [
            "Environmental change detected. Updating CI knowledge base.",
            "Position delta detected. Triggering bidirectional learning update.",
            "SAI detected coordinate shift. Persisting new state to context layer."
        ],
        ResponseStyle.MINIMAL: [
            "Position changed - updating",
            "Moved - noted"
        ]
    }
}

EXECUTION_START_TEMPLATES = {
    ResponseStyle.PROFESSIONAL: [
        "Executing {action} on {element}...",
        "Performing {action}...",
        "Initiating {action} operation..."
    ],
    ResponseStyle.CASUAL: [
        "Clicking {element}...",
        "Here we go...",
        "On it..."
    ],
    ResponseStyle.TECHNICAL: [
        "Executing {action} operation at coordinates ({x}, {y})...",
        "Initiating {action} with confidence {confidence}%...",
        "Performing {action} via {method}..."
    ],
    ResponseStyle.MINIMAL: [
        "Executing...",
        "Running..."
    ]
}

EXECUTION_COMPLETE_TEMPLATES = {
    'success': {
        ResponseStyle.PROFESSIONAL: [
            "Operation completed successfully.",
            "{action} completed.",
            "Success - operation finished in {duration}s."
        ],
        ResponseStyle.CASUAL: [
            "Done!",
            "All set!",
            "Success!"
        ],
        ResponseStyle.TECHNICAL: [
            "Execution successful. Duration: {duration}s. Verification: passed.",
            "Operation completed with {confidence}% confidence. Verified: {verified}",
            "{action} execution complete. Performance metrics logged."
        ],
        ResponseStyle.MINIMAL: [
            "Done",
            "Complete"
        ]
    },
    'failed': {
        ResponseStyle.PROFESSIONAL: [
            "Operation failed: {error}",
            "Unable to complete {action}: {error}",
            "Execution unsuccessful - {error}"
        ],
        ResponseStyle.CASUAL: [
            "Hmm, that didn't work: {error}",
            "Couldn't do it - {error}",
            "Failed: {error}"
        ],
        ResponseStyle.TECHNICAL: [
            "Execution failed with error: {error}. Stack trace logged.",
            "Operation terminated abnormally: {error}",
            "Fatal execution error: {error}. Diagnostics available."
        ],
        ResponseStyle.MINIMAL: [
            "Failed: {error}",
            "Error: {error}"
        ]
    }
}

LEARNING_EVENT_TEMPLATES = {
    ResponseStyle.PROFESSIONAL: [
        "Learning from this interaction. Total experience: {count} interactions.",
        "Updating knowledge base. Experience level: {count} samples.",
        "Recording results for future optimization."
    ],
    ResponseStyle.CASUAL: [
        "Getting smarter! {count} interactions learned.",
        "I'm learning - {count} experiences now.",
        "Another one for the memory bank! Total: {count}"
    ],
    ResponseStyle.TECHNICAL: [
        "Bidirectional learning update complete. Dataset: {count} executions.",
        "CI/SAI sync in progress. Historical samples: {count}",
        "Knowledge base update: {count} total training instances."
    ],
    ResponseStyle.MINIMAL: [
        "Learned ({count} total)",
        "Updated"
    ]
}

DEVICE_CONNECTION_TEMPLATES = {
    'start': {
        ResponseStyle.PROFESSIONAL: [
            "Initiating connection to {device}...",
            "Connecting to {device}...",
            "Starting device connection sequence for {device}..."
        ],
        ResponseStyle.CASUAL: [
            "Connecting to {device}...",
            "Let me connect to {device}...",
            "On it - connecting to {device}..."
        ],
        ResponseStyle.TECHNICAL: [
            "Initiating multi-step AirPlay connection to {device}...",
            "Beginning device connection protocol for {device}...",
            "Starting Screen Mirroring connection sequence: target={device}"
        ],
        ResponseStyle.MINIMAL: [
            "Connecting to {device}...",
            "Connecting..."
        ]
    },

    'open_control_center': {
        ResponseStyle.PROFESSIONAL: [
            "Opening Control Center...",
            "Accessing Control Center...",
            "Step 1: Control Center access..."
        ],
        ResponseStyle.CASUAL: [
            "Opening Control Center...",
            "Getting to Control Center...",
            "First, opening Control Center..."
        ],
        ResponseStyle.TECHNICAL: [
            "Step 1/3: Executing Control Center click at menubar position...",
            "Opening Control Center via UAE-guided navigation...",
            "Accessing macOS Control Center interface..."
        ],
        ResponseStyle.MINIMAL: [
            "Control Center...",
            "Step 1..."
        ]
    },

    'open_screen_mirroring': {
        ResponseStyle.PROFESSIONAL: [
            "Opening Screen Mirroring menu...",
            "Accessing Screen Mirroring options...",
            "Step 2: Screen Mirroring access..."
        ],
        ResponseStyle.CASUAL: [
            "Opening Screen Mirroring...",
            "Getting to mirroring options...",
            "Now opening Screen Mirroring..."
        ],
        ResponseStyle.TECHNICAL: [
            "Step 2/3: Navigating to Screen Mirroring submenu...",
            "Accessing AirPlay device selection interface...",
            "Opening Screen Mirroring control panel..."
        ],
        ResponseStyle.MINIMAL: [
            "Screen Mirroring...",
            "Step 2..."
        ]
    },

    'select_device': {
        ResponseStyle.PROFESSIONAL: [
            "Selecting {device}...",
            "Choosing {device} from available devices...",
            "Step 3: Device selection - {device}..."
        ],
        ResponseStyle.CASUAL: [
            "Selecting {device}...",
            "Picking {device}...",
            "Almost there - selecting {device}..."
        ],
        ResponseStyle.TECHNICAL: [
            "Step 3/3: Executing device selection for {device}...",
            "Initiating AirPlay connection handshake with {device}...",
            "Selecting target device from AirPlay mesh: {device}"
        ],
        ResponseStyle.MINIMAL: [
            "Selecting {device}...",
            "Step 3..."
        ]
    },

    'complete': {
        ResponseStyle.PROFESSIONAL: [
            "Successfully connected to {device}.",
            "Connection established with {device}.",
            "Device connection complete - {device} is now active."
        ],
        ResponseStyle.CASUAL: [
            "Connected to {device}!",
            "All done - you're on {device}!",
            "Success! Connected to {device}."
        ],
        ResponseStyle.TECHNICAL: [
            "AirPlay connection established. Target: {device}. Duration: {duration}s",
            "Screen Mirroring active. Device: {device}. Handshake complete.",
            "Connection successful. {device} mirroring enabled ({duration}s)"
        ],
        ResponseStyle.MINIMAL: [
            "Connected to {device}",
            "Done"
        ]
    },

    'error': {
        ResponseStyle.PROFESSIONAL: [
            "Connection failed: {error}",
            "Unable to connect to {device}: {error}",
            "Connection error - {error}"
        ],
        ResponseStyle.CASUAL: [
            "Couldn't connect: {error}",
            "Connection failed - {error}",
            "Hmm, something went wrong: {error}"
        ],
        ResponseStyle.TECHNICAL: [
            "Connection sequence terminated abnormally: {error}",
            "Fatal connection error at step {step}: {error}",
            "AirPlay handshake failed: {error}"
        ],
        ResponseStyle.MINIMAL: [
            "Failed: {error}",
            "Error: {error}"
        ]
    }
}


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_RESPONSE_STYLE = ResponseStyle.CASUAL
DEFAULT_INCLUDE_CONFIDENCE = True
DEFAULT_INCLUDE_METRICS = False
DEFAULT_USE_PERCENTAGE = True  # Show confidence as percentage (95%) vs decimal (0.95)


def get_response_templates(
    scenario: str,
    source: str = None,
    style: ResponseStyle = DEFAULT_RESPONSE_STYLE
) -> List[str]:
    """
    Get response templates for a scenario

    Args:
        scenario: Scenario name (decision_start, decision_made, etc.)
        source: Decision source (context/situation/fused) if applicable
        style: Response style

    Returns:
        List of template strings
    """
    template_map = {
        'decision_start': DECISION_START_TEMPLATES,
        'decision_made': DECISION_MADE_TEMPLATES,
        'execution_start': EXECUTION_START_TEMPLATES,
        'execution_complete': EXECUTION_COMPLETE_TEMPLATES,
        'learning_event': LEARNING_EVENT_TEMPLATES,
        'device_connection': DEVICE_CONNECTION_TEMPLATES
    }

    templates = template_map.get(scenario)
    if not templates:
        return ["Processing..."]

    # Handle nested templates (with source)
    if source and isinstance(templates, dict) and source in templates:
        templates = templates[source]

    # Get style-specific templates
    if isinstance(templates, dict) and style in templates:
        return templates[style]

    # Fallback to casual style
    if isinstance(templates, dict):
        return templates.get(ResponseStyle.CASUAL, ["Processing..."])

    return templates


def format_confidence(confidence: float, as_percentage: bool = DEFAULT_USE_PERCENTAGE) -> str:
    """
    Format confidence value

    Args:
        confidence: Confidence value (0.0 - 1.0)
        as_percentage: Format as percentage

    Returns:
        Formatted confidence string
    """
    if as_percentage:
        return f"{int(confidence * 100)}"
    else:
        return f"{confidence:.2f}"


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import random

    print("\n" + "=" * 80)
    print("UAE Communication Templates - Demo")
    print("=" * 80)

    # Test different scenarios and styles
    scenarios = [
        ('decision_start', None, ResponseStyle.CASUAL),
        ('decision_made', 'context', ResponseStyle.PROFESSIONAL),
        ('decision_made', 'situation', ResponseStyle.TECHNICAL),
        ('decision_made', 'fused', ResponseStyle.CASUAL),
        ('execution_complete', 'success', ResponseStyle.MINIMAL),
        ('device_connection', 'complete', ResponseStyle.CASUAL)
    ]

    for scenario, source, style in scenarios:
        templates = get_response_templates(scenario, source, style)
        template = random.choice(templates)

        print(f"\n{scenario} ({source or 'N/A'}) - {style.value}:")
        print(f"  Template: {template}")
        print(f"  Example:  {template.format(element='Control Center', confidence=95, device='Living Room TV', duration=1.2, error='timeout')}")

    print("\n" + "=" * 80)
    print("✅ Demo complete!")
    print("=" * 80)
