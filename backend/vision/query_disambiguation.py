"""
Query Disambiguation for Multi-Monitor Support
===============================================

Handles ambiguous monitor/display references in user queries with
intelligent resolution and clarification strategies.

Features:
- Resolve ordinal references ("second monitor", "3rd display")
- Handle positional references ("left monitor", "right screen")
- Detect primary/main display requests
- Ask clarifying questions when ambiguous
- Support natural language variations

Author: Derek Russell
Date: 2025-10-14
"""

import logging
from typing import List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MonitorReference:
    """Resolved monitor reference"""
    display_id: int
    confidence: float
    resolution_method: str
    ambiguous: bool = False


class QueryDisambiguator:
    """
    Intelligent disambiguation for multi-monitor queries
    
    Handles natural language references to monitors and provides
    clarification when queries are ambiguous.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def resolve_monitor_reference(
        self, 
        query: str, 
        available_displays: List[Any]  # List[DisplayInfo]
    ) -> Optional[MonitorReference]:
        """
        Resolve ambiguous monitor references to specific display_id
        
        Supports:
        - Ordinals: "second monitor", "third display", "2nd screen"
        - Primary: "primary monitor", "main display", "main screen"
        - Positional: "left monitor", "right monitor"
        - Numeric: "monitor 1", "display 2", "screen 3"
        
        Args:
            query: User query string
            available_displays: List of DisplayInfo objects
            
        Returns:
            MonitorReference with resolved display_id and confidence
        """
        query_lower = query.lower()
        
        # Primary/main/first display (highest priority)
        if any(keyword in query_lower for keyword in ["primary", "main", "first", "monitor 1"]):
            for display in available_displays:
                if display.is_primary:
                    self.logger.info(f"Resolved 'primary' to display {display.display_id}")
                    return MonitorReference(
                        display_id=display.display_id,
                        confidence=1.0,
                        resolution_method="primary_keyword",
                        ambiguous=False
                    )
            # If no primary found, return first display
            if available_displays:
                return MonitorReference(
                    display_id=available_displays[0].display_id,
                    confidence=0.8,
                    resolution_method="first_display_fallback",
                    ambiguous=False
                )
        
        # Ordinal references (second, third, etc.)
        ordinals = {
            "second": 1, "2nd": 1, "monitor 2": 1, "display 2": 1, "screen 2": 1,
            "third": 2, "3rd": 2, "monitor 3": 2, "display 3": 2, "screen 3": 2,
            "fourth": 3, "4th": 3, "monitor 4": 3, "display 4": 3, "screen 4": 3,
            "fifth": 4, "5th": 4, "monitor 5": 4, "display 5": 4, "screen 5": 4
        }
        
        for ordinal, index in ordinals.items():
            if ordinal in query_lower:
                if index < len(available_displays):
                    display_id = available_displays[index].display_id
                    self.logger.info(f"Resolved '{ordinal}' to display {display_id}")
                    return MonitorReference(
                        display_id=display_id,
                        confidence=1.0,
                        resolution_method=f"ordinal_{ordinal}",
                        ambiguous=False
                    )
                else:
                    self.logger.warning(f"'{ordinal}' requested but only {len(available_displays)} displays available")
                    return None
        
        # Positional references (left, right, top, bottom)
        if "left" in query_lower or "leftmost" in query_lower:
            # Find leftmost display (lowest x position)
            if available_displays:
                leftmost = min(available_displays, key=lambda d: d.position[0])
                self.logger.info(f"Resolved 'left' to display {leftmost.display_id} at position {leftmost.position}")
                return MonitorReference(
                    display_id=leftmost.display_id,
                    confidence=0.9,
                    resolution_method="positional_left",
                    ambiguous=False
                )
        
        if "right" in query_lower or "rightmost" in query_lower:
            # Find rightmost display (highest x position)
            if available_displays:
                rightmost = max(available_displays, key=lambda d: d.position[0])
                self.logger.info(f"Resolved 'right' to display {rightmost.display_id} at position {rightmost.position}")
                return MonitorReference(
                    display_id=rightmost.display_id,
                    confidence=0.9,
                    resolution_method="positional_right",
                    ambiguous=False
                )
        
        # Generic "monitor" or "display" without qualifier = ambiguous
        if any(keyword in query_lower for keyword in ["monitor", "display", "screen"]):
            self.logger.info("Ambiguous monitor reference detected, clarification needed")
            return MonitorReference(
                display_id=available_displays[0].display_id if available_displays else 0,
                confidence=0.3,
                resolution_method="ambiguous",
                ambiguous=True
            )
        
        # Could not resolve
        self.logger.info("Could not resolve monitor reference from query")
        return None
    
    async def ask_clarification(
        self, 
        query: str, 
        available_displays: List[Any]
    ) -> str:
        """
        Generate clarification question for ambiguous queries
        
        Args:
            query: Original user query
            available_displays: List of DisplayInfo objects
            
        Returns:
            Natural language clarification question
        """
        if len(available_displays) == 0:
            return "Sir, I cannot detect any displays."
        
        if len(available_displays) == 1:
            return "Sir, you have only one display connected."
        
        # Build display descriptions
        display_descriptions = []
        for i, display in enumerate(available_displays):
            if display.is_primary:
                position = "Primary"
            else:
                position = f"Monitor {i+1}"
            
            resolution = f"{display.resolution[0]}x{display.resolution[1]}"
            
            # Add position hint if available
            if display.position[0] < 0:
                position_hint = " (left)"
            elif display.position[0] > 0:
                position_hint = " (right)"
            else:
                position_hint = ""
            
            display_descriptions.append(f"{position} ({resolution}{position_hint})")
        
        return (f"Sir, I see {len(available_displays)} displays: " +
                ", ".join(display_descriptions) + 
                ". Which one would you like me to analyze?")
    
    def extract_all_monitor_keywords(self, query: str) -> List[str]:
        """Extract all monitor-related keywords from query for logging/analysis"""
        query_lower = query.lower()
        keywords = []
        
        monitor_keywords = [
            "monitor", "display", "screen", "primary", "main", "second", "third",
            "left", "right", "1", "2", "3", "4", "5"
        ]
        
        for keyword in monitor_keywords:
            if keyword in query_lower:
                keywords.append(keyword)
        
        return keywords


# Singleton instance
_disambiguator = None

def get_query_disambiguator() -> QueryDisambiguator:
    """Get singleton QueryDisambiguator instance"""
    global _disambiguator
    if _disambiguator is None:
        _disambiguator = QueryDisambiguator()
    return _disambiguator
