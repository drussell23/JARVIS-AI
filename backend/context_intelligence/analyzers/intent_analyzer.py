"""
Intent Analyzer for JARVIS Context Intelligence
==============================================

Analyzes command intent to determine context requirements
"""

import logging
import re
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Types of command intents"""
    SCREEN_CONTROL = "screen_control"      # Lock/unlock screen
    APP_LAUNCH = "app_launch"              # Open applications
    WEB_BROWSE = "web_browse"              # Browse websites
    FILE_OPERATION = "file_operation"      # File operations
    DOCUMENT_CREATION = "document_creation"  # Create documents/essays
    SYSTEM_QUERY = "system_query"          # System info queries
    TIME_WEATHER = "time_weather"          # Time/weather queries
    PREDICTIVE_QUERY = "predictive_query"  # Predictive/analytical queries
    GENERAL_CHAT = "general_chat"          # General conversation
    UNKNOWN = "unknown"                    # Unknown intent


@dataclass
class Intent:
    """Represents analyzed command intent"""
    type: IntentType
    confidence: float
    entities: Dict[str, Any]
    requires_screen: bool
    original_command: str
    metadata: Dict[str, Any] = None


class IntentAnalyzer:
    """Analyzes command intent for context intelligence"""
    
    def __init__(self):
        self.intent_patterns = self._initialize_patterns()
        
    def _initialize_patterns(self) -> Dict[IntentType, List[re.Pattern]]:
        """Initialize intent patterns"""
        return {
            IntentType.SCREEN_CONTROL: [
                re.compile(r'\b(lock|unlock)\s+(my\s+)?(screen|computer|mac)\b', re.I),
                re.compile(r'\bscreen\s+(lock|unlock)\b', re.I),
            ],
            IntentType.APP_LAUNCH: [
                re.compile(r'\b(open|launch|start|run)\s+(\w+)', re.I),
                re.compile(r'\b(switch|go)\s+to\s+(\w+)', re.I),
            ],
            IntentType.WEB_BROWSE: [
                re.compile(r'\b(open|search|google|look up|find online)\b.*\b(safari|chrome|firefox|browser)\b', re.I),
                re.compile(r'\b(go to|navigate to|visit)\s+(.*\.com|.*\.org|website)', re.I),
                re.compile(r'\bsearch\s+for\s+(.*)', re.I),
            ],
            IntentType.FILE_OPERATION: [
                re.compile(r'\b(create|edit|save|close|open)\s+(file|document|folder)', re.I),
                re.compile(r'\bfind\s+file\s+(.*)', re.I),
            ],
            IntentType.DOCUMENT_CREATION: [
                re.compile(r'\b(write|create|compose|draft|generate)\s+(me\s+)?(an?\s+)?(essay|document|report|paper|article|blog\s+post)', re.I),
                re.compile(r'\bwrite\s+(me\s+)?(\d+\s+)?(word|page)s?\s+(essay|document|report|paper|article)', re.I),
                re.compile(r'\b(help\s+me\s+write|can\s+you\s+write)', re.I),
                re.compile(r'\bcreate\s+a\s+(.*?)\s+(document|essay|report)', re.I),
            ],
            IntentType.SYSTEM_QUERY: [
                re.compile(r'\b(show|display|what|check)\s+(system|memory|cpu|disk)', re.I),
                re.compile(r'\bhow\s+much\s+(memory|storage|cpu)', re.I),
            ],
            IntentType.TIME_WEATHER: [
                re.compile(r'\b(what|tell|show)\s+(time|weather|temperature)', re.I),
                re.compile(r'\bwhat\s+time\s+is\s+it\b', re.I),
                re.compile(r'\bhow\'s\s+the\s+weather\b', re.I),
            ],
            IntentType.PREDICTIVE_QUERY: [
                # Progress checks
                re.compile(r'\b(am i|are we)\s+(making|seeing)\s+progress\b', re.I),
                re.compile(r'\bhow\s+(much|far|well)\s+(progress|am i doing)\b', re.I),
                re.compile(r'\bwhat\'?s\s+my\s+progress\b', re.I),
                # Next steps
                re.compile(r'\bwhat\s+should\s+i\s+(do|work on)\s+next\b', re.I),
                re.compile(r'\b(next\s+steps|what\'?s\s+next)\b', re.I),
                re.compile(r'\bwhat\s+to\s+do\s+next\b', re.I),
                # Bug detection
                re.compile(r'\b(are there|any|find)\s+(any\s+)?(bugs|errors|issues|problems)\b', re.I),
                re.compile(r'\bpotential\s+(bugs|issues)\b', re.I),
                re.compile(r'\bwhat\'?s\s+wrong\b', re.I),
                # Code explanation
                re.compile(r'\bexplain\s+(this|that|the)\s+code\b', re.I),
                re.compile(r'\bwhat\s+does\s+(this|that|the)\s+code\s+do\b', re.I),
                re.compile(r'\bhow\s+does\s+(this|that)\s+work\b', re.I),
                # Pattern analysis
                re.compile(r'\bwhat\s+patterns?\s+do\s+you\s+see\b', re.I),
                re.compile(r'\banalyze\s+(the\s+)?patterns?\b', re.I),
                # Workflow optimization
                re.compile(r'\bhow\s+can\s+i\s+improve\s+my\s+workflow\b', re.I),
                re.compile(r'\boptimize\s+my\s+workflow\b', re.I),
                re.compile(r'\bwork\s+more\s+efficiently\b', re.I),
                # Quality assessment
                re.compile(r'\bhow\'?s\s+my\s+code\s+quality\b', re.I),
                re.compile(r'\bcode\s+quality\s+(assessment|check)\b', re.I),
            ],
        }
        
    async def analyze(self, command: str, context: Dict[str, Any] = None) -> Intent:
        """
        Analyze command to determine intent
        
        Args:
            command: The command to analyze
            context: Additional context information
            
        Returns:
            Intent object with analysis results
        """
        command_lower = command.lower()
        
        # Try to match intent patterns
        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = pattern.search(command)
                if match:
                    entities = self._extract_entities(match, command)
                    requires_screen = self._requires_screen(intent_type, command_lower)
                    
                    return Intent(
                        type=intent_type,
                        confidence=0.9,
                        entities=entities,
                        requires_screen=requires_screen,
                        original_command=command,
                        metadata={"pattern": pattern.pattern}
                    )
        
        # Default to general chat if no pattern matches
        return Intent(
            type=IntentType.GENERAL_CHAT,
            confidence=0.5,
            entities={},
            requires_screen=self._requires_screen_fallback(command_lower),
            original_command=command,
            metadata={}
        )
        
    def _extract_entities(self, match: re.Match, command: str) -> Dict[str, Any]:
        """Extract entities from command"""
        entities = {}
        
        # Extract matched groups
        groups = match.groups()
        if groups:
            if len(groups) >= 1:
                entities["action"] = groups[0]
            if len(groups) >= 2:
                entities["target"] = groups[1]
                
        # Extract search terms
        if "search for" in command.lower():
            search_match = re.search(r'search\s+for\s+(.*?)(?:\s+in\s+|\s+on\s+|$)', command, re.I)
            if search_match:
                entities["search_term"] = search_match.group(1)
                
        return entities
        
    def _requires_screen(self, intent_type: IntentType, command: str) -> bool:
        """Determine if intent requires screen access"""
        # Screen control commands that lock don't need screen
        if intent_type == IntentType.SCREEN_CONTROL and "lock" in command:
            return False

        # Predictive queries may need screen for visual analysis
        if intent_type == IntentType.PREDICTIVE_QUERY:
            # Only need screen if asking to explain visible code
            visual_keywords = ["explain", "code", "this", "that", "see", "screen"]
            return any(keyword in command.lower() for keyword in visual_keywords)

        # These intents typically require screen
        screen_required_intents = {
            IntentType.APP_LAUNCH,
            IntentType.WEB_BROWSE,
            IntentType.FILE_OPERATION,
            IntentType.DOCUMENT_CREATION,
        }

        return intent_type in screen_required_intents
        
    def _requires_screen_fallback(self, command: str) -> bool:
        """Fallback check for screen requirement"""
        # Commands that typically require screen
        screen_keywords = [
            'open', 'launch', 'start', 'show', 'display',
            'create', 'edit', 'save', 'close',
            'search', 'google', 'browse', 'navigate',
            'click', 'type', 'scroll', 'move',
            'window', 'tab', 'desktop'
        ]
        
        # Exceptions that don't need screen
        exceptions = ['lock screen', 'lock my screen', 'what time', "what's the time", 'weather']
        
        # Check exceptions first
        for exception in exceptions:
            if exception in command:
                return False
                
        # Check if any screen keywords present
        for keyword in screen_keywords:
            if keyword in command:
                return True
                
        return False
        
    def validate_intent(self, intent: Intent) -> Tuple[bool, Optional[str]]:
        """
        Validate an intent
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if intent.confidence < 0.3:
            return False, "Low confidence in intent analysis"
            
        if intent.type == IntentType.UNKNOWN and intent.requires_screen:
            return True, None  # Allow unknown intents that might need screen
            
        return True, None


# Global instance
_analyzer = None

def get_intent_analyzer() -> IntentAnalyzer:
    """Get or create intent analyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = IntentAnalyzer()
    return _analyzer