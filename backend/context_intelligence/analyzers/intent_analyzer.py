"""
Intent Analyzer for JARVIS Context Intelligence
==============================================

This module provides intent analysis capabilities for the JARVIS context intelligence system.
It analyzes user commands to determine their intent, extract relevant entities, and assess
whether screen access is required for execution.

The analyzer uses pattern matching with regular expressions to classify commands into
different intent types such as screen control, app launching, web browsing, file operations,
document creation, system queries, and more.

Example:
    >>> analyzer = get_intent_analyzer()
    >>> intent = await analyzer.analyze("open Safari")
    >>> print(intent.type)
    IntentType.APP_LAUNCH
"""

import logging
import re
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Types of command intents that can be recognized by the analyzer.
    
    Each intent type represents a different category of user command that may
    require different handling and context requirements.
    """
    SCREEN_CONTROL = "screen_control"      # Lock/unlock screen
    APP_LAUNCH = "app_launch"              # Open applications
    WEB_BROWSE = "web_browse"              # Browse websites
    FILE_OPERATION = "file_operation"      # File operations
    DOCUMENT_CREATION = "document_creation"  # Create documents/essays
    SYSTEM_QUERY = "system_query"          # System info queries
    TIME_WEATHER = "time_weather"          # Time/weather queries
    PREDICTIVE_QUERY = "predictive_query"  # Predictive/analytical queries
    ACTION_QUERY = "action_query"          # Action-oriented queries (switch, close, fix, run)
    GENERAL_CHAT = "general_chat"          # General conversation
    UNKNOWN = "unknown"                    # Unknown intent


@dataclass
class Intent:
    """Represents the analyzed intent of a user command.
    
    This dataclass encapsulates all the information extracted from analyzing
    a user command, including the intent type, confidence level, extracted
    entities, and metadata.
    
    Attributes:
        type: The classified intent type
        confidence: Confidence score between 0.0 and 1.0
        entities: Dictionary of extracted entities from the command
        requires_screen: Whether this intent requires screen access
        original_command: The original user command that was analyzed
        metadata: Additional metadata about the analysis process
    """
    type: IntentType
    confidence: float
    entities: Dict[str, Any]
    requires_screen: bool
    original_command: str
    metadata: Dict[str, Any] = None


class IntentAnalyzer:
    """Analyzes command intent for context intelligence.
    
    This class provides the core functionality for analyzing user commands
    to determine their intent. It uses pattern matching with regular expressions
    to classify commands and extract relevant entities.
    
    Attributes:
        intent_patterns: Dictionary mapping intent types to regex patterns
    """
    
    def __init__(self):
        """Initialize the intent analyzer with predefined patterns."""
        self.intent_patterns = self._initialize_patterns()
        
    def _initialize_patterns(self) -> Dict[IntentType, List[re.Pattern]]:
        """Initialize regex patterns for each intent type.
        
        Creates a comprehensive set of regular expression patterns that are used
        to match user commands to specific intent types. Each intent type has
        multiple patterns to handle variations in how users might express the
        same intent.
        
        Returns:
            Dictionary mapping IntentType to list of compiled regex patterns
        """
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
            IntentType.ACTION_QUERY: [
                # Space/window actions
                re.compile(r'\b(switch|go|move|jump)\s+(to\s+)?space\s+\d+\b', re.I),
                re.compile(r'\b(close|quit|kill)\s+(the\s+)?(\w+|it|that|this)', re.I),
                re.compile(r'\bmove\s+\w+\s+to\s+space\s+\d+', re.I),
                re.compile(r'\b(focus|show)\s+(the\s+)?(\w+|it|that)', re.I),
                # Fix/solve actions
                re.compile(r'\bfix\s+(the\s+)?(error|bug|issue|problem|it|that)', re.I),
                re.compile(r'\bsolve\s+(the\s+)?(error|issue|problem|it|that)', re.I),
                # Run/execute actions
                re.compile(r'\b(run|execute)\s+(the\s+)?(tests|test suite|build)', re.I),
                re.compile(r'\btest\s+(it|this|that)', re.I),
                re.compile(r'\bbuild\s+(it|this|the project)', re.I),
                # App control
                re.compile(r'\b(launch|start|open|restart)\s+\w+', re.I),
                # URL opening
                re.compile(r'\bopen\s+(https?://|[\w-]+\.(com|org|net))', re.I),
            ],
        }
        
    async def analyze(self, command: str, context: Dict[str, Any] = None) -> Intent:
        """Analyze a command to determine its intent.
        
        This is the main method for intent analysis. It processes the input command
        through pattern matching to classify the intent type, extract entities,
        and determine context requirements.
        
        Args:
            command: The user command to analyze
            context: Optional additional context information that may influence analysis
            
        Returns:
            Intent object containing the analysis results including type, confidence,
            entities, and screen access requirements
            
        Example:
            >>> analyzer = IntentAnalyzer()
            >>> intent = await analyzer.analyze("open Safari")
            >>> print(f"Intent: {intent.type}, Confidence: {intent.confidence}")
            Intent: IntentType.APP_LAUNCH, Confidence: 0.9
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
        """Extract entities from a regex match and command.
        
        Processes the regex match groups and command text to extract relevant
        entities such as actions, targets, and search terms that provide
        additional context about the user's intent.
        
        Args:
            match: The regex match object from pattern matching
            command: The original command string
            
        Returns:
            Dictionary containing extracted entities with keys like 'action',
            'target', and 'search_term'
        """
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
        """Determine if an intent type requires screen access.
        
        Analyzes the intent type and command content to determine whether
        executing this command would require access to the user's screen.
        This is important for context intelligence to know when screen
        capture or visual analysis might be needed.
        
        Args:
            intent_type: The classified intent type
            command: The original command string (lowercased)
            
        Returns:
            True if screen access is required, False otherwise
        """
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
        """Fallback method to determine screen requirement for unclassified commands.
        
        When a command doesn't match any specific intent patterns, this method
        uses keyword-based heuristics to determine if screen access might be
        needed. It checks for common action words and exceptions.
        
        Args:
            command: The command string (lowercased)
            
        Returns:
            True if screen access is likely required, False otherwise
        """
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
        """Validate an analyzed intent for correctness and completeness.
        
        Performs validation checks on an Intent object to ensure it meets
        quality standards and contains valid data. This can be used to
        filter out low-confidence or problematic intent analyses.
        
        Args:
            intent: The Intent object to validate
            
        Returns:
            Tuple containing:
                - bool: True if intent is valid, False otherwise
                - Optional[str]: Error message if invalid, None if valid
                
        Example:
            >>> intent = Intent(type=IntentType.APP_LAUNCH, confidence=0.2, ...)
            >>> is_valid, error = analyzer.validate_intent(intent)
            >>> print(f"Valid: {is_valid}, Error: {error}")
            Valid: False, Error: Low confidence in intent analysis
        """
        if intent.confidence < 0.3:
            return False, "Low confidence in intent analysis"
            
        if intent.type == IntentType.UNKNOWN and intent.requires_screen:
            return True, None  # Allow unknown intents that might need screen
            
        return True, None


# Global instance
_analyzer = None

def get_intent_analyzer() -> IntentAnalyzer:
    """Get or create a global intent analyzer instance.
    
    This function implements the singleton pattern to ensure only one
    IntentAnalyzer instance is created and reused throughout the application.
    This is efficient since the analyzer's patterns are expensive to compile.
    
    Returns:
        IntentAnalyzer: The global analyzer instance
        
    Example:
        >>> analyzer = get_intent_analyzer()
        >>> intent = await analyzer.analyze("open Chrome")
    """
    global _analyzer
    if _analyzer is None:
        _analyzer = IntentAnalyzer()
    return _analyzer