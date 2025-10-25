"""
Action-Oriented Query Analyzer for JARVIS
==========================================

Analyzes action-oriented queries to determine:
- Action type (fix, switch, close, run, etc.)
- Target (error, space, browser, tests, etc.)
- Parameters (space number, app name, etc.)
- Safety level (safe, needs_confirmation, risky)

Handles queries like:
- "Fix the error in space 3"
- "Switch to space 5"
- "Close the browser in space 2"
- "Run the tests"
- "Fix it" (with implicit reference resolution)

Author: Derek Russell
Date: 2025-10-19
"""

import logging
import re
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# ACTION TYPES
# ============================================================================

class ActionType(Enum):
    """Types of actions that can be performed"""
    # Space/Window Management
    SWITCH_SPACE = "switch_space"              # Switch to a different space
    FOCUS_WINDOW = "focus_window"              # Focus a specific window
    CLOSE_WINDOW = "close_window"              # Close a window
    MOVE_WINDOW = "move_window"                # Move window to different space
    ARRANGE_WINDOWS = "arrange_windows"        # Arrange/tile windows

    # Application Control
    LAUNCH_APP = "launch_app"                  # Launch an application
    QUIT_APP = "quit_app"                      # Quit an application
    RESTART_APP = "restart_app"                # Restart an application

    # Development Actions
    RUN_TESTS = "run_tests"                    # Run test suite
    RUN_BUILD = "run_build"                    # Run build process
    FIX_ERROR = "fix_error"                    # Fix an error/bug
    DEBUG = "debug"                            # Debug an issue
    DEPLOY = "deploy"                          # Deploy code

    # Browser Actions
    OPEN_URL = "open_url"                      # Open URL in browser
    CLOSE_TAB = "close_tab"                    # Close browser tab
    REFRESH_PAGE = "refresh_page"              # Refresh page

    # System Actions
    EXECUTE_COMMAND = "execute_command"        # Execute shell command
    COPY_TO_CLIPBOARD = "copy_to_clipboard"    # Copy to clipboard

    # File Operations
    OPEN_FILE = "open_file"                    # Open a file
    SAVE_FILE = "save_file"                    # Save a file
    DELETE_FILE = "delete_file"                # Delete a file

    # Generic
    UNKNOWN = "unknown"


class ActionSafety(Enum):
    """Safety level of actions"""
    SAFE = "safe"                              # Safe to execute without confirmation
    NEEDS_CONFIRMATION = "needs_confirmation"  # Needs user confirmation
    RISKY = "risky"                           # High risk, extra confirmation needed
    BLOCKED = "blocked"                        # Should not be executed


class ActionTarget(Enum):
    """What the action targets"""
    SPACE = "space"                # Desktop space
    WINDOW = "window"              # Application window
    APPLICATION = "application"    # Application
    ERROR = "error"                # Error/bug
    TESTS = "tests"                # Test suite
    FILE = "file"                  # File
    URL = "url"                    # Web URL
    COMMAND = "command"            # Shell command
    UNKNOWN = "unknown"


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ActionParameter:
    """A parameter for an action"""
    name: str
    value: Any
    explicit: bool = True  # Was this explicitly stated or inferred?
    confidence: float = 1.0


@dataclass
class ActionIntent:
    """Parsed action intent"""
    action_type: ActionType
    target_type: ActionTarget
    parameters: Dict[str, ActionParameter] = field(default_factory=dict)
    safety_level: ActionSafety = ActionSafety.NEEDS_CONFIRMATION
    confidence: float = 0.0
    requires_resolution: bool = False  # Needs implicit reference resolution?
    original_query: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_param(self, name: str, default=None) -> Any:
        """Get parameter value"""
        param = self.parameters.get(name)
        return param.value if param else default

    def has_param(self, name: str) -> bool:
        """Check if parameter exists"""
        return name in self.parameters


# ============================================================================
# ACTION ANALYZER
# ============================================================================

class ActionAnalyzer:
    """
    Analyzes action-oriented queries to determine intent and parameters

    Uses pattern matching and NLP to understand:
    - What action to perform
    - What to act on
    - Where to perform the action
    - Any additional parameters
    """

    def __init__(self):
        """Initialize the action analyzer"""
        self.action_patterns = self._initialize_patterns()
        self.safety_rules = self._initialize_safety_rules()

    def _initialize_patterns(self) -> Dict[ActionType, List[re.Pattern]]:
        """Initialize action patterns"""
        return {
            # Space/Window Management
            ActionType.SWITCH_SPACE: [
                re.compile(r'\b(switch|go|move|jump)\s+(to\s+)?space\s+(\d+)\b', re.I),
                re.compile(r'\bspace\s+(\d+)\b', re.I),
                re.compile(r'\b(switch|go)\s+to\s+(desktop\s+)?(\d+)\b', re.I),
            ],
            ActionType.FOCUS_WINDOW: [
                re.compile(r'\b(focus|show|bring up)\s+(the\s+)?(\w+)\s+(window|app)\b', re.I),
                re.compile(r'\bfocus\s+on\s+(\w+)', re.I),
            ],
            ActionType.CLOSE_WINDOW: [
                re.compile(r'\bclose\s+(the\s+)?(\w+)\s+(window|app|browser|tab)', re.I),
                re.compile(r'\bclose\s+(it|that|this)', re.I),
                re.compile(r'\bshut\s+down\s+(\w+)', re.I),
            ],
            ActionType.MOVE_WINDOW: [
                re.compile(r'\bmove\s+(\w+)\s+to\s+space\s+(\d+)', re.I),
                re.compile(r'\bsend\s+(\w+)\s+to\s+space\s+(\d+)', re.I),
            ],

            # Application Control
            ActionType.LAUNCH_APP: [
                re.compile(r'\b(open|launch|start|run)\s+(\w+)', re.I),
            ],
            ActionType.QUIT_APP: [
                re.compile(r'\b(quit|exit|close)\s+(\w+)', re.I),
                re.compile(r'\bkill\s+(\w+)', re.I),
            ],
            ActionType.RESTART_APP: [
                re.compile(r'\brestart\s+(\w+)', re.I),
                re.compile(r'\brelaunch\s+(\w+)', re.I),
            ],

            # Development Actions
            ActionType.RUN_TESTS: [
                re.compile(r'\b(run|execute)\s+(the\s+)?(tests|test suite)', re.I),
                re.compile(r'\btest\s+(it|this|that)', re.I),
            ],
            ActionType.RUN_BUILD: [
                re.compile(r'\b(run|execute|start)\s+(the\s+)?build', re.I),
                re.compile(r'\bbuild\s+(it|this|the\s+project)', re.I),
            ],
            ActionType.FIX_ERROR: [
                re.compile(r'\bfix\s+(the\s+)?(error|bug|issue|problem)', re.I),
                re.compile(r'\bfix\s+(it|that|this)', re.I),
                re.compile(r'\bsolve\s+(the\s+)?(error|issue)', re.I),
            ],
            ActionType.DEBUG: [
                re.compile(r'\bdebug\s+(this|that|it)', re.I),
                re.compile(r'\b(investigate|diagnose)\s+(the\s+)?(error|issue)', re.I),
            ],

            # Browser Actions
            ActionType.OPEN_URL: [
                re.compile(r'\b(open|go to|navigate to)\s+(.*?)(\.com|\.org|\.net|https?://)', re.I),
            ],
            ActionType.CLOSE_TAB: [
                re.compile(r'\bclose\s+(the\s+)?(tab|browser)', re.I),
            ],
            ActionType.REFRESH_PAGE: [
                re.compile(r'\b(refresh|reload)\s+(the\s+)?(page|browser)', re.I),
            ],

            # System Actions
            ActionType.EXECUTE_COMMAND: [
                re.compile(r'\b(execute|run)\s+command\s+["\'](.+?)["\']', re.I),
                re.compile(r'\brun\s+["\'](.+?)["\']', re.I),
            ],
        }

    def _initialize_safety_rules(self) -> Dict[ActionType, ActionSafety]:
        """Initialize safety rules for each action type"""
        return {
            # Safe actions
            ActionType.SWITCH_SPACE: ActionSafety.SAFE,
            ActionType.FOCUS_WINDOW: ActionSafety.SAFE,
            ActionType.OPEN_URL: ActionSafety.SAFE,
            ActionType.REFRESH_PAGE: ActionSafety.SAFE,
            ActionType.LAUNCH_APP: ActionSafety.SAFE,
            ActionType.RUN_TESTS: ActionSafety.SAFE,
            ActionType.RUN_BUILD: ActionSafety.SAFE,

            # Needs confirmation
            ActionType.CLOSE_WINDOW: ActionSafety.NEEDS_CONFIRMATION,
            ActionType.MOVE_WINDOW: ActionSafety.NEEDS_CONFIRMATION,
            ActionType.QUIT_APP: ActionSafety.NEEDS_CONFIRMATION,
            ActionType.CLOSE_TAB: ActionSafety.NEEDS_CONFIRMATION,
            ActionType.FIX_ERROR: ActionSafety.NEEDS_CONFIRMATION,

            # Risky actions
            ActionType.RESTART_APP: ActionSafety.RISKY,
            ActionType.DELETE_FILE: ActionSafety.RISKY,
            ActionType.DEPLOY: ActionSafety.RISKY,
            ActionType.EXECUTE_COMMAND: ActionSafety.RISKY,
        }

    async def analyze(self, query: str, context: Optional[Dict[str, Any]] = None) -> ActionIntent:
        """
        Analyze an action query

        Args:
            query: The action query
            context: Optional context information

        Returns:
            ActionIntent with parsed information
        """
        logger.info(f"[ACTION-ANALYZER] Analyzing query: '{query}'")

        # Step 1: Identify action type
        action_type, match, pattern_confidence = await self._identify_action(query)

        # Step 2: Extract parameters from the match
        parameters = await self._extract_parameters(action_type, query, match)

        # Step 3: Determine target type
        target_type = await self._determine_target_type(action_type, parameters, query)

        # Step 4: Check if implicit resolution is needed
        requires_resolution = await self._check_needs_resolution(query, parameters)

        # Step 5: Determine safety level
        safety_level = self._get_safety_level(action_type, parameters)

        # Step 6: Calculate overall confidence
        confidence = await self._calculate_confidence(
            action_type, target_type, parameters, pattern_confidence
        )

        intent = ActionIntent(
            action_type=action_type,
            target_type=target_type,
            parameters=parameters,
            safety_level=safety_level,
            confidence=confidence,
            requires_resolution=requires_resolution,
            original_query=query,
            metadata={
                "context": context,
                "pattern_confidence": pattern_confidence
            }
        )

        logger.info(f"[ACTION-ANALYZER] Parsed intent: {action_type.value}, target: {target_type.value}, safety: {safety_level.value}")

        return intent

    async def _identify_action(self, query: str) -> Tuple[ActionType, Optional[re.Match], float]:
        """Identify the action type from query"""
        for action_type, patterns in self.action_patterns.items():
            for pattern in patterns:
                match = pattern.search(query)
                if match:
                    confidence = 0.9 if action_type != ActionType.UNKNOWN else 0.3
                    return action_type, match, confidence

        return ActionType.UNKNOWN, None, 0.1

    async def _extract_parameters(
        self,
        action_type: ActionType,
        query: str,
        match: Optional[re.Match]
    ) -> Dict[str, ActionParameter]:
        """Extract parameters from the query"""
        parameters = {}

        if not match:
            return parameters

        # Extract space numbers
        space_match = re.search(r'space\s+(\d+)', query, re.I)
        if space_match:
            parameters["space_id"] = ActionParameter(
                name="space_id",
                value=int(space_match.group(1)),
                explicit=True,
                confidence=1.0
            )

        # Extract "in space X" context
        in_space_match = re.search(r'in\s+space\s+(\d+)', query, re.I)
        if in_space_match:
            parameters["context_space"] = ActionParameter(
                name="context_space",
                value=int(in_space_match.group(1)),
                explicit=True,
                confidence=1.0
            )

        # Extract application names
        if action_type in [ActionType.LAUNCH_APP, ActionType.QUIT_APP, ActionType.RESTART_APP]:
            app_patterns = [
                r'(open|launch|quit|restart|close|kill)\s+(\w+)',
                r'(the\s+)?(\w+)\s+(app|window|browser)',
            ]
            for pattern in app_patterns:
                app_match = re.search(pattern, query, re.I)
                if app_match:
                    app_name = app_match.group(2) if len(app_match.groups()) >= 2 else app_match.group(1)
                    parameters["app_name"] = ActionParameter(
                        name="app_name",
                        value=app_name.capitalize(),
                        explicit=True,
                        confidence=0.8
                    )
                    break

        # Extract error/bug references
        if action_type == ActionType.FIX_ERROR:
            error_ref = re.search(r'(the\s+)?(error|bug|issue)', query, re.I)
            if error_ref:
                parameters["error_reference"] = ActionParameter(
                    name="error_reference",
                    value=error_ref.group(0),
                    explicit=False,  # Will need resolution
                    confidence=0.7
                )

        # Extract pronouns (for implicit resolution)
        pronoun_match = re.search(r'\b(it|that|this|them)\b', query, re.I)
        if pronoun_match:
            parameters["pronoun"] = ActionParameter(
                name="pronoun",
                value=pronoun_match.group(1),
                explicit=False,
                confidence=0.6
            )

        # Extract URLs
        url_match = re.search(r'(https?://[^\s]+|[\w-]+\.(com|org|net|io))', query, re.I)
        if url_match:
            parameters["url"] = ActionParameter(
                name="url",
                value=url_match.group(0),
                explicit=True,
                confidence=0.9
            )

        return parameters

    async def _determine_target_type(
        self,
        action_type: ActionType,
        parameters: Dict[str, ActionParameter],
        query: str
    ) -> ActionTarget:
        """Determine what the action targets"""
        # Based on action type
        action_to_target = {
            ActionType.SWITCH_SPACE: ActionTarget.SPACE,
            ActionType.FOCUS_WINDOW: ActionTarget.WINDOW,
            ActionType.CLOSE_WINDOW: ActionTarget.WINDOW,
            ActionType.MOVE_WINDOW: ActionTarget.WINDOW,
            ActionType.LAUNCH_APP: ActionTarget.APPLICATION,
            ActionType.QUIT_APP: ActionTarget.APPLICATION,
            ActionType.RESTART_APP: ActionTarget.APPLICATION,
            ActionType.RUN_TESTS: ActionTarget.TESTS,
            ActionType.FIX_ERROR: ActionTarget.ERROR,
            ActionType.OPEN_URL: ActionTarget.URL,
            ActionType.EXECUTE_COMMAND: ActionTarget.COMMAND,
        }

        if action_type in action_to_target:
            return action_to_target[action_type]

        # Check query for explicit target mentions
        target_keywords = {
            ActionTarget.SPACE: ["space", "desktop", "workspace"],
            ActionTarget.WINDOW: ["window", "tab"],
            ActionTarget.APPLICATION: ["app", "application", "browser"],
            ActionTarget.ERROR: ["error", "bug", "issue", "problem"],
            ActionTarget.TESTS: ["test", "tests", "test suite"],
            ActionTarget.FILE: ["file", "document"],
        }

        query_lower = query.lower()
        for target, keywords in target_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return target

        return ActionTarget.UNKNOWN

    async def _check_needs_resolution(
        self,
        query: str,
        parameters: Dict[str, ActionParameter]
    ) -> bool:
        """Check if the query needs implicit reference resolution"""
        # Has pronouns?
        if "pronoun" in parameters:
            return True

        # Has "the error/bug" without specific details?
        if "error_reference" in parameters and not parameters["error_reference"].explicit:
            return True

        # Has vague references like "that", "this"?
        vague_refs = re.search(r'\b(that|this|the)\s+(error|bug|window|app|browser)\b', query, re.I)
        if vague_refs:
            return True

        # Has implicit context (no explicit space/app mentioned)?
        has_explicit_target = any(
            p.explicit for p in parameters.values()
            if p.name in ["space_id", "app_name", "url"]
        )

        return not has_explicit_target

    def _get_safety_level(
        self,
        action_type: ActionType,
        parameters: Dict[str, ActionParameter]
    ) -> ActionSafety:
        """Determine safety level for the action"""
        # Get base safety from rules
        base_safety = self.safety_rules.get(action_type, ActionSafety.NEEDS_CONFIRMATION)

        # Increase safety concern if action is ambiguous
        has_ambiguous_params = any(
            not p.explicit or p.confidence < 0.7
            for p in parameters.values()
        )

        if has_ambiguous_params and base_safety == ActionSafety.SAFE:
            return ActionSafety.NEEDS_CONFIRMATION

        return base_safety

    async def _calculate_confidence(
        self,
        action_type: ActionType,
        target_type: ActionTarget,
        parameters: Dict[str, ActionParameter],
        pattern_confidence: float
    ) -> float:
        """Calculate overall confidence in the action intent"""
        confidence = pattern_confidence

        # Boost if we have explicit parameters
        explicit_params = [p for p in parameters.values() if p.explicit]
        if explicit_params:
            confidence += 0.1 * min(len(explicit_params), 3)

        # Reduce if target is unknown
        if target_type == ActionTarget.UNKNOWN:
            confidence -= 0.2

        # Reduce if action is unknown
        if action_type == ActionType.UNKNOWN:
            confidence -= 0.3

        return max(0.0, min(1.0, confidence))


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_global_analyzer: Optional[ActionAnalyzer] = None


def get_action_analyzer() -> ActionAnalyzer:
    """Get the global action analyzer instance"""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = ActionAnalyzer()
    return _global_analyzer


def initialize_action_analyzer() -> ActionAnalyzer:
    """Initialize the global action analyzer"""
    global _global_analyzer
    _global_analyzer = ActionAnalyzer()
    logger.info("[ACTION-ANALYZER] Global instance initialized")
    return _global_analyzer


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

async def analyze_action(query: str, **kwargs) -> ActionIntent:
    """Convenience function to analyze an action query"""
    analyzer = get_action_analyzer()
    return await analyzer.analyze(query, **kwargs)
