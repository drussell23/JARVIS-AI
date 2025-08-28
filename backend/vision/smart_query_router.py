#!/usr/bin/env python3
"""
Smart Query Router for JARVIS Multi-Window Intelligence
Routes queries to relevant windows based on intent
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .window_detector import WindowInfo
from .window_relationship_detector import WindowRelationshipDetector, WindowGroup

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Types of query intents"""
    MESSAGES = "messages"
    ERRORS = "errors"
    DOCUMENTATION = "documentation"
    CURRENT_WORK = "current_work"
    WORKSPACE_OVERVIEW = "workspace_overview"
    SPECIFIC_APP = "specific_app"
    NOTIFICATIONS = "notifications"
    CODE_SEARCH = "code_search"
    GENERAL = "general"

@dataclass
class QueryRoute:
    """Routing decision for a query"""
    intent: QueryIntent
    target_windows: List[WindowInfo]
    confidence: float
    reasoning: str
    capture_all: bool = False  # Whether to capture all windows

class SmartQueryRouter:
    """Routes queries to relevant windows intelligently"""
    
    def __init__(self):
        # Query patterns for different intents
        self.intent_patterns = {
            QueryIntent.MESSAGES: [
                r'\b(message|messages|chat|dm|notification|unread)\b',
                r'\b(discord|slack|whatsapp|telegram|teams|mail|email)\b',
                r'\b(any messages|check messages|new messages)\b'
            ],
            QueryIntent.ERRORS: [
                r'\b(error|errors|exception|bug|issue|problem|crash|fail)\b',
                r'\b(terminal|console|log|logs|debug|stack trace)\b',
                r'\b(what.*wrong|broken|not work)\b'
            ],
            QueryIntent.DOCUMENTATION: [
                r'\b(docs|documentation|api|reference|guide|tutorial)\b',
                r'\b(how to|example|sample|usage|syntax)\b',
                r'\b(stack overflow|github|mdn|official)\b'
            ],
            QueryIntent.CURRENT_WORK: [
                r'\b(working on|current|doing|task|focused on)\b',
                r'\b(what am i|what\'s my|tell me what)\b',
                r'\b(project|coding|developing|editing)\b'
            ],
            QueryIntent.WORKSPACE_OVERVIEW: [
                r'\b(workspace|screen|everything|all windows|overview)\b',
                r'\b(what\'s on|show me|list all|describe)\b',
                r'\b(open|running|active)\b'
            ],
            QueryIntent.NOTIFICATIONS: [
                r'\b(notification|notifications|alert|update|badge|reminder)\b',
                r'\b(anything new|important|urgent|attention)\b',
                r'\b(whatsapp|discord|slack|telegram|teams|mail|email)\s+(notification|notifications)\b',
                r'\b(notification|notifications)\s+from\s+(whatsapp|discord|slack|telegram|teams|mail|email)\b',
                r'\b(any|check|new|unread)\s+(notification|notifications)\b'
            ],
            QueryIntent.CODE_SEARCH: [
                r'\b(find|search|locate|where is|look for)\b',
                r'\b(function|class|method|variable|code)\b',
                r'\b(definition|implementation|usage)\b'
            ]
        }
        
        # App categories for routing - now more generic with patterns
        # Instead of hardcoding specific apps, use patterns to identify app types
        self.app_categories = {
            'communication': self._create_app_pattern(['discord', 'slack', 'message', 'mail', 
                                                      'whatsapp', 'telegram', 'signal', 'teams', 
                                                      'zoom', 'chat', 'skype', 'imessage']),
            'development': self._create_app_pattern(['code', 'studio', 'xcode', 'intellij', 
                                                    'pycharm', 'webstorm', 'sublime', 'atom',
                                                    'cursor', 'vim', 'emacs', 'ide']),
            'terminal': self._create_app_pattern(['terminal', 'iterm', 'alacritty', 'hyper', 
                                                 'warp', 'console', 'shell']),
            'browser': self._create_app_pattern(['chrome', 'safari', 'firefox', 'edge', 
                                               'brave', 'opera', 'browser']),
            'documentation': self._create_app_pattern(['preview', 'books', 'dash', 'notion', 
                                                      'obsidian', 'pdf', 'reader', 'notes'])
        }
        
        self.relationship_detector = WindowRelationshipDetector()
    
    def _create_app_pattern(self, keywords: List[str]) -> List[str]:
        """Create flexible patterns for app detection"""
        return keywords  # Can be expanded to regex patterns if needed
    
    def route_query(self, query: str, windows: List[WindowInfo]) -> QueryRoute:
        """Route a query to relevant windows"""
        query_lower = query.lower()
        
        # Detect intent
        intent, confidence = self._detect_intent(query_lower)
        
        # Route based on intent
        if intent == QueryIntent.MESSAGES:
            return self._route_messages_query(query_lower, windows)
        elif intent == QueryIntent.ERRORS:
            return self._route_errors_query(query_lower, windows)
        elif intent == QueryIntent.DOCUMENTATION:
            return self._route_documentation_query(query_lower, windows)
        elif intent == QueryIntent.CURRENT_WORK:
            return self._route_current_work_query(query_lower, windows)
        elif intent == QueryIntent.WORKSPACE_OVERVIEW:
            return self._route_workspace_overview_query(query_lower, windows)
        elif intent == QueryIntent.NOTIFICATIONS:
            return self._route_notifications_query(query_lower, windows)
        elif intent == QueryIntent.SPECIFIC_APP:
            return self._route_specific_app_query(query_lower, windows)
        elif intent == QueryIntent.CODE_SEARCH:
            return self._route_code_search_query(query_lower, windows)
        else:
            return self._route_general_query(query_lower, windows)
    
    def _detect_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """Detect the intent of a query"""
        best_intent = QueryIntent.GENERAL
        best_confidence = 0.0
        
        # Check each intent pattern
        for intent, patterns in self.intent_patterns.items():
            matches = sum(1 for pattern in patterns 
                         if re.search(pattern, query, re.IGNORECASE))
            
            # Calculate confidence based on matches
            if matches > 0:
                confidence = min(matches * 0.4, 1.0)
                if confidence > best_confidence:
                    best_intent = intent
                    best_confidence = confidence
        
        # Check for specific app mentions
        if best_confidence < 0.5:
            for category, apps in self.app_categories.items():
                for app in apps:
                    if app.lower() in query:
                        best_intent = QueryIntent.SPECIFIC_APP
                        best_confidence = 0.9
                        break
        
        return best_intent, best_confidence
    
    def _route_messages_query(self, query: str, windows: List[WindowInfo]) -> QueryRoute:
        """Route queries about messages to communication apps"""
        target_windows = []
        
        # Find all communication windows
        for window in windows:
            if self._is_communication_app(window):
                target_windows.append(window)
        
        # Sort by focus and then by common usage
        target_windows.sort(key=lambda w: (
            not w.is_focused,  # Focused first
            self._get_app_priority(w.app_name)  # Then by priority
        ))
        
        # Limit to top 5 communication windows
        target_windows = target_windows[:5]
        
        return QueryRoute(
            intent=QueryIntent.MESSAGES,
            target_windows=target_windows,
            confidence=0.9 if target_windows else 0.3,
            reasoning=f"Found {len(target_windows)} communication apps",
            capture_all=False
        )
    
    def _route_errors_query(self, query: str, windows: List[WindowInfo]) -> QueryRoute:
        """Route error queries to terminals and development tools"""
        target_windows = []
        
        # Priority 1: Terminal windows
        for window in windows:
            if self._is_terminal_app(window):
                target_windows.append(window)
        
        # Priority 2: Development IDEs (might have error panels)
        for window in windows:
            if self._is_development_app(window) and window not in target_windows:
                # Check if window title suggests errors
                if window.window_title and any(keyword in window.window_title.lower() 
                                              for keyword in ['problem', 'error', 'debug']):
                    target_windows.append(window)
        
        # Priority 3: Browser windows with error-related content
        for window in windows:
            if self._is_browser_app(window) and window not in target_windows:
                if window.window_title and 'error' in window.window_title.lower():
                    target_windows.append(window)
        
        # Sort by relevance
        target_windows.sort(key=lambda w: (
            not self._is_terminal_app(w),  # Terminals first
            not w.is_focused  # Then focused
        ))
        
        return QueryRoute(
            intent=QueryIntent.ERRORS,
            target_windows=target_windows[:5],
            confidence=0.85 if target_windows else 0.2,
            reasoning=f"Found {len(target_windows)} windows that might contain errors",
            capture_all=False
        )
    
    def _route_documentation_query(self, query: str, windows: List[WindowInfo]) -> QueryRoute:
        """Route documentation queries to browsers and doc apps"""
        target_windows = []
        
        # Find documentation windows
        for window in windows:
            if self._is_documentation_window(window):
                target_windows.append(window)
        
        # Sort by relevance (focused first, then by title relevance)
        target_windows.sort(key=lambda w: (
            not w.is_focused,
            not self._has_documentation_keywords(w)
        ))
        
        return QueryRoute(
            intent=QueryIntent.DOCUMENTATION,
            target_windows=target_windows[:5],
            confidence=0.8 if target_windows else 0.3,
            reasoning=f"Found {len(target_windows)} documentation windows",
            capture_all=False
        )
    
    def _route_current_work_query(self, query: str, windows: List[WindowInfo]) -> QueryRoute:
        """Route current work queries to focused window and related windows"""
        target_windows = []
        
        # Always include focused window
        focused_window = next((w for w in windows if w.is_focused), None)
        if focused_window:
            target_windows.append(focused_window)
            
            # Find related windows using relationship detection
            relationships = self.relationship_detector.detect_relationships(windows)
            groups = self.relationship_detector.group_windows(windows, relationships)
            
            # Find the group containing the focused window
            for group in groups:
                if focused_window in group.windows:
                    # Add other windows from the same group
                    for window in group.windows:
                        if window not in target_windows:
                            target_windows.append(window)
                    break
        
        # If no focused window, include top development windows
        if not target_windows:
            for window in windows:
                if self._is_development_app(window) or self._is_terminal_app(window):
                    target_windows.append(window)
                    if len(target_windows) >= 3:
                        break
        
        return QueryRoute(
            intent=QueryIntent.CURRENT_WORK,
            target_windows=target_windows[:5],
            confidence=0.95 if focused_window else 0.6,
            reasoning="Analyzing focused window and related context",
            capture_all=False
        )
    
    def _route_workspace_overview_query(self, query: str, windows: List[WindowInfo]) -> QueryRoute:
        """Route workspace overview queries to all windows"""
        # For overview queries, we want a representative sample
        target_windows = []
        
        # Include focused window first
        focused = next((w for w in windows if w.is_focused), None)
        if focused:
            target_windows.append(focused)
        
        # Include one window from each major category
        categories_seen = set()
        for window in windows:
            if window not in target_windows:
                category = self._get_window_category(window)
                if category and category not in categories_seen:
                    target_windows.append(window)
                    categories_seen.add(category)
        
        # Fill remaining slots with largest windows
        remaining_windows = [w for w in windows if w not in target_windows]
        remaining_windows.sort(key=lambda w: w.bounds['width'] * w.bounds['height'], 
                             reverse=True)
        
        for window in remaining_windows:
            if len(target_windows) < 5:
                target_windows.append(window)
        
        return QueryRoute(
            intent=QueryIntent.WORKSPACE_OVERVIEW,
            target_windows=target_windows,
            confidence=0.9,
            reasoning="Capturing representative windows from workspace",
            capture_all=True  # Indicates we want overview of all
        )
    
    def _route_notifications_query(self, query: str, windows: List[WindowInfo]) -> QueryRoute:
        """Route notification queries to apps that might have notifications"""
        target_windows = []
        
        # Check if a specific app is mentioned
        mentioned_app = None
        for app in ['whatsapp', 'discord', 'slack', 'telegram', 'teams', 'mail', 'messages']:
            if app in query:
                mentioned_app = app
                break
        
        if mentioned_app:
            # Look for the specific app
            for window in windows:
                if mentioned_app.lower() in window.app_name.lower():
                    target_windows.append(window)
            
            # If specific app not found, still check all communication apps
            if not target_windows:
                for window in windows:
                    if self._is_communication_app(window):
                        target_windows.append(window)
        else:
            # Check all communication apps
            for window in windows:
                if self._is_communication_app(window):
                    target_windows.append(window)
        
        # Check browsers (might have web app notifications)
        for window in windows:
            if self._is_browser_app(window) and window not in target_windows:
                if window.window_title and any(keyword in window.window_title.lower()
                                              for keyword in ['inbox', 'notification', 'message', mentioned_app or '']):
                    target_windows.append(window)
        
        # Sort by relevance - mentioned app first
        if mentioned_app:
            target_windows.sort(key=lambda w: (
                mentioned_app.lower() not in w.app_name.lower(),
                not w.is_focused
            ))
        
        reasoning = f"Checking {mentioned_app or 'communication apps'} for notifications"
        
        return QueryRoute(
            intent=QueryIntent.NOTIFICATIONS,
            target_windows=target_windows[:5],
            confidence=0.9 if target_windows and mentioned_app else 0.8 if target_windows else 0.4,
            reasoning=reasoning,
            capture_all=False
        )
    
    def _route_specific_app_query(self, query: str, windows: List[WindowInfo]) -> QueryRoute:
        """Route queries mentioning specific apps"""
        target_windows = []
        mentioned_app = None
        
        # Find which app was mentioned
        for category, apps in self.app_categories.items():
            for app in apps:
                if app.lower() in query:
                    mentioned_app = app
                    break
            if mentioned_app:
                break
        
        # Find windows for that app
        if mentioned_app:
            for window in windows:
                if mentioned_app.lower() in window.app_name.lower():
                    target_windows.append(window)
        
        # Sort by focus
        target_windows.sort(key=lambda w: not w.is_focused)
        
        return QueryRoute(
            intent=QueryIntent.SPECIFIC_APP,
            target_windows=target_windows[:5],
            confidence=0.95 if target_windows else 0.3,
            reasoning=f"Found {len(target_windows)} {mentioned_app or 'app'} windows",
            capture_all=False
        )
    
    def _route_code_search_query(self, query: str, windows: List[WindowInfo]) -> QueryRoute:
        """Route code search queries to development environments"""
        target_windows = []
        
        # Include all development windows
        for window in windows:
            if self._is_development_app(window):
                target_windows.append(window)
        
        # Include terminal windows (might show search results)
        for window in windows:
            if self._is_terminal_app(window) and window not in target_windows:
                target_windows.append(window)
        
        # Sort by focus and window size (larger windows more likely to show code)
        target_windows.sort(key=lambda w: (
            not w.is_focused,
            -(w.bounds['width'] * w.bounds['height'])
        ))
        
        return QueryRoute(
            intent=QueryIntent.CODE_SEARCH,
            target_windows=target_windows[:5],
            confidence=0.8 if target_windows else 0.3,
            reasoning=f"Searching in {len(target_windows)} development windows",
            capture_all=False
        )
    
    def _route_general_query(self, query: str, windows: List[WindowInfo]) -> QueryRoute:
        """Route general queries using current focus context"""
        # Default to current work context
        return self._route_current_work_query(query, windows)
    
    def _is_communication_app(self, window: WindowInfo) -> bool:
        """Check if window is a communication app using pattern matching"""
        app_name_lower = window.app_name.lower()
        # Check if any communication pattern matches
        return any(pattern in app_name_lower for pattern in self.app_categories['communication'])
    
    def _is_development_app(self, window: WindowInfo) -> bool:
        """Check if window is a development app using pattern matching"""
        app_name_lower = window.app_name.lower()
        return any(pattern in app_name_lower for pattern in self.app_categories['development'])
    
    def _is_terminal_app(self, window: WindowInfo) -> bool:
        """Check if window is a terminal using pattern matching"""
        app_name_lower = window.app_name.lower()
        return any(pattern in app_name_lower for pattern in self.app_categories['terminal'])
    
    def _is_browser_app(self, window: WindowInfo) -> bool:
        """Check if window is a browser using pattern matching"""
        app_name_lower = window.app_name.lower()
        return any(pattern in app_name_lower for pattern in self.app_categories['browser'])
    
    def _is_documentation_window(self, window: WindowInfo) -> bool:
        """Check if window contains documentation using pattern matching"""
        app_name_lower = window.app_name.lower()
        
        # Check doc app patterns
        if any(pattern in app_name_lower for pattern in self.app_categories['documentation']):
            return True
        
        # Check browsers with doc content
        if self._is_browser_app(window):
            return self._has_documentation_keywords(window)
        
        return False
    
    def _has_documentation_keywords(self, window: WindowInfo) -> bool:
        """Check if window title suggests documentation"""
        if not window.window_title:
            return False
        
        title_lower = window.window_title.lower()
        doc_keywords = ['docs', 'documentation', 'api', 'reference', 'guide', 
                       'tutorial', 'stackoverflow', 'github', 'mdn', 'npm']
        
        return any(keyword in title_lower for keyword in doc_keywords)
    
    def _get_window_category(self, window: WindowInfo) -> Optional[str]:
        """Get the category of a window"""
        for category, apps in self.app_categories.items():
            if any(app in window.app_name for app in apps):
                return category
        return None
    
    def _get_app_priority(self, app_name: str) -> int:
        """Get priority score for an app (lower is higher priority)"""
        # Define priority order for common apps
        priority_order = ['Discord', 'Slack', 'Messages', 'Mail', 'WhatsApp']
        
        for i, app in enumerate(priority_order):
            if app in app_name:
                return i
        
        return 999  # Low priority for unknown apps

async def test_smart_query_router():
    """Test smart query routing"""
    from .window_detector import WindowDetector
    
    print("🧭 Testing Smart Query Router")
    print("=" * 50)
    
    detector = WindowDetector()
    router = SmartQueryRouter()
    
    # Get current windows
    windows = detector.get_all_windows()
    print(f"\n📊 Found {len(windows)} windows")
    
    # Test various queries
    test_queries = [
        "Do I have any messages?",
        "Are there any errors in my terminal?",
        "Show me the documentation",
        "What am I working on?",
        "What's on my screen?",
        "Check Discord",
        "Find the save function",
        "Any important notifications?"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        route = router.route_query(query, windows)
        
        print(f"   Intent: {route.intent.value}")
        print(f"   Confidence: {route.confidence:.0%}")
        print(f"   Reasoning: {route.reasoning}")
        print(f"   Target windows: {len(route.target_windows)}")
        
        for i, window in enumerate(route.target_windows[:3]):
            print(f"   {i+1}. {window.app_name} - {window.window_title or 'Untitled'}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_smart_query_router())