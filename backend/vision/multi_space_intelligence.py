#!/usr/bin/env python3
"""
Multi-Space Intelligence Extensions for JARVIS Pure Vision System
Adds space-aware query detection and response generation
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class SpaceQueryType(Enum):
    """Types of multi-space queries"""
    SIMPLE_PRESENCE = "simple_presence"      # "Is X open?"
    LOCATION_QUERY = "location_query"        # "Where is X?"
    SPACE_CONTENT = "space_content"          # "What's on Desktop 2?"
    ALL_SPACES = "all_spaces"                # "Show me all my workspaces"
    SPECIFIC_DETAIL = "specific_detail"      # "What's the error in VSCode?"
    WORKSPACE_OVERVIEW = "workspace_overview" # "What am I working on?"

@dataclass
class SpaceQueryIntent:
    """Detected intent for space-related queries"""
    query_type: SpaceQueryType
    target_app: Optional[str] = None
    target_space: Optional[int] = None
    requires_screenshot: bool = False
    confidence: float = 1.0
    metadata_sufficient: bool = True

class MultiSpaceQueryDetector:
    """Detects and classifies multi-space query intents"""
    
    def __init__(self):
        # Query patterns
        self.patterns = {
            'simple_presence': [
                r'\b(is|are)\s+(\w+)\s+(open|running|active)\b',
                r'\bdo i have\s+(\w+)\s+open\b',
                r'\b(\w+)\s+running\?',
            ],
            'location_query': [
                r'\bwhere\s+is\s+(\w+)',
                r'\bwhich\s+(desktop|space|screen)\s+.*\s+(\w+)',
                r'\bfind\s+(\w+)',
                r'\blocation\s+of\s+(\w+)',
            ],
            'space_content': [
                r'\bwhat\'?s?\s+on\s+(desktop|space|screen)\s+(\d+)',
                r'\b(desktop|space|screen)\s+(\d+)\s+content',
                r'\bshow\s+me\s+(desktop|space|screen)\s+(\d+)',
            ],
            'all_spaces': [
                r'\ball\s+(my\s+)?(desktops?|spaces|screens|workspaces?)',
                r'\beverything\s+(open|running)',
                r'\bworkspace\s+overview',
                r'\bwhat\'?s?\s+on\s+all',
            ],
            'specific_detail': [
                r'\bread\s+(the\s+)?(\w+)\s+in\s+(\w+)',
                r'\berror\s+(message|in|on)\s+(\w+)',
                r'\bwhat\s+does\s+(\w+)\s+say',
                r'\bcontent\s+of\s+(\w+)',
            ],
            'workspace_overview': [
                r'\bwhat\s+am\s+i\s+working\s+on',
                r'\bmy\s+current\s+(work|tasks?|projects?)',
                r'\bworkspace\s+status',
                r'\bactive\s+projects?',
            ]
        }
        
        # Application name normalization
        self.app_aliases = {
            'code': 'Visual Studio Code',
            'vscode': 'Visual Studio Code',
            'chrome': 'Google Chrome',
            'firefox': 'Firefox',
            'safari': 'Safari',
            'slack': 'Slack',
            'terminal': 'Terminal',
            'iterm': 'iTerm2',
            'messages': 'Messages',
            'mail': 'Mail',
            'spotify': 'Spotify',
        }
        
    def detect_intent(self, query: str) -> SpaceQueryIntent:
        """Detect the intent of a space-related query"""
        query_lower = query.lower()
        
        # Check each pattern type
        for query_type, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    return self._build_intent(
                        query_type, 
                        match, 
                        query_lower
                    )
                    
        # Default: treat as general query
        return SpaceQueryIntent(
            query_type=SpaceQueryType.SIMPLE_PRESENCE,
            metadata_sufficient=True
        )
        
    def _build_intent(self, query_type: str, match: re.Match, query: str) -> SpaceQueryIntent:
        """Build intent from regex match"""
        intent = SpaceQueryIntent(
            query_type=SpaceQueryType[query_type.upper()]
        )
        
        # Extract app name if present
        groups = match.groups()
        for group in groups:
            if group and group in self.app_aliases:
                intent.target_app = self.app_aliases[group]
            elif group and group.isdigit():
                intent.target_space = int(group)
                
        # Determine if screenshot is required
        if query_type == 'specific_detail':
            intent.requires_screenshot = True
            intent.metadata_sufficient = False
        elif query_type == 'space_content' and 'show' in query:
            intent.requires_screenshot = True
            intent.metadata_sufficient = False
        elif query_type in ['simple_presence', 'location_query']:
            intent.requires_screenshot = False
            intent.metadata_sufficient = True
            
        return intent
        
    def extract_app_name(self, query: str) -> Optional[str]:
        """Extract and normalize application name from query"""
        query_lower = query.lower()
        
        # Check for known aliases
        for alias, full_name in self.app_aliases.items():
            if alias in query_lower:
                return full_name
                
        # Extract potential app name using patterns
        app_patterns = [
            r'\b(\w+)\s+(?:app|application|window)\b',
            r'(?:open|launch|start|find)\s+(\w+)',
            r'(?:is|are)\s+(\w+)\s+(?:open|running)',
        ]
        
        for pattern in app_patterns:
            match = re.search(pattern, query_lower)
            if match:
                potential_app = match.group(1)
                if potential_app in self.app_aliases:
                    return self.app_aliases[potential_app]
                # Return capitalized version if not in aliases
                return potential_app.title()
                
        return None

class SpaceAwarePromptEnhancer:
    """Enhances prompts with multi-space context"""
    
    def __init__(self):
        self.confidence_levels = {
            'screenshot': 'certain',
            'recent_cache': 'confident',
            'metadata': 'based on window information',
            'stale_cache': 'from earlier observation',
            'inference': 'likely',
        }
        
    def enhance_prompt(self, 
                      base_prompt: str, 
                      query_intent: SpaceQueryIntent,
                      space_data: Dict[str, Any]) -> str:
        """Enhance prompt with multi-space context"""
        
        # Add space awareness instructions
        space_context = self._build_space_context(space_data)
        
        enhanced_prompt = f"""{base_prompt}

Multi-Space Context:
{space_context}

Query Type: {query_intent.query_type.value}
{"Target App: " + query_intent.target_app if query_intent.target_app else ""}
{"Target Space: Desktop " + str(query_intent.target_space) if query_intent.target_space else ""}

Instructions for Multi-Space Response:
1. If asked about app presence, check ALL spaces, not just current
2. Specify which desktop/space contains what
3. Use natural space references: "Desktop 2", "your other space", etc.
4. If using cached or metadata info, subtly indicate freshness
5. For location queries, be specific about space number and what else is there
6. Never say you "can't see" other spaces - use available metadata

Response Confidence:
- With screenshot: "I can see..."
- With recent metadata: "VSCode is on Desktop 2..."
- With cache: "Desktop 2 has..." (implying recent observation)
- Metadata only: "Based on window information..."
"""
        
        return enhanced_prompt
        
    def _build_space_context(self, space_data: Dict[str, Any]) -> str:
        """Build natural language space context"""
        if not space_data:
            return "Unable to determine space information"
            
        context_parts = []
        
        # Current space info
        current_space = space_data.get('current_space', {})
        context_parts.append(
            f"You're currently on Desktop {current_space.get('id', 1)} "
            f"with {current_space.get('window_count', 0)} windows"
        )
        
        # Other spaces
        spaces = space_data.get('spaces', [])
        if len(spaces) > 1:
            context_parts.append(
                f"Total {len(spaces)} desktops active"
            )
            
        # Window distribution
        space_window_map = space_data.get('space_window_map', {})
        for space_id, window_ids in space_window_map.items():
            if space_id != current_space.get('id'):
                context_parts.append(
                    f"Desktop {space_id}: {len(window_ids)} windows"
                )
                
        return "\n".join(context_parts)
        
    def generate_confidence_prefix(self, data_source: str) -> str:
        """Generate appropriate confidence prefix for response"""
        return self.confidence_levels.get(data_source, "")

class MultiSpaceResponseBuilder:
    """Builds natural multi-space aware responses"""
    
    def __init__(self):
        self.space_descriptors = [
            "Desktop {}", 
            "Space {}", 
            "your {} workspace",
            "the {} desktop"
        ]
        
    def build_location_response(self, 
                               app_name: str,
                               window_info: Dict[str, Any],
                               confidence: str = "certain") -> str:
        """Build response for app location query"""
        
        # Handle both object and dict formats
        if hasattr(window_info, 'space_id'):
            space_id = window_info.space_id
            is_current = getattr(window_info, 'is_current_space', False)
        else:
            # It's a dict
            space_id = window_info.get('space_id', 1)
            is_current = window_info.get('is_current_space', False)
        
        # Build base response
        if is_current:
            location = "on your current desktop"
        else:
            location = f"on Desktop {space_id}"
            
        response_parts = [f"{app_name} is {location}"]
        
        # Add context if available
        # Handle window attributes
        if hasattr(window_info, 'window_title'):
            window_title = window_info.window_title
            is_fullscreen = getattr(window_info, 'is_fullscreen', False)
            is_minimized = getattr(window_info, 'is_minimized', False)
            companion_apps = getattr(window_info, 'companion_apps', [])
        else:
            # It's a dict
            window_title = window_info.get('window_title', '')
            is_fullscreen = window_info.get('is_fullscreen', False)
            is_minimized = window_info.get('is_minimized', False)
            companion_apps = window_info.get('companion_apps', [])
        
        if window_title:
            response_parts.append(f'with "{window_title}"')
            
        if is_fullscreen:
            response_parts.append("in fullscreen mode")
        elif is_minimized:
            response_parts.append("(minimized)")
            
        # Add companion apps if known
        if companion_apps:
            response_parts.append(
                f"alongside {self._format_app_list(companion_apps)}"
            )
            
        return " ".join(response_parts) + "."
        
    def build_space_overview(self,
                           space_id: int,
                           space_summary: Dict[str, Any],
                           include_screenshot: bool = False) -> str:
        """Build overview of a specific space"""
        
        if not space_summary.get('applications'):
            return f"Desktop {space_id} appears to be empty."
            
        # Build application summary
        app_descriptions = []
        for app, windows in space_summary['applications'].items():
            if len(windows) == 1:
                app_descriptions.append(f"{app} ({windows[0]})")
            else:
                app_descriptions.append(
                    f"{app} ({len(windows)} windows)"
                )
                
        # Format response
        if include_screenshot:
            prefix = f"I can see Desktop {space_id} has"
        else:
            prefix = f"Desktop {space_id} has"
            
        return f"{prefix}: {self._format_app_list(app_descriptions)}."
        
    def build_workspace_overview(self,
                               all_spaces: List[Dict[str, Any]]) -> str:
        """Build complete workspace overview"""
        
        overview_parts = [
            f"You have {len(all_spaces)} desktops active:"
        ]
        
        for space in all_spaces:
            space_id = space['space_id'] if isinstance(space, dict) else space.space_id
            applications = space['applications'] if isinstance(space, dict) else getattr(space, 'applications', {})
            is_current = space['is_current'] if isinstance(space, dict) else getattr(space, 'is_current', False)
            
            # Determine primary activity
            primary_activity = self._determine_space_activity(applications)
            
            if is_current:
                desc = f"Desktop {space_id} (current): {primary_activity}"
            else:
                desc = f"Desktop {space_id}: {primary_activity}"
                
            overview_parts.append(desc)
            
        return "\n".join(overview_parts)
        
    def _format_app_list(self, apps: List[str]) -> str:
        """Format list of apps naturally"""
        if not apps:
            return "no applications"
        if len(apps) == 1:
            return apps[0]
        if len(apps) == 2:
            return f"{apps[0]} and {apps[1]}"
        return f"{', '.join(apps[:-1])}, and {apps[-1]}"
        
    def _determine_space_activity(self, applications: Dict[str, List[str]]) -> str:
        """Determine primary activity on a space"""
        if not applications:
            return "Empty"
            
        # Check for common patterns
        app_names = list(applications.keys())
        
        if any(dev_app in app_names for dev_app in 
               ['Visual Studio Code', 'Xcode', 'Terminal']):
            return "Development work"
        elif any(comm_app in app_names for comm_app in 
                ['Slack', 'Messages', 'Mail']):
            return "Communication"
        elif any(browser in app_names for browser in 
                ['Safari', 'Chrome', 'Firefox']):
            return "Web browsing/research"
        else:
            # Default to listing main apps
            return self._format_app_list(app_names[:2])

# Integration class for pure_vision_intelligence.py
class MultiSpaceIntelligenceExtension:
    """Extension to add multi-space awareness to PureVisionIntelligence"""
    
    def __init__(self):
        self.query_detector = MultiSpaceQueryDetector()
        self.prompt_enhancer = SpaceAwarePromptEnhancer()
        self.response_builder = MultiSpaceResponseBuilder()
        
    def should_use_multi_space(self, query: str) -> bool:
        """Determine if query needs multi-space handling"""
        intent = self.query_detector.detect_intent(query)
        
        # Any query about apps, windows, or spaces needs multi-space
        return (
            intent.query_type != SpaceQueryType.SIMPLE_PRESENCE or
            intent.target_app is not None or
            intent.target_space is not None or
            any(keyword in query.lower() for keyword in 
                ['where', 'all', 'desktop', 'space', 'workspace', 'everywhere'])
        )
        
    def process_multi_space_query(self, 
                                query: str, 
                                window_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a multi-space query with available data"""
        
        # Detect intent
        intent = self.query_detector.detect_intent(query)
        
        # Determine what data we need vs what we have
        data_requirements = self._analyze_data_requirements(intent, window_data)
        
        # Build response based on available data
        response_data = {
            'intent': intent,
            'data_requirements': data_requirements,
            'can_answer': data_requirements['can_answer_with_current_data'],
            'confidence': data_requirements['confidence_level'],
            'suggested_response': None
        }
        
        # Generate suggested response if we can answer
        if response_data['can_answer']:
            response_data['suggested_response'] = self._generate_response(
                intent, window_data, data_requirements
            )
            
        return response_data
        
    def _analyze_data_requirements(self, 
                                 intent: SpaceQueryIntent,
                                 available_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze what data is needed vs available"""
        
        requirements = {
            'needs_screenshot': intent.requires_screenshot,
            'needs_metadata': True,  # Always useful
            'can_answer_with_current_data': False,
            'confidence_level': 'low',
            'missing_data': []
        }
        
        # Check what we have
        has_metadata = 'windows' in available_data
        has_current_screenshot = available_data.get('has_current_screenshot', False)
        has_cached_screenshots = any(
            (hasattr(space, 'cached_screenshot') and space.cached_screenshot is not None) if hasattr(space, 'cached_screenshot')
            else (isinstance(space, dict) and space.get('cached_screenshot') is not None)
            for space in available_data.get('spaces', [])
        )
        
        # Determine if we can answer
        if intent.metadata_sufficient and has_metadata:
            requirements['can_answer_with_current_data'] = True
            requirements['confidence_level'] = 'high'
        elif intent.requires_screenshot:
            if intent.target_space:
                # Check if we have screenshot for target space
                target_space_data = next(
                    (s for s in available_data.get('spaces', []) 
                     if (hasattr(s, 'space_id') and s.space_id == intent.target_space) or
                        (isinstance(s, dict) and s.get('space_id') == intent.target_space)),
                    None
                )
                if target_space_data:
                    has_screenshot = (hasattr(target_space_data, 'cached_screenshot') and target_space_data.cached_screenshot) if hasattr(target_space_data, 'cached_screenshot') \
                                   else (isinstance(target_space_data, dict) and target_space_data.get('cached_screenshot'))
                    if has_screenshot:
                        requirements['can_answer_with_current_data'] = True
                        requirements['confidence_level'] = 'medium'
                    else:
                        requirements['missing_data'].append('screenshot_for_space')
                else:
                    requirements['missing_data'].append('screenshot_for_space')
            elif has_current_screenshot:
                requirements['can_answer_with_current_data'] = True
                requirements['confidence_level'] = 'high'
                
        return requirements
        
    def _generate_response(self,
                         intent: SpaceQueryIntent,
                         window_data: Dict[str, Any],
                         requirements: Dict[str, Any]) -> str:
        """Generate appropriate response based on intent and data"""
        
        if intent.query_type == SpaceQueryType.LOCATION_QUERY:
            # Find the app
            windows = window_data.get('windows', [])
            target_windows = [
                w for w in windows 
                if intent.target_app and intent.target_app.lower() in (
                    w.app_name.lower() if hasattr(w, 'app_name') else w.get('app_name', '').lower()
                )
            ]
            
            if target_windows:
                return self.response_builder.build_location_response(
                    intent.target_app,
                    target_windows[0],
                    requirements['confidence_level']
                )
            else:
                return f"I don't see {intent.target_app} open on any desktop."
                
        elif intent.query_type == SpaceQueryType.ALL_SPACES:
            spaces_summary = []
            for space in window_data.get('spaces', []):
                if hasattr(space, 'space_id'):
                    # It's a SpaceInfo object
                    space_id = space.space_id
                    is_current = space.is_current
                else:
                    # It's a dictionary
                    space_id = space['space_id']
                    is_current = space['is_current']
                    
                summary = {
                    'space_id': space_id,
                    'is_current': is_current,
                    'applications': self._get_space_applications(
                        space_id, 
                        window_data
                    )
                }
                spaces_summary.append(summary)
                
            return self.response_builder.build_workspace_overview(spaces_summary)
            
        # Add more response types as needed
        return "I can help you with that query about your workspaces."
        
    def _get_space_applications(self, 
                              space_id: int, 
                              window_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Get applications on a specific space"""
        space_windows = window_data.get('space_window_map', {}).get(space_id, [])
        windows = window_data.get('windows', [])
        
        apps = {}
        for window_id in space_windows:
            window = next((w for w in windows if (
                (hasattr(w, 'window_id') and w.window_id == window_id) or
                (isinstance(w, dict) and w.get('window_id') == window_id)
            )), None)
            if window:
                app_name = window.app_name if hasattr(window, 'app_name') else window.get('app_name', 'Unknown')
                if app_name not in apps:
                    apps[app_name] = []
                window_title = window.window_title if hasattr(window, 'window_title') else window.get('window_title', '')
                apps[app_name].append(window_title)
                
        return apps