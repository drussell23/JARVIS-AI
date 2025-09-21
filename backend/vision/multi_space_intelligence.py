#!/usr/bin/env python3
"""
Multi-Space Intelligence Extensions for JARVIS Pure Vision System
Adds space-aware query detection and response generation
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import difflib

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
    context_hints: List[str] = field(default_factory=list)
    space_references: List[str] = field(default_factory=list)
    detected_patterns: List[str] = field(default_factory=list)

class MultiSpaceQueryDetector:
    """Detects and classifies multi-space query intents with dynamic pattern matching"""
    
    def __init__(self):
        # Dynamic pattern components
        self._space_terms = {'desktop', 'space', 'screen', 'workspace', 'monitor', 'display', 'area', 'environment'}
        self._location_verbs = {'where', 'which', 'find', 'locate', 'search', 'look', 'check'}
        self._presence_verbs = {'is', 'are', 'have', 'has', 'got', 'running', 'open', 'active', 'visible', 'showing'}
        self._other_terms = {'other', 'another', 'different', 'alternate', 'secondary', 'next', 'previous', 'adjacent'}
        self._all_terms = {'all', 'every', 'each', 'entire', 'whole', 'complete', 'full'}
        
        # Build dynamic patterns
        self._build_dynamic_patterns()
        
        # Application detection patterns
        self._app_indicators = {
            'suffix': ['app', 'application', 'program', 'software', 'tool', 'ide', 'editor', 'browser', 'client'],
            'context': ['window', 'instance', 'session', 'process']
        }
        
        # Initialize app name cache
        self._app_name_cache = {}
        self._common_app_words = self._build_common_app_words()
        
    def _build_dynamic_patterns(self):
        """Build patterns dynamically from components"""
        space_alt = '|'.join(self._space_terms)
        loc_alt = '|'.join(self._location_verbs)
        pres_alt = '|'.join(self._presence_verbs)
        other_alt = '|'.join(self._other_terms)
        all_alt = '|'.join(self._all_terms)
        
        self.patterns = {
            'simple_presence': self._generate_presence_patterns(pres_alt),
            'location_query': self._generate_location_patterns(loc_alt, space_alt, other_alt),
            'space_content': self._generate_space_content_patterns(space_alt, other_alt),
            'all_spaces': self._generate_all_spaces_patterns(all_alt, space_alt),
            'specific_detail': self._generate_detail_patterns(),
            'workspace_overview': self._generate_overview_patterns()
        }
        
    def _generate_presence_patterns(self, pres_alt):
        """Generate presence detection patterns"""
        return [
            rf'\b({pres_alt})\s+(\S+(?:\s+\S+)?)\s+(open|running|active|visible)\b',
            rf'\b(do|does)\s+\S+\s+have\s+(\S+(?:\s+\S+)?)\s+open\b',
            rf'\b(\S+(?:\s+\S+)?)\s+({pres_alt})\s*\?',
            rf'\bcan\s+(?:you\s+)?(?:see|find)\s+(\S+(?:\s+\S+)?)\b',
        ]
        
    def _generate_location_patterns(self, loc_alt, space_alt, other_alt):
        """Generate location query patterns"""
        return [
            rf'\b({loc_alt})\s+(?:is|are)\s+(\S+(?:\s+\S+)?)',
            rf'\b({loc_alt})\s+(?:can\s+)?(?:I|you)\s+find\s+(\S+(?:\s+\S+)?)',
            rf'\b(?:on\s+)?which\s+({space_alt})\s+(?:is|are)\s+(\S+(?:\s+\S+)?)',
            rf'\b(\S+(?:\s+\S+)?)\s+(?:in|on)\s+(?:the\s+)?({other_alt})\s+({space_alt})',
            rf'\bcan\s+you\s+see\s+(?:if\s+)?(\S+(?:\s+\S+)?)\s+.*\s+({other_alt})\s+({space_alt})',
            rf'\b(?:show|tell)\s+me\s+where\s+(\S+(?:\s+\S+)?)\s+is',
        ]
        
    def _generate_space_content_patterns(self, space_alt, other_alt):
        """Generate space content patterns"""
        return [
            rf'\bwhat(?:\'s|s| is)\s+(?:on|in)\s+({space_alt})\s+(\d+)',
            rf'\b({space_alt})\s+(\d+)\s+(?:content|contents|has|shows)',
            rf'\bshow\s+(?:me\s+)?({space_alt})\s+(\d+)',
            rf'\bwhat(?:\'s|s| is)\s+(?:on|in)\s+(?:the\s+)?({other_alt})\s+({space_alt})',
            rf'\b(?:display|show|list)\s+(?:the\s+)?({other_alt})\s+({space_alt})',
            rf'\bwhat\s+do\s+(?:I|you)\s+(?:have|see)\s+(?:on|in)\s+(?:the\s+)?({other_alt})',
        ]
        
    def _generate_all_spaces_patterns(self, all_alt, space_alt):
        """Generate all spaces patterns"""
        return [
            rf'\b({all_alt})\s+(?:my\s+)?({space_alt})s?\b',
            rf'\b(?:show|list|display)\s+(?:me\s+)?everything\s+(?:that(?:\'s|s)?\s+)?(?:open|running)',
            rf'\b({space_alt})\s+(?:overview|summary|status)',
            rf'\bwhat(?:\'s|s| is)\s+(?:on|in)\s+({all_alt})',
            rf'\b(?:across|throughout)\s+(?:all\s+)?(?:my\s+)?({space_alt})s?',
        ]
        
    def _generate_detail_patterns(self):
        """Generate specific detail patterns"""
        return [
            r'\b(?:read|show|display)\s+(?:the\s+)?(\S+)\s+(?:in|on|from)\s+(\S+)',
            r'\b(?:error|warning|message|alert)\s+(?:in|on|from)\s+(\S+)',
            r'\bwhat\s+(?:does|says)\s+(\S+)\s+(?:say|show|display)',
            r'\b(?:content|contents|text)\s+(?:of|in|from)\s+(\S+)',
            r'\b(?:check|examine|inspect)\s+(\S+)\s+in\s+(\S+)',
        ]
        
    def _generate_overview_patterns(self):
        """Generate workspace overview patterns"""
        return [
            r'\bwhat\s+am\s+I\s+(?:working|doing|focused)\s+on',
            r'\b(?:my|current)\s+(?:work|tasks?|projects?|activities)',
            r'\b(?:workspace|desktop|screen)\s+(?:status|state|overview)',
            r'\b(?:active|current|ongoing)\s+(?:work|projects?|tasks?)',
            r'\b(?:show|display|list)\s+(?:my\s+)?(?:current\s+)?(?:work|activity)',
        ]
        
    def _build_common_app_words(self):
        """Build set of common application-related words"""
        words = set()
        
        # Common app name patterns
        tech_terms = {'visual', 'studio', 'code', 'android', 'web', 'dev', 'tools'}
        generic_terms = {'pro', 'plus', 'lite', 'express', 'community', 'professional'}
        
        words.update(tech_terms)
        words.update(generic_terms)
        
        return words
        
    def detect_intent(self, query: str) -> SpaceQueryIntent:
        """Detect the intent of a space-related query with confidence scoring"""
        query_lower = query.lower()
        
        # Track all matches with confidence scores
        matches = []
        
        for query_type, patterns in self.patterns.items():
            for pattern_idx, pattern in enumerate(patterns):
                match = re.search(pattern, query_lower)
                if match:
                    # Calculate confidence based on match quality
                    confidence = self._calculate_match_confidence(
                        match, pattern, query_lower, query_type
                    )
                    matches.append({
                        'type': query_type,
                        'match': match,
                        'pattern': pattern,
                        'confidence': confidence,
                        'priority': pattern_idx
                    })
        
        # Select best match based on confidence and priority
        if matches:
            best_match = max(matches, key=lambda x: (x['confidence'], -x['priority']))
            return self._build_intent(
                best_match['type'], 
                best_match['match'], 
                query,
                confidence=best_match['confidence']
            )
                    
        # Default with context analysis
        return self._analyze_unmatched_query(query)
        
    def _build_intent(self, query_type: str, match: re.Match, query: str, confidence: float = 1.0) -> SpaceQueryIntent:
        """Build intent from regex match with enhanced context"""
        intent = SpaceQueryIntent(
            query_type=SpaceQueryType[query_type.upper()],
            confidence=confidence
        )
        
        # Extract app name with context
        app_info = self._extract_app_with_context(query)
        if app_info:
            intent.target_app = app_info['name']
            intent.context_hints.extend(app_info.get('hints', []))
        
        # Extract space references
        space_refs = self._extract_space_references(query, match)
        intent.space_references = space_refs['references']
        if space_refs.get('number'):
            intent.target_space = space_refs['number']
                
        # Determine requirements based on query analysis
        requirements = self._analyze_requirements(query_type, query, intent)
        intent.requires_screenshot = requirements['screenshot']
        intent.metadata_sufficient = requirements['metadata']
        
        # Add detected patterns for transparency
        intent.detected_patterns.append(match.re.pattern)
            
        return intent
        
    def extract_app_name(self, query: str) -> Optional[str]:
        """Legacy method - delegates to new context-aware extraction"""
        app_info = self._extract_app_with_context(query)
        return app_info['name'] if app_info else None
        
    def _extract_app_with_context(self, query: str) -> Optional[Dict[str, Any]]:
        """Extract application name with contextual hints"""
        query_lower = query.lower()
        
        # Check if this is about general space content
        if self._is_general_space_query(query_lower):
            return None
            
        # Try multiple extraction strategies
        strategies = [
            self._extract_by_known_apps,
            self._extract_by_patterns,
            self._extract_by_context_clues,
            self._extract_by_fuzzy_match
        ]
        
        for strategy in strategies:
            result = strategy(query, query_lower)
            if result:
                return result
                
        return None
        
    def _is_general_space_query(self, query_lower: str) -> bool:
        """Check if query is about general space content"""
        general_patterns = [
            r"what'?s?\s+(?:on|in)\s+(?:the\s+)?(?:other|another|different)\s+",
            r"show\s+me\s+(?:the\s+)?(?:other|another|different)\s+",
            r"(?:list|display)\s+(?:everything|all)\s+(?:on|in)\s+",
        ]
        
        return any(re.search(p, query_lower) for p in general_patterns)
        
    def _extract_by_known_apps(self, query: str, query_lower: str) -> Optional[Dict[str, Any]]:
        """Extract using known application database"""
        # Dynamic app discovery from query
        words = query_lower.split()
        
        # Build potential app names from word combinations
        candidates = []
        for i in range(len(words)):
            # Single word
            candidates.append(words[i])
            # Two words
            if i < len(words) - 1:
                candidates.append(f"{words[i]} {words[i+1]}")
            # Three words
            if i < len(words) - 2:
                candidates.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        # Check against known patterns
        for candidate in candidates:
            # Direct match in aliases
            if candidate in self._app_name_cache:
                return self._app_name_cache[candidate]
                
            # Check common patterns
            if self._looks_like_app_name(candidate):
                app_info = {
                    'name': self._normalize_app_name(candidate),
                    'hints': ['detected_by_pattern'],
                    'confidence': 0.8
                }
                self._app_name_cache[candidate] = app_info
                return app_info
                
        return None
        
    def _extract_by_patterns(self, query: str, query_lower: str) -> Optional[Dict[str, Any]]:
        """Extract using regex patterns"""
        pattern_configs = [
            {
                'pattern': r'\b(?:the\s+)?([A-Z]\w+(?:\s+[A-Z]\w+)*)\s+(?:' + '|'.join(self._app_indicators['suffix']) + r')\b',
                'hint': 'app_suffix',
                'confidence': 0.9
            },
            {
                'pattern': r'(?:can\s+you\s+see\s+)?(?:the\s+)?([A-Z]\w+(?:\s+\w+)?)\s+(?:in|on)\s+',
                'hint': 'location_context',
                'confidence': 0.7
            },
            {
                'pattern': r'(?:' + '|'.join(self._presence_verbs) + r')\s+([A-Z]\w+(?:\s+\w+)?)\s+(?:open|running)',
                'hint': 'presence_check',
                'confidence': 0.8
            }
        ]
        
        for config in pattern_configs:
            match = re.search(config['pattern'], query)
            if match:
                app_name = match.group(1).strip()
                if self._validate_app_name(app_name):
                    return {
                        'name': app_name,
                        'hints': [config['hint']],
                        'confidence': config['confidence']
                    }
                    
        return None
        
    def _extract_by_context_clues(self, query: str, query_lower: str) -> Optional[Dict[str, Any]]:
        """Extract using contextual clues"""
        # Look for capitalized words near app indicators
        tokens = query.split()
        
        for i, token in enumerate(tokens):
            if token[0].isupper() and len(token) > 1:
                # Check surrounding context
                context_score = 0
                hints = []
                
                # Check previous word
                if i > 0:
                    prev = tokens[i-1].lower()
                    if prev in ['the', 'open', 'launch', 'start', 'find']:
                        context_score += 0.3
                        hints.append(f'preceded_by_{prev}')
                        
                # Check next word
                if i < len(tokens) - 1:
                    next_word = tokens[i+1].lower()
                    if next_word in self._app_indicators['suffix'] + self._app_indicators['context']:
                        context_score += 0.5
                        hints.append(f'followed_by_{next_word}')
                        
                # Multi-word app name check
                if i < len(tokens) - 1 and tokens[i+1][0].isupper():
                    potential_app = f"{token} {tokens[i+1]}"
                    if self._validate_app_name(potential_app):
                        return {
                            'name': potential_app,
                            'hints': hints + ['multi_word_caps'],
                            'confidence': min(0.9, 0.5 + context_score)
                        }
                        
                # Single word check
                if context_score > 0.2 and self._validate_app_name(token):
                    return {
                        'name': token,
                        'hints': hints,
                        'confidence': min(0.8, 0.4 + context_score)
                    }
                    
        return None
        
    def _extract_by_fuzzy_match(self, query: str, query_lower: str) -> Optional[Dict[str, Any]]:
        """Extract using fuzzy matching against known apps"""
        # Get all capitalized sequences
        cap_sequences = re.findall(r'\b[A-Z]\w+(?:\s+[A-Z]\w+)*\b', query)
        
        for sequence in cap_sequences:
            # Check similarity to known apps
            for known_app in self._common_app_words:
                similarity = difflib.SequenceMatcher(None, sequence.lower(), known_app.lower()).ratio()
                if similarity > 0.8:
                    return {
                        'name': sequence,
                        'hints': ['fuzzy_match', f'similar_to_{known_app}'],
                        'confidence': similarity * 0.7
                    }
                    
        return None
        
    def _calculate_match_confidence(self, match: re.Match, pattern: str, query_lower: str, query_type: str) -> float:
        """Calculate confidence score for a pattern match"""
        base_confidence = 0.7
        
        # Boost for exact phrase matches
        if match.group(0) == query_lower.strip():
            base_confidence += 0.2
            
        # Boost for specific query types
        type_boosts = {
            'location_query': 0.1,
            'all_spaces': 0.15,
            'space_content': 0.1
        }
        base_confidence += type_boosts.get(query_type, 0)
        
        # Boost for multiple space indicators
        space_indicators = sum(1 for term in self._space_terms if term in query_lower)
        if space_indicators > 1:
            base_confidence += 0.05 * (space_indicators - 1)
            
        return min(1.0, base_confidence)
        
    def _analyze_unmatched_query(self, query: str) -> SpaceQueryIntent:
        """Analyze queries that don't match patterns"""
        query_lower = query.lower()
        
        # Check for space-related keywords
        has_space_term = any(term in query_lower for term in self._space_terms)
        has_other_term = any(term in query_lower for term in self._other_terms)
        
        if has_space_term and has_other_term:
            return SpaceQueryIntent(
                query_type=SpaceQueryType.LOCATION_QUERY,
                confidence=0.5,
                metadata_sufficient=True,
                context_hints=['unmatched_but_spatial']
            )
            
        return SpaceQueryIntent(
            query_type=SpaceQueryType.SIMPLE_PRESENCE,
            confidence=0.3,
            metadata_sufficient=True,
            context_hints=['fallback']
        )
        
    def _extract_space_references(self, query: str, match: re.Match) -> Dict[str, Any]:
        """Extract space references from query"""
        query_lower = query.lower()
        refs = []
        
        # Check for numbered spaces
        space_nums = re.findall(r'\b(?:desktop|space|screen|workspace)\s+(\d+)\b', query_lower)
        if space_nums:
            return {
                'references': [f'space_{num}' for num in space_nums],
                'number': int(space_nums[0])
            }
            
        # Check for relative references
        for term in self._other_terms:
            if term in query_lower:
                refs.append(f'relative_{term}')
                
        # Check for all spaces
        for term in self._all_terms:
            if term in query_lower:
                refs.append(f'scope_{term}')
                
        return {'references': refs, 'number': None}
        
    def _analyze_requirements(self, query_type: str, query: str, intent: SpaceQueryIntent) -> Dict[str, bool]:
        """Analyze what data is required for the query"""
        query_lower = query.lower()
        
        # Visual indicators that need screenshots
        visual_keywords = {'show', 'display', 'see', 'look', 'view', 'check', 'examine', 'read', 'content'}
        needs_visual = any(kw in query_lower for kw in visual_keywords)
        
        # Detail requirements
        if query_type == 'specific_detail' or needs_visual:
            return {'screenshot': True, 'metadata': False}
        elif query_type in ['simple_presence', 'location_query']:
            return {'screenshot': False, 'metadata': True}
        elif query_type == 'space_content':
            # If asking to "show", need screenshot
            return {'screenshot': needs_visual, 'metadata': not needs_visual}
        else:
            return {'screenshot': False, 'metadata': True}
            
    def _looks_like_app_name(self, text: str) -> bool:
        """Check if text looks like an app name"""
        if not text or len(text) < 2:
            return False
            
        # Check capitalization patterns
        words = text.split()
        if all(w[0].isupper() for w in words if w):
            return True
            
        # Check for known patterns
        app_patterns = [
            r'.*(?:app|application|ide|editor|browser)$',
            r'^(?:microsoft|adobe|jetbrains|google)\s+',
        ]
        
        return any(re.match(p, text.lower()) for p in app_patterns)
        
    def _normalize_app_name(self, name: str) -> str:
        """Normalize app name to standard format"""
        # Handle common variations
        normalizations = {
            'vscode': 'Visual Studio Code',
            'vs code': 'Visual Studio Code',
            'chrome': 'Google Chrome',
            'iterm': 'iTerm2',
        }
        
        lower_name = name.lower()
        if lower_name in normalizations:
            return normalizations[lower_name]
            
        # Ensure proper capitalization
        return ' '.join(w.capitalize() for w in name.split())
        
    def _validate_app_name(self, name: str) -> bool:
        """Validate if extracted text is likely an app name"""
        if not name or len(name) < 2:
            return False
            
        # Skip common non-app words
        skip_words = {
            'what', 'the', 'on', 'in', 'is', 'are', 'show', 'me', 'can', 'you',
            'see', 'find', 'where', 'which', 'other', 'desktop', 'space'
        }
        
        if name.lower() in skip_words:
            return False
            
        # Must start with capital or be a known app
        return name[0].isupper() or self._looks_like_app_name(name)

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
2. When asked about "the other desktop space" or "another space", analyze ALL spaces except the current one
3. Specify which desktop/space contains what
4. Use natural space references: "Desktop 2", "your other space", etc.
5. If using cached or metadata info, subtly indicate freshness
6. For location queries, be specific about space number and what else is there
7. Never say you "can't see" other spaces - use available metadata
8. When user asks about apps in "other" spaces, provide specific space numbers and locations

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
    """Builds natural multi-space aware responses with dynamic generation"""
    
    def __init__(self):
        self._init_response_templates()
        self._init_contextual_phrases()
        
    def _init_response_templates(self):
        """Initialize dynamic response templates"""
        self.space_descriptors = [
            "Desktop {}", 
            "Space {}", 
            "your {} workspace",
            "the {} desktop",
            "workspace {}",
            "screen {}"
        ]
        
        self.location_templates = {
            'current': [
                "on your current desktop",
                "here on this screen",
                "in your active workspace",
                "on the desktop you're viewing"
            ],
            'other': [
                "on Desktop {}",
                "in Space {}",
                "over on workspace {}",
                "on your {} screen"
            ],
            'multiple': [
                "across {} desktops",
                "on multiple screens",
                "in several workspaces",
                "distributed across spaces"
            ]
        }
        
    def _init_contextual_phrases(self):
        """Initialize contextual phrase builders"""
        self.context_phrases = {
            'with_activity': {
                'single': "{app} is {location} with \"{title}\"",
                'multiple': "{app} has {count} windows {location}",
                'active': "{app} is actively {location}",
            },
            'state_modifiers': {
                'fullscreen': "in fullscreen mode",
                'minimized': "(minimized)",
                'hidden': "(hidden)",
                'focused': "with focus",
            },
            'companion_phrases': {
                'single': "alongside {apps}",
                'multiple': "along with {apps}",
                'working_with': "working with {apps}",
            }
        }
        
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
        """Determine if query needs multi-space handling with dynamic analysis"""
        intent = self.query_detector.detect_intent(query)
        query_lower = query.lower()
        
        # Decision factors with weights
        factors = []
        
        # Factor 1: Intent type suggests multi-space
        multi_space_intents = {
            SpaceQueryType.LOCATION_QUERY: 0.8,
            SpaceQueryType.ALL_SPACES: 1.0,
            SpaceQueryType.SPACE_CONTENT: 0.7,
            SpaceQueryType.WORKSPACE_OVERVIEW: 0.9
        }
        if intent.query_type in multi_space_intents:
            factors.append(('intent_type', multi_space_intents[intent.query_type]))
            
        # Factor 2: Has app target (looking for specific app)
        if intent.target_app:
            factors.append(('has_app_target', 0.7))
            
        # Factor 3: Has space target or references
        if intent.target_space or intent.space_references:
            factors.append(('has_space_ref', 0.9))
            
        # Factor 4: Dynamic keyword analysis
        keyword_score = self._calculate_keyword_score(query_lower)
        if keyword_score > 0:
            factors.append(('keywords', keyword_score))
            
        # Factor 5: Pattern complexity
        pattern_score = self._analyze_pattern_complexity(query_lower)
        if pattern_score > 0:
            factors.append(('patterns', pattern_score))
            
        # Factor 6: Context hints suggest multi-space
        if any('spatial' in hint for hint in intent.context_hints):
            factors.append(('context_hints', 0.6))
            
        # Calculate weighted decision
        if not factors:
            # Check for simple presence queries that might still need multi-space
            return self._check_simple_presence_override(query_lower, intent)
            
        # Use highest factor or combination threshold
        max_score = max(score for _, score in factors) if factors else 0
        total_score = sum(score for _, score in factors)
        
        return max_score >= 0.7 or total_score >= 1.2
        
    def _calculate_keyword_score(self, query_lower: str) -> float:
        """Calculate keyword-based score for multi-space detection"""
        score = 0.0
        
        # Dynamic keyword categories with weights
        keyword_categories = {
            'location': (self.query_detector._location_verbs, 0.3),
            'spatial': (self.query_detector._space_terms, 0.2),
            'other': (self.query_detector._other_terms, 0.4),
            'all': (self.query_detector._all_terms, 0.5),
        }
        
        for category, (terms, weight) in keyword_categories.items():
            matches = sum(1 for term in terms if term in query_lower)
            if matches:
                score += weight * min(matches, 2)  # Cap contribution
                
        return min(score, 1.0)
        
    def _analyze_pattern_complexity(self, query_lower: str) -> float:
        """Analyze query pattern complexity"""
        score = 0.0
        
        # Complex patterns that suggest multi-space
        complex_patterns = [
            # Cross-space references
            (r'\b(?:across|between|among)\s+(?:\w+\s+)?(?:spaces?|desktops?)', 0.8),
            # Comparative queries  
            (r'\b(?:compare|difference|both|either)\s+', 0.6),
            # Navigation queries
            (r'\b(?:switch|move|go|jump)\s+(?:to|between)\s+', 0.7),
            # Listing queries
            (r'\b(?:list|show|display)\s+(?:all|every)\s+', 0.7),
        ]
        
        for pattern, weight in complex_patterns:
            if re.search(pattern, query_lower):
                score += weight
                
        return min(score, 1.0)
        
    def _check_simple_presence_override(self, query_lower: str, intent: SpaceQueryIntent) -> bool:
        """Check if simple presence query should use multi-space"""
        # Even simple "Is X open?" might need multi-space if:
        
        # 1. Query implies checking everywhere
        if any(term in query_lower for term in ['anywhere', 'somewhere', 'everywhere']):
            return True
            
        # 2. Query has uncertainty markers
        if any(term in query_lower for term in ['might', 'could', 'possibly', 'maybe']):
            return True
            
        # 3. Has app target (user asking about specific app)
        if intent.target_app:
            return True
            
        return False
        
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
            else (isinstance(space, dict) and (hasattr(space, 'cached_screenshot') and space.cached_screenshot) if hasattr(space, 'cached_screenshot') else (isinstance(space, dict) and space.get('cached_screenshot')) is not None)
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
            if intent.target_app:
                target_windows = [
                    w for w in windows 
                    if intent.target_app.lower() in (
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
            else:
                # Query about "other desktop" without specific app
                current_space_id = window_data.get('current_space', {}).get('id', 1)
                other_spaces = [
                    space for space in window_data.get('spaces', [])
                    if (hasattr(space, 'space_id') and space.space_id != current_space_id) or
                       (isinstance(space, dict) and space.get('space_id') != current_space_id)
                ]
                
                if other_spaces:
                    # Build response about other spaces
                    response_parts = []
                    for space in other_spaces:
                        space_id = space.space_id if hasattr(space, 'space_id') else space.get('space_id')
                        apps = self._get_space_applications(space_id, window_data)
                        if apps:
                            app_list = list(apps.keys())
                            response_parts.append(
                                f"Desktop {space_id} has {self.response_builder._format_app_list(app_list)}"
                            )
                    
                    if response_parts:
                        return ". ".join(response_parts) + "."
                    else:
                        return "The other desktops appear to be empty."
                else:
                    return "You only have one desktop active."
                
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