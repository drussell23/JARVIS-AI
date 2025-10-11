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
import numpy as np
import hashlib

logger = logging.getLogger(__name__)


class SpaceQueryType(Enum):
    """Types of multi-space queries"""

    SIMPLE_PRESENCE = "simple_presence"  # "Is X open?"
    LOCATION_QUERY = "location_query"  # "Where is X?"
    SPACE_CONTENT = "space_content"  # "What's on Desktop 2?"
    ALL_SPACES = "all_spaces"  # "Show me all my workspaces"
    SPECIFIC_DETAIL = "specific_detail"  # "What's the error in VSCode?"
    WORKSPACE_OVERVIEW = "workspace_overview"  # "What am I working on?"


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
        self._space_terms = {
            "desktop",
            "space",
            "screen",
            "workspace",
            "monitor",
            "display",
            "area",
            "environment",
            "window",
        }
        self._location_verbs = {
            "where",
            "which",
            "find",
            "locate",
            "search",
            "look",
            "check",
        }
        self._presence_verbs = {
            "is",
            "are",
            "have",
            "has",
            "got",
            "running",
            "open",
            "active",
            "visible",
            "showing",
        }
        self._other_terms = {
            "other",
            "another",
            "different",
            "alternate",
            "secondary",
            "next",
            "previous",
            "adjacent",
        }
        self._all_terms = {
            "all",
            "every",
            "each",
            "entire",
            "whole",
            "complete",
            "full",
        }
        
        # Build dynamic patterns
        self._build_dynamic_patterns()
        
        # Application detection patterns
        self._app_indicators = {
            "suffix": [
                "app",
                "application",
                "program",
                "software",
                "tool",
                "ide",
                "editor",
                "browser",
                "client",
            ],
            "context": ["window", "instance", "session", "process"],
        }
        
        # Initialize app name cache
        self._app_name_cache = {}
        self._common_app_words = self._build_common_app_words()
        
    def _build_dynamic_patterns(self):
        """Build patterns dynamically from components"""
        space_alt = "|".join(self._space_terms)
        loc_alt = "|".join(self._location_verbs)
        pres_alt = "|".join(self._presence_verbs)
        other_alt = "|".join(self._other_terms)
        all_alt = "|".join(self._all_terms)
        
        self.patterns = {
            "simple_presence": self._generate_presence_patterns(pres_alt),
            "location_query": self._generate_location_patterns(
                loc_alt, space_alt, other_alt
            ),
            "space_content": self._generate_space_content_patterns(
                space_alt, other_alt
            ),
            "all_spaces": self._generate_all_spaces_patterns(all_alt, space_alt),
            "specific_detail": self._generate_detail_patterns(),
            "workspace_overview": self._generate_overview_patterns(),
        }
        
    def _generate_presence_patterns(self, pres_alt):
        """Generate presence detection patterns"""
        return [
            rf"\b({pres_alt})\s+(\S+(?:\s+\S+)?)\s+(open|running|active|visible)\b",
            rf"\b(do|does)\s+\S+\s+have\s+(\S+(?:\s+\S+)?)\s+open\b",
            rf"\b(\S+(?:\s+\S+)?)\s+({pres_alt})\s*\?",
            rf"\bcan\s+(?:you\s+)?(?:see|find)\s+(\S+(?:\s+\S+)?)\b",
        ]
        
    def _generate_location_patterns(self, loc_alt, space_alt, other_alt):
        """Generate location query patterns"""
        return [
            rf"\b({loc_alt})\s+(?:is|are)\s+(\S+(?:\s+\S+)?)",
            rf"\b({loc_alt})\s+(?:can\s+)?(?:I|you)\s+find\s+(\S+(?:\s+\S+)?)",
            rf"\b(?:on\s+)?which\s+({space_alt})\s+(?:is|are)\s+(\S+(?:\s+\S+)?)",
            rf"\b(\S+(?:\s+\S+)?)\s+(?:in|on)\s+(?:the\s+)?({other_alt})\s+({space_alt})",
            rf"\bcan\s+you\s+see\s+(?:if\s+)?(\S+(?:\s+\S+)?)\s+.*\s+({other_alt})\s+({space_alt})",
            rf"\b(?:show|tell)\s+me\s+where\s+(\S+(?:\s+\S+)?)\s+is",
        ]
        
    def _generate_space_content_patterns(self, space_alt, other_alt):
        """Generate space content patterns"""
        return [
            rf"\bwhat(?:\'s|s| is)\s+(?:on|in)\s+({space_alt})\s+(\d+)",
            rf"\b({space_alt})\s+(\d+)\s+(?:content|contents|has|shows)",
            rf"\bshow\s+(?:me\s+)?({space_alt})\s+(\d+)",
            rf"\bwhat(?:\'s|s| is)\s+(?:on|in)\s+(?:the\s+)?({other_alt})\s+({space_alt})",
            rf"\b(?:display|show|list)\s+(?:the\s+)?({other_alt})\s+({space_alt})",
            rf"\bwhat\s+do\s+(?:I|you)\s+(?:have|see)\s+(?:on|in)\s+(?:the\s+)?({other_alt})",
        ]
        
    def _generate_all_spaces_patterns(self, all_alt, space_alt):
        """Generate all spaces patterns"""
        return [
            rf"\b({all_alt})\s+(?:my\s+)?({space_alt})s?\b",
            rf"\b(?:show|list|display)\s+(?:me\s+)?everything\s+(?:that(?:\'s|s)?\s+)?(?:open|running)",
            rf"\b({space_alt})\s+(?:overview|summary|status)",
            rf"\bwhat(?:\'s|s| is)\s+(?:on|in)\s+({all_alt})",
            rf"\b(?:across|throughout)\s+(?:all\s+)?(?:my\s+)?({space_alt})s?",
        ]
        
    def _generate_detail_patterns(self):
        """Generate specific detail patterns"""
        return [
            r"\b(?:read|show|display)\s+(?:the\s+)?(\S+)\s+(?:in|on|from)\s+(\S+)",
            r"\b(?:error|warning|message|alert)\s+(?:in|on|from)\s+(\S+)",
            r"\bwhat\s+(?:does|says)\s+(\S+)\s+(?:say|show|display)",
            r"\b(?:content|contents|text)\s+(?:of|in|from)\s+(\S+)",
            r"\b(?:check|examine|inspect)\s+(\S+)\s+in\s+(\S+)",
        ]
        
    def _generate_overview_patterns(self):
        """Generate workspace overview patterns"""
        return [
            r"\bwhat\s+am\s+I\s+(?:working|doing|focused)\s+on",
            r"\b(?:my|current)\s+(?:work|tasks?|projects?|activities)",
            r"\b(?:workspace|desktop|screen)\s+(?:status|state|overview)",
            r"\b(?:active|current|ongoing)\s+(?:work|projects?|tasks?)",
            r"\b(?:show|display|list)\s+(?:my\s+)?(?:current\s+)?(?:work|activity)",
        ]
        
    def _build_common_app_words(self):
        """Build set of common application-related words"""
        words = set()
        
        # Common app name patterns
        tech_terms = {"visual", "studio", "code", "android", "web", "dev", "tools"}
        generic_terms = {"pro", "plus", "lite", "express", "community", "professional"}
        
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
                    matches.append(
                        {
                            "type": query_type,
                            "match": match,
                            "pattern": pattern,
                            "confidence": confidence,
                            "priority": pattern_idx,
                        }
                    )
        
        # Select best match based on confidence and priority
        if matches:
            best_match = max(matches, key=lambda x: (x["confidence"], -x["priority"]))
            return self._build_intent(
                best_match["type"],
                best_match["match"],
                query,
                confidence=best_match["confidence"],
            )
                    
        # Default with context analysis
        return self._analyze_unmatched_query(query)
        
    def _build_intent(
        self, query_type: str, match: re.Match, query: str, confidence: float = 1.0
    ) -> SpaceQueryIntent:
        """Build intent from regex match with enhanced context"""
        intent = SpaceQueryIntent(
            query_type=SpaceQueryType[query_type.upper()], confidence=confidence
        )
        
        # Extract app name with context
        app_info = self._extract_app_with_context(query)
        if app_info:
            intent.target_app = app_info["name"]
            intent.context_hints.extend(app_info.get("hints", []))
        
        # Extract space references
        space_refs = self._extract_space_references(query, match)
        intent.space_references = space_refs["references"]
        if space_refs.get("number"):
            intent.target_space = space_refs["number"]
                
        # Determine requirements based on query analysis
        requirements = self._analyze_requirements(query_type, query, intent)
        intent.requires_screenshot = requirements["screenshot"]
        intent.metadata_sufficient = requirements["metadata"]
        
        # Add detected patterns for transparency
        intent.detected_patterns.append(match.re.pattern)
            
        return intent
        
    def extract_app_name(self, query: str) -> Optional[str]:
        """Legacy method - delegates to new context-aware extraction"""
        app_info = self._extract_app_with_context(query)
        return app_info["name"] if app_info else None
        
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
            self._extract_by_fuzzy_match,
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
        
    def _extract_by_known_apps(
        self, query: str, query_lower: str
    ) -> Optional[Dict[str, Any]]:
        """Extract using known application database"""
        # Check for common apps directly in query - return info
        common_apps = {
            "terminal": [
                "terminal",
                "term",
                "shell",
                "bash",
                "zsh",
                "command line",
            ],  # Common terminal keywords for detection
            "chrome": [
                "chrome",
                "google chrome",
            ],  # Common chrome keywords for detection
            "safari": ["safari"],  # Common safari keywords for detection
            "firefox": ["firefox"],  # Common firefox keywords for detection
            "vscode": [
                "vscode",
                "vs code",
                "visual studio code",
                "code editor",
            ],  # Common vscode keywords for detection
        }

        # Check for common app keywords in query - return info
        for app_name, keywords in common_apps.items(): 
            for (
                keyword
            ) in keywords:  # Check each keyword in the list of keywords for the app
                if (
                    keyword in query_lower
                ):  # Match found in query lower case - return info
                    # Build app info and return immediately with confidence 0.95
                    app_info = {
                        "name": app_name.capitalize(),  # Capitalize app name for consistency with other app info
                        "hints": [
                            "common_app_detected"
                        ],  # Add hint for common app detection
                        "confidence": 0.95,  # Set confidence to 0.95 for common app detection
                    }
                    return app_info  # Return app info with confidence 0.95 immediately upon match

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
                    "name": self._normalize_app_name(candidate),
                    "hints": ["detected_by_pattern"],
                    "confidence": 0.8,
                }
                self._app_name_cache[candidate] = app_info
                return app_info
                
        return None
        
    def _extract_by_patterns(
        self, query: str, query_lower: str
    ) -> Optional[Dict[str, Any]]:
        """Extract using regex patterns"""
        pattern_configs = [
            {
                "pattern": r"\b(?:the\s+)?([A-Z]\w+(?:\s+[A-Z]\w+)*)\s+(?:"
                + "|".join(self._app_indicators["suffix"])
                + r")\b",
                "hint": "app_suffix",
                "confidence": 0.9,
            },
            {
                "pattern": r"(?:can\s+you\s+see\s+)?(?:the\s+)?([A-Z]\w+(?:\s+\w+)?)\s+(?:in|on)\s+",
                "hint": "location_context",
                "confidence": 0.7,
            },
            {
                "pattern": r"(?:"
                + "|".join(self._presence_verbs)
                + r")\s+([A-Z]\w+(?:\s+\w+)?)\s+(?:open|running)",
                "hint": "presence_check",
                "confidence": 0.8,
            },
        ]
        
        for config in pattern_configs:
            match = re.search(config["pattern"], query)
            if match:
                app_name = match.group(1).strip()
                if self._validate_app_name(app_name):
                    return {
                        "name": app_name,
                        "hints": [config["hint"]],
                        "confidence": config["confidence"],
                    }
                    
        return None
        
    def _extract_by_context_clues(
        self, query: str, query_lower: str
    ) -> Optional[Dict[str, Any]]:
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
                    prev = tokens[i - 1].lower()
                    if prev in ["the", "open", "launch", "start", "find"]:
                        context_score += 0.3
                        hints.append(f"preceded_by_{prev}")
                        
                # Check next word
                if i < len(tokens) - 1:
                    next_word = tokens[i + 1].lower()
                    if (
                        next_word
                        in self._app_indicators["suffix"]
                        + self._app_indicators["context"]
                    ):
                        context_score += 0.5
                        hints.append(f"followed_by_{next_word}")
                        
                # Multi-word app name check
                if i < len(tokens) - 1 and tokens[i + 1][0].isupper():
                    potential_app = f"{token} {tokens[i+1]}"
                    if self._validate_app_name(potential_app):
                        return {
                            "name": potential_app,
                            "hints": hints + ["multi_word_caps"],
                            "confidence": min(0.9, 0.5 + context_score),
                        }
                        
                # Single word check
                if context_score > 0.2 and self._validate_app_name(token):
                    return {
                        "name": token,
                        "hints": hints,
                        "confidence": min(0.8, 0.4 + context_score),
                    }
                    
        return None
        
    def _extract_by_fuzzy_match(
        self, query: str, query_lower: str
    ) -> Optional[Dict[str, Any]]:
        """Extract using fuzzy matching against known apps"""
        # Get all capitalized sequences
        cap_sequences = re.findall(r"\b[A-Z]\w+(?:\s+[A-Z]\w+)*\b", query)
        
        for sequence in cap_sequences:
            # Check similarity to known apps
            for known_app in self._common_app_words:
                similarity = difflib.SequenceMatcher(
                    None, sequence.lower(), known_app.lower()
                ).ratio()
                if similarity > 0.8:
                    return {
                        "name": sequence,
                        "hints": ["fuzzy_match", f"similar_to_{known_app}"],
                        "confidence": similarity * 0.7,
                    }
                    
        return None
        
    def _calculate_match_confidence(
        self, match: re.Match, pattern: str, query_lower: str, query_type: str
    ) -> float:
        """Calculate confidence score for a pattern match"""
        base_confidence = 0.7
        
        # Boost for exact phrase matches
        if match.group(0) == query_lower.strip():
            base_confidence += 0.2
            
        # Boost for specific query types
        type_boosts = {"location_query": 0.1, "all_spaces": 0.15, "space_content": 0.1}
        base_confidence += type_boosts.get(query_type, 0)
        
        # Boost for multiple space indicators
        space_indicators = sum(1 for term in self._space_terms if term in query_lower)
        if space_indicators > 1:
            base_confidence += 0.05 * (space_indicators - 1)
            
        return min(1.0, base_confidence)
        
    def _analyze_unmatched_query(self, query: str) -> SpaceQueryIntent:
        """Analyze queries that don't match patterns - now more comprehensive"""
        query_lower = query.lower()

        # Check for space-related keywords
        has_space_term = any(term in query_lower for term in self._space_terms)
        has_other_term = any(term in query_lower for term in self._other_terms)

        # Dynamic app detection - extract any app names mentioned
        detected_apps = self._extract_app_mentions(query_lower)

        # Check for visual/content keywords that need screenshots
        visual_keywords = {
            "see",
            "look",
            "show",
            "display",
            "view",
            "check",
            "examine",
            "read",
            "content",
            "what",
            "how",
            "where",
            "tell me",
            "explain",
            "analyze",
            "describe",
            "details",
            "information",
        }
        needs_visual = any(word in query_lower for word in visual_keywords)

        # Check for activity keywords that suggest multi-space capture
        activity_keywords = {
            "working",
            "doing",
            "using",
            "running",
            "open",
            "active",
            "happening",
            "going on",
            "status",
            "current",
            "right now",
        }
        mentions_activity = any(word in query_lower for word in activity_keywords)

        # Enhanced logic for when to trigger multi-space capture
        should_capture_all_spaces = (
            (has_space_term and has_other_term)  # "other desktop"
            or (detected_apps and needs_visual)  # "see my terminal"
            or (mentions_activity and has_space_term)  # "what am I working on"
            or (needs_visual and not detected_apps)  # "what's happening"
        )

        if should_capture_all_spaces:
            return SpaceQueryIntent(
                query_type=SpaceQueryType.ALL_SPACES,
                confidence=0.7,
                metadata_sufficient=True,
                context_hints=["comprehensive_analysis", "visual_content_needed"],
                space_references=["all_spaces"],
                detected_patterns=[
                    (
                        f"detected_apps: {detected_apps}"
                        if detected_apps
                        else "visual_analysis"
                    )
                ],
            )

        # If query mentions specific apps, treat as location query
        if detected_apps:
            return SpaceQueryIntent(
                query_type=SpaceQueryType.LOCATION_QUERY,
                confidence=0.6,
                metadata_sufficient=True,
                context_hints=["app_mention", f"detected_apps: {detected_apps}"],
                space_references=["all_spaces"],
                detected_patterns=[f"app_mention: {detected_apps}"],
            )

        if has_space_term and has_other_term:
            return SpaceQueryIntent(
                query_type=SpaceQueryType.LOCATION_QUERY,
                confidence=0.5,
                metadata_sufficient=True,
                context_hints=["spatial_reference"],
            )

        return SpaceQueryIntent(
            query_type=SpaceQueryType.SIMPLE_PRESENCE,
            confidence=0.3,
            metadata_sufficient=True,
            context_hints=["fallback"],
        )

    def _extract_app_mentions(self, query_lower: str) -> List[str]:
        """Extract any application names mentioned in the query"""
        # Common app name patterns
        app_indicators = {
            "browsers": ["chrome", "safari", "firefox", "edge", "brave", "opera"],
            "editors": [
                "vscode",
                "code",
                "atom",
                "sublime",
                "vim",
                "emacs",
                "textedit",
            ],
            "terminals": ["terminal", "iterm", "iterm2", "alacritty", "kitty", "hyper"],
            "communication": ["slack", "teams", "discord", "zoom", "skype", "facetime"],
            "productivity": ["notes", "reminders", "calendar", "mail", "outlook"],
            "development": [
                "xcode",
                "android studio",
                "intellij",
                "pycharm",
                "webstorm",
            ],
            "design": ["photoshop", "illustrator", "figma", "sketch", "xd"],
            "music": ["spotify", "apple music", "music", "itunes"],
            "video": ["vlc", "quicktime", "youtube", "netflix"],
        }

        detected_apps = []

        # Check each category
        for category, apps in app_indicators.items():
            for app in apps:
                if app in query_lower:
                    detected_apps.append(app)

        # Also check for generic app patterns (AppName, MyApp, etc.)
        generic_patterns = [
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",  # Capitalized words
        ]

        for pattern in generic_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                # Filter out common words that aren't app names
                if len(match) > 2 and match.lower() not in {
                    "what",
                    "where",
                    "when",
                    "which",
                    "tell",
                    "show",
                    "look",
                    "help",
                    "find",
                    "open",
                    "close",
                }:
                    detected_apps.append(match.lower())

        return list(set(detected_apps))  # Remove duplicates

    def _extract_space_references(self, query: str, match: re.Match) -> Dict[str, Any]:
        """Extract space references from query"""
        query_lower = query.lower()
        refs = []
        
        # Check for numbered spaces
        space_nums = re.findall(
            r"\b(?:desktop|space|screen|workspace)\s+(\d+)\b", query_lower
        )
        if space_nums:
            return {
                "references": [f"space_{num}" for num in space_nums],
                "number": int(space_nums[0]),
            }
            
        # Check for relative references
        for term in self._other_terms:
            if term in query_lower:
                refs.append(f"relative_{term}")
                
        # Check for all spaces
        for term in self._all_terms:
            if term in query_lower:
                refs.append(f"scope_{term}")
                
        return {"references": refs, "number": None}
        
    def _analyze_requirements(
        self, query_type: str, query: str, intent: SpaceQueryIntent
    ) -> Dict[str, bool]:
        """Analyze what data is required for the query"""
        query_lower = query.lower()
        
        # Visual indicators that need screenshots
        visual_keywords = {
            "show",
            "display",
            "see",
            "look",
            "view",
            "check",
            "examine",
            "read",
            "content",
        }
        needs_visual = any(kw in query_lower for kw in visual_keywords)
        
        # Detail requirements
        if query_type == "specific_detail" or needs_visual:
            return {"screenshot": True, "metadata": False}
        elif query_type in ["simple_presence", "location_query"]:
            return {"screenshot": False, "metadata": True}
        elif query_type == "space_content":
            # If asking to "show", need screenshot
            return {"screenshot": needs_visual, "metadata": not needs_visual}
        else:
            return {"screenshot": False, "metadata": True}
            
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
            r".*(?:app|application|ide|editor|browser)$",
            r"^(?:microsoft|adobe|jetbrains|google)\s+",
        ]
        
        return any(re.match(p, text.lower()) for p in app_patterns)
        
    def _normalize_app_name(self, name: str) -> str:
        """Normalize app name to standard format"""
        # Handle common variations
        normalizations = {
            "vscode": "Visual Studio Code",
            "vs code": "Visual Studio Code",
            "chrome": "Google Chrome",
            "iterm": "iTerm2",
        }
        
        lower_name = name.lower()
        if lower_name in normalizations:
            return normalizations[lower_name]
            
        # Ensure proper capitalization
        return " ".join(w.capitalize() for w in name.split())
        
    def _validate_app_name(self, name: str) -> bool:
        """Validate if extracted text is likely an app name"""
        if not name or len(name) < 2:
            return False
            
        # Skip common non-app words
        skip_words = {
            "what",
            "the",
            "on",
            "in",
            "is",
            "are",
            "show",
            "me",
            "can",
            "you",
            "see",
            "find",
            "where",
            "which",
            "other",
            "desktop",
            "space",
        }
        
        if name.lower() in skip_words:
            return False
            
        # Must start with capital or be a known app
        return name[0].isupper() or self._looks_like_app_name(name)


class SpaceAwarePromptEnhancer:
    """Enhances prompts with multi-space context"""
    
    def __init__(self):
        self.confidence_levels = {
            "screenshot": "certain",
            "recent_cache": "confident",
            "metadata": "based on window information",
            "stale_cache": "from earlier observation",
            "inference": "likely",
        }

    def enhance_prompt(
        self,
                      base_prompt: str, 
                      query_intent: SpaceQueryIntent,
        space_data: Dict[str, Any],
    ) -> str:
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
        current_space = space_data.get("current_space", {})
        context_parts.append(
            f"You're currently on Desktop {current_space.get('id', 1)} "
            f"with {current_space.get('window_count', 0)} windows"
        )
        
        # Other spaces
        spaces = space_data.get("spaces", [])
        if len(spaces) > 1:
            context_parts.append(f"Total {len(spaces)} desktops active")
            
        # Window distribution
        space_window_map = space_data.get("space_window_map", {})
        for space_id, window_ids in space_window_map.items():
            if space_id != current_space.get("id"):
                context_parts.append(f"Desktop {space_id}: {len(window_ids)} windows")
                
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
            "screen {}",
        ]
        
        self.location_templates = {
            "current": [
                "on your current desktop",
                "here on this screen",
                "in your active workspace",
                "on the desktop you're viewing",
            ],
            "other": [
                "on Desktop {}",
                "in Space {}",
                "over on workspace {}",
                "on your {} screen",
            ],
            "multiple": [
                "across {} desktops",
                "on multiple screens",
                "in several workspaces",
                "distributed across spaces",
            ],
        }
        
    def _init_contextual_phrases(self):
        """Initialize contextual phrase builders"""
        self.context_phrases = {
            "with_activity": {
                "single": '{app} is {location} with "{title}"',
                "multiple": "{app} has {count} windows {location}",
                "active": "{app} is actively {location}",
            },
            "state_modifiers": {
                "fullscreen": "in fullscreen mode",
                "minimized": "(minimized)",
                "hidden": "(hidden)",
                "focused": "with focus",
            },
            "companion_phrases": {
                "single": "alongside {apps}",
                "multiple": "along with {apps}",
                "working_with": "working with {apps}",
            },
        }

    def build_location_response(
        self, app_name: str, window_info: Dict[str, Any], confidence: str = "certain"
    ) -> str:
        """Build response for app location query"""
        
        # Handle both object and dict formats
        if hasattr(window_info, "space_id"):
            space_id = window_info.space_id
            is_current = getattr(window_info, "is_current_space", False)
        else:
            # It's a dict
            space_id = window_info.get("space_id", 1)
            is_current = window_info.get("is_current_space", False)
        
        # Build base response
        if is_current:
            location = "on your current desktop"
        else:
            location = f"on Desktop {space_id}"
            
        response_parts = [f"{app_name} is {location}"]
        
        # Add context if available
        # Handle window attributes
        if hasattr(window_info, "window_title"):
            window_title = window_info.window_title
            is_fullscreen = getattr(window_info, "is_fullscreen", False)
            is_minimized = getattr(window_info, "is_minimized", False)
            companion_apps = getattr(window_info, "companion_apps", [])
        else:
            # It's a dict
            window_title = window_info.get("window_title", "")
            is_fullscreen = window_info.get("is_fullscreen", False)
            is_minimized = window_info.get("is_minimized", False)
            companion_apps = window_info.get("companion_apps", [])
        
        if window_title:
            response_parts.append(f'with "{window_title}"')
            
        if is_fullscreen:
            response_parts.append("in fullscreen mode")
        elif is_minimized:
            response_parts.append("(minimized)")
            
        # Add companion apps if known
        if companion_apps:
            response_parts.append(f"alongside {self._format_app_list(companion_apps)}")
            
        return " ".join(response_parts) + "."
        
    def build_space_overview(
        self,
                           space_id: int,
                           space_summary: Dict[str, Any],
        include_screenshot: bool = False,
    ) -> str:
        """Build overview of a specific space"""
        
        if not space_summary.get("applications"):
            return f"Desktop {space_id} appears to be empty."
            
        # Build application summary
        app_descriptions = []
        for app, windows in space_summary["applications"].items():
            if len(windows) == 1:
                app_descriptions.append(f"{app} ({windows[0]})")
            else:
                app_descriptions.append(f"{app} ({len(windows)} windows)")
                
        # Format response
        if include_screenshot:
            prefix = f"I can see Desktop {space_id} has"
        else:
            prefix = f"Desktop {space_id} has"
            
        return f"{prefix}: {self._format_app_list(app_descriptions)}."
        
    def build_workspace_overview(self, all_spaces: List[Dict[str, Any]]) -> str:
        """Build complete workspace overview"""
        
        overview_parts = [f"You have {len(all_spaces)} desktops active:"]
        
        for space in all_spaces:
            space_id = space["space_id"] if isinstance(space, dict) else space.space_id
            applications = (
                space["applications"]
                if isinstance(space, dict)
                else getattr(space, "applications", {})
            )
            is_current = (
                space["is_current"]
                if isinstance(space, dict)
                else getattr(space, "is_current", False)
            )
            
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
        
        if any(
            dev_app in app_names
            for dev_app in ["Visual Studio Code", "Xcode", "Terminal"]
        ):
            return "Development work"
        elif any(comm_app in app_names for comm_app in ["Slack", "Messages", "Mail"]):
            return "Communication"
        elif any(browser in app_names for browser in ["Safari", "Chrome", "Firefox"]):
            return "Web browsing/research"
        else:
            # Default to listing main apps
            return self._format_app_list(app_names[:2])


# Enhanced content analysis system
class DynamicContentAnalyzer:
    """Analyzes content from any application type dynamically"""

    def __init__(self):
        self.app_analyzers = self._init_app_analyzers()

    def _init_app_analyzers(self) -> Dict[str, callable]:
        """Initialize analyzers for different app types"""
        return {
            "terminal": self._analyze_terminal_content,
            "browser": self._analyze_browser_content,
            "editor": self._analyze_editor_content,
            "communication": self._analyze_communication_content,
            "productivity": self._analyze_productivity_content,
            "design": self._analyze_design_content,
            "music": self._analyze_music_content,
            "video": self._analyze_video_content,
            "system": self._analyze_system_content,
            "unknown": self._analyze_generic_content,
        }

    def analyze_window_content(
        self,
        app_name: str,
        window_title: str,
        screenshot: np.ndarray,
        ocr_text: str = None,
    ) -> Dict[str, Any]:
        """Analyze content from any window dynamically"""
        # Classify app type
        app_type = self._classify_app_type(app_name, window_title)

        # Get appropriate analyzer
        analyzer = self.app_analyzers.get(app_type, self.app_analyzers["unknown"])

        # Analyze content
        analysis = analyzer(app_name, window_title, screenshot, ocr_text)

        # Add metadata
        analysis.update(
            {
                "app_name": app_name,
                "window_title": window_title,
                "app_type": app_type,
                "analysis_timestamp": datetime.now(),
                "content_hash": hashlib.md5(
                    ocr_text.encode() if ocr_text else str(screenshot).encode()
                ).hexdigest()[:12],
            }
        )

        return analysis

    def _classify_app_type(self, app_name: str, window_title: str) -> str:
        """Dynamically classify application type"""
        name_lower = (app_name + " " + window_title).lower()

        # Classification rules
        classifications = {
            "terminal": [
                "terminal",
                "iterm",
                "alacritty",
                "kitty",
                "hyper",
                "bash",
                "zsh",
                "fish",
            ],
            "browser": [
                "chrome",
                "safari",
                "firefox",
                "edge",
                "brave",
                "opera",
                "browser",
            ],
            "editor": [
                "vscode",
                "code",
                "atom",
                "sublime",
                "vim",
                "emacs",
                "textedit",
                "xcode",
                "intellij",
                "pycharm",
                "webstorm",
            ],
            "communication": [
                "slack",
                "teams",
                "discord",
                "zoom",
                "skype",
                "facetime",
                "messages",
                "mail",
            ],
            "productivity": [
                "notes",
                "reminders",
                "calendar",
                "outlook",
                "onenote",
                "evernote",
            ],
            "design": [
                "photoshop",
                "illustrator",
                "figma",
                "sketch",
                "xd",
                "lightroom",
                "after effects",
            ],
            "music": ["spotify", "apple music", "music", "itunes", "vlc"],
            "video": ["youtube", "netflix", "quicktime", "vlc", "plex"],
            "system": [
                "finder",
                "system preferences",
                "activity monitor",
                "console",
                "terminal",
            ],
        }

        for app_type, keywords in classifications.items():
            if any(keyword in name_lower for keyword in keywords):
                return app_type

        return "unknown"

    def _analyze_terminal_content(
        self, app_name: str, window_title: str, screenshot: np.ndarray, ocr_text: str
    ) -> Dict[str, Any]:
        """Analyze terminal content"""
        analysis = {
            "content_type": "terminal",
            "commands": [],
            "errors": [],
            "warnings": [],
            "current_directory": None,
            "shell_type": "unknown",
            "exit_code": None,
            "is_running_command": False,
            "recent_output": None,
        }

        if not ocr_text:
            return analysis

        lines = ocr_text.strip().split("\n")

        # Detect shell type
        if "$ " in ocr_text or "$" in ocr_text:
            analysis["shell_type"] = "bash/zsh"
        elif "%" in ocr_text:
            analysis["shell_type"] = "fish"

        # Extract commands (lines starting with $ or %)
        for line in lines:
            line = line.strip()
            if line.startswith("$") or line.startswith("%"):
                command = line[1:].strip()
                if command and len(command) > 1:
                    analysis["commands"].append(command)

        # Detect errors
        error_patterns = [
            r"error:",
            r"Error:",
            r"ERROR:",
            r"traceback",
            r"Traceback",
            r"TRACEBACK",
            r"Exception:",
            r"exception:",
            r"failed",
            r"Failed",
            r"FAILED",
            r"not found",
            r"Not found",
            r"NOT FOUND",
            r"permission denied",
            r"Permission denied",
            r"command not found",
            r"Command not found",
        ]

        for line in lines:
            for pattern in error_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    if line not in analysis["errors"]:
                        analysis["errors"].append(line.strip())

        # Detect warnings
        warning_patterns = [
            r"warning:",
            r"Warning:",
            r"WARNING:",
            r"deprecated",
            r"Deprecated",
            r"note:",
            r"Note:",
        ]

        for line in lines:
            for pattern in warning_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    if line not in analysis["warnings"]:
                        analysis["warnings"].append(line.strip())

        # Detect current directory (usually in prompt)
        dir_patterns = [
            r"\$ .*?(/.*?)(?:\s*\$|#|$)",  # $ /path/to/dir $
            r"% .*?(/.*?)(?:\s*%|$)",  # % /path/to/dir %
        ]

        for line in lines:
            for pattern in dir_patterns:
                match = re.search(pattern, line)
                if match:
                    potential_dir = match.group(1).strip()
                    if potential_dir.startswith("/") and len(potential_dir) > 1:
                        analysis["current_directory"] = potential_dir
                        break

        # Check if running command (cursor at end)
        if lines and (lines[-1].endswith("$ ") or lines[-1].endswith("% ")):
            analysis["is_running_command"] = True

        # Get recent output (last few lines before prompt)
        analysis["recent_output"] = "\n".join(lines[-5:]) if lines else None

        return analysis

    def _analyze_browser_content(
        self, app_name: str, window_title: str, screenshot: np.ndarray, ocr_text: str
    ) -> Dict[str, Any]:
        """Analyze browser content"""
        analysis = {
            "content_type": "browser",
            "current_url": None,
            "page_title": None,
            "tabs_count": 0,
            "is_search_page": False,
            "search_terms": None,
            "is_document": False,
            "is_video": False,
            "is_social_media": False,
            "is_email": False,
        }

        if not window_title and not ocr_text:
            return analysis

        # Extract from window title
        if window_title:
            # Common browser title patterns
            title_lower = window_title.lower()

            # Check for search pages
            if any(
                engine in title_lower
                for engine in [
                    " - google search",
                    "search results",
                    " - bing",
                    " - duckduckgo",
                ]
            ):
                analysis["is_search_page"] = True

            # Check for document types
            if any(
                doc_type in title_lower
                for doc_type in [
                    " - google docs",
                    " - notion",
                    " - github",
                    " - stack overflow",
                ]
            ):
                analysis["is_document"] = True

            # Check for video sites
            if any(
                video_site in title_lower
                for video_site in [" - youtube", " - netflix", " - twitch"]
            ):
                analysis["is_video"] = True

            # Check for social media
            if any(
                social in title_lower
                for social in [
                    " - twitter",
                    " - facebook",
                    " - instagram",
                    " - linkedin",
                ]
            ):
                analysis["is_social_media"] = True

            # Extract potential search terms (from search result titles)
            if " - " in window_title:
                parts = window_title.split(" - ")
                if len(parts) >= 2 and parts[1]:
                    analysis["page_title"] = parts[1].strip()

        # Extract from OCR text (address bar, content)
        if ocr_text:
            lines = ocr_text.strip().split("\n")

            # Look for URL patterns
            url_pattern = r"https?://[^\s]+"
            for line in lines:
                urls = re.findall(url_pattern, line)
                if urls:
                    analysis["current_url"] = urls[0]
                    break

            # Count potential tabs (look for tab indicators)
            tab_indicators = ["|", "•", "●", "▶", "◀"]
            tab_count = 0
            for line in lines:
                if any(indicator in line for indicator in tab_indicators):
                    tab_count += line.count("|") if "|" in line else 1

            if tab_count > 0:
                analysis["tabs_count"] = tab_count

        return analysis

    def _analyze_editor_content(
        self, app_name: str, window_title: str, screenshot: np.ndarray, ocr_text: str
    ) -> Dict[str, Any]:
        """Analyze code editor content"""
        analysis = {
            "content_type": "editor",
            "file_name": None,
            "file_type": None,
            "language": None,
            "is_modified": False,
            "cursor_position": None,
            "errors": [],
            "warnings": [],
            "todo_items": [],
            "line_count": 0,
        }

        if not window_title and not ocr_text:
            return analysis

        # Extract filename and type from window title
        if window_title:
            # Common patterns: "filename.ext - AppName" or "filename.ext (modified) - AppName"
            title_patterns = [
                r"(.+?)(?:\s*-\s*.+)?$",  # Basic filename extraction
                r"(.+?)\s*\([^)]*modified[^)]*\)",  # Modified indicator
            ]

            for pattern in title_patterns:
                match = re.search(pattern, window_title)
                if match:
                    filename = match.group(1).strip()
                    if "." in filename:
                        name_part, ext_part = filename.rsplit(".", 1)
                        analysis["file_name"] = filename
                        analysis["file_type"] = ext_part.lower()

                        # Determine language
                        lang_map = {
                            "py": "python",
                            "js": "javascript",
                            "ts": "typescript",
                            "html": "html",
                            "css": "css",
                            "scss": "scss",
                            "json": "json",
                            "md": "markdown",
                            "txt": "text",
                            "cpp": "c++",
                            "c": "c",
                            "h": "c",
                            "hpp": "c++",
                            "java": "java",
                            "kt": "kotlin",
                            "swift": "swift",
                            "php": "php",
                            "rb": "ruby",
                            "go": "go",
                            "rs": "rust",
                            "sh": "shell",
                            "sql": "sql",
                            "xml": "xml",
                            "yaml": "yaml",
                            "yml": "yaml",
                        }
                        analysis["language"] = lang_map.get(
                            ext_part.lower(), ext_part.lower()
                        )
                    break

        # Analyze content if available
        if ocr_text:
            lines = ocr_text.strip().split("\n")

            # Count lines
            analysis["line_count"] = len(lines)

            # Look for errors/warnings (common IDE indicators)
            for line in lines:
                line_lower = line.lower()
                if any(
                    error_word in line_lower
                    for error_word in ["error:", "error", "exception:", "traceback"]
                ):
                    if line not in analysis["errors"]:
                        analysis["errors"].append(line.strip())
                elif any(
                    warn_word in line_lower
                    for warn_word in ["warning:", "warn:", "deprecated"]
                ):
                    if line not in analysis["warnings"]:
                        analysis["warnings"].append(line.strip())
                elif any(
                    todo_word in line_lower
                    for todo_word in [
                        "todo:",
                        "todo",
                        "fixme:",
                        "fixme",
                        "hack:",
                        "hack",
                    ]
                ):
                    if line not in analysis["todo_items"]:
                        analysis["todo_items"].append(line.strip())

        return analysis

    def _analyze_communication_content(
        self, app_name: str, window_title: str, screenshot: np.ndarray, ocr_text: str
    ) -> Dict[str, Any]:
        """Analyze communication app content"""
        analysis = {
            "content_type": "communication",
            "conversation_count": 0,
            "active_conversation": None,
            "unread_messages": 0,
            "participants": [],
            "is_group_chat": False,
            "last_message": None,
        }

        if not window_title and not ocr_text:
            return analysis

        # Extract from window title
        if window_title:
            title_lower = window_title.lower()

            # Check for group indicators
            if any(
                indicator in title_lower
                for indicator in ["group", "channel", "room", "#"]
            ):
                analysis["is_group_chat"] = True

            # Extract potential participant names
            # Look for patterns like "Chat with John" or "John, Mary - Chat"
            name_patterns = [
                r"chat with (.+)",
                r"(.+?) - chat",
                r"(.+?) \(",
                r"(.+?),\s",
            ]

            for pattern in name_patterns:
                match = re.search(pattern, title_lower)
                if match:
                    names = match.group(1).strip()
                    if "," in names:
                        participants = [n.strip() for n in names.split(",")]
                    else:
                        participants = [names]
                    analysis["participants"] = participants
                    break

        return analysis

    def _analyze_productivity_content(
        self, app_name: str, window_title: str, screenshot: np.ndarray, ocr_text: str
    ) -> Dict[str, Any]:
        """Analyze productivity app content"""
        analysis = {
            "content_type": "productivity",
            "document_type": None,
            "is_calendar": False,
            "is_notes": False,
            "is_tasks": False,
            "event_count": 0,
            "note_count": 0,
            "task_count": 0,
        }

        if not app_name and not window_title:
            return analysis

        app_lower = app_name.lower()
        title_lower = window_title.lower()

        # Classify document type
        if "calendar" in app_lower or "calendar" in title_lower:
            analysis["is_calendar"] = True
            analysis["document_type"] = "calendar"
        elif "notes" in app_lower or "notes" in title_lower:
            analysis["is_notes"] = True
            analysis["document_type"] = "notes"
        elif "reminders" in app_lower or "tasks" in title_lower:
            analysis["is_tasks"] = True
            analysis["document_type"] = "tasks"
        else:
            analysis["document_type"] = "document"

        return analysis

    def _analyze_design_content(
        self, app_name: str, window_title: str, screenshot: np.ndarray, ocr_text: str
    ) -> Dict[str, Any]:
        """Analyze design application content"""
        analysis = {
            "content_type": "design",
            "project_name": None,
            "is_modified": False,
            "layer_count": 0,
            "tool_active": None,
        }

        if window_title:
            # Extract project name (usually in title)
            if " - " in window_title:
                project_name = window_title.split(" - ")[0]
                analysis["project_name"] = project_name

        return analysis

    def _analyze_music_content(
        self, app_name: str, window_title: str, screenshot: np.ndarray, ocr_text: str
    ) -> Dict[str, Any]:
        """Analyze music application content"""
        analysis = {
            "content_type": "music",
            "current_song": None,
            "artist": None,
            "album": None,
            "is_playing": False,
            "playlist": None,
        }

        if window_title:
            # Extract song info from title
            title_lower = window_title.lower()

            # Look for "Song - Artist" pattern
            if " - " in window_title:
                parts = window_title.split(" - ")
                if len(parts) >= 2:
                    analysis["current_song"] = parts[0].strip()
                    analysis["artist"] = parts[1].strip()

        return analysis

    def _analyze_video_content(
        self, app_name: str, window_title: str, screenshot: np.ndarray, ocr_text: str
    ) -> Dict[str, Any]:
        """Analyze video application content"""
        analysis = {
            "content_type": "video",
            "current_video": None,
            "is_playing": False,
            "timestamp": None,
            "quality": None,
        }

        if window_title:
            analysis["current_video"] = window_title

        return analysis

    def _analyze_system_content(
        self, app_name: str, window_title: str, screenshot: np.ndarray, ocr_text: str
    ) -> Dict[str, Any]:
        """Analyze system application content"""
        analysis = {
            "content_type": "system",
            "system_info": {},
            "processes": [],
            "memory_usage": None,
            "cpu_usage": None,
        }

        app_lower = app_name.lower()

        if "activity monitor" in app_lower:
            # Activity Monitor specific analysis
            if ocr_text:
                # Look for CPU/Memory percentages
                cpu_pattern = r"(\d+(?:\.\d+)?)%\s*(?:cpu|CPU)"
                memory_pattern = r"(\d+(?:\.\d+)?)%\s*(?:memory|Memory)"

                cpu_matches = re.findall(cpu_pattern, ocr_text)
                memory_matches = re.findall(memory_pattern, ocr_text)

                if cpu_matches:
                    analysis["cpu_usage"] = max([float(x) for x in cpu_matches])
                if memory_matches:
                    analysis["memory_usage"] = max([float(x) for x in memory_matches])

        return analysis

    def _analyze_generic_content(
        self, app_name: str, window_title: str, screenshot: np.ndarray, ocr_text: str
    ) -> Dict[str, Any]:
        """Analyze unknown/generic application content"""
        analysis = {
            "content_type": "unknown",
            "has_text": bool(ocr_text and ocr_text.strip()),
            "text_length": len(ocr_text) if ocr_text else 0,
            "has_images": False,  # Would need more advanced analysis
            "is_active": True,
        }

        if ocr_text:
            analysis["sample_text"] = (
                ocr_text[:200] + "..." if len(ocr_text) > 200 else ocr_text
            )

        return analysis


# Enhanced intelligent response generation
class AdaptiveResponseGenerator:
    """Generates intelligent responses that adapt to any workspace configuration"""

    def __init__(self):
        self.content_analyzer = DynamicContentAnalyzer()
        self.response_templates = self._init_response_templates()
        self.response_builder = MultiSpaceResponseBuilder()

    def _init_response_templates(self) -> Dict[str, Any]:
        """Initialize adaptive response templates"""
        return {
            "workspace_overview": {
                "template": "You have {space_count} desktop spaces active. Here's what's happening:\n\n{space_details}",
                "conditions": ["multiple_spaces", "overview_request"],
            },
            "app_location": {
                "template": "{app_name} is {location}. {additional_info}",
                "conditions": ["single_app", "location_request"],
            },
            "content_summary": {
                "template": "{app_type}: {content_summary}. {context_info}",
                "conditions": ["single_app", "content_request"],
            },
            "cross_app_correlation": {
                "template": "I notice {correlation_summary}. {explanation}",
                "conditions": ["multiple_apps", "relationship_insight"],
            },
            "activity_analysis": {
                "template": "You're currently {activity_summary}. {workspace_context}",
                "conditions": ["activity_analysis", "workspace_overview"],
            },
        }

    def generate_response(
        self,
        query: str,
        workspace_data: Dict[str, Any],
        intent: SpaceQueryIntent,
        user_context: Dict[str, Any] = None,
    ) -> str:
        """Generate intelligent response based on query intent and workspace data"""

        # Determine response type based on intent
        response_type = self._determine_response_type(intent, workspace_data)

        # Get relevant data for response
        response_data = self._prepare_response_data(
            response_type, workspace_data, intent
        )

        # Generate response using appropriate template
        response = self._apply_response_template(response_type, response_data)

        # Add contextual enhancements
        response = self._enhance_response_with_context(
            response, response_data, user_context
        )

        return response

    def _determine_response_type(
        self, intent: SpaceQueryIntent, workspace_data: Dict[str, Any]
    ) -> str:
        """Determine the best response type for the query"""

        query_type = intent.query_type
        spaces = workspace_data.get("spaces", [])

        # Handle both dict and SpaceInfo objects
        if spaces:
            space_count = len(spaces)
            app_count = sum(
                len(getattr(space, 'applications', {})) if hasattr(space, 'applications')
                else (len(space.get("applications", {})) if isinstance(space, dict) else 0)
                for space in spaces
            )
        else:
            space_count = 0
            app_count = 0

        # Determine primary response type
        if query_type == SpaceQueryType.ALL_SPACES:
            if space_count > 1:
                return "workspace_overview"
            else:
                return "single_space_overview"

        elif query_type == SpaceQueryType.SPECIFIC_DETAIL:
            if intent.target_app:
                return "app_content_analysis"
            else:
                return "generic_content_analysis"

        elif query_type == SpaceQueryType.LOCATION_QUERY:
            if intent.target_app:
                return "app_location"
            else:
                return "space_location"

        elif query_type == SpaceQueryType.WORKSPACE_OVERVIEW:
            return "activity_analysis"

        # Default to comprehensive overview
        return "workspace_overview"

    def _prepare_response_data(
        self,
        response_type: str,
        workspace_data: Dict[str, Any],
        intent: SpaceQueryIntent,
    ) -> Dict[str, Any]:
        """Prepare data needed for the response"""

        data = {
            "response_type": response_type,
            "workspace_summary": self._summarize_workspace(workspace_data),
            "app_analyses": {},
            "correlations": [],
            "intent": intent,
        }

        # Analyze content for each application if needed
        if response_type in [
            "app_content_analysis",
            "generic_content_analysis",
            "workspace_overview",
        ]:
            data["app_analyses"] = self._analyze_all_applications(workspace_data)

        # Find correlations if multiple apps
        if len(data["app_analyses"]) > 1:
            data["correlations"] = self._find_cross_app_correlations(
                data["app_analyses"]
            )

        return data

    def _summarize_workspace(self, workspace_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the entire workspace"""

        spaces = workspace_data.get("spaces", [])
        current_space = workspace_data.get("current_space", {})

        summary = {
            "total_spaces": len(spaces),
            "current_space_id": current_space.get("id", 1),
            "total_apps": sum(
                len(getattr(space, 'applications', {})) if hasattr(space, 'applications')
                else (len(space.get("applications", {})) if isinstance(space, dict) else 0)
                for space in spaces
            ),
            "space_details": [],
        }

        for space in spaces:
            # Handle both dict and SpaceInfo objects
            if hasattr(space, 'space_id'):
                # SpaceInfo object
                space_id = space.space_id
                applications = getattr(space, 'applications', {})
                is_current = space_id == summary["current_space_id"]
            elif isinstance(space, dict):
                # Dict format
                space_id = space.get("space_id", 1)
                applications = space.get("applications", {})
                is_current = space_id == summary["current_space_id"]
            else:
                # Skip unknown format
                continue

            # Get app names safely
            if hasattr(applications, 'keys'):
                app_names = list(applications.keys())
            else:
                app_names = []

            space_summary = {
                "space_id": space_id,
                "is_current": is_current,
                "app_count": len(applications),
                "primary_activity": self.response_builder._determine_space_activity(applications),
                "applications": app_names,
            }

            summary["space_details"].append(space_summary)

        return summary

    def _analyze_all_applications(
        self, workspace_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze content for all applications across all spaces"""

        analyses = {}

        for space in workspace_data.get("spaces", []):
            # Handle both dict and SpaceInfo objects
            if hasattr(space, 'applications'):
                # SpaceInfo object
                applications = space.applications
            elif isinstance(space, dict):
                # Dict format
                applications = space.get("applications", {})
            else:
                # Skip unknown format
                continue

            # Get app windows safely
            if hasattr(applications, 'items'):
                app_windows = applications.items()
            else:
                app_windows = []

            for app_name, windows in app_windows:
                for window_info in windows:
                    # Handle both dict and object formats for window_info
                    if isinstance(window_info, dict):
                        # Dict format
                        window_title = window_info.get("title", "")
                        screenshot = window_info.get("screenshot")
                        ocr_text = window_info.get("ocr_text")
                    elif hasattr(window_info, 'title'):
                        # Object format
                        window_title = getattr(window_info, 'title', '')
                        screenshot = getattr(window_info, 'screenshot', None)
                        ocr_text = getattr(window_info, 'ocr_text', None)
                    else:
                        # Unknown format
                        window_title = ""
                        screenshot = None
                        ocr_text = None

                    # Analyze content
                    analysis = self.content_analyzer.analyze_window_content(
                        app_name, window_title, screenshot, ocr_text
                    )

                    # Use space_id + app_name + window_id as key
                    key = f"space_{space.get('space_id', 1)}_{app_name}_{window_info.get('id', 0)}"
                    analyses[key] = analysis

        return analyses

    def _find_cross_app_correlations(
        self, app_analyses: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find interesting correlations between applications"""

        correlations = []

        # Group analyses by app type
        by_type = {}
        for key, analysis in app_analyses.items():
            app_type = analysis.get("content_type", "unknown")
            if app_type not in by_type:
                by_type[app_type] = []
            by_type[app_type].append(analysis)

        # Look for interesting patterns
        patterns = [
            self._check_development_correlation,
            self._check_research_correlation,
            self._check_communication_correlation,
            self._check_productivity_correlation,
        ]

        for pattern_func in patterns:
            correlation = pattern_func(by_type)
            if correlation:
                correlations.append(correlation)

        return correlations

    def _check_development_correlation(
        self, by_type: Dict[str, List[Dict]]
    ) -> Optional[Dict[str, Any]]:
        """Check for development-related correlations"""

        editors = by_type.get("editor", [])
        terminals = by_type.get("terminal", [])
        browsers = by_type.get("browser", [])

        if len(editors) > 0 and len(terminals) > 0:
            # Check if they're related (similar filenames, errors, etc.)
            editor_files = [
                e.get("file_name", "") for e in editors if e.get("file_name")
            ]
            terminal_commands = [t.get("commands", []) for t in terminals]

            # Simple correlation: if editor file and terminal command mention similar things
            for editor_file in editor_files:
                for commands in terminal_commands:
                    for command in commands:
                        if any(
                            word in command.lower()
                            for word in editor_file.lower().split(".")
                        ):
                            return {
                                "type": "development_workflow",
                                "description": f"You're working on {editor_file} with terminal commands",
                                "confidence": 0.8,
                                "related_apps": ["editor", "terminal"],
                            }

        return None

    def _check_research_correlation(
        self, by_type: Dict[str, List[Dict]]
    ) -> Optional[Dict[str, Any]]:
        """Check for research-related correlations"""

        browsers = by_type.get("browser", [])
        editors = by_type.get("editor", [])

        if len(browsers) > 0 and len(editors) > 0:
            # Look for research patterns (search pages + documentation)
            search_browsers = [b for b in browsers if b.get("is_search_page", False)]
            doc_browsers = [b for b in browsers if b.get("is_document", False)]

            if search_browsers and (editors or doc_browsers):
                return {
                    "type": "research_workflow",
                    "description": "You're researching while coding",
                    "confidence": 0.7,
                    "related_apps": ["browser", "editor"],
                }

        return None

    def _check_communication_correlation(
        self, by_type: Dict[str, List[Dict]]
    ) -> Optional[Dict[str, Any]]:
        """Check for communication-related correlations"""

        communication = by_type.get("communication", [])
        browsers = by_type.get("browser", [])
        editors = by_type.get("editor", [])

        if len(communication) > 0 and (len(browsers) > 0 or len(editors) > 0):
            return {
                "type": "collaboration_workflow",
                "description": "You're communicating while working",
                "confidence": 0.6,
                "related_apps": ["communication", "browser", "editor"],
            }

        return None

    def _check_productivity_correlation(
        self, by_type: Dict[str, List[Dict]]
    ) -> Optional[Dict[str, Any]]:
        """Check for productivity-related correlations"""

        productivity = by_type.get("productivity", [])
        browsers = by_type.get("browser", [])

        if len(productivity) > 0 and len(browsers) > 0:
            return {
                "type": "planning_workflow",
                "description": "You're organizing work while browsing",
                "confidence": 0.6,
                "related_apps": ["productivity", "browser"],
            }

        return None

    def _apply_response_template(
        self, response_type: str, response_data: Dict[str, Any]
    ) -> str:
        """Apply the appropriate response template"""

        templates = self.response_templates

        if response_type == "workspace_overview":
            return self._generate_workspace_overview(response_data)
        elif response_type == "app_location":
            return self._generate_app_location_response(response_data)
        elif response_type == "app_content_analysis":
            return self._generate_content_analysis_response(response_data)
        elif response_type == "activity_analysis":
            return self._generate_activity_analysis_response(response_data)
        elif response_type == "cross_app_correlation":
            return self._generate_correlation_response(response_data)
        else:
            return self._generate_generic_response(response_data)

    def _generate_workspace_overview(self, response_data: Dict[str, Any]) -> str:
        """Generate comprehensive workspace overview"""

        workspace = response_data["workspace_summary"]
        correlations = response_data.get("correlations", [])

        overview_parts = [
            f"You have {workspace['total_spaces']} desktop spaces active with {workspace['total_apps']} applications total."
        ]

        # Add space details
        for space_detail in workspace["space_details"]:
            space_id = space_detail["space_id"]
            is_current = space_detail["is_current"]
            activity = space_detail["primary_activity"]
            apps = space_detail["applications"]

            if is_current:
                overview_parts.append(
                    f"**Desktop {space_id} (current)**: {activity} - {', '.join(apps[:3])}"
                )
            else:
                overview_parts.append(
                    f"Desktop {space_id}: {activity} - {', '.join(apps[:2])}"
                )

        # Add correlations if found
        if correlations:
            correlation = correlations[0]  # Use the strongest correlation
            overview_parts.append(
                f"\n💡 **Pattern detected**: {correlation['description']}"
            )

        return "\n".join(overview_parts)

    def _generate_app_location_response(self, response_data: Dict[str, Any]) -> str:
        """Generate response for app location queries"""

        intent = response_data["intent"]
        app_name = intent.target_app or "the application"
        workspace = response_data["workspace_summary"]

        # Find where the app is located
        app_location = None
        for space_detail in workspace["space_details"]:
            if app_name.lower() in [
                app.lower() for app in space_detail["applications"]
            ]:
                space_id = space_detail["space_id"]
                is_current = space_detail["is_current"]
                app_location = {
                    "space_id": space_id,
                    "is_current": is_current,
                    "other_apps": [
                        app for app in space_detail["applications"] if app != app_name
                    ],
                }
                break

        if not app_location:
            return f"I don't see {app_name} in any of your active desktop spaces."

        # Generate location response
        if app_location["is_current"]:
            location_text = "on your current desktop"
        else:
            location_text = f"on Desktop {app_location['space_id']}"

        response_parts = [f"{app_name} is {location_text}"]

        if app_location["other_apps"]:
            response_parts.append(
                f"alongside {', '.join(app_location['other_apps'][:2])}"
            )

        return ". ".join(response_parts) + "."

    def _generate_content_analysis_response(self, response_data: Dict[str, Any]) -> str:
        """Generate response for content analysis queries"""

        app_analyses = response_data["app_analyses"]
        intent = response_data["intent"]

        # Find relevant app analysis
        target_app = intent.target_app
        relevant_analysis = None

        for key, analysis in app_analyses.items():
            if target_app and target_app.lower() in analysis["app_name"].lower():
                relevant_analysis = analysis
                break

        if not relevant_analysis:
            return f"I don't have detailed information about {target_app or 'that application'}."

        # Generate content-specific response
        return self._generate_app_specific_response(relevant_analysis)

    def _generate_app_specific_response(self, analysis: Dict[str, Any]) -> str:
        """Generate response specific to application type"""

        content_type = analysis["content_type"]

        if content_type == "terminal":
            return self._generate_terminal_response(analysis)
        elif content_type == "browser":
            return self._generate_browser_response(analysis)
        elif content_type == "editor":
            return self._generate_editor_response(analysis)
        elif content_type == "communication":
            return self._generate_communication_response(analysis)
        elif content_type == "productivity":
            return self._generate_productivity_response(analysis)
        elif content_type == "design":
            return self._generate_design_response(analysis)
        elif content_type == "music":
            return self._generate_music_response(analysis)
        elif content_type == "video":
            return self._generate_video_response(analysis)
        elif content_type == "system":
            return self._generate_system_response(analysis)
        else:
            return self._generate_generic_response(analysis)

    def _generate_terminal_response(self, analysis: Dict[str, Any]) -> str:
        """Generate terminal-specific response"""

        parts = []

        if analysis.get("current_directory"):
            parts.append(f"Working directory: `{analysis['current_directory']}`")

        if analysis.get("commands"):
            recent_commands = analysis["commands"][-3:]  # Last 3 commands
            parts.append(
                f"Recent commands: {', '.join(f'`{cmd}`' for cmd in recent_commands)}"
            )

        if analysis.get("errors"):
            errors = analysis["errors"][:2]  # First 2 errors
            parts.append(f"Errors detected: {', '.join(errors)}")

        if analysis.get("warnings"):
            warnings = analysis["warnings"][:2]
            parts.append(f"Warnings: {', '.join(warnings)}")

        if analysis.get("is_running_command"):
            parts.append("Currently running a command")

        if not parts:
            parts.append("Terminal session active")

        return ". ".join(parts) + "."

    def _generate_browser_response(self, analysis: Dict[str, Any]) -> str:
        """Generate browser-specific response"""

        parts = []

        if analysis.get("current_url"):
            url = analysis["current_url"]
            if len(url) > 60:
                url = url[:57] + "..."
            parts.append(f"Viewing: {url}")

        if analysis.get("page_title"):
            parts.append(f"Page: {analysis['page_title']}")

        if analysis.get("is_search_page"):
            parts.append("This appears to be a search results page")

        if analysis.get("is_document"):
            parts.append("This looks like a document or collaborative workspace")

        if analysis.get("is_video"):
            parts.append("This appears to be a video streaming site")

        if analysis.get("is_social_media"):
            parts.append("This looks like social media")

        if analysis.get("tabs_count", 0) > 1:
            parts.append(f"Multiple tabs open ({analysis['tabs_count']})")

        if not parts:
            parts.append("Web browser active")

        return ". ".join(parts) + "."

    def _generate_editor_response(self, analysis: Dict[str, Any]) -> str:
        """Generate editor-specific response"""

        parts = []

        if analysis.get("file_name"):
            parts.append(f"Editing: `{analysis['file_name']}`")

        if analysis.get("language"):
            parts.append(f"Language: {analysis['language']}")

        if analysis.get("line_count"):
            parts.append(f"{analysis['line_count']} lines")

        if analysis.get("errors"):
            error_count = len(analysis["errors"])
            parts.append(f"{error_count} error{'s' if error_count != 1 else ''}")

        if analysis.get("warnings"):
            warning_count = len(analysis["warnings"])
            parts.append(f"{warning_count} warning{'s' if warning_count != 1 else ''}")

        if analysis.get("todo_items"):
            todo_count = len(analysis["todo_items"])
            parts.append(f"{todo_count} TODO item{'s' if todo_count != 1 else ''}")

        if not parts:
            parts.append("Code editor active")

        return ". ".join(parts) + "."

    def _generate_communication_response(self, analysis: Dict[str, Any]) -> str:
        """Generate communication app response"""

        parts = []

        if analysis.get("participants"):
            participants = analysis["participants"]
            if len(participants) == 1:
                parts.append(f"Chatting with {participants[0]}")
            elif len(participants) <= 3:
                parts.append(f"Group chat with {', '.join(participants)}")
            else:
                parts.append(
                    f"Group chat with {participants[0]} and {len(participants)-1} others"
                )

        if analysis.get("is_group_chat"):
            parts.append("This is a group conversation")

        if not parts:
            parts.append("Communication app active")

        return ". ".join(parts) + "."

    def _generate_productivity_response(self, analysis: Dict[str, Any]) -> str:
        """Generate productivity app response"""

        parts = []

        if analysis.get("document_type"):
            parts.append(f"Using {analysis['document_type']} application")

        if analysis.get("is_calendar"):
            parts.append("Calendar application")
        elif analysis.get("is_notes"):
            parts.append("Note-taking application")
        elif analysis.get("is_tasks"):
            parts.append("Task management application")

        if not parts:
            parts.append("Productivity application active")

        return ". ".join(parts) + "."

    def _generate_design_response(self, analysis: Dict[str, Any]) -> str:
        """Generate design app response"""

        parts = []

        if analysis.get("project_name"):
            parts.append(f"Working on: {analysis['project_name']}")

        parts.append("Design application active")
        return ". ".join(parts) + "."

    def _generate_music_response(self, analysis: Dict[str, Any]) -> str:
        """Generate music app response"""

        parts = []

        if analysis.get("current_song") and analysis.get("artist"):
            parts.append(f"Playing: {analysis['current_song']} by {analysis['artist']}")
        elif analysis.get("current_song"):
            parts.append(f"Playing: {analysis['current_song']}")

        if not parts:
            parts.append("Music application active")

        return ". ".join(parts) + "."

    def _generate_video_response(self, analysis: Dict[str, Any]) -> str:
        """Generate video app response"""

        parts = []

        if analysis.get("current_video"):
            parts.append(f"Watching: {analysis['current_video']}")

        parts.append("Video application active")
        return ". ".join(parts) + "."

    def _generate_system_response(self, analysis: Dict[str, Any]) -> str:
        """Generate system app response"""

        parts = []

        if analysis.get("cpu_usage"):
            parts.append(f"CPU usage: {analysis['cpu_usage']}%")

        if analysis.get("memory_usage"):
            parts.append(f"Memory usage: {analysis['memory_usage']}%")

        if not parts:
            parts.append("System monitoring application active")

        return ". ".join(parts) + "."

    def _generate_generic_response(self, analysis: Dict[str, Any]) -> str:
        """Generate generic response for unknown apps"""

        parts = []

        if analysis.get("has_text"):
            parts.append("Text content detected")
            if analysis.get("text_length", 0) > 100:
                parts.append("Substantial amount of text")

        if analysis.get("sample_text"):
            # Show a snippet
            sample = (
                analysis["sample_text"][:100] + "..."
                if len(analysis["sample_text"]) > 100
                else analysis["sample_text"]
            )
            parts.append(f'Content: "{sample}"')

        if not parts:
            parts.append("Application window active")

        return ". ".join(parts) + "."

    def _generate_activity_analysis_response(
        self, response_data: Dict[str, Any]
    ) -> str:
        """Generate activity analysis response"""

        workspace = response_data["workspace_summary"]
        correlations = response_data.get("correlations", [])

        # Determine overall activity
        all_activities = [
            space["primary_activity"] for space in workspace["space_details"]
        ]
        unique_activities = list(set(all_activities))

        if len(unique_activities) == 1:
            activity_summary = unique_activities[0]
        elif len(unique_activities) <= 3:
            activity_summary = "doing a mix of " + " and ".join(unique_activities)
        else:
            activity_summary = "working on multiple different tasks"

        response_parts = [
            f"You're currently {activity_summary} across your {workspace['total_spaces']} desktop spaces"
        ]

        # Add correlation insights
        if correlations:
            correlation = correlations[0]
            response_parts.append(f"💡 {correlation['description']}")

        return ". ".join(response_parts) + "."

    def _generate_correlation_response(self, response_data: Dict[str, Any]) -> str:
        """Generate correlation-focused response"""

        correlations = response_data.get("correlations", [])

        if not correlations:
            return "I don't see any particular patterns in your workspace activity."

        correlation = correlations[0]  # Use the strongest correlation
        return f"💡 **Pattern detected**: {correlation['description']}."

    def _enhance_response_with_context(
        self,
        response: str,
        response_data: Dict[str, Any],
        user_context: Dict[str, Any] = None,
    ) -> str:
        """Add contextual enhancements to the response"""

        enhancements = []

        # Add confidence indicators
        intent = response_data["intent"]
        if intent.confidence < 0.6:
            enhancements.append("(Based on limited information)")

        # Add follow-up suggestions for complex responses
        if len(response.split()) > 50:  # Long response
            enhancements.append(
                "Would you like me to explain any specific part in more detail?"
            )

        # Add the enhancements
        if enhancements:
            response += "\n\n" + " ".join(enhancements)

        return response


# Dynamic adaptation and learning system
class WorkspaceAdaptationEngine:
    """Makes the system dynamically adapt to any workspace configuration"""

    def __init__(self):
        self.adaptation_rules = self._init_adaptation_rules()
        self.user_preferences = {}
        self.workspace_patterns = {}
        self.performance_metrics = {}

    def _init_adaptation_rules(self) -> Dict[str, Any]:
        """Initialize dynamic adaptation rules"""
        return {
            "response_style": {
                "terse": {"max_length": 100, "detail_level": "low"},
                "balanced": {"max_length": 200, "detail_level": "medium"},
                "detailed": {"max_length": 400, "detail_level": "high"},
            },
            "content_priority": {
                "terminal": 0.9,  # High priority for terminal errors
                "browser": 0.7,  # Medium priority for research
                "editor": 0.8,  # High priority for code work
                "communication": 0.5,  # Lower priority for chat
                "productivity": 0.6,  # Medium for notes/calendar
                "design": 0.4,  # Lower for design work
                "music": 0.2,  # Low for entertainment
                "video": 0.2,  # Low for entertainment
                "system": 0.3,  # Low for system monitoring
                "unknown": 0.1,  # Very low for unknown
            },
            "correlation_thresholds": {
                "development": 0.7,
                "research": 0.6,
                "communication": 0.5,
                "productivity": 0.5,
            },
        }

    def adapt_to_workspace(
        self,
        workspace_data: Dict[str, Any],
        user_query: str,
        user_preferences: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Adapt system behavior to current workspace configuration"""

        adaptation = {
            "response_style": "balanced",
            "content_priorities": {},
            "correlation_sensitivity": 0.6,
            "detail_level": "medium",
            "focus_areas": [],
        }

        # Analyze workspace composition
        spaces = workspace_data.get("spaces", [])
        app_types = self._analyze_workspace_composition(spaces)

        # Determine primary work mode
        primary_mode = self._determine_primary_work_mode(app_types)

        # Adapt response style based on work mode
        if primary_mode in ["development", "research"]:
            adaptation["response_style"] = "detailed"
            adaptation["detail_level"] = "high"
        elif primary_mode == "communication":
            adaptation["response_style"] = "terse"
            adaptation["detail_level"] = "low"
        else:
            adaptation["response_style"] = "balanced"
            adaptation["detail_level"] = "medium"

        # Set content priorities based on what's open
        for app_type, count in app_types.items():
            if count > 0:
                base_priority = self.adaptation_rules["content_priority"].get(
                    app_type, 0.5
                )
                # Boost priority if multiple windows of same type
                priority = min(1.0, base_priority + (count - 1) * 0.1)
                adaptation["content_priorities"][app_type] = priority

        # Determine correlation sensitivity
        if primary_mode in ["development", "research"]:
            adaptation["correlation_sensitivity"] = 0.8  # High sensitivity for work
        elif primary_mode == "communication":
            adaptation["correlation_sensitivity"] = 0.4  # Low sensitivity for social
        else:
            adaptation["correlation_sensitivity"] = 0.6  # Medium

        # Identify focus areas based on user's query and workspace
        focus_areas = self._identify_focus_areas(user_query, app_types, primary_mode)
        adaptation["focus_areas"] = focus_areas

        # Apply user preferences if available
        if user_preferences:
            adaptation = self._apply_user_preferences(adaptation, user_preferences)

        return adaptation

    def _analyze_workspace_composition(
        self, spaces: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Analyze what types of applications are in the workspace"""

        app_types = {}

        for space in spaces:
            # Handle both dict and SpaceInfo objects
            if hasattr(space, 'applications'):
                # SpaceInfo object
                applications = space.applications
            elif isinstance(space, dict):
                # Dict format
                applications = space.get("applications", {})
            else:
                # Unknown format
                applications = {}

            # Get app names safely
            if hasattr(applications, 'keys'):
                app_names = list(applications.keys())
            else:
                app_names = []

            for app_name in app_names:
                # Use content analyzer to classify app type
                analyzer = DynamicContentAnalyzer()
                app_type = analyzer._classify_app_type(app_name, "")

                if app_type not in app_types:
                    app_types[app_type] = 0
                app_types[app_type] += 1

        return app_types

    def _determine_primary_work_mode(self, app_types: Dict[str, int]) -> str:
        """Determine the primary mode of work based on app composition"""

        # Score each work mode
        work_modes = {
            "development": 0,
            "research": 0,
            "communication": 0,
            "productivity": 0,
            "design": 0,
            "entertainment": 0,
            "system": 0,
        }

        # Weight different app types for each work mode
        mode_weights = {
            "development": {"terminal": 3, "editor": 3, "browser": 1, "system": 1},
            "research": {"browser": 3, "editor": 2, "productivity": 1},
            "communication": {"communication": 3, "browser": 1},
            "productivity": {"productivity": 3, "browser": 1, "editor": 1},
            "design": {"design": 3, "browser": 1},
            "entertainment": {"music": 2, "video": 2, "browser": 1},
            "system": {"system": 3},
        }

        # Calculate scores
        for mode, weights in mode_weights.items():
            for app_type, count in app_types.items():
                if app_type in weights:
                    work_modes[mode] += weights[app_type] * count

        # Return mode with highest score
        return max(work_modes, key=work_modes.get)

    def _identify_focus_areas(
        self, user_query: str, app_types: Dict[str, int], primary_mode: str
    ) -> List[str]:
        """Identify what areas the user is likely focusing on"""

        focus_areas = []
        query_lower = user_query.lower()

        # Query-based focus areas
        if any(
            word in query_lower for word in ["terminal", "command", "error", "debug"]
        ):
            focus_areas.append("terminal_analysis")

        if any(word in query_lower for word in ["code", "file", "edit", "programming"]):
            focus_areas.append("code_analysis")

        if any(word in query_lower for word in ["browse", "research", "search", "web"]):
            focus_areas.append("web_research")

        if any(
            word in query_lower for word in ["chat", "message", "email", "communicate"]
        ):
            focus_areas.append("communication")

        # Workspace-based focus areas
        if app_types.get("terminal", 0) > 0:
            focus_areas.append("terminal_activity")

        if app_types.get("editor", 0) > 0:
            focus_areas.append("development_work")

        if app_types.get("browser", 0) > 1:
            focus_areas.append("research_activity")

        # Primary mode focus
        if primary_mode == "development":
            focus_areas.append("coding_focus")
        elif primary_mode == "research":
            focus_areas.append("research_focus")
        elif primary_mode == "communication":
            focus_areas.append("collaboration_focus")

        return list(set(focus_areas))  # Remove duplicates

    def _apply_user_preferences(
        self, adaptation: Dict[str, Any], user_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply learned user preferences to adaptation"""

        # Response style preference
        if "response_style" in user_preferences:
            adaptation["response_style"] = user_preferences["response_style"]

        # Detail level preference
        if "detail_level" in user_preferences:
            adaptation["detail_level"] = user_preferences["detail_level"]

        # Focus areas preference
        if "focus_areas" in user_preferences:
            adaptation["focus_areas"] = user_preferences["focus_areas"]

        return adaptation

    def learn_from_interaction(
        self,
        query: str,
        response: str,
        user_feedback: str,
        workspace_context: Dict[str, Any],
    ):
        """Learn from user interactions to improve future responses"""

        # Extract learning signals
        learning_data = {
            "query_length": len(query.split()),
            "response_length": len(response.split()),
            "workspace_complexity": len(workspace_context.get("spaces", [])),
            "user_satisfaction": self._interpret_feedback(user_feedback),
            "timestamp": datetime.now(),
        }

        # Update preferences based on feedback
        if user_feedback:
            self._update_preferences_from_feedback(learning_data, user_feedback)

        # Store for pattern analysis
        self._store_interaction_pattern(learning_data, workspace_context)

    def _interpret_feedback(self, feedback: str) -> float:
        """Interpret user feedback as satisfaction score"""

        feedback_lower = feedback.lower()

        positive_keywords = [
            "good",
            "great",
            "excellent",
            "perfect",
            "helpful",
            "thanks",
            "useful",
        ]
        negative_keywords = [
            "bad",
            "wrong",
            "incorrect",
            "not helpful",
            "confusing",
            "too long",
            "too short",
        ]

        positive_count = sum(1 for word in positive_keywords if word in feedback_lower)
        negative_count = sum(1 for word in negative_keywords if word in feedback_lower)

        if positive_count > negative_count:
            return 0.8 + (positive_count * 0.1)  # 0.8 to 1.0
        elif negative_count > positive_count:
            return 0.2 - (negative_count * 0.1)  # 0.0 to 0.2
        else:
            return 0.5  # Neutral

    def _update_preferences_from_feedback(
        self, learning_data: Dict[str, Any], feedback: str
    ):
        """Update user preferences based on feedback"""

        feedback_lower = feedback.lower()

        # Response style preferences
        if any(word in feedback_lower for word in ["too long", "verbose", "detailed"]):
            self.user_preferences["response_style"] = "terse"
        elif any(
            word in feedback_lower for word in ["too short", "brief", "need more"]
        ):
            self.user_preferences["response_style"] = "detailed"

        # Detail level preferences
        if "too much detail" in feedback_lower:
            self.user_preferences["detail_level"] = "low"
        elif "not enough detail" in feedback_lower:
            self.user_preferences["detail_level"] = "high"

    def _store_interaction_pattern(
        self, learning_data: Dict[str, Any], workspace_context: Dict[str, Any]
    ):
        """Store interaction pattern for future analysis"""

        pattern_key = f"query_len_{learning_data['query_length']}_spaces_{learning_data['workspace_complexity']}"

        if pattern_key not in self.workspace_patterns:
            self.workspace_patterns[pattern_key] = []

        self.workspace_patterns[pattern_key].append(
            {
                "satisfaction": learning_data["user_satisfaction"],
                "response_length": learning_data["response_length"],
                "timestamp": learning_data["timestamp"],
            }
        )

        # Keep only recent patterns (last 100)
        if len(self.workspace_patterns[pattern_key]) > 100:
            self.workspace_patterns[pattern_key] = self.workspace_patterns[pattern_key][
                -100:
            ]

    def get_adaptation_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for adapting to current usage patterns"""

        recommendations = {
            "response_style_changes": [],
            "content_priority_adjustments": [],
            "correlation_improvements": [],
        }

        # Analyze response length patterns
        for pattern_key, interactions in self.workspace_patterns.items():
            if len(interactions) >= 10:  # Need sufficient data
                recent_interactions = interactions[-10:]

                avg_satisfaction = sum(
                    i["satisfaction"] for i in recent_interactions
                ) / len(recent_interactions)
                avg_response_length = sum(
                    i["response_length"] for i in recent_interactions
                ) / len(recent_interactions)

                # Recommend style changes based on satisfaction
                if avg_satisfaction < 0.6 and avg_response_length > 150:
                    recommendations["response_style_changes"].append(
                        "Consider using more concise responses for better user satisfaction"
                    )
                elif avg_satisfaction > 0.8 and avg_response_length < 100:
                    recommendations["response_style_changes"].append(
                        "Users seem to prefer more detailed responses"
                    )

        return recommendations


# Integration class for pure_vision_intelligence.py
class MultiSpaceIntelligenceExtension:
    """Extension to add multi-space awareness to PureVisionIntelligence"""
    
    def __init__(self):
        self.query_detector = MultiSpaceQueryDetector()
        self.prompt_enhancer = SpaceAwarePromptEnhancer()
        self.content_analyzer = DynamicContentAnalyzer()
        self.response_generator = AdaptiveResponseGenerator()
        self.adaptation_engine = WorkspaceAdaptationEngine()
        self.response_builder = MultiSpaceResponseBuilder()

    def analyze_comprehensive_workspace(
        self,
        query: str,
        workspace_data: Dict[str, Any],
        user_context: Dict[str, Any] = None,
    ) -> str:
        """Main entry point for comprehensive multi-space analysis"""

        # Step 1: Detect query intent
        intent = self.query_detector.detect_intent(query)

        # Step 2: Adapt to workspace configuration
        adaptation = self.adaptation_engine.adapt_to_workspace(
            workspace_data, query, user_context
        )

        # Step 3: Generate intelligent response
        response = self.response_generator.generate_response(
            query, workspace_data, intent, user_context
        )

        # Step 4: Apply adaptation rules
        response = self._apply_adaptation_rules(response, adaptation)

        return response

    def generate_enhanced_workspace_response(
        self,
        query: str,
        workspace_data: Dict[str, Any],
        screenshots: Dict[int, Any] = None,
    ) -> str:
        """Generate enhanced response with screenshots if available

        NOTE: This method now returns None to signal that Claude API should be used
        when screenshots are available, allowing for intelligent vision analysis
        instead of template-based responses.
        """

        # If we have screenshots, return None to signal Claude API should be used
        if screenshots:
            # Return None to indicate that the caller should use Claude API
            # with the screenshots for intelligent analysis
            return None
        else:
            # No screenshots, use workspace data only for basic response
            return self.analyze_comprehensive_workspace(query, workspace_data)

    def _apply_adaptation_rules(self, response: str, adaptation: Dict[str, Any]) -> str:
        """Apply dynamic adaptation rules to the response"""

        # Apply response style adaptation
        style = adaptation.get("response_style", "balanced")
        style_rules = self.adaptation_engine.adaptation_rules["response_style"][style]

        # Truncate if too long
        max_length = style_rules["max_length"]
        if len(response) > max_length:
            # Try to truncate at a sentence boundary
            sentences = response.split(". ")
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence + ". ") <= max_length:
                    truncated += sentence + ". "
                else:
                    break
            response = truncated.rstrip(". ") + "..."

        return response

    def test_enhanced_system(self):
        """Test the enhanced multi-space vision system with various configurations"""

        print("🚀 Testing Enhanced Multi-Space Vision Intelligence System")
        print("=" * 70)

        # Test 1: Development workspace
        print("\n📋 Test 1: Development Workspace")
        print("-" * 40)

        dev_workspace = {
            "current_space": {"id": 1, "window_count": 3},
            "spaces": [
                {
                    "space_id": 1,
                    "applications": {
                        "Terminal": [{"title": "zsh", "id": 1}],
                        "Visual Studio Code": [{"title": "main.py", "id": 2}],
                        "Chrome": [{"title": "Stack Overflow", "id": 3}],
                    },
                },
                {
                    "space_id": 2,
                    "applications": {
                        "Terminal": [{"title": "npm test", "id": 4}],
                        "Chrome": [{"title": "GitHub Issues", "id": 5}],
                    },
                },
            ],
        }

        queries = [
            "What's happening in my workspace?",
            "Can you see my terminal?",
            "What am I working on?",
            "Tell me about the code I'm editing",
            "What's the error in my terminal?",
        ]

        for query in queries:
            print(f"\nQuery: '{query}'")
            intent = self.query_detector.detect_intent(query)
            adaptation = self.adaptation_engine.adapt_to_workspace(dev_workspace, query)

            print(
                f"Intent: {intent.query_type.value} (confidence: {intent.confidence})"
            )
            print(
                f"Adaptation: {adaptation['response_style']} style, focus: {adaptation['focus_areas']}"
            )

            response = self.analyze_comprehensive_workspace(query, dev_workspace)
            print(f"Response: {response[:100]}{'...' if len(response) > 100 else ''}")

        # Test 2: Research workspace
        print("\n\n📚 Test 2: Research Workspace")
        print("-" * 40)

        research_workspace = {
            "current_space": {"id": 1, "window_count": 2},
            "spaces": [
                {
                    "space_id": 1,
                    "applications": {
                        "Chrome": [
                            {"title": "Python Documentation", "id": 1},
                            {"title": "Stack Overflow", "id": 2},
                        ],
                        "Safari": [{"title": "Research Paper", "id": 3}],
                    },
                },
                {
                    "space_id": 2,
                    "applications": {
                        "Notes": [{"title": "Research Notes", "id": 4}],
                        "Chrome": [{"title": "Academic Database", "id": 5}],
                    },
                },
            ],
        }

        research_queries = [
            "What research am I doing?",
            "What's in my browser?",
            "Tell me about the document I'm reading",
            "What notes do I have?",
        ]

        for query in research_queries:
            print(f"\nQuery: '{query}'")
            adaptation = self.adaptation_engine.adapt_to_workspace(
                research_workspace, query
            )
            print(
                f"Adaptation: {adaptation['response_style']} style, focus: {adaptation['focus_areas']}"
            )

            response = self.analyze_comprehensive_workspace(query, research_workspace)
            print(f"Response: {response[:100]}{'...' if len(response) > 100 else ''}")

        # Test 3: Mixed workspace
        print("\n\n🎯 Test 3: Mixed Workspace")
        print("-" * 40)

        mixed_workspace = {
            "current_space": {"id": 1, "window_count": 4},
            "spaces": [
                {
                    "space_id": 1,
                    "applications": {
                        "Slack": [{"title": "Team Chat", "id": 1}],
                        "Chrome": [{"title": "Email", "id": 2}],
                        "Calendar": [{"title": "Today", "id": 3}],
                    },
                },
                {
                    "space_id": 2,
                    "applications": {
                        "Terminal": [{"title": "git status", "id": 4}],
                        "Visual Studio Code": [{"title": "config.py", "id": 5}],
                    },
                },
                {
                    "space_id": 3,
                    "applications": {"Spotify": [{"title": "Now Playing", "id": 6}]},
                },
            ],
        }

        mixed_queries = [
            "What's going on in my workspace?",
            "Am I working or taking a break?",
            "What should I focus on next?",
        ]

        for query in mixed_queries:
            print(f"\nQuery: '{query}'")
            adaptation = self.adaptation_engine.adapt_to_workspace(
                mixed_workspace, query
            )
            print(
                f"Adaptation: {adaptation['response_style']} style, focus: {adaptation['focus_areas']}"
            )

            response = self.analyze_comprehensive_workspace(query, mixed_workspace)
            print(f"Response: {response[:100]}{'...' if len(response) > 100 else ''}")

        print("\n✅ Enhanced Multi-Space Vision Intelligence Test Complete!")
        print("\nKey Features Demonstrated:")
        print("• Dynamic app detection (no hardcoding)")
        print("• Comprehensive content analysis (any app type)")
        print("• Intelligent response generation (adapts to workspace)")
        print("• Cross-application correlation (finds patterns)")
        print("• Dynamic adaptation (learns and adapts)")
        
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
            SpaceQueryType.WORKSPACE_OVERVIEW: 0.9,
        }
        if intent.query_type in multi_space_intents:
            factors.append(("intent_type", multi_space_intents[intent.query_type]))
            
        # Factor 2: Has app target (looking for specific app)
        if intent.target_app:
            factors.append(("has_app_target", 0.7))
            
        # Factor 3: Has space target or references
        if intent.target_space or intent.space_references:
            factors.append(("has_space_ref", 0.9))
            
        # Factor 4: Dynamic keyword analysis
        keyword_score = self._calculate_keyword_score(query_lower)
        if keyword_score > 0:
            factors.append(("keywords", keyword_score))
            
        # Factor 5: Pattern complexity
        pattern_score = self._analyze_pattern_complexity(query_lower)
        if pattern_score > 0:
            factors.append(("patterns", pattern_score))
            
        # Factor 6: Context hints suggest multi-space
        if any("spatial" in hint for hint in intent.context_hints):
            factors.append(("context_hints", 0.6))
            
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
            "location": (self.query_detector._location_verbs, 0.3),
            "spatial": (self.query_detector._space_terms, 0.2),
            "other": (self.query_detector._other_terms, 0.4),
            "all": (self.query_detector._all_terms, 0.5),
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
            (r"\b(?:across|between|among)\s+(?:\w+\s+)?(?:spaces?|desktops?)", 0.8),
            # Comparative queries  
            (r"\b(?:compare|difference|both|either)\s+", 0.6),
            # Navigation queries
            (r"\b(?:switch|move|go|jump)\s+(?:to|between)\s+", 0.7),
            # Listing queries
            (r"\b(?:list|show|display)\s+(?:all|every)\s+", 0.7),
        ]
        
        for pattern, weight in complex_patterns:
            if re.search(pattern, query_lower):
                score += weight
                
        return min(score, 1.0)
        
    def _check_simple_presence_override(
        self, query_lower: str, intent: SpaceQueryIntent
    ) -> bool:
        """Check if simple presence query should use multi-space"""
        # Even simple "Is X open?" might need multi-space if:
        
        # 1. Query implies checking everywhere
        if any(term in query_lower for term in ["anywhere", "somewhere", "everywhere"]):
            return True
            
        # 2. Query has uncertainty markers
        if any(term in query_lower for term in ["might", "could", "possibly", "maybe"]):
            return True
            
        # 3. Has app target (user asking about specific app)
        if intent.target_app:
            return True
            
        return False
        
    def process_multi_space_query(
        self, query: str, window_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a multi-space query with available data"""
        
        # Detect intent
        intent = self.query_detector.detect_intent(query)
        
        # Determine what data we need vs what we have
        data_requirements = self._analyze_data_requirements(intent, window_data)
        
        # Build response based on available data
        response_data = {
            "intent": intent,
            "data_requirements": data_requirements,
            "can_answer": data_requirements["can_answer_with_current_data"],
            "confidence": data_requirements["confidence_level"],
            "suggested_response": None,
        }
        
        # Generate suggested response if we can answer
        if response_data["can_answer"]:
            response_data["suggested_response"] = self._generate_response(
                intent, window_data, data_requirements
            )
            
        return response_data
        
    def _analyze_data_requirements(
        self, intent: SpaceQueryIntent, available_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze what data is needed vs available"""
        
        requirements = {
            "needs_screenshot": intent.requires_screenshot,
            "needs_metadata": True,  # Always useful
            "can_answer_with_current_data": False,
            "confidence_level": "low",
            "missing_data": [],
        }
        
        # Check what we have
        has_metadata = "windows" in available_data
        has_current_screenshot = available_data.get("has_current_screenshot", False)
        has_cached_screenshots = any(
            (
                (
                    hasattr(space, "cached_screenshot")
                    and space.cached_screenshot is not None
                )
                if hasattr(space, "cached_screenshot")
                else (
                    isinstance(space, dict)
                    and (
                        hasattr(space, "cached_screenshot") and space.cached_screenshot
                    )
                    if hasattr(space, "cached_screenshot")
                    else (isinstance(space, dict) and space.get("cached_screenshot"))
                    is not None
                )
            )
            for space in available_data.get("spaces", [])
        )
        
        # Determine if we can answer
        if intent.metadata_sufficient and has_metadata:
            requirements["can_answer_with_current_data"] = True
            requirements["confidence_level"] = "high"
        elif intent.requires_screenshot:
            if intent.target_space:
                # Check if we have screenshot for target space
                target_space_data = next(
                    (
                        s
                        for s in available_data.get("spaces", [])
                        if (
                            hasattr(s, "space_id") and s.space_id == intent.target_space
                        )
                        or (
                            isinstance(s, dict)
                            and s.get("space_id") == intent.target_space
                        )
                    ),
                    None,
                )
                if target_space_data:
                    has_screenshot = (
                        (
                            hasattr(target_space_data, "cached_screenshot")
                            and target_space_data.cached_screenshot
                        )
                        if hasattr(target_space_data, "cached_screenshot")
                        else (
                            isinstance(target_space_data, dict)
                            and target_space_data.get("cached_screenshot")
                        )
                    )
                    if has_screenshot:
                        requirements["can_answer_with_current_data"] = True
                        requirements["confidence_level"] = "medium"
                    else:
                        requirements["missing_data"].append("screenshot_for_space")
                else:
                    requirements["missing_data"].append("screenshot_for_space")
            elif has_current_screenshot:
                requirements["can_answer_with_current_data"] = True
                requirements["confidence_level"] = "high"
                
        return requirements
        
    def _generate_response(
        self,
                         intent: SpaceQueryIntent,
                         window_data: Dict[str, Any],
        requirements: Dict[str, Any],
    ) -> str:
        """Generate appropriate response based on intent and data"""
        
        if intent.query_type == SpaceQueryType.LOCATION_QUERY:
            # Find the app
            windows = window_data.get("windows", [])
            if intent.target_app:
                target_windows = [
                    w
                    for w in windows
                    if intent.target_app.lower()
                    in (
                        w.app_name.lower()
                        if hasattr(w, "app_name")
                        else w.get("app_name", "").lower()
                    )
                ]
                
                if target_windows:
                    return self.response_builder.build_location_response(
                        intent.target_app,
                        target_windows[0],
                        requirements["confidence_level"],
                    )
                else:
                    return f"I don't see {intent.target_app} open on any desktop."
            else:
                # Query about "other desktop" without specific app
                current_space_id = window_data.get("current_space", {}).get("id", 1)
                other_spaces = [
                    space
                    for space in window_data.get("spaces", [])
                    if (
                        hasattr(space, "space_id")
                        and space.space_id != current_space_id
                    )
                    or (
                        isinstance(space, dict)
                        and space.get("space_id") != current_space_id
                    )
                ]
                
                if other_spaces:
                    # Build response about other spaces
                    response_parts = []
                    for space in other_spaces:
                        space_id = (
                            space.space_id
                            if hasattr(space, "space_id")
                            else space.get("space_id")
                        )
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
            for space in window_data.get("spaces", []):
                if hasattr(space, "space_id"):
                    # It's a SpaceInfo object
                    space_id = space.space_id
                    is_current = space.is_current
                else:
                    # It's a dictionary
                    space_id = space["space_id"]
                    is_current = space["is_current"]
                    
                summary = {
                    "space_id": space_id,
                    "is_current": is_current,
                    "applications": self._get_space_applications(space_id, window_data),
                }
                spaces_summary.append(summary)
                
            return self.response_builder.build_workspace_overview(spaces_summary)
            
        # Add more response types as needed
        return "I can help you with that query about your workspaces."
        
    def _get_space_applications(
        self, space_id: int, window_data: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Get applications on a specific space"""
        space_windows = window_data.get("space_window_map", {}).get(space_id, [])
        windows = window_data.get("windows", [])
        
        apps = {}
        for window_id in space_windows:
            window = next(
                (
                    w
                    for w in windows
                    if (
                        (hasattr(w, "window_id") and w.window_id == window_id)
                        or (isinstance(w, dict) and w.get("window_id") == window_id)
                    )
                ),
                None,
            )
            if window:
                app_name = (
                    window.app_name
                    if hasattr(window, "app_name")
                    else window.get("app_name", "Unknown")
                )
                if app_name not in apps:
                    apps[app_name] = []
                window_title = (
                    window.window_title
                    if hasattr(window, "window_title")
                    else window.get("window_title", "")
                )
                apps[app_name].append(window_title)
                
        return apps
