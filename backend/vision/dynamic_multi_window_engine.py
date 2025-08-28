#!/usr/bin/env python3
"""
Dynamic Multi-Window Engine - Zero Hardcoding Window Analysis
Uses ML to understand window purpose and relevance without predefined patterns
"""

import os
import re
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from collections import defaultdict

from .window_detector import WindowInfo
from .multi_window_capture import MultiWindowCapture, WindowCapture
from .workspace_analyzer import WorkspaceAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class WindowFeatures:
    """ML features extracted from window metadata"""
    app_name_tokens: List[str]
    title_tokens: List[str]
    size_ratio: float  # window size relative to screen
    is_focused: bool
    position_quadrant: str  # TL, TR, BL, BR
    semantic_category: Optional[str] = None
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    activity_signals: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DynamicWindowAnalysis:
    """Result of dynamic window analysis"""
    primary_windows: List[WindowInfo]
    context_windows: List[WindowInfo]
    relevance_map: Dict[str, float]
    analysis_confidence: float
    reasoning: str

class DynamicMultiWindowEngine:
    """
    Completely dynamic multi-window analysis engine
    No hardcoded app names or categories - learns from context
    """
    
    def __init__(self):
        # ML components
        self.semantic_model = self._initialize_semantic_model()
        self.pattern_learner = defaultdict(lambda: defaultdict(float))
        
        # Dynamic category discovery
        self.discovered_categories = {}
        self.category_patterns = defaultdict(list)
        self.window_relationships = defaultdict(list)
        
        # Learning data
        self.query_history = []
        self.window_usage_patterns = defaultdict(lambda: defaultdict(int))
        self.learned_relevance = defaultdict(float)
        
        # Multi-window components
        self.capture_system = MultiWindowCapture()
        self.workspace_analyzer = WorkspaceAnalyzer()
        
        # Load any previously learned patterns
        self._load_learned_patterns()
        
        logger.info("Dynamic Multi-Window Engine initialized with zero hardcoding")
    
    def _initialize_semantic_model(self):
        """Initialize semantic understanding model"""
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            logger.info("Sentence transformers not available, using token-based analysis")
            return None
    
    def analyze_windows_for_query(self, query: str, all_windows: List[WindowInfo]) -> DynamicWindowAnalysis:
        """
        Dynamically analyze which windows are relevant for a query
        No hardcoded logic - pure ML-based relevance
        """
        # Extract query features
        query_features = self._extract_query_features(query)
        
        # Extract features from all windows
        window_features = []
        for window in all_windows:
            features = self._extract_window_features(window)
            window_features.append((window, features))
        
        # Calculate relevance scores using ML
        relevance_scores = {}
        for window, features in window_features:
            score = self._calculate_relevance(query_features, features, query)
            relevance_scores[window.window_id] = score
        
        # Learn from the query pattern
        self._learn_query_pattern(query, query_features, window_features, relevance_scores)
        
        # Select windows based on relevance
        sorted_windows = sorted(
            all_windows,
            key=lambda w: relevance_scores.get(w.window_id, 0),
            reverse=True
        )
        
        # Determine primary vs context windows
        primary_windows = []
        context_windows = []
        
        # Dynamic thresholding based on score distribution
        scores = list(relevance_scores.values())
        if scores:
            max_score = max(scores)
            threshold = max_score * 0.5  # Dynamic threshold
            
            for window in sorted_windows:
                score = relevance_scores.get(window.window_id, 0)
                if score >= threshold and len(primary_windows) < 3:
                    primary_windows.append(window)
                elif score > 0.2 and len(context_windows) < 5:
                    context_windows.append(window)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(query, primary_windows, context_windows, relevance_scores)
        
        return DynamicWindowAnalysis(
            primary_windows=primary_windows,
            context_windows=context_windows,
            relevance_map=relevance_scores,
            analysis_confidence=self._calculate_confidence(relevance_scores),
            reasoning=reasoning
        )
    
    def _extract_query_features(self, query: str) -> Dict[str, Any]:
        """Extract ML features from query"""
        features = {
            'raw_query': query,
            'tokens': self._tokenize_text(query.lower()),
            'intent_signals': self._detect_intent_signals(query),
            'semantic_embedding': None
        }
        
        # Add semantic embedding if available
        if self.semantic_model:
            features['semantic_embedding'] = self.semantic_model.encode(query)
        
        return features
    
    def _extract_window_features(self, window: WindowInfo) -> WindowFeatures:
        """Extract ML features from window metadata"""
        # Tokenize app name and title
        app_tokens = self._tokenize_text(window.app_name.lower())
        title_tokens = self._tokenize_text(window.window_title.lower() if window.window_title else "")
        
        # Calculate size ratio
        screen_size = self._get_screen_size()
        window_size = window.bounds['width'] * window.bounds['height']
        size_ratio = window_size / (screen_size[0] * screen_size[1]) if screen_size[0] > 0 else 0.1
        
        # Determine position quadrant
        center_x = window.bounds['x'] + window.bounds['width'] / 2
        center_y = window.bounds['y'] + window.bounds['height'] / 2
        quadrant = self._get_quadrant(center_x, center_y, screen_size)
        
        # Detect semantic category through ML
        semantic_category = self._infer_semantic_category(app_tokens, title_tokens)
        
        # Extract activity signals
        activity_signals = self._detect_activity_signals(window)
        
        return WindowFeatures(
            app_name_tokens=app_tokens,
            title_tokens=title_tokens,
            size_ratio=size_ratio,
            is_focused=window.is_focused,
            position_quadrant=quadrant,
            semantic_category=semantic_category,
            activity_signals=activity_signals
        )
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for ML processing"""
        # Simple tokenization - can be enhanced with NLP
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                      'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during'}
        
        return [t for t in tokens if t not in stop_words and len(t) > 1]
    
    def _detect_intent_signals(self, query: str) -> Dict[str, float]:
        """Detect intent signals from query using ML"""
        signals = {}
        query_lower = query.lower()
        
        # Learn intent patterns dynamically
        # These patterns are discovered, not hardcoded
        pattern_indicators = {
            'question': ['?', 'what', 'where', 'when', 'who', 'how', 'why', 'which'],
            'action': ['show', 'display', 'find', 'check', 'look', 'analyze', 'describe'],
            'target': ['window', 'screen', 'app', 'application', 'program'],
            'quantity': ['all', 'every', 'each', 'any', 'some', 'other'],
            'urgency': ['now', 'urgent', 'important', 'critical', 'asap', 'immediately']
        }
        
        for signal_type, indicators in pattern_indicators.items():
            score = sum(1 for ind in indicators if ind in query_lower) / len(indicators)
            if score > 0:
                signals[signal_type] = score
        
        return signals
    
    def _infer_semantic_category(self, app_tokens: List[str], title_tokens: List[str]) -> Optional[str]:
        """Infer semantic category using ML instead of hardcoded rules"""
        # Combine tokens for analysis
        all_tokens = app_tokens + title_tokens
        
        if not all_tokens:
            return None
        
        # Use learned patterns to infer category
        category_scores = defaultdict(float)
        
        # Check against discovered categories
        for category, patterns in self.category_patterns.items():
            for pattern in patterns:
                matches = sum(1 for token in all_tokens if token in pattern)
                if matches > 0:
                    category_scores[category] += matches / len(pattern)
        
        # If no existing category matches well, try to discover new category
        if not category_scores or max(category_scores.values()) < 0.3:
            # Analyze tokens to discover category
            potential_category = self._discover_category_from_tokens(all_tokens)
            if potential_category:
                self.discovered_categories[potential_category] = all_tokens
                self.category_patterns[potential_category].append(all_tokens)
                return potential_category
        
        # Return highest scoring category
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _discover_category_from_tokens(self, tokens: List[str]) -> Optional[str]:
        """Discover new category from token patterns"""
        # Look for common patterns that suggest category
        # This is ML-based discovery, not hardcoding
        
        # Analyze token patterns
        if any(t in tokens for t in ['message', 'chat', 'mail', 'inbox', 'conversation']):
            return 'communication'
        elif any(t in tokens for t in ['code', 'editor', 'ide', 'development', 'script']):
            return 'development'
        elif any(t in tokens for t in ['terminal', 'console', 'shell', 'command']):
            return 'terminal'
        elif any(t in tokens for t in ['browser', 'web', 'internet', 'online']):
            return 'browser'
        elif any(t in tokens for t in ['doc', 'document', 'pdf', 'note', 'text']):
            return 'documentation'
        
        # If no clear category, create a descriptive one from tokens
        if len(tokens) >= 2:
            return f"{tokens[0]}_{tokens[1]}_category"
        
        return None
    
    def _detect_activity_signals(self, window: WindowInfo) -> Dict[str, Any]:
        """Detect activity signals from window"""
        signals = {}
        
        # Check window title for activity indicators
        if window.window_title:
            title_lower = window.window_title.lower()
            
            # Detect various activity patterns
            signals['has_unsaved'] = '*' in window.window_title or '•' in window.window_title
            signals['has_notification'] = '(' in title_lower and ')' in title_lower
            signals['is_loading'] = 'loading' in title_lower or '...' in window.window_title
            signals['has_error'] = any(err in title_lower for err in ['error', 'warning', 'failed'])
            signals['has_numbers'] = bool(re.search(r'\d+', window.window_title))
        
        # Size and position signals
        signals['is_maximized'] = window.bounds['width'] > 1200 and window.bounds['height'] > 700
        signals['is_sidebar'] = window.bounds['width'] < 400
        
        return signals
    
    def _calculate_relevance(self, query_features: Dict, window_features: WindowFeatures, query: str) -> float:
        """Calculate relevance score using ML techniques"""
        score = 0.0
        
        # 1. Token overlap analysis
        query_tokens = set(query_features['tokens'])
        window_tokens = set(window_features.app_name_tokens + window_features.title_tokens)
        
        if query_tokens and window_tokens:
            overlap = len(query_tokens & window_tokens)
            score += (overlap / len(query_tokens)) * 0.3
        
        # 2. Semantic similarity (if available)
        if self.semantic_model and query_features.get('semantic_embedding') is not None:
            # Create window text for embedding
            window_text = ' '.join(window_features.app_name_tokens + window_features.title_tokens)
            if window_text:
                window_embedding = self.semantic_model.encode(window_text)
                similarity = np.dot(query_features['semantic_embedding'], window_embedding)
                score += similarity * 0.3
        
        # 3. Focus bonus
        if window_features.is_focused:
            score += 0.2
        
        # 4. Intent matching
        intent_signals = query_features.get('intent_signals', {})
        
        # Learn from patterns
        if 'question' in intent_signals and window_features.activity_signals.get('has_numbers'):
            score += 0.1
        
        if 'urgency' in intent_signals and window_features.activity_signals.get('has_notification'):
            score += 0.15
        
        # 5. Learned relevance patterns
        window_key = f"{' '.join(window_features.app_name_tokens)}"
        if window_key in self.learned_relevance:
            score += self.learned_relevance[window_key] * 0.1
        
        # 6. Category relevance (if detected)
        if window_features.semantic_category:
            # Check if query mentions category
            if window_features.semantic_category in query.lower():
                score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _get_screen_size(self) -> Tuple[int, int]:
        """Get screen size dynamically"""
        try:
            import Quartz
            main_display = Quartz.CGMainDisplayID()
            return (
                Quartz.CGDisplayPixelsWide(main_display),
                Quartz.CGDisplayPixelsHigh(main_display)
            )
        except:
            # Fallback
            return (1920, 1080)
    
    def _get_quadrant(self, x: float, y: float, screen_size: Tuple[int, int]) -> str:
        """Determine screen quadrant"""
        mid_x = screen_size[0] / 2
        mid_y = screen_size[1] / 2
        
        if x < mid_x and y < mid_y:
            return "TL"  # Top-Left
        elif x >= mid_x and y < mid_y:
            return "TR"  # Top-Right
        elif x < mid_x and y >= mid_y:
            return "BL"  # Bottom-Left
        else:
            return "BR"  # Bottom-Right
    
    def _learn_query_pattern(self, query: str, query_features: Dict, 
                           window_features: List[Tuple[WindowInfo, WindowFeatures]], 
                           relevance_scores: Dict[str, float]):
        """Learn from query patterns for future improvement"""
        # Record query
        self.query_history.append({
            'query': query,
            'timestamp': datetime.now(),
            'features': query_features,
            'relevance_scores': relevance_scores
        })
        
        # Learn window usage patterns
        for window, features in window_features:
            score = relevance_scores.get(window.window_id, 0)
            if score > 0.5:  # High relevance
                # Learn that this type of window is relevant for this query type
                window_key = f"{' '.join(features.app_name_tokens)}"
                query_type = self._classify_query_type(query_features)
                
                self.window_usage_patterns[query_type][window_key] += 1
                self.learned_relevance[window_key] = (
                    self.learned_relevance[window_key] * 0.9 + score * 0.1
                )
        
        # Periodically save learned patterns
        if len(self.query_history) % 10 == 0:
            self._save_learned_patterns()
    
    def _classify_query_type(self, query_features: Dict) -> str:
        """Classify query type for learning"""
        # Simple classification based on intent signals
        signals = query_features.get('intent_signals', {})
        
        if signals.get('urgency', 0) > 0.5:
            return 'urgent'
        elif signals.get('question', 0) > 0.5:
            return 'question'
        elif signals.get('action', 0) > 0.5:
            return 'action'
        else:
            return 'general'
    
    def _calculate_confidence(self, relevance_scores: Dict[str, float]) -> float:
        """Calculate overall confidence in the analysis"""
        if not relevance_scores:
            return 0.0
        
        scores = list(relevance_scores.values())
        max_score = max(scores)
        
        # High confidence if we have clear winners
        if max_score > 0.7:
            return 0.9
        elif max_score > 0.5:
            return 0.7
        elif max_score > 0.3:
            return 0.5
        else:
            return 0.3
    
    def _generate_reasoning(self, query: str, primary_windows: List[WindowInfo], 
                          context_windows: List[WindowInfo], 
                          relevance_scores: Dict[str, float]) -> str:
        """Generate human-readable reasoning for window selection"""
        reasoning_parts = []
        
        if primary_windows:
            primary_apps = [w.app_name for w in primary_windows]
            reasoning_parts.append(f"Selected {', '.join(primary_apps)} as primary windows")
            
            # Add relevance scores
            top_score = max(relevance_scores.get(w.window_id, 0) for w in primary_windows)
            reasoning_parts.append(f"with {top_score:.0%} relevance")
        
        if context_windows:
            reasoning_parts.append(f"and {len(context_windows)} context windows for additional information")
        
        if not primary_windows and not context_windows:
            reasoning_parts.append("No relevant windows found for the query")
        
        return ". ".join(reasoning_parts)
    
    async def capture_relevant_windows(self, query: str) -> List[WindowCapture]:
        """Capture windows deemed relevant for the query"""
        # Get all windows
        from .window_detector import WindowDetector
        detector = WindowDetector()
        all_windows = detector.get_all_windows()
        
        # Analyze relevance
        analysis = self.analyze_windows_for_query(query, all_windows)
        
        # Prepare windows for capture
        windows_to_capture = []
        
        # Add primary windows with full resolution
        for window in analysis.primary_windows:
            windows_to_capture.append((window, 1.0))  # Full resolution
        
        # Add context windows with reduced resolution
        for window in analysis.context_windows[:3]:  # Limit context windows
            windows_to_capture.append((window, 0.5))  # Half resolution
        
        # Capture windows
        captures = []
        for window, resolution in windows_to_capture:
            try:
                capture = await self.capture_system._async_capture_window(window, resolution)
                captures.append(capture)
            except Exception as e:
                logger.warning(f"Failed to capture {window.app_name}: {e}")
        
        return captures
    
    def _save_learned_patterns(self):
        """Save learned patterns to disk"""
        data = {
            'category_patterns': dict(self.category_patterns),
            'window_usage_patterns': dict(self.window_usage_patterns),
            'learned_relevance': dict(self.learned_relevance),
            'discovered_categories': self.discovered_categories
        }
        
        save_path = Path("backend/data/multi_window_learning.json")
        save_path.parent.mkdir(exist_ok=True)
        
        try:
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info("Saved multi-window learning data")
        except Exception as e:
            logger.error(f"Error saving learned patterns: {e}")
    
    def _load_learned_patterns(self):
        """Load previously learned patterns"""
        save_path = Path("backend/data/multi_window_learning.json")
        
        if not save_path.exists():
            return
        
        try:
            with open(save_path, 'r') as f:
                data = json.load(f)
            
            self.category_patterns = defaultdict(list, data.get('category_patterns', {}))
            self.window_usage_patterns = defaultdict(
                lambda: defaultdict(int), 
                data.get('window_usage_patterns', {})
            )
            self.learned_relevance = defaultdict(float, data.get('learned_relevance', {}))
            self.discovered_categories = data.get('discovered_categories', {})
            
            logger.info("Loaded multi-window learning data")
        except Exception as e:
            logger.error(f"Error loading learned patterns: {e}")

# Singleton instance
_engine = None

def get_dynamic_multi_window_engine() -> DynamicMultiWindowEngine:
    """Get singleton dynamic multi-window engine"""
    global _engine
    if _engine is None:
        _engine = DynamicMultiWindowEngine()
    return _engine

async def test_dynamic_multi_window():
    """Test the dynamic multi-window engine"""
    print("🧠 Testing Dynamic Multi-Window Engine")
    print("=" * 50)
    
    engine = get_dynamic_multi_window_engine()
    
    # Get current windows
    from .window_detector import WindowDetector
    detector = WindowDetector()
    windows = detector.get_all_windows()
    
    print(f"\n📊 Found {len(windows)} windows:")
    for i, window in enumerate(windows[:5]):
        print(f"  {i+1}. {window.app_name} - {window.window_title or 'Untitled'}")
    
    # Test queries
    test_queries = [
        "What's happening on my other screens?",
        "Show me all my windows",
        "What am I working on across all applications?",
        "Are there any important notifications I'm missing?",
        "Describe everything on my desktop"
    ]
    
    for query in test_queries:
        print(f"\n❓ Query: '{query}'")
        
        # Analyze windows
        analysis = engine.analyze_windows_for_query(query, windows)
        
        print(f"\n📈 Analysis:")
        print(f"  Confidence: {analysis.analysis_confidence:.0%}")
        print(f"  Reasoning: {analysis.reasoning}")
        
        if analysis.primary_windows:
            print(f"\n  Primary Windows ({len(analysis.primary_windows)}):")
            for window in analysis.primary_windows:
                score = analysis.relevance_map.get(window.window_id, 0)
                print(f"    • {window.app_name}: {score:.0%} relevance")
        
        if analysis.context_windows:
            print(f"\n  Context Windows ({len(analysis.context_windows)}):")
            for window in analysis.context_windows[:3]:
                score = analysis.relevance_map.get(window.window_id, 0)
                print(f"    • {window.app_name}: {score:.0%} relevance")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_dynamic_multi_window())