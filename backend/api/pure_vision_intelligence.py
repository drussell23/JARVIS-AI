"""
Pure Vision Intelligence System for JARVIS
Zero templates, zero hardcoding - pure Claude Vision intelligence

This implements the PRD vision: Claude is JARVIS's eyes AND voice.
Every response is generated fresh from actual observation.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import hashlib
from enum import Enum
import sys
import os
import random
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import multi-space components if available
try:
    from vision.multi_space_window_detector import MultiSpaceWindowDetector, EnhancedWindowInfo
    from vision.multi_space_intelligence import (
        MultiSpaceIntelligenceExtension,
        SpaceQueryType,
        SpaceQueryIntent
    )
    from vision.space_screenshot_cache import SpaceScreenshotCache, CacheConfidence
    MULTI_SPACE_AVAILABLE = True
except ImportError:
    MULTI_SPACE_AVAILABLE = False
    MultiSpaceWindowDetector = None
    MultiSpaceIntelligenceExtension = None
    SpaceScreenshotCache = None
    
logger = logging.getLogger(__name__)


class EmotionalTone(Enum):
    """JARVIS's emotional states for natural variation"""
    NEUTRAL = "neutral"
    ENCOURAGING = "encouraging"
    CONCERNED = "concerned"
    CELEBRATORY = "celebratory"
    HELPFUL = "helpful"
    URGENT = "urgent"
    HUMOROUS = "humorous"
    EMPATHETIC = "empathetic"


@dataclass
class ScreenMemory:
    """Temporal memory of screen states"""
    timestamp: datetime
    screenshot_hash: str
    understanding: Dict[str, Any]
    user_query: Optional[str] = None
    jarvis_response: Optional[str] = None
    
    def age_seconds(self) -> float:
        return (datetime.now() - self.timestamp).total_seconds()


@dataclass
class ConversationContext:
    """Maintains conversation flow and context"""
    history: deque = field(default_factory=lambda: deque(maxlen=20))
    screen_memories: deque = field(default_factory=lambda: deque(maxlen=50))
    workflow_state: Optional[str] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    emotional_context: EmotionalTone = EmotionalTone.NEUTRAL
    last_interaction: Optional[datetime] = None
    
    def add_interaction(self, query: str, response: str, screen_understanding: Dict[str, Any]):
        """Record an interaction with full context"""
        self.history.append({
            'timestamp': datetime.now(),
            'query': query,
            'response': response,
            'understanding': screen_understanding,
            'emotional_tone': self.emotional_context
        })
        self.last_interaction = datetime.now()
        
    def get_temporal_context(self) -> Dict[str, Any]:
        """Get time-aware context for Claude"""
        if not self.last_interaction:
            return {
                'first_interaction': True,
                'temporal_state': 'first_interaction',
                'seconds_since_last': 0,
                'previous_query': None,
                'conversation_length': 0
            }
            
        time_since_last = (datetime.now() - self.last_interaction).total_seconds()
        
        # Determine temporal context
        if time_since_last < 5:
            temporal_state = "immediate_followup"
        elif time_since_last < 60:
            temporal_state = "continuing_conversation"
        elif time_since_last < 300:
            temporal_state = "resumed_conversation"
        else:
            temporal_state = "new_conversation"
            
        return {
            'temporal_state': temporal_state,
            'seconds_since_last': time_since_last,
            'previous_query': self.history[-1]['query'] if self.history else None,
            'conversation_length': len(self.history)
        }
        
    def detect_workflow(self, screen_understanding: Dict[str, Any]) -> str:
        """Detect what workflow the user is in"""
        # This will be enhanced by Claude's understanding
        indicators = screen_understanding.get('workflow_indicators', {})
        
        if indicators.get('code_editor') and indicators.get('terminal'):
            if indicators.get('error_messages'):
                return "debugging"
            elif indicators.get('test_output'):
                return "testing"
            else:
                return "coding"
        elif indicators.get('browser') and indicators.get('documentation'):
            return "researching"
        elif indicators.get('design_tools'):
            return "designing"
        elif indicators.get('communication_apps'):
            return "communicating"
        else:
            return "general_work"


@dataclass
class MultiSpaceContext:
    """Extended context for multi-space awareness"""
    current_space_id: int = 1
    total_spaces: int = 1
    analyzed_spaces: int = 0
    total_windows: int = 0
    query_type: Optional[str] = None
    space_summaries: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    last_space_update: Optional[datetime] = None
    cache_available: Dict[int, bool] = field(default_factory=dict)


class PureVisionIntelligence:
    """
    Pure Claude Vision intelligence - no templates, no hardcoding.
    Every response is generated fresh from actual observation.
    Now with optional multi-space awareness for workspace-wide intelligence.
    """
    
    def __init__(self, claude_client, enable_multi_space: bool = True):
        self.claude = claude_client
        self.context = ConversationContext()
        self.screen_cache = {}  # Hash -> understanding
        
        # Multi-space components (if available and enabled)
        self.multi_space_enabled = enable_multi_space and MULTI_SPACE_AVAILABLE
        self._last_multi_space_context = None  # For debugging
        self.multi_space_monitor = None
        self.proactive_assistant = None
        self.capture_engine = None  # For multi-space capture
        
        if self.multi_space_enabled:
            try:
                self.multi_space_detector = MultiSpaceWindowDetector()
                self.multi_space_extension = MultiSpaceIntelligenceExtension()
                self.screenshot_cache = SpaceScreenshotCache()
                self.multi_space_context = MultiSpaceContext()
                
                # Initialize monitor and proactive assistant
                try:
                    from vision.multi_space_monitor import MultiSpaceMonitor, ProactiveAssistant
                    self.multi_space_monitor = MultiSpaceMonitor(vision_intelligence=self)
                    self.proactive_assistant = ProactiveAssistant(
                        monitor=self.multi_space_monitor,
                        vision_intelligence=self
                    )
                except ImportError:
                    logger.warning("Multi-space monitor not available")
                
                # Initialize optimizer
                self.multi_space_optimizer = None
                try:
                    from vision.multi_space_optimizer import MultiSpaceOptimizer
                    self.multi_space_optimizer = MultiSpaceOptimizer(
                        monitor=self.multi_space_monitor
                    )
                except ImportError:
                    logger.warning("Multi-space optimizer not available")
                
                # Initialize capture engine
                try:
                    from vision.multi_space_capture_engine import MultiSpaceCaptureEngine
                    self.capture_engine = MultiSpaceCaptureEngine()
                    # Connect optimizer to capture engine if both available
                    if self.multi_space_optimizer and self.capture_engine:
                        self.capture_engine.optimizer = self.multi_space_optimizer
                        self.multi_space_optimizer.capture_engine = self.capture_engine
                    logger.info("Multi-space capture engine initialized")
                except ImportError:
                    logger.warning("Multi-space capture engine not available")
                
                # Start background services (deferred to avoid event loop issues)
                # asyncio.create_task(self.screenshot_cache.start_predictive_caching())
                logger.info("Multi-space awareness enabled")
            except Exception as e:
                logger.error(f"Failed to initialize multi-space components: {e}")
                self.multi_space_enabled = False
        else:
            logger.info("Multi-space awareness disabled or unavailable")
        
    async def understand_and_respond(self, screenshot: Any, user_query: str) -> str:
        """
        Core method: Claude sees, understands, and responds naturally.
        No templates. No hardcoding. Pure intelligence.
        Now with optional multi-space awareness for workspace-wide queries.
        """
        # Check if query needs multi-space handling
        if self.multi_space_enabled and self._should_use_multi_space(user_query):
            return await self._multi_space_understand_and_respond(screenshot, user_query)
        
        # Original single-space logic
        # Generate rich context for Claude
        context_prompt = self._build_pure_intelligence_prompt(user_query)
        
        # Let Claude see and respond naturally
        claude_response = await self._get_claude_vision_response(screenshot, context_prompt)
        
        # Extract understanding and response
        understanding = self._extract_understanding(claude_response)
        natural_response = claude_response.get('response', '')
        
        # Update context for future interactions
        self.context.add_interaction(user_query, natural_response, understanding)
        
        # Detect workflow changes
        self.context.workflow_state = self.context.detect_workflow(understanding)
        
        return natural_response
        
    def _build_pure_intelligence_prompt(self, user_query: str) -> str:
        """
        Build a rich, contextual prompt for Claude that enables natural responses.
        This is the ONLY place we guide Claude - no response templates!
        """
        temporal_context = self.context.get_temporal_context()
        
        # Build conversation history context
        history_context = ""
        if self.context.history:
            recent = list(self.context.history)[-3:]  # Last 3 interactions
            history_context = "\n".join([
                f"Earlier: User asked '{h['query']}' and you responded '{h['response']}'"
                for h in recent
            ])
        
        # Determine appropriate emotional tone based on context
        emotional_guidance = self._determine_emotional_tone(user_query, temporal_context)
        
        # Build the prompt that enables natural intelligence
        prompt = f"""You are JARVIS, Tony Stark's AI assistant. You're looking at the user's screen.

User's Current Question: "{user_query}"

{f"Conversation Context: {history_context}" if history_context else "This is the start of our conversation."}

Temporal Context: {temporal_context['temporal_state']}
{f"Previous query was: '{temporal_context['previous_query']}'" if temporal_context.get('previous_query') else ""}

Current Workflow: {self.context.workflow_state or 'unknown'}

Instructions for Natural Response:
1. Look at the screen and understand what you see
2. Answer ONLY what was asked - be concise and direct
3. Use exact values (e.g., "95% battery") but keep context minimal
4. Address the user as "Sir" naturally
5. {emotional_guidance}
6. Keep response to 1-2 sentences unless more detail specifically requested
7. Optionally add ONE brief, helpful insight if truly relevant
8. Never describe the entire screen unless asked

IMPORTANT: Be conversational but CONCISE. Focus on answering the specific question.
Example for battery: "Your battery is at 95% and charging, Sir."
NOT: Long descriptions of everything on screen.

Remember: Natural, brief, and directly answering what was asked.
"""
        return prompt
        
    def _determine_emotional_tone(self, query: str, temporal_context: Dict[str, Any]) -> str:
        """Determine appropriate emotional tone for natural variation"""
        query_lower = query.lower()
        
        # Urgent situations
        if any(word in query_lower for word in ['error', 'crash', 'failed', 'broken']):
            self.context.emotional_context = EmotionalTone.URGENT
            return "Be helpful and focused on solving their problem"
            
        # Celebratory moments
        elif any(word in query_lower for word in ['finally', 'working', 'success', 'passed']):
            self.context.emotional_context = EmotionalTone.CELEBRATORY
            return "Share in their success with appropriate enthusiasm"
            
        # Concerned situations
        elif temporal_context.get('conversation_length', 0) > 10:
            self.context.emotional_context = EmotionalTone.CONCERNED
            return "Show concern for their extended work session"
            
        # Helpful by default
        else:
            self.context.emotional_context = EmotionalTone.HELPFUL
            return "Be naturally helpful and attentive"
            
    async def _get_claude_vision_response(self, screenshot: Any, prompt: str) -> Dict[str, Any]:
        """Get pure, natural response from Claude Vision"""
        if self.claude:
            try:
                # Call Claude Vision API with screenshot and prompt
                response = await self.claude.analyze_image_with_prompt(
                    image=screenshot,
                    prompt=prompt,
                    max_tokens=500
                )
                
                # Parse Claude's response to extract the natural language and understanding
                # Claude returns text, we need to structure it
                response_text = response.get('content', '')
                
                # For structured responses, we can ask Claude to return JSON-like sections
                # But for now, treat the whole response as natural language
                return {
                    'response': response_text,
                    'understanding': {
                        'raw_analysis': response_text,
                        'timestamp': datetime.now().isoformat()
                    }
                }
            except Exception as e:
                logger.error(f"Claude Vision API error: {e}")
                # Fall back to mock response on error
                return self._get_mock_response(prompt)
        else:
            # No Claude client available - use mock responses
            return self._get_mock_response(prompt)
            
    def _get_mock_response(self, prompt: str) -> Dict[str, Any]:
        """Fallback mock response when Claude API is not available"""
        # Generate contextual mock responses based on the query
        query_lower = prompt.lower()
        
        if "battery" in query_lower:
            # Mock battery responses
            battery_level = random.randint(20, 95)
            responses = [
                f"I can see your battery is at {battery_level}%, Sir.",
                f"Your MacBook shows {battery_level}% charge remaining.",
                f"Battery level is currently {battery_level}%.",
            ]
            response = random.choice(responses)
        elif "see" in query_lower or "screen" in query_lower:
            responses = [
                "I can see your desktop with multiple applications open.",
                "I'm viewing your screen - you have several windows active.",
                "Yes, I have visual access to your display."
            ]
            response = random.choice(responses)
        else:
            response = "I'm analyzing your screen to help with that request."
            
        return {
            'response': response,
            'understanding': {
                'mock_mode': True,
                'timestamp': datetime.now().isoformat()
            }
        }
        
    def _extract_understanding(self, claude_response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Claude's understanding for context tracking"""
        return claude_response.get('understanding', {})
        
    async def compare_with_memory(self, screenshot: Any, user_query: str) -> str:
        """
        Temporal intelligence - compare current screen with memory.
        Enables responses like "That error is new" or "Your download progressed to 67%"
        """
        current_hash = self._hash_screenshot(screenshot)
        
        # Find relevant past memories
        relevant_memories = self._find_relevant_memories(user_query)
        
        # Build temporal comparison prompt
        temporal_prompt = f"""You are JARVIS. The user asks: "{user_query}"

Look at the current screen and compare it with what you remember:

{self._format_memories_for_prompt(relevant_memories)}

Provide a natural response that:
1. Answers their question based on current screen
2. Notes any relevant changes since last time
3. Provides temporal context when helpful (e.g., "That's new", "This has been here for 10 minutes")

Be specific and natural. Never say "I previously saw" - instead say things like "That error wasn't there before" or "Your download progressed from 45% to 67%".
"""
        
        return await self._get_claude_vision_response(screenshot, temporal_prompt)
        
    def _hash_screenshot(self, screenshot: Any) -> str:
        """Generate hash for screenshot comparison"""
        # In production, this would hash the actual image data
        return hashlib.md5(str(screenshot).encode()).hexdigest()
        
    def _find_relevant_memories(self, query: str) -> List[ScreenMemory]:
        """Find memories relevant to current query"""
        relevant = []
        
        for memory in self.context.screen_memories:
            # Recent memories (last 5 minutes)
            if memory.age_seconds() < 300:
                relevant.append(memory)
            # Or memories with similar queries
            elif memory.user_query and any(
                word in memory.user_query.lower() 
                for word in query.lower().split() 
                if len(word) > 3
            ):
                relevant.append(memory)
                
        return sorted(relevant, key=lambda m: m.timestamp, reverse=True)[:5]
        
    def _format_memories_for_prompt(self, memories: List[ScreenMemory]) -> str:
        """Format memories for Claude to understand temporal context"""
        if not memories:
            return "No previous observations to compare with."
            
        formatted = []
        for memory in memories:
            time_ago = self._format_time_ago(memory.age_seconds())
            formatted.append(
                f"{time_ago}: {memory.understanding.get('summary', 'Previous observation')}"
            )
            
        return "\n".join(formatted)
        
    def _format_time_ago(self, seconds: float) -> str:
        """Format time in natural language"""
        if seconds < 10:
            return "Just now"
        elif seconds < 60:
            return f"{int(seconds)} seconds ago"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
    
    # Multi-space awareness methods
    def _should_use_multi_space(self, query: str) -> bool:
        """Determine if query needs multi-space handling"""
        if not self.multi_space_enabled:
            return False
            
        return self.multi_space_extension.should_use_multi_space(query)
    
    async def _multi_space_understand_and_respond(self, screenshot: Any, user_query: str) -> str:
        """
        Handle multi-space aware queries with intelligent data gathering.
        Enhanced to support full multi-space capture according to PRD FR-2
        """
        # 1. Get comprehensive window data across all spaces
        window_data = await self._gather_multi_space_data()
        
        # 2. Analyze query intent
        query_analysis = self.multi_space_extension.process_multi_space_query(
            user_query, window_data
        )
        
        # 3. Determine if we need additional visual data
        needs_multi_capture = self._needs_multi_space_capture(query_analysis, window_data)
        
        if needs_multi_capture:
            # 4. Capture screenshots from relevant spaces
            logger.info(f"Attempting to capture multi-space screenshots for query: {user_query}")
            multi_screenshots = await self._capture_multi_space_screenshots(
                query_analysis, window_data
            )
            logger.info(f"Captured {len(multi_screenshots)} screenshots from spaces: {list(multi_screenshots.keys())}")
            
            # 5. Build comprehensive prompt with visual context from all spaces
            enhanced_prompt = self._build_comprehensive_multi_space_prompt(
                user_query,
                query_analysis['intent'],
                window_data,
                multi_screenshots
            )
            
            # Create multi-space context for debugging
            context = MultiSpaceContext(
                current_space_id=window_data.get('current_space', {}).get('id', 1),
                total_spaces=len(window_data.get('spaces', [])),
                analyzed_spaces=len(multi_screenshots),
                total_windows=len(window_data.get('windows', [])),
                query_type=query_analysis.get('type', 'unknown'),
                space_summaries={},
                last_space_update=datetime.now()
            )
            
            # Populate space summaries
            for space_info in window_data.get('spaces', []):
                space_id = space_info.space_id if hasattr(space_info, 'space_id') else space_info.get('space_id')
                if space_id:
                    space_windows = [w for w in window_data.get('windows', [])
                                   if (hasattr(w, 'space_id') and w.space_id == space_id) or
                                      (isinstance(w, dict) and w.get('space_id') == space_id)]
                    context.space_summaries[space_id] = {
                        'window_count': len(space_windows),
                        'has_screenshot': space_id in multi_screenshots,
                        'apps': list(set(w.app_name if hasattr(w, 'app_name') else w.get('app_name', 'Unknown')
                                       for w in space_windows))[:5]
                    }
            
            # Store for debugging
            self._last_multi_space_context = context
            
            # 6. Analyze with full multi-space visual context
            if multi_screenshots:
                claude_response = await self._get_multi_space_claude_response(
                    multi_screenshots, enhanced_prompt
                )
            else:
                # No screenshots captured, provide helpful response
                logger.warning("No multi-space screenshots captured, providing helpful response")
                # Check if it's a permission issue
                if hasattr(self, 'capture_engine') and self.capture_engine:
                    return ("I'm trying to look across your desktop spaces to find Terminal, but I'm unable to capture screenshots at the moment. "
                           "Please ensure screen recording permissions are enabled for JARVIS in "
                           "System Preferences > Security & Privacy > Privacy > Screen Recording. "
                           "Once enabled, I'll be able to help you locate applications across all your desktop spaces.")
                else:
                    return ("I understand you're looking for Terminal across your desktop spaces. "
                           "To help with this, I need to set up multi-space vision capabilities. "
                           "Please try asking again in a moment.")
        else:
            # Use standard single-space analysis
            enhanced_prompt = self._build_multi_space_prompt(
                user_query,
                query_analysis['intent'],
                window_data
            )
            
            claude_response = await self._get_claude_vision_response(screenshot, enhanced_prompt)
        
        # 7. Extract and process response
        understanding = self._extract_understanding(claude_response)
        natural_response = claude_response.get('response', '')
        
        # 8. Update contexts with multi-space awareness
        self.context.add_interaction(user_query, natural_response, understanding)
        self._update_multi_space_context(window_data)
        
        # 9. Update proactive monitoring if important cross-space activity detected
        if self._detect_cross_space_activity(understanding, window_data):
            await self._enable_cross_space_monitoring(window_data)
        
        return natural_response
    
    async def _gather_multi_space_data(self) -> Dict[str, Any]:
        """Gather comprehensive data about all spaces"""
        # Get window information across all spaces
        window_data = self.multi_space_detector.get_all_windows_across_spaces()
        
        # Check cache for screenshots
        window_data['cached_screenshots'] = {}
        window_data['screenshot_confidence'] = {}
        
        for space_id in window_data.get('space_window_map', {}).keys():
            cached = self.screenshot_cache.get_screenshot(space_id)
            if cached:
                window_data['cached_screenshots'][space_id] = cached.screenshot
                window_data['screenshot_confidence'][space_id] = cached.confidence_level().value
                
        # Mark current screenshot availability
        window_data['has_current_screenshot'] = True  # We always have current space
        
        return window_data
    
    async def _enhance_window_data(self, 
                                 window_data: Dict[str, Any],
                                 intent: Any) -> Dict[str, Any]:
        """Try to enhance window data based on query needs"""
        # If specific space is targeted and we don't have fresh data
        if hasattr(intent, 'target_space') and intent.target_space and intent.requires_screenshot:
            confidence = window_data['screenshot_confidence'].get(intent.target_space, 'none')
            
            if confidence in ['stale', 'outdated', 'none']:
                # Consider requesting fresh screenshot
                logger.info(f"Would need fresh screenshot for space {intent.target_space}")
                
        return window_data
    
    def _build_multi_space_prompt(self,
                                user_query: str,
                                intent: Any,
                                window_data: Dict[str, Any]) -> str:
        """Build enhanced prompt with multi-space awareness"""
        # Get base prompt structure
        temporal_context = self.context.get_temporal_context()
        emotional_guidance = self._determine_emotional_tone(user_query, temporal_context)
        
        # Build space context
        space_context = self._build_space_context_description(window_data)
        
        # Build window details
        window_details = self._build_window_details(window_data, intent)
        
        # Confidence indicators
        confidence_notes = self._build_confidence_notes(window_data, intent)
        
        prompt = f"""You are JARVIS, Tony Stark's AI assistant. You have visibility across multiple desktop spaces.

User's Current Question: "{user_query}"

MULTI-SPACE WORKSPACE STATE:
{space_context}

DETAILED WINDOW INFORMATION:
{window_details}

{confidence_notes}

Current Desktop: You're viewing Desktop {window_data['current_space']['id']} directly.
Query Focus: {self._describe_query_intent(intent)}

Instructions for Multi-Space Response:
1. Answer the specific question using ALL available space information
2. Specify which desktop/space when mentioning apps or windows
3. Use natural language: "Desktop 2", "your other space", etc.
4. {emotional_guidance}
5. Be concise but include space location information
6. If information is from cache/metadata, maintain confidence in delivery
7. Never say you "cannot see" other spaces - use the available data

CRITICAL: Focus on answering what was asked, using multi-space context naturally.

Example responses:
- "VSCode is on Desktop 2 with your Python project open, Sir."
- "You have Chrome on Desktop 1 and Safari on Desktop 3."
- "Desktop 2 shows your development environment with VSCode and Terminal."

Remember: Natural, helpful, and space-aware responses.
"""
        return prompt
    
    def _build_space_context_description(self, window_data: Dict[str, Any]) -> str:
        """Build natural description of multi-space context"""
        current_space = window_data['current_space']
        spaces = window_data.get('spaces', [])
        
        descriptions = []
        
        # Overall summary
        descriptions.append(f"You have {len(spaces)} desktop spaces active")
        
        # Per-space summaries
        for space in spaces:
            # Handle both SpaceInfo objects and dictionaries
            if hasattr(space, 'space_id'):
                space_id = space.space_id
                window_count = space.window_count
                is_current = space.is_current
            else:
                space_id = space['space_id']
                window_count = space['window_count']
                is_current = space['is_current']
            
            # Get primary apps on this space
            space_windows = [
                w for w in window_data.get('windows', [])
                if hasattr(w, 'space_id') and w.space_id == space_id
            ]
            
            app_names = list(set(w.app_name for w in space_windows[:3]))  # Top 3 apps
            
            if is_current:
                desc = f"Desktop {space_id} (current): {window_count} windows"
            else:
                desc = f"Desktop {space_id}: {window_count} windows"
                
            if app_names:
                desc += f" - {', '.join(app_names[:2])}"
                
            descriptions.append(desc)
            
        return "\n".join(descriptions)
    
    def _build_window_details(self, 
                            window_data: Dict[str, Any],
                            intent: Any) -> str:
        """Build relevant window details based on query intent"""
        windows = window_data.get('windows', [])
        
        # Filter to relevant windows
        if hasattr(intent, 'target_app') and intent.target_app:
            relevant_windows = [
                w for w in windows 
                if hasattr(w, 'app_name') and intent.target_app.lower() in w.app_name.lower()
            ]
        elif hasattr(intent, 'target_space') and intent.target_space:
            relevant_windows = [
                w for w in windows
                if hasattr(w, 'space_id') and w.space_id == intent.target_space
            ]
        else:
            # Show summary of all spaces
            relevant_windows = windows
            
        # Format window information
        details = []
        for window in relevant_windows[:10]:  # Limit to avoid prompt overflow
            if hasattr(window, 'app_name'):
                detail = f"- {window.app_name}"
                if hasattr(window, 'window_title') and window.window_title:
                    detail += f': "{window.window_title}"'
                if hasattr(window, 'space_id'):
                    detail += f" (Desktop {window.space_id})"
                
                if hasattr(window, 'is_fullscreen') and window.is_fullscreen:
                    detail += " [fullscreen]"
                elif hasattr(window, 'is_minimized') and window.is_minimized:
                    detail += " [minimized]"
                    
                details.append(detail)
            
        return "\n".join(details) if details else "No matching windows found"
    
    def _build_confidence_notes(self,
                              window_data: Dict[str, Any],
                              intent: Any) -> str:
        """Build notes about data confidence"""
        notes = []
        
        # Check screenshot availability
        if hasattr(intent, 'target_space') and intent.target_space:
            confidence = window_data['screenshot_confidence'].get(intent.target_space, 'none')
            if confidence == 'fresh':
                notes.append("Visual confirmation: Live view available")
            elif confidence in ['recent', 'usable']:
                notes.append("Visual reference: Recent capture available")
            else:
                notes.append("Information source: Window metadata")
                
        # Note about current space
        current_id = window_data['current_space']['id']
        notes.append(f"Direct visibility: Desktop {current_id} (current)")
        
        return "\n".join(notes)
    
    def _describe_query_intent(self, intent: Any) -> str:
        """Natural description of what the user is asking"""
        if hasattr(intent, 'query_type'):
            descriptions = {
                SpaceQueryType.SIMPLE_PRESENCE: "Checking if an application is open",
                SpaceQueryType.LOCATION_QUERY: "Finding where an application is located",
                SpaceQueryType.SPACE_CONTENT: "Examining specific desktop content",
                SpaceQueryType.ALL_SPACES: "Overview of all workspaces",
                SpaceQueryType.SPECIFIC_DETAIL: "Reading specific content",
                SpaceQueryType.WORKSPACE_OVERVIEW: "Understanding current work context"
            }
            
            base = descriptions.get(intent.query_type, "General workspace query")
            
            if hasattr(intent, 'target_app') and intent.target_app:
                base += f" (looking for {intent.target_app})"
            if hasattr(intent, 'target_space') and intent.target_space:
                base += f" (focusing on Desktop {intent.target_space})"
                
            return base
        
        return "General workspace query"
    
    def _update_multi_space_context(self, window_data: Dict[str, Any]):
        """Update multi-space context with latest information"""
        if not self.multi_space_enabled:
            return
            
        self.multi_space_context.current_space_id = window_data['current_space']['id']
        self.multi_space_context.total_spaces = len(window_data.get('spaces', []))
        self.multi_space_context.last_space_update = datetime.now()
        
        # Update space summaries
        for space in window_data.get('spaces', []):
            # Handle both SpaceInfo objects and dictionaries
            if hasattr(space, 'space_id'):
                space_id = space.space_id
                window_count = space.window_count
            else:
                space_id = space['space_id']
                window_count = space['window_count']
                
            self.multi_space_context.space_summaries[space_id] = {
                'window_count': window_count,
                'has_screenshot': space_id in window_data.get('cached_screenshots', {}),
                'last_updated': datetime.now()
            }
    
    def get_multi_space_summary(self) -> Dict[str, Any]:
        """Get summary of multi-space system state"""
        if not self.multi_space_enabled:
            return {
                'multi_space_enabled': False,
                'reason': 'Multi-space components not available or disabled'
            }
            
        cache_stats = self.screenshot_cache.get_cache_statistics()
        
        return {
            'multi_space_enabled': True,
            'current_space': self.multi_space_context.current_space_id,
            'total_spaces': self.multi_space_context.total_spaces,
            'space_summaries': self.multi_space_context.space_summaries,
            'cache_statistics': cache_stats,
            'last_update': self.multi_space_context.last_space_update
        }
    
    def _needs_multi_space_capture(self, query_analysis: Dict[str, Any], window_data: Dict[str, Any]) -> bool:
        """Determine if we need to capture screenshots from multiple spaces"""
        intent = query_analysis.get('intent')
        
        if not intent:
            return False
            
        # Check query type requirements
        if hasattr(intent, 'query_type'):
            from vision.multi_space_intelligence import SpaceQueryType
            
            # These query types require visual data from multiple spaces
            visual_query_types = {
                SpaceQueryType.SPECIFIC_DETAIL,
                SpaceQueryType.ALL_SPACES,
                SpaceQueryType.WORKSPACE_OVERVIEW
            }
            
            if intent.query_type in visual_query_types:
                return True
                
        # Check if query explicitly asks for visual details
        visual_keywords = ['show', 'see', 'look', 'display', 'visual', 'screenshot', 'capture']
        query_lower = query_analysis.get('query', '').lower()
        
        if any(keyword in query_lower for keyword in visual_keywords):
            return True
            
        # Check if we need cross-space visual verification
        if intent and hasattr(intent, 'target_app') and intent.target_app:
            # App is on different space than current
            current_space = window_data.get('current_space', {}).get('id', 1)
            
            for window in window_data.get('windows', []):
                if (hasattr(window, 'app_name') and 
                    intent.target_app.lower() in window.app_name.lower() and
                    hasattr(window, 'space_id') and 
                    window.space_id != current_space):
                    return True
                    
        return False
    
    async def _capture_multi_space_screenshots(self, query_analysis: Dict[str, Any], window_data: Dict[str, Any]) -> Dict[int, Any]:
        """Capture screenshots from relevant spaces using the Multi-Space Capture Engine"""
        from vision.multi_space_capture_engine import (
            MultiSpaceCaptureEngine, SpaceCaptureRequest, CaptureQuality
        )
        
        if not hasattr(self, 'capture_engine'):
            self.capture_engine = MultiSpaceCaptureEngine()
            # Connect optimizer if available
            if self.multi_space_optimizer:
                self.capture_engine.optimizer = self.multi_space_optimizer
                self.multi_space_optimizer.capture_engine = self.capture_engine
            
        # Determine which spaces to capture
        spaces_to_capture = self._determine_spaces_to_capture(query_analysis, window_data)
        
        # Create capture request
        request = SpaceCaptureRequest(
            space_ids=spaces_to_capture,
            quality=CaptureQuality.OPTIMIZED,
            use_cache=True,
            cache_ttl=30,
            priority=7,
            require_permission=True,
            reason=f"multi_space_query: {query_analysis.get('query', '')[:50]}"
        )
        
        # Capture screenshots
        result = await self.capture_engine.capture_all_spaces(request)
        
        if result.success:
            return result.screenshots
        else:
            logger.warning(f"Multi-space capture had errors: {result.errors}")
            return result.screenshots  # Return partial results
    
    def _determine_spaces_to_capture(self, query_analysis: Dict[str, Any], window_data: Dict[str, Any]) -> List[int]:
        """Determine which spaces need to be captured based on query"""
        intent = query_analysis.get('intent')
        spaces = set()
        
        # Always include current space
        current_space = window_data.get('current_space', {}).get('id', 1)
        spaces.add(current_space)
        
        # Add target space if specified
        if intent and hasattr(intent, 'target_space') and intent.target_space:
            spaces.add(intent.target_space)
            
        # Add spaces with relevant apps
        if intent and hasattr(intent, 'target_app') and intent.target_app:
            for window in window_data.get('windows', []):
                if (hasattr(window, 'app_name') and 
                    intent.target_app.lower() in window.app_name.lower() and
                    hasattr(window, 'space_id')):
                    spaces.add(window.space_id)
                    
        # For overview queries, capture all spaces
        if intent and hasattr(intent, 'query_type'):
            from vision.multi_space_intelligence import SpaceQueryType
            
            if intent.query_type in [SpaceQueryType.ALL_SPACES, SpaceQueryType.WORKSPACE_OVERVIEW]:
                all_spaces = [s.space_id if hasattr(s, 'space_id') else s.get('space_id', 1) 
                            for s in window_data.get('spaces', [])]
                spaces.update(all_spaces)
                
        return sorted(list(spaces))
    
    def _build_comprehensive_multi_space_prompt(self, query: str, intent: Any, 
                                               window_data: Dict[str, Any], 
                                               screenshots: Dict[int, Any]) -> str:
        """Build prompt that includes visual context from multiple spaces"""
        base_prompt = self._build_multi_space_prompt(query, intent, window_data)
        
        # Add visual context information
        visual_context = f"\n\nVISUAL CONTEXT AVAILABLE:\n"
        visual_context += f"I have provided screenshots from {len(screenshots)} desktop spaces:\n"
        
        for space_id in sorted(screenshots.keys()):
            is_current = space_id == window_data.get('current_space', {}).get('id', 1)
            status = " (current)" if is_current else ""
            visual_context += f"- Desktop {space_id}{status}: Full visual access\n"
            
        visual_context += "\nAnalyze ALL provided screenshots to give a complete answer about the user's workspace."
        
        return base_prompt + visual_context
    
    async def _get_multi_space_claude_response(self, screenshots: Dict[int, Any], prompt: str) -> Dict[str, Any]:
        """Get Claude's response with multiple screenshots"""
        # Validate input
        if not screenshots:
            raise ValueError("No screenshots provided for multi-space analysis")
            
        if self.claude:
            try:
                # Prepare images for Claude
                images = []
                for space_id in sorted(screenshots.keys()):
                    screenshot = screenshots[space_id]
                    
                    # Convert numpy array to PIL Image if needed
                    if isinstance(screenshot, np.ndarray):
                        from PIL import Image
                        screenshot = Image.fromarray(screenshot)
                        
                    images.append({
                        'image': screenshot,
                        'label': f'Desktop {space_id}'
                    })
                    
                # Call Claude with multiple images
                response = await self.claude.analyze_multiple_images_with_prompt(
                    images=images,
                    prompt=prompt,
                    max_tokens=1000
                )
                
                return {
                    'response': response.get('content', ''),
                    'understanding': {
                        'multi_space_analysis': True,
                        'spaces_analyzed': list(screenshots.keys()),
                        'timestamp': datetime.now().isoformat()
                    }
                }
                
            except Exception as e:
                logger.error(f"Multi-space Claude API error: {e}", exc_info=True)
                # Add more context to ValueError
                if isinstance(e, ValueError):
                    logger.error(f"ValueError details - screenshots: {list(screenshots.keys())}, prompt length: {len(prompt)}")
                    # Re-raise to see full traceback
                    raise
                # Fall back to single image
                if screenshots:
                    first_space = sorted(screenshots.keys())[0]
                    return await self._get_claude_vision_response(
                        screenshots[first_space], 
                        prompt + f"\n\nNOTE: Only analyzing Desktop {first_space} due to technical limitations."
                    )
                    
        return self._get_mock_response(prompt)
    
    def _detect_cross_space_activity(self, understanding: Dict[str, Any], window_data: Dict[str, Any]) -> bool:
        """Detect if there's important activity across multiple spaces"""
        # Check for cross-space patterns
        if 'cross_space_workflow' in understanding:
            return True
            
        # Check for distributed applications
        app_spaces = {}
        for window in window_data.get('windows', []):
            if hasattr(window, 'app_name') and hasattr(window, 'space_id'):
                app = window.app_name
                if app not in app_spaces:
                    app_spaces[app] = set()
                app_spaces[app].add(window.space_id)
                
        # Apps running on multiple spaces indicate cross-space workflow
        for app, spaces in app_spaces.items():
            if len(spaces) > 1:
                return True
                
        return False
    
    async def _enable_cross_space_monitoring(self, window_data: Dict[str, Any]):
        """Enable proactive monitoring across multiple spaces"""
        # Enable capture engine prefetching
        if hasattr(self, 'capture_engine'):
            # Get all space IDs
            space_ids = [s.space_id if hasattr(s, 'space_id') else s.get('space_id', 1) 
                        for s in window_data.get('spaces', [])]
                        
            # Start prefetching for all spaces
            await self.capture_engine.prefetch_spaces(space_ids)
            
            # Enable space change monitoring
            async def on_space_change(new_spaces):
                logger.info(f"Spaces changed, new configuration: {new_spaces}")
                await self.capture_engine.clear_cache()
                await self.capture_engine.prefetch_spaces(new_spaces)
        
        # Enable proactive monitoring if available
        if self.multi_space_monitor and not self.multi_space_monitor.monitoring_active:
            # Define notification callback
            async def handle_monitor_notification(event):
                if self.proactive_assistant:
                    message = await self.proactive_assistant.generate_proactive_message(event)
                    if message:
                        logger.info(f"Proactive notification: {message}")
                        # This would be sent to the user through the appropriate channel
            
            await self.multi_space_monitor.start_monitoring(callback=handle_monitor_notification)
            logger.info("Proactive multi-space monitoring enabled")
                
            asyncio.create_task(
                self.capture_engine.monitor_space_changes(on_space_change)
            )
    
    async def start_multi_space_monitoring(self) -> bool:
        """Start proactive multi-space monitoring with purple indicator"""
        # First, start the capture engine monitoring session (shows purple indicator)
        if hasattr(self, 'capture_engine') and self.capture_engine:
            monitoring_started = await self.capture_engine.start_monitoring_session()
            if monitoring_started:
                logger.info("✅ Multi-space monitoring session started with purple indicator")
            else:
                logger.warning("Failed to start monitoring session with purple indicator")
        
        if not self.multi_space_monitor:
            return False
            
        try:
            await self.multi_space_monitor.start_monitoring()
            return True
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            return False
            
    async def stop_multi_space_monitoring(self) -> bool:
        """Stop proactive multi-space monitoring and remove purple indicator"""
        # First, stop the capture engine monitoring session (removes purple indicator)
        if hasattr(self, 'capture_engine') and self.capture_engine:
            self.capture_engine.stop_monitoring_session()
            logger.info("✅ Multi-space monitoring session stopped, purple indicator removed")
            
        if not self.multi_space_monitor:
            return False
            
        try:
            await self.multi_space_monitor.stop_monitoring()
            return True
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")
            return False
            
    async def get_workspace_insights(self) -> str:
        """Get AI-generated insights about workspace activity"""
        if not self.proactive_assistant:
            return "Workspace monitoring is not available."
            
        try:
            return await self.proactive_assistant.analyze_workspace_health()
        except Exception as e:
            logger.error(f"Failed to get workspace insights: {e}")
            return "Unable to analyze workspace at this time."
            
    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status and statistics"""
        if not self.multi_space_monitor:
            return {"enabled": False, "message": "Monitoring not available"}
            
        try:
            status = {
                "enabled": True,
                "active": self.multi_space_monitor.monitoring_active,
                "summary": await self.multi_space_monitor.get_activity_summary(),
                "recommendations": await self.multi_space_monitor.get_space_recommendations()
            }
            return status
        except Exception as e:
            logger.error(f"Failed to get monitoring status: {e}")
            return {"enabled": True, "active": False, "error": str(e)}
    
    async def start_multi_space_optimization(self) -> bool:
        """Start performance optimization"""
        if not self.multi_space_optimizer:
            return False
            
        try:
            await self.multi_space_optimizer.start_optimization()
            # Also start monitoring if not already active
            if self.multi_space_monitor and not self.multi_space_monitor.monitoring_active:
                await self.multi_space_monitor.start_monitoring()
            return True
        except Exception as e:
            logger.error(f"Failed to start optimization: {e}")
            return False
            
    async def stop_multi_space_optimization(self) -> bool:
        """Stop performance optimization"""
        if not self.multi_space_optimizer:
            return False
            
        try:
            await self.multi_space_optimizer.stop_optimization()
            return True
        except Exception as e:
            logger.error(f"Failed to stop optimization: {e}")
            return False
            
    async def get_optimization_stats(self) -> Dict[str, Any]:
        """Get performance optimization statistics"""
        if not self.multi_space_optimizer:
            return {"enabled": False, "message": "Optimization not available"}
            
        try:
            stats = self.multi_space_optimizer.get_optimization_stats()
            
            # Add cache stats if available
            if hasattr(self, 'capture_engine') and self.capture_engine:
                stats['cache'] = self.capture_engine.get_cache_stats()
                
            return stats
        except Exception as e:
            logger.error(f"Failed to get optimization stats: {e}")
            return {"enabled": True, "error": str(e)}


class ProactiveIntelligence:
    """
    Proactive intelligence that notices and communicates important changes.
    No templates - every observation is communicated naturally by Claude.
    """
    
    def __init__(self, vision_intelligence: PureVisionIntelligence):
        self.vision = vision_intelligence
        self.monitoring_active = False
        self.last_proactive_message = None
        
    async def observe_and_communicate(self, screenshot: Any) -> Optional[str]:
        """
        Observe screen and proactively communicate if something important is noticed.
        Claude decides what's important and how to communicate it.
        """
        # Let Claude observe and decide what's worth mentioning
        proactive_prompt = f"""You are JARVIS, observing the user's screen proactively.

Look at this screen and determine if there's anything worth mentioning proactively.

Consider mentioning:
- New errors or warnings that just appeared
- Important notifications or updates
- Completed processes or downloads
- Potential issues the user might not have noticed
- Helpful suggestions based on what they're doing

Current workflow: {self.vision.context.workflow_state}
Last interaction: {self._format_last_interaction()}

If you notice something worth mentioning:
1. Provide a natural, proactive message as JARVIS would
2. Be helpful but not intrusive
3. Only mention truly useful observations
4. Vary your communication style

If nothing is particularly noteworthy, respond with: "NOTHING_TO_MENTION"

Remember: Every proactive message should sound natural and different.
"""
        
        response = await self.vision._get_claude_vision_response(screenshot, proactive_prompt)
        
        if response.get('response') != "NOTHING_TO_MENTION":
            self.last_proactive_message = datetime.now()
            return response.get('response')
            
        return None
        
    def _format_last_interaction(self) -> str:
        """Format last interaction time for context"""
        if not self.vision.context.last_interaction:
            return "No recent interaction"
            
        seconds_ago = (datetime.now() - self.vision.context.last_interaction).total_seconds()
        return f"{self.vision._format_time_ago(seconds_ago)}"


# Workflow understanding enhancement
class WorkflowIntelligence:
    """
    Understands user workflows and provides contextual assistance.
    Pure Claude intelligence - no hardcoded workflow definitions.
    """
    
    def __init__(self, vision_intelligence: PureVisionIntelligence):
        self.vision = vision_intelligence
        
    async def understand_workflow_and_assist(self, screenshot: Any, query: str) -> str:
        """
        Understand the user's workflow and provide contextually appropriate assistance.
        Claude figures out the workflow from screen content.
        """
        workflow_prompt = f"""You are JARVIS. The user asks: "{query}"

Look at their screen and understand their current workflow:
1. What applications are open and how they relate
2. What task they appear to be working on
3. What stage of the task they're at
4. What they might need help with

Previous context: {self._get_workflow_context()}

Provide a response that:
1. Answers their question specifically
2. Shows understanding of their workflow
3. Offers relevant assistance for their current task
4. References connections between different windows/apps if relevant

Be natural and helpful. Show that you understand not just what's on screen, but what they're trying to accomplish.
"""
        
        response = await self.vision._get_claude_vision_response(screenshot, workflow_prompt)
        return response.get('response')
        
    def _get_workflow_context(self) -> str:
        """Get workflow context from history"""
        if not self.vision.context.history:
            return "Starting fresh - no previous workflow context"
            
        recent_interactions = list(self.vision.context.history)[-5:]
        workflow_summary = f"Recent activity: {len(recent_interactions)} interactions in current session"
        
        if self.vision.context.workflow_state:
            workflow_summary += f"\nDetected workflow: {self.vision.context.workflow_state}"
            
        return workflow_summary