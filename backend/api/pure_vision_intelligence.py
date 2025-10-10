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
    from vision.multi_space_window_detector import (
        MultiSpaceWindowDetector,
        EnhancedWindowInfo,
    )
    from vision.multi_space_intelligence import (
        MultiSpaceIntelligenceExtension,
        SpaceQueryType,
        SpaceQueryIntent,
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

    def add_interaction(
        self, query: str, response: str, screen_understanding: Dict[str, Any]
    ):
        """Record an interaction with full context"""
        self.history.append(
            {
                "timestamp": datetime.now(),
                "query": query,
                "response": response,
                "understanding": screen_understanding,
                "emotional_tone": self.emotional_context,
            }
        )
        self.last_interaction = datetime.now()

    def get_temporal_context(self) -> Dict[str, Any]:
        """Get time-aware context for Claude"""
        if not self.last_interaction:
            return {
                "first_interaction": True,
                "temporal_state": "first_interaction",
                "seconds_since_last": 0,
                "previous_query": None,
                "conversation_length": 0,
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
            "temporal_state": temporal_state,
            "seconds_since_last": time_since_last,
            "previous_query": self.history[-1]["query"] if self.history else None,
            "conversation_length": len(self.history),
        }

    def detect_workflow(self, screen_understanding: Dict[str, Any]) -> str:
        """Detect what workflow the user is in"""
        # This will be enhanced by Claude's understanding
        indicators = screen_understanding.get("workflow_indicators", {})

        if indicators.get("code_editor") and indicators.get("terminal"):
            if indicators.get("error_messages"):
                return "debugging"
            elif indicators.get("test_output"):
                return "testing"
            else:
                return "coding"
        elif indicators.get("browser") and indicators.get("documentation"):
            return "researching"
        elif indicators.get("design_tools"):
            return "designing"
        elif indicators.get("communication_apps"):
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

    def __init__(self, claude_client, enable_multi_space: bool = True, context_store=None, context_bridge=None):
        self.claude = claude_client
        self.context = ConversationContext()
        self.screen_cache = {}  # Hash -> understanding

        # ═══════════════════════════════════════════════════════════════
        # Context Intelligence Bridge Integration
        # ═══════════════════════════════════════════════════════════════
        self.context_bridge = context_bridge  # Access to Context Intelligence (Priority 1-3)

        # ═══════════════════════════════════════════════════════════════
        # Follow-Up Context Tracking
        # ═══════════════════════════════════════════════════════════════
        self.context_store = context_store
        if self.context_store is None:
            try:
                from backend.core.context.memory_store import InMemoryContextStore
                self.context_store = InMemoryContextStore(max_size=50)
                logger.info("Initialized in-memory context store for follow-up tracking")
            except Exception as e:
                logger.warning(f"Could not initialize context store: {e}")
                self.context_store = None

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
                    from vision.multi_space_monitor import (
                        MultiSpaceMonitor,
                        ProactiveAssistant,
                    )

                    self.multi_space_monitor = MultiSpaceMonitor(
                        vision_intelligence=self
                    )
                    self.proactive_assistant = ProactiveAssistant(
                        monitor=self.multi_space_monitor, vision_intelligence=self
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
                    from vision.multi_space_capture_engine import (
                        MultiSpaceCaptureEngine,
                    )

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

    # ═══════════════════════════════════════════════════════════════
    # Follow-Up Context Tracking Methods
    # ═══════════════════════════════════════════════════════════════

    async def track_pending_question(
        self,
        question_text: str,
        window_type: str,
        window_id: str,
        space_id: str,
        snapshot_id: str,
        summary: str,
        ocr_text: str | None = None,
        ttl_seconds: int = 120,
    ) -> str | None:
        """
        Track a pending question after JARVIS asks the user something.
        Returns context ID if successful, None if store unavailable.
        """
        if not self.context_store:
            logger.warning("Cannot track pending question: context store not available")
            return None

        try:
            from backend.core.models.context_envelope import (
                ContextEnvelope,
                ContextMetadata,
                ContextCategory,
                ContextPriority,
                VisionContextPayload,
            )

            # Create vision context envelope
            metadata = ContextMetadata(
                category=ContextCategory.VISION,
                priority=ContextPriority.HIGH,
                source="pure_vision_intelligence",
                tags=(window_type, "pending_question"),
            )

            payload = VisionContextPayload(
                window_type=window_type,
                window_id=window_id,
                space_id=space_id,
                snapshot_id=snapshot_id,
                summary=summary,
                ocr_text=ocr_text,
            )

            envelope = ContextEnvelope(
                metadata=metadata,
                payload=payload,
                ttl_seconds=ttl_seconds,
                decay_rate=0.01,  # 1% per second
            )

            context_id = await self.context_store.add(envelope)
            logger.info(
                f"[FOLLOW-UP] Tracked pending question: '{question_text}' "
                f"(context_id={context_id}, window={window_type})"
            )

            # Telemetry: Track pending context creation
            try:
                from backend.core.telemetry.events import get_telemetry
                telemetry = get_telemetry()
                await telemetry.track_event(
                    "follow_up.pending_created",
                    {
                        "context_id": context_id,
                        "window_type": window_type,
                        "question_text": question_text[:100],
                        "ttl_seconds": ttl_seconds,
                        "has_ocr_text": ocr_text is not None,
                    }
                )
            except:
                pass  # Telemetry is optional

            return context_id

        except Exception as e:
            logger.error(f"Failed to track pending question: {e}", exc_info=True)
            return None

    async def get_active_pending(self):
        """Retrieve the most recent valid pending context."""
        if not self.context_store:
            return None

        try:
            # Clear expired contexts first
            await self.context_store.clear_expired()

            # Get most relevant
            contexts = await self.context_store.get_most_relevant(limit=1)
            return contexts[0] if contexts else None

        except Exception as e:
            logger.error(f"Failed to get active pending context: {e}", exc_info=True)
            return None

    async def clear_all_pending(self):
        """Clear all pending contexts."""
        if self.context_store:
            try:
                await self.context_store.clear_all()
                logger.info("[FOLLOW-UP] Cleared all pending contexts")
            except Exception as e:
                logger.error(f"Failed to clear pending contexts: {e}", exc_info=True)

    async def _track_followup_opportunity(
        self,
        user_query: str,
        jarvis_response: str,
        screenshot: Any,
        understanding: Dict[str, Any]
    ):
        """
        Intelligently detect when JARVIS mentions seeing a terminal/window
        but hasn't provided deep analysis yet, and track it for follow-up.

        This enables conversations like:
        User: "can you see my terminal?"
        JARVIS: "Yes, I can see Terminal on Desktop 2..."
        User: "what does it say?" or "are there any errors?"
        JARVIS: [Actually reads and analyzes the terminal content]
        """
        if not self.context_store:
            return  # No context store, skip tracking

        try:
            response_lower = jarvis_response.lower()
            query_lower = user_query.lower()

            # Detect if JARVIS mentioned seeing something but didn't analyze it deeply
            mentioned_terminal = any(word in response_lower for word in [
                'terminal', 'command line', 'shell', 'bash', 'zsh', 'iterm'
            ])
            mentioned_browser = any(word in response_lower for word in [
                'browser', 'chrome', 'safari', 'firefox', 'webpage', 'tab'
            ])
            mentioned_code = any(word in response_lower for word in [
                'code', 'editor', 'vscode', 'vs code', 'ide', 'file'
            ])
            mentioned_window = 'window' in response_lower or 'desktop' in response_lower

            # Check if user asked "can you see" or similar discovery question
            is_discovery_query = any(phrase in query_lower for phrase in [
                'can you see', 'do you see', 'what do you see',
                'show me', 'what\'s on', 'what\'s in'
            ])

            # Check if response is brief (indicates surface-level observation)
            is_brief_response = len(jarvis_response.split()) < 50

            # If JARVIS mentioned a window/terminal/etc in a brief response to a discovery query,
            # track it as a follow-up opportunity
            should_track = (
                is_discovery_query and
                is_brief_response and
                (mentioned_terminal or mentioned_browser or mentioned_code or mentioned_window)
            )

            if should_track:
                # Determine window type
                if mentioned_terminal:
                    window_type = "terminal"
                    follow_up_question = "Would you like me to read what's in the Terminal and check for any errors or important output?"
                elif mentioned_browser:
                    window_type = "browser"
                    follow_up_question = "Would you like me to read the page content or summarize what's displayed?"
                elif mentioned_code:
                    window_type = "code"
                    follow_up_question = "Would you like me to analyze the code for any issues or errors?"
                else:
                    window_type = "general"
                    follow_up_question = "Would you like me to describe what's in that window in more detail?"

                # Extract space/window info from response if available
                import re
                space_match = re.search(r'desktop\s*(\d+)', response_lower)
                space_id = space_match.group(1) if space_match else "unknown"

                # Generate snapshot ID from screenshot hash
                import hashlib
                snapshot_id = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:12]

                # Track the pending question
                context_id = await self.track_pending_question(
                    question_text=follow_up_question,
                    window_type=window_type,
                    window_id=f"{window_type}_{space_id}",
                    space_id=str(space_id),
                    snapshot_id=snapshot_id,
                    summary=f"User asked '{user_query}', JARVIS responded '{jarvis_response[:100]}'",
                    ocr_text=None,  # Will be extracted on follow-up
                    ttl_seconds=180,  # 3 minutes to ask follow-up
                )

                if context_id:
                    logger.info(
                        f"[FOLLOW-UP] Tracked opportunity for {window_type} deep-dive "
                        f"(context_id={context_id}, space={space_id})"
                    )

        except Exception as e:
            logger.error(f"[FOLLOW-UP] Failed to track followup opportunity: {e}", exc_info=True)

    # ═══════════════════════════════════════════════════════════════

    async def understand_and_respond(self, screenshot: Any, user_query: str) -> str:
        """
        Core method: Claude sees, understands, and responds naturally.
        No templates. No hardcoding. Pure intelligence.
        Now with optional multi-space awareness for workspace-wide queries.
        ENHANCED: Integrates with Context Intelligence Bridge for structured context.

        Args:
            screenshot: Can be a single image OR Dict[int, image] for multi-space
            user_query: The user's question
        """
        # Check if we already have multi-space screenshots
        if isinstance(screenshot, dict) and len(screenshot) > 1:
            # We already have multi-space screenshots from vision_command_handler
            return await self._analyze_multi_space_screenshots(screenshot, user_query)

        # Check if query needs multi-space handling (but we don't have screenshots yet)
        if self.multi_space_enabled and self._should_use_multi_space(user_query):
            return await self._multi_space_understand_and_respond(
                screenshot, user_query
            )

        # Original single-space logic
        # ENHANCED: Get structured context from Context Intelligence Bridge
        structured_context = await self._get_structured_context(user_query)

        # Generate rich context for Claude
        context_prompt = self._build_pure_intelligence_prompt(user_query, structured_context)

        # Let Claude see and respond naturally
        claude_response = await self._get_claude_vision_response(
            screenshot, context_prompt
        )

        # Extract understanding and response
        understanding = self._extract_understanding(claude_response)
        natural_response = claude_response.get("response", "")

        # Update context for future interactions
        self.context.add_interaction(user_query, natural_response, understanding)

        # Detect workflow changes
        self.context.workflow_state = self.context.detect_workflow(understanding)

        # ═══════════════════════════════════════════════════════════════
        # Feed understanding back to Context Intelligence Bridge
        # This updates the context graph with what Claude observed
        # ═══════════════════════════════════════════════════════════════
        await self._update_context_bridge(understanding, user_query, natural_response)

        # ═══════════════════════════════════════════════════════════════
        # Track pending follow-up questions when JARVIS mentions seeing
        # a terminal/window but hasn't provided detailed analysis yet
        # ═══════════════════════════════════════════════════════════════
        await self._track_followup_opportunity(
            user_query, natural_response, screenshot, understanding
        )

        return natural_response

    async def _update_context_bridge(self, understanding: Dict[str, Any], user_query: str, response: str = ""):
        """
        Feed Claude's understanding back to Context Intelligence Bridge.
        This creates a feedback loop: Vision sees → Context stores → Vision uses context.
        Also saves conversational context for multi-turn conversations.
        """
        if not self.context_bridge:
            return

        try:
            # ═══════════════════════════════════════════════════════════════
            # Save conversational context for follow-up queries
            # ═══════════════════════════════════════════════════════════════
            if response and ("would you like" in response.lower() or "can i help" in response.lower()):
                # JARVIS asked a follow-up question, save context for continuation
                self.context_bridge._save_conversation_context(
                    query=user_query,
                    response=response,
                    current_space_id=understanding.get("current_space_id")
                )
                logger.info("[VISION→CONTEXT] Saved conversational context for follow-up")

            # ═══════════════════════════════════════════════════════════════
            # Extract what Claude observed and update context graph
            # ═══════════════════════════════════════════════════════════════
            observed_apps = understanding.get("observed_applications", [])
            screen_text = understanding.get("screen_text", "")
            detected_errors = understanding.get("detected_errors", [])
            current_space = understanding.get("current_space_id", 1)

            # If Claude saw a terminal, extract terminal-specific data
            for app in observed_apps:
                app_name = app.get("name", "")
                app_type = app.get("type", "").lower()

                if "terminal" in app_name.lower() or app_type == "terminal":
                    # Extract terminal context from what Claude saw
                    terminal_text = app.get("content", screen_text)

                    # Use TerminalCommandIntelligence to analyze the text
                    if self.context_bridge.terminal_intelligence:
                        term_ctx = await self.context_bridge.terminal_intelligence.analyze_terminal_context(terminal_text)

                        # Update the context graph with terminal data
                        space_id = app.get("space_id", current_space)
                        self.context_bridge.context_graph.update_terminal_context(
                            space_id=space_id,
                            app_name=app_name,
                            command=term_ctx.last_command,
                            output=term_ctx.command_output,
                            errors=term_ctx.errors,
                            working_dir=term_ctx.current_directory
                        )

                        logger.info(f"[VISION→CONTEXT] Updated terminal context for {app_name} in Space {space_id}")

            # Update based on detected errors (even if not from terminal)
            if detected_errors:
                for error in detected_errors:
                    error_text = error if isinstance(error, str) else error.get("text", "")
                    app_name = error.get("app", "Unknown") if isinstance(error, dict) else "Unknown"

                    # Try to update terminal context if error looks terminal-related
                    if any(keyword in error_text.lower() for keyword in ["error:", "exception:", "traceback", "failed"]):
                        self.context_bridge.context_graph.update_terminal_context(
                            space_id=current_space,
                            app_name=app_name,
                            errors=[error_text]
                        )

        except Exception as e:
            logger.error(f"[VISION→CONTEXT] Error updating context bridge: {e}", exc_info=True)

    async def _get_structured_context(self, user_query: str) -> Optional[Dict[str, Any]]:
        """
        Get structured context from Context Intelligence Bridge.
        Combines multi-space tracking, terminal intelligence, and cross-space correlation.
        """
        if not self.context_bridge:
            return None

        try:
            # Get comprehensive workspace summary
            summary = self.context_bridge.context_graph.get_summary()

            # Extract relevant context
            structured = {
                "total_spaces": summary.get("total_spaces", 0),
                "active_spaces": summary.get("active_spaces", []),
                "applications": {},
                "errors": [],
                "recent_commands": []
            }

            # Gather context from all spaces
            for space_id, space_data in summary.get("spaces", {}).items():
                apps = space_data.get("applications", {})
                for app_name, app_data in apps.items():
                    if app_data.get("activity_count", 0) > 0:
                        structured["applications"][f"{app_name} (Space {space_id})"] = {
                            "type": app_data.get("context_type"),
                            "last_activity": app_data.get("last_activity"),
                            "significance": app_data.get("significance")
                        }

                        # Extract terminal errors and commands
                        if app_data.get("context_type") == "terminal":
                            # Try to get actual terminal context
                            space = self.context_bridge.context_graph.spaces.get(space_id)
                            if space:
                                app_ctx = space.applications.get(app_name)
                                if app_ctx and app_ctx.terminal_context:
                                    term_ctx = app_ctx.terminal_context
                                    if term_ctx.errors:
                                        for error in term_ctx.errors:
                                            structured["errors"].append({
                                                "space": space_id,
                                                "app": app_name,
                                                "error": error,
                                                "command": term_ctx.last_command
                                            })
                                    if term_ctx.last_command:
                                        structured["recent_commands"].append({
                                            "space": space_id,
                                            "app": app_name,
                                            "command": term_ctx.last_command,
                                            "directory": term_ctx.working_directory
                                        })

            return structured if (structured["applications"] or structured["errors"]) else None

        except Exception as e:
            logger.error(f"[VISION-INTEL] Error getting structured context: {e}")
            return None

    def _build_pure_intelligence_prompt(self, user_query: str, structured_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a rich, contextual prompt for Claude that enables natural responses.
        This is the ONLY place we guide Claude - no response templates!
        ENHANCED: Now includes structured context from Context Intelligence Bridge.
        """
        temporal_context = self.context.get_temporal_context()

        # Build conversation history context
        history_context = ""
        if self.context.history:
            recent = list(self.context.history)[-3:]  # Last 3 interactions
            history_context = "\n".join(
                [
                    f"Earlier: User asked '{h['query']}' and you responded '{h['response']}'"
                    for h in recent
                ]
            )

        # Determine appropriate emotional tone based on context
        emotional_guidance = self._determine_emotional_tone(
            user_query, temporal_context
        )

        # Build structured context section if available
        structured_section = ""
        if structured_context:
            structured_section = "\n\n═══ WORKSPACE INTELLIGENCE (from Context System) ═══\n"

            if structured_context.get("errors"):
                structured_section += "\n🔴 DETECTED ERRORS:\n"
                for err in structured_context["errors"]:
                    structured_section += f"  • {err['app']} (Space {err['space']}): {err['error']}\n"
                    if err.get('command'):
                        structured_section += f"    Command: {err['command']}\n"

            if structured_context.get("recent_commands"):
                structured_section += "\n💻 RECENT TERMINAL ACTIVITY:\n"
                for cmd in structured_context["recent_commands"][-3:]:
                    structured_section += f"  • Space {cmd['space']}: `{cmd['command']}`"
                    if cmd.get('directory'):
                        structured_section += f" in {cmd['directory']}"
                    structured_section += "\n"

            if structured_context.get("applications"):
                structured_section += f"\n📱 ACTIVE APPLICATIONS ({len(structured_context['applications'])} windows):\n"
                for app_name, app_info in list(structured_context["applications"].items())[:5]:
                    structured_section += f"  • {app_name}: {app_info['type']}\n"

            structured_section += "\nUse this context to enhance your visual analysis!\n═══════════════════════════════════════════════════\n"

        # Build the prompt that enables natural intelligence
        prompt = f"""You are JARVIS, Tony Stark's AI assistant. You're looking at the user's screen.

User's Current Question: "{user_query}"

{f"Conversation Context: {history_context}" if history_context else "This is the start of our conversation."}

Temporal Context: {temporal_context['temporal_state']}
{f"Previous query was: '{temporal_context['previous_query']}'" if temporal_context.get('previous_query') else ""}

Current Workflow: {self.context.workflow_state or 'unknown'}
{structured_section}
Instructions for Natural Response:
1. Look at the screen and understand what you see
2. COMBINE visual analysis with the structured workspace context above
3. **If you see a Terminal window, READ what's on the screen:**
   - Look for error messages (lines starting with "Error:", "Traceback", etc.)
   - Note the last command that was run (look for $ or % prompts)
   - Identify the working directory if visible
   - Be SPECIFIC about what you see - quote the exact error text
4. If errors are detected (in structured context OR visible on screen), mention them proactively
5. Answer ONLY what was asked - be concise and direct
6. Use exact values when visible but keep context minimal
7. Address the user as "Sir" naturally
8. {emotional_guidance}
9. Keep response to 1-2 sentences unless more detail specifically requested
10. Optionally add ONE brief, helpful insight if truly relevant
11. Never describe the entire screen unless asked

IMPORTANT: Be conversational but CONCISE. Focus on answering the specific question.
Example Good Response: "Yes, I can see Terminal in Space 2, Sir. I notice there's a ModuleNotFoundError when running `python app.py`. Would you like me to explain?"
Example Bad Response: "Yes, I can see your terminal is open on Desktop 2."

Remember: Natural, brief, directly answering what was asked, and SPECIFIC about errors/commands you see.
"""
        return prompt

    def _determine_emotional_tone(
        self, query: str, temporal_context: Dict[str, Any]
    ) -> str:
        """Determine appropriate emotional tone for natural variation"""
        query_lower = query.lower()

        # Urgent situations
        if any(word in query_lower for word in ["error", "crash", "failed", "broken"]):
            self.context.emotional_context = EmotionalTone.URGENT
            return "Be helpful and focused on solving their problem"

        # Celebratory moments
        elif any(
            word in query_lower for word in ["finally", "working", "success", "passed"]
        ):
            self.context.emotional_context = EmotionalTone.CELEBRATORY
            return "Share in their success with appropriate enthusiasm"

        # Concerned situations
        elif temporal_context.get("conversation_length", 0) > 10:
            self.context.emotional_context = EmotionalTone.CONCERNED
            return "Show concern for their extended work session"

        # Helpful by default
        else:
            self.context.emotional_context = EmotionalTone.HELPFUL
            return "Be naturally helpful and attentive"

    async def _get_claude_vision_response(
        self, screenshot: Any, prompt: str
    ) -> Dict[str, Any]:
        """Get pure, natural response from Claude Vision"""
        if self.claude:
            try:
                # Call Claude Vision API with screenshot and prompt
                response = await self.claude.analyze_image_with_prompt(
                    image=screenshot, prompt=prompt, max_tokens=500
                )

                # Parse Claude's response to extract the natural language and understanding
                # Claude returns text, we need to structure it
                response_text = response.get("content", "")

                # For structured responses, we can ask Claude to return JSON-like sections
                # But for now, treat the whole response as natural language
                return {
                    "response": response_text,
                    "understanding": {
                        "raw_analysis": response_text,
                        "timestamp": datetime.now().isoformat(),
                    },
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
                "Yes, I have visual access to your display.",
            ]
            response = random.choice(responses)
        else:
            response = "I'm analyzing your screen to help with that request."

        return {
            "response": response,
            "understanding": {
                "mock_mode": True,
                "timestamp": datetime.now().isoformat(),
            },
        }

    def _extract_understanding(self, claude_response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Claude's understanding for context tracking"""
        return claude_response.get("understanding", {})

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

    async def _analyze_multi_space_screenshots(
        self, screenshots: Dict[int, Any], user_query: str
    ) -> str:
        """
        Analyze already-captured multi-space screenshots
        This is called when vision_command_handler has already captured multiple spaces
        """
        logger.info(f"Analyzing {len(screenshots)} pre-captured space screenshots")

        # Get window data for context
        window_data = await self._gather_multi_space_data()

        # Analyze query intent
        query_analysis = self.multi_space_extension.process_multi_space_query(
            user_query, window_data
        )

        # Build comprehensive prompt with all screenshots
        enhanced_prompt = self._build_comprehensive_multi_space_prompt(
            user_query, query_analysis["intent"], window_data, screenshots
        )

        # Analyze with Claude
        claude_response = await self._get_multi_space_claude_response(
            screenshots, enhanced_prompt
        )

        # Extract and process response
        understanding = self._extract_understanding(claude_response)
        natural_response = claude_response.get("response", "")

        # Update context
        self.context.add_interaction(user_query, natural_response, understanding)

        return natural_response

    async def _multi_space_understand_and_respond(
        self, screenshot: Any, user_query: str
    ) -> str:
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
        needs_multi_capture = self._needs_multi_space_capture(
            query_analysis, window_data
        )

        if needs_multi_capture:
            # 4. Capture screenshots from relevant spaces
            logger.info(
                f"Attempting to capture multi-space screenshots for query: {user_query}"
            )
            multi_screenshots = await self._capture_multi_space_screenshots(
                query_analysis, window_data
            )
            logger.info(
                f"Captured {len(multi_screenshots)} screenshots from spaces: {list(multi_screenshots.keys())}"
            )

            # 5. Build comprehensive prompt with visual context from all spaces
            enhanced_prompt = self._build_comprehensive_multi_space_prompt(
                user_query, query_analysis["intent"], window_data, multi_screenshots
            )

            # Create multi-space context for debugging
            context = MultiSpaceContext(
                current_space_id=window_data.get("current_space", {}).get("id", 1),
                total_spaces=len(window_data.get("spaces", [])),
                analyzed_spaces=len(multi_screenshots),
                total_windows=len(window_data.get("windows", [])),
                query_type=query_analysis.get("type", "unknown"),
                space_summaries={},
                last_space_update=datetime.now(),
            )

            # Populate space summaries
            for space_info in window_data.get("spaces", []):
                space_id = (
                    space_info.space_id
                    if hasattr(space_info, "space_id")
                    else space_info.get("space_id")
                )
                if space_id:
                    space_windows = [
                        w
                        for w in window_data.get("windows", [])
                        if (hasattr(w, "space_id") and w.space_id == space_id)
                        or (isinstance(w, dict) and w.get("space_id") == space_id)
                    ]
                    context.space_summaries[space_id] = {
                        "window_count": len(space_windows),
                        "has_screenshot": space_id in multi_screenshots,
                        "apps": list(
                            set(
                                (
                                    w.app_name
                                    if hasattr(w, "app_name")
                                    else w.get("app_name", "Unknown")
                                )
                                for w in space_windows
                            )
                        )[:5],
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
                logger.warning(
                    "No multi-space screenshots captured, providing helpful response"
                )
                # Check if it's a permission issue
                if hasattr(self, "capture_engine") and self.capture_engine:
                    return (
                        "I'm trying to look across your desktop spaces to find Terminal, but I'm unable to capture screenshots at the moment. "
                        "Please ensure screen recording permissions are enabled for JARVIS in "
                        "System Preferences > Security & Privacy > Privacy > Screen Recording. "
                        "Once enabled, I'll be able to help you locate applications across all your desktop spaces."
                    )
                else:
                    return (
                        "I understand you're looking for Terminal across your desktop spaces. "
                        "To help with this, I need to set up multi-space vision capabilities. "
                        "Please try asking again in a moment."
                    )
        else:
            # Use standard single-space analysis
            enhanced_prompt = self._build_multi_space_prompt(
                user_query, query_analysis["intent"], window_data
            )

            claude_response = await self._get_claude_vision_response(
                screenshot, enhanced_prompt
            )

        # 7. Extract and process response
        understanding = self._extract_understanding(claude_response)
        natural_response = claude_response.get("response", "")

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
        window_data["cached_screenshots"] = {}
        window_data["screenshot_confidence"] = {}

        for space_id in window_data.get("space_window_map", {}).keys():
            cached = self.screenshot_cache.get_screenshot(space_id)
            if cached:
                window_data["cached_screenshots"][space_id] = cached.screenshot
                window_data["screenshot_confidence"][
                    space_id
                ] = cached.confidence_level().value

        # Mark current screenshot availability
        window_data["has_current_screenshot"] = True  # We always have current space

        return window_data

    async def _enhance_window_data(
        self, window_data: Dict[str, Any], intent: Any
    ) -> Dict[str, Any]:
        """Try to enhance window data based on query needs"""
        # If specific space is targeted and we don't have fresh data
        if (
            hasattr(intent, "target_space")
            and intent.target_space
            and intent.requires_screenshot
        ):
            confidence = window_data["screenshot_confidence"].get(
                intent.target_space, "none"
            )

            if confidence in ["stale", "outdated", "none"]:
                # Consider requesting fresh screenshot
                logger.info(
                    f"Would need fresh screenshot for space {intent.target_space}"
                )

        return window_data

    def _build_multi_space_prompt(
        self, user_query: str, intent: Any, window_data: Dict[str, Any]
    ) -> str:
        """Build enhanced prompt with multi-space awareness"""
        # Get base prompt structure
        temporal_context = self.context.get_temporal_context()
        emotional_guidance = self._determine_emotional_tone(
            user_query, temporal_context
        )

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
        current_space = window_data["current_space"]
        spaces = window_data.get("spaces", [])

        descriptions = []

        # Overall summary
        descriptions.append(f"You have {len(spaces)} desktop spaces active")

        # Per-space summaries
        for space in spaces:
            # Handle both SpaceInfo objects and dictionaries
            if hasattr(space, "space_id"):
                space_id = space.space_id
                window_count = space.window_count
                is_current = space.is_current
            else:
                space_id = space["space_id"]
                window_count = space["window_count"]
                is_current = space["is_current"]

            # Get primary apps on this space
            space_windows = [
                w
                for w in window_data.get("windows", [])
                if hasattr(w, "space_id") and w.space_id == space_id
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

    def _build_window_details(self, window_data: Dict[str, Any], intent: Any) -> str:
        """Build relevant window details based on query intent"""
        windows = window_data.get("windows", [])

        # Filter to relevant windows
        if hasattr(intent, "target_app") and intent.target_app:
            relevant_windows = [
                w
                for w in windows
                if hasattr(w, "app_name")
                and intent.target_app.lower() in w.app_name.lower()
            ]
        elif hasattr(intent, "target_space") and intent.target_space:
            relevant_windows = [
                w
                for w in windows
                if hasattr(w, "space_id") and w.space_id == intent.target_space
            ]
        else:
            # Show summary of all spaces
            relevant_windows = windows

        # Format window information
        details = []
        for window in relevant_windows[:10]:  # Limit to avoid prompt overflow
            if hasattr(window, "app_name"):
                detail = f"- {window.app_name}"
                if hasattr(window, "window_title") and window.window_title:
                    detail += f': "{window.window_title}"'
                if hasattr(window, "space_id"):
                    detail += f" (Desktop {window.space_id})"

                if hasattr(window, "is_fullscreen") and window.is_fullscreen:
                    detail += " [fullscreen]"
                elif hasattr(window, "is_minimized") and window.is_minimized:
                    detail += " [minimized]"

                details.append(detail)

        return "\n".join(details) if details else "No matching windows found"

    def _build_confidence_notes(self, window_data: Dict[str, Any], intent: Any) -> str:
        """Build notes about data confidence"""
        notes = []

        # Check screenshot availability
        if hasattr(intent, "target_space") and intent.target_space:
            confidence = window_data["screenshot_confidence"].get(
                intent.target_space, "none"
            )
            if confidence == "fresh":
                notes.append("Visual confirmation: Live view available")
            elif confidence in ["recent", "usable"]:
                notes.append("Visual reference: Recent capture available")
            else:
                notes.append("Information source: Window metadata")

        # Note about current space
        current_id = window_data["current_space"]["id"]
        notes.append(f"Direct visibility: Desktop {current_id} (current)")

        return "\n".join(notes)

    def _describe_query_intent(self, intent: Any) -> str:
        """Natural description of what the user is asking"""
        if hasattr(intent, "query_type"):
            descriptions = {
                SpaceQueryType.SIMPLE_PRESENCE: "Checking if an application is open",
                SpaceQueryType.LOCATION_QUERY: "Finding where an application is located",
                SpaceQueryType.SPACE_CONTENT: "Examining specific desktop content",
                SpaceQueryType.ALL_SPACES: "Overview of all workspaces",
                SpaceQueryType.SPECIFIC_DETAIL: "Reading specific content",
                SpaceQueryType.WORKSPACE_OVERVIEW: "Understanding current work context",
            }

            base = descriptions.get(intent.query_type, "General workspace query")

            if hasattr(intent, "target_app") and intent.target_app:
                base += f" (looking for {intent.target_app})"
            if hasattr(intent, "target_space") and intent.target_space:
                base += f" (focusing on Desktop {intent.target_space})"

            return base

        return "General workspace query"

    def _update_multi_space_context(self, window_data: Dict[str, Any]):
        """Update multi-space context with latest information"""
        if not self.multi_space_enabled:
            return

        self.multi_space_context.current_space_id = window_data["current_space"]["id"]
        self.multi_space_context.total_spaces = len(window_data.get("spaces", []))
        self.multi_space_context.last_space_update = datetime.now()

        # Update space summaries
        for space in window_data.get("spaces", []):
            # Handle both SpaceInfo objects and dictionaries
            if hasattr(space, "space_id"):
                space_id = space.space_id
                window_count = space.window_count
            else:
                space_id = space["space_id"]
                window_count = space["window_count"]

            self.multi_space_context.space_summaries[space_id] = {
                "window_count": window_count,
                "has_screenshot": space_id in window_data.get("cached_screenshots", {}),
                "last_updated": datetime.now(),
            }

    def get_multi_space_summary(self) -> Dict[str, Any]:
        """Get summary of multi-space system state"""
        if not self.multi_space_enabled:
            return {
                "multi_space_enabled": False,
                "reason": "Multi-space components not available or disabled",
            }

        cache_stats = self.screenshot_cache.get_cache_statistics()

        return {
            "multi_space_enabled": True,
            "current_space": self.multi_space_context.current_space_id,
            "total_spaces": self.multi_space_context.total_spaces,
            "space_summaries": self.multi_space_context.space_summaries,
            "cache_statistics": cache_stats,
            "last_update": self.multi_space_context.last_space_update,
        }

    def _needs_multi_space_capture(
        self, query_analysis: Dict[str, Any], window_data: Dict[str, Any]
    ) -> bool:
        """Determine if we need to capture screenshots from multiple spaces"""
        intent = query_analysis.get("intent")

        if not intent:
            return False

        # Check query type requirements
        if hasattr(intent, "query_type"):
            from vision.multi_space_intelligence import SpaceQueryType

            # These query types require visual data from multiple spaces
            visual_query_types = {
                SpaceQueryType.SPECIFIC_DETAIL,
                SpaceQueryType.ALL_SPACES,
                SpaceQueryType.WORKSPACE_OVERVIEW,
            }

            if intent.query_type in visual_query_types:
                return True

        # Check if query explicitly asks for visual details
        visual_keywords = [
            "show",
            "see",
            "look",
            "display",
            "visual",
            "screenshot",
            "capture",
        ]
        query_lower = query_analysis.get("query", "").lower()

        if any(keyword in query_lower for keyword in visual_keywords):
            return True

        # Check if we need cross-space visual verification
        if intent and hasattr(intent, "target_app") and intent.target_app:
            # App is on different space than current
            current_space = window_data.get("current_space", {}).get("id", 1)

            for window in window_data.get("windows", []):
                if (
                    hasattr(window, "app_name")
                    and intent.target_app.lower() in window.app_name.lower()
                    and hasattr(window, "space_id")
                    and window.space_id != current_space
                ):
                    return True

        return False

    async def _capture_multi_space_screenshots(
        self, query_analysis: Dict[str, Any], window_data: Dict[str, Any]
    ) -> Dict[int, Any]:
        """Capture screenshots from relevant spaces using the Multi-Space Capture Engine"""
        from vision.multi_space_capture_engine import (
            MultiSpaceCaptureEngine,
            SpaceCaptureRequest,
            CaptureQuality,
        )

        if not hasattr(self, "capture_engine"):
            self.capture_engine = MultiSpaceCaptureEngine()
            # Connect optimizer if available
            if self.multi_space_optimizer:
                self.capture_engine.optimizer = self.multi_space_optimizer
                self.multi_space_optimizer.capture_engine = self.capture_engine

        # Determine which spaces to capture
        spaces_to_capture = self._determine_spaces_to_capture(
            query_analysis, window_data
        )

        # Create capture request
        request = SpaceCaptureRequest(
            space_ids=spaces_to_capture,
            quality=CaptureQuality.OPTIMIZED,
            use_cache=True,
            cache_ttl=30,
            priority=7,
            require_permission=True,
            reason=f"multi_space_query: {query_analysis.get('query', '')[:50]}",
        )

        # Capture screenshots
        result = await self.capture_engine.capture_all_spaces(request)

        if result.success:
            return result.screenshots
        else:
            logger.warning(f"Multi-space capture had errors: {result.errors}")
            return result.screenshots  # Return partial results

    def _determine_spaces_to_capture(
        self, query_analysis: Dict[str, Any], window_data: Dict[str, Any]
    ) -> List[int]:
        """Determine which spaces need to be captured based on query"""
        intent = query_analysis.get("intent")
        spaces = set()

        # Always include current space
        current_space = window_data.get("current_space", {}).get("id", 1)
        spaces.add(current_space)

        # Add target space if specified
        if intent and hasattr(intent, "target_space") and intent.target_space:
            spaces.add(intent.target_space)

        # Add spaces with relevant apps
        if intent and hasattr(intent, "target_app") and intent.target_app:
            for window in window_data.get("windows", []):
                if (
                    hasattr(window, "app_name")
                    and intent.target_app.lower() in window.app_name.lower()
                    and hasattr(window, "space_id")
                ):
                    spaces.add(window.space_id)

        # For overview queries, capture all spaces
        if intent and hasattr(intent, "query_type"):
            from vision.multi_space_intelligence import SpaceQueryType

            if intent.query_type in [
                SpaceQueryType.ALL_SPACES,
                SpaceQueryType.WORKSPACE_OVERVIEW,
            ]:
                all_spaces = [
                    s.space_id if hasattr(s, "space_id") else s.get("space_id", 1)
                    for s in window_data.get("spaces", [])
                ]
                spaces.update(all_spaces)

        return sorted(list(spaces))

    def _build_comprehensive_multi_space_prompt(
        self,
        query: str,
        intent: Any,
        window_data: Dict[str, Any],
        screenshots: Dict[int, Any],
    ) -> str:
        """Build prompt that includes visual context from multiple spaces"""
        base_prompt = self._build_multi_space_prompt(query, intent, window_data)

        # Add visual context information
        visual_context = f"\n\nVISUAL CONTEXT AVAILABLE:\n"
        visual_context += (
            f"I have provided screenshots from {len(screenshots)} desktop spaces:\n"
        )

        for space_id in sorted(screenshots.keys()):
            is_current = space_id == window_data.get("current_space", {}).get("id", 1)
            status = " (current)" if is_current else ""
            visual_context += f"- Desktop {space_id}{status}: Full visual access\n"

        visual_context += "\nAnalyze ALL provided screenshots to give a complete answer about the user's workspace."

        return base_prompt + visual_context

    async def _get_multi_space_claude_response(
        self, screenshots: Dict[int, Any], prompt: str
    ) -> Dict[str, Any]:
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

                    images.append({"image": screenshot, "label": f"Desktop {space_id}"})

                # Call Claude with multiple images
                response = await self.claude.analyze_multiple_images_with_prompt(
                    images=images, prompt=prompt, max_tokens=1000
                )

                return {
                    "response": response.get("content", ""),
                    "understanding": {
                        "multi_space_analysis": True,
                        "spaces_analyzed": list(screenshots.keys()),
                        "timestamp": datetime.now().isoformat(),
                    },
                }

            except Exception as e:
                logger.error(f"Multi-space Claude API error: {e}", exc_info=True)
                # Add more context to ValueError
                if isinstance(e, ValueError):
                    logger.error(
                        f"ValueError details - screenshots: {list(screenshots.keys())}, prompt length: {len(prompt)}"
                    )
                    # Re-raise to see full traceback
                    raise
                # Fall back to single image
                if screenshots:
                    first_space = sorted(screenshots.keys())[0]
                    return await self._get_claude_vision_response(
                        screenshots[first_space],
                        prompt
                        + f"\n\nNOTE: Only analyzing Desktop {first_space} due to technical limitations.",
                    )

        return self._get_mock_response(prompt)

    def _detect_cross_space_activity(
        self, understanding: Dict[str, Any], window_data: Dict[str, Any]
    ) -> bool:
        """Detect if there's important activity across multiple spaces"""
        # Check for cross-space patterns
        if "cross_space_workflow" in understanding:
            return True

        # Check for distributed applications
        app_spaces = {}
        for window in window_data.get("windows", []):
            if hasattr(window, "app_name") and hasattr(window, "space_id"):
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
        if hasattr(self, "capture_engine"):
            # Get all space IDs
            space_ids = [
                s.space_id if hasattr(s, "space_id") else s.get("space_id", 1)
                for s in window_data.get("spaces", [])
            ]

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
                    message = await self.proactive_assistant.generate_proactive_message(
                        event
                    )
                    if message:
                        logger.info(f"Proactive notification: {message}")
                        # This would be sent to the user through the appropriate channel

            await self.multi_space_monitor.start_monitoring(
                callback=handle_monitor_notification
            )
            logger.info("Proactive multi-space monitoring enabled")

            asyncio.create_task(
                self.capture_engine.monitor_space_changes(on_space_change)
            )

    async def start_multi_space_monitoring(self) -> bool:
        """Start proactive multi-space monitoring with purple indicator"""
        # First, start the capture engine monitoring session (shows purple indicator)
        if hasattr(self, "capture_engine") and self.capture_engine:
            monitoring_started = await self.capture_engine.start_monitoring_session()
            if monitoring_started:
                logger.info(
                    "✅ Multi-space monitoring session started with purple indicator"
                )
            else:
                logger.warning(
                    "Failed to start monitoring session with purple indicator"
                )

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
        if hasattr(self, "capture_engine") and self.capture_engine:
            self.capture_engine.stop_monitoring_session()
            logger.info(
                "✅ Multi-space monitoring session stopped, purple indicator removed"
            )

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
                "recommendations": await self.multi_space_monitor.get_space_recommendations(),
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
            if (
                self.multi_space_monitor
                and not self.multi_space_monitor.monitoring_active
            ):
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
            if hasattr(self, "capture_engine") and self.capture_engine:
                stats["cache"] = self.capture_engine.get_cache_stats()

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

        response = await self.vision._get_claude_vision_response(
            screenshot, proactive_prompt
        )

        if response.get("response") != "NOTHING_TO_MENTION":
            self.last_proactive_message = datetime.now()
            return response.get("response")

        return None

    def _format_last_interaction(self) -> str:
        """Format last interaction time for context"""
        if not self.vision.context.last_interaction:
            return "No recent interaction"

        seconds_ago = (
            datetime.now() - self.vision.context.last_interaction
        ).total_seconds()
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

        response = await self.vision._get_claude_vision_response(
            screenshot, workflow_prompt
        )
        return response.get("response")

    def _get_workflow_context(self) -> str:
        """Get workflow context from history"""
        if not self.vision.context.history:
            return "Starting fresh - no previous workflow context"

        recent_interactions = list(self.vision.context.history)[-5:]
        workflow_summary = f"Recent activity: {len(recent_interactions)} interactions in current session"

        if self.vision.context.workflow_state:
            workflow_summary += (
                f"\nDetected workflow: {self.vision.context.workflow_state}"
            )

        return workflow_summary
