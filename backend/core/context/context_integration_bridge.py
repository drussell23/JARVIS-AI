"""
Context Integration Bridge - Connects Multi-Space Context Graph with Existing Systems
======================================================================================

This bridge integrates the new MultiSpaceContextGraph with:
1. MultiSpaceMonitor (vision/multi_space_monitor.py) - For space/app detection
2. TerminalCommandIntelligence (vision/handlers/terminal_command_intelligence.py) - For terminal analysis
3. FeedbackLearningLoop (core/learning/feedback_loop.py) - For adaptive notifications
4. ContextStore (core/context/memory_store.py) - For persistence
5. ProactiveVisionIntelligence (vision/proactive_vision_intelligence.py) - For OCR analysis

Architecture:

    Vision Systems → ContextIntegrationBridge → MultiSpaceContextGraph
         ↓                      ↓                        ↓
    MultiSpaceMonitor    Event Translation      Rich Context Storage
    OCR Analysis         Context Enrichment     Cross-Space Correlation
    Terminal Intel       Automatic Detection    Temporal Decay

The bridge acts as an adapter, translating events from existing systems
into rich context updates for the graph.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# INTEGRATION BRIDGE - Main Coordinator
# ============================================================================

class ContextIntegrationBridge:
    """
    Bridges existing JARVIS systems with the new multi-space context graph.

    This is the glue that makes everything work together seamlessly.
    """

    def __init__(self, context_graph, multi_space_monitor=None, terminal_intelligence=None, feedback_loop=None, implicit_resolver=None, cross_space_intelligence=None):
        """
        Initialize the integration bridge.

        Args:
            context_graph: MultiSpaceContextGraph instance
            multi_space_monitor: Optional MultiSpaceMonitor instance
            terminal_intelligence: Optional TerminalCommandIntelligence instance
            feedback_loop: Optional FeedbackLearningLoop instance
            implicit_resolver: Optional ImplicitReferenceResolver instance
            cross_space_intelligence: Optional CrossSpaceIntelligence instance
        """
        self.context_graph = context_graph
        self.multi_space_monitor = multi_space_monitor
        self.terminal_intelligence = terminal_intelligence
        self.feedback_loop = feedback_loop
        self.implicit_resolver = implicit_resolver
        self.cross_space_intelligence = cross_space_intelligence

        # State
        self.is_running = False
        self._monitoring_task: Optional[asyncio.Task] = None

        # Configuration
        self.ocr_analysis_enabled = True
        self.auto_detect_app_types = True

        # Conversational context tracking for follow-up queries
        self._last_query = None
        self._last_response = None
        self._last_context = {}  # Store what we talked about (apps, spaces, errors)
        self._conversation_timestamp = None

        logger.info("[INTEGRATION-BRIDGE] Initialized")

    async def start(self):
        """Start the integration bridge and all connected systems"""
        if self.is_running:
            logger.warning("[INTEGRATION-BRIDGE] Already running")
            return

        self.is_running = True

        # Start context graph
        await self.context_graph.start()

        # Start multi-space monitor if provided
        if self.multi_space_monitor:
            await self._setup_monitor_integration()

        # Start monitoring task
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info("[INTEGRATION-BRIDGE] Started all systems")

    async def stop(self):
        """Stop the integration bridge and all connected systems"""
        self.is_running = False

        # Stop monitoring task
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        # Stop context graph
        await self.context_graph.stop()

        # Stop multi-space monitor if provided
        if self.multi_space_monitor:
            await self.multi_space_monitor.stop_monitoring()

        logger.info("[INTEGRATION-BRIDGE] Stopped all systems")

    # ========================================================================
    # MULTI-SPACE MONITOR INTEGRATION
    # ========================================================================

    async def _setup_monitor_integration(self):
        """Setup integration with MultiSpaceMonitor"""
        from backend.vision.multi_space_monitor import MonitorEventType

        # Register event handlers for all event types
        self.multi_space_monitor.register_event_handler(
            MonitorEventType.SPACE_SWITCHED,
            self._handle_space_switched
        )
        self.multi_space_monitor.register_event_handler(
            MonitorEventType.APP_LAUNCHED,
            self._handle_app_launched
        )
        self.multi_space_monitor.register_event_handler(
            MonitorEventType.APP_CLOSED,
            self._handle_app_closed
        )
        self.multi_space_monitor.register_event_handler(
            MonitorEventType.SPACE_CREATED,
            self._handle_space_created
        )
        self.multi_space_monitor.register_event_handler(
            MonitorEventType.SPACE_REMOVED,
            self._handle_space_removed
        )

        # Start monitoring
        await self.multi_space_monitor.start_monitoring(
            callback=self._handle_monitor_notification
        )

        logger.info("[INTEGRATION-BRIDGE] Set up MultiSpaceMonitor integration")

    async def _handle_space_switched(self, event):
        """Handle space switch event from monitor"""
        to_space = event.details.get('to_space')
        if to_space:
            self.context_graph.set_active_space(to_space)
            logger.debug(f"[INTEGRATION-BRIDGE] Space switched to {to_space}")

    async def _handle_app_launched(self, event):
        """Handle app launch event from monitor"""
        space_id = event.space_id
        app_name = event.app_name

        if space_id and app_name:
            # Detect app type automatically
            context_type = self._detect_app_type(app_name)

            # Create application context in the graph
            space = self.context_graph.get_or_create_space(space_id)
            space.add_application(app_name, context_type)

            logger.debug(f"[INTEGRATION-BRIDGE] App launched: {app_name} ({context_type.value}) in Space {space_id}")

    async def _handle_app_closed(self, event):
        """Handle app close event from monitor"""
        space_id = event.space_id
        app_name = event.app_name

        if space_id and app_name and space_id in self.context_graph.spaces:
            space = self.context_graph.spaces[space_id]
            space.remove_application(app_name)

            logger.debug(f"[INTEGRATION-BRIDGE] App closed: {app_name} in Space {space_id}")

    async def _handle_space_created(self, event):
        """Handle space creation event from monitor"""
        space_id = event.space_id
        if space_id:
            self.context_graph.get_or_create_space(space_id)
            logger.debug(f"[INTEGRATION-BRIDGE] Space created: {space_id}")

    async def _handle_space_removed(self, event):
        """Handle space removal event from monitor"""
        space_id = event.space_id
        if space_id:
            self.context_graph.remove_space(space_id)
            logger.debug(f"[INTEGRATION-BRIDGE] Space removed: {space_id}")

    async def _handle_monitor_notification(self, event):
        """Handle notifications from the monitor that require user attention"""
        # Check with feedback loop if we should show this notification
        if self.feedback_loop:
            from backend.core.learning.feedback_loop import NotificationPattern

            # Map event type to notification pattern
            pattern_map = {
                "WORKFLOW_DETECTED": NotificationPattern.CROSS_SPACE_WORKFLOW,
                "ACTIVITY_SURGE": NotificationPattern.SYSTEM_WARNING,
            }

            pattern = pattern_map.get(event.event_type.name)
            if pattern:
                should_show, adjusted_importance = self.feedback_loop.should_show_notification(
                    pattern=pattern,
                    base_importance=event.importance / 10.0,  # Normalize to 0-1
                    context={"space_id": event.space_id, "app_name": event.app_name}
                )

                if not should_show:
                    logger.debug(f"[INTEGRATION-BRIDGE] Suppressed notification based on feedback learning: {event.event_type.name}")
                    return

        # If we get here, show the notification
        # (Actual notification display would be handled by the calling system)
        logger.info(f"[INTEGRATION-BRIDGE] Notification: {event.event_type.name} - {event.details}")

    # ========================================================================
    # OCR ANALYSIS INTEGRATION
    # ========================================================================

    async def process_ocr_update(self,
                                 space_id: int,
                                 app_name: str,
                                 ocr_text: str,
                                 screenshot_path: Optional[str] = None):
        """
        Process OCR text from a screenshot and update context graph.

        This is called whenever we capture and OCR a screenshot.
        It analyzes the content and updates the appropriate context.

        Args:
            space_id: Which space the screenshot is from
            app_name: Which application
            ocr_text: Extracted OCR text
            screenshot_path: Optional path to the screenshot
        """
        if not self.ocr_analysis_enabled:
            return

        # Detect app type
        context_type = self._detect_app_type(app_name)

        # Add screenshot reference to context graph
        if screenshot_path:
            self.context_graph.add_screenshot_reference(space_id, app_name, screenshot_path, ocr_text)

        # Determine content type and significance
        content_type = context_type.value
        significance = "normal"
        has_error = False

        # Check for critical content (errors, etc.)
        if "error" in ocr_text.lower() or "exception" in ocr_text.lower() or "failed" in ocr_text.lower():
            significance = "critical"
            content_type = "error"
            has_error = True

        # Record visual attention if implicit resolver is available
        if self.implicit_resolver:
            self.implicit_resolver.record_visual_attention(
                space_id=space_id,
                app_name=app_name,
                ocr_text=ocr_text,
                content_type=content_type,
                significance=significance
            )

        # Record activity in cross-space intelligence if available
        if self.cross_space_intelligence:
            self.cross_space_intelligence.record_activity(
                space_id=space_id,
                app_name=app_name,
                content=ocr_text,
                activity_type=context_type.value,
                has_error=has_error,
                significance=significance
            )

        # Route to appropriate analyzer based on app type
        if context_type.value == "terminal":
            await self._analyze_terminal_ocr(space_id, app_name, ocr_text)
        elif context_type.value == "browser":
            await self._analyze_browser_ocr(space_id, app_name, ocr_text)
        elif context_type.value == "ide":
            await self._analyze_ide_ocr(space_id, app_name, ocr_text)
        else:
            # Generic context update
            self.context_graph.update_generic_context(space_id, app_name, ocr_text)

        logger.debug(f"[INTEGRATION-BRIDGE] Processed OCR for {app_name} in Space {space_id}")

    async def _analyze_terminal_ocr(self, space_id: int, app_name: str, ocr_text: str):
        """Analyze terminal OCR text using TerminalCommandIntelligence"""
        if not self.terminal_intelligence:
            # Lazy load if not provided
            try:
                from backend.vision.handlers.terminal_command_intelligence import get_terminal_intelligence
                self.terminal_intelligence = get_terminal_intelligence()
            except Exception as e:
                logger.warning(f"[INTEGRATION-BRIDGE] Could not load terminal intelligence: {e}")
                return

        try:
            # Analyze terminal context
            context = await self.terminal_intelligence.analyze_terminal_context(ocr_text)

            # Extract command and errors
            command = context.last_command
            errors = context.errors if context.errors else []
            working_dir = context.current_directory

            # Determine exit code based on errors
            exit_code = 1 if errors else 0

            # Update context graph
            self.context_graph.update_terminal_context(
                space_id=space_id,
                app_name=app_name,
                command=command,
                output=ocr_text,
                errors=errors,
                exit_code=exit_code,
                working_dir=working_dir
            )

            # If there are errors, trigger critical event handling
            if errors:
                logger.info(f"[INTEGRATION-BRIDGE] Detected terminal error in Space {space_id}: {errors[0][:100]}")

                # Get fix suggestions if available
                if self.terminal_intelligence:
                    suggestions = await self.terminal_intelligence.suggest_fix_commands(context)
                    if suggestions:
                        logger.info(f"[INTEGRATION-BRIDGE] Generated {len(suggestions)} fix suggestions")

        except Exception as e:
            logger.error(f"[INTEGRATION-BRIDGE] Error analyzing terminal OCR: {e}")

    async def _analyze_browser_ocr(self, space_id: int, app_name: str, ocr_text: str):
        """Analyze browser OCR text"""
        # Extract URL if visible in OCR (browsers usually show URL in address bar)
        url = self._extract_url_from_text(ocr_text)

        # Extract title (usually at top of page)
        title = self._extract_title_from_text(ocr_text)

        # Detect if this looks like documentation/research
        research_indicators = [
            "documentation", "docs", "stack overflow", "github",
            "tutorial", "guide", "reference", "api", "example"
        ]
        is_research = any(indicator in ocr_text.lower() for indicator in research_indicators)

        # Update context graph
        self.context_graph.update_browser_context(
            space_id=space_id,
            app_name=app_name,
            url=url,
            title=title,
            extracted_text=ocr_text
        )

        if is_research:
            logger.debug(f"[INTEGRATION-BRIDGE] Detected research activity in Space {space_id}")

    async def _analyze_ide_ocr(self, space_id: int, app_name: str, ocr_text: str):
        """Analyze IDE OCR text"""
        # Extract filename from common IDE patterns
        # IDEs usually show filename in title bar or tab
        active_file = self._extract_filename_from_text(ocr_text)

        # Detect errors/warnings (IDEs often show these with specific markers)
        errors = []
        if "error:" in ocr_text.lower() or "✗" in ocr_text:
            # Extract error lines
            for line in ocr_text.split('\n'):
                if "error" in line.lower() or "✗" in line:
                    errors.append(line.strip())

        # Update context graph
        self.context_graph.update_ide_context(
            space_id=space_id,
            app_name=app_name,
            active_file=active_file,
            errors=errors if errors else None
        )

    # ========================================================================
    # NATURAL LANGUAGE QUERY INTERFACE
    # ========================================================================

    def _compute_followup_score(self, query: str, last_query: str, last_response: str) -> float:
        """
        Compute semantic similarity score to determine if query is a follow-up.
        Uses lightweight NLP techniques - no heavy ML models.

        Returns: 0.0-1.0 confidence score
        """
        query_lower = query.lower()

        # Calculate multiple signals and combine them
        signals = []

        # Signal 1: Pronoun/Anaphora detection (references to "it", "that", "there")
        anaphora_words = ['it', 'that', 'this', 'there', 'them', 'those', 'these']
        has_anaphora = any(f" {word} " in f" {query_lower} " for word in anaphora_words)
        if has_anaphora:
            signals.append(0.4)  # Strong signal

        # Signal 2: Interrogative continuations (what, why, how about previous topic)
        interrogatives = ['what', 'why', 'how', 'when', 'where', 'which', 'who']
        starts_with_interrogative = any(query_lower.startswith(word) for word in interrogatives)
        if starts_with_interrogative and has_anaphora:
            signals.append(0.5)  # Very strong

        # Signal 3: Request verbs indicating elaboration
        elaboration_verbs = ['explain', 'tell', 'show', 'describe', 'detail', 'elaborate', 'clarify']
        has_elaboration = any(verb in query_lower for verb in elaboration_verbs)
        if has_elaboration:
            signals.append(0.3)

        # Signal 4: Affirmative responses
        affirmatives = ['yes', 'yeah', 'yep', 'sure', 'okay', 'ok', 'please', 'go ahead', 'proceed']
        is_affirmative = any(f" {word} " in f" {query_lower} " or query_lower.startswith(word) for word in affirmatives)
        if is_affirmative and len(query.split()) <= 5:  # Short affirmative
            signals.append(0.6)

        # Signal 5: Entity overlap (same apps/spaces mentioned in previous conversation)
        if self._last_context and self._last_context.get("apps"):
            mentioned_apps = set()
            for app_info in self._last_context["apps"]:
                app_name = app_info.get("name", "").lower()
                if app_name and app_name in query_lower:
                    mentioned_apps.add(app_name)

            if mentioned_apps:
                signals.append(0.4)  # References same apps

        # Signal 6: Query brevity (follow-ups tend to be shorter)
        word_count = len(query.split())
        if word_count <= 8:
            signals.append(0.2)

        # Signal 7: No new question structure (lacks independent context)
        # If query doesn't introduce new topics/apps, likely a follow-up
        has_new_topic_words = any(word in query_lower for word in [
            'can you see', 'what do you see', 'show me', 'open', 'close', 'launch',
            'start', 'run', 'execute', 'search for', 'find'
        ])
        if not has_new_topic_words:
            signals.append(0.3)

        # Combine signals with weighted average
        if not signals:
            return 0.0

        # Boost score if multiple signals present
        base_score = sum(signals) / len(signals)
        signal_diversity_bonus = min(0.2, len(signals) * 0.03)

        return min(1.0, base_score + signal_diversity_bonus)

    async def check_followup_query(self, query: str, current_space_id: Optional[int] = None) -> Optional[str]:
        """
        Check if query is a follow-up to previous conversation.
        Returns detailed response if it's a follow-up, None otherwise.

        Uses semantic analysis instead of hardcoded keywords.
        Lightweight NLP - optimized for M1 Mac (no heavy ML models).
        """
        query_normalized = self._normalize_speech_query(query.lower())

        # Check if we have recent conversational context (within last 2 minutes)
        if not self._conversation_timestamp:
            return None

        time_since = datetime.now() - self._conversation_timestamp
        if time_since.total_seconds() >= 120:  # 2 minutes
            return None

        # Compute semantic followup score
        followup_score = self._compute_followup_score(
            query_normalized,
            self._last_query or "",
            self._last_response or ""
        )

        # Adaptive threshold based on time since last interaction
        # Recent queries need lower threshold (more likely to be follow-ups)
        time_factor = max(0.0, 1.0 - (time_since.total_seconds() / 120))
        adaptive_threshold = 0.5 - (time_factor * 0.2)  # Range: 0.3-0.5

        if followup_score >= adaptive_threshold:
            logger.info(
                f"[CONTEXT-BRIDGE] Follow-up detected (score: {followup_score:.2f}, "
                f"threshold: {adaptive_threshold:.2f}): '{query_normalized}'"
            )
            return await self._handle_detail_followup(query_normalized, current_space_id)

        logger.debug(
            f"[CONTEXT-BRIDGE] Not a follow-up (score: {followup_score:.2f}, "
            f"threshold: {adaptive_threshold:.2f})"
        )
        return None

    async def answer_query(self, query: str, current_space_id: Optional[int] = None) -> str:
        """
        Answer natural language queries about workspace context.

        This is the foundation for "what does it say?" queries.

        Examples:
            - "what does it say?" → Find and explain most recent error
            - "what's the error?" → Find most recent error
            - "what's happening?" → Summarize current space activity
            - "explain that" → Explain the thing we just discussed
            - "what am I working on?" → Synthesize workspace-wide context
            - "can you see my terminal?" → Proactively offer to explain what's there
            - "explain in detail" → Follow-up query for detailed explanation

        Args:
            query: Natural language query
            current_space_id: Optional current space ID for context

        Returns:
            Natural language response
        """
        # Normalize speech-to-text errors FIRST
        query_lower = self._normalize_speech_query(query.lower())

        # Log the normalization for debugging
        if query_lower != query.lower():
            logger.debug(f"[CONTEXT-BRIDGE] Normalized query: '{query}' → '{query_lower}'")

        # Check for follow-up queries requesting more detail
        # This includes affirmative responses to "Would you like me to help/explain?"
        detail_keywords = [
            "explain in detail", "more detail", "tell me more", "what's happening",
            "explain what's happening", "what is happening", "explain that",
            "give me details", "explain it", "what's going on", "what is going on",
            # Affirmative continuations
            "yes", "yeah", "yep", "sure", "please", "go ahead",
            # Natural continuations that reference previous context
            "explain", "what's in", "what is in", "what about", "help me with"
        ]

        # Check if we have recent conversational context (within last 2 minutes)
        if self._conversation_timestamp:
            time_since = datetime.now() - self._conversation_timestamp
            if time_since.total_seconds() < 120:  # 2 minutes
                # Check for follow-up patterns
                is_followup = False

                # Direct match on keywords
                if any(kw in query_lower for kw in detail_keywords):
                    is_followup = True

                # Detect affirmative + context reference
                # E.g., "yes, explain what's happening" or "yes jarvis, can you explain"
                if any(affirm in query_lower for affirm in ["yes", "yeah", "sure", "please"]):
                    if any(ctx in query_lower for ctx in ["explain", "tell me", "what", "help"]):
                        is_followup = True

                if is_followup:
                    logger.info(f"[CONTEXT-BRIDGE] Detected follow-up query: '{query_lower}'")
                    return await self._handle_detail_followup(query_lower, current_space_id)

        # Handle "can you see" queries - be proactive about explaining
        visibility_keywords = ["can you see", "do you see", "are you seeing", "what do you see"]
        if any(kw in query_lower for kw in visibility_keywords):
            response = await self._handle_visibility_query(query_lower, current_space_id)
            # Save conversational context for follow-up
            self._save_conversation_context(query, response, current_space_id)
            return response

        # Use cross-space intelligence for workspace-wide queries
        if self.cross_space_intelligence:
            # Check if this is a workspace-wide query
            workspace_queries = ["working on", "related", "connected", "across", "all spaces"]
            if any(kw in query.lower() for kw in workspace_queries):
                try:
                    result = await self.cross_space_intelligence.answer_workspace_query(
                        query, current_space_id
                    )
                    if result.get("found"):
                        return result["response"]
                except Exception as e:
                    logger.error(f"[INTEGRATION-BRIDGE] Error in cross-space intelligence: {e}")

        # Use implicit resolver if available (advanced understanding)
        if self.implicit_resolver:
            try:
                result = await self.implicit_resolver.resolve_query(query)
                return result["response"]
            except Exception as e:
                logger.error(f"[INTEGRATION-BRIDGE] Error in implicit resolver: {e}")
                # Fall through to basic resolution

        # Fallback: Use basic context graph query
        context = self.context_graph.find_context_for_query(query)

        # Generate natural language response
        if context["type"] == "error":
            return self._format_error_response(context)
        elif context["type"] == "terminal":
            return self._format_terminal_response(context)
        elif context["type"] == "current_space":
            return self._format_current_space_response(context)
        elif context["type"] == "no_relevant_context":
            return context["message"]
        else:
            return "I'm not sure what you're referring to. Could you be more specific?"

    def _normalize_speech_query(self, query: str) -> str:
        """
        Normalize common speech-to-text transcription errors and variations.

        Common issues:
        - "and" → "in" (e.g., "terminal and the other window" → "terminal in the other window")
        - "on" → "in" (e.g., "terminal on the other space")
        - Missing words (e.g., "see terminal" → "see my terminal")
        - Filler words (e.g., "um", "uh", "like")
        """
        # Remove common filler words (handle start/end of string too)
        filler_words = ["um ", "uh ", "like ", "you know ", "basically ", "actually "]

        # Handle filler words at the beginning
        for filler in filler_words:
            if query.startswith(filler):
                query = query[len(filler):]

        # Handle filler words in the middle (with spaces on both sides)
        for filler in filler_words:
            query = query.replace(f" {filler}", " ")
            query = query.replace(f" {filler.strip()} ", " ")

        # Fix common speech-to-text errors
        speech_corrections = {
            # "and the other" → "in the other" (most common mishearing)
            " and the other window": " in the other window",
            " and the other space": " in the other space",
            " and the other tab": " in the other tab",
            " and the other screen": " in the other screen",
            " and another window": " in another window",
            " and another space": " in another space",
            " and other window": " in the other window",
            " and other space": " in the other space",

            # "on" → "in"
            " on the other window": " in the other window",
            " on the other space": " in the other space",
            " on another window": " in another window",
            " on another space": " in another space",
            " on other window": " in the other window",
            " on other space": " in the other space",

            # "of" → "in"
            " of the other window": " in the other window",
            " of another window": " in another window",

            # "at" → "in"
            " at the other window": " in the other window",

            # Add missing possessives
            "see terminal": "see my terminal",
            "see the terminal": "see my terminal",
            "see browser": "see my browser",
            "see the browser": "see my browser",
            "see code": "see my code",
            "see editor": "see my editor",

            # Common word confusions
            " termonal ": " terminal ",
            " terminol ": " terminal ",
            " console ": " terminal ",
            " crome ": " chrome ",
            " safari ": " browser ",
            " firefox ": " browser ",

            # Variations of "the"
            " da ": " the ",
            " de ": " the ",
            " duh ": " the ",
        }

        for wrong, correct in speech_corrections.items():
            query = query.replace(wrong, correct)

        # Clean up extra spaces
        query = " ".join(query.split())

        return query

    async def _handle_visibility_query(self, query: str, current_space_id: Optional[int]) -> str:
        """
        Handle "can you see X?" queries - be proactive about explaining what's visible.

        Examples:
        - "can you see my terminal?" → "Yes, I can see Terminal in Space 2. I notice there's an error..."
        - "do you see the error?" → "Yes, I see an error in Terminal (Space 1): ModuleNotFoundError..."
        - "can you see my terminal and the other window?" → Handles speech-to-text "and" vs "in"

        Note: query is already normalized by answer_query()
        """
        query_lower = query  # Already normalized and lowercased

        # Extract what they're asking about
        target_keywords = {
            "terminal": ["terminal", "console", "command line", "shell"],
            "browser": ["browser", "chrome", "safari", "firefox", "web"],
            "editor": ["editor", "vscode", "code", "ide", "cursor"],
            "error": ["error", "problem", "issue", "failed"]
        }

        target_type = None
        for app_type, keywords in target_keywords.items():
            if any(kw in query_lower for kw in keywords):
                target_type = app_type
                break

        # Get all spaces summary
        summary = self.context_graph.get_summary()
        spaces = summary.get("spaces", {})

        # Look for the target across all spaces
        found_apps = []
        for space_id, space_data in spaces.items():
            apps = space_data.get("applications", {})
            for app_name, app_data in apps.items():
                if target_type:
                    # Check if app matches target type
                    context_type = app_data.get("context_type", "").lower()
                    if target_type in context_type or target_type == "error":
                        found_apps.append((space_id, app_name, app_data))
                else:
                    # No specific target, include all active apps
                    if app_data.get("activity_count", 0) > 0:
                        found_apps.append((space_id, app_name, app_data))

        if not found_apps:
            return f"I don't see any {target_type or 'activity'} in your workspace right now. Would you like me to start monitoring?"

        # Build proactive response
        response_parts = []

        # Affirmative answer
        if len(found_apps) == 1:
            space_id, app_name, app_data = found_apps[0]
            response_parts.append(f"Yes, I can see {app_name} in Space {space_id}.")
        else:
            response_parts.append(f"Yes, I can see {len(found_apps)} windows across your workspace:")
            for space_id, app_name, _ in found_apps[:3]:  # Show first 3
                response_parts.append(f"  • {app_name} (Space {space_id})")

        # Check for errors or significant content
        has_errors = False
        error_details = []

        for space_id, app_name, app_data in found_apps:
            # Check for terminal errors
            if app_data.get("context_type") == "terminal":
                terminal_ctx = app_data.get("terminal_context", {})
                errors = terminal_ctx.get("errors", [])
                if errors:
                    has_errors = True
                    error_details.append({
                        "space_id": space_id,
                        "app_name": app_name,
                        "error": errors[0]  # Most recent error
                    })

        # Proactively offer to explain if there's something interesting
        if has_errors:
            response_parts.append("")  # Blank line
            if len(error_details) == 1:
                err = error_details[0]
                response_parts.append(f"I notice there's an error in {err['app_name']} (Space {err['space_id']}):")
                response_parts.append(f"  {err['error'][:150]}...")
                response_parts.append("")
                response_parts.append("Would you like me to explain what's happening in detail?")
            else:
                response_parts.append(f"I notice {len(error_details)} errors across your workspace.")
                response_parts.append("")
                response_parts.append("Would you like me to explain what's happening?")
        else:
            # No errors, but still offer help
            response_parts.append("")
            response_parts.append("Everything looks normal. Would you like me to explain what's happening?")

        return "\n".join(response_parts)

    async def _handle_detail_followup(self, query: str, current_space_id: Optional[int]) -> str:
        """
        Handle follow-up queries asking for more detail about what was just discussed.

        This provides dynamic, detailed explanations based on actual terminal/app context.
        NO HARDCODED RESPONSES - everything is generated from real context data.

        Examples:
        - User: "can you see my terminal in the other window?"
        - JARVIS: "Yes, I can see Terminal in Space 2..."
        - User: "explain what's happening in detail"
        - JARVIS: [Dynamic explanation based on terminal context]

        Args:
            query: Follow-up query (already normalized)
            current_space_id: Current space ID

        Returns:
            Detailed natural language explanation
        """
        logger.info("[CONTEXT-BRIDGE] Handling follow-up detail query")

        # Get the context from the last query (what apps/spaces we discussed)
        last_context = self._last_context

        if not last_context:
            return "I'm not sure what you'd like me to explain. Could you ask about something specific?"

        # Get the apps we were discussing
        discussed_apps = last_context.get("apps", [])

        if not discussed_apps:
            return "I don't have enough context to provide details. What would you like to know about?"

        # Build comprehensive explanation by analyzing each app
        response_parts = []

        for app_info in discussed_apps:
            space_id = app_info["space_id"]
            app_name = app_info["app_name"]
            app_data = app_info["app_data"]
            context_type = app_data.get("context_type", "").lower()

            # Terminal apps - use TerminalCommandIntelligence for rich analysis
            if context_type == "terminal" and self.terminal_intelligence:
                terminal_ctx = app_data.get("terminal_context", {})

                # Generate dynamic explanation based on terminal state
                explanation = await self._explain_terminal_context(
                    app_name, space_id, terminal_ctx
                )
                response_parts.append(explanation)

            # Browser apps - explain what they're viewing/researching
            elif context_type == "browser":
                browser_ctx = app_data.get("browser_context", {})
                explanation = self._explain_browser_context(
                    app_name, space_id, browser_ctx
                )
                response_parts.append(explanation)

            # IDE/Editor apps - explain code context
            elif context_type in ["ide", "editor"]:
                ide_ctx = app_data.get("ide_context", {})
                explanation = self._explain_ide_context(
                    app_name, space_id, ide_ctx
                )
                response_parts.append(explanation)

            # Generic apps - basic context
            else:
                response_parts.append(
                    f"**{app_name} (Space {space_id})**\n"
                    f"Last activity: {app_data.get('last_activity', 'Unknown')}"
                )

        # Use cross-space intelligence to find relationships
        if self.cross_space_intelligence and len(discussed_apps) > 1:
            try:
                relationships = await self._find_cross_space_relationships(discussed_apps)
                if relationships:
                    response_parts.append("\n**Cross-Space Analysis:**")
                    response_parts.append(relationships)
            except Exception as e:
                logger.error(f"[CONTEXT-BRIDGE] Error finding relationships: {e}")

        if not response_parts:
            return "I don't have detailed information available for what we were discussing."

        return "\n\n".join(response_parts)

    async def _explain_terminal_context(
        self, app_name: str, space_id: int, terminal_ctx: Dict[str, Any]
    ) -> str:
        """
        Generate dynamic explanation of terminal context.
        Uses TerminalCommandIntelligence for rich analysis.
        """
        parts = [f"**{app_name} (Space {space_id})**"]

        # Extract terminal context
        last_command = terminal_ctx.get("last_command")
        last_output = terminal_ctx.get("last_output")
        errors = terminal_ctx.get("errors", [])
        working_dir = terminal_ctx.get("working_directory")
        recent_commands = terminal_ctx.get("recent_commands", [])

        # Working directory
        if working_dir:
            parts.append(f"Working directory: `{working_dir}`")

        # Recent commands context
        if recent_commands and len(recent_commands) > 0:
            parts.append(f"\nRecent commands:")
            for cmd_tuple in list(recent_commands)[-3:]:  # Last 3 commands
                # Handle both string and tuple formats
                if isinstance(cmd_tuple, tuple):
                    cmd = cmd_tuple[0]
                else:
                    cmd = cmd_tuple
                parts.append(f"  • `{cmd}`")

        # Last command executed
        if last_command:
            parts.append(f"\nLast command: `{last_command}`")

        # Errors - provide detailed analysis
        if errors:
            parts.append(f"\n**Error Analysis:**")

            for error in errors[:2]:  # Show up to 2 errors
                parts.append(f"\n{error}")

                # Use TerminalCommandIntelligence to suggest fixes
                if self.terminal_intelligence:
                    try:
                        # Create minimal OCR text for analysis
                        ocr_text = f"{last_command}\n{error}"
                        term_ctx = await self.terminal_intelligence.analyze_terminal_context(ocr_text)
                        suggestions = await self.terminal_intelligence.suggest_fix_commands(term_ctx)

                        if suggestions:
                            parts.append("\n**Suggested Fix:**")
                            for i, suggestion in enumerate(suggestions[:2], 1):
                                parts.append(
                                    f"{i}. `{suggestion.command}`\n"
                                    f"   Purpose: {suggestion.purpose}\n"
                                    f"   Safety: {suggestion.safety_tier.upper()}\n"
                                    f"   Impact: {suggestion.estimated_impact}"
                                )
                    except Exception as e:
                        logger.error(f"[CONTEXT-BRIDGE] Error getting command suggestions: {e}")

        # Command output (if no errors, show what happened)
        elif last_output:
            parts.append(f"\nOutput:")
            # Truncate long output
            output_preview = last_output[:300]
            if len(last_output) > 300:
                output_preview += "..."
            parts.append(f"```\n{output_preview}\n```")

        # Activity summary
        if not errors and not last_output:
            parts.append("\nNo recent activity detected.")

        return "\n".join(parts)

    def _explain_browser_context(
        self, app_name: str, space_id: int, browser_ctx: Dict[str, Any]
    ) -> str:
        """Generate explanation of browser context."""
        parts = [f"**{app_name} (Space {space_id})**"]

        active_url = browser_ctx.get("active_url")
        page_title = browser_ctx.get("page_title")
        search_query = browser_ctx.get("search_query")
        is_researching = browser_ctx.get("is_researching", False)
        research_topic = browser_ctx.get("research_topic")

        if active_url:
            parts.append(f"Current page: {page_title or active_url}")

        if search_query:
            parts.append(f"Recent search: \"{search_query}\"")

        if is_researching and research_topic:
            parts.append(f"Research topic: {research_topic}")

        if not active_url and not search_query:
            parts.append("No active browsing detected.")

        return "\n".join(parts)

    def _explain_ide_context(
        self, app_name: str, space_id: int, ide_ctx: Dict[str, Any]
    ) -> str:
        """Generate explanation of IDE/editor context."""
        parts = [f"**{app_name} (Space {space_id})**"]

        active_file = ide_ctx.get("active_file")
        open_files = ide_ctx.get("open_files", [])
        project_name = ide_ctx.get("project_name")

        if project_name:
            parts.append(f"Project: {project_name}")

        if active_file:
            parts.append(f"Editing: `{active_file}`")

        if open_files and len(open_files) > 1:
            parts.append(f"Open files: {len(open_files)}")
            for file in open_files[:3]:
                parts.append(f"  • {file}")

        if not active_file and not open_files:
            parts.append("No files currently open.")

        return "\n".join(parts)

    async def _find_cross_space_relationships(self, apps: List[Dict[str, Any]]) -> str:
        """
        Find relationships between apps across different spaces.
        Uses CrossSpaceIntelligence for semantic correlation.
        """
        if not self.cross_space_intelligence or len(apps) < 2:
            return ""

        # Extract activity contexts for correlation
        contexts = []
        for app in apps:
            contexts.append({
                "space_id": app["space_id"],
                "app_name": app["app_name"],
                "content": self._extract_app_content(app["app_data"])
            })

        # Find correlations
        try:
            correlations = await self.cross_space_intelligence.find_correlations(contexts)

            if correlations:
                parts = []
                for corr in correlations[:3]:  # Top 3 correlations
                    parts.append(
                        f"• {corr.get('description', 'Related activity detected')} "
                        f"(confidence: {corr.get('confidence', 0):.0%})"
                    )
                return "\n".join(parts)
        except Exception as e:
            logger.error(f"[CONTEXT-BRIDGE] Error in cross-space correlation: {e}")

        return ""

    def _extract_app_content(self, app_data: Dict[str, Any]) -> str:
        """Extract relevant text content from app data for correlation."""
        content_parts = []

        # Terminal content
        if "terminal_context" in app_data:
            term_ctx = app_data["terminal_context"]
            if term_ctx.get("last_command"):
                content_parts.append(term_ctx["last_command"])
            if term_ctx.get("errors"):
                content_parts.extend(term_ctx["errors"])

        # Browser content
        if "browser_context" in app_data:
            browser_ctx = app_data["browser_context"]
            if browser_ctx.get("page_title"):
                content_parts.append(browser_ctx["page_title"])
            if browser_ctx.get("search_query"):
                content_parts.append(browser_ctx["search_query"])

        # IDE content
        if "ide_context" in app_data:
            ide_ctx = app_data["ide_context"]
            if ide_ctx.get("active_file"):
                content_parts.append(ide_ctx["active_file"])

        return " ".join(content_parts)

    def _save_conversation_context(
        self, query: str, response: str, current_space_id: Optional[int]
    ):
        """
        Save conversational context for follow-up queries.
        Stores what apps/spaces we discussed for intelligent follow-ups.
        """
        self._last_query = query
        self._last_response = response
        self._conversation_timestamp = datetime.now()

        # Extract which apps we talked about from the current workspace
        # Access the actual context objects, not the serialized dict
        discussed_apps = []
        for space_id, space in self.context_graph.spaces.items():
            for app_name, app_ctx in space.applications.items():
                # Include apps with recent activity
                if app_ctx.activity_count > 0:
                    # Build app_data dict with the actual context objects
                    app_data = {
                        "app_name": app_ctx.app_name,
                        "context_type": app_ctx.context_type.value,
                        "last_activity": app_ctx.last_activity.isoformat(),
                        "activity_count": app_ctx.activity_count,
                        "significance": app_ctx.significance.value
                    }

                    # Add the actual context objects
                    if app_ctx.terminal_context:
                        app_data["terminal_context"] = asdict(app_ctx.terminal_context)
                    if app_ctx.browser_context:
                        app_data["browser_context"] = asdict(app_ctx.browser_context)
                    if app_ctx.ide_context:
                        app_data["ide_context"] = asdict(app_ctx.ide_context)

                    discussed_apps.append({
                        "space_id": space_id,
                        "app_name": app_name,
                        "app_data": app_data
                    })

        self._last_context = {
            "apps": discussed_apps,
            "current_space_id": current_space_id,
            "timestamp": datetime.now()
        }

        logger.debug(
            f"[CONTEXT-BRIDGE] Saved conversation context: "
            f"{len(discussed_apps)} apps across {len(self.context_graph.spaces)} spaces"
        )

    def _format_error_response(self, context: Dict[str, Any]) -> str:
        """Format error context into natural language"""
        space_id = context.get("space_id")
        app_name = context.get("app_name")
        error = context["details"].get("error", "Unknown error")
        command = context["details"].get("command")

        response = f"The error in {app_name} (Space {space_id}) is:\n\n{error}"

        if command:
            response += f"\n\nThis happened when you ran: `{command}`"

        # Check if there are fix suggestions
        if self.terminal_intelligence and app_name in ["Terminal", "iTerm", "iTerm2"]:
            # Note: Would need to pass terminal context here for actual suggestions
            response += "\n\nI can suggest a fix if you'd like."

        return response

    def _format_terminal_response(self, context: Dict[str, Any]) -> str:
        """Format terminal context into natural language"""
        space_id = context.get("space_id")
        app_name = context.get("app_name")
        last_command = context.get("last_command")
        errors = context.get("errors", [])

        response = f"In {app_name} (Space {space_id}):\n\n"

        if last_command:
            response += f"Last command: `{last_command}`\n"

        if errors:
            response += f"\nErrors:\n"
            for error in errors[:3]:  # Show first 3 errors
                response += f"  • {error}\n"
        else:
            response += "\nNo errors detected."

        return response

    def _format_current_space_response(self, context: Dict[str, Any]) -> str:
        """Format current space context into natural language"""
        space_id = context.get("space_id")
        applications = context.get("applications", [])
        recent_events = context.get("recent_events", [])

        response = f"In Space {space_id}:\n\n"

        if applications:
            response += f"Open applications: {', '.join(applications)}\n\n"

        # Check for cross-space relationships
        cross_space_summary = self.context_graph.get_cross_space_summary()
        if cross_space_summary and "No" not in cross_space_summary:
            response += f"\n{cross_space_summary}"

        return response

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _detect_app_type(self, app_name: str):
        """Automatically detect application context type"""
        from backend.core.context.multi_space_context_graph import ContextType

        app_lower = app_name.lower()

        # Terminal apps
        if any(term in app_lower for term in ["terminal", "iterm", "console", "cmd", "powershell"]):
            return ContextType.TERMINAL

        # Browsers
        elif any(browser in app_lower for browser in ["safari", "chrome", "firefox", "arc", "brave", "edge"]):
            return ContextType.BROWSER

        # IDEs
        elif any(ide in app_lower for ide in ["code", "vscode", "intellij", "pycharm", "sublime", "atom", "vim", "emacs"]):
            return ContextType.IDE

        # Communication
        elif any(comm in app_lower for comm in ["slack", "discord", "zoom", "teams", "messages"]):
            return ContextType.COMMUNICATION

        # Editors
        elif any(ed in app_lower for ed in ["notes", "textedit", "word", "pages", "notion"]):
            return ContextType.EDITOR

        else:
            return ContextType.GENERIC

    def _extract_url_from_text(self, text: str) -> Optional[str]:
        """Extract URL from OCR text (basic implementation)"""
        import re
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        match = re.search(url_pattern, text)
        return match.group(0) if match else None

    def _extract_title_from_text(self, text: str) -> Optional[str]:
        """Extract page title from OCR text (basic implementation)"""
        # Usually first line or first non-URL line
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines[:5]:  # Check first 5 lines
            if not line.startswith('http') and len(line) > 3 and len(line) < 200:
                return line
        return None

    def _extract_filename_from_text(self, text: str) -> Optional[str]:
        """Extract filename from OCR text (basic implementation)"""
        import re
        # Look for common file extensions
        file_pattern = r'\b[\w\-]+\.(py|js|ts|jsx|tsx|java|cpp|c|h|go|rs|rb|php|html|css|json|yaml|yml|md|txt)\b'
        match = re.search(file_pattern, text)
        return match.group(0) if match else None

    async def _monitoring_loop(self):
        """Background monitoring loop for periodic updates"""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                # Update space inferred tags
                for space in self.context_graph.spaces.values():
                    space.infer_tags()

                # Log summary
                summary = self.context_graph.get_summary()
                logger.debug(f"[INTEGRATION-BRIDGE] Spaces: {summary['total_spaces']}, Active: {len(summary['active_spaces'])}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[INTEGRATION-BRIDGE] Error in monitoring loop: {e}")

    # ========================================================================
    # PUBLIC API - For External Systems
    # ========================================================================

    def get_context_summary(self) -> Dict[str, Any]:
        """Get comprehensive context summary for external systems"""
        return self.context_graph.get_summary()

    def get_workspace_intelligence_summary(self) -> Dict[str, Any]:
        """Get workspace-wide intelligence summary including cross-space relationships"""
        summary = self.context_graph.get_summary()

        # Add cross-space intelligence if available
        if self.cross_space_intelligence:
            workspace_summary = self.cross_space_intelligence.get_workspace_summary()
            summary["cross_space_intelligence"] = workspace_summary

        return summary

    async def handle_user_query(self, query: str, current_space_id: Optional[int] = None) -> str:
        """
        Handle user query (main entry point for conversational interface).

        This is what gets called when the user says things like:
        - "what does it say?"
        - "what's the error?"
        - "what's happening in the terminal?"
        - "what am I working on?"
        """
        return await self.answer_query(query, current_space_id)

    def export_context(self, filepath: Path):
        """Export current context to file (for debugging/analysis)"""
        self.context_graph.export_to_json(filepath)


# ============================================================================
# GLOBAL INSTANCE MANAGEMENT
# ============================================================================

_global_bridge: Optional[ContextIntegrationBridge] = None


def get_integration_bridge() -> Optional[ContextIntegrationBridge]:
    """Get the global integration bridge instance"""
    return _global_bridge


def set_integration_bridge(bridge: ContextIntegrationBridge):
    """Set the global integration bridge instance"""
    global _global_bridge
    _global_bridge = bridge


async def initialize_integration_bridge(
    context_graph=None,
    multi_space_monitor=None,
    terminal_intelligence=None,
    feedback_loop=None,
    auto_start: bool = True
) -> ContextIntegrationBridge:
    """
    Initialize and configure the integration bridge.

    This is the main initialization function that should be called at JARVIS startup.

    Args:
        context_graph: Optional MultiSpaceContextGraph (created if None)
        multi_space_monitor: Optional MultiSpaceMonitor (created if None)
        terminal_intelligence: Optional TerminalCommandIntelligence (loaded if None)
        feedback_loop: Optional FeedbackLearningLoop (loaded if None)
        auto_start: Whether to automatically start all systems

    Returns:
        Configured ContextIntegrationBridge instance
    """
    global _global_bridge

    # Create context graph if not provided
    if context_graph is None:
        from backend.core.context.multi_space_context_graph import MultiSpaceContextGraph
        context_graph = MultiSpaceContextGraph(
            decay_ttl_seconds=300,  # 5 minutes
            enable_cross_space_correlation=True
        )
        logger.info("[INTEGRATION-BRIDGE] Created new MultiSpaceContextGraph")

    # Create multi-space monitor if not provided
    if multi_space_monitor is None:
        try:
            from backend.vision.multi_space_monitor import MultiSpaceMonitor
            multi_space_monitor = MultiSpaceMonitor()
            logger.info("[INTEGRATION-BRIDGE] Created new MultiSpaceMonitor")
        except Exception as e:
            logger.warning(f"[INTEGRATION-BRIDGE] Could not create MultiSpaceMonitor: {e}")

    # Load terminal intelligence if not provided
    if terminal_intelligence is None:
        try:
            from backend.vision.handlers.terminal_command_intelligence import get_terminal_intelligence
            terminal_intelligence = get_terminal_intelligence()
            logger.info("[INTEGRATION-BRIDGE] Loaded TerminalCommandIntelligence")
        except Exception as e:
            logger.warning(f"[INTEGRATION-BRIDGE] Could not load terminal intelligence: {e}")

    # Load feedback loop if not provided
    if feedback_loop is None:
        try:
            from backend.core.learning.feedback_loop import get_feedback_loop
            feedback_loop = get_feedback_loop()
            logger.info("[INTEGRATION-BRIDGE] Loaded FeedbackLearningLoop")
        except Exception as e:
            logger.warning(f"[INTEGRATION-BRIDGE] Could not load feedback loop: {e}")

    # Create implicit reference resolver
    implicit_resolver = None
    try:
        from backend.core.nlp.implicit_reference_resolver import initialize_implicit_resolver
        implicit_resolver = initialize_implicit_resolver(context_graph)
        logger.info("[INTEGRATION-BRIDGE] Created ImplicitReferenceResolver")
    except Exception as e:
        logger.warning(f"[INTEGRATION-BRIDGE] Could not create implicit resolver: {e}")

    # Create cross-space intelligence
    cross_space_intelligence = None
    try:
        from backend.core.intelligence.cross_space_intelligence import initialize_cross_space_intelligence
        cross_space_intelligence = initialize_cross_space_intelligence()
        logger.info("[INTEGRATION-BRIDGE] Created CrossSpaceIntelligence")
    except Exception as e:
        logger.warning(f"[INTEGRATION-BRIDGE] Could not create cross-space intelligence: {e}")

    # Create bridge
    bridge = ContextIntegrationBridge(
        context_graph=context_graph,
        multi_space_monitor=multi_space_monitor,
        terminal_intelligence=terminal_intelligence,
        feedback_loop=feedback_loop,
        implicit_resolver=implicit_resolver,
        cross_space_intelligence=cross_space_intelligence
    )

    # Set as global instance
    _global_bridge = bridge

    # Auto-start if requested
    if auto_start:
        await bridge.start()
        logger.info("[INTEGRATION-BRIDGE] All systems started and integrated")

    return bridge
