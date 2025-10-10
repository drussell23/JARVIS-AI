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
from dataclasses import dataclass
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

        # Handle "can you see" queries - be proactive about explaining
        visibility_keywords = ["can you see", "do you see", "are you seeing", "what do you see"]
        if any(kw in query_lower for kw in visibility_keywords):
            return await self._handle_visibility_query(query_lower, current_space_id)

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
