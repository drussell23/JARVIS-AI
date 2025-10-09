"""
Vision Follow-Up Handler Plugin
Handles follow-up responses to vision-related questions.
"""
from __future__ import annotations

import logging
from typing import Any

from backend.core.routing.adaptive_router import (
    BaseRouteHandler,
    RouteConfig,
    RoutingResult,
    HandlerPlugin,
)
from backend.core.models.context_envelope import (
    ContextEnvelope,
    VisionContextPayload,
    InteractionContextPayload,
)
from backend.core.intent.adaptive_classifier import IntentResult

logger = logging.getLogger(__name__)


class VisionFollowUpHandler(BaseRouteHandler):
    """
    Handles follow-up responses to vision questions.
    Dynamically dispatches to specialized handlers based on window type.
    """

    def __init__(self, name: str = "vision_follow_up", priority: int = 80):
        super().__init__(name, priority)
        self._window_handlers: dict[str, callable] = {}

    def register_window_handler(
        self, window_type: str, handler: callable
    ) -> None:
        """Register handler for specific window type."""
        self._window_handlers[window_type] = handler
        logger.debug(f"Registered window handler: {window_type}")

    async def handle(
        self,
        user_input: str,
        intent: IntentResult,
        context: ContextEnvelope | None,
        extras: dict[str, Any],
    ) -> RoutingResult:
        """Route follow-up to appropriate window handler."""
        if not context:
            return RoutingResult(
                success=False,
                response="I don't have any pending context to follow up on. What would you like me to look at?",
            )

        # Extract payload
        if not isinstance(context.payload, (VisionContextPayload, InteractionContextPayload)):
            return RoutingResult(
                success=False,
                response="The pending context doesn't seem to be vision-related.",
            )

        # Determine response type (affirmative/negative/inquiry)
        response_type = self._classify_response(user_input)

        if response_type == "negative":
            return RoutingResult(
                success=True,
                response="No problem! Let me know if you need anything else.",
                metadata={"action": "dismissed"},
            )

        # Handle vision context
        if isinstance(context.payload, VisionContextPayload):
            return await self._handle_vision_context(
                user_input, response_type, context, extras
            )

        # Handle interaction context (linked to vision)
        if isinstance(context.payload, InteractionContextPayload):
            linked_id = context.payload.linked_context_id
            if linked_id:
                # TODO: Retrieve linked vision context from store
                return RoutingResult(
                    success=True,
                    response=f"Following up on interaction context (linked: {linked_id})",
                )

        return RoutingResult(
            success=False,
            response="I'm not sure what you're referring to. Can you be more specific?",
        )

    async def _handle_vision_context(
        self,
        user_input: str,
        response_type: str,
        context: ContextEnvelope[VisionContextPayload],
        extras: dict[str, Any],
    ) -> RoutingResult:
        """Handle vision-specific follow-up."""
        payload = context.payload
        window_type = payload.window_type

        # Get specialized handler
        handler = self._window_handlers.get(window_type)
        if not handler:
            # Generic fallback
            return RoutingResult(
                success=True,
                response=f"I can see your {window_type} window. {payload.summary}",
                metadata={"window_type": window_type, "generic_fallback": True},
            )

        # Delegate to specialized handler
        try:
            result = await handler(user_input, response_type, payload, extras)
            return RoutingResult(
                success=True,
                response=result,
                metadata={"window_type": window_type, "handler": handler.__name__},
            )
        except Exception as e:
            logger.error(f"Window handler for '{window_type}' failed: {e}", exc_info=True)
            return RoutingResult(
                success=False,
                response=f"I had trouble analyzing your {window_type} window. Want me to try again?",
                error=str(e),
            )

    def _classify_response(self, text: str) -> str:
        """
        Classify follow-up response type.
        Returns: "affirmative", "negative", or "inquiry"
        """
        text_lower = text.strip().lower()

        # Negative responses
        negative_patterns = {"no", "nope", "nah", "not now", "skip", "never mind"}
        if any(p in text_lower for p in negative_patterns):
            return "negative"

        # Inquiry responses
        inquiry_patterns = {"tell me", "show me", "what", "describe", "explain", "analyze"}
        if any(p in text_lower for p in inquiry_patterns):
            return "inquiry"

        # Default to affirmative
        return "affirmative"


# Specialized window handlers

async def handle_terminal_follow_up(
    user_input: str,
    response_type: str,
    payload: VisionContextPayload,
    extras: dict[str, Any],
) -> str:
    """Handle terminal window follow-up."""
    from backend.vision.adapters import ocr_text_from_snapshot, extract_errors, suggest_fix

    # Get OCR text (cached or fresh)
    ocr_text = payload.ocr_text
    if not ocr_text:
        ocr_text = await ocr_text_from_snapshot(payload.snapshot_id)

    if not ocr_text:
        return "I couldn't read your Terminal text. Let me take a fresh snapshot and try again."

    # Analyze for errors
    errors = extract_errors(ocr_text)

    if errors:
        # Format error report
        error_report = "\n".join(f"• {err}" for err in errors[:3])  # Top 3 errors
        suggestions = [suggest_fix(err) for err in errors[:3]]
        suggestion_text = "\n".join(f"  → {s}" for s in suggestions if s)

        response = f"**Terminal Analysis:**\n\n{error_report}"
        if suggestion_text:
            response += f"\n\n**Suggestions:**\n{suggestion_text}"

        response += "\n\nWould you like me to help fix any of these?"
        return response

    # No errors - provide summary
    lines = ocr_text.strip().split("\n")
    summary = "\n".join(lines[-10:])  # Last 10 lines

    return f"**Terminal Output (last 10 lines):**\n\n```\n{summary}\n```\n\nEverything looks good! What would you like to do next?"


async def handle_browser_follow_up(
    user_input: str,
    response_type: str,
    payload: VisionContextPayload,
    extras: dict[str, Any],
) -> str:
    """Handle browser window follow-up."""
    from backend.vision.adapters import extract_page_content

    # Extract page content
    content = await extract_page_content(payload.window_id, payload.snapshot_id)

    if not content:
        return "I couldn't read the browser content. Want me to refresh the snapshot?"

    # Format response
    title = content.get("title", "Untitled")
    text = content.get("text", "")[:500]  # First 500 chars
    links = content.get("links", [])[:5]  # Top 5 links

    response = f"**Browser: {title}**\n\n{text}"

    if links:
        links_text = "\n".join(f"• {link}" for link in links)
        response += f"\n\n**Key links:**\n{links_text}"

    return response


async def handle_code_follow_up(
    user_input: str,
    response_type: str,
    payload: VisionContextPayload,
    extras: dict[str, Any],
) -> str:
    """Handle code editor window follow-up."""
    from backend.vision.adapters import analyze_code_window

    # Analyze code
    analysis = await analyze_code_window(payload.window_id, payload.snapshot_id)

    if not analysis:
        return "I couldn't analyze the code. Want me to take another look?"

    # Format response
    file_path = analysis.get("file_path", "Unknown file")
    language = analysis.get("language", "unknown")
    diagnostics = analysis.get("diagnostics", [])

    response = f"**Code: {file_path}** ({language})\n\n"

    if diagnostics:
        diag_text = "\n".join(f"• Line {d['line']}: {d['message']}" for d in diagnostics[:5])
        response += f"**Issues found:**\n{diag_text}\n\nWant me to suggest fixes?"
    else:
        response += "No issues detected! The code looks good."

    return response


async def handle_general_window_follow_up(
    user_input: str,
    response_type: str,
    payload: VisionContextPayload,
    extras: dict[str, Any],
) -> str:
    """Fallback for general windows."""
    ocr_text = payload.ocr_text or "No text detected"
    return f"**Window Analysis:**\n\n{payload.summary}\n\n**Detected text:**\n{ocr_text[:300]}"


# Plugin definition

class VisionFollowUpPlugin(HandlerPlugin):
    """Vision follow-up handler plugin."""

    def __init__(self):
        self._handler = VisionFollowUpHandler()

        # Register window-specific handlers
        self._handler.register_window_handler("terminal", handle_terminal_follow_up)
        self._handler.register_window_handler("browser", handle_browser_follow_up)
        self._handler.register_window_handler("code", handle_code_follow_up)
        self._handler.register_window_handler("general", handle_general_window_follow_up)

    @property
    def routes(self) -> list[RouteConfig]:
        return [
            RouteConfig(
                intent_label="follow_up",
                handler=self._handler,
                requires_context=True,
                context_categories=("VISION", "INTERACTION"),
                min_confidence=0.75,
            )
        ]

    async def on_load(self) -> None:
        logger.info("Vision Follow-Up Plugin loaded")

    async def on_unload(self) -> None:
        logger.info("Vision Follow-Up Plugin unloaded")
