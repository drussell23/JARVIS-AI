"""
JARVIS Context Intelligence API
===============================

Context awareness, OCR processing, and intelligent context queries.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/context", tags=["context"])


# Models
class ContextQuery(BaseModel):
    query: str
    include_history: bool = True
    max_history_items: int = 10


class OCRUpdate(BaseModel):
    ocr_text: str
    timestamp: Optional[str] = None
    source: Optional[str] = "screen_capture"
    confidence: Optional[float] = 1.0


class ContextEvent(BaseModel):
    event_type: str
    data: Dict[str, Any]
    timestamp: Optional[str] = None


# In-memory context storage
_context_store = {
    "current_context": {
        "active_app": None,
        "screen_content": None,
        "last_command": None,
        "user_activity": None,
        "timestamp": None
    },
    "ocr_history": [],
    "event_history": [],
    "user_patterns": {
        "common_commands": {},
        "active_hours": {},
        "app_usage": {}
    }
}


def _get_context_summary() -> Dict[str, Any]:
    """Generate a summary of current context"""
    ctx = _context_store["current_context"]
    ocr_history = _context_store["ocr_history"]

    return {
        "active_app": ctx.get("active_app"),
        "has_screen_content": ctx.get("screen_content") is not None,
        "last_command": ctx.get("last_command"),
        "last_update": ctx.get("timestamp"),
        "ocr_history_count": len(ocr_history),
        "recent_activity": ctx.get("user_activity")
    }


@router.post("/query")
async def query_context(request: ContextQuery) -> Dict[str, Any]:
    """
    Query the context intelligence system.

    Ask questions about current context, recent activity, or patterns.
    """
    try:
        query = request.query.lower()
        response = {
            "query": request.query,
            "timestamp": datetime.now().isoformat(),
            "results": {}
        }

        # Handle different query types
        if "working on" in query or "current" in query:
            # What am I working on?
            ctx = _context_store["current_context"]
            response["results"] = {
                "type": "current_activity",
                "active_app": ctx.get("active_app"),
                "recent_content": ctx.get("screen_content", "")[:500] if ctx.get("screen_content") else None,
                "last_command": ctx.get("last_command")
            }

        elif "history" in query or "earlier" in query or "before" in query:
            # What was I doing earlier?
            history = _context_store["ocr_history"][-request.max_history_items:]
            response["results"] = {
                "type": "history",
                "items": [
                    {
                        "timestamp": h.get("timestamp"),
                        "source": h.get("source"),
                        "content_preview": h.get("text", "")[:200]
                    }
                    for h in history
                ]
            }

        elif "pattern" in query or "usually" in query:
            # What do I usually do?
            patterns = _context_store["user_patterns"]
            response["results"] = {
                "type": "patterns",
                "common_commands": dict(sorted(
                    patterns.get("common_commands", {}).items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]),
                "app_usage": patterns.get("app_usage", {}),
                "active_hours": patterns.get("active_hours", {})
            }

        elif "screen" in query or "see" in query or "display" in query:
            # What's on my screen?
            ctx = _context_store["current_context"]
            response["results"] = {
                "type": "screen_content",
                "content": ctx.get("screen_content"),
                "ocr_timestamp": ctx.get("timestamp")
            }

        else:
            # General context
            response["results"] = {
                "type": "general",
                "summary": _get_context_summary(),
                "hint": "Try asking: 'What am I working on?', 'What did I do earlier?', 'What's on my screen?'"
            }

        # Include history if requested
        if request.include_history:
            response["recent_events"] = _context_store["event_history"][-5:]

        return response

    except Exception as e:
        logger.error(f"Context query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_context_summary() -> Dict[str, Any]:
    """
    Get a summary of current context.

    Returns high-level overview of what JARVIS knows about current state.
    """
    try:
        summary = _get_context_summary()

        # Add additional metadata
        summary["event_count"] = len(_context_store["event_history"])
        summary["pattern_data_available"] = bool(_context_store["user_patterns"].get("common_commands"))

        # Time-based context
        now = datetime.now()
        summary["time_context"] = {
            "hour": now.hour,
            "day_of_week": now.strftime("%A"),
            "is_work_hours": 9 <= now.hour <= 17,
            "is_weekend": now.weekday() >= 5
        }

        return {
            "summary": summary,
            "timestamp": now.isoformat()
        }

    except Exception as e:
        logger.error(f"Context summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ocr_update")
async def update_ocr_context(request: OCRUpdate) -> Dict[str, Any]:
    """
    Update context with OCR data from screen capture.

    Called by vision system when new screen content is captured.
    """
    try:
        timestamp = request.timestamp or datetime.now().isoformat()

        # Update current context
        _context_store["current_context"]["screen_content"] = request.ocr_text
        _context_store["current_context"]["timestamp"] = timestamp

        # Add to history
        ocr_entry = {
            "text": request.ocr_text,
            "timestamp": timestamp,
            "source": request.source,
            "confidence": request.confidence
        }
        _context_store["ocr_history"].append(ocr_entry)

        # Keep history limited
        if len(_context_store["ocr_history"]) > 100:
            _context_store["ocr_history"] = _context_store["ocr_history"][-100:]

        # Extract active app if possible
        if request.ocr_text:
            # Simple heuristic - could be enhanced
            first_line = request.ocr_text.split('\n')[0] if request.ocr_text else ""
            _context_store["current_context"]["active_app"] = first_line[:50]

        return {
            "success": True,
            "message": "OCR context updated",
            "history_size": len(_context_store["ocr_history"]),
            "timestamp": timestamp
        }

    except Exception as e:
        logger.error(f"OCR update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/event")
async def record_context_event(event: ContextEvent) -> Dict[str, Any]:
    """
    Record a context event.

    Events can be commands, app switches, user actions, etc.
    """
    try:
        timestamp = event.timestamp or datetime.now().isoformat()

        event_record = {
            "type": event.event_type,
            "data": event.data,
            "timestamp": timestamp
        }

        _context_store["event_history"].append(event_record)

        # Keep history limited
        if len(_context_store["event_history"]) > 500:
            _context_store["event_history"] = _context_store["event_history"][-500:]

        # Update patterns
        if event.event_type == "command":
            cmd = event.data.get("command", "")
            _context_store["user_patterns"]["common_commands"][cmd] = \
                _context_store["user_patterns"]["common_commands"].get(cmd, 0) + 1

            # Track last command
            _context_store["current_context"]["last_command"] = cmd

        elif event.event_type == "app_switch":
            app = event.data.get("app", "")
            _context_store["user_patterns"]["app_usage"][app] = \
                _context_store["user_patterns"]["app_usage"].get(app, 0) + 1
            _context_store["current_context"]["active_app"] = app

        # Track active hours
        hour = datetime.now().hour
        hour_key = str(hour)
        _context_store["user_patterns"]["active_hours"][hour_key] = \
            _context_store["user_patterns"]["active_hours"].get(hour_key, 0) + 1

        return {
            "success": True,
            "event_id": len(_context_store["event_history"]),
            "timestamp": timestamp
        }

    except Exception as e:
        logger.error(f"Context event error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_context_history(
    event_type: Optional[str] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Get context event history.

    Args:
        event_type: Filter by event type (optional)
        limit: Maximum number of events to return
    """
    try:
        history = _context_store["event_history"]

        if event_type:
            history = [e for e in history if e.get("type") == event_type]

        return {
            "events": history[-limit:],
            "total_count": len(history),
            "filtered_by": event_type
        }

    except Exception as e:
        logger.error(f"Context history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns")
async def get_user_patterns() -> Dict[str, Any]:
    """
    Get learned user patterns.

    Returns command frequency, app usage, and active hours.
    """
    try:
        patterns = _context_store["user_patterns"]

        # Sort and limit
        sorted_commands = dict(sorted(
            patterns.get("common_commands", {}).items(),
            key=lambda x: x[1],
            reverse=True
        )[:20])

        sorted_apps = dict(sorted(
            patterns.get("app_usage", {}).items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])

        return {
            "common_commands": sorted_commands,
            "app_usage": sorted_apps,
            "active_hours": patterns.get("active_hours", {}),
            "data_points": sum(patterns.get("common_commands", {}).values())
        }

    except Exception as e:
        logger.error(f"User patterns error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
async def reset_context(
    reset_history: bool = True,
    reset_patterns: bool = False
) -> Dict[str, Any]:
    """
    Reset context data.

    Args:
        reset_history: Clear event and OCR history
        reset_patterns: Clear learned user patterns
    """
    try:
        if reset_history:
            _context_store["ocr_history"] = []
            _context_store["event_history"] = []
            _context_store["current_context"] = {
                "active_app": None,
                "screen_content": None,
                "last_command": None,
                "user_activity": None,
                "timestamp": None
            }

        if reset_patterns:
            _context_store["user_patterns"] = {
                "common_commands": {},
                "active_hours": {},
                "app_usage": {}
            }

        return {
            "success": True,
            "reset_history": reset_history,
            "reset_patterns": reset_patterns,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Context reset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/active-app")
async def get_active_app() -> Dict[str, Any]:
    """Get the currently active application."""
    try:
        ctx = _context_store["current_context"]
        return {
            "active_app": ctx.get("active_app"),
            "last_update": ctx.get("timestamp"),
            "confidence": "high" if ctx.get("timestamp") else "unknown"
        }
    except Exception as e:
        logger.error(f"Active app error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
