"""
Dynamic Intent Registry
Load intent patterns from configuration files instead of hardcoding.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, TypedDict

logger = logging.getLogger(__name__)


class IntentDefinition(TypedDict):
    """Schema for intent configuration."""
    label: str
    patterns: list[str]
    examples: list[str]
    embeddings: list[list[float]] | None
    metadata: dict[str, Any]


class IntentRegistry:
    """
    Central registry for intent definitions.
    Loads from JSON/YAML, supports hot-reloading.
    """

    def __init__(self, config_path: Path | None = None):
        self._config_path = config_path
        self._intents: dict[str, IntentDefinition] = {}
        self._pattern_cache: dict[str, list[str]] = {}

        if config_path and config_path.exists():
            self.load_from_file(config_path)

    def load_from_file(self, path: Path) -> None:
        """Load intent definitions from JSON file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)

            self._intents.clear()
            self._pattern_cache.clear()

            for intent_data in data.get("intents", []):
                intent_def = IntentDefinition(
                    label=intent_data["label"],
                    patterns=intent_data.get("patterns", []),
                    examples=intent_data.get("examples", []),
                    embeddings=intent_data.get("embeddings"),
                    metadata=intent_data.get("metadata", {}),
                )
                self._intents[intent_def["label"]] = intent_def
                self._pattern_cache[intent_def["label"]] = intent_def["patterns"]

            logger.info(f"Loaded {len(self._intents)} intent definitions from {path}")

        except Exception as e:
            logger.error(f"Failed to load intent registry from {path}: {e}", exc_info=True)

    def register_intent(
        self,
        label: str,
        patterns: list[str],
        examples: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register intent programmatically."""
        intent_def = IntentDefinition(
            label=label,
            patterns=patterns,
            examples=examples or [],
            embeddings=embeddings,
            metadata=metadata or {},
        )
        self._intents[label] = intent_def
        self._pattern_cache[label] = patterns

        logger.debug(f"Registered intent: {label} with {len(patterns)} patterns")

    def get_intent(self, label: str) -> IntentDefinition | None:
        """Retrieve intent definition."""
        return self._intents.get(label)

    def get_all_patterns(self) -> dict[str, list[str]]:
        """Get all intent patterns for lexical classifier."""
        return self._pattern_cache.copy()

    def get_all_embeddings(self) -> dict[str, list[list[float]]]:
        """Get all intent embeddings for semantic classifier."""
        return {
            label: intent["embeddings"]
            for label, intent in self._intents.items()
            if intent["embeddings"]
        }

    def reload(self) -> None:
        """Hot-reload from config file."""
        if self._config_path:
            self.load_from_file(self._config_path)

    @property
    def intent_labels(self) -> list[str]:
        """Get all registered intent labels."""
        return list(self._intents.keys())

    def export_to_file(self, path: Path) -> None:
        """Export current registry to JSON."""
        data = {"intents": list(self._intents.values())}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported intent registry to {path}")


# Default bootstrap patterns (can be overridden by config)
DEFAULT_FOLLOW_UP_PATTERNS = [
    # Affirmative
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "alright",
    "please", "please do", "go ahead", "proceed", "continue",
    "that would be helpful", "sounds good", "absolutely", "definitely",
    "correct", "right", "exactly", "indeed",

    # Negative
    "no", "nope", "nah", "not now", "maybe later", "skip",
    "never mind", "don't bother", "no thanks",

    # Inquiry
    "tell me more", "show me", "what is it", "what does it say",
    "what's in it", "what do you see", "describe it", "explain",
    "analyze it", "what's happening", "what's there", "details",
    "expand", "elaborate", "break it down",

    # Context references
    "that one", "this", "that", "the one you mentioned",
    "from before", "the previous", "what you just showed",
]

DEFAULT_VISION_PATTERNS = [
    # Terminal/CLI
    "terminal", "command line", "cli", "shell", "bash", "console",
    "what's in my terminal", "check the terminal", "see the output",
    "what does the error say", "command output",

    # Browser/Web
    "browser", "webpage", "web page", "website", "chrome", "safari",
    "what's on the page", "read the page", "see the website",
    "what does it say", "page content",

    # Code/IDE
    "code", "editor", "vscode", "vim", "ide", "file",
    "what's in the file", "read the code", "check my code",
    "see the function", "look at line",

    # General window
    "window", "screen", "display", "what do you see",
    "what's on my screen", "check my screen", "look at",
    "can you see", "visible", "showing",

    # Analysis requests
    "analyze", "examine", "inspect", "review", "check",
    "find errors", "debug", "what's wrong", "any issues",
]


def create_default_registry() -> IntentRegistry:
    """Factory for default intent registry."""
    registry = IntentRegistry()

    registry.register_intent(
        label="follow_up",
        patterns=DEFAULT_FOLLOW_UP_PATTERNS,
        examples=[
            "yes",
            "tell me more about that",
            "show me the details",
            "no thanks",
            "what does it say?",
        ],
        metadata={
            "category": "interaction",
            "priority": 100,  # highest - check first
            "requires_pending_context": True,
        },
    )

    registry.register_intent(
        label="vision",
        patterns=DEFAULT_VISION_PATTERNS,
        examples=[
            "what's in my terminal?",
            "can you see my browser?",
            "check the code in my editor",
            "analyze what's on screen",
        ],
        metadata={
            "category": "vision",
            "priority": 80,
        },
    )

    return registry
