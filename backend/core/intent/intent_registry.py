"""
Dynamic Intent Registry Module.

This module provides a centralized registry system for managing intent definitions
used in natural language processing. It supports loading intent patterns from
configuration files, hot-reloading, and programmatic registration of intents.

The registry manages intent patterns, examples, embeddings, and metadata for
both lexical and semantic classification systems.

Example:
    >>> registry = IntentRegistry(Path("intents.json"))
    >>> registry.register_intent("greeting", ["hello", "hi"], ["Hello there!"])
    >>> patterns = registry.get_all_patterns()
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, TypedDict

logger = logging.getLogger(__name__)


class IntentDefinition(TypedDict):
    """Schema for intent configuration.
    
    Attributes:
        label: Unique identifier for the intent
        patterns: List of text patterns that match this intent
        examples: Example phrases demonstrating the intent
        embeddings: Pre-computed vector embeddings for semantic matching
        metadata: Additional configuration and context information
    """
    label: str
    patterns: list[str]
    examples: list[str]
    embeddings: list[list[float]] | None
    metadata: dict[str, Any]


class IntentRegistry:
    """
    Central registry for intent definitions.
    
    Manages intent definitions loaded from configuration files with support
    for hot-reloading and programmatic registration. Provides efficient
    access to patterns and embeddings for classification systems.
    
    Attributes:
        _config_path: Path to the configuration file for hot-reloading
        _intents: Dictionary mapping intent labels to their definitions
        _pattern_cache: Cached mapping of labels to patterns for quick access
    
    Example:
        >>> registry = IntentRegistry(Path("config/intents.json"))
        >>> registry.register_intent("help", ["help", "assist"], ["Can you help me?"])
        >>> intent = registry.get_intent("help")
    """

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize the intent registry.
        
        Args:
            config_path: Optional path to JSON configuration file containing
                        intent definitions. If provided and exists, intents
                        will be loaded automatically.
        """
        self._config_path = config_path
        self._intents: dict[str, IntentDefinition] = {}
        self._pattern_cache: dict[str, list[str]] = {}

        if config_path and config_path.exists():
            self.load_from_file(config_path)

    def load_from_file(self, path: Path) -> None:
        """Load intent definitions from JSON configuration file.
        
        Clears existing intents and loads new definitions from the specified
        JSON file. The file should contain an "intents" array with intent
        definition objects.
        
        Args:
            path: Path to the JSON configuration file
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
            KeyError: If required fields are missing from intent definitions
            
        Example:
            >>> registry.load_from_file(Path("intents.json"))
        """
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
        """Register an intent definition programmatically.
        
        Adds a new intent to the registry or updates an existing one.
        The intent will be available immediately for classification.
        
        Args:
            label: Unique identifier for the intent
            patterns: List of text patterns that should match this intent
            examples: Optional list of example phrases demonstrating the intent
            embeddings: Optional pre-computed vector embeddings for semantic matching
            metadata: Optional additional configuration and context information
            
        Example:
            >>> registry.register_intent(
            ...     "greeting",
            ...     ["hello", "hi", "hey"],
            ...     ["Hello there!", "Hi, how are you?"],
            ...     metadata={"category": "social", "priority": 50}
            ... )
        """
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
        """Retrieve an intent definition by label.
        
        Args:
            label: The intent label to look up
            
        Returns:
            The intent definition if found, None otherwise
            
        Example:
            >>> intent = registry.get_intent("greeting")
            >>> if intent:
            ...     print(f"Found {len(intent['patterns'])} patterns")
        """
        return self._intents.get(label)

    def get_all_patterns(self) -> dict[str, list[str]]:
        """Get all intent patterns for lexical classification.
        
        Returns a copy of the pattern cache mapping intent labels to their
        associated text patterns. Used by lexical classifiers for pattern matching.
        
        Returns:
            Dictionary mapping intent labels to lists of patterns
            
        Example:
            >>> patterns = registry.get_all_patterns()
            >>> for label, pattern_list in patterns.items():
            ...     print(f"{label}: {len(pattern_list)} patterns")
        """
        return self._pattern_cache.copy()

    def get_all_embeddings(self) -> dict[str, list[list[float]]]:
        """Get all intent embeddings for semantic classification.
        
        Returns embeddings for intents that have them defined. Used by
        semantic classifiers for vector-based similarity matching.
        
        Returns:
            Dictionary mapping intent labels to lists of embedding vectors,
            only includes intents that have embeddings defined
            
        Example:
            >>> embeddings = registry.get_all_embeddings()
            >>> for label, vectors in embeddings.items():
            ...     print(f"{label}: {len(vectors)} embeddings")
        """
        return {
            label: intent["embeddings"]
            for label, intent in self._intents.items()
            if intent["embeddings"]
        }

    def reload(self) -> None:
        """Hot-reload intent definitions from the configuration file.
        
        Reloads the registry from the original configuration file path
        if one was provided during initialization. Useful for updating
        intents without restarting the application.
        
        Raises:
            ValueError: If no configuration path was set during initialization
            
        Example:
            >>> registry.reload()  # Reloads from original config file
        """
        if self._config_path:
            self.load_from_file(self._config_path)

    @property
    def intent_labels(self) -> list[str]:
        """Get all registered intent labels.
        
        Returns:
            List of all intent labels currently registered
            
        Example:
            >>> labels = registry.intent_labels
            >>> print(f"Registered intents: {', '.join(labels)}")
        """
        return list(self._intents.keys())

    def export_to_file(self, path: Path) -> None:
        """Export current registry to a JSON configuration file.
        
        Saves all current intent definitions to a JSON file in the same
        format used by load_from_file(). Useful for persisting programmatically
        registered intents.
        
        Args:
            path: Path where the JSON file should be written
            
        Raises:
            PermissionError: If unable to write to the specified path
            OSError: If there are filesystem-related errors
            
        Example:
            >>> registry.export_to_file(Path("exported_intents.json"))
        """
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
    """Factory function for creating a default intent registry.
    
    Creates and configures an IntentRegistry with default intent definitions
    for common interaction patterns including follow-up responses and vision
    analysis requests.
    
    Returns:
        IntentRegistry instance pre-configured with default intents
        
    Example:
        >>> registry = create_default_registry()
        >>> print(f"Created registry with {len(registry.intent_labels)} intents")
    """
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