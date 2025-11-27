#!/usr/bin/env python3
"""
JARVIS AGI OS - Advanced Startup Greeter

Dynamic, context-aware startup greetings for JARVIS.
Handles system startup, laptop wake events, and provides
personalized greetings based on:
- Time of day
- Day of week
- Owner identity
- System state
- Recent activity
- Weather (if available)
- Calendar events (if available)

Example:
    >>> greeter = await get_startup_greeter()
    >>> await greeter.greet_on_startup()
    # JARVIS: "Good morning, Derek. Ready for your command."

    >>> await greeter.greet_on_wake()
    # JARVIS: "Welcome back, Derek. You have 2 meetings today."
"""

import asyncio
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class GreetingContext(Enum):
    """Context for when greeting is triggered."""
    STARTUP = "startup"           # System startup
    WAKE = "wake"                 # Laptop wake from sleep
    UNLOCK = "unlock"             # Screen unlock
    RETURN = "return"             # User returns after absence
    SCHEDULED = "scheduled"       # Scheduled check-in


class GreetingStyle(Enum):
    """Style of greeting delivery."""
    FORMAL = "formal"             # "Good morning, sir."
    CASUAL = "casual"             # "Hey Derek, what's up?"
    BRIEF = "brief"               # "Online and ready."
    DETAILED = "detailed"         # Full status report


@dataclass
class GreetingConfig:
    """Configuration for startup greeter."""
    enabled: bool = True
    style: GreetingStyle = GreetingStyle.FORMAL
    include_time_greeting: bool = True
    include_status_report: bool = False
    include_calendar_preview: bool = True
    include_weather: bool = False
    max_greeting_length: int = 100  # Characters
    cooldown_seconds: int = 30      # Min time between greetings
    wake_detection_enabled: bool = True


@dataclass
class GreetingResult:
    """Result of a greeting operation."""
    success: bool
    greeting_text: str
    context: GreetingContext
    owner_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    spoken: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)


class StartupGreeter:
    """
    Advanced startup greeter for JARVIS AGI OS.

    Provides dynamic, context-aware greetings that adapt to:
    - Time of day (morning/afternoon/evening/night)
    - Day of week (weekday vs weekend)
    - Owner identity (dynamic name resolution)
    - System state (component status)
    - Recent activity patterns
    """

    def __init__(self, config: Optional[GreetingConfig] = None):
        """Initialize the startup greeter."""
        self.config = config or GreetingConfig()
        self._voice = None
        self._owner_service = None
        self._last_greeting_time: Optional[datetime] = None
        self._greeting_history: List[GreetingResult] = []
        self._wake_monitor_task: Optional[asyncio.Task] = None
        self._initialized = False

        # Greeting templates by time of day
        self._morning_greetings = [
            "Good morning, {name}. Ready for your command.",
            "Good morning, {name}. All systems operational.",
            "Morning, {name}. JARVIS online and at your service.",
            "Good morning, {name}. How may I assist you today?",
        ]

        self._afternoon_greetings = [
            "Good afternoon, {name}. Ready for your command.",
            "Good afternoon, {name}. How can I help?",
            "Afternoon, {name}. JARVIS standing by.",
            "Good afternoon, {name}. At your service.",
        ]

        self._evening_greetings = [
            "Good evening, {name}. Ready for your command.",
            "Good evening, {name}. How may I assist?",
            "Evening, {name}. JARVIS online.",
            "Good evening, {name}. All systems ready.",
        ]

        self._late_night_greetings = [
            "JARVIS online, {name}. Working late?",
            "Good evening, {name}. Burning the midnight oil?",
            "JARVIS online, {name}. Ready when you are.",
            "Evening, {name}. Let me know if you need anything.",
        ]

        self._wake_greetings = [
            "Welcome back, {name}.",
            "JARVIS online. Welcome back, {name}.",
            "Back online, {name}. Ready for your command.",
            "Resuming, {name}. How can I help?",
        ]

        self._brief_greetings = [
            "Online and ready.",
            "JARVIS online.",
            "At your service.",
            "Ready for your command.",
        ]

    async def initialize(self) -> bool:
        """Initialize the greeter with voice and owner services."""
        if self._initialized:
            return True

        try:
            from .realtime_voice_communicator import get_voice_communicator, VoiceMode
            from .owner_identity_service import get_owner_identity

            self._voice = await get_voice_communicator()
            self._owner_service = await get_owner_identity()
            self._VoiceMode = VoiceMode

            # Start wake detection if enabled
            if self.config.wake_detection_enabled:
                self._wake_monitor_task = asyncio.create_task(
                    self._monitor_system_wake()
                )

            self._initialized = True
            logger.info("✅ StartupGreeter initialized")
            return True

        except Exception as e:
            logger.error(f"❌ StartupGreeter initialization failed: {e}")
            return False

    async def _get_owner_name(self) -> str:
        """Get the owner's first name dynamically."""
        if self._owner_service:
            try:
                owner = await self._owner_service.get_current_owner()
                if owner and owner.name:
                    return owner.name.split()[0]
            except Exception as e:
                logger.debug(f"Could not get owner name: {e}")
        return "sir"

    def _get_time_of_day(self) -> str:
        """Get the current time of day category."""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 22:
            return "evening"
        else:
            return "late_night"

    def _select_greeting_template(
        self,
        context: GreetingContext,
        time_of_day: str,
    ) -> str:
        """Select an appropriate greeting template."""
        if self.config.style == GreetingStyle.BRIEF:
            return random.choice(self._brief_greetings)

        if context == GreetingContext.WAKE:
            return random.choice(self._wake_greetings)

        # Select based on time of day
        templates = {
            "morning": self._morning_greetings,
            "afternoon": self._afternoon_greetings,
            "evening": self._evening_greetings,
            "late_night": self._late_night_greetings,
        }

        return random.choice(templates.get(time_of_day, self._morning_greetings))

    async def _build_greeting(
        self,
        context: GreetingContext,
        extras: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a complete greeting message."""
        owner_name = await self._get_owner_name()
        time_of_day = self._get_time_of_day()

        # Get base greeting
        template = self._select_greeting_template(context, time_of_day)
        greeting = template.format(name=owner_name)

        # Add extras if detailed style
        if self.config.style == GreetingStyle.DETAILED and extras:
            if extras.get("meetings_today"):
                count = extras["meetings_today"]
                greeting += f" You have {count} meeting{'s' if count != 1 else ''} today."

            if extras.get("unread_messages"):
                count = extras["unread_messages"]
                greeting += f" {count} unread message{'s' if count != 1 else ''}."

        return greeting

    def _can_greet(self) -> bool:
        """Check if enough time has passed since last greeting."""
        if not self._last_greeting_time:
            return True

        elapsed = (datetime.now() - self._last_greeting_time).total_seconds()
        return elapsed >= self.config.cooldown_seconds

    async def greet(
        self,
        context: GreetingContext = GreetingContext.STARTUP,
        extras: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> GreetingResult:
        """
        Deliver a greeting via voice.

        Args:
            context: The context triggering the greeting
            extras: Additional context data (meetings, messages, etc.)
            force: Bypass cooldown check

        Returns:
            GreetingResult with success status and greeting text
        """
        if not self.config.enabled:
            return GreetingResult(
                success=False,
                greeting_text="",
                context=context,
                owner_name="",
                extras={"reason": "greeter_disabled"}
            )

        if not force and not self._can_greet():
            return GreetingResult(
                success=False,
                greeting_text="",
                context=context,
                owner_name="",
                extras={"reason": "cooldown_active"}
            )

        # Ensure initialized
        if not self._initialized:
            await self.initialize()

        owner_name = await self._get_owner_name()
        greeting_text = await self._build_greeting(context, extras)

        # Speak the greeting
        spoken = False
        if self._voice:
            try:
                await self._voice.speak(
                    greeting_text,
                    mode=self._VoiceMode.NORMAL
                )
                spoken = True
            except Exception as e:
                logger.error(f"Failed to speak greeting: {e}")

        # Record the greeting
        self._last_greeting_time = datetime.now()
        result = GreetingResult(
            success=True,
            greeting_text=greeting_text,
            context=context,
            owner_name=owner_name,
            spoken=spoken,
            extras=extras or {}
        )

        self._greeting_history.append(result)

        # Keep history bounded
        if len(self._greeting_history) > 100:
            self._greeting_history = self._greeting_history[-100:]

        logger.info(f"🎙️ Greeting delivered: '{greeting_text}'")
        return result

    async def greet_on_startup(
        self,
        extras: Optional[Dict[str, Any]] = None,
    ) -> GreetingResult:
        """Deliver a startup greeting."""
        return await self.greet(GreetingContext.STARTUP, extras, force=True)

    async def greet_on_wake(
        self,
        extras: Optional[Dict[str, Any]] = None,
    ) -> GreetingResult:
        """Deliver a wake-from-sleep greeting."""
        return await self.greet(GreetingContext.WAKE, extras)

    async def greet_on_unlock(
        self,
        extras: Optional[Dict[str, Any]] = None,
    ) -> GreetingResult:
        """Deliver a screen unlock greeting."""
        return await self.greet(GreetingContext.UNLOCK, extras)

    async def _monitor_system_wake(self):
        """Monitor for system wake events (macOS)."""
        logger.info("🔄 Starting system wake monitor")

        last_check = datetime.now()

        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds

                current_time = datetime.now()
                time_diff = (current_time - last_check).total_seconds()

                # If more than 30 seconds passed in a 5 second sleep,
                # the system likely woke from sleep
                if time_diff > 30:
                    logger.info(f"🌅 System wake detected (gap: {time_diff:.1f}s)")
                    await self.greet_on_wake()

                last_check = current_time

            except asyncio.CancelledError:
                logger.info("Wake monitor stopped")
                break
            except Exception as e:
                logger.error(f"Wake monitor error: {e}")
                await asyncio.sleep(10)

    async def stop(self):
        """Stop the greeter and clean up."""
        if self._wake_monitor_task:
            self._wake_monitor_task.cancel()
            try:
                await self._wake_monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("StartupGreeter stopped")


# Global instance
_greeter_instance: Optional[StartupGreeter] = None


async def get_startup_greeter(
    config: Optional[GreetingConfig] = None,
) -> StartupGreeter:
    """Get or create the global startup greeter instance."""
    global _greeter_instance

    if _greeter_instance is None:
        _greeter_instance = StartupGreeter(config)
        await _greeter_instance.initialize()

    return _greeter_instance


async def greet_on_startup() -> GreetingResult:
    """Convenience function to deliver startup greeting."""
    greeter = await get_startup_greeter()
    return await greeter.greet_on_startup()


async def greet_on_wake() -> GreetingResult:
    """Convenience function to deliver wake greeting."""
    greeter = await get_startup_greeter()
    return await greeter.greet_on_wake()
