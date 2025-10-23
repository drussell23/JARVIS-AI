#!/usr/bin/env python3
"""
Yabai Spatial Intelligence Engine
==================================

24/7 workspace monitoring and spatial pattern learning using Yabai window manager.

This module provides:
- Real-time Space/Desktop monitoring
- Window position and focus tracking
- App usage pattern detection
- Cross-workspace behavioral analysis
- Spatial intelligence for UAE + SAI integration

Features:
- Monitors all macOS Spaces continuously
- Tracks app locations and movements
- Detects workflow patterns
- Learns Space-specific behaviors
- Feeds spatial data to Learning Database

Author: Derek J. Russell
Date: October 2025
Version: 1.0.0 - 24/7 Spatial Intelligence
"""

import asyncio
import logging
import subprocess
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from pathlib import Path
import calendar

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class WindowInfo:
    """Information about a window"""
    window_id: int
    app_name: str
    title: str
    frame: Dict[str, float]  # {x, y, w, h}
    is_focused: bool
    is_fullscreen: bool
    space_id: int
    stack_index: int

@dataclass
class SpaceInfo:
    """Information about a Space/Desktop"""
    space_id: int
    space_index: int
    space_label: Optional[str]
    is_visible: bool
    is_native_fullscreen: bool
    windows: List[WindowInfo]
    focused_window: Optional[WindowInfo]


@dataclass
class AppUsageSession:
    """Tracks an app usage session"""
    app_name: str
    space_id: int
    start_time: float
    end_time: Optional[float]
    duration: Optional[float]
    window_title: str
    focus_count: int


@dataclass
class SpaceTransition:
    """Records a Space transition"""
    from_space_id: int
    to_space_id: int
    trigger_app: Optional[str]
    timestamp: float
    hour_of_day: int
    day_of_week: int


# ============================================================================
# Yabai Spatial Intelligence Engine
# ============================================================================

class YabaiSpatialIntelligence:
    """
    24/7 Spatial intelligence engine using Yabai

    Continuously monitors:
    - All Spaces/Desktops
    - Window positions and movements
    - App focus and usage patterns
    - Workspace transitions
    - Behavioral patterns
    """

    def __init__(
        self,
        learning_db=None,
        monitoring_interval: float = 5.0,  # Match SAI interval
        enable_24_7_mode: bool = True
    ):
        """
        Initialize Yabai Spatial Intelligence

        Args:
            learning_db: Learning Database instance
            monitoring_interval: Monitoring interval in seconds
            enable_24_7_mode: Enable continuous 24/7 monitoring
        """
        self.learning_db = learning_db
        self.monitoring_interval = monitoring_interval
        self.enable_24_7_mode = enable_24_7_mode

        # State tracking
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None

        # Current state
        self.current_spaces: Dict[int, SpaceInfo] = {}
        self.current_focused_space: Optional[int] = None
        self.previous_focused_space: Optional[int] = None

        # App usage tracking
        self.active_sessions: Dict[str, AppUsageSession] = {}
        self.session_history: deque = deque(maxlen=1000)

        # Space transition tracking
        self.space_transition_history: deque = deque(maxlen=500)

        # Metrics
        self.metrics = {
            'total_space_changes': 0,
            'total_app_switches': 0,
            'total_sessions_tracked': 0,
            'spaces_monitored': 0,
            'windows_tracked': 0,
            'monitoring_cycles': 0
        }

        # Check Yabai availability
        self.yabai_available = self._check_yabai()

        logger.info("[YABAI-SI] Yabai Spatial Intelligence initialized")
        logger.info(f"[YABAI-SI] Yabai available: {self.yabai_available}")
        logger.info(f"[YABAI-SI] 24/7 mode: {self.enable_24_7_mode}")
        logger.info(f"[YABAI-SI] Monitoring interval: {self.monitoring_interval}s")

    def _check_yabai(self) -> bool:
        """Check if Yabai is installed and accessible"""
        try:
            result = subprocess.run(
                ['yabai', '-m', 'query', '--spaces'],
                capture_output=True,
                text=True,
                timeout=2
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"[YABAI-SI] Yabai not available: {e}")
            return False

    async def start_monitoring(self):
        """Start 24/7 spatial monitoring"""
        if self.is_monitoring:
            logger.warning("[YABAI-SI] Already monitoring")
            return

        if not self.yabai_available:
            logger.error("[YABAI-SI] Cannot start monitoring - Yabai not available")
            return

        logger.info("[YABAI-SI] Starting 24/7 spatial monitoring...")
        self.is_monitoring = True

        # Initial scan
        await self._scan_workspace()

        # Start continuous monitoring
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info("[YABAI-SI] ✅ 24/7 spatial monitoring active")

    async def stop_monitoring(self):
        """Stop spatial monitoring"""
        if not self.is_monitoring:
            return

        logger.info("[YABAI-SI] Stopping spatial monitoring...")
        self.is_monitoring = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        # Close any active sessions
        await self._close_all_sessions()

        logger.info("[YABAI-SI] ✅ Spatial monitoring stopped")

    async def _monitoring_loop(self):
        """Main 24/7 monitoring loop"""
        logger.info("[YABAI-SI] 24/7 monitoring loop started")

        while self.is_monitoring:
            try:
                # Scan workspace
                await self._scan_workspace()

                # Track patterns
                await self._track_patterns()

                # Store to Learning DB
                if self.learning_db:
                    await self._store_to_learning_db()

                self.metrics['monitoring_cycles'] += 1

                # Sleep until next cycle
                await asyncio.sleep(self.monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[YABAI-SI] Error in monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(self.monitoring_interval)

    async def _scan_workspace(self):
        """Scan entire workspace state"""
        try:
            # Query all Spaces
            spaces_data = await self._query_spaces()

            # Query all windows
            windows_data = await self._query_windows()

            # Build SpaceInfo objects
            new_spaces = {}

            for space in spaces_data:
                space_id = space['id']
                space_windows = [
                    w for w in windows_data if w['space'] == space_id
                ]

                # Convert to WindowInfo
                window_infos = []
                focused_window = None

                for w in space_windows:
                    window_info = WindowInfo(
                        window_id=w['id'],
                        app_name=w['app'],
                        title=w.get('title', ''),
                        frame=w['frame'],
                        is_focused=w.get('has-focus', False),
                        is_fullscreen=w.get('is-native-fullscreen', False),
                        space_id=space_id,
                        stack_index=w.get('stack-index', 0)
                    )
                    window_infos.append(window_info)

                    if window_info.is_focused:
                        focused_window = window_info

                space_info = SpaceInfo(
                    space_id=space_id,
                    space_index=space['index'],
                    space_label=space.get('label'),
                    is_visible=space.get('is-visible', False),
                    is_native_fullscreen=space.get('is-native-fullscreen', False),
                    windows=window_infos,
                    focused_window=focused_window
                )

                new_spaces[space_id] = space_info

            # Update state
            self.previous_focused_space = self.current_focused_space
            self.current_spaces = new_spaces

            # Find currently focused Space
            for space_id, space_info in new_spaces.items():
                if space_info.is_visible:
                    self.current_focused_space = space_id
                    break

            # Detect Space transition
            if (self.previous_focused_space is not None and
                self.current_focused_space is not None and
                self.previous_focused_space != self.current_focused_space):
                await self._handle_space_transition(
                    self.previous_focused_space,
                    self.current_focused_space
                )

            # Update metrics
            self.metrics['spaces_monitored'] = len(new_spaces)
            self.metrics['windows_tracked'] = sum(len(s.windows) for s in new_spaces.values())

        except Exception as e:
            logger.error(f"[YABAI-SI] Error scanning workspace: {e}")

    async def _query_spaces(self) -> List[Dict]:
        """Query Yabai for all Spaces"""
        try:
            result = await asyncio.create_subprocess_exec(
                'yabai', '-m', 'query', '--spaces',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                return json.loads(stdout.decode())
            else:
                logger.error(f"[YABAI-SI] Failed to query spaces: {stderr.decode()}")
                return []

        except Exception as e:
            logger.error(f"[YABAI-SI] Error querying spaces: {e}")
            return []

    async def _query_windows(self) -> List[Dict]:
        """Query Yabai for all windows"""
        try:
            result = await asyncio.create_subprocess_exec(
                'yabai', '-m', 'query', '--windows',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                return json.loads(stdout.decode())
            else:
                logger.error(f"[YABAI-SI] Failed to query windows: {stderr.decode()}")
                return []

        except Exception as e:
            logger.error(f"[YABAI-SI] Error querying windows: {e}")
            return []

    async def _handle_space_transition(self, from_space: int, to_space: int):
        """Handle Space transition"""
        now = time.time()
        dt = datetime.now()

        # Determine trigger app (what was focused before transition)
        trigger_app = None
        if from_space in self.current_spaces:
            from_space_info = self.current_spaces[from_space]
            if from_space_info.focused_window:
                trigger_app = from_space_info.focused_window.app_name

        # Create transition record
        transition = SpaceTransition(
            from_space_id=from_space,
            to_space_id=to_space,
            trigger_app=trigger_app,
            timestamp=now,
            hour_of_day=dt.hour,
            day_of_week=dt.weekday()
        )

        self.space_transition_history.append(transition)
        self.metrics['total_space_changes'] += 1

        logger.info(f"[YABAI-SI] Space transition: {from_space} → {to_space} (trigger: {trigger_app})")

    async def _track_patterns(self):
        """Track usage patterns from current state"""
        now = time.time()
        dt = datetime.now()

        # Track app usage in each Space
        for space_id, space_info in self.current_spaces.items():
            if space_info.focused_window:
                app_name = space_info.focused_window.app_name
                session_key = f"{app_name}_{space_id}"

                if session_key not in self.active_sessions:
                    # Start new session
                    session = AppUsageSession(
                        app_name=app_name,
                        space_id=space_id,
                        start_time=now,
                        end_time=None,
                        duration=None,
                        window_title=space_info.focused_window.title,
                        focus_count=1
                    )
                    self.active_sessions[session_key] = session
                    logger.debug(f"[YABAI-SI] Started session: {app_name} on Space {space_id}")
                else:
                    # Update existing session
                    self.active_sessions[session_key].focus_count += 1

    async def _close_all_sessions(self):
        """Close all active app usage sessions"""
        now = time.time()

        for session_key, session in list(self.active_sessions.items()):
            session.end_time = now
            session.duration = now - session.start_time
            self.session_history.append(session)
            self.metrics['total_sessions_tracked'] += 1

        self.active_sessions.clear()

    async def _store_to_learning_db(self):
        """Store spatial intelligence data to Learning Database"""
        if not self.learning_db:
            return

        try:
            # Store workspace usage
            await self._store_workspace_usage()

            # Store app usage patterns
            await self._store_app_usage_patterns()

            # Store Space transitions
            await self._store_space_transitions()

            # Store temporal patterns (including leap year support!)
            await self._store_temporal_patterns()

        except Exception as e:
            logger.error(f"[YABAI-SI] Error storing to Learning DB: {e}")

    async def _store_workspace_usage(self):
        """Store workspace usage data"""
        now = datetime.now()

        for space_id, space_info in self.current_spaces.items():
            if space_info.focused_window:
                try:
                    async with self.learning_db.db.cursor() as cursor:
                        await cursor.execute("""
                            INSERT INTO workspace_usage
                            (space_id, space_label, app_name, window_title, window_position,
                             focus_duration_seconds, timestamp, day_of_week, hour_of_day,
                             is_fullscreen, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            space_id,
                            space_info.space_label,
                            space_info.focused_window.app_name,
                            space_info.focused_window.title,
                            json.dumps(space_info.focused_window.frame),
                            self.monitoring_interval,  # Approximation
                            now,
                            now.weekday(),
                            now.hour,
                            space_info.focused_window.is_fullscreen,
                            json.dumps({'space_index': space_info.space_index})
                        ))

                    await self.learning_db.db.commit()

                except Exception as e:
                    logger.error(f"[YABAI-SI] Error storing workspace usage: {e}")

    async def _store_app_usage_patterns(self):
        """Store or update app usage patterns"""
        now = datetime.now()

        for space_id, space_info in self.current_spaces.items():
            if space_info.focused_window:
                app_name = space_info.focused_window.app_name

                try:
                    async with self.learning_db.db.cursor() as cursor:
                        # Check if pattern exists
                        await cursor.execute("""
                            SELECT * FROM app_usage_patterns
                            WHERE app_name = ? AND space_id = ?
                            AND typical_time_of_day = ? AND typical_day_of_week = ?
                        """, (app_name, space_id, now.hour, now.weekday()))

                        existing = await cursor.fetchone()

                        if existing:
                            # Update existing pattern
                            await cursor.execute("""
                                UPDATE app_usage_patterns SET
                                    usage_frequency = usage_frequency + 1,
                                    total_usage_time = total_usage_time + ?,
                                    last_used = ?,
                                    confidence = MIN(confidence + 0.01, 0.95)
                                WHERE app_name = ? AND space_id = ?
                                AND typical_time_of_day = ? AND typical_day_of_week = ?
                            """, (
                                self.monitoring_interval,
                                now,
                                app_name,
                                space_id,
                                now.hour,
                                now.weekday()
                            ))
                        else:
                            # Insert new pattern
                            await cursor.execute("""
                                INSERT INTO app_usage_patterns
                                (app_name, space_id, usage_frequency, avg_session_duration,
                                 total_usage_time, typical_time_of_day, typical_day_of_week,
                                 last_used, confidence, metadata)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                app_name,
                                space_id,
                                1,
                                self.monitoring_interval,
                                self.monitoring_interval,
                                now.hour,
                                now.weekday(),
                                now,
                                0.5,
                                json.dumps({})
                            ))

                    await self.learning_db.db.commit()

                except Exception as e:
                    logger.error(f"[YABAI-SI] Error storing app usage pattern: {e}")

    async def _store_space_transitions(self):
        """Store Space transition data"""
        if not self.space_transition_history:
            return

        # Process recent transitions
        transitions_to_store = list(self.space_transition_history)[-10:]  # Last 10

        for transition in transitions_to_store:
            try:
                async with self.learning_db.db.cursor() as cursor:
                    await cursor.execute("""
                        INSERT INTO space_transitions
                        (from_space_id, to_space_id, trigger_app, trigger_action,
                         frequency, timestamp, hour_of_day, day_of_week, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT DO UPDATE SET
                            frequency = frequency + 1
                    """, (
                        transition.from_space_id,
                        transition.to_space_id,
                        transition.trigger_app,
                        'space_change',
                        1,
                        datetime.fromtimestamp(transition.timestamp),
                        transition.hour_of_day,
                        transition.day_of_week,
                        json.dumps({})
                    ))

                await self.learning_db.db.commit()

            except Exception as e:
                logger.error(f"[YABAI-SI] Error storing space transition: {e}")

    async def _store_temporal_patterns(self):
        """Store temporal patterns including leap year support"""
        now = datetime.now()
        is_leap = calendar.isleap(now.year)

        for space_id, space_info in self.current_spaces.items():
            if space_info.focused_window:
                app_name = space_info.focused_window.app_name

                try:
                    async with self.learning_db.db.cursor() as cursor:
                        await cursor.execute("""
                            INSERT INTO temporal_patterns
                            (pattern_type, time_of_day, day_of_week, day_of_month,
                             month_of_year, is_leap_year, action_type, target,
                             frequency, confidence, last_occurrence, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT DO NOTHING
                        """, (
                            'app_usage',
                            now.hour,
                            now.weekday(),
                            now.day,
                            now.month,
                            is_leap,
                            'focus_app',
                            app_name,
                            1,
                            0.5,
                            now,
                            json.dumps({'space_id': space_id})
                        ))

                    await self.learning_db.db.commit()

                except Exception as e:
                    logger.error(f"[YABAI-SI] Error storing temporal pattern: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get spatial intelligence metrics"""
        return {
            **self.metrics,
            'yabai_available': self.yabai_available,
            'is_monitoring': self.is_monitoring,
            'active_sessions': len(self.active_sessions),
            'current_focused_space': self.current_focused_space
        }


# ============================================================================
# Singleton Instance
# ============================================================================

_yabai_instance: Optional[YabaiSpatialIntelligence] = None


async def get_yabai_intelligence(
    learning_db=None,
    monitoring_interval: float = 5.0,
    enable_24_7_mode: bool = True
) -> YabaiSpatialIntelligence:
    """
    Get singleton Yabai Spatial Intelligence instance

    Args:
        learning_db: Learning Database instance
        monitoring_interval: Monitoring interval
        enable_24_7_mode: Enable 24/7 monitoring

    Returns:
        YabaiSpatialIntelligence instance
    """
    global _yabai_instance

    if _yabai_instance is None:
        _yabai_instance = YabaiSpatialIntelligence(
            learning_db=learning_db,
            monitoring_interval=monitoring_interval,
            enable_24_7_mode=enable_24_7_mode
        )

    return _yabai_instance


# ============================================================================
# Example Usage
# ============================================================================

async def main():
    """Example usage"""
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 80)
    print("Yabai Spatial Intelligence - 24/7 Monitoring")
    print("=" * 80)

    # Initialize
    yabai = await get_yabai_intelligence()

    # Start monitoring
    await yabai.start_monitoring()

    # Monitor for 30 seconds
    print("\nMonitoring workspace for 30 seconds...")
    await asyncio.sleep(30)

    # Get metrics
    metrics = yabai.get_metrics()
    print(f"\n📊 Metrics:")
    print(f"   Spaces monitored: {metrics['spaces_monitored']}")
    print(f"   Windows tracked: {metrics['windows_tracked']}")
    print(f"   Space changes: {metrics['total_space_changes']}")
    print(f"   Monitoring cycles: {metrics['monitoring_cycles']}")

    # Stop
    await yabai.stop_monitoring()

    print("\n" + "=" * 80)
    print("✅ Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
