"""
Unit tests for Phase 2 Real-Time Intelligence components.

Tests cover:
- Screen Context Analyzer
- Proactive Suggestion Engine
- Notification Triage System
- Focus Tracker
- Intelligence Coordinator

Note: Uses fixtures from conftest.py to avoid import issues.
"""

import pytest
import asyncio
from datetime import datetime, timedelta


# =============================================================================
# Screen Context Analyzer Tests
# =============================================================================

class TestActivityClassifier:
    """Tests for activity classification."""

    def test_classify_coding_apps(self, screen_context_module):
        """Test classification of coding applications."""
        ActivityClassifier = screen_context_module.ActivityClassifier
        ActivityType = screen_context_module.ActivityType
        ContextConfidence = screen_context_module.ContextConfidence

        classifier = ActivityClassifier()

        activity, confidence = classifier.classify(
            app_name="Cursor",
            bundle_id="com.todesktop.230313mzl4w4u92",
        )
        assert activity == ActivityType.CODING
        assert confidence in [ContextConfidence.MEDIUM, ContextConfidence.HIGH]

        activity, confidence = classifier.classify(
            app_name="Visual Studio Code",
            bundle_id="com.microsoft.VSCode",
        )
        assert activity == ActivityType.CODING

    def test_classify_communication_apps(self, screen_context_module):
        """Test classification of communication apps."""
        ActivityClassifier = screen_context_module.ActivityClassifier
        ActivityType = screen_context_module.ActivityType

        classifier = ActivityClassifier()

        activity, _ = classifier.classify(
            app_name="Slack",
            bundle_id="com.tinyspeck.slackmacgap",
        )
        assert activity == ActivityType.COMMUNICATION

        activity, _ = classifier.classify(
            app_name="Messages",
            bundle_id="com.apple.MobileSMS",
        )
        assert activity == ActivityType.COMMUNICATION

    def test_classify_meeting_apps(self, screen_context_module):
        """Test classification of meeting apps."""
        ActivityClassifier = screen_context_module.ActivityClassifier
        ActivityType = screen_context_module.ActivityType

        classifier = ActivityClassifier()

        activity, _ = classifier.classify(
            app_name="Zoom",
            bundle_id="us.zoom.xos",
        )
        assert activity == ActivityType.MEETING

    def test_classify_from_window_title(self, screen_context_module):
        """Test classification based on window title."""
        ActivityClassifier = screen_context_module.ActivityClassifier
        ActivityType = screen_context_module.ActivityType

        classifier = ActivityClassifier()

        activity, _ = classifier.classify(
            app_name="Unknown App",
            bundle_id=None,
            window_title="test_file.py - Some Editor",
        )
        assert activity == ActivityType.CODING

    def test_learn_mapping(self, screen_context_module):
        """Test learning new app mappings."""
        ActivityClassifier = screen_context_module.ActivityClassifier
        ActivityType = screen_context_module.ActivityType
        ContextConfidence = screen_context_module.ContextConfidence

        classifier = ActivityClassifier()

        # Unknown app
        activity, _ = classifier.classify(
            app_name="CustomApp",
            bundle_id="com.custom.app",
        )
        assert activity == ActivityType.UNKNOWN

        # Learn mapping
        classifier.learn_mapping("com.custom.app", ActivityType.PRODUCTIVITY)

        # Should now classify correctly
        activity, confidence = classifier.classify(
            app_name="CustomApp",
            bundle_id="com.custom.app",
        )
        assert activity == ActivityType.PRODUCTIVITY
        assert confidence == ContextConfidence.HIGH


class TestScreenContext:
    """Tests for screen context data model."""

    def test_screen_context_creation(self, screen_context_module):
        """Test creating screen context."""
        ScreenContext = screen_context_module.ScreenContext
        ActivityType = screen_context_module.ActivityType

        context = ScreenContext(
            active_app="Cursor",
            active_bundle_id="com.cursor",
            window_title="main.py",
            activity_type=ActivityType.CODING,
        )

        assert context.active_app == "Cursor"
        assert context.activity_type == ActivityType.CODING
        assert context.context_id is not None

    def test_screen_context_to_dict(self, screen_context_module):
        """Test converting context to dictionary."""
        ScreenContext = screen_context_module.ScreenContext
        ActivityType = screen_context_module.ActivityType

        context = ScreenContext(
            active_app="Test",
            activity_type=ActivityType.CODING,
        )

        data = context.to_dict()
        assert "context_id" in data
        assert data["active_app"] == "Test"
        assert data["activity_type"] == "coding"

    def test_screen_context_summary(self, screen_context_module):
        """Test context summary generation."""
        ScreenContext = screen_context_module.ScreenContext
        ActivityType = screen_context_module.ActivityType

        context = ScreenContext(
            active_app="Cursor",
            activity_type=ActivityType.CODING,
            current_task="Writing tests",
        )

        summary = context.get_summary()
        assert "Cursor" in summary
        assert "coding" in summary
        assert "Writing tests" in summary


class TestScreenContextAnalyzer:
    """Tests for the screen context analyzer."""

    def test_analyzer_creation(self, screen_context_module):
        """Test creating analyzer."""
        ScreenContextAnalyzer = screen_context_module.ScreenContextAnalyzer
        ScreenContextConfig = screen_context_module.ScreenContextConfig

        config = ScreenContextConfig(
            capture_interval_seconds=1.0,
            enable_vision_analysis=False,
        )
        analyzer = ScreenContextAnalyzer(config)

        assert not analyzer._running
        assert analyzer.config == config

    @pytest.mark.asyncio
    async def test_analyzer_start_stop(self, screen_context_module):
        """Test starting and stopping analyzer."""
        ScreenContextAnalyzer = screen_context_module.ScreenContextAnalyzer
        ScreenContextConfig = screen_context_module.ScreenContextConfig

        config = ScreenContextConfig(
            capture_interval_seconds=1.0,
            enable_vision_analysis=False,
        )
        analyzer = ScreenContextAnalyzer(config)

        result = await analyzer.start()
        assert result
        assert analyzer._running

        await analyzer.stop()
        assert not analyzer._running

    def test_get_stats(self, screen_context_module):
        """Test getting analyzer stats."""
        ScreenContextAnalyzer = screen_context_module.ScreenContextAnalyzer
        ScreenContextConfig = screen_context_module.ScreenContextConfig

        analyzer = ScreenContextAnalyzer(ScreenContextConfig(enable_vision_analysis=False))
        stats = analyzer.get_stats()

        assert "running" in stats
        assert "analyses_performed" in stats
        assert "errors" in stats


# =============================================================================
# Proactive Suggestion Engine Tests
# =============================================================================

class TestSuggestion:
    """Tests for suggestion data model."""

    def test_suggestion_creation(self, suggestion_engine_module):
        """Test creating a suggestion."""
        Suggestion = suggestion_engine_module.Suggestion
        SuggestionType = suggestion_engine_module.SuggestionType
        SuggestionPriority = suggestion_engine_module.SuggestionPriority

        suggestion = Suggestion(
            suggestion_type=SuggestionType.ACTION,
            priority=SuggestionPriority.HIGH,
            title="Test",
            message="Test message",
        )

        assert suggestion.suggestion_id is not None
        assert suggestion.suggestion_type == SuggestionType.ACTION
        assert suggestion.priority == SuggestionPriority.HIGH

    def test_suggestion_expiration(self, suggestion_engine_module):
        """Test suggestion expiration check."""
        Suggestion = suggestion_engine_module.Suggestion
        SuggestionState = suggestion_engine_module.SuggestionState

        # Expired by time
        suggestion = Suggestion(
            expires_at=datetime.now() - timedelta(minutes=5)
        )
        assert suggestion.is_expired()

        # Expired by state
        suggestion2 = Suggestion()
        suggestion2.state = SuggestionState.EXPIRED
        assert suggestion2.is_expired()

        # Not expired
        suggestion3 = Suggestion(
            expires_at=datetime.now() + timedelta(minutes=5)
        )
        assert not suggestion3.is_expired()

    def test_suggestion_combined_score(self, suggestion_engine_module):
        """Test combined score calculation."""
        Suggestion = suggestion_engine_module.Suggestion

        suggestion = Suggestion(
            confidence=0.8,
            relevance_score=0.6,
            timing_score=0.9,
        )

        score = suggestion.get_combined_score()
        expected = 0.8 * 0.4 + 0.6 * 0.4 + 0.9 * 0.2
        assert abs(score - expected) < 0.001


class TestPatternLearner:
    """Tests for pattern learning."""

    def test_observation_recording(self, suggestion_engine_module):
        """Test recording observations."""
        PatternLearner = suggestion_engine_module.PatternLearner

        learner = PatternLearner()

        learner.observe(
            activity_type="coding",
            app_name="Cursor",
            context={"project": "jarvis"},
        )

        assert len(learner._observations) == 1
        assert learner._observations[0]["activity_type"] == "coding"

    def test_hourly_patterns(self, suggestion_engine_module):
        """Test hourly pattern tracking."""
        PatternLearner = suggestion_engine_module.PatternLearner

        learner = PatternLearner()

        for _ in range(5):
            learner.observe(
                activity_type="coding",
                app_name="Cursor",
                context={},
            )

        current_hour = datetime.now().hour
        assert len(learner._hourly_patterns[current_hour]) == 5


class TestTimingOptimizer:
    """Tests for suggestion timing optimization."""

    def test_rate_limiting(self, suggestion_engine_module):
        """Test rate limiting."""
        TimingOptimizer = suggestion_engine_module.TimingOptimizer
        SuggestionEngineConfig = suggestion_engine_module.SuggestionEngineConfig
        SuggestionPriority = suggestion_engine_module.SuggestionPriority

        # Disable quiet hours to test rate limiting specifically
        config = SuggestionEngineConfig(
            max_suggestions_per_hour=2,
            quiet_hours_start=0,
            quiet_hours_end=0,  # No quiet hours
        )
        optimizer = TimingOptimizer(config)

        for _ in range(3):
            optimizer.record_suggestion_shown()

        is_good, reason = optimizer.is_good_time_to_suggest(
            priority=SuggestionPriority.MEDIUM
        )
        assert not is_good
        assert reason == "rate_limit_hour"


class TestProactiveSuggestionEngine:
    """Tests for the suggestion engine."""

    def test_engine_creation(self, suggestion_engine_module):
        """Test creating engine."""
        ProactiveSuggestionEngine = suggestion_engine_module.ProactiveSuggestionEngine
        SuggestionEngineConfig = suggestion_engine_module.SuggestionEngineConfig

        config = SuggestionEngineConfig(check_interval=0.5)
        engine = ProactiveSuggestionEngine(config)

        assert not engine._running

    @pytest.mark.asyncio
    async def test_engine_start_stop(self, suggestion_engine_module):
        """Test starting and stopping engine."""
        ProactiveSuggestionEngine = suggestion_engine_module.ProactiveSuggestionEngine
        SuggestionEngineConfig = suggestion_engine_module.SuggestionEngineConfig

        config = SuggestionEngineConfig(check_interval=0.5)
        engine = ProactiveSuggestionEngine(config)

        result = await engine.start()
        assert result
        assert engine._running

        await engine.stop()
        assert not engine._running

    def test_update_context(self, suggestion_engine_module):
        """Test context updates."""
        ProactiveSuggestionEngine = suggestion_engine_module.ProactiveSuggestionEngine
        SuggestionEngineConfig = suggestion_engine_module.SuggestionEngineConfig

        engine = ProactiveSuggestionEngine(SuggestionEngineConfig())

        engine.update_context(
            activity_type="coding",
            app_name="Cursor",
            context={"project": "test"},
        )

        assert engine._current_activity == "coding"
        assert engine._current_app == "Cursor"
        assert "project" in engine._current_context

    def test_manual_trigger_suggestion(self, suggestion_engine_module):
        """Test manually triggering suggestions."""
        ProactiveSuggestionEngine = suggestion_engine_module.ProactiveSuggestionEngine
        SuggestionEngineConfig = suggestion_engine_module.SuggestionEngineConfig
        SuggestionType = suggestion_engine_module.SuggestionType
        SuggestionPriority = suggestion_engine_module.SuggestionPriority

        engine = ProactiveSuggestionEngine(SuggestionEngineConfig())

        suggestion = engine.manually_trigger_suggestion(
            suggestion_type=SuggestionType.REMINDER,
            title="Test",
            message="Test reminder",
            priority=SuggestionPriority.HIGH,
        )

        assert suggestion is not None
        assert suggestion.title == "Test"
        assert len(engine._pending_suggestions) == 1


# =============================================================================
# Notification Triage Tests
# =============================================================================

class TestNotificationCategory:
    """Tests for notification categorization."""

    def test_categorize_security(self, notification_triage_module):
        """Test categorizing security notifications."""
        CategoryEngine = notification_triage_module.CategoryEngine
        NotificationCategory = notification_triage_module.NotificationCategory

        engine = CategoryEngine()

        category, confidence = engine.categorize(
            app_name="1Password",
            title="Security Alert",
            body="Login attempt detected",
        )

        assert category == NotificationCategory.SECURITY
        assert confidence >= 0.7

    def test_categorize_communication(self, notification_triage_module):
        """Test categorizing communication notifications."""
        CategoryEngine = notification_triage_module.CategoryEngine
        NotificationCategory = notification_triage_module.NotificationCategory

        engine = CategoryEngine()

        category, _ = engine.categorize(
            app_name="Slack",
            title="New message",
            body="John: Hey there!",
        )

        assert category == NotificationCategory.COMMUNICATION

    def test_categorize_from_content(self, notification_triage_module):
        """Test categorizing from notification content."""
        CategoryEngine = notification_triage_module.CategoryEngine
        NotificationCategory = notification_triage_module.NotificationCategory

        engine = CategoryEngine()

        category, _ = engine.categorize(
            app_name="Unknown",
            title="50% OFF Sale!",
            body="Limited time offer, subscribe now!",
        )

        assert category == NotificationCategory.MARKETING


class TestPriorityScorer:
    """Tests for notification priority scoring."""

    def test_high_priority_apps(self, notification_triage_module):
        """Test high priority app scoring."""
        PriorityScorer = notification_triage_module.PriorityScorer
        TriagedNotification = notification_triage_module.TriagedNotification
        NotificationCategory = notification_triage_module.NotificationCategory
        UrgencyLevel = notification_triage_module.UrgencyLevel

        scorer = PriorityScorer()

        notif = TriagedNotification(
            app_name="1Password",
            title="Security",
            body="Alert",
            category=NotificationCategory.SECURITY,
        )

        score, urgency = scorer.score(notif)
        assert score >= 0.7
        assert urgency in [UrgencyLevel.HIGH, UrgencyLevel.CRITICAL]


class TestSmartBatcher:
    """Tests for notification batching."""

    def test_should_batch_low_priority(self, notification_triage_module):
        """Test that low priority notifications are batched."""
        SmartBatcher = notification_triage_module.SmartBatcher
        NotificationTriageConfig = notification_triage_module.NotificationTriageConfig
        TriagedNotification = notification_triage_module.TriagedNotification
        UrgencyLevel = notification_triage_module.UrgencyLevel

        batcher = SmartBatcher(NotificationTriageConfig())

        notif = TriagedNotification(urgency=UrgencyLevel.LOW)
        assert batcher.should_batch(notif)

    def test_should_not_batch_critical(self, notification_triage_module):
        """Test that critical notifications are not batched."""
        SmartBatcher = notification_triage_module.SmartBatcher
        NotificationTriageConfig = notification_triage_module.NotificationTriageConfig
        TriagedNotification = notification_triage_module.TriagedNotification
        UrgencyLevel = notification_triage_module.UrgencyLevel

        batcher = SmartBatcher(NotificationTriageConfig())

        notif = TriagedNotification(urgency=UrgencyLevel.CRITICAL)
        assert not batcher.should_batch(notif)


class TestFocusGuard:
    """Tests for focus mode guard."""

    def test_dnd_blocks_non_critical(self, notification_triage_module):
        """Test that DND blocks non-critical notifications."""
        FocusGuard = notification_triage_module.FocusGuard
        FocusMode = notification_triage_module.FocusMode
        TriagedNotification = notification_triage_module.TriagedNotification
        UrgencyLevel = notification_triage_module.UrgencyLevel

        guard = FocusGuard()
        guard.set_focus_mode(FocusMode.DND)

        notif = TriagedNotification(urgency=UrgencyLevel.MEDIUM)
        should_deliver, reason = guard.should_deliver(notif)

        assert not should_deliver
        assert reason == "dnd_active"

    def test_critical_always_delivers(self, notification_triage_module):
        """Test that critical notifications always deliver."""
        FocusGuard = notification_triage_module.FocusGuard
        FocusMode = notification_triage_module.FocusMode
        TriagedNotification = notification_triage_module.TriagedNotification
        UrgencyLevel = notification_triage_module.UrgencyLevel

        guard = FocusGuard()
        guard.set_focus_mode(FocusMode.DND)

        notif = TriagedNotification(urgency=UrgencyLevel.CRITICAL)
        should_deliver, reason = guard.should_deliver(notif)

        assert should_deliver
        assert reason == "critical_always"


class TestNotificationTriageSystem:
    """Tests for the notification triage system."""

    def test_triage_creation(self, notification_triage_module):
        """Test creating triage system."""
        NotificationTriageSystem = notification_triage_module.NotificationTriageSystem
        NotificationTriageConfig = notification_triage_module.NotificationTriageConfig

        config = NotificationTriageConfig(check_interval_seconds=0.5)
        triage = NotificationTriageSystem(config)

        assert not triage._running

    @pytest.mark.asyncio
    async def test_triage_start_stop(self, notification_triage_module):
        """Test starting and stopping triage."""
        NotificationTriageSystem = notification_triage_module.NotificationTriageSystem
        NotificationTriageConfig = notification_triage_module.NotificationTriageConfig

        config = NotificationTriageConfig(check_interval_seconds=0.5)
        triage = NotificationTriageSystem(config)

        result = await triage.start()
        assert result
        assert triage._running

        await triage.stop()
        assert not triage._running

    @pytest.mark.asyncio
    async def test_triage_notification(self, notification_triage_module):
        """Test triaging a notification."""
        NotificationTriageSystem = notification_triage_module.NotificationTriageSystem
        NotificationTriageConfig = notification_triage_module.NotificationTriageConfig

        triage = NotificationTriageSystem(NotificationTriageConfig())
        await triage.start()

        result = await triage.triage_notification(
            app_name="Slack",
            title="New message",
            body="Hello world",
        )

        assert result is not None
        assert result.app_name == "Slack"
        assert result.category is not None
        assert result.urgency is not None

        await triage.stop()


# =============================================================================
# Focus Tracker Tests
# =============================================================================

class TestActivityClassificationEngine:
    """Tests for activity classification in focus tracker."""

    def test_classify_productive_apps(self, focus_tracker_module):
        """Test classification of productive apps."""
        ActivityClassificationEngine = focus_tracker_module.ActivityClassificationEngine
        ActivityCategory = focus_tracker_module.ActivityCategory

        engine = ActivityClassificationEngine()

        category, _ = engine.classify("Cursor")
        assert category == ActivityCategory.PRODUCTIVE

        category, _ = engine.classify("Visual Studio Code")
        assert category == ActivityCategory.PRODUCTIVE

    def test_classify_distracting_apps(self, focus_tracker_module):
        """Test classification of distracting apps."""
        ActivityClassificationEngine = focus_tracker_module.ActivityClassificationEngine
        ActivityCategory = focus_tracker_module.ActivityCategory

        engine = ActivityClassificationEngine()

        category, _ = engine.classify("Twitter")
        assert category == ActivityCategory.DISTRACTING

        category, _ = engine.classify("YouTube")
        assert category == ActivityCategory.DISTRACTING


class TestFocusScorer:
    """Tests for focus scoring."""

    def test_score_productive_session(self, focus_tracker_module):
        """Test scoring a productive session."""
        FocusScorer = focus_tracker_module.FocusScorer
        FocusTrackerConfig = focus_tracker_module.FocusTrackerConfig

        scorer = FocusScorer(FocusTrackerConfig())

        score = scorer.score_session(
            productive_minutes=50,
            total_minutes=60,
            context_switches=3,
            distractions=1,
        )

        assert score >= 70

    def test_score_distracted_session(self, focus_tracker_module):
        """Test scoring a distracted session."""
        FocusScorer = focus_tracker_module.FocusScorer
        FocusTrackerConfig = focus_tracker_module.FocusTrackerConfig

        scorer = FocusScorer(FocusTrackerConfig())

        score = scorer.score_session(
            productive_minutes=10,
            total_minutes=60,
            context_switches=30,
            distractions=15,
        )

        assert score <= 50

    def test_focus_state_classification(self, focus_tracker_module):
        """Test focus state classification."""
        FocusScorer = focus_tracker_module.FocusScorer
        FocusTrackerConfig = focus_tracker_module.FocusTrackerConfig
        FocusState = focus_tracker_module.FocusState

        scorer = FocusScorer(FocusTrackerConfig())

        # High switch rate = distracted
        state = scorer.score_focus_state(
            recent_switches=10,
            window_seconds=60,
            in_productive_app=True,
        )
        assert state == FocusState.DISTRACTED

        # Low switch rate in productive app = deep focus
        state = scorer.score_focus_state(
            recent_switches=0,
            window_seconds=120,
            in_productive_app=True,
        )
        assert state == FocusState.DEEP_FOCUS


class TestBreakManager:
    """Tests for break management."""

    def test_no_break_initially(self, focus_tracker_module):
        """Test that no break is needed initially."""
        BreakManager = focus_tracker_module.BreakManager
        FocusTrackerConfig = focus_tracker_module.FocusTrackerConfig
        BreakRecommendation = focus_tracker_module.BreakRecommendation

        manager = BreakManager(FocusTrackerConfig())

        rec, _ = manager.check_break_needed()
        assert rec == BreakRecommendation.NONE

    def test_break_after_long_work(self, focus_tracker_module):
        """Test break recommendation after long work period."""
        BreakManager = focus_tracker_module.BreakManager
        FocusTrackerConfig = focus_tracker_module.FocusTrackerConfig
        BreakRecommendation = focus_tracker_module.BreakRecommendation

        config = FocusTrackerConfig(break_reminder_interval_minutes=30)
        manager = BreakManager(config)

        manager._continuous_work_started = datetime.now() - timedelta(minutes=60)

        rec, message = manager.check_break_needed()
        assert rec in [BreakRecommendation.RECOMMENDED, BreakRecommendation.URGENT]
        assert message != ""


class TestFocusSession:
    """Tests for focus session data model."""

    def test_session_creation(self, focus_tracker_module):
        """Test creating a focus session."""
        FocusSession = focus_tracker_module.FocusSession

        session = FocusSession(
            primary_app="Cursor",
            apps_used=["Cursor", "Terminal"],
        )

        assert session.session_id is not None
        assert session.is_active
        assert session.primary_app == "Cursor"

    def test_session_end(self, focus_tracker_module):
        """Test ending a session."""
        FocusSession = focus_tracker_module.FocusSession

        session = FocusSession()
        session.end_session()

        assert not session.is_active
        assert session.ended_at is not None
        assert session.duration_minutes >= 0


class TestFocusTracker:
    """Tests for the focus tracker."""

    def test_tracker_creation(self, focus_tracker_module):
        """Test creating tracker."""
        FocusTracker = focus_tracker_module.FocusTracker
        FocusTrackerConfig = focus_tracker_module.FocusTrackerConfig

        config = FocusTrackerConfig(update_interval_seconds=0.5)
        tracker = FocusTracker(config)

        assert not tracker._running

    @pytest.mark.asyncio
    async def test_tracker_start_stop(self, focus_tracker_module):
        """Test starting and stopping tracker."""
        FocusTracker = focus_tracker_module.FocusTracker
        FocusTrackerConfig = focus_tracker_module.FocusTrackerConfig

        config = FocusTrackerConfig(
            update_interval_seconds=0.5,
            enable_insights=False,
        )
        tracker = FocusTracker(config)

        result = await tracker.start()
        assert result
        assert tracker._running

        await tracker.stop()
        assert not tracker._running

    def test_get_productivity_score(self, focus_tracker_module):
        """Test getting productivity score."""
        FocusTracker = focus_tracker_module.FocusTracker
        FocusTrackerConfig = focus_tracker_module.FocusTrackerConfig

        tracker = FocusTracker(FocusTrackerConfig())
        score = tracker.get_productivity_score()
        assert 0 <= score <= 100


# =============================================================================
# Intelligence Coordinator Tests
# =============================================================================

class TestIntelligenceConfig:
    """Tests for intelligence configuration."""

    def test_default_config(self, coordinator_module):
        """Test default configuration values."""
        IntelligenceConfig = coordinator_module.IntelligenceConfig

        config = IntelligenceConfig()

        assert config.enable_screen_context is True
        assert config.enable_suggestions is True
        assert config.enable_notification_triage is True
        assert config.enable_focus_tracking is True


class TestComponentStatus:
    """Tests for component status tracking."""

    def test_status_creation(self, coordinator_module):
        """Test creating component status."""
        ComponentStatus = coordinator_module.ComponentStatus

        status = ComponentStatus(name="test_component")

        assert status.name == "test_component"
        assert not status.is_running
        assert status.is_healthy
        assert status.error_count == 0

    def test_status_to_dict(self, coordinator_module):
        """Test converting status to dictionary."""
        ComponentStatus = coordinator_module.ComponentStatus

        status = ComponentStatus(
            name="test",
            is_running=True,
            is_healthy=True,
        )

        data = status.to_dict()
        assert data["name"] == "test"
        assert data["is_running"] is True


class TestIntelligenceCoordinator:
    """Tests for the intelligence coordinator."""

    def test_coordinator_creation(self, coordinator_module):
        """Test creating coordinator."""
        IntelligenceCoordinator = coordinator_module.IntelligenceCoordinator
        IntelligenceConfig = coordinator_module.IntelligenceConfig

        config = IntelligenceConfig(
            enable_screen_context=False,
            enable_suggestions=False,
            enable_notification_triage=False,
            enable_focus_tracking=False,
        )
        coordinator = IntelligenceCoordinator(config)

        assert not coordinator._running

    @pytest.mark.asyncio
    async def test_coordinator_start_stop(self, coordinator_module):
        """Test starting and stopping coordinator."""
        IntelligenceCoordinator = coordinator_module.IntelligenceCoordinator
        IntelligenceConfig = coordinator_module.IntelligenceConfig

        config = IntelligenceConfig(
            enable_screen_context=False,
            enable_suggestions=False,
            enable_notification_triage=False,
            enable_focus_tracking=False,
        )
        coordinator = IntelligenceCoordinator(config)

        result = await coordinator.start()
        assert result
        assert coordinator._running

        await coordinator.stop()
        assert not coordinator._running

    def test_get_component_status(self, coordinator_module):
        """Test getting component status."""
        IntelligenceCoordinator = coordinator_module.IntelligenceCoordinator
        IntelligenceConfig = coordinator_module.IntelligenceConfig

        coordinator = IntelligenceCoordinator(IntelligenceConfig())
        status = coordinator.get_component_status()

        assert "screen_context" in status
        assert "suggestions" in status
        assert "notification_triage" in status
        assert "focus_tracker" in status

    def test_callback_registration(self, coordinator_module):
        """Test registering callbacks."""
        IntelligenceCoordinator = coordinator_module.IntelligenceCoordinator
        IntelligenceConfig = coordinator_module.IntelligenceConfig

        coordinator = IntelligenceCoordinator(IntelligenceConfig())

        async def dummy_callback(data):
            pass

        coordinator.on_suggestion(dummy_callback)
        coordinator.on_notification(dummy_callback)
        coordinator.on_insight(dummy_callback)
        coordinator.on_context_changed(dummy_callback)

        assert len(coordinator._on_suggestion) == 1
        assert len(coordinator._on_notification) == 1
        assert len(coordinator._on_insight) == 1
        assert len(coordinator._on_context_changed) == 1
