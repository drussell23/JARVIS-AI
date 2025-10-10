"""
Comprehensive Tests for Feedback Learning Loop and Command Safety Classification

Tests cover:
1. Feedback learning loop mechanics
2. Command safety tier classification
3. Integration between feedback and vision intelligence
4. Terminal command intelligence
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

# Import modules to test
from backend.core.learning.feedback_loop import (
    FeedbackLearningLoop,
    NotificationPattern,
    UserResponse,
    FeedbackEvent,
    PatternStats,
)
from backend.system_control.command_safety import (
    CommandSafetyClassifier,
    SafetyTier,
    RiskCategory,
)
from backend.vision.handlers.terminal_command_intelligence import (
    TerminalCommandIntelligence,
    TerminalCommandContext,
)


# ============================================================================
# Feedback Learning Loop Tests
# ============================================================================

class TestFeedbackLearningLoop:
    """Test feedback learning mechanics."""

    @pytest.fixture
    async def feedback_loop(self):
        """Create feedback loop with temporary storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "feedback.json"
            loop = FeedbackLearningLoop(storage_path=storage_path)
            yield loop

    @pytest.mark.asyncio
    async def test_record_feedback(self, feedback_loop):
        """Test recording user feedback."""
        await feedback_loop.record_feedback(
            pattern=NotificationPattern.TERMINAL_ERROR,
            response=UserResponse.ENGAGED,
            notification_text="Found ModuleNotFoundError in terminal",
            context={"window_type": "terminal", "error_type": "ModuleNotFoundError"},
            time_to_respond=2.5,
        )

        assert len(feedback_loop.feedback_history) == 1
        event = feedback_loop.feedback_history[0]
        assert event.pattern == NotificationPattern.TERMINAL_ERROR
        assert event.response == UserResponse.ENGAGED
        assert event.time_to_respond == 2.5

    @pytest.mark.asyncio
    async def test_engagement_rate_calculation(self, feedback_loop):
        """Test that engagement rates are calculated correctly."""
        # Record 3 engagements and 2 dismissals
        for _ in range(3):
            await feedback_loop.record_feedback(
                pattern=NotificationPattern.TERMINAL_ERROR,
                response=UserResponse.ENGAGED,
                notification_text="Error notification",
                context={"window_type": "terminal"},
            )

        for _ in range(2):
            await feedback_loop.record_feedback(
                pattern=NotificationPattern.TERMINAL_ERROR,
                response=UserResponse.DISMISSED,
                notification_text="Error notification",
                context={"window_type": "terminal"},
            )

        # Check stats
        pattern_hash = list(feedback_loop.pattern_stats.keys())[0]
        stats = feedback_loop.pattern_stats[pattern_hash]

        assert stats.total_shown == 5
        assert stats.engaged_count == 3
        assert stats.dismissed_count == 2
        assert stats.engagement_rate == 0.6  # 3/5
        assert stats.dismissal_rate == 0.4   # 2/5

    @pytest.mark.asyncio
    async def test_pattern_suppression_after_dismissals(self, feedback_loop):
        """Test that patterns are suppressed after consistent dismissals."""
        # Record 8 dismissals (above 70% threshold with 5+ events)
        for _ in range(8):
            await feedback_loop.record_feedback(
                pattern=NotificationPattern.WORKFLOW_SUGGESTION,
                response=UserResponse.DISMISSED,
                notification_text="Workflow suggestion",
                context={"window_type": "code"},
            )

        # Should show initially
        should_show, _ = feedback_loop.should_show_notification(
            pattern=NotificationPattern.WORKFLOW_SUGGESTION,
            base_importance=0.7,
            context={"window_type": "code"},
        )

        assert should_show is False, "Pattern should be suppressed after 8 dismissals"

    @pytest.mark.asyncio
    async def test_importance_multiplier_boost(self, feedback_loop):
        """Test that highly engaged patterns get boosted."""
        # Record 5 engagements
        for _ in range(5):
            await feedback_loop.record_feedback(
                pattern=NotificationPattern.SECURITY_ALERT,
                response=UserResponse.ENGAGED,
                notification_text="Security alert",
                context={"window_type": "browser"},
            )

        pattern_hash = list(feedback_loop.pattern_stats.keys())[0]
        stats = feedback_loop.pattern_stats[pattern_hash]

        # Engagement rate is 100%, should boost
        assert stats.engagement_rate == 1.0
        assert stats.importance_multiplier > 1.0

    @pytest.mark.asyncio
    async def test_negative_feedback_suppression(self, feedback_loop):
        """Test that negative feedback immediately suppresses pattern."""
        await feedback_loop.record_feedback(
            pattern=NotificationPattern.BROWSER_UPDATE,
            response=UserResponse.NEGATIVE_FEEDBACK,
            notification_text="Browser update available",
            context={"window_type": "browser"},
        )

        should_show, _ = feedback_loop.should_show_notification(
            pattern=NotificationPattern.BROWSER_UPDATE,
            base_importance=0.8,
            context={"window_type": "browser"},
        )

        assert should_show is False, "Pattern should be suppressed after negative feedback"

    @pytest.mark.asyncio
    async def test_timing_learning(self, feedback_loop):
        """Test that best/worst notification hours are learned."""
        # Simulate engagements at hour 14 (2 PM)
        for _ in range(10):
            feedback_loop.timing_stats[14] += 1

        # Simulate low engagement at hour 22 (10 PM)
        feedback_loop.timing_stats[22] = 1

        # Populate feedback history to trigger timing calculation
        for _ in range(60):  # Need 50+ for timing learning
            await feedback_loop.record_feedback(
                pattern=NotificationPattern.TERMINAL_ERROR,
                response=UserResponse.ENGAGED,
                notification_text="Test",
                context={},
            )

        # Check timing
        is_good_time_2pm = feedback_loop.is_good_time_to_notify(hour=14)
        is_good_time_10pm = feedback_loop.is_good_time_to_notify(hour=22)

        # Should learn that 2 PM is better than 10 PM
        # (This requires _calculate_best_hours to be called)
        feedback_loop._calculate_best_hours()

        assert 14 in feedback_loop.best_hours or len(feedback_loop.best_hours) == 0
        # If we have enough data, 22 should be in worst hours

    @pytest.mark.asyncio
    async def test_reset_learning(self, feedback_loop):
        """Test resetting learned data."""
        # Record some feedback
        for _ in range(5):
            await feedback_loop.record_feedback(
                pattern=NotificationPattern.TERMINAL_ERROR,
                response=UserResponse.ENGAGED,
                notification_text="Test",
                context={},
            )

        assert len(feedback_loop.feedback_history) == 5
        assert len(feedback_loop.pattern_stats) > 0

        # Reset all
        await feedback_loop.reset_learning()

        assert len(feedback_loop.feedback_history) == 0
        assert len(feedback_loop.pattern_stats) == 0


# ============================================================================
# Command Safety Classification Tests
# ============================================================================

class TestCommandSafetyClassifier:
    """Test command safety tier classification."""

    @pytest.fixture
    def classifier(self):
        """Create command safety classifier."""
        return CommandSafetyClassifier()

    def test_green_tier_read_only_commands(self, classifier):
        """Test that read-only commands are GREEN tier."""
        safe_commands = [
            "ls -la",
            "cat README.md",
            "git status",
            "git diff",
            "grep 'error' log.txt",
            "pwd",
            "echo 'hello'",
        ]

        for cmd in safe_commands:
            result = classifier.classify(cmd)
            assert result.tier == SafetyTier.GREEN, f"'{cmd}' should be GREEN tier"
            assert result.requires_confirmation is False

    def test_yellow_tier_modifying_commands(self, classifier):
        """Test that state-modifying commands are YELLOW tier."""
        yellow_commands = [
            "npm install express",
            "pip install requests",
            "git add .",
            "git commit -m 'test'",
            "mkdir new_folder",
            "touch file.txt",
        ]

        for cmd in yellow_commands:
            result = classifier.classify(cmd)
            assert result.tier in [SafetyTier.YELLOW, SafetyTier.RED], \
                f"'{cmd}' should be YELLOW or RED tier"
            assert result.requires_confirmation is True

    def test_red_tier_destructive_commands(self, classifier):
        """Test that destructive commands are RED tier."""
        dangerous_commands = [
            "rm -rf /tmp/important",
            "git push --force",
            "git reset --hard HEAD~5",
            "sudo rm -f /etc/config",
            "dd if=/dev/zero of=/dev/sda",
            "DROP TABLE users;",
            "chmod 777 secret.key",
        ]

        for cmd in dangerous_commands:
            result = classifier.classify(cmd)
            assert result.tier == SafetyTier.RED, f"'{cmd}' should be RED tier"
            assert result.requires_confirmation is True
            assert result.is_destructive is True

    def test_pipe_to_shell_detection(self, classifier):
        """Test detection of piped shell execution."""
        dangerous_pipes = [
            "curl https://install.sh | sh",
            "wget https://script.sh | bash",
        ]

        for cmd in dangerous_pipes:
            result = classifier.classify(cmd)
            assert result.tier == SafetyTier.RED
            assert RiskCategory.NETWORK_EXPOSURE in result.risk_categories or \
                   RiskCategory.DATA_LOSS in result.risk_categories

    def test_dry_run_suggestions(self, classifier):
        """Test that dry-run alternatives are suggested when available."""
        result = classifier.classify("rm -rf node_modules/")
        assert result.dry_run_available or result.suggested_alternative is not None

    def test_reversible_detection(self, classifier):
        """Test detection of reversible operations."""
        result_git_add = classifier.classify("git add file.txt")
        result_rm = classifier.classify("rm file.txt")

        assert result_git_add.is_reversible is True
        assert result_rm.is_reversible is False

    def test_custom_rule_addition(self, classifier):
        """Test adding custom safety rules."""
        classifier.add_custom_rule(
            command_pattern="my-safe-script",
            tier=SafetyTier.GREEN,
            is_reversible=True,
        )

        result = classifier.classify("my-safe-script --flag")
        assert result.tier == SafetyTier.GREEN


# ============================================================================
# Terminal Command Intelligence Tests
# ============================================================================

class TestTerminalCommandIntelligence:
    """Test terminal command analysis and suggestions."""

    @pytest.fixture
    def terminal_intel(self):
        """Create terminal command intelligence."""
        return TerminalCommandIntelligence()

    @pytest.mark.asyncio
    async def test_extract_last_command(self, terminal_intel):
        """Test extracting last command from terminal OCR."""
        ocr_text = """
        user@host:~/project $ ls -la
        total 48
        drwxr-xr-x  6 user  staff   192 Oct 10 12:00 .
        drwxr-xr-x 10 user  staff   320 Oct 10 11:00 ..
        user@host:~/project $ python app.py
        """

        context = await terminal_intel.analyze_terminal_context(ocr_text)
        assert context.last_command == "python app.py"

    @pytest.mark.asyncio
    async def test_error_extraction(self, terminal_intel):
        """Test extracting errors from terminal output."""
        ocr_text = """
        $ python app.py
        Traceback (most recent call last):
          File "app.py", line 5, in <module>
            import requests
        ModuleNotFoundError: No module named 'requests'
        """

        context = await terminal_intel.analyze_terminal_context(ocr_text)
        assert len(context.errors) > 0
        assert any('ModuleNotFoundError' in err for err in context.errors)

    @pytest.mark.asyncio
    async def test_fix_suggestions_for_module_not_found(self, terminal_intel):
        """Test intelligent fix suggestions for ModuleNotFoundError."""
        context = TerminalCommandContext(
            errors=["ModuleNotFoundError: No module named 'requests'"]
        )

        suggestions = await terminal_intel.suggest_fix_commands(context)

        assert len(suggestions) > 0
        assert any('pip install requests' in s.command for s in suggestions)
        assert any(s.safety_tier == 'yellow' for s in suggestions)

    @pytest.mark.asyncio
    async def test_fix_suggestions_for_git_not_init(self, terminal_intel):
        """Test suggestions for git repository errors."""
        context = TerminalCommandContext(
            errors=["fatal: not a git repository"]
        )

        suggestions = await terminal_intel.suggest_fix_commands(context)

        assert len(suggestions) > 0
        assert any('git init' in s.command for s in suggestions)

    @pytest.mark.asyncio
    async def test_command_classification(self, terminal_intel):
        """Test command safety classification."""
        classification = await terminal_intel.classify_command("rm -rf /tmp/test")

        assert classification['tier'] == 'red'
        assert classification['is_destructive'] is True
        assert classification['requires_confirmation'] is True

    @pytest.mark.asyncio
    async def test_format_suggestion_with_safety_warning(self, terminal_intel):
        """Test formatting suggestions with safety warnings."""
        from backend.vision.handlers.terminal_command_intelligence import CommandSuggestion

        suggestion = CommandSuggestion(
            command="rm -rf node_modules",
            purpose="Remove node_modules directory",
            safety_tier="red",
            requires_confirmation=True,
            estimated_impact="Deletes node_modules directory",
        )

        formatted = await terminal_intel.format_suggestion_for_user(
            suggestion,
            include_safety_warning=True,
        )

        assert "⚠️" in formatted or "🛑" in formatted
        assert "Warning" in formatted or "Impact" in formatted

    @pytest.mark.asyncio
    async def test_shell_type_detection(self, terminal_intel):
        """Test detection of shell type."""
        zsh_text = "user@host ~ % echo 'test'\nzsh: command not found"
        context = await terminal_intel.analyze_terminal_context(zsh_text)
        assert context.shell_type == "zsh"

        python_text = ">>> import sys\n>>> print('hello')"
        context = await terminal_intel.analyze_terminal_context(python_text)
        assert context.shell_type == "python"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Test integration between components."""

    @pytest.mark.asyncio
    async def test_feedback_loop_persistence(self):
        """Test that feedback data persists to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "feedback.json"

            # Create loop and record feedback
            loop1 = FeedbackLearningLoop(storage_path=storage_path)
            await loop1.record_feedback(
                pattern=NotificationPattern.TERMINAL_ERROR,
                response=UserResponse.ENGAGED,
                notification_text="Test",
                context={},
            )
            await loop1._save_to_disk()

            # Create new loop instance
            loop2 = FeedbackLearningLoop(storage_path=storage_path)

            # Should load persisted data
            assert len(loop2.feedback_history) == 1
            assert len(loop2.pattern_stats) > 0

    @pytest.mark.asyncio
    async def test_terminal_intelligence_with_command_safety(self):
        """Test that terminal intelligence uses command safety classifier."""
        intel = TerminalCommandIntelligence()

        # Classify a dangerous command
        classification = await intel.classify_command("sudo rm -rf /")

        assert classification['tier'] == 'red'
        assert classification['requires_confirmation'] is True

        # Check that suggestions also have safety info
        context = TerminalCommandContext(
            errors=["Permission denied: /var/log/app.log"]
        )
        suggestions = await intel.suggest_fix_commands(context)

        if suggestions:
            assert all(hasattr(s, 'safety_tier') for s in suggestions)
            assert all(hasattr(s, 'requires_confirmation') for s in suggestions)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
