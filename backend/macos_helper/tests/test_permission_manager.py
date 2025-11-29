"""
Tests for macOS helper permission manager.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from macos_helper.permission_manager import (
    PermissionType,
    PermissionStatus,
    PermissionResult,
    PermissionOverview,
    PermissionManager,
    PERMISSION_DEFINITIONS,
)


class TestPermissionType:
    """Tests for PermissionType enum."""

    def test_permission_types_exist(self):
        """Test that all permission types exist."""
        assert PermissionType.ACCESSIBILITY
        assert PermissionType.SCREEN_RECORDING
        assert PermissionType.MICROPHONE
        assert PermissionType.FULL_DISK_ACCESS
        assert PermissionType.CALENDAR
        assert PermissionType.REMINDERS
        assert PermissionType.LOCATION
        assert PermissionType.NOTIFICATIONS

    def test_permission_type_values(self):
        """Test permission type string values."""
        assert PermissionType.ACCESSIBILITY.value == "accessibility"
        assert PermissionType.SCREEN_RECORDING.value == "screen_recording"
        assert PermissionType.MICROPHONE.value == "microphone"


class TestPermissionStatus:
    """Tests for PermissionStatus enum."""

    def test_permission_statuses_exist(self):
        """Test that all permission statuses exist."""
        assert PermissionStatus.GRANTED
        assert PermissionStatus.DENIED
        assert PermissionStatus.NOT_DETERMINED
        assert PermissionStatus.RESTRICTED

    def test_permission_status_values(self):
        """Test permission status string values."""
        assert PermissionStatus.GRANTED.value == "granted"
        assert PermissionStatus.DENIED.value == "denied"
        assert PermissionStatus.NOT_DETERMINED.value == "not_determined"


class TestPermissionResult:
    """Tests for PermissionResult dataclass."""

    def test_permission_result_creation(self):
        """Test creating permission result."""
        result = PermissionResult(
            permission_type=PermissionType.ACCESSIBILITY,
            status=PermissionStatus.GRANTED,
        )

        assert result.permission_type == PermissionType.ACCESSIBILITY
        assert result.status == PermissionStatus.GRANTED
        assert result.error is None

    def test_permission_result_with_error(self):
        """Test permission result with error."""
        result = PermissionResult(
            permission_type=PermissionType.MICROPHONE,
            status=PermissionStatus.DENIED,
            error="User denied access",
        )

        assert result.status == PermissionStatus.DENIED
        assert result.error == "User denied access"


class TestPermissionOverview:
    """Tests for PermissionOverview dataclass."""

    def test_permission_overview_creation(self):
        """Test creating permission overview."""
        results = {
            PermissionType.ACCESSIBILITY: PermissionResult(
                permission_type=PermissionType.ACCESSIBILITY,
                status=PermissionStatus.GRANTED,
            ),
            PermissionType.MICROPHONE: PermissionResult(
                permission_type=PermissionType.MICROPHONE,
                status=PermissionStatus.DENIED,
            ),
        }

        overview = PermissionOverview(
            all_required_granted=False,
            required_missing=[PermissionType.MICROPHONE],
            results=results,
        )

        assert not overview.all_required_granted
        assert PermissionType.MICROPHONE in overview.required_missing
        assert len(overview.results) == 2


class TestPermissionDefinitions:
    """Tests for permission definitions."""

    def test_all_permissions_have_definitions(self):
        """Test that all permission types have definitions."""
        for perm_type in PermissionType:
            assert perm_type in PERMISSION_DEFINITIONS, f"Missing definition for {perm_type}"

    def test_definitions_have_required_fields(self):
        """Test that definitions have required fields."""
        for perm_type, definition in PERMISSION_DEFINITIONS.items():
            assert "name" in definition, f"Missing 'name' for {perm_type}"
            assert "description" in definition, f"Missing 'description' for {perm_type}"
            assert "why_needed" in definition, f"Missing 'why_needed' for {perm_type}"
            assert "required" in definition, f"Missing 'required' for {perm_type}"


class TestPermissionManager:
    """Tests for PermissionManager."""

    @pytest.fixture
    def permission_manager(self):
        """Create a permission manager for testing."""
        return PermissionManager()

    def test_permission_manager_creation(self, permission_manager):
        """Test permission manager creation."""
        assert permission_manager is not None
        assert not permission_manager._monitoring

    @pytest.mark.asyncio
    async def test_check_accessibility_permission(self, permission_manager):
        """Test checking accessibility permission."""
        # This will actually check the real permission on the system
        result = await permission_manager.check_accessibility()

        assert isinstance(result, PermissionResult)
        assert result.permission_type == PermissionType.ACCESSIBILITY
        assert result.status in [
            PermissionStatus.GRANTED,
            PermissionStatus.DENIED,
            PermissionStatus.NOT_DETERMINED,
        ]

    @pytest.mark.asyncio
    async def test_check_all_permissions(self, permission_manager):
        """Test checking all permissions."""
        overview = await permission_manager.check_all_permissions()

        assert isinstance(overview, PermissionOverview)
        assert isinstance(overview.all_required_granted, bool)
        assert isinstance(overview.results, dict)

        # Should have results for all permission types
        for perm_type in PermissionType:
            assert perm_type in overview.results

    @pytest.mark.asyncio
    async def test_generate_onboarding_steps(self, permission_manager):
        """Test generating onboarding steps."""
        steps = await permission_manager.generate_onboarding_steps()

        assert isinstance(steps, list)

        # Check structure of steps
        for step in steps:
            assert "permission_type" in step
            assert "name" in step
            assert "description" in step
            assert "why_needed" in step
            assert "settings_url" in step

    def test_get_stats(self, permission_manager):
        """Test getting permission manager stats."""
        stats = permission_manager.get_stats()

        assert "monitoring" in stats
        assert "cache_size" in stats

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, permission_manager):
        """Test starting and stopping permission monitoring."""
        await permission_manager.start_monitoring()
        assert permission_manager._monitoring

        await permission_manager.stop_monitoring()
        assert not permission_manager._monitoring

    @pytest.mark.asyncio
    async def test_open_permission_settings(self, permission_manager):
        """Test opening permission settings."""
        # This test just verifies the method doesn't crash
        # It won't actually open settings in CI
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            await permission_manager.open_permission_settings(PermissionType.ACCESSIBILITY)
            mock_run.assert_called_once()
