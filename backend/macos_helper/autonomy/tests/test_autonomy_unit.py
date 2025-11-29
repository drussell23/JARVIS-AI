"""
Unit tests for Phase 3 Autonomy components.

Tests cover:
- Action Registry
- Permission System
- Safety Validator
- Action Learning System
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


# =============================================================================
# ACTION REGISTRY TESTS
# =============================================================================


class TestActionRegistry:
    """Tests for ActionRegistry."""

    def test_action_type_enum(self, action_registry_module):
        """Test ActionType enumeration."""
        ActionType = action_registry_module.ActionType

        # Check some key action types exist
        assert hasattr(ActionType, 'APP_OPEN')
        assert hasattr(ActionType, 'APP_CLOSE')
        assert hasattr(ActionType, 'FILE_OPEN')
        assert hasattr(ActionType, 'SYSTEM_INFO')
        assert hasattr(ActionType, 'VOLUME_SET')
        assert hasattr(ActionType, 'SCREENSHOT')

    def test_action_category_enum(self, action_registry_module):
        """Test ActionCategory enumeration."""
        ActionCategory = action_registry_module.ActionCategory

        assert ActionCategory.APPLICATION.value == "application"
        assert ActionCategory.FILE_SYSTEM.value == "file_system"
        assert ActionCategory.SYSTEM.value == "system"
        assert ActionCategory.SECURITY.value == "security"

    def test_action_risk_level(self, action_registry_module):
        """Test ActionRiskLevel enumeration."""
        ActionRiskLevel = action_registry_module.ActionRiskLevel

        assert ActionRiskLevel.MINIMAL.value == 1
        assert ActionRiskLevel.LOW.value == 2
        assert ActionRiskLevel.MODERATE.value == 3
        assert ActionRiskLevel.HIGH.value == 4
        assert ActionRiskLevel.CRITICAL.value == 5

        # Test requires_confirmation
        assert not ActionRiskLevel.LOW.requires_confirmation()
        assert ActionRiskLevel.HIGH.requires_confirmation()
        assert ActionRiskLevel.CRITICAL.requires_confirmation()

    def test_action_parameter_validation(self, action_registry_module):
        """Test ActionParameter validation."""
        ActionParameter = action_registry_module.ActionParameter

        # Required string parameter
        param = ActionParameter(
            name="app_name",
            description="Application name",
            param_type=str,
            required=True
        )

        valid, error = param.validate("Safari")
        assert valid
        assert error is None

        valid, error = param.validate(None)
        assert not valid
        assert "Required parameter" in error

    def test_action_parameter_range_validation(self, action_registry_module):
        """Test ActionParameter range validation."""
        ActionParameter = action_registry_module.ActionParameter

        param = ActionParameter(
            name="volume",
            description="Volume level",
            param_type=int,
            required=True,
            min_value=0,
            max_value=100
        )

        valid, _ = param.validate(50)
        assert valid

        valid, error = param.validate(150)
        assert not valid
        assert "<=" in error

        valid, error = param.validate(-10)
        assert not valid
        assert ">=" in error

    def test_action_parameter_choices_validation(self, action_registry_module):
        """Test ActionParameter choices validation."""
        ActionParameter = action_registry_module.ActionParameter

        param = ActionParameter(
            name="button",
            description="Mouse button",
            param_type=str,
            required=True,
            choices=["left", "right", "middle"]
        )

        valid, _ = param.validate("left")
        assert valid

        valid, error = param.validate("up")
        assert not valid
        assert "must be one of" in error

    def test_action_metadata_param_validation(self, action_registry_module):
        """Test ActionMetadata parameter validation."""
        ActionMetadata = action_registry_module.ActionMetadata
        ActionParameter = action_registry_module.ActionParameter
        ActionType = action_registry_module.ActionType
        ActionCategory = action_registry_module.ActionCategory
        ActionRiskLevel = action_registry_module.ActionRiskLevel

        metadata = ActionMetadata(
            action_type=ActionType.VOLUME_SET,
            name="Set Volume",
            description="Set system volume",
            category=ActionCategory.MEDIA,
            risk_level=ActionRiskLevel.MINIMAL,
            parameters=[
                ActionParameter(
                    name="level",
                    description="Volume level",
                    param_type=int,
                    required=True,
                    min_value=0,
                    max_value=100
                )
            ]
        )

        valid, errors = metadata.validate_params({"level": 50})
        assert valid
        assert len(errors) == 0

        valid, errors = metadata.validate_params({"level": 150})
        assert not valid
        assert len(errors) == 1

    def test_registry_creation(self, action_registry_module, registry_config):
        """Test ActionRegistry creation."""
        ActionRegistry = action_registry_module.ActionRegistry

        registry = ActionRegistry(config=registry_config)
        assert registry is not None
        assert not registry.is_running

    @pytest.mark.asyncio
    async def test_registry_start_stop(self, action_registry_module, registry_config):
        """Test ActionRegistry start and stop."""
        ActionRegistry = action_registry_module.ActionRegistry

        registry = ActionRegistry(config=registry_config)

        await registry.start()
        assert registry.is_running

        await registry.stop()
        assert not registry.is_running

    def test_registry_default_actions_registered(self, action_registry_module, registry_config):
        """Test that default actions are registered."""
        ActionRegistry = action_registry_module.ActionRegistry
        ActionType = action_registry_module.ActionType

        registry = ActionRegistry(config=registry_config)

        # Check some default actions are registered
        assert registry.get_action(ActionType.APP_OPEN) is not None
        assert registry.get_action(ActionType.VOLUME_SET) is not None
        assert registry.get_action(ActionType.SCREENSHOT) is not None

    def test_registry_get_by_category(self, action_registry_module, registry_config):
        """Test getting actions by category."""
        ActionRegistry = action_registry_module.ActionRegistry
        ActionCategory = action_registry_module.ActionCategory

        registry = ActionRegistry(config=registry_config)

        app_actions = registry.get_by_category(ActionCategory.APPLICATION)
        assert len(app_actions) > 0

        media_actions = registry.get_by_category(ActionCategory.MEDIA)
        assert len(media_actions) > 0

    def test_registry_search_actions(self, action_registry_module, registry_config):
        """Test action search functionality."""
        ActionRegistry = action_registry_module.ActionRegistry
        ActionCategory = action_registry_module.ActionCategory
        ActionRiskLevel = action_registry_module.ActionRiskLevel

        registry = ActionRegistry(config=registry_config)

        # Search by query
        results = registry.search_actions(query="volume")
        assert len(results) > 0

        # Search by category
        results = registry.search_actions(category=ActionCategory.APPLICATION)
        assert all(r.metadata.category == ActionCategory.APPLICATION for r in results)

        # Search by max risk
        results = registry.search_actions(max_risk=ActionRiskLevel.LOW)
        assert all(r.metadata.risk_level.value <= ActionRiskLevel.LOW.value for r in results)

    def test_registry_statistics(self, action_registry_module, registry_config):
        """Test registry statistics."""
        ActionRegistry = action_registry_module.ActionRegistry

        registry = ActionRegistry(config=registry_config)

        stats = registry.get_statistics()
        assert "total_actions" in stats
        assert stats["total_actions"] > 0
        assert "by_category" in stats
        assert "by_risk_level" in stats


# =============================================================================
# PERMISSION SYSTEM TESTS
# =============================================================================


class TestPermissionSystem:
    """Tests for PermissionSystem."""

    def test_permission_level_comparison(self, permission_system_module):
        """Test PermissionLevel comparison."""
        PermissionLevel = permission_system_module.PermissionLevel

        assert PermissionLevel.DENY < PermissionLevel.ASK
        assert PermissionLevel.ASK < PermissionLevel.ALLOW
        assert PermissionLevel.ALLOW < PermissionLevel.AUTO

    def test_permission_creation(self, permission_system_module):
        """Test Permission creation."""
        Permission = permission_system_module.Permission
        PermissionScope = permission_system_module.PermissionScope
        PermissionLevel = permission_system_module.PermissionLevel

        perm = Permission(
            scope=PermissionScope.ACTION,
            level=PermissionLevel.ALLOW,
            target="APP_OPEN",
            reason="Test permission"
        )

        assert perm.is_valid
        assert not perm.is_expired

    def test_permission_expiry(self, permission_system_module):
        """Test Permission expiry."""
        Permission = permission_system_module.Permission
        PermissionScope = permission_system_module.PermissionScope
        PermissionLevel = permission_system_module.PermissionLevel

        # Expired permission
        perm = Permission(
            scope=PermissionScope.ACTION,
            level=PermissionLevel.ALLOW,
            target="APP_OPEN",
            expires_at=datetime.now() - timedelta(hours=1)
        )

        assert perm.is_expired
        assert not perm.is_valid

    def test_permission_system_creation(self, permission_system_module, permission_config):
        """Test PermissionSystem creation."""
        PermissionSystem = permission_system_module.PermissionSystem

        system = PermissionSystem(config=permission_config)
        assert system is not None
        assert not system.is_running

    @pytest.mark.asyncio
    async def test_permission_system_start_stop(self, permission_system_module, permission_config):
        """Test PermissionSystem start and stop."""
        PermissionSystem = permission_system_module.PermissionSystem

        system = PermissionSystem(config=permission_config)

        await system.start()
        assert system.is_running

        await system.stop()
        assert not system.is_running

    @pytest.mark.asyncio
    async def test_permission_check_minimal_risk(self, permission_system_module, permission_config, action_registry_module):
        """Test permission check for minimal risk action."""
        PermissionSystem = permission_system_module.PermissionSystem
        PermissionContext = permission_system_module.PermissionContext
        ActionType = action_registry_module.ActionType
        ActionCategory = action_registry_module.ActionCategory
        ActionRiskLevel = action_registry_module.ActionRiskLevel

        system = PermissionSystem(config=permission_config)
        await system.start()

        try:
            context = PermissionContext(
                action_type=ActionType.VOLUME_GET,
                action_category=ActionCategory.MEDIA,
                risk_level=ActionRiskLevel.MINIMAL,
            )

            decision = await system.check_permission(context)

            # Minimal risk should be auto-approved with default config
            assert decision.allowed
            assert not decision.requires_confirmation

        finally:
            await system.stop()

    @pytest.mark.asyncio
    async def test_permission_check_high_risk(self, permission_system_module, permission_config, action_registry_module):
        """Test permission check for high risk action."""
        PermissionSystem = permission_system_module.PermissionSystem
        PermissionContext = permission_system_module.PermissionContext
        ActionType = action_registry_module.ActionType
        ActionCategory = action_registry_module.ActionCategory
        ActionRiskLevel = action_registry_module.ActionRiskLevel

        system = PermissionSystem(config=permission_config)
        await system.start()

        try:
            context = PermissionContext(
                action_type=ActionType.FILE_DELETE,
                action_category=ActionCategory.FILE_SYSTEM,
                risk_level=ActionRiskLevel.HIGH,
            )

            decision = await system.check_permission(context)

            # High risk should require confirmation
            assert decision.requires_confirmation

        finally:
            await system.stop()

    @pytest.mark.asyncio
    async def test_permission_grant(self, permission_system_module, permission_config, action_registry_module):
        """Test permission grant."""
        PermissionSystem = permission_system_module.PermissionSystem
        PermissionScope = permission_system_module.PermissionScope
        PermissionLevel = permission_system_module.PermissionLevel

        system = PermissionSystem(config=permission_config)
        await system.start()

        try:
            # Grant explicit permission
            perm = system.grant_permission(
                scope=PermissionScope.ACTION,
                target="APP_OPEN",
                level=PermissionLevel.ALLOW,
                reason="Test grant"
            )

            assert perm is not None
            assert perm.level == PermissionLevel.ALLOW

            # Check permissions list
            perms = system.list_permissions()
            assert len(perms) > 0

        finally:
            await system.stop()

    @pytest.mark.asyncio
    async def test_permission_revoke(self, permission_system_module, permission_config):
        """Test permission revoke."""
        PermissionSystem = permission_system_module.PermissionSystem
        PermissionScope = permission_system_module.PermissionScope
        PermissionLevel = permission_system_module.PermissionLevel

        system = PermissionSystem(config=permission_config)
        await system.start()

        try:
            # Grant permission
            system.grant_permission(
                scope=PermissionScope.ACTION,
                target="TEST_ACTION",
                level=PermissionLevel.ALLOW,
            )

            initial_count = len(system.list_permissions())

            # Revoke
            revoked = system.revoke_permission("TEST_ACTION")
            assert revoked > 0

            assert len(system.list_permissions()) < initial_count

        finally:
            await system.stop()


# =============================================================================
# SAFETY VALIDATOR TESTS
# =============================================================================


class TestSafetyValidator:
    """Tests for SafetyValidator."""

    def test_safety_level_enum(self, safety_validator_module):
        """Test SafetyLevel enumeration."""
        SafetyLevel = safety_validator_module.SafetyLevel

        assert SafetyLevel.STRICT.value == "strict"
        assert SafetyLevel.STANDARD.value == "standard"
        assert SafetyLevel.RELAXED.value == "relaxed"

    def test_constraint_type_enum(self, safety_validator_module):
        """Test ConstraintType enumeration."""
        ConstraintType = safety_validator_module.ConstraintType

        assert hasattr(ConstraintType, 'PATH_RESTRICTION')
        assert hasattr(ConstraintType, 'APP_RESTRICTION')
        assert hasattr(ConstraintType, 'RESOURCE_LIMIT')

    def test_safety_constraint_creation(self, safety_validator_module):
        """Test SafetyConstraint creation."""
        SafetyConstraint = safety_validator_module.SafetyConstraint
        ConstraintType = safety_validator_module.ConstraintType
        CheckSeverity = safety_validator_module.CheckSeverity

        constraint = SafetyConstraint(
            name="test_constraint",
            description="Test constraint",
            constraint_type=ConstraintType.CUSTOM,
            severity=CheckSeverity.WARNING,
            condition=lambda d: d.get("allowed", True),
            error_message="Test failed"
        )

        assert constraint.name == "test_constraint"
        assert constraint.enabled

    def test_safety_validator_creation(self, safety_validator_module, safety_config):
        """Test SafetyValidator creation."""
        SafetyValidator = safety_validator_module.SafetyValidator

        validator = SafetyValidator(config=safety_config)
        assert validator is not None
        assert not validator.is_running

    @pytest.mark.asyncio
    async def test_safety_validator_start_stop(self, safety_validator_module, safety_config):
        """Test SafetyValidator start and stop."""
        SafetyValidator = safety_validator_module.SafetyValidator

        validator = SafetyValidator(config=safety_config)

        await validator.start()
        assert validator.is_running

        await validator.stop()
        assert not validator.is_running

    @pytest.mark.asyncio
    async def test_validate_safe_action(self, safety_validator_module, safety_config, action_registry_module):
        """Test validation of a safe action."""
        SafetyValidator = safety_validator_module.SafetyValidator
        ActionType = action_registry_module.ActionType
        ActionMetadata = action_registry_module.ActionMetadata
        ActionCategory = action_registry_module.ActionCategory
        ActionRiskLevel = action_registry_module.ActionRiskLevel

        validator = SafetyValidator(config=safety_config)
        await validator.start()

        try:
            metadata = ActionMetadata(
                action_type=ActionType.APP_FOCUS,
                name="Focus App",
                description="Focus an application",
                category=ActionCategory.APPLICATION,
                risk_level=ActionRiskLevel.MINIMAL,
            )

            result = await validator.validate(
                action_type=ActionType.APP_FOCUS,
                metadata=metadata,
                params={"app_name": "Safari"},
                context={}
            )

            assert result.passed
            assert result.result.value in ["pass", "pass_with_warnings"]

        finally:
            await validator.stop()

    @pytest.mark.asyncio
    async def test_validate_blocked_path(self, safety_validator_module, safety_config, action_registry_module):
        """Test validation blocks system paths."""
        SafetyValidator = safety_validator_module.SafetyValidator
        ActionType = action_registry_module.ActionType
        ActionMetadata = action_registry_module.ActionMetadata
        ActionCategory = action_registry_module.ActionCategory
        ActionRiskLevel = action_registry_module.ActionRiskLevel

        validator = SafetyValidator(config=safety_config)
        await validator.start()

        try:
            metadata = ActionMetadata(
                action_type=ActionType.FILE_DELETE,
                name="Delete File",
                description="Delete a file",
                category=ActionCategory.FILE_SYSTEM,
                risk_level=ActionRiskLevel.HIGH,
            )

            result = await validator.validate(
                action_type=ActionType.FILE_DELETE,
                metadata=metadata,
                params={"file_path": "/System/Library/test.txt"},
                context={}
            )

            # Should fail due to blocked path
            assert not result.passed

        finally:
            await validator.stop()

    @pytest.mark.asyncio
    async def test_risk_assessment(self, safety_validator_module, safety_config, action_registry_module):
        """Test risk assessment."""
        SafetyValidator = safety_validator_module.SafetyValidator
        ActionType = action_registry_module.ActionType
        ActionMetadata = action_registry_module.ActionMetadata
        ActionCategory = action_registry_module.ActionCategory
        ActionRiskLevel = action_registry_module.ActionRiskLevel

        validator = SafetyValidator(config=safety_config)
        await validator.start()

        try:
            metadata = ActionMetadata(
                action_type=ActionType.APP_OPEN,
                name="Open App",
                description="Open an application",
                category=ActionCategory.APPLICATION,
                risk_level=ActionRiskLevel.LOW,
                supports_rollback=True,
            )

            result = await validator.validate(
                action_type=ActionType.APP_OPEN,
                metadata=metadata,
                params={"app_name": "Safari"},
                context={}
            )

            assert result.risk_assessment is not None
            assert 0 <= result.risk_assessment.total_score <= 1

        finally:
            await validator.stop()

    def test_constraint_management(self, safety_validator_module, safety_config):
        """Test constraint management."""
        SafetyValidator = safety_validator_module.SafetyValidator
        SafetyConstraint = safety_validator_module.SafetyConstraint
        ConstraintType = safety_validator_module.ConstraintType
        CheckSeverity = safety_validator_module.CheckSeverity

        validator = SafetyValidator(config=safety_config)

        # Add custom constraint
        custom = SafetyConstraint(
            name="custom_test",
            description="Custom test constraint",
            constraint_type=ConstraintType.CUSTOM,
            severity=CheckSeverity.WARNING,
            condition=lambda d: True,
            error_message="Custom test"
        )

        validator.add_constraint(custom)

        constraints = validator.list_constraints()
        assert any(c.name == "custom_test" for c in constraints)

        # Remove constraint
        removed = validator.remove_constraint("custom_test")
        assert removed

        constraints = validator.list_constraints()
        assert not any(c.name == "custom_test" for c in constraints)


# =============================================================================
# ACTION LEARNING TESTS
# =============================================================================


class TestActionLearning:
    """Tests for ActionLearningSystem."""

    def test_pattern_type_enum(self, action_learning_module):
        """Test PatternType enumeration."""
        PatternType = action_learning_module.PatternType

        assert hasattr(PatternType, 'SUCCESS_RATE')
        assert hasattr(PatternType, 'EXECUTION_TIME')
        assert hasattr(PatternType, 'TIMING_PREFERENCE')

    def test_prediction_confidence(self, action_learning_module):
        """Test PredictionConfidence."""
        PredictionConfidence = action_learning_module.PredictionConfidence

        assert PredictionConfidence.from_score(0.95) == PredictionConfidence.VERY_HIGH
        assert PredictionConfidence.from_score(0.8) == PredictionConfidence.HIGH
        assert PredictionConfidence.from_score(0.6) == PredictionConfidence.MEDIUM
        assert PredictionConfidence.from_score(0.3) == PredictionConfidence.LOW
        assert PredictionConfidence.from_score(0.1) == PredictionConfidence.VERY_LOW

    def test_action_outcome_creation(self, action_learning_module, action_registry_module):
        """Test ActionOutcome creation."""
        ActionOutcome = action_learning_module.ActionOutcome
        ActionType = action_registry_module.ActionType

        outcome = ActionOutcome(
            action_type=ActionType.APP_OPEN,
            success=True,
            execution_time_ms=150.0,
            params={"app_name": "Safari"},
            context={"screen_locked": False}
        )

        assert outcome.success
        assert outcome.execution_time_ms == 150.0

    def test_action_outcome_feature_vector(self, action_learning_module, action_registry_module):
        """Test ActionOutcome feature vector generation."""
        ActionOutcome = action_learning_module.ActionOutcome
        ActionType = action_registry_module.ActionType

        outcome = ActionOutcome(
            action_type=ActionType.APP_OPEN,
            success=True,
            execution_time_ms=150.0,
            params={"app_name": "Safari"},
            context={"screen_locked": False, "focus_mode": True}
        )

        features = outcome.to_feature_vector()

        assert "success" in features
        assert features["success"] == 1.0
        assert "execution_time_ms" in features
        assert "ctx_screen_locked" in features
        assert features["ctx_screen_locked"] == 0.0

    def test_learning_system_creation(self, action_learning_module, learning_config):
        """Test ActionLearningSystem creation."""
        ActionLearningSystem = action_learning_module.ActionLearningSystem

        system = ActionLearningSystem(config=learning_config)
        assert system is not None
        assert not system.is_running

    @pytest.mark.asyncio
    async def test_learning_system_start_stop(self, action_learning_module, learning_config):
        """Test ActionLearningSystem start and stop."""
        ActionLearningSystem = action_learning_module.ActionLearningSystem

        system = ActionLearningSystem(config=learning_config)

        await system.start()
        assert system.is_running

        await system.stop()
        assert not system.is_running

    @pytest.mark.asyncio
    async def test_record_outcome(self, action_learning_module, learning_config, action_registry_module):
        """Test recording outcomes."""
        ActionLearningSystem = action_learning_module.ActionLearningSystem
        ActionOutcome = action_learning_module.ActionOutcome
        ActionType = action_registry_module.ActionType

        system = ActionLearningSystem(config=learning_config)
        await system.start()

        try:
            outcome = ActionOutcome(
                action_type=ActionType.APP_OPEN,
                success=True,
                execution_time_ms=150.0,
                params={"app_name": "Safari"},
                context={}
            )

            await system.record_outcome(outcome)

            stats = system.get_statistics()
            assert stats["total_outcomes_recorded"] == 1

        finally:
            await system.stop()

    @pytest.mark.asyncio
    async def test_predict_success_insufficient_data(self, action_learning_module, learning_config, action_registry_module):
        """Test prediction with insufficient data."""
        ActionLearningSystem = action_learning_module.ActionLearningSystem
        ActionType = action_registry_module.ActionType
        PredictionConfidence = action_learning_module.PredictionConfidence

        system = ActionLearningSystem(config=learning_config)
        await system.start()

        try:
            # No data recorded, should return default prediction
            prediction = await system.predict_success(
                action_type=ActionType.APP_OPEN,
                params={"app_name": "Safari"},
                context={}
            )

            # Should have low confidence due to insufficient data
            assert prediction.confidence == PredictionConfidence.LOW
            assert "Insufficient" in prediction.risk_factors[0]

        finally:
            await system.stop()

    @pytest.mark.asyncio
    async def test_predict_success_with_data(self, action_learning_module, learning_config, action_registry_module):
        """Test prediction with sufficient data."""
        ActionLearningSystem = action_learning_module.ActionLearningSystem
        ActionOutcome = action_learning_module.ActionOutcome
        ActionType = action_registry_module.ActionType

        system = ActionLearningSystem(config=learning_config)
        await system.start()

        try:
            # Record multiple successful outcomes
            for i in range(5):
                outcome = ActionOutcome(
                    action_type=ActionType.APP_OPEN,
                    success=True,
                    execution_time_ms=100 + i * 10,
                    params={"app_name": "Safari"},
                    context={"screen_locked": False}
                )
                await system.record_outcome(outcome)

            # Now predict
            prediction = await system.predict_success(
                action_type=ActionType.APP_OPEN,
                params={"app_name": "Safari"},
                context={"screen_locked": False}
            )

            # Should have high predicted success rate
            assert prediction.predicted_success_rate > 0.5
            assert prediction.similar_outcomes > 0

        finally:
            await system.stop()

    @pytest.mark.asyncio
    async def test_action_summary(self, action_learning_module, learning_config, action_registry_module):
        """Test action summary generation."""
        ActionLearningSystem = action_learning_module.ActionLearningSystem
        ActionOutcome = action_learning_module.ActionOutcome
        ActionType = action_registry_module.ActionType

        system = ActionLearningSystem(config=learning_config)
        await system.start()

        try:
            # Record outcomes with varying success
            for i in range(10):
                outcome = ActionOutcome(
                    action_type=ActionType.APP_OPEN,
                    success=i % 2 == 0,  # 50% success
                    execution_time_ms=100 + i * 10,
                    params={"app_name": "Safari"},
                    context={}
                )
                await system.record_outcome(outcome)

            summary = system.get_action_summary(ActionType.APP_OPEN)

            assert summary["sample_count"] == 10
            assert summary["success_rate"] == 0.5
            assert "avg_execution_time_ms" in summary

        finally:
            await system.stop()

    def test_learning_config_from_env(self, action_learning_module):
        """Test configuration from environment."""
        ActionLearningConfig = action_learning_module.ActionLearningConfig

        # Test with default values
        config = ActionLearningConfig.from_env()
        assert config.enabled
        assert config.min_samples_for_prediction == 10
        assert config.decay_factor == 0.95


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestAutonomyIntegration:
    """Integration tests for autonomy components."""

    @pytest.mark.asyncio
    async def test_registry_and_permission_integration(
        self,
        action_registry_module,
        permission_system_module,
        registry_config,
        permission_config
    ):
        """Test integration between registry and permission system."""
        ActionRegistry = action_registry_module.ActionRegistry
        ActionType = action_registry_module.ActionType
        PermissionSystem = permission_system_module.PermissionSystem
        PermissionContext = permission_system_module.PermissionContext

        registry = ActionRegistry(config=registry_config)
        permission_system = PermissionSystem(config=permission_config)

        await permission_system.start()

        try:
            # Get action metadata from registry
            registered = registry.get_action(ActionType.APP_OPEN)
            assert registered is not None

            metadata = registered.metadata

            # Create permission context from metadata
            context = PermissionContext(
                action_type=metadata.action_type,
                action_category=metadata.category,
                risk_level=metadata.risk_level,
            )

            # Check permission
            decision = await permission_system.check_permission(context)
            assert decision is not None

        finally:
            await permission_system.stop()

    @pytest.mark.asyncio
    async def test_full_validation_pipeline(
        self,
        action_registry_module,
        permission_system_module,
        safety_validator_module,
        registry_config,
        permission_config,
        safety_config
    ):
        """Test full validation pipeline."""
        ActionRegistry = action_registry_module.ActionRegistry
        ActionType = action_registry_module.ActionType
        PermissionSystem = permission_system_module.PermissionSystem
        PermissionContext = permission_system_module.PermissionContext
        SafetyValidator = safety_validator_module.SafetyValidator

        registry = ActionRegistry(config=registry_config)
        permission_system = PermissionSystem(config=permission_config)
        safety_validator = SafetyValidator(config=safety_config)

        await permission_system.start()
        await safety_validator.start()

        try:
            action_type = ActionType.APP_OPEN
            registered = registry.get_action(action_type)
            metadata = registered.metadata
            params = {"app_name": "Safari"}

            # 1. Safety validation
            safety_result = await safety_validator.validate(
                action_type=action_type,
                metadata=metadata,
                params=params,
                context={}
            )
            assert safety_result.passed

            # 2. Permission check
            perm_context = PermissionContext(
                action_type=action_type,
                action_category=metadata.category,
                risk_level=metadata.risk_level,
            )
            decision = await permission_system.check_permission(perm_context)
            assert decision.allowed

        finally:
            await permission_system.stop()
            await safety_validator.stop()


# =============================================================================
# RUN TESTS
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
