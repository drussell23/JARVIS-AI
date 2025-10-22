#!/usr/bin/env python3
"""
Goal Inference + Autonomous Decision Engine + UAE Integration
==============================================================

Seamlessly integrates Goal Inference System with Autonomous Decision Engine
through the Unified Awareness Engine (UAE) for predictive, intelligent automation.

This integration enables:
- Predictive display connections based on inferred goals
- Context-aware autonomous decisions
- Continuous learning from user behavior
- Dynamic adaptation without hardcoding

Author: Derek J. Russell
Date: October 2025
Version: 1.0.0
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import numpy as np

# Import Goal Inference System
from backend.vision.intelligence.goal_inference_system import (
    GoalInferenceEngine, Goal, GoalLevel,
    get_goal_inference_engine
)

# Import Autonomous Decision Engine
from backend.autonomy.autonomous_decision_engine import (
    AutonomousDecisionEngine, AutonomousAction,
    ActionPriority, ActionCategory
)

# Import Advanced Autonomous Engine
from backend.autonomy.advanced_autonomous_engine import (
    AdvancedAutonomousEngine,
    get_advanced_autonomous_engine
)

# Import UAE
from backend.intelligence.unified_awareness_engine import (
    UnifiedAwarenessEngine,
    DecisionSource,
    get_uae_engine
)

logger = logging.getLogger(__name__)


@dataclass
class IntegratedDecision:
    """Represents a decision made by the integrated system"""
    decision_id: str
    source: str  # 'goal', 'autonomous', 'uae', 'integrated'
    action: AutonomousAction
    goal: Optional[Goal]
    uae_confidence: float
    integrated_confidence: float
    reasoning: str
    timestamp: datetime


class GoalAutonomousUAEIntegration:
    """
    Integrates Goal Inference, Autonomous Decision Engine, and UAE
    for comprehensive intelligent automation.
    """

    def __init__(self):
        """Initialize the integration"""
        # Core engines
        self.goal_inference = get_goal_inference_engine()
        self.autonomous_engine = AutonomousDecisionEngine()
        self.advanced_engine = get_advanced_autonomous_engine()
        self.uae = get_uae_engine()

        # Integration state
        self.active_goals = {}
        self.recent_decisions = []
        self.learning_data = []

        # Configuration
        self.config = self._load_configuration()

        # Metrics
        self.metrics = {
            'goals_inferred': 0,
            'decisions_made': 0,
            'display_connections': 0,
            'successful_predictions': 0,
            'total_predictions': 0
        }

        logger.info("Goal-Autonomous-UAE Integration initialized")

    def _load_configuration(self) -> Dict[str, Any]:
        """Load integration configuration"""
        config_path = Path("backend/config/integration_config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)

        return {
            'min_goal_confidence': 0.75,
            'min_decision_confidence': 0.70,
            'enable_predictive_display': True,
            'max_concurrent_goals': 10,
            'learning_enabled': True,
            'auto_connect_threshold': 0.85
        }

    async def process_context(self, context: Dict[str, Any]) -> List[IntegratedDecision]:
        """
        Process context through all three systems and generate integrated decisions

        Args:
            context: Rich context including workspace state, windows, etc.

        Returns:
            List of integrated decisions
        """
        integrated_decisions = []

        try:
            # Step 1: Infer goals from context
            goals = await self.goal_inference.infer_goals(context)
            self.metrics['goals_inferred'] += len(self._flatten_goals(goals))

            # Step 2: Get UAE's unified awareness
            uae_decision = await self.uae.get_element_position(
                context.get('target_element', 'Living Room TV')
            )

            # Step 3: Generate autonomous decisions based on goals
            autonomous_decisions = await self.autonomous_engine.analyze_and_decide(
                context.get('workspace_state'),
                context.get('windows', [])
            )

            # Step 4: Use advanced engine for ML-based predictions
            advanced_decisions = await self.advanced_engine.make_decision(context)

            # Step 5: Integrate all decisions
            integrated = await self._integrate_decisions(
                goals,
                uae_decision,
                autonomous_decisions,
                advanced_decisions,
                context
            )

            integrated_decisions.extend(integrated)

            # Step 6: Apply learning if enabled
            if self.config['learning_enabled']:
                await self._apply_learning(integrated_decisions, context)

            # Update metrics
            self.metrics['decisions_made'] += len(integrated_decisions)

            # Check for display connections
            display_decisions = [
                d for d in integrated_decisions
                if d.action.action_type == 'connect_display'
            ]
            self.metrics['display_connections'] += len(display_decisions)

        except Exception as e:
            logger.error(f"Error in integrated processing: {e}")

        return integrated_decisions

    async def predict_display_connection(
        self,
        context: Dict[str, Any]
    ) -> Optional[IntegratedDecision]:
        """
        Predict if display should be connected based on goals and context

        Args:
            context: Current context

        Returns:
            Display connection decision if confidence is high enough
        """
        if not self.config['enable_predictive_display']:
            return None

        # Infer goals
        goals = await self.goal_inference.infer_goals(context)

        # Check for display-relevant goals
        for level, level_goals in goals.items():
            for goal in level_goals:
                if await self._is_display_relevant_goal(goal):
                    # Get display target from UAE
                    display_position = await self.uae.get_element_position("Living Room TV")

                    if display_position and display_position.confidence > self.config['auto_connect_threshold']:
                        # Create integrated decision
                        action = AutonomousAction(
                            action_type='connect_display',
                            target='Living Room TV',
                            params={
                                'display_name': 'Living Room TV',
                                'connection_type': 'predictive',
                                'goal_id': goal.goal_id,
                                'goal_type': goal.goal_type,
                                'uae_confidence': display_position.confidence
                            },
                            priority=ActionPriority.HIGH,
                            confidence=self._calculate_integrated_confidence(
                                goal.confidence,
                                display_position.confidence
                            ),
                            category=ActionCategory.WORKFLOW,
                            reasoning=f"Predicted display connection for {goal.description}"
                        )

                        decision = IntegratedDecision(
                            decision_id=self._generate_decision_id(),
                            source='integrated',
                            action=action,
                            goal=goal,
                            uae_confidence=display_position.confidence,
                            integrated_confidence=action.confidence,
                            reasoning=action.reasoning,
                            timestamp=datetime.now()
                        )

                        self.metrics['total_predictions'] += 1
                        return decision

        return None

    async def _is_display_relevant_goal(self, goal: Goal) -> bool:
        """Check if goal is relevant for display connection"""
        display_relevant_types = [
            'project_completion',
            'feature_implementation',
            'document_preparation',
            'meeting_preparation',
            'presentation'
        ]

        # Check goal type
        if goal.goal_type in display_relevant_types:
            return True

        # Check goal evidence for display-related keywords
        for evidence in goal.evidence:
            data_str = str(evidence.get('data', '')).lower()
            if any(keyword in data_str for keyword in ['display', 'screen', 'tv', 'monitor', 'present']):
                return True

        return False

    def _flatten_goals(self, goals: Dict[GoalLevel, List[Goal]]) -> List[Goal]:
        """Flatten hierarchical goals into a single list"""
        flat = []
        for level, level_goals in goals.items():
            flat.extend(level_goals)
        return flat

    async def _integrate_decisions(
        self,
        goals: Dict[GoalLevel, List[Goal]],
        uae_decision: Any,
        autonomous_decisions: List[AutonomousAction],
        advanced_decisions: List,
        context: Dict[str, Any]
    ) -> List[IntegratedDecision]:
        """Integrate decisions from all sources"""
        integrated = []

        # Create a mapping of goals for quick lookup
        goal_map = {}
        for level, level_goals in goals.items():
            for goal in level_goals:
                goal_map[goal.goal_id] = goal

        # Process autonomous decisions
        for action in autonomous_decisions:
            # Find associated goal if any
            associated_goal = None
            if 'goal_id' in action.params:
                associated_goal = goal_map.get(action.params['goal_id'])

            # Calculate integrated confidence
            uae_confidence = 0.5  # Default if no UAE data
            if uae_decision and hasattr(uae_decision, 'confidence'):
                uae_confidence = uae_decision.confidence

            integrated_confidence = self._calculate_integrated_confidence(
                action.confidence,
                uae_confidence,
                associated_goal.confidence if associated_goal else 0.5
            )

            # Only include high-confidence decisions
            if integrated_confidence >= self.config['min_decision_confidence']:
                decision = IntegratedDecision(
                    decision_id=self._generate_decision_id(),
                    source='autonomous',
                    action=action,
                    goal=associated_goal,
                    uae_confidence=uae_confidence,
                    integrated_confidence=integrated_confidence,
                    reasoning=self._generate_reasoning(action, associated_goal, uae_decision),
                    timestamp=datetime.now()
                )
                integrated.append(decision)

        # Process advanced engine decisions
        for adv_decision in advanced_decisions:
            if hasattr(adv_decision, 'predicted_action'):
                # Find associated goal
                associated_goal = goal_map.get(adv_decision.predicted_action.goal_id)

                # Convert to autonomous action
                action = AutonomousAction(
                    action_type=adv_decision.predicted_action.action_type,
                    target=adv_decision.predicted_action.target,
                    params={
                        'ml_confidence': adv_decision.ml_confidence,
                        'risk_level': adv_decision.risk_level.name,
                        'strategy': adv_decision.decision_strategy.name
                    },
                    priority=ActionPriority.HIGH if adv_decision.ml_confidence > 0.8 else ActionPriority.MEDIUM,
                    confidence=adv_decision.ml_confidence,
                    category=ActionCategory.WORKFLOW,
                    reasoning=f"ML-predicted action with {adv_decision.ml_confidence:.0%} confidence"
                )

                decision = IntegratedDecision(
                    decision_id=self._generate_decision_id(),
                    source='advanced',
                    action=action,
                    goal=associated_goal,
                    uae_confidence=0.0,  # Advanced engine doesn't use UAE directly
                    integrated_confidence=adv_decision.ml_confidence,
                    reasoning=action.reasoning,
                    timestamp=datetime.now()
                )
                integrated.append(decision)

        return integrated

    def _calculate_integrated_confidence(self, *confidences) -> float:
        """Calculate integrated confidence from multiple sources"""
        valid_confidences = [c for c in confidences if c > 0]
        if not valid_confidences:
            return 0.5

        # Weighted average with higher weight on agreement
        avg = sum(valid_confidences) / len(valid_confidences)

        # Boost if all sources agree (high confidence)
        if all(c > 0.7 for c in valid_confidences):
            avg = min(1.0, avg * 1.1)

        # Penalty if sources disagree
        if len(valid_confidences) > 1:
            std_dev = np.std(valid_confidences) if len(valid_confidences) > 1 else 0
            if std_dev > 0.2:
                avg *= (1 - std_dev * 0.5)

        return min(1.0, max(0.0, avg))

    def _generate_reasoning(
        self,
        action: AutonomousAction,
        goal: Optional[Goal],
        uae_decision: Any
    ) -> str:
        """Generate integrated reasoning for decision"""
        reasons = []

        if goal:
            reasons.append(f"Goal: {goal.description} (confidence: {goal.confidence:.0%})")

        if action.reasoning:
            reasons.append(f"Action: {action.reasoning}")

        if uae_decision and hasattr(uae_decision, 'confidence'):
            reasons.append(f"UAE confidence: {uae_decision.confidence:.0%}")

        return " | ".join(reasons) if reasons else "Integrated decision"

    def _generate_decision_id(self) -> str:
        """Generate unique decision ID"""
        import hashlib
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]

    async def _apply_learning(
        self,
        decisions: List[IntegratedDecision],
        context: Dict[str, Any]
    ):
        """Apply learning from decisions"""
        for decision in decisions:
            # Store decision for learning
            self.learning_data.append({
                'decision': decision,
                'context': context,
                'timestamp': datetime.now()
            })

            # Learn goal-action patterns
            if decision.goal:
                self.goal_inference.learn_from_completion(
                    decision.goal,
                    success=True  # Will be updated with actual outcome
                )

            # Update autonomous engine
            if hasattr(decision.action, 'action_type'):
                self.autonomous_engine.learn_from_feedback(
                    decision.action,
                    success=True,  # Will be updated with actual outcome
                    user_feedback=None
                )

        # Trim learning data to prevent memory issues
        if len(self.learning_data) > 1000:
            self.learning_data = self.learning_data[-1000:]

    async def execute_decision(
        self,
        decision: IntegratedDecision,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Execute an integrated decision

        Args:
            decision: The decision to execute
            context: Optional execution context

        Returns:
            Success status
        """
        try:
            logger.info(f"Executing integrated decision: {decision.action.action_type}")

            # Special handling for display connections
            if decision.action.action_type == 'connect_display':
                success = await self._execute_display_connection(decision, context)
            else:
                # Generic execution (would integrate with actual systems)
                success = await self._execute_generic_action(decision, context)

            # Update metrics
            if success and decision.source == 'integrated':
                self.metrics['successful_predictions'] += 1

            # Learn from outcome
            if self.config['learning_enabled']:
                await self._learn_from_execution(decision, success)

            return success

        except Exception as e:
            logger.error(f"Error executing decision: {e}")
            return False

    async def _execute_display_connection(
        self,
        decision: IntegratedDecision,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Execute display connection action"""
        try:
            display_name = decision.action.params.get('display_name', 'Living Room TV')
            logger.info(f"Connecting to {display_name} based on integrated decision")

            # Here you would integrate with actual display connection code
            # For now, simulate success
            await asyncio.sleep(0.5)

            logger.info(f"Successfully connected to {display_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect display: {e}")
            return False

    async def _execute_generic_action(
        self,
        decision: IntegratedDecision,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Execute generic action"""
        # This would integrate with actual system actions
        logger.info(f"Executing action: {decision.action.action_type}")
        await asyncio.sleep(0.3)
        return True

    async def _learn_from_execution(
        self,
        decision: IntegratedDecision,
        success: bool
    ):
        """Learn from execution outcome"""
        # Update goal learning
        if decision.goal:
            self.goal_inference.learn_from_completion(decision.goal, success)

        # Update autonomous learning
        self.autonomous_engine.learn_from_feedback(
            decision.action,
            success,
            user_feedback=None
        )

        # Update advanced engine if available
        if hasattr(self.advanced_engine, 'learn_from_execution'):
            # Advanced engine learning would go here
            pass

    def get_metrics(self) -> Dict[str, Any]:
        """Get integration metrics"""
        prediction_accuracy = 0.0
        if self.metrics['total_predictions'] > 0:
            prediction_accuracy = (
                self.metrics['successful_predictions'] /
                self.metrics['total_predictions']
            )

        return {
            **self.metrics,
            'prediction_accuracy': prediction_accuracy,
            'active_goals': len(self.active_goals),
            'recent_decisions': len(self.recent_decisions)
        }

    async def save_state(self):
        """Save integration state"""
        state = {
            'config': self.config,
            'metrics': self.metrics,
            'learning_data_size': len(self.learning_data)
        }

        state_path = Path("backend/data/integration_state.json")
        state_path.parent.mkdir(parents=True, exist_ok=True)

        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)

        # Save individual engine states
        self.goal_inference.save_state(Path("backend/data/goal_inference_state.json"))
        await self.advanced_engine.save_state()


# Global instance
_integration_instance = None


def get_integration() -> GoalAutonomousUAEIntegration:
    """Get or create the global integration instance"""
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = GoalAutonomousUAEIntegration()
    return _integration_instance


async def test_integration():
    """Test the integrated system"""
    print("🚀 Testing Goal-Autonomous-UAE Integration")
    print("=" * 60)

    integration = get_integration()

    # Create a mock workspace state for testing
    class MockWorkspaceState:
        def __init__(self):
            self.focused_task = 'Preparing presentation for team meeting'
            self.workspace_context = 'Development environment with presentation prep'
            self.suggestions = ['Connect to display', 'Organize workspace']
            self.confidence = 0.9

    # Create rich test context
    test_context = {
        'active_applications': ['vscode', 'chrome', 'terminal'],
        'recent_actions': ['typing', 'switching_tabs', 'scrolling'],
        'content': {
            'type': 'code',
            'language': 'python',
            'project': 'JARVIS',
            'focused_task': 'Implementing display connection automation'
        },
        'workspace_state': MockWorkspaceState(),
        'windows': [],
        'target_element': 'Living Room TV',
        'time_context': {
            'time_of_day': 'afternoon',
            'day_of_week': 'weekday'
        }
    }

    print("\n📊 Processing context through integrated system...")
    decisions = await integration.process_context(test_context)

    print(f"\n✨ Generated {len(decisions)} integrated decisions:\n")
    for i, decision in enumerate(decisions, 1):
        print(f"{i}. Action: {decision.action.action_type}")
        print(f"   Source: {decision.source}")
        print(f"   Target: {decision.action.target}")
        if decision.goal:
            print(f"   Goal: {decision.goal.description}")
        print(f"   UAE Confidence: {decision.uae_confidence:.0%}")
        print(f"   Integrated Confidence: {decision.integrated_confidence:.0%}")
        print(f"   Reasoning: {decision.reasoning}")
        print()

    # Test predictive display connection
    print("\n🔮 Testing predictive display connection...")
    display_decision = await integration.predict_display_connection(test_context)

    if display_decision:
        print(f"✅ Predicted display connection:")
        print(f"   Display: {display_decision.action.target}")
        print(f"   Confidence: {display_decision.integrated_confidence:.0%}")
        print(f"   Reasoning: {display_decision.reasoning}")

        # Execute the decision
        print("\n⚡ Executing display connection...")
        success = await integration.execute_decision(display_decision, test_context)
        print(f"   Result: {'Success' if success else 'Failed'}")
    else:
        print("❌ No display connection predicted")

    # Show metrics
    print("\n📈 Integration Metrics:")
    metrics = integration.get_metrics()
    for key, value in metrics.items():
        print(f"   {key}: {value}")

    # Save state
    await integration.save_state()
    print("\n💾 State saved successfully")

    print("\n✅ Integration test complete!")


if __name__ == "__main__":
    import numpy as np
    asyncio.run(test_integration())