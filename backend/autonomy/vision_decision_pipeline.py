#!/usr/bin/env python3
"""
Vision Decision Pipeline for JARVIS
Integrates vision system with decision engine for autonomous processing
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Import vision components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.screen_capture_module import ScreenCaptureModule, ScreenCapture
from vision.ocr_processor import OCRProcessor, OCRResult
from vision.window_analysis import WindowAnalyzer, WindowContent, ApplicationCategory
from vision.enhanced_monitoring import EnhancedWorkspaceMonitor

# Import autonomy components
from .autonomous_decision_engine import (
    AutonomousDecisionEngine, AutonomousAction, 
    ActionPriority, ActionCategory
)
from .action_queue import ActionQueueManager
from .autonomous_behaviors import AutonomousBehaviorManager
from .system_states import SystemStateManager, SystemState, ComponentState, TransitionReason
from .error_recovery import ErrorRecoveryManager, ErrorCategory, ErrorSeverity
from .monitoring_metrics import SystemMonitor

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Stages of vision processing pipeline"""
    CAPTURE = "capture"
    OCR = "ocr"
    ANALYSIS = "analysis"
    DECISION = "decision"
    QUEUING = "queuing"
    EXECUTION = "execution"


@dataclass
class PipelineContext:
    """Context passed through processing pipeline"""
    stage: ProcessingStage
    timestamp: datetime = field(default_factory=datetime.now)
    screen_capture: Optional[ScreenCapture] = None
    ocr_results: Dict[str, OCRResult] = field(default_factory=dict)
    window_analyses: List[WindowContent] = field(default_factory=list)
    workspace_state: Dict[str, Any] = field(default_factory=dict)
    decisions: List[AutonomousAction] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def add_metric(self, name: str, value: float):
        """Add performance metric"""
        self.metrics[name] = value
        
    def add_error(self, error: str):
        """Add error to context"""
        self.errors.append(f"[{self.stage.value}] {error}")
        logger.error(f"Pipeline error in {self.stage.value}: {error}")


class VisionDecisionPipeline:
    """Main pipeline for processing vision data into autonomous actions"""
    
    def __init__(self):
        # Vision components
        self.screen_capture = ScreenCaptureModule(capture_interval=2.0)
        self.ocr_processor = OCRProcessor()
        self.window_analyzer = WindowAnalyzer()
        self.enhanced_monitor = EnhancedWorkspaceMonitor()
        
        # Decision components
        self.decision_engine = AutonomousDecisionEngine()
        self.behavior_manager = AutonomousBehaviorManager()
        self.action_queue = ActionQueueManager()
        
        # State manager
        self.state_manager = SystemStateManager()
        
        # Error recovery manager
        self.error_manager = ErrorRecoveryManager()
        
        # System monitor
        self.monitor = SystemMonitor()
        
        # Pipeline configuration
        self.config = {
            'enable_ocr': True,
            'enable_enhanced_monitoring': True,
            'confidence_threshold': 0.7,
            'max_actions_per_cycle': 5,
            'priority_boost_urgent': 2.0
        }
        
        # Pipeline state
        self.is_running = False
        self.processing_task = None
        self.cycle_count = 0
        self.last_context: Optional[PipelineContext] = None
        
        # Performance tracking
        self.performance_history = []
        
        # Initialize system components
        self._register_components()
        
        # Register error recovery callbacks
        self._setup_error_recovery()
        
    async def start_pipeline(self):
        """Start the vision decision pipeline"""
        if self.is_running:
            logger.warning("Pipeline already running")
            return
            
        self.is_running = True
        
        # Start screen capture
        await self.screen_capture.start_continuous_capture()
        
        # Set up capture callback
        self.screen_capture.add_capture_callback(self._process_capture)
        
        # Start processing loop
        self.processing_task = asyncio.create_task(self._pipeline_loop())
        
        # Initialize state manager
        await self.state_manager.initialize_system()
        await self.state_manager.start_monitoring()
        
        # Start system monitoring
        await self.monitor.start_monitoring()
        
        logger.info("Vision Decision Pipeline started")
        
    async def stop_pipeline(self):
        """Stop the vision decision pipeline"""
        self.is_running = False
        
        # Stop screen capture
        await self.screen_capture.stop_continuous_capture()
        
        # Cancel processing task
        if self.processing_task:
            self.processing_task.cancel()
            
        # Stop system monitoring
        await self.monitor.stop_monitoring()
        
        # Shutdown state manager
        await self.state_manager.shutdown_system()
        
        logger.info("Vision Decision Pipeline stopped")
        
    async def _pipeline_loop(self):
        """Main pipeline processing loop"""
        while self.is_running:
            try:
                # Wait for next capture (handled by callback)
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Pipeline loop error: {e}")
                await asyncio.sleep(1)
                
    async def _process_capture(self, capture: ScreenCapture):
        """Process a new screen capture through the pipeline"""
        self.cycle_count += 1
        capture_start = datetime.now()
        context = PipelineContext(stage=ProcessingStage.CAPTURE)
        context.screen_capture = capture
        
        # Record capture metric
        capture_duration = (datetime.now() - capture_start).total_seconds()
        self.monitor.record_capture(capture_duration)
        
        try:
            # Transition to processing state
            await self.state_manager.transition_to(
                SystemState.PROCESSING,
                TransitionReason.AUTOMATIC
            )
            
            # Run through pipeline stages
            await self._stage_ocr(context)
            await self._stage_analysis(context)
            
            # Transition to deciding state
            await self.state_manager.transition_to(
                SystemState.DECIDING,
                TransitionReason.COMPLETION
            )
            
            await self._stage_decision(context)
            
            # Transition to executing state if actions to queue
            if context.decisions:
                await self.state_manager.transition_to(
                    SystemState.EXECUTING,
                    TransitionReason.COMPLETION
                )
                await self._stage_queuing(context)
            
            # Return to monitoring
            await self.state_manager.transition_to(
                SystemState.MONITORING,
                TransitionReason.COMPLETION
            )
            
            # Store context
            self.last_context = context
            
            # Track performance
            self._track_performance(context)
            
        except Exception as e:
            context.add_error(f"Pipeline processing failed: {e}")
            logger.error(f"Pipeline error: {e}", exc_info=True)
            
            # Transition to error recovery
            await self.state_manager.transition_to(
                SystemState.ERROR_RECOVERY,
                TransitionReason.ERROR,
                metadata={'error': str(e)}
            )
            
    async def _stage_ocr(self, context: PipelineContext):
        """OCR processing stage"""
        context.stage = ProcessingStage.OCR
        start_time = datetime.now()
        
        if not self.config['enable_ocr']:
            return
            
        try:
            # Get current windows
            windows = self.window_analyzer.window_detector.get_all_windows()
            
            # Process visible windows with OCR
            for window in windows[:5]:  # Limit to top 5 windows
                if window.is_visible and window.width > 100 and window.height > 100:
                    # Define region for window
                    region = (window.x, window.y, window.width, window.height)
                    
                    # Crop screenshot to window
                    window_image = context.screen_capture.image.crop(region)
                    
                    # Run OCR
                    ocr_result = await self.ocr_processor.process_image(window_image)
                    context.ocr_results[window.window_id] = ocr_result
                    
            # Record metrics
            elapsed = (datetime.now() - start_time).total_seconds()
            context.add_metric('ocr_time', elapsed)
            context.add_metric('ocr_windows', len(context.ocr_results))
            
            # Record monitoring metrics
            total_regions = sum(len(r.regions) for r in context.ocr_results.values())
            self.monitor.record_ocr(elapsed, total_regions)
            
            # Update component state
            self.state_manager.update_component_state(
                'ocr_processor',
                ComponentState.ACTIVE,
                health_score=1.0
            )
            
        except Exception as e:
            context.add_error(f"OCR failed: {e}")
            self.state_manager.update_component_state(
                'ocr_processor',
                ComponentState.ERROR,
                health_score=0.3
            )
            # Handle error
            await self.error_manager.handle_error(
                e,
                component='ocr_processor',
                category=ErrorCategory.OCR,
                severity=ErrorSeverity.MEDIUM
            )
            self.monitor.record_error('ocr_processor', 'medium')
            
    async def _stage_analysis(self, context: PipelineContext):
        """Window and workspace analysis stage"""
        context.stage = ProcessingStage.ANALYSIS
        start_time = datetime.now()
        
        try:
            # Get enhanced workspace state if enabled
            if self.config['enable_enhanced_monitoring']:
                workspace_state = await self.enhanced_monitor.get_complete_workspace_state()
                context.workspace_state = workspace_state
                
                # Extract window analyses
                windows = workspace_state.get('windows', [])
                for window in windows:
                    # Analyze window with OCR results if available
                    ocr_result = context.ocr_results.get(window.window_id)
                    
                    window_content = await self.window_analyzer.analyze_window(
                        window, 
                        context.screen_capture
                    )
                    context.window_analyses.append(window_content)
            else:
                # Basic window analysis
                windows = self.window_analyzer.window_detector.get_all_windows()
                for window in windows[:5]:
                    if window.is_visible:
                        window_content = await self.window_analyzer.analyze_window(window)
                        context.window_analyses.append(window_content)
                        
            # Record metrics
            elapsed = (datetime.now() - start_time).total_seconds()
            context.add_metric('analysis_time', elapsed)
            context.add_metric('analyzed_windows', len(context.window_analyses))
            
            # Record monitoring metrics
            self.monitor.record_analysis(elapsed, len(context.window_analyses))
            
            # Update component state
            self.state_manager.update_component_state(
                'window_analyzer',
                ComponentState.ACTIVE,
                health_score=1.0
            )
            self.monitor.update_component_health('window_analyzer', 1.0)
            
        except Exception as e:
            context.add_error(f"Analysis failed: {e}")
            self.state_manager.update_component_state(
                'window_analyzer',
                ComponentState.ERROR,
                health_score=0.3
            )
            # Handle error
            await self.error_manager.handle_error(
                e,
                component='window_analyzer',
                category=ErrorCategory.VISION,
                severity=ErrorSeverity.MEDIUM
            )
            self.monitor.record_error('window_analyzer', 'medium')
            
    async def _stage_decision(self, context: PipelineContext):
        """Decision making stage"""
        context.stage = ProcessingStage.DECISION
        start_time = datetime.now()
        
        try:
            # Get actionable windows
            actionable = self.window_analyzer.get_actionable_windows(
                context.window_analyses
            )
            
            # Generate decisions for each actionable window
            for window_content in actionable:
                decisions = await self._generate_window_decisions(
                    window_content,
                    context
                )
                context.decisions.extend(decisions)
                
            # Get behavior-based decisions
            if context.workspace_state:
                behavior_actions = await self.behavior_manager.process_workspace_state(
                    context.workspace_state,
                    context.workspace_state.get('windows', [])
                )
                context.decisions.extend(behavior_actions)
                
            # Filter by confidence threshold
            context.decisions = [
                d for d in context.decisions 
                if d.confidence >= self.config['confidence_threshold']
            ]
            
            # Limit number of actions
            if len(context.decisions) > self.config['max_actions_per_cycle']:
                # Sort by priority and confidence
                context.decisions.sort(
                    key=lambda x: (x.priority.value, -x.confidence)
                )
                context.decisions = context.decisions[:self.config['max_actions_per_cycle']]
                
            # Record metrics
            elapsed = (datetime.now() - start_time).total_seconds()
            context.add_metric('decision_time', elapsed)
            context.add_metric('decisions_made', len(context.decisions))
            
            # Record monitoring metrics
            self.monitor.record_decision(elapsed, len(context.decisions))
            
            # Update component state
            self.state_manager.update_component_state(
                'decision_engine',
                ComponentState.ACTIVE,
                health_score=1.0
            )
            self.monitor.update_component_health('decision_engine', 1.0)
            
        except Exception as e:
            context.add_error(f"Decision making failed: {e}")
            self.state_manager.update_component_state(
                'decision_engine',
                ComponentState.ERROR,
                health_score=0.3
            )
            # Handle error
            await self.error_manager.handle_error(
                e,
                component='decision_engine',
                category=ErrorCategory.DECISION,
                severity=ErrorSeverity.HIGH
            )
            self.monitor.record_error('decision_engine', 'high')
            
    async def _generate_window_decisions(self, window_content: WindowContent,
                                       context: PipelineContext) -> List[AutonomousAction]:
        """Generate decisions for a specific window"""
        decisions = []
        
        # Handle urgent notifications
        if window_content.has_urgent_items:
            for notification in window_content.notifications:
                if notification.get('urgent'):
                    action = AutonomousAction(
                        action_type="handle_urgent_notification",
                        target=window_content.app_name,
                        params={
                            'window_id': window_content.window_id,
                            'notification': notification
                        },
                        priority=ActionPriority.HIGH,
                        confidence=0.9,
                        category=ActionCategory.NOTIFICATION,
                        reasoning=f"Urgent notification in {window_content.app_name}",
                        requires_permission=True
                    )
                    decisions.append(action)
                    
        # Handle error states
        if window_content.state.value == 'error':
            action = AutonomousAction(
                action_type="handle_application_error",
                target=window_content.app_name,
                params={
                    'window_id': window_content.window_id,
                    'error_context': window_content.key_information
                },
                priority=ActionPriority.HIGH,
                confidence=0.85,
                category=ActionCategory.MAINTENANCE,
                reasoning=f"Error detected in {window_content.app_name}",
                requires_permission=True
            )
            decisions.append(action)
            
        # Handle actionable items
        for action_item in window_content.action_items[:2]:  # Limit per window
            if action_item['type'] == 'button':
                # Determine action based on button text
                button_text = action_item['text'].lower()
                
                if button_text in ['ok', 'accept', 'yes', 'continue']:
                    action = AutonomousAction(
                        action_type="click_confirmation",
                        target=window_content.app_name,
                        params={
                            'window_id': window_content.window_id,
                            'button': action_item
                        },
                        priority=ActionPriority.MEDIUM,
                        confidence=0.7,
                        category=ActionCategory.WORKFLOW,
                        reasoning=f"Confirmation needed in {window_content.app_name}",
                        requires_permission=True
                    )
                    decisions.append(action)
                    
        return decisions
        
    async def _stage_queuing(self, context: PipelineContext):
        """Action queuing stage"""
        context.stage = ProcessingStage.QUEUING
        start_time = datetime.now()
        
        try:
            if context.decisions:
                # Add decisions to action queue
                added = await self.action_queue.add_actions(context.decisions)
                
                logger.info(f"Queued {added}/{len(context.decisions)} actions")
                
                # Record metrics
                context.add_metric('actions_queued', added)
                
                # Record queue depth
                self.monitor.record_queue_depth(len(self.action_queue.action_queue))
                
            # Record timing
            elapsed = (datetime.now() - start_time).total_seconds()
            context.add_metric('queuing_time', elapsed)
            
        except Exception as e:
            context.add_error(f"Queuing failed: {e}")
            
    def _track_performance(self, context: PipelineContext):
        """Track pipeline performance metrics"""
        performance = {
            'cycle': self.cycle_count,
            'timestamp': context.timestamp,
            'total_time': sum(context.metrics.values()),
            'metrics': context.metrics,
            'errors': len(context.errors),
            'decisions': len(context.decisions)
        }
        
        self.performance_history.append(performance)
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
            
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        avg_metrics = {}
        if self.performance_history:
            # Calculate average metrics
            for metric in ['ocr_time', 'analysis_time', 'decision_time', 'total_time']:
                values = [p['metrics'].get(metric, 0) for p in self.performance_history[-10:]]
                if values:
                    avg_metrics[f'avg_{metric}'] = sum(values) / len(values)
                    
        # Get system state status
        system_status = self.state_manager.get_system_status()
                    
        return {
            'is_running': self.is_running,
            'cycle_count': self.cycle_count,
            'queue_length': len(self.action_queue.action_queue),
            'last_run': self.last_context.timestamp.isoformat() if self.last_context else None,
            'performance': avg_metrics,
            'recent_errors': self.last_context.errors if self.last_context else [],
            'system_state': system_status,
            'error_statistics': self.error_manager.get_error_statistics(),
            'monitoring_report': self.monitor.get_monitoring_report()
        }
        
    def _register_components(self):
        """Register pipeline components with state manager"""
        self.state_manager.register_component('vision_pipeline', ComponentState.READY)
        self.state_manager.register_component('ocr_processor', ComponentState.READY)
        self.state_manager.register_component('window_analyzer', ComponentState.READY)
        self.state_manager.register_component('decision_engine', ComponentState.READY)
        self.state_manager.register_component('action_queue', ComponentState.READY)
        
    def _setup_error_recovery(self):
        """Setup error recovery mechanisms"""
        # Register component reset functions
        self.error_manager.register_component_reset(
            'ocr_processor',
            self._reset_ocr_processor
        )
        self.error_manager.register_component_reset(
            'vision_pipeline',
            self._reset_vision_pipeline
        )
        
        # Add error callbacks
        async def on_error(error_record, **kwargs):
            if kwargs.get('shutdown'):
                logger.critical("Emergency shutdown requested")
                await self.stop_pipeline()
                
        self.error_manager.add_error_callback(on_error)
        
    async def _reset_ocr_processor(self):
        """Reset OCR processor"""
        logger.info("Resetting OCR processor...")
        self.ocr_processor = OCRProcessor()
        self.state_manager.update_component_state(
            'ocr_processor',
            ComponentState.READY,
            health_score=1.0
        )
        
    async def _reset_vision_pipeline(self):
        """Reset vision pipeline"""
        logger.info("Resetting vision pipeline...")
        # Restart screen capture
        await self.screen_capture.stop_continuous_capture()
        await asyncio.sleep(1)
        await self.screen_capture.start_continuous_capture()
        self.state_manager.update_component_state(
            'vision_pipeline',
            ComponentState.READY,
            health_score=1.0
        )
        
    async def process_single_capture(self) -> PipelineContext:
        """Process a single screen capture (for testing)"""
        # Capture screen
        capture = self.screen_capture.capture_screen()
        if not capture:
            raise Exception("Failed to capture screen")
            
        # Process through pipeline
        context = PipelineContext(stage=ProcessingStage.CAPTURE)
        context.screen_capture = capture
        
        await self._stage_ocr(context)
        await self._stage_analysis(context)
        await self._stage_decision(context)
        
        return context


async def test_vision_pipeline():
    """Test the vision decision pipeline"""
    print("🤖 Testing Vision Decision Pipeline")
    print("=" * 50)
    
    pipeline = VisionDecisionPipeline()
    
    # Process single capture
    print("\n📸 Processing single capture...")
    
    try:
        context = await pipeline.process_single_capture()
        
        print(f"\n✅ Pipeline completed")
        print(f"   OCR Results: {len(context.ocr_results)}")
        print(f"   Window Analyses: {len(context.window_analyses)}")
        print(f"   Decisions Made: {len(context.decisions)}")
        print(f"   Errors: {len(context.errors)}")
        
        if context.decisions:
            print("\n📋 Generated Decisions:")
            for decision in context.decisions[:3]:
                print(f"   - {decision.action_type} on {decision.target}")
                print(f"     Priority: {decision.priority.name}")
                print(f"     Confidence: {decision.confidence:.2f}")
                print(f"     Reasoning: {decision.reasoning}")
                
        if context.metrics:
            print("\n⏱️ Performance Metrics:")
            for metric, value in context.metrics.items():
                print(f"   {metric}: {value:.3f}s")
                
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}")
        
    print("\n✅ Vision pipeline test complete!")


if __name__ == "__main__":
    asyncio.run(test_vision_pipeline())