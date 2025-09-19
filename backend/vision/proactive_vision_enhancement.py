#!/usr/bin/env python3
"""
Proactive Vision Enhancement for ClaudeVisionAnalyzer
Integrates the new proactive intelligence system with the existing analyzer
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .proactive_vision_intelligence import ProactiveVisionIntelligence
from .intelligent_notification_filter import NotificationFilter, NotificationContext  
from .proactive_communication_module import ProactiveCommunicator
from .claude_vision_analyzer_main import ClaudeVisionAnalyzer

logger = logging.getLogger(__name__)


class ProactiveVisionEnhancement:
    """
    Enhancement layer that adds proactive intelligence to ClaudeVisionAnalyzer
    Designed to be integrated seamlessly with the existing system
    """
    
    @staticmethod
    async def enhance_analyzer(analyzer: ClaudeVisionAnalyzer, voice_api=None) -> None:
        """
        Enhance an existing ClaudeVisionAnalyzer with proactive capabilities
        
        Args:
            analyzer: The ClaudeVisionAnalyzer instance to enhance
            voice_api: Optional voice API for notifications
        """
        logger.info("Enhancing ClaudeVisionAnalyzer with proactive intelligence")
        
        # Create proactive components
        proactive_intelligence = ProactiveVisionIntelligence(
            vision_analyzer=analyzer,
            notification_callback=None  # Will be set later
        )
        
        notification_filter = NotificationFilter()
        
        communicator = ProactiveCommunicator(
            voice_callback=voice_api.speak if voice_api else None,
            text_callback=None  # Can be added later
        )
        
        # Create integrated notification handler
        async def handle_proactive_notification(notification_data: Dict[str, Any]):
            """Handle notifications with filtering and communication"""
            try:
                # Parse change from notification
                from .proactive_vision_intelligence import ScreenChange, Priority, ChangeCategory
                
                change = ScreenChange(
                    description=notification_data.get('message', ''),
                    importance=Priority(notification_data.get('priority', 'medium')),
                    confidence=notification_data.get('confidence', 0.8),
                    category=ChangeCategory(notification_data.get('category', 'other')),
                    suggested_message=notification_data.get('message', ''),
                    location=notification_data.get('location', ''),
                    timestamp=datetime.now(),
                    screenshot_hash=''
                )
                
                # Build context
                context = NotificationContext(
                    user_activity='unknown',  # Could be enhanced
                    focus_level=0.5,
                    recent_notifications=[],
                    time_of_day=datetime.now().strftime("%H:%M"),
                    workflow_phase='normal',
                    interaction_history=[]
                )
                
                # Apply filtering
                should_notify, reason = notification_filter.should_notify(change, context)
                
                if should_notify:
                    # Send through communicator
                    await communicator.send_proactive_message(change)
                    notification_filter.record_notification(change, context)
                else:
                    logger.debug(f"Notification filtered: {reason}")
                    
            except Exception as e:
                logger.error(f"Error handling proactive notification: {e}")
                
        # Set the notification callback
        proactive_intelligence.notification_callback = handle_proactive_notification
        
        # Store components on analyzer
        analyzer._proactive_intelligence = proactive_intelligence
        analyzer._notification_filter = notification_filter
        analyzer._proactive_communicator = communicator
        
        # Add proactive methods to analyzer
        analyzer.start_proactive_monitoring = lambda: proactive_intelligence.start_monitoring()
        analyzer.stop_proactive_monitoring = lambda: proactive_intelligence.stop_monitoring()
        analyzer.get_proactive_stats = lambda: proactive_intelligence.get_monitoring_stats()
        
        # Update configuration
        if hasattr(analyzer, '_continuous_analyzer_config'):
            analyzer._continuous_analyzer_config['enable_proactive'] = True
            analyzer._continuous_analyzer_config['proactive_enhanced'] = True
            
        logger.info("Proactive enhancement complete")
        
    @staticmethod
    def add_proactive_methods(analyzer_class):
        """
        Add proactive methods directly to the ClaudeVisionAnalyzer class
        This is an alternative integration approach
        """
        
        async def start_proactive_intelligence(self, config: Optional[Dict[str, Any]] = None):
            """Start enhanced proactive monitoring with pure Claude intelligence"""
            logger.info("Starting enhanced proactive intelligence monitoring")
            
            # Initialize if not already done
            if not hasattr(self, '_proactive_intelligence'):
                from .proactive_vision_integration import create_proactive_vision_system
                self._proactive_system = await create_proactive_vision_system(self)
                
            # Update config if provided
            if config:
                self._proactive_system.update_config(config)
                
            # Start monitoring
            await self._proactive_system.start_proactive_monitoring()
            
            return True
            
        async def stop_proactive_intelligence(self):
            """Stop enhanced proactive monitoring"""
            if hasattr(self, '_proactive_system'):
                await self._proactive_system.stop_proactive_monitoring()
                
        async def handle_user_response(self, response: str):
            """Handle user response to proactive notifications"""
            if hasattr(self, '_proactive_system'):
                return await self._proactive_system.handle_user_response(response)
            return "Proactive system not initialized"
            
        def get_proactive_intelligence_stats(self):
            """Get comprehensive proactive system statistics"""
            if hasattr(self, '_proactive_system'):
                return self._proactive_system.get_system_stats()
            return {}
            
        # Add methods to class
        analyzer_class.start_proactive_intelligence = start_proactive_intelligence
        analyzer_class.stop_proactive_intelligence = stop_proactive_intelligence
        analyzer_class.handle_user_response = handle_user_response
        analyzer_class.get_proactive_intelligence_stats = get_proactive_intelligence_stats
        
        logger.info("Added proactive intelligence methods to ClaudeVisionAnalyzer")


# Configuration integration
def update_vision_config_for_proactive(config_class):
    """Update VisionConfig class with proactive settings"""
    
    # Add new configuration fields
    new_fields = {
        # Proactive intelligence settings
        'proactive_analysis_interval': 3.0,
        'proactive_importance_threshold': 0.6,
        'proactive_confidence_threshold': 0.7,
        'proactive_max_notifications_per_minute': 3,
        'proactive_cooldown_seconds': 30,
        
        # Communication settings
        'proactive_notification_style': 'balanced',  # minimal/balanced/detailed/conversational
        'proactive_voice_enabled': True,
        'proactive_progressive_disclosure': True,
        
        # Learning settings
        'proactive_enable_learning': True,
        'proactive_adapt_to_user': True,
        
        # Context awareness
        'proactive_context_awareness': True,
        'proactive_workflow_detection': True
    }
    
    # This would need to be integrated into the VisionConfig dataclass
    return new_fields


# Usage example for integration
async def integrate_proactive_vision(api_key: str):
    """Example of how to integrate proactive vision with existing analyzer"""
    
    # Create standard analyzer
    analyzer = ClaudeVisionAnalyzer(api_key)
    
    # Method 1: Enhance existing instance
    await ProactiveVisionEnhancement.enhance_analyzer(analyzer)
    
    # Method 2: Add methods to class (do this once at startup)
    ProactiveVisionEnhancement.add_proactive_methods(ClaudeVisionAnalyzer)
    
    # Now analyzer has proactive capabilities
    await analyzer.start_proactive_intelligence({
        'importance_threshold': 0.6,
        'notification_style': 'balanced',
        'enable_voice': True
    })
    
    # Use it
    # ... monitoring happens automatically ...
    
    # Get stats
    stats = analyzer.get_proactive_intelligence_stats()
    print(f"Proactive stats: {stats}")
    
    # Stop when done
    await analyzer.stop_proactive_intelligence()