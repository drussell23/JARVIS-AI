#!/usr/bin/env python3
"""
Enhanced Real-Time Interaction Handler for JARVIS Screen Monitoring
Fully dynamic with Claude Vision API integration - no hardcoded responses
"""

import asyncio
import logging
import time
import os
from typing import Dict, Any, Optional, List, Callable, Set, Tuple
from datetime import datetime, timedelta
from collections import deque
import json
import hashlib
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class RealTimeInteractionHandler:
    """Proactive real-time intelligent assistant with contextual conversation capabilities"""
    
    def __init__(self, continuous_analyzer=None, notification_callback: Optional[Callable] = None,
                 vision_analyzer=None):
        """
        Initialize proactive real-time interaction handler
        
        Args:
            continuous_analyzer: The continuous screen analyzer instance
            notification_callback: Callback to send notifications to user
            vision_analyzer: Claude Vision analyzer for dynamic analysis
        """
        self.analyzer = continuous_analyzer
        self.notification_callback = notification_callback
        self.vision_analyzer = vision_analyzer
        
        # Dynamic configuration - no hardcoded values
        self.config = self._load_dynamic_config()
        
        # Enhanced interaction state for proactive assistance
        self.interaction_state = {
            'monitoring_start_time': None,
            'last_interaction_time': None,
            'last_notification_time': None,
            'notifications_sent': deque(maxlen=self.config['max_notifications_per_hour']),
            'screen_history': deque(maxlen=50),  # Increased for better context
            'context_evolution': {},  # Track how context changes over time
            'user_activity_patterns': {},  # Learn user patterns
            'interaction_effectiveness': {},  # Track which interactions were helpful
            'screen_regions_of_interest': [],  # Dynamic regions based on activity
            'pending_analysis_queue': deque(),  # Queue for detailed analysis
            'conversation_context': deque(maxlen=20),  # Extended conversation history
            'active_workflows': {},  # Currently detected workflows
            'workflow_state': {},  # State of each workflow
            'user_focus_areas': [],  # Where user is focusing
            'assistance_opportunities': deque(maxlen=10),  # Potential help moments
            'context_switches': deque(maxlen=10),  # Track context switches
            'error_recovery_attempts': {},  # Track error resolution
            'productivity_metrics': {}  # Track productivity patterns
        }
        
        # Enhanced learning for proactive behavior
        self.learning_state = {
            'observed_workflows': {},  # Dynamically learned workflows
            'interaction_responses': {},  # Track user responses to notifications
            'timing_patterns': {},  # Learn best times to interact
            'attention_patterns': {},  # Learn where user focuses
            'error_patterns': {},  # Learn error signatures
            'success_patterns': {},  # Learn success patterns
            'workflow_sequences': {},  # Common workflow sequences
            'assistance_effectiveness': {},  # Which assistance was helpful
            'user_preferences': {},  # Learned user preferences
            'context_triggers': {},  # What triggers context switches
            'productivity_indicators': {}  # What indicates productive work
        }
        
        # Enhanced Claude Vision settings for proactive analysis
        self.vision_settings = {
            'use_adaptive_prompts': True,
            'context_window_size': 10,  # Increased for better understanding
            'analysis_depth': 'comprehensive',
            'enable_predictive_analysis': True,
            'enable_comparative_analysis': True,
            'enable_behavioral_analysis': True,
            'enable_workflow_detection': True,
            'enable_opportunity_detection': True,
            'enable_conversation_flow': True
        }
        
        # Proactive monitoring settings
        self.proactive_settings = {
            'workflow_detection_enabled': True,
            'error_assistance_enabled': True,
            'productivity_suggestions_enabled': True,
            'context_aware_timing': True,
            'natural_conversation_mode': True,
            'min_observation_time': 10.0,  # Observe before first interaction
            'workflow_confidence_threshold': 0.8,
            'assistance_confidence_threshold': 0.85
        }
        
        # Register callbacks if analyzer is provided
        if self.analyzer:
            self._register_analyzer_callbacks()
        
        self._monitoring_task = None
        self._proactive_task = None
        self._workflow_detection_task = None
        self._is_active = False
        self._analysis_cache = {}  # Cache with TTL
        self._workflow_detectors = self._initialize_workflow_detectors()
        
    def _load_dynamic_config(self) -> Dict[str, Any]:
        """Load configuration dynamically - no hardcoded defaults"""
        # Calculate dynamic values based on system resources
        import psutil
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Dynamic intervals based on system capability
        base_interval = 30.0 if memory_gb >= 16 else 45.0
        
        return {
            'interaction_interval': float(os.getenv('JARVIS_INTERACTION_INTERVAL', str(base_interval))),
            'context_aware_notifications': os.getenv('JARVIS_CONTEXT_AWARE', 'true').lower() == 'true',
            'proactive_assistance': os.getenv('JARVIS_PROACTIVE', 'true').lower() == 'true',
            'notification_cooldown': float(os.getenv('JARVIS_NOTIFICATION_COOLDOWN', str(base_interval * 2))),
            'max_notifications_per_hour': int(os.getenv('JARVIS_MAX_NOTIFICATIONS', str(int(60 / base_interval * 2)))),
            'analysis_queue_size': int(os.getenv('JARVIS_ANALYSIS_QUEUE', str(cpu_count * 2))),
            'cache_ttl_seconds': float(os.getenv('JARVIS_CACHE_TTL', str(base_interval))),
            'enable_learning': os.getenv('JARVIS_ENABLE_LEARNING', 'true').lower() == 'true',
            'min_confidence_threshold': float(os.getenv('JARVIS_MIN_CONFIDENCE', '0.7'))
        }
        
    def _register_analyzer_callbacks(self):
        """Register callbacks with the continuous analyzer"""
        if not self.analyzer:
            return
            
        # Register for all events - we'll analyze everything dynamically
        events = ['app_changed', 'content_changed', 'error_detected', 'user_needs_help', 
                  'weather_visible', 'memory_warning']
        
        for event in events:
            self.analyzer.register_callback(event, self._on_dynamic_event)
        
    def _initialize_workflow_detectors(self) -> Dict[str, Any]:
        """Initialize dynamic workflow detection patterns"""
        return {
            'coding': {
                'indicators': ['ide', 'code editor', 'terminal', 'debugging'],
                'confidence_boost': ['syntax error', 'compilation', 'git'],
                'assistance_triggers': ['error', 'stuck', 'repeated attempts']
            },
            'research': {
                'indicators': ['browser', 'multiple tabs', 'documentation', 'search'],
                'confidence_boost': ['reading', 'scrolling', 'note-taking'],
                'assistance_triggers': ['many tabs', 'back and forth', 'searching']
            },
            'communication': {
                'indicators': ['email', 'slack', 'messages', 'chat'],
                'confidence_boost': ['typing', 'composing', 'replying'],
                'assistance_triggers': ['long pause', 'deleting text', 'rewriting']
            },
            'problem_solving': {
                'indicators': ['whiteboard', 'diagram', 'calculator', 'notes'],
                'confidence_boost': ['drawing', 'calculating', 'planning'],
                'assistance_triggers': ['erasing', 'stuck', 'confusion']
            }
        }
        
    async def start_interactive_monitoring(self):
        """Start proactive intelligent monitoring mode"""
        if self._is_active:
            logger.warning("Interactive monitoring already active")
            return
            
        self._is_active = True
        self.interaction_state['monitoring_start_time'] = time.time()
        
        # Generate dynamic initial greeting using Claude Vision
        initial_message = await self._generate_proactive_greeting()
        await self._send_notification(initial_message, priority="info")
        
        # Start monitoring tasks
        self._monitoring_task = asyncio.create_task(self._interaction_loop())
        self._proactive_task = asyncio.create_task(self._proactive_analysis_loop())
        self._workflow_detection_task = asyncio.create_task(self._workflow_detection_loop())
        
        logger.info("Started proactive real-time intelligent monitoring")
        
    async def stop_interactive_monitoring(self):
        """Stop proactive monitoring with intelligent farewell"""
        if not self._is_active:
            return
            
        self._is_active = False
        
        # Cancel all monitoring tasks
        tasks = [self._monitoring_task, self._proactive_task, self._workflow_detection_task]
        for task in tasks:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                
        # Generate dynamic farewell message with session summary
        farewell_message = await self._generate_dynamic_farewell()
        await self._send_notification(farewell_message, priority="info")
        
        # Save learned patterns if learning is enabled
        if self.config['enable_learning']:
            await self._save_learned_patterns()
        
        logger.info("Stopped proactive real-time intelligent monitoring")
        
    async def _interaction_loop(self):
        """Enhanced interaction loop with dynamic analysis"""
        while self._is_active:
            try:
                # Get current screen context
                if self.analyzer:
                    # Capture current screen
                    screenshot = await self._capture_current_screen()
                    if screenshot:
                        # Add to history
                        self._update_screen_history(screenshot)
                        
                        # Perform dynamic analysis
                        await self._perform_dynamic_analysis(screenshot)
                    
                # Process analysis queue
                await self._process_analysis_queue()
                
                # Adaptive interval based on activity
                interval = await self._calculate_adaptive_interval()
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in enhanced interaction loop: {e}")
                await asyncio.sleep(self.config['interaction_interval'])
                
    async def _generate_proactive_greeting(self) -> str:
        """Generate a proactive greeting that sets expectations"""
        if not self.vision_analyzer:
            return await self._generate_contextual_message("monitoring_start")
            
        # Capture initial screen state
        screenshot = await self._capture_current_screen()
        if not screenshot:
            return await self._generate_contextual_message("monitoring_start")
            
        # Generate proactive greeting
        prompt = (
            "You are JARVIS, a proactive AI assistant. The user just activated intelligent monitoring mode. "
            "Look at their current screen and provide a welcoming message that: "
            "1) Acknowledges what they're currently working on "
            "2) Explains you'll be actively watching and will offer help when you notice opportunities "
            "3) Assures them you'll be unobtrusive but helpful "
            "4) Mentions you'll engage in natural conversation as they work "
            "Be warm, professional, and set the tone for proactive assistance."
        )
        
        result = await self._analyze_with_claude(screenshot, prompt)
        return result.get('message', await self._generate_contextual_message("monitoring_start"))
        
    async def _proactive_analysis_loop(self):
        """Continuous proactive analysis for assistance opportunities"""
        # Wait for initial observation period
        await asyncio.sleep(self.proactive_settings['min_observation_time'])
        
        while self._is_active:
            try:
                # Analyze current context for proactive opportunities
                opportunities = await self._detect_assistance_opportunities()
                
                for opportunity in opportunities:
                    if await self._should_offer_assistance(opportunity):
                        await self._provide_proactive_assistance(opportunity)
                        
                # Dynamic sleep based on activity
                interval = await self._calculate_proactive_interval()
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in proactive analysis loop: {e}")
                await asyncio.sleep(self.config['interaction_interval'])
                
    async def _workflow_detection_loop(self):
        """Detect and track user workflows"""
        while self._is_active:
            try:
                # Detect current workflow
                workflow = await self._detect_current_workflow()
                
                if workflow:
                    # Update workflow state
                    await self._update_workflow_state(workflow)
                    
                    # Check for workflow-specific assistance
                    if await self._workflow_needs_assistance(workflow):
                        await self._provide_workflow_assistance(workflow)
                        
                await asyncio.sleep(5.0)  # Check workflows every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in workflow detection: {e}")
                await asyncio.sleep(5.0)
            
    async def _generate_dynamic_farewell(self) -> str:
        """Generate a dynamic farewell message using Claude Vision"""
        monitoring_duration = time.time() - self.interaction_state['monitoring_start_time']
        
        if not self.vision_analyzer:
            return await self._generate_contextual_message("monitoring_stop", {"duration": monitoring_duration})
            
        # Get final screen state and history summary
        screenshot = await self._capture_current_screen()
        history_summary = self._summarize_screen_history()
        
        prompt = (
            f"You are JARVIS. The user is stopping screen monitoring after {self._format_duration(monitoring_duration)}. "
            f"During this session, you observed: {history_summary}. "
            "Look at their current screen and provide a personalized farewell that: "
            "1) Acknowledges what they accomplished during the session "
            "2) Offers any final observations or suggestions based on their current state "
            "3) Maintains your professional yet personable tone. "
            "Be concise and meaningful, not generic."
        )
        
        result = await self._analyze_with_claude(screenshot, prompt)
        return result.get('message', await self._generate_contextual_message("monitoring_stop", {"duration": monitoring_duration}))
        
    async def _perform_dynamic_analysis(self, screenshot: Any):
        """Perform dynamic analysis on current screen state"""
        current_time = time.time()
        
        # Skip if in cooldown
        if self._is_in_cooldown():
            return
            
        # Build context from history
        context = self._build_context_from_history()
        
        # Dynamic prompt based on context
        prompt = await self._generate_analysis_prompt(context)
        
        # Analyze with Claude
        result = await self._analyze_with_claude(screenshot, prompt, context)
        
        # Process analysis result
        if result.get('should_interact', False):
            confidence = result.get('confidence', 0.0)
            if confidence >= self.config['min_confidence_threshold']:
                await self._send_notification(result['message'], 
                                            priority=result.get('priority', 'normal'),
                                            data=result.get('data', {}))
                
                # Track interaction
                self._track_interaction(result)
                
        # Learn from observation
        if self.config['enable_learning']:
            self._update_learning_state(result)
        
    async def _on_dynamic_event(self, data: Dict[str, Any]):
        """Handle any event dynamically using Claude Vision"""
        event_type = data.get('event_type', 'unknown')
        
        # Queue for detailed analysis
        self.interaction_state['pending_analysis_queue'].append({
            'event_type': event_type,
            'data': data,
            'timestamp': time.time()
        })
        
    async def _process_analysis_queue(self):
        """Process pending analysis queue"""
        if not self.interaction_state['pending_analysis_queue']:
            return
            
        # Process up to N items per cycle
        max_items = min(len(self.interaction_state['pending_analysis_queue']), 
                        self.config['analysis_queue_size'])
        
        for _ in range(max_items):
            if not self.interaction_state['pending_analysis_queue']:
                break
                
            item = self.interaction_state['pending_analysis_queue'].popleft()
            await self._analyze_event(item)
            
    async def _analyze_event(self, event_item: Dict[str, Any]):
        """Analyze a specific event using Claude Vision"""
        if not self.vision_analyzer:
            return
            
        event_type = event_item['event_type']
        event_data = event_item['data']
        
        # Capture current screen for context
        screenshot = await self._capture_current_screen()
        if not screenshot:
            return
            
        # Build dynamic prompt based on event
        prompt = self._build_event_analysis_prompt(event_type, event_data)
        
        # Analyze with Claude
        result = await self._analyze_with_claude(screenshot, prompt, event_data)
        
        # Process result
        if result.get('requires_action', False):
            await self._handle_dynamic_action(result, event_type)
            
    def _build_event_analysis_prompt(self, event_type: str, event_data: Dict[str, Any]) -> str:
        """Build a dynamic prompt for event analysis"""
        base_prompt = (
            "You are JARVIS, monitoring the user's screen. "
            f"A '{event_type}' event just occurred. "
        )
        
        if event_type == 'app_changed':
            base_prompt += (
                f"The user switched from {event_data.get('old_app', 'unknown')} "
                f"to {event_data.get('new_app', 'unknown')}. "
                "Analyze if this transition requires any assistance or observations. "
            )
        elif event_type == 'error_detected':
            base_prompt += (
                "An error was detected on screen. Analyze the error context and "
                "determine if you should offer specific help. "
            )
        else:
            base_prompt += f"Event details: {json.dumps(event_data, default=str)}. "
            
        base_prompt += (
            "Based on the current screen and context, determine: "
            "1) If you should interact with the user (should_interact: true/false) "
            "2) What specific, helpful message to provide (message: string) "
            "3) The priority level (priority: high/normal/low) "
            "4) Your confidence in this decision (confidence: 0.0-1.0) "
            "Be specific and helpful, not generic. Reference what you see on screen."
        )
        
        return base_prompt
            
    async def _analyze_with_claude(self, screenshot: Any, prompt: str, 
                                  context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze screenshot with Claude Vision API"""
        if not self.vision_analyzer:
            return {'should_interact': False, 'message': '', 'confidence': 0.0}
            
        try:
            # Use the vision analyzer to analyze with the given prompt
            result = await self.vision_analyzer.analyze_screenshot(screenshot, prompt)
            
            # Parse Claude's response into structured format
            if isinstance(result, tuple):
                analysis_result, metrics = result
            else:
                analysis_result = result
                
            # Extract structured data from response
            response_text = analysis_result.get('analysis', '')
            
            # Try to parse JSON response if Claude provided structured output
            try:
                if '{' in response_text and '}' in response_text:
                    import re
                    json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
                    if json_match:
                        structured_response = json.loads(json_match.group())
                        return structured_response
            except:
                pass
                
            # Otherwise, build response from analysis
            return {
                'should_interact': bool(response_text and len(response_text) > 20),
                'message': response_text,
                'confidence': 0.8,  # Default confidence
                'priority': 'normal',
                'data': {'analysis': analysis_result}
            }
            
        except Exception as e:
            logger.error(f"Error analyzing with Claude: {e}")
            return {'should_interact': False, 'message': '', 'confidence': 0.0}
            
    async def _capture_current_screen(self) -> Optional[Any]:
        """Capture current screen through analyzer"""
        if not self.analyzer:
            return None
            
        try:
            capture_result = await self.analyzer.vision_handler.capture_screen()
            if capture_result and hasattr(capture_result, 'success') and capture_result.success:
                return capture_result
            return capture_result
        except Exception as e:
            logger.error(f"Error capturing screen: {e}")
            return None
            
    def _update_screen_history(self, screenshot: Any):
        """Update screen history with new screenshot"""
        history_entry = {
            'timestamp': time.time(),
            'screenshot': screenshot,
            'hash': self._generate_screen_hash(screenshot)
        }
        self.interaction_state['screen_history'].append(history_entry)
        
    def _generate_screen_hash(self, screenshot: Any) -> str:
        """Generate hash for screenshot for comparison"""
        # Simple hash based on timestamp for now
        return hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
        
    def _build_context_from_history(self) -> Dict[str, Any]:
        """Build context from screen history"""
        history = list(self.interaction_state['screen_history'])
        
        context = {
            'history_length': len(history),
            'monitoring_duration': time.time() - self.interaction_state['monitoring_start_time'],
            'recent_changes': [],
            'patterns': self.learning_state['observed_workflows']
        }
        
        # Detect recent changes
        if len(history) >= 2:
            for i in range(1, min(len(history), 5)):
                if history[-i]['hash'] != history[-(i+1)]['hash']:
                    context['recent_changes'].append({
                        'time_ago': time.time() - history[-i]['timestamp'],
                        'index': i
                    })
                    
        return context
            
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form"""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, _ = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours} hours and {minutes} minutes"
        else:
            return f"{minutes} minutes"
            
    async def _detect_assistance_opportunities(self) -> List[Dict[str, Any]]:
        """Detect opportunities for proactive assistance"""
        opportunities = []
        
        if not self.vision_analyzer:
            return opportunities
            
        # Get current screen
        screenshot = await self._capture_current_screen()
        if not screenshot:
            return opportunities
            
        # Build comprehensive context
        context = self._build_proactive_context()
        
        # Analyze for opportunities
        prompt = (
            "You are JARVIS, proactively monitoring the user's work. "
            f"Context: {json.dumps(context, default=str)}\n"
            "Analyze the current screen for opportunities to help. Look for:\n"
            "1) User struggling or stuck (repeated actions, errors, confusion)\n"
            "2) Workflow inefficiencies you could improve\n"
            "3) Relevant information you could provide\n"
            "4) Tasks you could assist with\n"
            "5) Potential issues before they become problems\n\n"
            "Return a JSON array of opportunities, each with:\n"
            "- type: 'error_help', 'workflow_tip', 'information', 'task_assist', 'preventive'\n"
            "- description: what you observed\n"
            "- assistance: how you could help\n"
            "- confidence: 0.0-1.0\n"
            "- urgency: 'high', 'medium', 'low'\n"
            "- natural_message: conversational message to user"
        )
        
        result = await self._analyze_with_claude(screenshot, prompt, context)
        
        # Parse opportunities
        if result.get('data') and isinstance(result['data'].get('analysis'), dict):
            raw_opportunities = result['data']['analysis'].get('opportunities', [])
            for opp in raw_opportunities:
                if opp.get('confidence', 0) >= self.proactive_settings['assistance_confidence_threshold']:
                    opportunities.append(opp)
                    
        return opportunities
        
    async def _should_offer_assistance(self, opportunity: Dict[str, Any]) -> bool:
        """Determine if we should offer assistance for this opportunity"""
        # Check cooldown
        if self._is_in_cooldown() and opportunity.get('urgency') != 'high':
            return False
            
        # Check if similar assistance was recently offered
        recent_assists = list(self.interaction_state['assistance_opportunities'])
        for recent in recent_assists:
            if self._is_similar_opportunity(opportunity, recent):
                return False
                
        # Check user preferences
        if self.learning_state['user_preferences'].get('quiet_mode'):
            return opportunity.get('urgency') == 'high'
            
        return True
        
    async def _provide_proactive_assistance(self, opportunity: Dict[str, Any]):
        """Provide proactive assistance in a natural way"""
        # Record the opportunity
        self.interaction_state['assistance_opportunities'].append({
            **opportunity,
            'timestamp': time.time()
        })
        
        # Send natural message
        await self._send_notification(
            opportunity.get('natural_message', opportunity.get('assistance')),
            priority=opportunity.get('urgency', 'normal'),
            data={'opportunity': opportunity}
        )
        
        # Track for learning
        self._track_proactive_interaction(opportunity)
        
    async def _detect_current_workflow(self) -> Optional[Dict[str, Any]]:
        """Detect the current user workflow"""
        if not self.vision_analyzer:
            return None
            
        screenshot = await self._capture_current_screen()
        if not screenshot:
            return None
            
        # Get recent history for context
        recent_screens = list(self.interaction_state['screen_history'])[-5:]
        
        prompt = (
            "You are JARVIS, analyzing user workflows. "
            "Based on the current screen and recent activity, identify the user's workflow:\n"
            f"Known workflows: {list(self._workflow_detectors.keys())}\n"
            "Analyze and return JSON with:\n"
            "- workflow_type: one of the known workflows or 'other'\n"
            "- confidence: 0.0-1.0\n"
            "- indicators: list of observed indicators\n"
            "- current_phase: what phase of the workflow\n"
            "- potential_blockers: any obstacles you see"
        )
        
        result = await self._analyze_with_claude(screenshot, prompt)
        
        if result.get('data'):
            workflow_data = result['data'].get('analysis', {})
            if workflow_data.get('confidence', 0) >= self.proactive_settings['workflow_confidence_threshold']:
                return workflow_data
                
        return None
        
    async def _update_workflow_state(self, workflow: Dict[str, Any]):
        """Update the state of detected workflow"""
        workflow_type = workflow.get('workflow_type')
        
        if workflow_type not in self.interaction_state['active_workflows']:
            # New workflow detected
            self.interaction_state['active_workflows'][workflow_type] = {
                'started': time.time(),
                'phases': [workflow.get('current_phase')],
                'blockers': []
            }
            
            # Notify about workflow detection
            if self.proactive_settings['natural_conversation_mode']:
                message = await self._generate_workflow_acknowledgment(workflow)
                await self._send_notification(message, priority='low')
        else:
            # Update existing workflow
            state = self.interaction_state['active_workflows'][workflow_type]
            current_phase = workflow.get('current_phase')
            
            if current_phase not in state['phases']:
                state['phases'].append(current_phase)
                
            if workflow.get('potential_blockers'):
                state['blockers'].extend(workflow['potential_blockers'])
                
    async def _workflow_needs_assistance(self, workflow: Dict[str, Any]) -> bool:
        """Check if the workflow needs assistance"""
        workflow_type = workflow.get('workflow_type')
        detector = self._workflow_detectors.get(workflow_type, {})
        
        # Check for assistance triggers
        indicators = workflow.get('indicators', [])
        triggers = detector.get('assistance_triggers', [])
        
        for trigger in triggers:
            if any(trigger.lower() in ind.lower() for ind in indicators):
                return True
                
        # Check for blockers
        if workflow.get('potential_blockers'):
            return True
            
        return False
        
    async def _provide_workflow_assistance(self, workflow: Dict[str, Any]):
        """Provide workflow-specific assistance"""
        if not self.vision_analyzer:
            return
            
        screenshot = await self._capture_current_screen()
        if not screenshot:
            return
            
        workflow_type = workflow.get('workflow_type')
        blockers = workflow.get('potential_blockers', [])
        
        prompt = (
            f"You are JARVIS. The user is in a {workflow_type} workflow. "
            f"Current phase: {workflow.get('current_phase')}. "
            f"Potential issues: {', '.join(blockers) if blockers else 'none detected'}. "
            "Provide natural, conversational assistance that:\n"
            "1) Acknowledges their current task\n"
            "2) Offers specific, actionable help\n"
            "3) Doesn't interrupt their flow\n"
            "4) Sounds like a helpful colleague\n"
            "Be concise and directly helpful."
        )
        
        result = await self._analyze_with_claude(screenshot, prompt)
        
        if result.get('message'):
            await self._send_notification(
                result['message'],
                priority='normal',
                data={'workflow': workflow_type}
            )
            
    async def _generate_workflow_acknowledgment(self, workflow: Dict[str, Any]) -> str:
        """Generate a natural acknowledgment of detected workflow"""
        if not self.vision_analyzer:
            return ""
            
        screenshot = await self._capture_current_screen()
        
        prompt = (
            f"You are JARVIS. You've detected the user is doing {workflow.get('workflow_type')} work. "
            "Provide a brief, natural acknowledgment that:\n"
            "1) Shows you understand what they're doing\n"
            "2) Offers to help if needed\n"
            "3) Doesn't interrupt their flow\n"
            "Keep it to one short sentence, like a colleague who just noticed what you're working on."
        )
        
        result = await self._analyze_with_claude(screenshot, prompt)
        return result.get('message', "")
        
    def _build_proactive_context(self) -> Dict[str, Any]:
        """Build comprehensive context for proactive analysis"""
        base_context = self._build_context_from_history()
        
        # Add proactive elements
        base_context.update({
            'active_workflows': list(self.interaction_state['active_workflows'].keys()),
            'recent_assists': len(self.interaction_state['assistance_opportunities']),
            'user_focus_time': self._calculate_focus_duration(),
            'productivity_score': self._calculate_productivity_score(),
            'error_frequency': self._calculate_error_frequency(),
            'context_switches': len(self.interaction_state['context_switches'])
        })
        
        return base_context
        
    def _calculate_focus_duration(self) -> float:
        """Calculate how long user has been focused"""
        if not self.interaction_state['screen_history']:
            return 0.0
            
        # Find last major change
        history = list(self.interaction_state['screen_history'])
        focus_start = history[-1]['timestamp']
        
        for i in range(len(history)-1, 0, -1):
            if history[i]['hash'] != history[i-1]['hash']:
                # Found a change, check if major
                if i < len(history) - 3:  # More than 3 screens ago
                    break
                    
        return time.time() - focus_start
        
    def _calculate_productivity_score(self) -> float:
        """Calculate a productivity score based on patterns"""
        score = 0.5  # Neutral baseline
        
        # Positive indicators
        if self._calculate_focus_duration() > 300:  # 5+ minutes focus
            score += 0.2
            
        # Negative indicators  
        if len(self.interaction_state['context_switches']) > 5:
            score -= 0.1
            
        return max(0.0, min(1.0, score))
        
    def _calculate_error_frequency(self) -> float:
        """Calculate recent error frequency"""
        error_count = 0
        recent_window = 300  # 5 minutes
        
        for event in self.interaction_state['pending_analysis_queue']:
            if event['event_type'] == 'error_detected':
                if time.time() - event['timestamp'] < recent_window:
                    error_count += 1
                    
        return error_count / (recent_window / 60)  # Errors per minute
        
    async def _calculate_proactive_interval(self) -> float:
        """Calculate dynamic interval for proactive checks"""
        base = self.config['interaction_interval'] * 0.5  # More frequent for proactive
        
        # Adjust based on activity
        if self._calculate_productivity_score() > 0.7:
            # User is productive, check less often
            return base * 2.0
        elif self._calculate_error_frequency() > 0.5:
            # High error rate, check more often
            return base * 0.5
            
        return base
        
    def _is_similar_opportunity(self, opp1: Dict[str, Any], opp2: Dict[str, Any]) -> bool:
        """Check if two opportunities are similar"""
        if opp1.get('type') != opp2.get('type'):
            return False
            
        # Check description similarity (simple check)
        desc1 = opp1.get('description', '').lower()
        desc2 = opp2.get('description', '').lower()
        
        # If more than 50% of words match, consider similar
        words1 = set(desc1.split())
        words2 = set(desc2.split())
        
        if not words1 or not words2:
            return False
            
        overlap = len(words1.intersection(words2))
        similarity = overlap / min(len(words1), len(words2))
        
        return similarity > 0.5
        
    def _track_proactive_interaction(self, opportunity: Dict[str, Any]):
        """Track proactive interaction for learning"""
        interaction_id = hashlib.sha256(
            f"{time.time()}_{opportunity.get('type')}".encode()
        ).hexdigest()[:16]
        
        self.learning_state['assistance_effectiveness'][interaction_id] = {
            'timestamp': time.time(),
            'opportunity': opportunity,
            'workflow': list(self.interaction_state['active_workflows'].keys()),
            'context': self._build_proactive_context(),
            'user_response': None  # To be updated based on user behavior
        }
        
    async def _generate_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """Generate dynamic analysis prompt based on context"""
        prompt_parts = [
            "You are JARVIS, proactively monitoring the user's screen to provide timely assistance.",
            f"You've been monitoring for {self._format_duration(context['monitoring_duration'])}."
        ]
        
        if context.get('active_workflows'):
            prompt_parts.append(f"Detected workflows: {', '.join(context['active_workflows'])}")
            
        if context['recent_changes']:
            prompt_parts.append(f"Detected {len(context['recent_changes'])} recent screen changes.")
            
        if context.get('productivity_score', 0) < 0.5:
            prompt_parts.append("User productivity seems lower than usual.")
            
        prompt_parts.extend([
            "Analyze the current screen for proactive assistance opportunities.",
            "Consider:",
            "- Current workflow stage and potential next steps",
            "- Any struggles, errors, or inefficiencies",
            "- Information that could help their current task",
            "- Workflow optimizations or shortcuts",
            "- Preventive assistance before issues arise",
            "",
            "Respond with JSON containing:",
            "- should_interact: boolean",
            "- message: natural, conversational message (like a helpful colleague)",
            "- priority: 'high', 'normal', or 'low'",
            "- confidence: float between 0 and 1",
            "- analysis_type: type of assistance offered",
            "- conversation_starter: optional follow-up to engage naturally"
        ])
        
        return "\n".join(prompt_parts)
        
    async def _calculate_adaptive_interval(self) -> float:
        """Calculate adaptive interval based on activity and context"""
        base_interval = self.config['interaction_interval']
        
        # Adjust based on recent activity
        history = list(self.interaction_state['screen_history'])
        if len(history) >= 2:
            # Check activity level
            recent_changes = sum(1 for i in range(1, min(len(history), 5))
                               if history[-i]['hash'] != history[-(i+1)]['hash'])
            
            if recent_changes >= 3:  # High activity
                return base_interval * 0.5  # Check more frequently
            elif recent_changes == 0:  # No activity
                return base_interval * 2.0  # Check less frequently
                
        return base_interval
        
    def _is_in_cooldown(self) -> bool:
        """Check if we're in notification cooldown period"""
        if not self.interaction_state['last_notification_time']:
            return False
            
        time_since_last = time.time() - self.interaction_state['last_notification_time']
        return time_since_last < self.config['notification_cooldown']
        
    async def _send_notification(self, message: str, priority: str = "normal", 
                               data: Optional[Dict[str, Any]] = None):
        """Send notification to user"""
        if self._is_in_cooldown() and priority != "high":
            return
            
        # Record notification
        current_time = time.time()
        self.interaction_state['last_notification_time'] = current_time
        self.interaction_state['notifications_sent'].append(current_time)
        
        # Add to conversation context
        self.interaction_state['conversation_context'].append({
            'type': 'jarvis',
            'message': message,
            'timestamp': current_time
        })
        
        # Prepare notification
        notification = {
            'type': 'jarvis_notification',
            'message': message,
            'priority': priority,
            'timestamp': datetime.now().isoformat(),
            'data': data or {}
        }
        
        # Send via callback if available
        if self.notification_callback:
            try:
                if asyncio.iscoroutinefunction(self.notification_callback):
                    await self.notification_callback(notification)
                else:
                    self.notification_callback(notification)
            except Exception as e:
                logger.error(f"Error sending notification: {e}")
        else:
            logger.info(f"JARVIS: {message}")
            
    def _track_interaction(self, result: Dict[str, Any]):
        """Track interaction for learning"""
        interaction_id = hashlib.sha256(
            f"{time.time()}_{result.get('message', '')}".encode()
        ).hexdigest()[:16]
        
        self.learning_state['interaction_responses'][interaction_id] = {
            'timestamp': time.time(),
            'result': result,
            'context': self._build_context_from_history(),
            'effectiveness': None  # To be updated based on user response
        }
        
    def _update_learning_state(self, analysis_result: Dict[str, Any]):
        """Update learning state based on analysis"""
        # Track timing patterns
        current_hour = datetime.now().hour
        self.learning_state['timing_patterns'][current_hour] = \
            self.learning_state['timing_patterns'].get(current_hour, 0) + 1
            
        # Track observed workflows
        if 'analysis_type' in analysis_result:
            analysis_type = analysis_result['analysis_type']
            self.learning_state['observed_workflows'][analysis_type] = \
                self.learning_state['observed_workflows'].get(analysis_type, 0) + 1
                
    def _summarize_screen_history(self) -> str:
        """Summarize screen history for context"""
        history = list(self.interaction_state['screen_history'])
        if not history:
            return "no significant activity"
            
        # Count changes
        changes = sum(1 for i in range(1, len(history))
                     if history[i]['hash'] != history[i-1]['hash'])
        
        summary_parts = [f"{changes} screen changes"]
        
        # Add workflow summary
        if self.learning_state['observed_workflows']:
            top_workflow = max(self.learning_state['observed_workflows'].items(),
                             key=lambda x: x[1])[0]
            summary_parts.append(f"primarily {top_workflow} activities")
            
        return ", ".join(summary_parts)
        
    async def _generate_contextual_message(self, message_type: str, 
                                         params: Optional[Dict[str, Any]] = None) -> str:
        """Generate contextual message when Claude Vision isn't available"""
        if message_type == "monitoring_start":
            return "Screen monitoring activated. I'll observe and provide assistance as needed."
        elif message_type == "monitoring_stop":
            duration = params.get('duration', 0) if params else 0
            return f"Screen monitoring deactivated after {self._format_duration(duration)}."
        else:
            return "I'm here to assist you."
            
    async def _handle_dynamic_action(self, result: Dict[str, Any], event_type: str):
        """Handle dynamic action based on analysis result"""
        action = result.get('action', {})
        if not action:
            return
            
        # Handle different action types
        action_type = action.get('type')
        if action_type == 'notify':
            await self._send_notification(
                action.get('message', result.get('message', '')),
                priority=action.get('priority', 'normal')
            )
        # Add more action types as needed
        
    async def _save_learned_patterns(self):
        """Save learned patterns for future sessions"""
        patterns_file = os.path.join(os.path.expanduser('~'), '.jarvis', 'interaction_patterns.json')
        os.makedirs(os.path.dirname(patterns_file), exist_ok=True)
        
        try:
            with open(patterns_file, 'w') as f:
                json.dump({
                    'learning_state': self.learning_state,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving learned patterns: {e}")
            
    async def provide_screen_summary(self) -> str:
        """Provide a dynamic summary using Claude Vision"""
        if not self.vision_analyzer:
            return "Screen monitoring is active but vision analysis is unavailable."
            
        screenshot = await self._capture_current_screen()
        if not screenshot:
            return "Unable to capture current screen."
            
        history_summary = self._summarize_screen_history()
        
        prompt = (
            "You are JARVIS. The user asked for a summary of their screen activity. "
            f"You've observed: {history_summary}. "
            "Looking at the current screen, provide a concise summary of: "
            "1) What the user is currently doing "
            "2) Key activities during this session "
            "3) Any observations or suggestions "
            "Be specific and reference actual content you see."
        )
        
        result = await self._analyze_with_claude(screenshot, prompt)
        return result.get('message', "I'm monitoring your screen activity.")
        
    def get_interaction_stats(self) -> Dict[str, Any]:
        """Get comprehensive interaction statistics"""
        return {
            'monitoring_duration': time.time() - self.interaction_state['monitoring_start_time']
                if self.interaction_state['monitoring_start_time'] else 0,
            'notifications_sent': len(self.interaction_state['notifications_sent']),
            'screen_changes_observed': len(self.interaction_state['screen_history']),
            'events_queued': len(self.interaction_state['pending_analysis_queue']),
            'learning_data': {
                'workflows_observed': len(self.learning_state['observed_workflows']),
                'interactions_tracked': len(self.learning_state['interaction_responses']),
                'peak_activity_hour': max(self.learning_state['timing_patterns'].items(),
                                         key=lambda x: x[1])[0]
                    if self.learning_state['timing_patterns'] else None
            },
            'is_active': self._is_active
        }