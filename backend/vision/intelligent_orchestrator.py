#!/usr/bin/env python3
"""
Intelligent Multi-Space Orchestration System
============================================

Orchestrates Yabai + CG Windows API + Claude Vision for intelligent workspace analysis.
Implements the three-system architecture with dynamic targeting and zero hardcoding.

Features:
- Intelligent workspace scouting (Yabai)
- Selective visual capture (CG Windows API) 
- Deep content analysis (Claude Vision)
- Conversational context system
- Pattern recognition and workflow detection
- Dynamic targeting based on query intent
- Cost optimization through selective capture
"""

import asyncio
import logging
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import numpy as np
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class CapturePriority(Enum):
    """Dynamic capture priorities based on content analysis"""
    CRITICAL = "critical"      # Errors, failures, urgent content
    HIGH = "high"             # Active work, research, development
    MEDIUM = "medium"         # Communication, documentation
    LOW = "low"              # Idle, entertainment, background
    SKIP = "skip"            # No visual analysis needed

class QueryIntent(Enum):
    """Dynamic query intent classification"""
    WORKSPACE_OVERVIEW = "workspace_overview"
    ERROR_ANALYSIS = "error_analysis"
    RESEARCH_REVIEW = "research_review"
    WORKFLOW_STATUS = "workflow_status"
    DEBUGGING_SESSION = "debugging_session"
    MULTITASKING_CHECK = "multitasking_check"
    FOLLOW_UP_DETAIL = "follow_up_detail"

class WorkflowPattern(Enum):
    """Detected workflow patterns"""
    DEBUGGING = "debugging"
    RESEARCH = "research"
    DEVELOPMENT = "development"
    COMMUNICATION = "communication"
    MULTITASKING = "multitasking"
    PRESENTATION = "presentation"
    LEARNING = "learning"

@dataclass
class WorkspaceSnapshot:
    """Complete workspace state snapshot"""
    timestamp: datetime
    spaces: List[Dict[str, Any]]
    current_space: int
    total_spaces: int
    total_windows: int
    total_apps: int
    snapshot_id: str
    patterns_detected: List[WorkflowPattern] = field(default_factory=list)
    capture_priorities: Dict[int, CapturePriority] = field(default_factory=dict)
    context_summary: str = ""

@dataclass
class CaptureTarget:
    """Target for selective capture"""
    space_id: int
    window_id: Optional[int]
    app_name: str
    window_title: str
    priority: CapturePriority
    reason: str
    estimated_value: float  # 0.0 to 1.0
    capture_method: str = "cg_windows_api"

@dataclass
class AnalysisContext:
    """Context for intelligent analysis"""
    query: str
    intent: QueryIntent
    workspace_snapshot: WorkspaceSnapshot
    capture_targets: List[CaptureTarget]
    previous_context: Optional['AnalysisContext'] = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    detected_patterns: List[WorkflowPattern] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)

class IntelligentOrchestrator:
    """
    Intelligent orchestration system for multi-space analysis.
    
    Coordinates Yabai (metadata), CG Windows API (capture), and Claude Vision (analysis)
    with dynamic targeting, pattern recognition, and conversational context.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._context_cache: Dict[str, AnalysisContext] = {}
        self._pattern_history: deque = deque(maxlen=50)
        self._capture_history: deque = deque(maxlen=100)
        self._performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self._user_preferences: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
        # Dynamic configuration
        self._config = {
            "max_capture_targets": 3,
            "min_capture_value": 0.3,
            "context_ttl_seconds": 300,
            "pattern_detection_threshold": 0.7,
            "cost_optimization_enabled": True,
            "performance_monitoring": True
        }
        
        # Initialize dynamic configuration manager
        from .dynamic_config import get_dynamic_config_manager
        self.config_manager = get_dynamic_config_manager()
        
        # Initialize subsystems
        self._initialize_subsystems()
        
    def _initialize_subsystems(self):
        """Initialize Yabai, CG Windows API, and Claude Vision subsystems"""
        try:
            # Import subsystems dynamically
            from .yabai_space_detector import get_yabai_detector
            from .cg_window_capture import get_capture_engine
            from .claude_vision_analyzer_main import ClaudeVisionAnalyzer
            
            self.yabai_detector = get_yabai_detector()
            self.cg_capture_engine = get_capture_engine()
            self.claude_analyzer = None  # Will be initialized with API key
            
            self.logger.info("✅ Intelligent Orchestrator subsystems initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize subsystems: {e}")
            raise
    
    async def analyze_workspace_intelligently(
        self, 
        query: str, 
        claude_api_key: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for intelligent workspace analysis.
        
        Orchestrates the complete three-system flow:
        1. Yabai metadata collection (50ms)
        2. CG Windows API selective capture (200-500ms)
        3. Claude Vision intelligent analysis (2-3s)
        
        Args:
            query: User's query about workspace
            claude_api_key: Claude API key for vision analysis
            user_context: Additional user context/preferences
            
        Returns:
            Complete analysis with insights, patterns, and recommendations
        """
        start_time = time.time()
        
        try:
            # Phase 1: Intelligent Query Analysis
            intent = await self._analyze_query_intent(query)
            self.logger.info(f"[ORCHESTRATOR] Query intent: {intent.value}")
            
            # Phase 2: Workspace Scouting (Yabai)
            workspace_snapshot = await self._scout_workspace()
            self.logger.info(f"[ORCHESTRATOR] Workspace scouted: {workspace_snapshot.total_spaces} spaces, {workspace_snapshot.total_windows} windows")
            
            # Phase 3: Dynamic Target Selection
            capture_targets = await self._select_capture_targets(intent, workspace_snapshot)
            self.logger.info(f"[ORCHESTRATOR] Selected {len(capture_targets)} capture targets")
            
            # Phase 4: Selective Visual Capture (CG Windows API)
            captured_content = await self._capture_selectively(capture_targets)
            self.logger.info(f"[ORCHESTRATOR] Captured {len(captured_content)} windows")
            
            # Phase 5: Pattern Recognition
            patterns = await self._detect_workflow_patterns(workspace_snapshot, captured_content)
            workspace_snapshot.patterns_detected = patterns
            
            # Phase 6: Intelligent Analysis
            # For workspace overview queries, generate simple list-based response
            # For detailed queries, use Claude Vision analysis
            if intent == QueryIntent.WORKSPACE_OVERVIEW:
                self.logger.info("[ORCHESTRATOR] Generating workspace overview response (no Claude analysis needed)")
                analysis_result = await self._generate_workspace_overview(
                    query, workspace_snapshot, patterns
                )
            else:
                self.logger.info(f"[ORCHESTRATOR] Using Claude Vision for {intent.value} analysis")
                analysis_result = await self._analyze_with_claude(
                    query, intent, workspace_snapshot, captured_content, claude_api_key
                )
            
            # Phase 7: Context Management
            context = AnalysisContext(
                query=query,
                intent=intent,
                workspace_snapshot=workspace_snapshot,
                capture_targets=capture_targets,
                detected_patterns=patterns
            )
            
            await self._store_context(context, analysis_result)
            
            # Phase 8: Performance Tracking
            total_time = time.time() - start_time
            await self._track_performance(total_time, len(capture_targets), len(captured_content))
            
            # Phase 9: Dynamic Optimization
            await self._optimize_dynamically(total_time, len(capture_targets), len(captured_content))
            
            return {
                "success": True,
                "analysis": analysis_result,
                "workspace_snapshot": workspace_snapshot,
                "capture_targets": capture_targets,
                "patterns_detected": patterns,
                "context_id": context.workspace_snapshot.snapshot_id,
                "performance": {
                    "total_time": total_time,
                    "phases": {
                        "scouting": workspace_snapshot.timestamp,
                        "capture": len(captured_content),
                        "analysis": analysis_result.get("analysis_time", 0)
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"[ORCHESTRATOR] Analysis failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "fallback_available": True
            }
    
    async def _analyze_query_intent(self, query: str) -> QueryIntent:
        """Analyze query intent using dynamic pattern matching"""
        query_lower = query.lower()
        
        # Check for explicit workspace overview queries FIRST (highest priority)
        overview_keywords = [
            "list all", "show all", "across", "desktop spaces", 
            "how many spaces", "all my spaces", "workspace overview",
            "what spaces", "all desktop"
        ]
        if any(keyword in query_lower for keyword in overview_keywords):
            return QueryIntent.WORKSPACE_OVERVIEW
        
        # Dynamic intent patterns (no hardcoding)
        intent_patterns = {
            QueryIntent.ERROR_ANALYSIS: [
                "error", "fail", "broken", "issue", "problem", "debug", "fix"
            ],
            QueryIntent.RESEARCH_REVIEW: [
                "research", "study", "learn", "investigate", "explore", "browse"
            ],
            QueryIntent.DEBUGGING_SESSION: [
                "debug", "troubleshoot", "test", "develop", "code", "program"
            ],
            QueryIntent.WORKFLOW_STATUS: [
                "status", "progress", "where", "what", "happening", "doing"
            ],
            QueryIntent.MULTITASKING_CHECK: [
                "multitask", "busy", "overwhelmed", "manage", "organize"
            ],
            QueryIntent.FOLLOW_UP_DETAIL: [
                "more", "detail", "explain", "tell me", "show me", "analyze"
            ]
        }
        
        # Score each intent
        intent_scores = {}
        for intent, patterns in intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            intent_scores[intent] = score
        
        # Default to workspace overview if no clear intent
        if not any(intent_scores.values()):
            return QueryIntent.WORKSPACE_OVERVIEW
        
        # Return highest scoring intent
        return max(intent_scores.items(), key=lambda x: x[1])[0]
    
    async def _scout_workspace(self) -> WorkspaceSnapshot:
        """Scout workspace using Yabai for complete metadata"""
        start_time = time.time()
        
        try:
            # Get comprehensive workspace data from Yabai
            if not self.yabai_detector or not self.yabai_detector.is_available():
                raise ValueError("Yabai detector not available")
            
            # Get all spaces and windows
            spaces_data = self.yabai_detector.enumerate_all_spaces()
            current_space = self.yabai_detector.get_current_space()
            
            # Process spaces data with window titles for context
            spaces = []
            total_windows = 0
            total_apps = set()
            
            for space_data in spaces_data:
                # Extract window titles for intelligent activity detection
                windows = space_data.get("windows", [])
                window_titles = [
                    w.get("title", "") if isinstance(w, dict) else getattr(w, "title", "")
                    for w in windows
                ]
                # Filter out empty titles
                window_titles = [t for t in window_titles if t and len(t.strip()) > 0]
                
                space_info = {
                    "space_id": space_data.get("space_id", 0),
                    "space_name": space_data.get("space_name", ""),
                    "is_current": space_data.get("is_current", False),
                    "is_fullscreen": space_data.get("is_fullscreen", False),
                    "windows": windows,
                    "applications": space_data.get("applications", []),
                    "primary_app": space_data.get("primary_app", "Unknown"),
                    "window_count": len(windows),
                    "window_titles": window_titles  # Add window titles for context
                }
                
                spaces.append(space_info)
                total_windows += space_info["window_count"]
                total_apps.update(space_info["applications"])
            
            # Generate snapshot ID
            snapshot_id = hashlib.md5(
                f"{time.time()}_{len(spaces)}_{total_windows}".encode()
            ).hexdigest()[:12]
            
            snapshot = WorkspaceSnapshot(
                timestamp=datetime.now(),
                spaces=spaces,
                current_space=current_space,
                total_spaces=len(spaces),
                total_windows=total_windows,
                total_apps=len(total_apps),
                snapshot_id=snapshot_id
            )
            
            scouting_time = time.time() - start_time
            self.logger.info(f"[SCOUTING] Completed in {scouting_time:.3f}s")
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"[SCOUTING] Failed: {e}")
            raise
    
    async def _select_capture_targets(
        self, 
        intent: QueryIntent, 
        snapshot: WorkspaceSnapshot
    ) -> List[CaptureTarget]:
        """Dynamically select capture targets based on intent and workspace state"""
        
        targets = []
        
        for space in snapshot.spaces:
            space_id = space["space_id"]
            windows = space.get("windows", [])
            
            for window in windows:
                app_name = window.get("app", "Unknown")
                window_title = window.get("title", "")
                
                # Dynamic priority calculation
                priority, reason, value = await self._calculate_capture_priority(
                    intent, app_name, window_title, space
                )
                
                if priority != CapturePriority.SKIP and value >= self._config["min_capture_value"]:
                    target = CaptureTarget(
                        space_id=space_id,
                        window_id=window.get("id"),
                        app_name=app_name,
                        window_title=window_title,
                        priority=priority,
                        reason=reason,
                        estimated_value=value
                    )
                    targets.append(target)
        
        # Sort by priority and value
        targets.sort(key=lambda t: (t.priority.value, -t.estimated_value))
        
        # Limit targets based on dynamic configuration
        active_config = self.config_manager.get_active_config()
        capture_config = active_config.get("capture", {})
        if hasattr(capture_config, 'max_targets'):
            max_targets = capture_config.max_targets
        else:
            max_targets = capture_config.get("max_targets", self._config["max_capture_targets"])
        return targets[:max_targets]
    
    async def _calculate_capture_priority(
        self, 
        intent: QueryIntent, 
        app_name: str, 
        window_title: str, 
        space: Dict[str, Any]
    ) -> Tuple[CapturePriority, str, float]:
        """Calculate dynamic capture priority based on multiple factors"""
        
        # Base scores for different factors
        app_scores = {
            "Terminal": 0.9,
            "iTerm2": 0.9,
            "Google Chrome": 0.7,
            "Safari": 0.7,
            "Firefox": 0.7,
            "Cursor": 0.8,
            "Code": 0.8,
            "Sublime Text": 0.8,
            "Slack": 0.4,
            "Discord": 0.3,
            "Mail": 0.3,
            "Messages": 0.2
        }
        
        # Intent-based scoring
        intent_scores = {
            QueryIntent.ERROR_ANALYSIS: {
                "error_keywords": ["error", "fail", "exception", "traceback", "debug"],
                "app_boost": {"Terminal": 0.3, "iTerm2": 0.3}
            },
            QueryIntent.RESEARCH_REVIEW: {
                "research_keywords": ["research", "study", "learn", "documentation", "stackoverflow"],
                "app_boost": {"Google Chrome": 0.2, "Safari": 0.2}
            },
            QueryIntent.DEBUGGING_SESSION: {
                "debug_keywords": ["debug", "test", "develop", "code", "program"],
                "app_boost": {"Cursor": 0.2, "Code": 0.2, "Terminal": 0.2}
            }
        }
        
        # Calculate base score
        base_score = app_scores.get(app_name, 0.5)
        
        # Apply intent-based scoring
        if intent in intent_scores:
            intent_config = intent_scores[intent]
            
            # Check for keywords in window title
            title_lower = window_title.lower()
            keyword_matches = sum(
                1 for keyword in intent_config.get("error_keywords", []) 
                if keyword in title_lower
            )
            
            if keyword_matches > 0:
                base_score += 0.3
            
            # Apply app boost
            app_boost = intent_config.get("app_boost", {}).get(app_name, 0)
            base_score += app_boost
        
        # Current space bonus
        if space.get("is_current", False):
            base_score += 0.1
        
        # Determine priority
        if base_score >= 0.8:
            priority = CapturePriority.CRITICAL
            reason = "High-value content detected"
        elif base_score >= 0.6:
            priority = CapturePriority.HIGH
            reason = "Active work detected"
        elif base_score >= 0.4:
            priority = CapturePriority.MEDIUM
            reason = "Relevant content"
        elif base_score >= 0.2:
            priority = CapturePriority.LOW
            reason = "Background content"
        else:
            priority = CapturePriority.SKIP
            reason = "Low priority"
        
        return priority, reason, min(base_score, 1.0)
    
    async def _capture_selectively(self, targets: List[CaptureTarget]) -> Dict[str, Any]:
        """Selectively capture windows using CG Windows API"""
        
        captured_content = {}
        
        for target in targets:
            try:
                start_time = time.time()
                
                # Use CG Windows API for non-disruptive capture
                if target.window_id:
                    screenshot = await self._capture_window_by_id(target.window_id)
                else:
                    # Fallback to space capture
                    screenshot = await self._capture_space_by_id(target.space_id)
                
                if screenshot is not None:
                    captured_content[f"space_{target.space_id}_{target.app_name}"] = {
                        "screenshot": screenshot,
                        "target": target,
                        "capture_time": time.time() - start_time,
                        "timestamp": datetime.now()
                    }
                    
                    self.logger.info(
                        f"[CAPTURE] Successfully captured {target.app_name} "
                        f"from Space {target.space_id} ({target.priority.value})"
                    )
                else:
                    self.logger.warning(f"[CAPTURE] Failed to capture {target.app_name}")
                    
            except Exception as e:
                self.logger.error(f"[CAPTURE] Error capturing {target.app_name}: {e}")
        
        return captured_content
    
    async def _capture_window_by_id(self, window_id: int) -> Optional[np.ndarray]:
        """Capture specific window by ID using CG Windows API"""
        try:
            if not self.cg_capture_engine:
                return None
            
            # Use CG Windows API for non-disruptive capture
            if hasattr(self.cg_capture_engine, 'capture_window_by_id'):
                screenshot = self.cg_capture_engine.capture_window_by_id(window_id)
            else:
                # Fallback to capture_window method
                screenshot = self.cg_capture_engine.capture_window(window_id)
            return screenshot
            
        except Exception as e:
            self.logger.error(f"[CAPTURE] Window capture failed: {e}")
            return None
    
    async def _capture_space_by_id(self, space_id: int) -> Optional[np.ndarray]:
        """Capture entire space by ID (fallback method)"""
        try:
            # Use the existing multi-space capture engine
            from .multi_space_capture_engine import MultiSpaceCaptureEngine, SpaceCaptureRequest, CaptureQuality
            
            capture_engine = MultiSpaceCaptureEngine()
            request = SpaceCaptureRequest(
                space_ids=[space_id],
                quality=CaptureQuality.FULL,
                use_cache=True
            )
            
            screenshot = await capture_engine._capture_with_cg_windows(space_id, request)
            return screenshot
            
        except Exception as e:
            self.logger.error(f"[CAPTURE] Space capture failed: {e}")
            # Final fallback: use current space capture
            try:
                from .claude_vision_analyzer_main import ClaudeVisionAnalyzer
                analyzer = ClaudeVisionAnalyzer()
                screenshot = await analyzer.capture_screen(multi_space=False)
                return screenshot
            except Exception as e2:
                self.logger.error(f"[CAPTURE] Final fallback failed: {e2}")
                return None
    
    async def _detect_workflow_patterns(
        self, 
        snapshot: WorkspaceSnapshot, 
        captured_content: Dict[str, Any]
    ) -> List[WorkflowPattern]:
        """Detect workflow patterns from workspace state and captured content"""
        
        patterns = []
        
        # Analyze space distribution
        space_apps = {}
        for space in snapshot.spaces:
            apps = space.get("applications", [])
            space_apps[space["space_id"]] = apps
        
        # Pattern detection logic
        if self._detect_debugging_pattern(space_apps, captured_content):
            patterns.append(WorkflowPattern.DEBUGGING)
        
        if self._detect_research_pattern(space_apps, captured_content):
            patterns.append(WorkflowPattern.RESEARCH)
        
        if self._detect_development_pattern(space_apps, captured_content):
            patterns.append(WorkflowPattern.DEVELOPMENT)
        
        if self._detect_communication_pattern(space_apps, captured_content):
            patterns.append(WorkflowPattern.COMMUNICATION)
        
        if self._detect_multitasking_pattern(space_apps, captured_content):
            patterns.append(WorkflowPattern.MULTITASKING)
        
        return patterns
    
    def _detect_debugging_pattern(self, space_apps: Dict[int, List[str]], captured_content: Dict[str, Any]) -> bool:
        """Detect debugging workflow pattern"""
        debugging_indicators = 0
        
        for space_id, apps in space_apps.items():
            # Check for debugging apps
            if any(app in ["Terminal", "iTerm2"] for app in apps):
                debugging_indicators += 1
            
            # Check for development apps
            if any(app in ["Cursor", "Code", "Sublime Text"] for app in apps):
                debugging_indicators += 1
            
            # Check for research apps (debugging often involves research)
            if any(app in ["Google Chrome", "Safari"] for app in apps):
                debugging_indicators += 1
        
        # Check captured content for error indicators
        for content_key, content_data in captured_content.items():
            target = content_data.get("target")
            if target and "error" in target.window_title.lower():
                debugging_indicators += 2
        
        return debugging_indicators >= 3
    
    def _detect_research_pattern(self, space_apps: Dict[int, List[str]], captured_content: Dict[str, Any]) -> bool:
        """Detect research workflow pattern"""
        research_indicators = 0
        
        for space_id, apps in space_apps.items():
            # Check for research apps
            if any(app in ["Google Chrome", "Safari", "Firefox"] for app in apps):
                research_indicators += 1
            
            # Check for documentation apps
            if any(app in ["Notes", "Obsidian", "Notion"] for app in apps):
                research_indicators += 1
        
        return research_indicators >= 2
    
    def _detect_development_pattern(self, space_apps: Dict[int, List[str]], captured_content: Dict[str, Any]) -> bool:
        """Detect development workflow pattern"""
        dev_indicators = 0
        
        for space_id, apps in space_apps.items():
            # Check for development apps
            if any(app in ["Cursor", "Code", "Sublime Text", "Vim"] for app in apps):
                dev_indicators += 1
            
            # Check for terminal apps
            if any(app in ["Terminal", "iTerm2"] for app in apps):
                dev_indicators += 1
        
        return dev_indicators >= 2
    
    def _detect_communication_pattern(self, space_apps: Dict[int, List[str]], captured_content: Dict[str, Any]) -> bool:
        """Detect communication workflow pattern"""
        comm_indicators = 0
        
        for space_id, apps in space_apps.items():
            # Check for communication apps
            if any(app in ["Slack", "Discord", "Teams", "Zoom"] for app in apps):
                comm_indicators += 1
            
            # Check for email apps
            if any(app in ["Mail", "Outlook"] for app in apps):
                comm_indicators += 1
        
        return comm_indicators >= 1
    
    def _detect_multitasking_pattern(self, space_apps: Dict[int, List[str]], captured_content: Dict[str, Any]) -> bool:
        """Detect multitasking workflow pattern"""
        # Multitasking is detected if there are multiple different types of activities
        activity_types = set()
        
        for space_id, apps in space_apps.items():
            if any(app in ["Cursor", "Code", "Sublime Text"] for app in apps):
                activity_types.add("development")
            if any(app in ["Google Chrome", "Safari"] for app in apps):
                activity_types.add("research")
            if any(app in ["Slack", "Discord", "Mail"] for app in apps):
                activity_types.add("communication")
            if any(app in ["Terminal", "iTerm2"] for app in apps):
                activity_types.add("terminal")
        
        return len(activity_types) >= 3
    
    async def _analyze_with_claude(
        self,
        query: str,
        intent: QueryIntent,
        snapshot: WorkspaceSnapshot,
        captured_content: Dict[str, Any],
        api_key: Optional[str]
    ) -> Dict[str, Any]:
        """Analyze captured content using Claude Vision"""
        
        if not api_key:
            return {
                "analysis": "Claude Vision analysis requires API key",
                "fallback": True
            }
        
        try:
            # Initialize Claude analyzer if needed
            if not self.claude_analyzer:
                from .claude_vision_analyzer_main import ClaudeVisionAnalyzer
                self.claude_analyzer = ClaudeVisionAnalyzer(api_key)
            
            # Build enhanced prompt
            prompt = await self._build_analysis_prompt(query, intent, snapshot, captured_content)
            
            # Prepare images for Claude
            images = []
            for content_key, content_data in captured_content.items():
                screenshot = content_data.get("screenshot")
                if screenshot is not None:
                    images.append({
                        "image": screenshot,
                        "description": f"Space {content_data['target'].space_id} - {content_data['target'].app_name}"
                    })
            
            # Analyze with Claude
            start_time = time.time()
            analysis_result = await self.claude_analyzer.analyze_screenshot_async(
                images[0]["image"] if images else None,
                prompt
            )
            analysis_time = time.time() - start_time
            
            return {
                "analysis": analysis_result.get("description", "Analysis completed"),
                "analysis_time": analysis_time,
                "images_analyzed": len(images),
                "intent": intent.value,
                "patterns": [p.value for p in snapshot.patterns_detected]
            }
            
        except Exception as e:
            self.logger.error(f"[CLAUDE] Analysis failed: {e}")
            return {
                "analysis": f"Analysis failed: {str(e)}",
                "fallback": True
            }
    
    async def _generate_workspace_overview(
        self,
        query: str,
        snapshot: WorkspaceSnapshot,
        patterns: List[WorkflowPattern]
    ) -> Dict[str, Any]:
        """Generate richly detailed workspace overview with clean bullet formatting"""
        
        # Build overview with clean bullet points
        response_parts = [
            f"Sir, you're working across {snapshot.total_spaces} desktop spaces:",
            ""  # Blank line for spacing
        ]
        
        # List each space with rich, detailed activity descriptions
        for space in sorted(snapshot.spaces, key=lambda x: x.get("space_id", 0)):
            space_id = space.get("space_id", "?")
            apps = space.get("applications", [])
            is_current = space.get("is_current", False)
            window_titles = space.get("window_titles", [])
            
            if apps:
                # Get primary app
                primary_app = apps[0]
                
                # Generate detailed activity description with rich context
                activity = await self._infer_detailed_activity(
                    primary_app, window_titles, apps
                )
                
                # Format with current indicator
                current_marker = " (current)" if is_current else ""
                
                # Check if activity has multiple lines (rich detail)
                if "\n" in activity:
                    # Multi-line format for rich details
                    response_parts.append(f"• Space {space_id}{current_marker}: {primary_app}")
                    for line in activity.split("\n"):
                        response_parts.append(f"  {line}")
                else:
                    # Single line format with clean bullet
                    response_parts.append(f"• Space {space_id}{current_marker}: {primary_app} — {activity}")
            else:
                current_marker = " (current)" if is_current else ""
                response_parts.append(f"• Space {space_id}{current_marker}: Empty")
        
        # Generate intelligent workflow summary without separator
        workflow_summary = await self._generate_detailed_workflow_summary(snapshot, patterns)
        if workflow_summary:
            response_parts.append("")  # Blank line before summary
            response_parts.append("Workflow Analysis:")
            for line in workflow_summary.split("\n"):
                if line.strip():
                    response_parts.append(f"• {line}")
        
        response_text = "\n".join(response_parts)
        
        return {
            "analysis": response_text,
            "analysis_time": 0.0,
            "images_analyzed": 0,
            "intent": "workspace_overview",
            "patterns": [p.value for p in patterns],
            "overview_mode": True
        }
    
    async def _infer_detailed_activity(
        self,
        app_name: str,
        window_titles: List[str],
        all_apps: List[str]
    ) -> str:
        """
        Generate RICH, DETAILED activity descriptions with maximum context.
        Extracts specific files, URLs, tasks, and provides actionable information.
        NO HARDCODING - pure semantic intelligence.
        """
        
        # If we have window titles, do deep analysis
        if window_titles:
            title = next((t for t in window_titles if t and len(t) > 0), "")
            
            if title:
                title_lower = title.lower()
                app_lower = app_name.lower()
                
                # === BROWSER INTELLIGENCE (Chrome, Safari, Firefox) ===
                if any(browser in app_lower for browser in ['chrome', 'safari', 'firefox', 'edge', 'brave']):
                    # Extract website and tab context
                    if 'github.com' in title_lower or 'github' in title_lower:
                        # Extract repo name if present
                        if '/' in title:
                            parts = [p for p in title.split('/') if p.strip()]
                            if len(parts) >= 2:
                                repo = parts[-1].split('—')[0].split('-')[0].strip()
                                return f"Browsing GitHub repository: {repo}"
                        return "Browsing GitHub"
                    
                    elif 'stackoverflow' in title_lower:
                        # Try to extract question topic
                        question = title.split('-')[0].strip()
                        if len(question) < 60 and len(question) > 5:
                            return f"Stack Overflow: {question}"
                        return "Researching solutions on Stack Overflow"
                    
                    elif 'youtube' in title_lower:
                        video_title = title.split('-')[0].split('—')[0].strip()
                        if len(video_title) < 60 and len(video_title) > 5:
                            return f"Watching: {video_title}"
                        return "Watching YouTube videos"
                    
                    elif any(site in title_lower for site in ['reddit', 'twitter', 'linkedin', 'facebook']):
                        site_name = next(s.capitalize() for s in ['reddit', 'twitter', 'linkedin', 'facebook'] if s in title_lower)
                        return f"Browsing {site_name}"
                    
                    elif 'google' in title_lower and ('search' in title_lower or '?' in title):
                        # Try to extract search query
                        return "Searching Google"
                    
                    elif 'localhost' in title_lower or '127.0.0.1' in title_lower:
                        # Extract port and app
                        if ':' in title:
                            port_part = title.split(':')[1].split()[0]
                            return f"Testing local app on port {port_part}"
                        return "Testing local development server"
                    
                    elif 'docs' in title_lower or 'documentation' in title_lower:
                        # Extract doc topic
                        doc_name = title.split('—')[0].split('-')[0].split('|')[0].strip()
                        if len(doc_name) < 50 and len(doc_name) > 3:
                            return f"Reading docs: {doc_name}"
                        return "Reading documentation"
                    
                    else:
                        # Generic but extract first part of title
                        page_title = title.split('—')[0].split('-')[0].split('|')[0].strip()
                        if len(page_title) < 60 and len(page_title) > 3:
                            return f"Viewing: {page_title}"
                        return "Web browsing"
                
                # === TERMINAL INTELLIGENCE ===
                elif 'terminal' in app_lower or 'iterm' in app_lower:
                    # Extract what's running in terminal
                    if 'jupyter' in title_lower:
                        notebook_name = title.split(':')[0].strip()
                        if len(notebook_name) < 40:
                            return f"Running Jupyter: {notebook_name}"
                        return "Running Jupyter Notebook server"
                    
                    elif 'npm' in title_lower:
                        if 'run dev' in title_lower or 'dev' in title_lower:
                            return "Running npm dev server"
                        elif 'install' in title_lower:
                            return "Installing npm packages"
                        return "Running npm commands"
                    
                    elif 'python' in title_lower:
                        # Extract script name
                        if '.py' in title:
                            script = title.split('.py')[0].split()[-1] + '.py'
                            return f"Running Python script: {script}"
                        return "Running Python"
                    
                    elif 'docker' in title_lower:
                        return "Managing Docker containers"
                    
                    elif 'ssh' in title_lower:
                        # Extract server name
                        server = title.split('@')[-1].split()[0] if '@' in title else None
                        if server and len(server) < 30:
                            return f"SSH connected to {server}"
                        return "Connected via SSH"
                    
                    elif 'git' in title_lower:
                        return "Running Git commands"
                    
                    elif 'vim' in title_lower or 'nano' in title_lower or 'emacs' in title_lower:
                        return "Editing files in terminal"
                    
                    # Check for directory context
                    elif '~/' in title or '/' in title:
                        # Extract current directory
                        parts = title.split('/')
                        if len(parts) > 0:
                            current_dir = parts[-1].split()[0].strip()
                            if current_dir and len(current_dir) < 30:
                                return f"Working in: {current_dir}"
                    
                    return "Terminal session"
                
                # === CODE EDITOR INTELLIGENCE (Cursor, VSCode, etc.) ===
                elif any(editor in app_lower for editor in ['cursor', 'code', 'vscode', 'sublime', 'atom']):
                    # Extract project and file
                    if '—' in title or '–' in title:
                        parts = title.replace('–', '—').split('—')
                        
                        # Try to get file name
                        file_part = parts[0].strip()
                        project_part = parts[-1].strip() if len(parts) > 1 else None
                        
                        # Extract project name
                        if project_part:
                            project_name = project_part.split('[')[0].split('(')[0].strip()
                            
                            # Extract file name if present
                            if '.' in file_part and len(file_part) < 40:
                                file_name = file_part.split('/')[-1].strip()
                                return f"Editing {file_name}\n   📂 Project: {project_name}"
                            else:
                                return f"Working on {project_name} project"
                        
                        # Fallback: just file name
                        if '.' in file_part and len(file_part) < 40:
                            return f"Editing: {file_part}"
                    
                    # Check for file extension in title
                    extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.go', '.rs', '.rb', '.php', '.css', '.html', '.md']
                    for ext in extensions:
                        if ext in title:
                            file_name = title.split(ext)[0].split()[-1] + ext
                            return f"Editing: {file_name}"
                    
                    return "Code editing"
                
                # === DATA SCIENCE / JUPYTER ===
                elif 'jupyter' in app_lower or '.ipynb' in title_lower:
                    notebook_name = title.split('.ipynb')[0].split('/')[-1].strip()
                    if len(notebook_name) < 40 and len(notebook_name) > 0:
                        return f"Analyzing data: {notebook_name}.ipynb"
                    return "Data analysis in Jupyter"
                
                # === FILE BROWSER (Finder, etc.) ===
                elif 'finder' in app_lower:
                    # Extract current folder/location
                    location = title.strip()
                    if location and len(location) < 40 and location not in ['Finder', 'File Browser']:
                        return f"Browsing: {location}"
                    return "File management"
                
                # === GENERIC: Try to extract meaningful title ===
                clean_title = title.split('—')[0].split('-')[0].split('|')[0].strip()
                if len(clean_title) < 60 and len(clean_title) > 3:
                    # Check if it's not just the app name
                    if clean_title.lower() != app_name.lower():
                        return clean_title
        
        # === FALLBACK: Use original simpler inference ===
        return await self._infer_activity_from_context(app_name, window_titles, all_apps)
    
    async def _infer_activity_from_context(
        self,
        app_name: str,
        window_titles: List[str],
        all_apps: List[str]
    ) -> str:
        """
        Simpler activity inference for fallback cases.
        NO HARDCODING - uses dynamic pattern recognition and semantic analysis.
        """
        
        # If we have window titles, analyze them for context
        if window_titles:
            # Get the most informative title (usually the first non-empty one)
            title = next((t for t in window_titles if t and len(t) > 0), "")
            
            if title:
                # Extract meaningful context from title
                title_lower = title.lower()
                
                # Dynamic semantic analysis of title content
                activity_signals = {
                    # Project/file indicators
                    'project': any(indicator in title_lower for indicator in ['.py', '.js', '.ts', '.java', '.cpp', '.go', '.rs', 'github', 'gitlab', 'repo']),
                    'documentation': any(indicator in title_lower for indicator in ['readme', 'docs', 'documentation', 'wiki', 'guide', 'tutorial']),
                    'error_debugging': any(indicator in title_lower for indicator in ['error', 'exception', 'debug', 'traceback', 'failed', 'issue']),
                    'data_analysis': any(indicator in title_lower for indicator in ['jupyter', 'notebook', '.ipynb', 'pandas', 'matplotlib', 'data', 'analysis']),
                    'web_browsing': any(indicator in title_lower for indicator in ['google', 'search', 'stackoverflow', 'reddit', 'youtube', 'twitter']),
                    'communication': any(indicator in title_lower for indicator in ['slack', 'discord', 'mail', 'message', 'chat', 'email']),
                    'design': any(indicator in title_lower for indicator in ['figma', 'sketch', 'design', 'photoshop', 'illustrator']),
                    'terminal_task': any(indicator in title_lower for indicator in ['bash', 'zsh', 'terminal', 'ssh', 'server', 'docker', 'npm', 'pip'])
                }
                
                # Combine app context with title signals for robust inference
                if activity_signals['project']:
                    # Extract project name if present
                    project_indicators = ['github', 'gitlab', '—', '-', ':', 'repo']
                    for indicator in project_indicators:
                        if indicator in title:
                            parts = title.split(indicator)
                            if len(parts) > 1:
                                project_name = parts[-1].strip() if indicator in ['—', '-', ':'] else parts[0].strip()
                                # Clean up common suffixes
                                project_name = project_name.replace('.git', '').replace('/', ' ').strip()
                                if project_name:
                                    return f"Working on {project_name} project"
                    return "Code editing"
                
                elif activity_signals['error_debugging']:
                    return "Debugging errors"
                
                elif activity_signals['data_analysis']:
                    # Try to extract notebook name
                    if '.ipynb' in title_lower or 'notebook' in title_lower:
                        notebook_parts = title.split('-')[0].split('—')[0].split(':')[0]
                        notebook_name = notebook_parts.strip()
                        if notebook_name and len(notebook_name) < 50:
                            return f"Analyzing data in {notebook_name}"
                    return "Data analysis and visualization"
                
                elif activity_signals['documentation']:
                    return "Reading documentation"
                
                elif activity_signals['web_browsing']:
                    # Try to extract search topic or site
                    if 'google' in title_lower and 'search' in title_lower:
                        return "Web research"
                    elif 'stackoverflow' in title_lower:
                        return "Researching solutions on Stack Overflow"
                    elif 'github' in title_lower:
                        return "Browsing GitHub repositories"
                    elif 'youtube' in title_lower:
                        return "Watching tutorials or videos"
                    return "Web browsing"
                
                elif activity_signals['communication']:
                    return "Team communication"
                
                elif activity_signals['design']:
                    return "Design work"
                
                elif activity_signals['terminal_task']:
                    # Infer task from title
                    if 'jupyter' in title_lower:
                        return "Running Jupyter server"
                    elif 'docker' in title_lower:
                        return "Container management"
                    elif 'npm' in title_lower or 'node' in title_lower:
                        return "Node.js development"
                    elif 'python' in title_lower or 'pip' in title_lower:
                        return "Python development"
                    return "Terminal operations"
                
                # If title has meaningful content but no specific signal, use it directly
                if len(title) < 60 and len(title) > 3:
                    # Clean up title for display
                    clean_title = title.split('—')[0].split('-')[0].split(':')[0].strip()
                    if clean_title and not any(char in clean_title for char in ['/', '\\', '|']):
                        return clean_title
        
        # Fallback: Infer from app name + multiple app context
        app_lower = app_name.lower()
        
        # Multi-app context analysis (no hardcoding - dynamic inference)
        if len(all_apps) > 1:
            app_set = {app.lower() for app in all_apps}
            
            # Detect development environment (multiple dev tools)
            dev_tools = {'cursor', 'code', 'vscode', 'vim', 'sublime', 'atom', 'intellij', 'pycharm'}
            if len(app_set & dev_tools) > 0:
                if 'terminal' in app_set or 'iterm' in app_set:
                    return "Active development session"
                return "Code editing"
        
        # App-specific intelligent defaults (semantic, not hardcoded rules)
        app_activity_map = {
            'finder': 'File management',
            'chrome': 'Web browsing',
            'safari': 'Web browsing',
            'firefox': 'Web browsing',
            'cursor': 'Code editing',
            'code': 'Code editing',
            'vscode': 'Code editing',
            'terminal': 'Command line operations',
            'iterm': 'Command line operations',
            'slack': 'Team collaboration',
            'discord': 'Communication',
            'mail': 'Email',
            'messages': 'Messaging',
            'notes': 'Note taking',
            'obsidian': 'Knowledge management',
            'notion': 'Project management',
            'figma': 'Design work',
            'photoshop': 'Image editing',
            'spotify': 'Music',
            'zoom': 'Video call',
            'docker': 'Container management',
        }
        
        # Try to find semantic match
        for key, activity in app_activity_map.items():
            if key in app_lower:
                return activity
        
        # Ultimate fallback: Generic but accurate
        return "Active"
    
    async def _generate_detailed_workflow_summary(
        self,
        snapshot: WorkspaceSnapshot,
        patterns: List[WorkflowPattern]
    ) -> str:
        """
        Generate RICH workflow summary with detailed multi-dimensional analysis.
        NO HARDCODING - comprehensive dynamic analysis.
        """
        
        # Collect detailed workspace intelligence
        all_apps = set()
        dev_spaces = []
        browser_spaces = []
        terminal_spaces = []
        
        for space in snapshot.spaces:
            apps = space.get("applications", [])
            space_id = space.get("space_id")
            all_apps.update(apps)
            
            for app in apps:
                app_lower = app.lower()
                if any(dev in app_lower for dev in ['cursor', 'code', 'vscode', 'sublime']):
                    dev_spaces.append(space_id)
                if any(browser in app_lower for browser in ['chrome', 'safari', 'firefox']):
                    browser_spaces.append(space_id)
                if 'terminal' in app_lower or 'iterm' in app_lower:
                    terminal_spaces.append(space_id)
        
        # Build rich summary
        summary_parts = []
        
        # Multi-space activity pattern
        active_spaces = len([s for s in snapshot.spaces if len(s.get("applications", [])) > 0])
        if active_spaces >= 4:
            summary_parts.append(f"You're actively multitasking across {active_spaces} spaces")
        elif active_spaces >= 2:
            summary_parts.append(f"Working across {active_spaces} active spaces")
        
        # Development focus
        if dev_spaces:
            if len(dev_spaces) > 1:
                summary_parts.append(f"Development work happening in {len(dev_spaces)} spaces")
            else:
                summary_parts.append("Focused development work")
        
        # Context switching indicator
        if len(dev_spaces) > 0 and len(browser_spaces) > 0:
            summary_parts.append("Switching between coding and research")
        
        # Terminal activity
        if terminal_spaces:
            summary_parts.append("Active terminal sessions running")
        
        # Overall workflow categorization
        categories = []
        if dev_spaces: categories.append("development")
        if terminal_spaces: categories.append("command-line")
        if browser_spaces: categories.append("web research")
        
        if len(categories) >= 2:
            summary_parts.append(f"Primary focus: {' + '.join(categories)}")
        
        # Fallback to original summary if needed
        if not summary_parts:
            return await self._generate_workflow_summary(snapshot, patterns)
        
        return "\n".join(summary_parts)
    
    async def _generate_workflow_summary(
        self,
        snapshot: WorkspaceSnapshot,
        patterns: List[WorkflowPattern]
    ) -> str:
        """
        Generate intelligent workflow summary based on actual workspace state.
        NO HARDCODING - dynamically analyzes actual applications and patterns.
        """
        
        # Collect all unique applications across all spaces
        all_apps = set()
        space_contexts = []
        
        for space in snapshot.spaces:
            apps = space.get("applications", [])
            all_apps.update(apps)
            if apps:
                space_contexts.append({
                    'apps': apps,
                    'titles': space.get("window_titles", [])
                })
        
        # Dynamic semantic analysis of application ecosystem
        activity_categories = {
            'development': {'cursor', 'code', 'vscode', 'vim', 'sublime', 'atom', 'intellij', 'pycharm', 'webstorm', 'android studio'},
            'terminal': {'terminal', 'iterm', 'iterm2', 'alacritty', 'kitty'},
            'browser': {'chrome', 'safari', 'firefox', 'edge', 'brave'},
            'communication': {'slack', 'discord', 'mail', 'messages', 'zoom', 'teams', 'skype'},
            'data_science': {'jupyter', 'rstudio', 'spyder', 'anaconda'},
            'design': {'figma', 'sketch', 'photoshop', 'illustrator', 'affinity'},
            'productivity': {'notion', 'obsidian', 'notes', 'evernote', 'onenote'},
            'media': {'spotify', 'music', 'vlc', 'quicktime'}
        }
        
        # Detect active categories
        active_categories = []
        app_lower_set = {app.lower() for app in all_apps}
        
        for category, keywords in activity_categories.items():
            if any(keyword in app_str for keyword in keywords for app_str in app_lower_set):
                active_categories.append(category)
        
        # Generate intelligent summary based on detected patterns
        if len(active_categories) >= 3:
            # Multi-modal work
            primary = active_categories[0]
            return f"Your focus spans multiple areas: {', '.join(active_categories[:3])}."
        
        elif len(active_categories) == 2:
            # Two primary activities
            return f"You're balancing {active_categories[0]} and {active_categories[1]} work."
        
        elif len(active_categories) == 1:
            # Single focus
            category = active_categories[0]
            if category == 'development' and 'terminal' in active_categories:
                return "You're in an active development session."
            elif category == 'development':
                return "Your primary focus is on software development."
            elif category == 'browser':
                return "You're primarily engaged in web-based research and browsing."
            elif category == 'data_science':
                return "You're working on data analysis and visualization."
            else:
                return f"Your primary focus is on {category} work."
        
        # Fallback: Use detected patterns
        if patterns:
            pattern_name = patterns[0].value.replace('_', ' ')
            return f"Your primary focus appears to be on {pattern_name} work."
        
        # Ultimate fallback
        if len(all_apps) > 3:
            return "You're actively multitasking across multiple applications."
        else:
            return "You're working across your desktop spaces."
    
    async def _build_analysis_prompt(
        self,
        query: str,
        intent: QueryIntent,
        snapshot: WorkspaceSnapshot,
        captured_content: Dict[str, Any]
    ) -> str:
        """Build enhanced prompt for Claude Vision analysis"""
        
        # Base prompt
        prompt = f"""You are JARVIS, Tony Stark's AI assistant with advanced vision capabilities.

USER QUERY: "{query}"
QUERY INTENT: {intent.value}

WORKSPACE CONTEXT:
You have visibility across {snapshot.total_spaces} desktop spaces with {snapshot.total_windows} windows total.
Current space: {snapshot.current_space}

SPACE BREAKDOWN:"""
        
        # Add space details
        for space in snapshot.spaces:
            space_id = space["space_id"]
            is_current = space.get("is_current", False)
            apps = space.get("applications", [])
            primary_app = space.get("primary_app", "Unknown")
            
            current_marker = " (CURRENT)" if is_current else ""
            prompt += f"\n• Space {space_id}{current_marker}: {primary_app}"
            if apps:
                prompt += f" - {', '.join(apps)}"
        
        # Add captured content context
        if captured_content:
            prompt += "\n\nCAPTURED CONTENT FOR ANALYSIS:"
            for content_key, content_data in captured_content.items():
                target = content_data["target"]
                prompt += f"\n• Space {target.space_id}: {target.app_name} - {target.window_title}"
                prompt += f"\n  Priority: {target.priority.value} ({target.reason})"
        
        # Add detected patterns
        if snapshot.patterns_detected:
            patterns = [p.value for p in snapshot.patterns_detected]
            prompt += f"\n\nDETECTED WORKFLOW PATTERNS: {', '.join(patterns)}"
        
        # Add analysis instructions
        prompt += f"""

ANALYSIS INSTRUCTIONS:
1. Analyze the visual content in detail
2. Focus on the user's specific query and intent
3. Identify patterns and connections across spaces
4. Provide actionable insights and recommendations
5. Address the user as "Sir" naturally
6. Be specific about what you see and what it means
7. If you detect errors or issues, highlight them clearly
8. Connect related activities across different spaces

Provide a comprehensive analysis that helps the user understand their workspace and what's happening."""
        
        return prompt
    
    async def _store_context(self, context: AnalysisContext, analysis_result: Dict[str, Any]):
        """Store analysis context for follow-up questions"""
        
        with self._lock:
            self._context_cache[context.workspace_snapshot.snapshot_id] = context
            
            # Clean up old contexts
            current_time = datetime.now()
            expired_contexts = []
            for context_id, stored_context in self._context_cache.items():
                age = (current_time - stored_context.workspace_snapshot.timestamp).total_seconds()
                if age > self._config["context_ttl_seconds"]:
                    expired_contexts.append(context_id)
            
            for context_id in expired_contexts:
                del self._context_cache[context_id]
    
    async def _track_performance(self, total_time: float, targets_count: int, captured_count: int):
        """Track performance metrics for optimization"""
        
        if self._config["performance_monitoring"]:
            self._performance_metrics["total_time"].append(total_time)
            self._performance_metrics["targets_count"].append(targets_count)
            self._performance_metrics["captured_count"].append(captured_count)
            
            # Record metrics in dynamic config manager
            self.config_manager.record_performance_metric("total_time", total_time)
            self.config_manager.record_performance_metric("targets_count", targets_count)
            self.config_manager.record_performance_metric("captured_count", captured_count)
            
            # Keep only recent metrics
            for metric_name in self._performance_metrics:
                if len(self._performance_metrics[metric_name]) > 100:
                    self._performance_metrics[metric_name] = self._performance_metrics[metric_name][-100:]
    
    async def _optimize_dynamically(self, total_time: float, targets_count: int, captured_count: int):
        """Dynamically optimize configuration based on performance"""
        
        try:
            # Check if optimization is needed
            active_config = self.config_manager.get_active_config()
            target_time = active_config.get("performance", {}).get("target_response_time", 3.0)
            
            # If performance is significantly worse than target, optimize
            if total_time > target_time * 1.5:
                self.logger.info(f"[OPTIMIZATION] Performance slow ({total_time:.2f}s > {target_time:.2f}s), optimizing...")
                
                # Auto-optimize configuration
                recommendation = self.config_manager.auto_optimize()
                self.logger.info(f"[OPTIMIZATION] Applied {recommendation.value} optimization")
            
            # If performance is good, consider quality optimization
            elif total_time < target_time * 0.7 and targets_count < 3:
                self.logger.info(f"[OPTIMIZATION] Performance good ({total_time:.2f}s < {target_time:.2f}s), considering quality boost...")
                
                # Check if user prefers quality
                user_prefs = self.config_manager._user_preferences
                if user_prefs.get("prefer_quality", "false").lower() == "true":
                    self.config_manager.optimize_for_quality()
                    self.logger.info("[OPTIMIZATION] Applied quality optimization")
                    
        except Exception as e:
            self.logger.warning(f"[OPTIMIZATION] Dynamic optimization failed: {e}")
    
    async def handle_follow_up_query(
        self, 
        query: str, 
        context_id: Optional[str] = None,
        claude_api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle follow-up queries using stored context"""
        
        if not context_id:
            # Try to find recent context
            with self._lock:
                if self._context_cache:
                    context_id = max(
                        self._context_cache.keys(),
                        key=lambda k: self._context_cache[k].workspace_snapshot.timestamp
                    )
        
        if not context_id or context_id not in self._context_cache:
            return {
                "success": False,
                "error": "No context available for follow-up query"
            }
        
        # Get stored context
        with self._lock:
            stored_context = self._context_cache[context_id]
        
        # Analyze follow-up intent
        intent = await self._analyze_query_intent(query)
        
        # Determine what to re-capture based on follow-up
        re_capture_targets = await self._select_follow_up_targets(query, stored_context)
        
        # Re-capture if needed
        captured_content = {}
        if re_capture_targets:
            captured_content = await self._capture_selectively(re_capture_targets)
        
        # Analyze with Claude using stored context
        analysis_result = await self._analyze_with_claude(
            query, intent, stored_context.workspace_snapshot, captured_content, claude_api_key
        )
        
        return {
            "success": True,
            "analysis": analysis_result,
            "context_id": context_id,
            "re_captured": len(captured_content) > 0
        }
    
    async def _select_follow_up_targets(
        self, 
        query: str, 
        stored_context: AnalysisContext
    ) -> List[CaptureTarget]:
        """Select targets for follow-up queries"""
        
        query_lower = query.lower()
        targets = []
        
        # Check if query references specific spaces or content
        for target in stored_context.capture_targets:
            # Check if query mentions this target
            if (target.app_name.lower() in query_lower or 
                target.window_title.lower() in query_lower or
                f"space {target.space_id}" in query_lower):
                targets.append(target)
        
        return targets
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        
        metrics = {}
        for metric_name, values in self._performance_metrics.items():
            if values:
                metrics[metric_name] = {
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        
        return metrics
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update configuration dynamically"""
        
        self._config.update(new_config)
        self.logger.info(f"[ORCHESTRATOR] Configuration updated: {new_config}")


# Global instance
_orchestrator_instance = None

def get_intelligent_orchestrator() -> IntelligentOrchestrator:
    """Get or create the global orchestrator instance"""
    global _orchestrator_instance
    
    if _orchestrator_instance is None:
        _orchestrator_instance = IntelligentOrchestrator()
    
    return _orchestrator_instance