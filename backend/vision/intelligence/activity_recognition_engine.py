"""
Activity Recognition Engine - Understanding what task the user is performing and why
Part of the Intelligent Understanding System (IUS)
Memory-optimized for 100MB allocation
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter, deque
from enum import Enum, auto
import json
import logging
from pathlib import Path
import statistics
import re

logger = logging.getLogger(__name__)

# Memory allocation constants
MEMORY_LIMITS = {
    'task_templates': 30 * 1024 * 1024,    # 30MB
    'activity_history': 40 * 1024 * 1024,   # 40MB
    'progress_tracking': 30 * 1024 * 1024   # 30MB
}


class PrimaryActivity(Enum):
    """Primary activity categories"""
    CODING_DEVELOPMENT = "coding_development"
    COMMUNICATION = "communication"
    RESEARCH_BROWSING = "research_browsing"
    DOCUMENT_CREATION = "document_creation"
    DATA_ANALYSIS = "data_analysis"
    MEDIA_CONSUMPTION = "media_consumption"
    FILE_MANAGEMENT = "file_management"
    SYSTEM_ADMINISTRATION = "system_administration"
    LEARNING_EDUCATION = "learning_education"
    CREATIVE_WORK = "creative_work"
    UNKNOWN = "unknown"


class TaskStatus(Enum):
    """Task completion status"""
    NOT_STARTED = auto()
    IN_PROGRESS = auto()
    STUCK = auto()
    BLOCKED = auto()
    ABANDONED = auto()
    COMPLETED = auto()
    FAILED = auto()


class ProgressIndicator(Enum):
    """Indicators of task progress"""
    FORWARD_PROGRESS = "forward_progress"
    STALLED = "stalled"
    REGRESSION = "regression"
    CYCLIC_PATTERN = "cyclic_pattern"
    HELP_SEEKING = "help_seeking"
    ERROR_RECOVERY = "error_recovery"


@dataclass
class TaskTemplate:
    """Template for recognizable tasks"""
    task_id: str
    name: str
    primary_activity: PrimaryActivity
    typical_applications: List[str]
    common_patterns: List[str]
    expected_duration: timedelta
    success_indicators: List[str]
    failure_indicators: List[str]
    prerequisites: List[str] = field(default_factory=list)
    sub_tasks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def matches_pattern(self, pattern: str) -> float:
        """Check if a pattern matches this task template"""
        pattern_lower = pattern.lower()
        matches = 0
        total = len(self.common_patterns)
        
        for common_pattern in self.common_patterns:
            if common_pattern.lower() in pattern_lower:
                matches += 1
            elif self._fuzzy_match(common_pattern.lower(), pattern_lower):
                matches += 0.5
        
        return matches / total if total > 0 else 0.0
    
    def _fuzzy_match(self, pattern1: str, pattern2: str) -> bool:
        """Simple fuzzy matching"""
        words1 = set(pattern1.split())
        words2 = set(pattern2.split())
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        return overlap / total > 0.5 if total > 0 else False


@dataclass
class RecognizedTask:
    """A task that has been recognized"""
    task_id: str
    template_id: Optional[str]
    name: str
    primary_activity: PrimaryActivity
    start_time: datetime
    status: TaskStatus = TaskStatus.IN_PROGRESS
    confidence: float = 1.0
    
    # Task context
    active_applications: Set[str] = field(default_factory=set)
    visited_states: List[str] = field(default_factory=list)
    key_interactions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Progress tracking
    completion_percentage: float = 0.0
    progress_indicators: List[ProgressIndicator] = field(default_factory=list)
    last_progress_time: Optional[datetime] = None
    stuck_duration: Optional[timedelta] = None
    
    # Related tasks
    parent_task: Optional[str] = None
    sub_tasks: List[str] = field(default_factory=list)
    blocking_factors: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> timedelta:
        """Get task duration"""
        end_time = datetime.now() if self.status == TaskStatus.IN_PROGRESS else self.last_progress_time
        return end_time - self.start_time if end_time else timedelta()
    
    @property
    def is_stuck(self) -> bool:
        """Check if task is stuck"""
        return self.status == TaskStatus.STUCK or (
            self.stuck_duration and self.stuck_duration > timedelta(minutes=5)
        )


class TaskIdentifier:
    """Identifies and classifies user tasks"""
    
    def __init__(self):
        self.task_templates: Dict[str, TaskTemplate] = {}
        self.app_activity_map: Dict[str, PrimaryActivity] = {}
        self._load_default_templates()
        self._load_app_mappings()
    
    def _load_default_templates(self):
        """Load default task templates"""
        # Coding/Development templates
        self.add_template(TaskTemplate(
            task_id="code_debug",
            name="Debugging Code",
            primary_activity=PrimaryActivity.CODING_DEVELOPMENT,
            typical_applications=["vscode", "xcode", "intellij", "terminal"],
            common_patterns=["error", "debug", "breakpoint", "console", "log"],
            expected_duration=timedelta(minutes=30),
            success_indicators=["fix", "resolved", "working", "success"],
            failure_indicators=["stuck", "help", "confused", "error persists"],
            sub_tasks=["identify_error", "reproduce_issue", "fix_code", "test_fix"]
        ))
        
        self.add_template(TaskTemplate(
            task_id="code_feature",
            name="Implementing New Feature",
            primary_activity=PrimaryActivity.CODING_DEVELOPMENT,
            typical_applications=["vscode", "xcode", "terminal", "browser"],
            common_patterns=["implement", "feature", "function", "class", "test"],
            expected_duration=timedelta(hours=2),
            success_indicators=["complete", "tested", "commit", "push"],
            failure_indicators=["blocked", "unclear", "help needed"],
            prerequisites=["requirements_understood", "environment_setup"],
            sub_tasks=["design", "implement", "test", "document"]
        ))
        
        # Communication templates
        self.add_template(TaskTemplate(
            task_id="email_response",
            name="Responding to Emails",
            primary_activity=PrimaryActivity.COMMUNICATION,
            typical_applications=["mail", "gmail", "outlook"],
            common_patterns=["inbox", "reply", "compose", "send"],
            expected_duration=timedelta(minutes=15),
            success_indicators=["sent", "replied", "forwarded"],
            failure_indicators=["draft", "unsent", "cancelled"]
        ))
        
        self.add_template(TaskTemplate(
            task_id="meeting_participation",
            name="Participating in Meeting",
            primary_activity=PrimaryActivity.COMMUNICATION,
            typical_applications=["zoom", "teams", "slack", "discord"],
            common_patterns=["meeting", "call", "video", "audio", "screen share"],
            expected_duration=timedelta(minutes=45),
            success_indicators=["meeting ended", "disconnected", "left meeting"],
            failure_indicators=["connection issues", "audio problems", "kicked out"]
        ))
        
        # Research/Browsing templates
        self.add_template(TaskTemplate(
            task_id="research_topic",
            name="Researching Topic",
            primary_activity=PrimaryActivity.RESEARCH_BROWSING,
            typical_applications=["chrome", "safari", "firefox"],
            common_patterns=["search", "google", "wikipedia", "documentation", "tutorial"],
            expected_duration=timedelta(minutes=30),
            success_indicators=["found", "understand", "bookmarked", "noted"],
            failure_indicators=["not found", "confused", "too complex"],
            sub_tasks=["initial_search", "deep_dive", "note_taking"]
        ))
        
        # Document Creation templates
        self.add_template(TaskTemplate(
            task_id="write_document",
            name="Writing Document",
            primary_activity=PrimaryActivity.DOCUMENT_CREATION,
            typical_applications=["word", "pages", "google docs", "notion"],
            common_patterns=["write", "document", "edit", "format", "save"],
            expected_duration=timedelta(hours=1),
            success_indicators=["saved", "complete", "exported", "shared"],
            failure_indicators=["lost work", "crashed", "incomplete"],
            sub_tasks=["outline", "draft", "edit", "finalize"]
        ))
        
        # Data Analysis templates
        self.add_template(TaskTemplate(
            task_id="analyze_data",
            name="Analyzing Data",
            primary_activity=PrimaryActivity.DATA_ANALYSIS,
            typical_applications=["excel", "numbers", "jupyter", "tableau"],
            common_patterns=["data", "chart", "graph", "analyze", "visualize"],
            expected_duration=timedelta(hours=1),
            success_indicators=["insight", "conclusion", "report", "visualization"],
            failure_indicators=["error", "incorrect", "missing data"],
            prerequisites=["data_available", "tools_ready"],
            sub_tasks=["load_data", "clean_data", "analyze", "visualize"]
        ))
    
    def _load_app_mappings(self):
        """Load application to activity mappings"""
        # Development applications
        for app in ["vscode", "xcode", "intellij", "sublime", "atom", "vim", "emacs", "terminal", "iterm"]:
            self.app_activity_map[app] = PrimaryActivity.CODING_DEVELOPMENT
        
        # Communication applications
        for app in ["mail", "outlook", "gmail", "slack", "discord", "teams", "zoom", "messages", "whatsapp"]:
            self.app_activity_map[app] = PrimaryActivity.COMMUNICATION
        
        # Browsers (can be multiple activities)
        for app in ["chrome", "safari", "firefox", "edge", "brave"]:
            self.app_activity_map[app] = PrimaryActivity.RESEARCH_BROWSING
        
        # Document creation
        for app in ["word", "pages", "docs", "notion", "obsidian", "typora", "ia writer"]:
            self.app_activity_map[app] = PrimaryActivity.DOCUMENT_CREATION
        
        # Data analysis
        for app in ["excel", "numbers", "sheets", "jupyter", "rstudio", "tableau", "powerbi"]:
            self.app_activity_map[app] = PrimaryActivity.DATA_ANALYSIS
        
        # Media consumption
        for app in ["spotify", "music", "vlc", "quicktime", "youtube", "netflix"]:
            self.app_activity_map[app] = PrimaryActivity.MEDIA_CONSUMPTION
        
        # File management
        for app in ["finder", "explorer", "dropbox", "drive", "onedrive"]:
            self.app_activity_map[app] = PrimaryActivity.FILE_MANAGEMENT
        
        # System administration
        for app in ["activity monitor", "system preferences", "disk utility", "terminal"]:
            self.app_activity_map[app] = PrimaryActivity.SYSTEM_ADMINISTRATION
    
    def add_template(self, template: TaskTemplate):
        """Add a task template"""
        self.task_templates[template.task_id] = template
    
    def identify_activity_from_apps(self, active_apps: List[str]) -> Tuple[PrimaryActivity, float]:
        """Identify primary activity from active applications"""
        if not active_apps:
            return PrimaryActivity.UNKNOWN, 0.0
        
        activity_scores = Counter()
        
        for app in active_apps:
            app_lower = app.lower()
            # Direct mapping
            if app_lower in self.app_activity_map:
                activity_scores[self.app_activity_map[app_lower]] += 1
            else:
                # Try partial matching
                for known_app, activity in self.app_activity_map.items():
                    if known_app in app_lower or app_lower in known_app:
                        activity_scores[activity] += 0.5
        
        if not activity_scores:
            return PrimaryActivity.UNKNOWN, 0.0
        
        most_common = activity_scores.most_common(1)[0]
        confidence = most_common[1] / len(active_apps)
        
        return most_common[0], confidence
    
    def match_task_templates(self, 
                           activity: PrimaryActivity,
                           context: Dict[str, Any]) -> List[Tuple[TaskTemplate, float]]:
        """Match context against task templates"""
        matches = []
        
        # Filter templates by activity
        relevant_templates = [
            t for t in self.task_templates.values()
            if t.primary_activity == activity
        ]
        
        for template in relevant_templates:
            score = self._calculate_template_match_score(template, context)
            if score > 0.3:  # Minimum threshold
                matches.append((template, score))
        
        # Sort by score
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def _calculate_template_match_score(self, 
                                      template: TaskTemplate,
                                      context: Dict[str, Any]) -> float:
        """Calculate how well a template matches the context"""
        scores = []
        
        # Check application overlap
        context_apps = set(context.get('active_applications', []))
        template_apps = set(template.typical_applications)
        if context_apps and template_apps:
            app_overlap = len(context_apps & template_apps) / len(template_apps)
            scores.append(app_overlap)
        
        # Check pattern matching
        context_text = ' '.join(context.get('text_content', []))
        if context_text:
            pattern_score = template.matches_pattern(context_text)
            scores.append(pattern_score)
        
        # Check state patterns
        states = context.get('visited_states', [])
        if states:
            state_patterns = ' '.join(states)
            for pattern in template.common_patterns:
                if pattern in state_patterns:
                    scores.append(0.8)
                    break
            else:
                scores.append(0.2)
        
        return statistics.mean(scores) if scores else 0.0


class TaskInferencer:
    """Infers specific tasks from activity and context"""
    
    def __init__(self, task_identifier: TaskIdentifier):
        self.task_identifier = task_identifier
        self.confidence_threshold = 0.5
    
    async def infer_task(self, 
                        activity: PrimaryActivity,
                        context: Dict[str, Any],
                        history: Optional[List[RecognizedTask]] = None) -> RecognizedTask:
        """Infer the specific task being performed"""
        # Try template matching first
        template_matches = self.task_identifier.match_task_templates(activity, context)
        
        if template_matches and template_matches[0][1] > self.confidence_threshold:
            # Use best matching template
            template, confidence = template_matches[0]
            task = RecognizedTask(
                task_id=f"{template.task_id}_{datetime.now().timestamp()}",
                template_id=template.task_id,
                name=template.name,
                primary_activity=activity,
                start_time=datetime.now(),
                confidence=confidence
            )
            
            # Populate from template
            task.sub_tasks = template.sub_tasks.copy()
            
        else:
            # Create generic task
            task = RecognizedTask(
                task_id=f"generic_{activity.value}_{datetime.now().timestamp()}",
                template_id=None,
                name=f"Generic {activity.value.replace('_', ' ').title()}",
                primary_activity=activity,
                start_time=datetime.now(),
                confidence=0.5
            )
        
        # Update from context
        task.active_applications = set(context.get('active_applications', []))
        task.visited_states = context.get('visited_states', [])
        
        # Check for sub-task relationship
        if history:
            parent_candidate = self._find_parent_task(task, history)
            if parent_candidate:
                task.parent_task = parent_candidate.task_id
                parent_candidate.sub_tasks.append(task.task_id)
        
        return task
    
    def _find_parent_task(self, 
                         task: RecognizedTask,
                         history: List[RecognizedTask]) -> Optional[RecognizedTask]:
        """Find potential parent task"""
        # Look for recent incomplete tasks with matching activity
        for hist_task in reversed(history):
            if (hist_task.status == TaskStatus.IN_PROGRESS and
                hist_task.primary_activity == task.primary_activity and
                (datetime.now() - hist_task.start_time) < timedelta(hours=2)):
                return hist_task
        return None
    
    async def update_task_confidence(self, 
                                   task: RecognizedTask,
                                   new_evidence: Dict[str, Any]) -> float:
        """Update task confidence based on new evidence"""
        adjustments = []
        
        # Check if expected applications are still active
        current_apps = set(new_evidence.get('active_applications', []))
        if task.active_applications:
            app_persistence = len(current_apps & task.active_applications) / len(task.active_applications)
            adjustments.append(app_persistence)
        
        # Check for success indicators
        if task.template_id:
            template = self.task_identifier.task_templates.get(task.template_id)
            if template:
                text_content = ' '.join(new_evidence.get('text_content', [])).lower()
                
                # Check success indicators
                success_found = any(indicator in text_content for indicator in template.success_indicators)
                if success_found:
                    adjustments.append(1.2)  # Boost confidence
                
                # Check failure indicators
                failure_found = any(indicator in text_content for indicator in template.failure_indicators)
                if failure_found:
                    adjustments.append(0.5)  # Reduce confidence
        
        # Apply adjustments
        if adjustments:
            adjustment_factor = statistics.mean(adjustments)
            task.confidence = min(1.0, task.confidence * adjustment_factor)
        
        return task.confidence


class ProgressMonitor:
    """Monitors task progress and detects issues"""
    
    def __init__(self):
        self.progress_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.stuck_threshold = timedelta(minutes=5)
        self.abandonment_threshold = timedelta(minutes=15)
    
    async def update_progress(self, 
                            task: RecognizedTask,
                            context: Dict[str, Any]) -> ProgressIndicator:
        """Update task progress and return indicator"""
        task_id = task.task_id
        
        # Create progress snapshot
        snapshot = {
            'timestamp': datetime.now(),
            'visited_states': len(task.visited_states),
            'active_apps': len(task.active_applications),
            'interactions': len(task.key_interactions),
            'context': context
        }
        
        self.progress_history[task_id].append(snapshot)
        
        # Analyze progress
        indicator = self._analyze_progress_pattern(task_id)
        
        # Update task
        task.progress_indicators.append(indicator)
        task.last_progress_time = datetime.now()
        
        # Check if stuck
        if indicator in [ProgressIndicator.STALLED, ProgressIndicator.CYCLIC_PATTERN]:
            if not task.stuck_duration:
                task.stuck_duration = timedelta()
            else:
                task.stuck_duration += datetime.now() - task.last_progress_time
            
            if task.stuck_duration > self.stuck_threshold:
                task.status = TaskStatus.STUCK
        else:
            task.stuck_duration = None
        
        # Update completion percentage
        task.completion_percentage = self._estimate_completion(task)
        
        return indicator
    
    def _analyze_progress_pattern(self, task_id: str) -> ProgressIndicator:
        """Analyze progress pattern from history"""
        history = self.progress_history[task_id]
        
        if len(history) < 2:
            return ProgressIndicator.FORWARD_PROGRESS
        
        # Compare recent snapshots
        recent = history[-5:]  # Last 5 snapshots
        
        # Check for stalling (no changes)
        if all(s['visited_states'] == recent[0]['visited_states'] for s in recent[1:]):
            return ProgressIndicator.STALLED
        
        # Check for regression
        if recent[-1]['visited_states'] < recent[0]['visited_states']:
            return ProgressIndicator.REGRESSION
        
        # Check for cyclic pattern
        state_counts = [s['visited_states'] for s in recent]
        if len(set(state_counts)) <= 2 and len(state_counts) >= 4:
            return ProgressIndicator.CYCLIC_PATTERN
        
        # Check for help seeking
        context_text = ' '.join(recent[-1]['context'].get('text_content', [])).lower()
        help_keywords = ['help', 'error', 'stuck', 'how to', 'why', 'issue', 'problem']
        if any(keyword in context_text for keyword in help_keywords):
            return ProgressIndicator.HELP_SEEKING
        
        # Default to forward progress
        return ProgressIndicator.FORWARD_PROGRESS
    
    def _estimate_completion(self, task: RecognizedTask) -> float:
        """Estimate task completion percentage"""
        if task.status == TaskStatus.COMPLETED:
            return 100.0
        
        if task.status in [TaskStatus.ABANDONED, TaskStatus.FAILED]:
            return task.completion_percentage  # Keep last value
        
        # If we have sub-tasks, use them
        if task.sub_tasks and task.template_id:
            template = TaskTemplate  # Would get from task_identifier
            completed_subtasks = 0
            # This would check which subtasks are done
            # For now, estimate based on duration
            
        # Estimate based on duration vs expected
        if task.template_id:
            # Would get expected duration from template
            expected_duration = timedelta(hours=1)  # Placeholder
            actual_duration = task.duration
            
            if actual_duration >= expected_duration:
                return 80.0  # Likely near completion
            else:
                return (actual_duration.total_seconds() / expected_duration.total_seconds()) * 80
        
        # Generic estimation based on activity
        base_progress = 20.0  # Started
        
        # Add progress for states visited
        state_progress = min(30.0, len(task.visited_states) * 5)
        
        # Add progress for time spent
        time_progress = min(30.0, task.duration.total_seconds() / 3600 * 30)  # Up to 30% for an hour
        
        # Check for completion indicators
        if task.progress_indicators:
            if ProgressIndicator.FORWARD_PROGRESS in task.progress_indicators[-3:]:
                base_progress += 10
        
        return min(95.0, base_progress + state_progress + time_progress)
    
    def detect_abandonment(self, task: RecognizedTask) -> bool:
        """Detect if task has been abandoned"""
        if task.status != TaskStatus.IN_PROGRESS:
            return False
        
        # Check time since last progress
        time_since_progress = datetime.now() - (task.last_progress_time or task.start_time)
        
        if time_since_progress > self.abandonment_threshold:
            # Check if there's any recent activity
            history = self.progress_history.get(task.task_id, [])
            if history:
                last_snapshot = history[-1]
                if (datetime.now() - last_snapshot['timestamp']) > self.abandonment_threshold:
                    task.status = TaskStatus.ABANDONED
                    return True
        
        return False


class ActivityRecognitionEngine:
    """Main engine coordinating activity recognition"""
    
    def __init__(self):
        self.task_identifier = TaskIdentifier()
        self.task_inferencer = TaskInferencer(self.task_identifier)
        self.progress_monitor = ProgressMonitor()
        
        # Task tracking
        self.active_tasks: Dict[str, RecognizedTask] = {}
        self.completed_tasks: deque = deque(maxlen=100)
        self.task_history: List[RecognizedTask] = []
        
        # Memory management
        self.memory_usage = {
            'task_templates': 0,
            'activity_history': 0,
            'progress_tracking': 0
        }
        
        # Load saved data
        self._load_saved_state()
    
    async def recognize_activity(self, context: Dict[str, Any]) -> RecognizedTask:
        """Recognize current user activity and task"""
        # Step 1: Identify primary activity
        active_apps = context.get('active_applications', [])
        primary_activity, activity_confidence = self.task_identifier.identify_activity_from_apps(active_apps)
        
        logger.info(f"Identified primary activity: {primary_activity.value} (confidence: {activity_confidence:.2f})")
        
        # Step 2: Check if continuing existing task
        existing_task = self._find_continuing_task(primary_activity, context)
        
        if existing_task:
            # Update existing task
            await self.task_inferencer.update_task_confidence(existing_task, context)
            indicator = await self.progress_monitor.update_progress(existing_task, context)
            
            logger.info(f"Continuing task: {existing_task.name} (progress: {existing_task.completion_percentage:.1f}%)")
            
            return existing_task
        
        # Step 3: Infer new task
        new_task = await self.task_inferencer.infer_task(
            primary_activity, 
            context,
            list(self.active_tasks.values())
        )
        
        # Add to active tasks
        self.active_tasks[new_task.task_id] = new_task
        self.task_history.append(new_task)
        
        logger.info(f"Started new task: {new_task.name} (confidence: {new_task.confidence:.2f})")
        
        return new_task
    
    def _find_continuing_task(self, 
                            activity: PrimaryActivity,
                            context: Dict[str, Any]) -> Optional[RecognizedTask]:
        """Find if user is continuing an existing task"""
        current_apps = set(context.get('active_applications', []))
        
        for task in self.active_tasks.values():
            # Check if same activity and significant app overlap
            if task.primary_activity == activity and task.status == TaskStatus.IN_PROGRESS:
                app_overlap = len(current_apps & task.active_applications) / len(task.active_applications) if task.active_applications else 0
                
                # Check time gap
                time_gap = datetime.now() - (task.last_progress_time or task.start_time)
                
                if app_overlap > 0.5 and time_gap < timedelta(minutes=30):
                    return task
        
        return None
    
    async def update_task_progress(self, task_id: str, context: Dict[str, Any]) -> ProgressIndicator:
        """Update progress for a specific task"""
        task = self.active_tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        return await self.progress_monitor.update_progress(task, context)
    
    def complete_task(self, task_id: str, success: bool = True):
        """Mark a task as completed"""
        task = self.active_tasks.get(task_id)
        if not task:
            return
        
        task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        task.completion_percentage = 100.0 if success else task.completion_percentage
        
        # Move to completed
        self.completed_tasks.append(task)
        del self.active_tasks[task_id]
        
        logger.info(f"Task completed: {task.name} (success: {success})")
    
    def get_current_activities(self) -> List[RecognizedTask]:
        """Get all current active tasks"""
        # Check for abandoned tasks
        for task in list(self.active_tasks.values()):
            if self.progress_monitor.detect_abandonment(task):
                self.completed_tasks.append(task)
                del self.active_tasks[task.task_id]
        
        return list(self.active_tasks.values())
    
    def get_task_insights(self, task_id: str) -> Dict[str, Any]:
        """Get detailed insights for a task"""
        task = self.active_tasks.get(task_id)
        if not task:
            # Check completed tasks
            task = next((t for t in self.completed_tasks if t.task_id == task_id), None)
            if not task:
                return {'error': 'Task not found'}
        
        insights = {
            'task_id': task.task_id,
            'name': task.name,
            'primary_activity': task.primary_activity.value,
            'status': task.status.name,
            'duration': str(task.duration),
            'completion_percentage': task.completion_percentage,
            'confidence': task.confidence,
            'is_stuck': task.is_stuck,
            'stuck_duration': str(task.stuck_duration) if task.stuck_duration else None,
            'active_applications': list(task.active_applications),
            'visited_states': len(task.visited_states),
            'progress_history': [ind.value for ind in task.progress_indicators[-10:]],
            'sub_tasks': task.sub_tasks,
            'parent_task': task.parent_task,
            'blocking_factors': task.blocking_factors
        }
        
        # Add template info if available
        if task.template_id:
            template = self.task_identifier.task_templates.get(task.template_id)
            if template:
                insights['expected_duration'] = str(template.expected_duration)
                insights['success_indicators'] = template.success_indicators
                insights['prerequisites'] = template.prerequisites
        
        return insights
    
    def get_activity_summary(self) -> Dict[str, Any]:
        """Get summary of all activities"""
        summary = {
            'active_tasks': len(self.active_tasks),
            'completed_today': sum(1 for t in self.completed_tasks 
                                 if t.start_time.date() == datetime.now().date()),
            'tasks_by_activity': defaultdict(int),
            'average_durations': defaultdict(list),
            'success_rate': 0.0,
            'common_blockers': Counter(),
            'productivity_score': 0.0
        }
        
        # Analyze all tasks
        all_tasks = list(self.active_tasks.values()) + list(self.completed_tasks)
        
        for task in all_tasks:
            summary['tasks_by_activity'][task.primary_activity.value] += 1
            
            if task.status == TaskStatus.COMPLETED:
                summary['average_durations'][task.primary_activity.value].append(
                    task.duration.total_seconds() / 60  # Minutes
                )
        
        # Calculate averages
        for activity, durations in summary['average_durations'].items():
            if durations:
                summary['average_durations'][activity] = statistics.mean(durations)
        
        # Calculate success rate
        completed = [t for t in all_tasks if t.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]]
        if completed:
            successful = sum(1 for t in completed if t.status == TaskStatus.COMPLETED)
            summary['success_rate'] = successful / len(completed)
        
        # Find common blockers
        for task in all_tasks:
            for blocker in task.blocking_factors:
                summary['common_blockers'][blocker] += 1
        
        # Calculate productivity score
        summary['productivity_score'] = self._calculate_productivity_score(all_tasks)
        
        # Add memory usage
        summary['memory_usage'] = self._calculate_memory_usage()
        
        return dict(summary)
    
    def _calculate_productivity_score(self, tasks: List[RecognizedTask]) -> float:
        """Calculate overall productivity score"""
        if not tasks:
            return 0.0
        
        scores = []
        
        # Task completion rate
        completed = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        completion_rate = completed / len(tasks) if tasks else 0
        scores.append(completion_rate)
        
        # Time efficiency (actual vs expected for template tasks)
        template_tasks = [t for t in tasks if t.template_id and t.status == TaskStatus.COMPLETED]
        if template_tasks:
            efficiencies = []
            for task in template_tasks:
                template = self.task_identifier.task_templates.get(task.template_id)
                if template:
                    expected = template.expected_duration.total_seconds()
                    actual = task.duration.total_seconds()
                    efficiency = min(1.0, expected / actual) if actual > 0 else 0
                    efficiencies.append(efficiency)
            if efficiencies:
                scores.append(statistics.mean(efficiencies))
        
        # Stuck time ratio
        total_time = sum(t.duration.total_seconds() for t in tasks)
        stuck_time = sum(t.stuck_duration.total_seconds() for t in tasks 
                        if t.stuck_duration)
        if total_time > 0:
            stuck_ratio = 1.0 - (stuck_time / total_time)
            scores.append(stuck_ratio)
        
        return statistics.mean(scores) if scores else 0.0
    
    def _calculate_memory_usage(self) -> Dict[str, int]:
        """Calculate memory usage of components"""
        import sys
        
        # Task templates
        self.memory_usage['task_templates'] = sum(
            sys.getsizeof(template) for template in self.task_identifier.task_templates.values()
        )
        
        # Activity history
        self.memory_usage['activity_history'] = (
            sum(sys.getsizeof(task) for task in self.active_tasks.values()) +
            sum(sys.getsizeof(task) for task in self.completed_tasks)
        )
        
        # Progress tracking
        self.memory_usage['progress_tracking'] = sum(
            sys.getsizeof(history) for history in self.progress_monitor.progress_history.values()
        )
        
        return self.memory_usage
    
    def save_state(self):
        """Save current state to disk"""
        state = {
            'active_tasks': {
                task_id: self._serialize_task(task) 
                for task_id, task in self.active_tasks.items()
            },
            'completed_tasks': [
                self._serialize_task(task) for task in self.completed_tasks
            ][-50:],  # Keep last 50
            'task_history': [
                {'task_id': t.task_id, 'name': t.name, 'activity': t.primary_activity.value}
                for t in self.task_history[-100:]
            ]
        }
        
        try:
            save_path = Path("activity_recognition_state.json")
            with open(save_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            logger.info("Saved activity recognition state")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _serialize_task(self, task: RecognizedTask) -> Dict[str, Any]:
        """Serialize task for saving"""
        return {
            'task_id': task.task_id,
            'template_id': task.template_id,
            'name': task.name,
            'primary_activity': task.primary_activity.value,
            'start_time': task.start_time.isoformat(),
            'status': task.status.name,
            'confidence': task.confidence,
            'completion_percentage': task.completion_percentage,
            'active_applications': list(task.active_applications),
            'visited_states': task.visited_states[-20:],  # Keep last 20
            'parent_task': task.parent_task,
            'sub_tasks': task.sub_tasks,
            'blocking_factors': task.blocking_factors
        }
    
    def _load_saved_state(self):
        """Load saved state from disk"""
        try:
            load_path = Path("activity_recognition_state.json")
            if load_path.exists():
                with open(load_path, 'r') as f:
                    state = json.load(f)
                
                # Restore completed tasks (active tasks are likely stale)
                for task_data in state.get('completed_tasks', [])[-20:]:
                    task = self._deserialize_task(task_data)
                    if task:
                        self.completed_tasks.append(task)
                
                logger.info(f"Loaded {len(self.completed_tasks)} completed tasks")
        
        except Exception as e:
            logger.error(f"Failed to load saved state: {e}")
    
    def _deserialize_task(self, data: Dict[str, Any]) -> Optional[RecognizedTask]:
        """Deserialize task from saved data"""
        try:
            task = RecognizedTask(
                task_id=data['task_id'],
                template_id=data.get('template_id'),
                name=data['name'],
                primary_activity=PrimaryActivity(data['primary_activity']),
                start_time=datetime.fromisoformat(data['start_time']),
                status=TaskStatus[data['status']],
                confidence=data.get('confidence', 1.0)
            )
            
            task.completion_percentage = data.get('completion_percentage', 0.0)
            task.active_applications = set(data.get('active_applications', []))
            task.visited_states = data.get('visited_states', [])
            task.parent_task = data.get('parent_task')
            task.sub_tasks = data.get('sub_tasks', [])
            task.blocking_factors = data.get('blocking_factors', [])
            
            return task
        except Exception as e:
            logger.error(f"Failed to deserialize task: {e}")
            return None


# Global instance
_activity_engine_instance = None

def get_activity_engine() -> ActivityRecognitionEngine:
    """Get or create the global activity recognition engine"""
    global _activity_engine_instance
    if _activity_engine_instance is None:
        _activity_engine_instance = ActivityRecognitionEngine()
    return _activity_engine_instance