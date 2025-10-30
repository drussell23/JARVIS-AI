"""
Predictive Analytics Engine for JARVIS
======================================

This module provides comprehensive predictive and analytical capabilities for JARVIS,
handling high-level queries about progress, recommendations, bug detection, and workflow optimization.

The engine uses a multi-stage analysis pipeline:
    Query → Intent Classifier → Metric Analyzer → Claude Vision → Response Generator
       ↓            ↓                    ↓              ↓                ↓
    Parse      Determine           Calculate      Semantic      Natural Language
    Intent     Query Type          Metrics        Analysis         Response

Key Features:
1. **Progress Analysis** - Tracks git commits, code changes, test results
2. **Bug Detection** - Analyzes patterns in errors, test failures, code smells
3. **Recommendations** - Suggests next tasks based on project state
4. **Semantic Understanding** - Uses Claude Vision for deep code analysis
5. **Zero Hardcoding** - Fully dynamic, configurable query types

Architecture:
    - PredictiveAnalyzer: Main orchestrator
    - GitMetricsCollector: Collects git repository metrics
    - ErrorPatternCollector: Tracks and analyzes error patterns
    - WorkflowAnalyzer: Analyzes productivity and workflow patterns
    - RecommendationEngine: Generates actionable recommendations

Example:
    >>> analyzer = PredictiveAnalyzer()
    >>> result = await analyzer.analyze("Am I making progress?")
    >>> print(result.response_text)

Author: Derek Russell
Date: 2025-10-19
"""

import asyncio
import logging
import re
import json
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import hashlib

logger = logging.getLogger(__name__)


# ============================================================================
# QUERY TYPES
# ============================================================================

class PredictiveQueryType(Enum):
    """Types of predictive/analytical queries supported by the system.
    
    Each query type corresponds to a specific analysis pipeline and response format.
    """
    PROGRESS_CHECK = "progress_check"              # "Am I making progress?"
    NEXT_STEPS = "next_steps"                      # "What should I work on next?"
    BUG_DETECTION = "bug_detection"                # "Are there any bugs?"
    CODE_EXPLANATION = "code_explanation"           # "Explain this code"
    PATTERN_ANALYSIS = "pattern_analysis"          # "What patterns do you see?"
    WORKFLOW_OPTIMIZATION = "workflow_optimization" # "How can I improve my workflow?"
    QUALITY_ASSESSMENT = "quality_assessment"      # "How's my code quality?"
    TIME_ESTIMATION = "time_estimation"            # "How long will this take?"
    RISK_ASSESSMENT = "risk_assessment"            # "What could go wrong?"
    UNKNOWN = "unknown"


class AnalysisScope(Enum):
    """Scope of analysis for predictive queries.
    
    Determines what data sources and time ranges to include in the analysis.
    """
    CURRENT_SPACE = "current_space"      # Current visible workspace
    ALL_SPACES = "all_spaces"            # All monitored spaces
    SPECIFIC_PROJECT = "specific_project" # Specific git project
    TIMEFRAME = "timeframe"              # Specific time period


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ProgressMetrics:
    """Progress metrics for a project or workspace.
    
    Attributes:
        commits_today: Number of commits made today
        commits_this_week: Number of commits made this week
        lines_added: Lines of code added
        lines_removed: Lines of code removed
        files_modified: Number of files changed
        test_pass_rate: Percentage of tests passing (0.0-1.0)
        build_success_rate: Percentage of builds succeeding (0.0-1.0)
        active_time_minutes: Minutes of active development time
        last_commit_time: Timestamp of most recent commit
        velocity_trend: Trend in commit velocity ("increasing", "decreasing", "stable")
        metadata: Additional metrics and context data
    """
    commits_today: int = 0
    commits_this_week: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    files_modified: int = 0
    test_pass_rate: float = 0.0
    build_success_rate: float = 0.0
    active_time_minutes: int = 0
    last_commit_time: Optional[datetime] = None
    velocity_trend: str = "stable"  # "increasing", "decreasing", "stable"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BugPattern:
    """Detected bug pattern with analysis details.
    
    Attributes:
        pattern_type: Type of bug pattern (e.g., "syntax_error", "test_failure")
        frequency: Number of times this pattern has occurred
        last_occurrence: Timestamp of most recent occurrence
        locations: File paths or space IDs where pattern was detected
        severity: Severity level ("critical", "high", "medium", "low")
        description: Human-readable description of the pattern
        suggested_fix: Optional suggestion for fixing the pattern
        confidence: Confidence score in pattern detection (0.0-1.0)
    """
    pattern_type: str               # "syntax_error", "test_failure", "runtime_error"
    frequency: int                  # How many times seen
    last_occurrence: datetime
    locations: List[str]            # File paths or space IDs
    severity: str                   # "critical", "high", "medium", "low"
    description: str
    suggested_fix: Optional[str] = None
    confidence: float = 0.0


@dataclass
class Recommendation:
    """Actionable recommendation generated by the system.
    
    Attributes:
        category: Category of recommendation ("next_task", "optimization", "bug_fix")
        priority: Priority level ("high", "medium", "low")
        title: Short title for the recommendation
        description: Detailed description of what to do
        rationale: Explanation of why this recommendation was made
        action_items: List of specific actions to take
        estimated_time: Estimated time to complete (optional)
        confidence: Confidence in the recommendation (0.0-1.0)
        metadata: Additional context and data
    """
    category: str                   # "next_task", "optimization", "bug_fix"
    priority: str                   # "high", "medium", "low"
    title: str
    description: str
    rationale: str                  # Why this recommendation
    action_items: List[str]
    estimated_time: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyticsResult:
    """Result of a predictive/analytical query.
    
    Attributes:
        query_type: Type of query that was analyzed
        success: Whether the analysis completed successfully
        confidence: Overall confidence in the analysis results (0.0-1.0)
        insights: List of high-level insights discovered
        metrics: Progress metrics (if applicable)
        bug_patterns: List of detected bug patterns
        recommendations: List of actionable recommendations
        visualizations: List of text/markdown visualizations
        raw_data: Raw analysis data for debugging/extension
        response_text: Natural language response text
        metadata: Additional result metadata
    """
    query_type: PredictiveQueryType
    success: bool
    confidence: float
    insights: List[str]             # High-level insights
    metrics: Optional[ProgressMetrics] = None
    bug_patterns: List[BugPattern] = field(default_factory=list)
    recommendations: List[Recommendation] = field(default_factory=list)
    visualizations: List[str] = field(default_factory=list)  # Markdown/text visualizations
    raw_data: Dict[str, Any] = field(default_factory=dict)
    response_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# METRICS COLLECTORS
# ============================================================================

class GitMetricsCollector:
    """Collects metrics from git repositories.
    
    Provides caching to avoid expensive git operations on repeated calls.
    Tracks commit activity, code changes, and velocity trends.
    
    Attributes:
        _cache: Cache of metrics by repository path
        _cache_duration: How long to cache metrics before refreshing
    """

    def __init__(self):
        """Initialize the git metrics collector."""
        self._cache: Dict[str, Tuple[ProgressMetrics, datetime]] = {}
        self._cache_duration = timedelta(minutes=5)

    async def collect_metrics(self, repo_path: str) -> ProgressMetrics:
        """Collect git metrics for a repository.
        
        Args:
            repo_path: Path to the git repository
            
        Returns:
            ProgressMetrics object with collected data
            
        Example:
            >>> collector = GitMetricsCollector()
            >>> metrics = await collector.collect_metrics("/path/to/repo")
            >>> print(f"Commits today: {metrics.commits_today}")
        """
        # Check cache
        if repo_path in self._cache:
            metrics, timestamp = self._cache[repo_path]
            if datetime.now() - timestamp < self._cache_duration:
                logger.debug(f"[GIT-METRICS] Using cached metrics for {repo_path}")
                return metrics

        metrics = ProgressMetrics()

        try:
            import subprocess

            # Get commits today
            result = await asyncio.create_subprocess_exec(
                "git", "-C", repo_path, "log", "--since='1 day ago'", "--oneline",
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, _ = await result.communicate()
            metrics.commits_today = len(stdout.decode().strip().split('\n')) if stdout else 0

            # Get commits this week
            result = await asyncio.create_subprocess_exec(
                "git", "-C", repo_path, "log", "--since='1 week ago'", "--oneline",
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, _ = await result.communicate()
            metrics.commits_this_week = len(stdout.decode().strip().split('\n')) if stdout else 0

            # Get diff stats for today
            result = await asyncio.create_subprocess_exec(
                "git", "-C", repo_path, "diff", "--stat", "@{1.day.ago}..HEAD",
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, _ = await result.communicate()
            stats = stdout.decode()

            # Parse insertions/deletions
            insertions = re.findall(r'(\d+) insertions?', stats)
            deletions = re.findall(r'(\d+) deletions?', stats)
            metrics.lines_added = int(insertions[0]) if insertions else 0
            metrics.lines_removed = int(deletions[0]) if deletions else 0

            # Get files changed
            files_changed = re.findall(r'(\d+) files? changed', stats)
            metrics.files_modified = int(files_changed[0]) if files_changed else 0

            # Get last commit time
            result = await asyncio.create_subprocess_exec(
                "git", "-C", repo_path, "log", "-1", "--format=%ct",
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, _ = await result.communicate()
            if stdout:
                timestamp = int(stdout.decode().strip())
                metrics.last_commit_time = datetime.fromtimestamp(timestamp)

            # Calculate velocity trend
            metrics.velocity_trend = self._calculate_velocity_trend(metrics)

            # Cache the result
            self._cache[repo_path] = (metrics, datetime.now())

            logger.info(f"[GIT-METRICS] Collected metrics: {metrics.commits_today} commits today, {metrics.files_modified} files modified")

        except Exception as e:
            logger.error(f"[GIT-METRICS] Error collecting metrics: {e}")

        return metrics

    def _calculate_velocity_trend(self, metrics: ProgressMetrics) -> str:
        """Calculate velocity trend based on commit patterns.
        
        Args:
            metrics: Current progress metrics
            
        Returns:
            Velocity trend: "increasing", "decreasing", or "stable"
        """
        daily_avg = metrics.commits_this_week / 7 if metrics.commits_this_week > 0 else 0

        if metrics.commits_today > daily_avg * 1.2:
            return "increasing"
        elif metrics.commits_today < daily_avg * 0.8:
            return "decreasing"
        else:
            return "stable"


class ErrorPatternCollector:
    """Collects and analyzes error patterns over time.
    
    Tracks recurring errors, failures, and issues to identify patterns
    that might indicate systemic problems or areas needing attention.
    
    Attributes:
        error_history: Deque of recent error occurrences
        pattern_frequency: Counter of error types by frequency
    """

    def __init__(self, max_patterns: int = 100):
        """Initialize the error pattern collector.
        
        Args:
            max_patterns: Maximum number of error patterns to track
        """
        self.error_history: deque[Tuple[str, datetime, Dict[str, Any]]] = deque(maxlen=max_patterns)
        self.pattern_frequency: defaultdict[str, int] = defaultdict(int)

    def record_error(self, error_type: str, location: str, details: Dict[str, Any]):
        """Record an error occurrence for pattern analysis.
        
        Args:
            error_type: Type/category of error
            location: Where the error occurred (file path, space ID, etc.)
            details: Additional error details and context
            
        Example:
            >>> collector = ErrorPatternCollector()
            >>> collector.record_error("syntax_error", "main.py", {"line": 42})
        """
        self.error_history.append((error_type, datetime.now(), {
            "location": location,
            **details
        }))
        self.pattern_frequency[error_type] += 1
        logger.debug(f"[ERROR-PATTERN] Recorded error: {error_type} at {location}")

    async def analyze_patterns(self, within_hours: int = 24) -> List[BugPattern]:
        """Analyze error patterns within a time window.
        
        Args:
            within_hours: Time window to analyze (hours)
            
        Returns:
            List of detected bug patterns, sorted by severity and frequency
            
        Example:
            >>> patterns = await collector.analyze_patterns(within_hours=12)
            >>> for pattern in patterns:
            ...     print(f"{pattern.pattern_type}: {pattern.frequency}x")
        """
        cutoff = datetime.now() - timedelta(hours=within_hours)
        patterns = []

        # Group errors by type
        grouped_errors = defaultdict(list)
        for error_type, timestamp, details in self.error_history:
            if timestamp > cutoff:
                grouped_errors[error_type].append((timestamp, details))

        # Analyze each error type
        for error_type, occurrences in grouped_errors.items():
            if len(occurrences) >= 2:  # Only patterns that repeat
                locations = [occ[1].get("location", "unknown") for occ in occurrences]
                last_occurrence = max(occ[0] for occ in occurrences)

                # Determine severity
                severity = self._calculate_severity(error_type, len(occurrences))

                pattern = BugPattern(
                    pattern_type=error_type,
                    frequency=len(occurrences),
                    last_occurrence=last_occurrence,
                    locations=list(set(locations)),
                    severity=severity,
                    description=f"{error_type} occurred {len(occurrences)} times in the last {within_hours}h",
                    confidence=min(0.9, 0.5 + (len(occurrences) * 0.1))
                )

                patterns.append(pattern)

        # Sort by severity and frequency
        patterns.sort(key=lambda p: (
            {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(p.severity, 0),
            p.frequency
        ), reverse=True)

        logger.info(f"[ERROR-PATTERN] Found {len(patterns)} bug patterns")
        return patterns

    def _calculate_severity(self, error_type: str, frequency: int) -> str:
        """Calculate severity based on error type and frequency.
        
        Args:
            error_type: Type of error
            frequency: How many times it occurred
            
        Returns:
            Severity level: "critical", "high", "medium", or "low"
        """
        critical_keywords = ["crash", "segfault", "fatal", "critical"]
        high_keywords = ["exception", "failure", "error"]

        error_lower = error_type.lower()

        if any(kw in error_lower for kw in critical_keywords) or frequency > 10:
            return "critical"
        elif any(kw in error_lower for kw in high_keywords) or frequency > 5:
            return "high"
        elif frequency > 2:
            return "medium"
        else:
            return "low"


class WorkflowAnalyzer:
    """Analyzes workflow patterns and efficiency metrics.
    
    Tracks user activities, context switches, and focus patterns
    to provide insights about productivity and workflow optimization.
    
    Attributes:
        activity_log: Deque of recent workflow activities
    """

    def __init__(self):
        """Initialize the workflow analyzer."""
        self.activity_log: deque[Tuple[str, datetime, Dict[str, Any]]] = deque(maxlen=500)

    def record_activity(self, activity_type: str, details: Dict[str, Any]):
        """Record a workflow activity for analysis.
        
        Args:
            activity_type: Type of activity (e.g., "code_edit", "space_switch")
            details: Activity details and context
            
        Example:
            >>> analyzer = WorkflowAnalyzer()
            >>> analyzer.record_activity("code_edit", {"space_id": "main", "file": "app.py"})
        """
        self.activity_log.append((activity_type, datetime.now(), details))

    async def analyze_workflow(self, within_hours: int = 24) -> Dict[str, Any]:
        """Analyze workflow patterns within a time window.
        
        Args:
            within_hours: Time window to analyze (hours)
            
        Returns:
            Dictionary containing workflow metrics and analysis
            
        Example:
            >>> analysis = await analyzer.analyze_workflow(within_hours=8)
            >>> print(f"Efficiency score: {analysis['efficiency_score']:.1%}")
        """
        cutoff = datetime.now() - timedelta(hours=within_hours)

        recent_activities = [
            (activity, timestamp, details)
            for activity, timestamp, details in self.activity_log
            if timestamp > cutoff
        ]

        if not recent_activities:
            return {
                "total_activities": 0,
                "focus_time": 0,
                "context_switches": 0,
                "efficiency_score": 0.0
            }

        # Calculate metrics
        total_activities = len(recent_activities)

        # Count context switches (changing spaces/apps)
        context_switches = 0
        last_context = None
        for activity, _, details in recent_activities:
            current_context = details.get("space_id", details.get("app_name"))
            if last_context and current_context != last_context:
                context_switches += 1
            last_context = current_context

        # Estimate focus time (time between activities < 5 min)
        focus_periods = []
        for i in range(1, len(recent_activities)):
            time_diff = (recent_activities[i][1] - recent_activities[i-1][1]).total_seconds() / 60
            if time_diff < 5:  # Focused if < 5 min between activities
                focus_periods.append(time_diff)

        focus_time = sum(focus_periods)

        # Calculate efficiency score (0-1)
        # High focus time, low context switches = high efficiency
        max_possible_focus = within_hours * 60
        efficiency_score = (focus_time / max_possible_focus) * (1 - min(1.0, context_switches / total_activities))

        return {
            "total_activities": total_activities,
            "focus_time_minutes": int(focus_time),
            "context_switches": context_switches,
            "efficiency_score": round(efficiency_score, 2),
            "avg_focus_period_minutes": round(sum(focus_periods) / len(focus_periods), 1) if focus_periods else 0
        }


# ============================================================================
# RECOMMENDATION ENGINE
# ============================================================================

class RecommendationEngine:
    """Generates actionable recommendations based on analysis results.
    
    Takes metrics, patterns, and context to produce prioritized recommendations
    for improving productivity, fixing issues, and optimizing workflow.
    """

    async def generate_recommendations(
        self,
        query_type: PredictiveQueryType,
        metrics: Optional[ProgressMetrics],
        bug_patterns: List[BugPattern],
        workflow_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Recommendation]:
        """Generate recommendations based on analysis results.
        
        Args:
            query_type: Type of query being analyzed
            metrics: Progress metrics (if available)
            bug_patterns: Detected bug patterns
            workflow_data: Workflow analysis results
            context: Additional context and parameters
            
        Returns:
            List of recommendations sorted by priority
            
        Example:
            >>> engine = RecommendationEngine()
            >>> recs = await engine.generate_recommendations(
            ...     PredictiveQueryType.NEXT_STEPS, metrics, [], {}, {}
            ... )
        """
        recommendations = []

        if query_type == PredictiveQueryType.NEXT_STEPS:
            recommendations.extend(await self._recommend_next_steps(metrics, bug_patterns, context))
        elif query_type == PredictiveQueryType.BUG_DETECTION:
            recommendations.extend(await self._recommend_bug_fixes(bug_patterns))
        elif query_type == PredictiveQueryType.WORKFLOW_OPTIMIZATION:
            recommendations.extend(await self._recommend_workflow_improvements(workflow_data))
        elif query_type == PredictiveQueryType.PROGRESS_CHECK:
            recommendations.extend(await self._recommend_progress_actions(metrics, workflow_data))

        # Sort by priority
        priority_order = {"high": 3, "medium": 2, "low": 1}
        recommendations.sort(key=lambda r: priority_order.get(r.priority, 0), reverse=True)

        return recommendations

    async def _recommend_next_steps(
        self,
        metrics: Optional[ProgressMetrics],
        bug_patterns: List[BugPattern],
        context: Dict[str, Any]
    ) -> List[Recommendation]:
        """Recommend next steps based on current state.
        
        Args:
            metrics: Current progress metrics
            bug_patterns: Detected bug patterns
            context: Additional context
            
        Returns:
            List of next step recommendations
        """
        recommendations = []

        # If there are critical bugs, prioritize fixing them
        critical_bugs = [b for b in bug_patterns if b.severity == "critical"]
        if critical_bugs:
            recommendations.append(Recommendation(
                category="bug_fix",
                priority="high",
                title="Fix Critical Bugs",
                description=f"Address {len(critical_bugs)} critical bug(s) detected in your codebase",
                rationale="Critical bugs can cause system instability and should be addressed immediately",
                action_items=[f"Fix {bug.pattern_type} in {', '.join(bug.locations[:2])}" for bug in critical_bugs[:3]],
                estimated_time="30-60 minutes",
                confidence=0.9
            ))

        # If low commit activity, suggest getting started
        if metrics and metrics.commits_today == 0:
            recommendations.append(Recommendation(
                category="next_task",
                priority="medium",
                title="Start Daily Work",
                description="No commits today - consider starting with a small, manageable task",
                rationale="Consistent daily progress helps maintain momentum",
                action_items=[
                    "Review open issues or TODOs",
                    "Pick one small task to complete",
                    "Make your first commit"
                ],
                estimated_time="15-30 minutes",
                confidence=0.7
            ))

        # If high velocity, suggest optimization or refactoring
        if metrics and metrics.velocity_trend == "increasing" and metrics.commits_today > 5:
            recommendations.append(Recommendation(
                category="optimization",
                priority="low",
                title="Consider Refactoring",
                description="High commit velocity - good time to refactor and clean up",
                rationale="Active development periods are good opportunities to improve code quality",
                action_items=[
                    "Review recent changes for duplicate code",
                    "Add tests for new functionality",
                    "Update documentation"
                ],
                estimated_time="20-40 minutes",
                confidence=0.6
            ))

        return recommendations

    async def _recommend_bug_fixes(self, bug_patterns: List[BugPattern]) -> List[Recommendation]:
        """Recommend bug fixes based on detected patterns.
        
        Args:
            bug_patterns: List of detected bug patterns
            
        Returns:
            List of bug fix recommendations
        """
        recommendations = []

        for bug in bug_patterns[:5]:  # Top 5 bugs
            priority = "high" if bug.severity in ["critical", "high"] else "medium"

            recommendations.append(Recommendation(
                category="bug_fix",
                priority=priority,
                title=f"Fix {bug.pattern_type}",
                description=f"{bug.description} in {', '.join(bug.locations[:2])}",
                rationale=f"This {bug.severity} severity bug has occurred {bug.frequency} times",
                action_items=[
                    f"Investigate {bug.pattern_type} in {bug.locations[0]}",
                    "Review recent code changes in affected areas",
                    "Add tests to prevent regression"
                ],
                confidence=bug.confidence,
                metadata={"bug_pattern": bug}
            ))

        return recommendations

    async def _recommend_workflow_improvements(self, workflow_data: Dict[str, Any]) -> List[Recommendation]:
        """Recommend workflow improvements based on analysis.
        
        Args:
            workflow_data: Workflow analysis results
            
        Returns:
            List of workflow improvement recommendations
        """
        recommendations = []

        efficiency = workflow_data.get("efficiency_score", 0.0)
        context_switches = workflow_data.get("context_switches", 0)

        if efficiency < 0.5:
            recommendations.append(Recommendation(
                category="optimization",
                priority="medium",
                title="Improve Focus Time",
                description=f"Current efficiency score: {efficiency:.1%}",
                rationale="Low efficiency indicates frequent interruptions or context switching",
                action_items=[
                    "Try time-blocking for focused work",
                    "Minimize distractions during coding sessions",
                    "Use 'Do Not Disturb' mode"
                ],
                estimated_time="Ongoing",
                confidence=0.7
            ))

        if context_switches > 20:
            recommendations.append(Recommendation(
                category="optimization",
                priority="medium",
                title="Reduce Context Switching",
                description=f"{context_switches} context switches detected",
                rationale="Frequent context switching reduces productivity",
                action_items=[
                    "Group similar tasks together",
                    "Complete one task before starting another",
                    "Use single workspace for related work"
                ],
                estimated_time="Ongoing",
                confidence=0.8
            ))

        return recommendations

    async def _recommend_progress_actions(
        self,
        metrics: Optional[ProgressMetrics],
        workflow_data: Dict[str, Any]
    ) -> List[Recommendation]:
        """Recommend actions based on progress analysis.
        
        Args:
            metrics: Progress metrics
            workflow_data: Workflow analysis results
            
        Returns:
            List of progress-related recommendations
        """
        recommendations = []

        if metrics and metrics.velocity_trend == "decreasing":
            recommendations.append(Recommendation(
                category="next_task",
                priority="medium",
                title="Boost Velocity",
                description="Commit velocity is decreasing - consider taking action",
                rationale="Maintaining consistent progress prevents project delays",
                action_items=[
                    "Review blockers or impediments",
                    "Break large tasks into smaller ones",
                    "Set daily commit goals"
                ],
                estimated_time="15 minutes planning",
                confidence=0.6
            ))

        return recommendations


# ============================================================================
# MAIN PREDICTIVE ANALYZER
# ============================================================================

class PredictiveAnalyzer:
    """Main predictive analytics engine for JARVIS.

    Orchestrates the entire analysis pipeline from query classification
    to response generation. Handles high-level analytical queries with
    dynamic analysis and actionable recommendations.
    
    Attributes:
        context_graph: Optional context graph for enhanced analysis
        git_metrics: Git metrics collector (if enabled)
        error_patterns: Error pattern collector (if enabled)
        workflow_analyzer: Workflow analyzer (if enabled)
        recommender: Recommendation engine
        query_patterns: Regex patterns for query classification
    """

    def __init__(
        self,
        context_graph=None,
        enable_git_metrics: bool = True,
        enable_error_tracking: bool = True,
        enable_workflow_analysis: bool = True
    ):
        """Initialize the predictive analyzer.
        
        Args:
            context_graph: Optional context graph for enhanced analysis
            enable_git_metrics: Whether to enable git metrics collection
            enable_error_tracking: Whether to enable error pattern tracking
            enable_workflow_analysis: Whether to enable workflow analysis
        """
        self.context_graph = context_graph

        # Collectors
        self.git_metrics = GitMetricsCollector() if enable_git_metrics else None
        self.error_patterns = ErrorPatternCollector() if enable_error_tracking else None
        self.workflow_analyzer = WorkflowAnalyzer() if enable_workflow_analysis else None

        # Recommendation engine
        self.recommender = RecommendationEngine()

        # Query patterns (dynamic, extensible)
        self.query_patterns = self._initialize_query_patterns()

        logger.info("[PREDICTIVE-ANALYZER] Initialized")

    def _initialize_query_patterns(self) -> Dict[PredictiveQueryType, List[re.Pattern]]:
        """Initialize regex patterns for query classification.
        
        Returns:
            Dictionary mapping query types to regex patterns
        """
        return {
            PredictiveQueryType.PROGRESS_CHECK: [
                re.compile(r'\b(am i|are we)\s+(making|seeing)\s+progress\b', re.I),
                re.compile(r'\bhow\s+(much|far|well)\s+(progress|am i doing)\b', re.I),
                re.compile(r'\bwhat\'?s\s+my\s+progress\b', re.I),
            ],
            PredictiveQueryType.NEXT_STEPS: [
                re.compile(r'\bwhat\s+should\s+i\s+(do|work on)\s+next\b', re.I),
                re.compile(r'\b(next\s+steps|what\'?s\s+