#!/usr/bin/env python3
"""
Cross-Space Context Awareness System
Analyzes relationships and context across multiple desktop spaces
"""

import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class WorkflowContext:
    """Context about user's workflow across spaces"""
    primary_task: str
    related_spaces: List[int]
    key_applications: List[str]
    workflow_type: str  # development, research, communication, etc.
    intensity_level: float  # 0-1 scale of activity
    time_spent: timedelta
    transitions: List[Tuple[int, int, datetime]]  # space transitions

@dataclass
class SpaceContext:
    """Enhanced context for a single space"""
    space_id: int
    purpose: str
    active_task: str
    related_spaces: List[int]
    key_files: List[str]
    key_urls: List[str]
    activity_score: float
    last_interaction: datetime
    context_tags: Set[str]

@dataclass
class CrossSpaceInsight:
    """Insights derived from cross-space analysis"""
    insight_type: str
    description: str
    affected_spaces: List[int]
    confidence: float
    recommendations: List[str]
    evidence: Dict[str, Any]

class CrossSpaceContextAnalyzer:
    """
    Analyzes context and relationships across multiple desktop spaces
    """

    def __init__(self):
        self.space_contexts = {}
        self.workflow_patterns = []
        self.space_transitions = []
        self.activity_history = defaultdict(list)
        self._init_context_patterns()
        logger.info("Cross-Space Context Analyzer initialized")

    def _init_context_patterns(self):
        """Initialize patterns for context detection"""
        self.workflow_patterns_db = {
            'development': {
                'apps': ['code', 'cursor', 'xcode', 'terminal', 'docker'],
                'keywords': ['debug', 'compile', 'test', 'build', 'git'],
                'file_types': ['.py', '.js', '.ts', '.cpp', '.java']
            },
            'research': {
                'apps': ['chrome', 'safari', 'firefox', 'notion', 'obsidian'],
                'keywords': ['search', 'read', 'documentation', 'api', 'tutorial'],
                'file_types': ['.pdf', '.md', '.txt', '.doc']
            },
            'communication': {
                'apps': ['slack', 'discord', 'messages', 'mail', 'teams'],
                'keywords': ['meeting', 'chat', 'email', 'call', 'conference'],
                'file_types': []
            },
            'creative': {
                'apps': ['photoshop', 'illustrator', 'figma', 'sketch'],
                'keywords': ['design', 'mockup', 'prototype', 'ui', 'ux'],
                'file_types': ['.psd', '.ai', '.fig', '.sketch']
            },
            'productivity': {
                'apps': ['notion', 'todoist', 'things', 'calendar'],
                'keywords': ['task', 'todo', 'schedule', 'plan', 'organize'],
                'file_types': []
            }
        }

    def analyze_cross_space_context(
        self,
        spaces_data: Dict[int, Dict],
        screenshots: Dict[int, Any] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive cross-space context analysis
        """
        # Update space contexts
        self._update_space_contexts(spaces_data)

        # Detect workflow patterns
        workflow = self._detect_workflow_pattern(spaces_data)

        # Analyze space relationships
        relationships = self._analyze_space_relationships()

        # Generate insights
        insights = self._generate_cross_space_insights()

        # Build comprehensive context
        return {
            'workflow': workflow,
            'space_contexts': self.space_contexts,
            'relationships': relationships,
            'insights': insights,
            'activity_summary': self._generate_activity_summary(),
            'recommendations': self._generate_recommendations(workflow, insights)
        }

    def _update_space_contexts(self, spaces_data: Dict[int, Dict]):
        """Update context for each space"""
        for space_id, space_info in spaces_data.items():
            context = self._build_space_context(space_id, space_info)
            self.space_contexts[space_id] = context

            # Track activity
            self.activity_history[space_id].append({
                'timestamp': datetime.now(),
                'apps': space_info.get('applications', []),
                'window_count': len(space_info.get('windows', []))
            })

    def _build_space_context(self, space_id: int, space_info: Dict) -> SpaceContext:
        """Build context for a single space"""
        apps = space_info.get('applications', [])
        windows = space_info.get('windows', [])

        # Determine space purpose
        purpose = self._determine_space_purpose(apps, windows)

        # Extract key information
        key_files = self._extract_key_files(windows)
        key_urls = self._extract_key_urls(windows)

        # Calculate activity score
        activity_score = self._calculate_activity_score(space_info)

        # Generate context tags
        context_tags = self._generate_context_tags(apps, windows, key_files, key_urls)

        return SpaceContext(
            space_id=space_id,
            purpose=purpose,
            active_task=self._infer_active_task(windows, apps),
            related_spaces=self._find_related_spaces(space_id, apps),
            key_files=key_files,
            key_urls=key_urls,
            activity_score=activity_score,
            last_interaction=datetime.now(),
            context_tags=context_tags
        )

    def _determine_space_purpose(self, apps: List[str], windows: List[Dict]) -> str:
        """Determine the primary purpose of a space"""
        app_lower = [app.lower() for app in apps]

        scores = {}
        for workflow_type, pattern in self.workflow_patterns_db.items():
            score = 0
            for app in app_lower:
                if any(key in app for key in pattern['apps']):
                    score += 1

            # Check window titles for keywords
            for window in windows:
                title = window.get('kCGWindowName', '').lower()
                if any(keyword in title for keyword in pattern['keywords']):
                    score += 0.5

            scores[workflow_type] = score

        if scores:
            return max(scores, key=scores.get)
        return 'general'

    def _extract_key_files(self, windows: List[Dict]) -> List[str]:
        """Extract key file names from window titles"""
        files = []
        file_extensions = ['.py', '.js', '.ts', '.md', '.txt', '.pdf', '.doc']

        for window in windows:
            title = window.get('kCGWindowName', '')
            for ext in file_extensions:
                if ext in title:
                    # Extract filename
                    parts = title.split('/')
                    if parts:
                        filename = parts[-1].split(' ')[0]
                        if filename not in files:
                            files.append(filename)

        return files[:10]  # Limit to top 10

    def _extract_key_urls(self, windows: List[Dict]) -> List[str]:
        """Extract URLs from browser windows"""
        urls = []
        browser_apps = ['chrome', 'safari', 'firefox', 'edge']

        for window in windows:
            app = window.get('kCGWindowOwnerName', '').lower()
            if any(browser in app for browser in browser_apps):
                title = window.get('kCGWindowName', '')
                # Simple heuristic - browser titles often contain domain
                if '.' in title and ' ' in title:
                    # Extract potential domain
                    parts = title.split(' ')
                    for part in parts:
                        if '.' in part and len(part) > 4:
                            urls.append(part)
                            break

        return list(set(urls))[:10]

    def _calculate_activity_score(self, space_info: Dict) -> float:
        """Calculate activity intensity score"""
        window_count = len(space_info.get('windows', []))
        app_count = len(space_info.get('applications', []))

        # Normalize scores
        window_score = min(window_count / 10, 1.0)
        app_score = min(app_count / 5, 1.0)

        return (window_score + app_score) / 2

    def _generate_context_tags(
        self,
        apps: List[str],
        windows: List[Dict],
        files: List[str],
        urls: List[str]
    ) -> Set[str]:
        """Generate contextual tags for the space"""
        tags = set()

        # App-based tags
        app_lower = [app.lower() for app in apps]
        if any('code' in app or 'cursor' in app for app in app_lower):
            tags.add('coding')
        if any('chrome' in app or 'safari' in app for app in app_lower):
            tags.add('browsing')
        if any('terminal' in app for app in app_lower):
            tags.add('command-line')
        if any('slack' in app or 'discord' in app for app in app_lower):
            tags.add('communication')

        # File-based tags
        if any('.py' in f for f in files):
            tags.add('python')
        if any('.js' in f or '.ts' in f for f in files):
            tags.add('javascript')
        if any('.md' in f for f in files):
            tags.add('documentation')

        # URL-based tags
        if any('github' in url for url in urls):
            tags.add('github')
        if any('stackoverflow' in url for url in urls):
            tags.add('debugging')

        return tags

    def _infer_active_task(self, windows: List[Dict], apps: List[str]) -> str:
        """Infer the active task from window titles and apps"""
        # Look for common patterns in window titles
        for window in windows:
            title = window.get('kCGWindowName', '').lower()

            if 'jarvis' in title:
                return 'Working on JARVIS AI system'
            elif '.py' in title:
                return f'Python development: {title.split("/")[-1] if "/" in title else title}'
            elif 'terminal' in title or 'iterm' in title:
                return 'Command line operations'
            elif any(browser in title for browser in ['chrome', 'safari', 'firefox']):
                return 'Web browsing/research'

        # Fallback to app-based inference
        if apps:
            return f'Using {apps[0]}'

        return 'General workspace activity'

    def _find_related_spaces(self, space_id: int, apps: List[str]) -> List[int]:
        """Find spaces related to the current one"""
        related = []

        for other_id, context in self.space_contexts.items():
            if other_id != space_id:
                # Check for shared applications
                if context and hasattr(context, 'key_applications'):
                    shared_apps = set(apps) & set(context.key_applications)
                    if shared_apps:
                        related.append(other_id)

        return related

    def _detect_workflow_pattern(self, spaces_data: Dict[int, Dict]) -> WorkflowContext:
        """Detect the overall workflow pattern across spaces"""
        all_apps = []
        space_purposes = {}

        for space_id, space_info in spaces_data.items():
            all_apps.extend(space_info.get('applications', []))
            if space_id in self.space_contexts:
                space_purposes[space_id] = self.space_contexts[space_id].purpose

        # Determine primary workflow type
        workflow_scores = defaultdict(int)
        for purpose in space_purposes.values():
            workflow_scores[purpose] += 1

        primary_workflow = max(workflow_scores, key=workflow_scores.get) if workflow_scores else 'general'

        # Calculate intensity
        total_windows = sum(len(s.get('windows', [])) for s in spaces_data.values())
        intensity = min(total_windows / 30, 1.0)  # Normalize to 0-1

        return WorkflowContext(
            primary_task=self._describe_workflow(primary_workflow, all_apps),
            related_spaces=list(spaces_data.keys()),
            key_applications=list(set(all_apps))[:10],
            workflow_type=primary_workflow,
            intensity_level=intensity,
            time_spent=timedelta(minutes=30),  # Placeholder
            transitions=self.space_transitions[-10:]  # Last 10 transitions
        )

    def _describe_workflow(self, workflow_type: str, apps: List[str]) -> str:
        """Generate human-readable workflow description"""
        descriptions = {
            'development': 'Software development and coding',
            'research': 'Research and information gathering',
            'communication': 'Team collaboration and communication',
            'creative': 'Creative design work',
            'productivity': 'Task management and planning',
            'general': 'Mixed workspace activities'
        }

        base = descriptions.get(workflow_type, 'General work')

        # Add specific app context
        if 'jarvis' in ' '.join(apps).lower():
            base += ' (JARVIS AI development)'
        elif 'code' in ' '.join(apps).lower() or 'cursor' in ' '.join(apps).lower():
            base += ' (active coding session)'

        return base

    def _analyze_space_relationships(self) -> Dict[str, Any]:
        """Analyze relationships between spaces"""
        relationships = {
            'clusters': [],
            'dependencies': [],
            'workflow_flow': []
        }

        # Find clusters of related spaces
        clustered = set()
        for space_id, context in self.space_contexts.items():
            if space_id not in clustered:
                cluster = [space_id]
                for related_id in context.related_spaces:
                    if related_id not in clustered:
                        cluster.append(related_id)
                        clustered.add(related_id)

                if len(cluster) > 1:
                    relationships['clusters'].append({
                        'spaces': cluster,
                        'common_purpose': context.purpose
                    })

        # Detect workflow flow
        if self.space_transitions:
            flow = self._detect_workflow_flow()
            relationships['workflow_flow'] = flow

        return relationships

    def _detect_workflow_flow(self) -> List[Dict]:
        """Detect the flow of work across spaces"""
        flow = []
        transition_counts = defaultdict(int)

        for from_space, to_space, _ in self.space_transitions:
            transition_counts[(from_space, to_space)] += 1

        # Find common transitions
        for (from_space, to_space), count in transition_counts.items():
            if count > 2:  # Threshold for significant flow
                from_context = self.space_contexts.get(from_space)
                to_context = self.space_contexts.get(to_space)

                if from_context and to_context:
                    flow.append({
                        'from': from_space,
                        'to': to_space,
                        'from_purpose': from_context.purpose,
                        'to_purpose': to_context.purpose,
                        'frequency': count
                    })

        return flow

    def _generate_cross_space_insights(self) -> List[CrossSpaceInsight]:
        """Generate insights from cross-space analysis"""
        insights = []

        # Check for distributed workflow
        if len(self.space_contexts) > 2:
            purposes = [c.purpose for c in self.space_contexts.values()]
            if 'development' in purposes and 'research' in purposes:
                insights.append(CrossSpaceInsight(
                    insight_type='distributed_workflow',
                    description='You have a distributed development workflow across multiple spaces',
                    affected_spaces=list(self.space_contexts.keys()),
                    confidence=0.8,
                    recommendations=[
                        'Consider consolidating related tasks',
                        'Use keyboard shortcuts to switch between spaces efficiently'
                    ],
                    evidence={'purposes': purposes}
                ))

        # Check for high activity
        high_activity_spaces = [
            sid for sid, ctx in self.space_contexts.items()
            if ctx.activity_score > 0.7
        ]

        if len(high_activity_spaces) > 1:
            insights.append(CrossSpaceInsight(
                insight_type='high_activity',
                description='Multiple spaces show high activity levels',
                affected_spaces=high_activity_spaces,
                confidence=0.9,
                recommendations=[
                    'Consider taking a break to maintain productivity',
                    'Review if all open applications are necessary'
                ],
                evidence={'spaces': high_activity_spaces}
            ))

        # Check for context switching
        if len(self.space_transitions) > 20:
            insights.append(CrossSpaceInsight(
                insight_type='frequent_switching',
                description='Frequent switching between spaces detected',
                affected_spaces=list(set(t[0] for t in self.space_transitions)),
                confidence=0.7,
                recommendations=[
                    'Consider batching similar tasks',
                    'Use split view for related applications'
                ],
                evidence={'transition_count': len(self.space_transitions)}
            ))

        return insights

    def _generate_activity_summary(self) -> Dict[str, Any]:
        """Generate summary of activity across all spaces"""
        total_apps = set()
        total_windows = 0
        active_spaces = 0

        for space_id, context in self.space_contexts.items():
            if context.activity_score > 0.1:
                active_spaces += 1

            # Count from activity history
            if space_id in self.activity_history:
                latest = self.activity_history[space_id][-1] if self.activity_history[space_id] else {}
                total_apps.update(latest.get('apps', []))
                total_windows += latest.get('window_count', 0)

        return {
            'active_spaces': active_spaces,
            'total_spaces': len(self.space_contexts),
            'unique_applications': len(total_apps),
            'total_windows': total_windows,
            'primary_activity': self._get_primary_activity(),
            'intensity': self._calculate_overall_intensity()
        }

    def _get_primary_activity(self) -> str:
        """Determine the primary activity across all spaces"""
        purpose_counts = defaultdict(int)
        for context in self.space_contexts.values():
            purpose_counts[context.purpose] += 1

        if purpose_counts:
            return max(purpose_counts, key=purpose_counts.get)
        return 'general'

    def _calculate_overall_intensity(self) -> float:
        """Calculate overall activity intensity"""
        if not self.space_contexts:
            return 0.0

        scores = [ctx.activity_score for ctx in self.space_contexts.values()]
        return sum(scores) / len(scores)

    def _generate_recommendations(
        self,
        workflow: WorkflowContext,
        insights: List[CrossSpaceInsight]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Workflow-based recommendations
        if workflow.intensity_level > 0.8:
            recommendations.append("High activity detected - consider taking a short break")

        if workflow.workflow_type == 'development':
            recommendations.append("Development workflow active - ensure version control is up to date")

        # Insight-based recommendations
        for insight in insights:
            recommendations.extend(insight.recommendations)

        # Space-based recommendations
        if len(self.space_contexts) > 5:
            recommendations.append("Many spaces active - consider closing unused spaces")

        return list(set(recommendations))[:5]  # Limit to 5 unique recommendations

    def track_space_transition(self, from_space: int, to_space: int):
        """Track transition between spaces"""
        self.space_transitions.append((from_space, to_space, datetime.now()))

        # Keep only recent transitions
        if len(self.space_transitions) > 100:
            self.space_transitions = self.space_transitions[-100:]