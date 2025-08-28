#!/usr/bin/env python3
"""
Window Relationship Detection for JARVIS Multi-Window Intelligence
Identifies connections and relationships between windows
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import difflib

from .window_detector import WindowInfo

logger = logging.getLogger(__name__)

@dataclass
class WindowRelationship:
    """Represents a relationship between two windows"""
    window1_id: int
    window2_id: int
    relationship_type: str  # 'ide_documentation', 'ide_terminal', 'browser_reference', etc.
    confidence: float  # 0.0 to 1.0
    evidence: List[str]  # Reasons for the relationship

@dataclass
class WindowGroup:
    """A group of related windows"""
    group_id: str
    windows: List[WindowInfo]
    group_type: str  # 'project', 'communication', 'research', etc.
    confidence: float
    common_elements: List[str]

class WindowRelationshipDetector:
    """Detects relationships between windows"""
    
    def __init__(self):
        # Common IDE and editor applications
        self.ide_apps = {
            'Visual Studio Code', 'Cursor', 'Xcode', 'IntelliJ IDEA', 
            'PyCharm', 'WebStorm', 'Sublime Text', 'Atom', 'TextMate'
        }
        
        # Documentation and reference apps
        self.doc_apps = {
            'Chrome', 'Safari', 'Firefox', 'Preview', 'Books', 
            'Dash', 'DevDocs', 'Notion', 'Obsidian'
        }
        
        # Terminal applications
        self.terminal_apps = {
            'Terminal', 'iTerm', 'Alacritty', 'Hyper', 'Warp'
        }
        
        # Communication apps
        self.comm_apps = {
            'Discord', 'Slack', 'Messages', 'Mail', 'Telegram', 
            'WhatsApp', 'Signal', 'Teams'
        }
        
        # Common project indicators
        self.project_indicators = [
            r'\.git', r'package\.json', r'requirements\.txt', r'Cargo\.toml',
            r'pom\.xml', r'build\.gradle', r'\.xcodeproj', r'\.sln'
        ]
        
        # Common documentation domains
        self.doc_domains = [
            'stackoverflow.com', 'github.com', 'docs.', 'developer.',
            'api.', 'reference.', 'tutorial.', 'guide.', 'npm',
            'pypi.org', 'crates.io', 'rubygems.org'
        ]
    
    def detect_relationships(self, windows: List[WindowInfo]) -> List[WindowRelationship]:
        """Detect all relationships between windows"""
        relationships = []
        
        # Check each pair of windows
        for i, window1 in enumerate(windows):
            for j, window2 in enumerate(windows[i+1:], i+1):
                relationship = self._analyze_window_pair(window1, window2)
                if relationship and relationship.confidence >= 0.5:
                    relationships.append(relationship)
        
        return relationships
    
    def group_windows(self, windows: List[WindowInfo], 
                     relationships: List[WindowRelationship]) -> List[WindowGroup]:
        """Group windows by project or task"""
        groups = []
        
        # Build adjacency list from relationships
        window_graph = defaultdict(list)
        for rel in relationships:
            if rel.confidence >= 0.6:  # Only strong relationships
                window_graph[rel.window1_id].append(rel.window2_id)
                window_graph[rel.window2_id].append(rel.window1_id)
        
        # Find connected components (groups)
        visited = set()
        for window in windows:
            if window.window_id not in visited:
                group = self._find_connected_windows(
                    window.window_id, window_graph, windows, visited
                )
                if len(group) > 1:  # Only groups with 2+ windows
                    window_group = self._analyze_group(group)
                    if window_group:
                        groups.append(window_group)
        
        return groups
    
    def _analyze_window_pair(self, window1: WindowInfo, 
                           window2: WindowInfo) -> Optional[WindowRelationship]:
        """Analyze relationship between two windows"""
        evidence = []
        relationship_type = None
        confidence = 0.0
        
        # Check IDE + Documentation relationship
        if (self._is_ide(window1) and self._is_documentation(window2)) or \
           (self._is_ide(window2) and self._is_documentation(window1)):
            ide_window = window1 if self._is_ide(window1) else window2
            doc_window = window2 if self._is_documentation(window2) else window1
            
            # Check for project name match
            project_match = self._find_common_project(ide_window, doc_window)
            if project_match:
                evidence.append(f"Common project: {project_match}")
                confidence += 0.4
            
            # Check for language/framework match
            lang_match = self._find_common_language(ide_window, doc_window)
            if lang_match:
                evidence.append(f"Common language: {lang_match}")
                confidence += 0.3
            
            # Check if documentation is relevant
            if self._is_relevant_documentation(doc_window):
                evidence.append("Technical documentation")
                confidence += 0.2
            
            if confidence > 0:
                relationship_type = "ide_documentation"
                confidence = min(confidence + 0.2, 1.0)  # Base confidence boost
        
        # Check IDE + Terminal relationship
        elif (self._is_ide(window1) and self._is_terminal(window2)) or \
             (self._is_ide(window2) and self._is_terminal(window1)):
            ide_window = window1 if self._is_ide(window1) else window2
            terminal_window = window2 if self._is_terminal(window2) else window1
            
            # Check for project path match
            project_match = self._find_common_project(ide_window, terminal_window)
            if project_match:
                evidence.append(f"Same project: {project_match}")
                confidence += 0.5
            
            # Check for common commands/files
            if self._has_related_content(ide_window, terminal_window):
                evidence.append("Related commands/files")
                confidence += 0.3
            
            # If same user and both are development tools, likely related
            if not project_match and not evidence:
                # Give base confidence for IDE + Terminal on same machine
                evidence.append("Development environment pair")
                confidence = 0.6
            
            if confidence > 0:
                relationship_type = "ide_terminal"
                confidence = min(confidence + 0.1, 0.95)
        
        # Check Browser + Browser relationship (multiple tabs for same topic)
        elif self._is_documentation(window1) and self._is_documentation(window2):
            # Check for similar titles or domains
            similarity = self._calculate_title_similarity(window1, window2)
            if similarity > 0.6:
                evidence.append(f"Similar content (similarity: {similarity:.0%})")
                confidence = similarity
                relationship_type = "related_documentation"
        
        # Check Communication app relationships
        elif self._is_communication(window1) and self._is_communication(window2):
            evidence.append("Multiple communication channels")
            confidence = 0.7
            relationship_type = "communication_group"
        
        # Create relationship if found
        if relationship_type and confidence >= 0.5:
            return WindowRelationship(
                window1_id=window1.window_id,
                window2_id=window2.window_id,
                relationship_type=relationship_type,
                confidence=confidence,
                evidence=evidence
            )
        
        return None
    
    def _is_ide(self, window: WindowInfo) -> bool:
        """Check if window is an IDE or code editor"""
        return any(ide in window.app_name for ide in self.ide_apps)
    
    def _is_documentation(self, window: WindowInfo) -> bool:
        """Check if window is documentation or reference"""
        if any(app in window.app_name for app in self.doc_apps):
            # Additional check for browser windows
            if window.window_title:
                title_lower = window.window_title.lower()
                # Check for documentation indicators
                doc_keywords = ['docs', 'documentation', 'api', 'reference', 
                              'guide', 'tutorial', 'stackoverflow', 'github']
                return any(keyword in title_lower for keyword in doc_keywords)
            return True
        return False
    
    def _is_terminal(self, window: WindowInfo) -> bool:
        """Check if window is a terminal"""
        return any(term in window.app_name for term in self.terminal_apps)
    
    def _is_communication(self, window: WindowInfo) -> bool:
        """Check if window is a communication app"""
        return any(app in window.app_name for app in self.comm_apps)
    
    def _find_common_project(self, window1: WindowInfo, 
                           window2: WindowInfo) -> Optional[str]:
        """Find common project name between windows"""
        if not window1.window_title or not window2.window_title:
            return None
        
        # Extract potential project names
        title1_parts = re.findall(r'[\w\-]+', window1.window_title)
        title2_parts = re.findall(r'[\w\-]+', window2.window_title)
        
        # Special handling for common project patterns
        # Look for project names in paths or after dashes
        for title in [window1.window_title, window2.window_title]:
            # Pattern: "something — ProjectName"
            if ' — ' in title:
                parts = title.split(' — ')
                if len(parts) > 1:
                    project_candidate = parts[-1].strip()
                    # Check if this appears in the other window
                    other_title = window2.window_title if title == window1.window_title else window1.window_title
                    if project_candidate in other_title:
                        return project_candidate
        
        # Find common parts
        common_parts = set(title1_parts) & set(title2_parts)
        
        # Filter out common words
        common_words = {'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 
                       'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
                       'tree', 'working', 'file', 'edit', 'view'}
        
        significant_parts = [part for part in common_parts 
                           if len(part) > 3 and part.lower() not in common_words]
        
        # Prioritize hyphenated project names
        hyphenated = [part for part in significant_parts if '-' in part]
        if hyphenated:
            return max(hyphenated, key=len)
        
        if significant_parts:
            # Return the longest common part
            return max(significant_parts, key=len)
        
        return None
    
    def _find_common_language(self, ide_window: WindowInfo, 
                            doc_window: WindowInfo) -> Optional[str]:
        """Find common programming language/framework"""
        if not doc_window.window_title:
            return None
        
        title_lower = doc_window.window_title.lower()
        
        # Common languages and frameworks
        languages = {
            'python': ['python', 'django', 'flask', 'fastapi', 'numpy', 'pandas'],
            'javascript': ['javascript', 'js', 'react', 'vue', 'angular', 'node'],
            'typescript': ['typescript', 'ts'],
            'rust': ['rust', 'cargo', 'crates'],
            'go': ['golang', 'go '],
            'java': ['java', 'spring', 'maven', 'gradle'],
            'swift': ['swift', 'swiftui', 'ios', 'macos'],
            'ruby': ['ruby', 'rails'],
        }
        
        for lang, keywords in languages.items():
            if any(keyword in title_lower for keyword in keywords):
                return lang
        
        return None
    
    def _is_relevant_documentation(self, window: WindowInfo) -> bool:
        """Check if window contains technical documentation"""
        if not window.window_title:
            return False
        
        title_lower = window.window_title.lower()
        
        # Check for documentation domains
        if any(domain in title_lower for domain in self.doc_domains):
            return True
        
        # Check for technical keywords
        tech_keywords = ['api', 'docs', 'reference', 'guide', 'tutorial',
                        'example', 'documentation', 'manual', 'specification']
        
        return any(keyword in title_lower for keyword in tech_keywords)
    
    def _has_related_content(self, window1: WindowInfo, 
                           window2: WindowInfo) -> bool:
        """Check if windows have related content"""
        if not window1.window_title or not window2.window_title:
            return False
        
        # Extract file names or paths
        file_pattern = r'[\w\-]+\.\w+'
        files1 = set(re.findall(file_pattern, window1.window_title))
        files2 = set(re.findall(file_pattern, window2.window_title))
        
        # Check for common files
        return bool(files1 & files2)
    
    def _calculate_title_similarity(self, window1: WindowInfo, 
                                  window2: WindowInfo) -> float:
        """Calculate similarity between window titles"""
        if not window1.window_title or not window2.window_title:
            return 0.0
        
        # Use sequence matcher for similarity
        return difflib.SequenceMatcher(
            None, 
            window1.window_title.lower(), 
            window2.window_title.lower()
        ).ratio()
    
    def _find_connected_windows(self, start_id: int, graph: Dict[int, List[int]], 
                              all_windows: List[WindowInfo], 
                              visited: Set[int]) -> List[WindowInfo]:
        """Find all windows connected to the starting window"""
        connected = []
        stack = [start_id]
        
        while stack:
            window_id = stack.pop()
            if window_id not in visited:
                visited.add(window_id)
                
                # Find the window object
                window = next((w for w in all_windows if w.window_id == window_id), None)
                if window:
                    connected.append(window)
                
                # Add neighbors
                for neighbor_id in graph[window_id]:
                    if neighbor_id not in visited:
                        stack.append(neighbor_id)
        
        return connected
    
    def _analyze_group(self, windows: List[WindowInfo]) -> Optional[WindowGroup]:
        """Analyze a group of windows to determine its type"""
        if not windows:
            return None
        
        # Count window types
        ide_count = sum(1 for w in windows if self._is_ide(w))
        doc_count = sum(1 for w in windows if self._is_documentation(w))
        term_count = sum(1 for w in windows if self._is_terminal(w))
        comm_count = sum(1 for w in windows if self._is_communication(w))
        
        # Determine group type
        group_type = "mixed"
        confidence = 0.5
        
        if ide_count > 0 and (doc_count > 0 or term_count > 0):
            group_type = "project"
            confidence = 0.8
        elif comm_count >= 2:
            group_type = "communication"
            confidence = 0.9
        elif doc_count >= 2:
            group_type = "research"
            confidence = 0.7
        
        # Find common elements
        common_elements = []
        
        # Extract all title words
        all_words = []
        for window in windows:
            if window.window_title:
                words = re.findall(r'\b\w{4,}\b', window.window_title)
                all_words.extend(words)
        
        # Find frequent words
        word_counts = defaultdict(int)
        for word in all_words:
            word_lower = word.lower()
            if word_lower not in {'window', 'file', 'edit', 'view', 'help'}:
                word_counts[word_lower] += 1
        
        # Common elements are words that appear in multiple windows
        common_elements = [word for word, count in word_counts.items() 
                         if count >= len(windows) * 0.4]
        
        return WindowGroup(
            group_id=f"{group_type}_{windows[0].window_id}",
            windows=windows,
            group_type=group_type,
            confidence=confidence,
            common_elements=common_elements[:5]  # Top 5 common elements
        )

async def test_relationship_detection():
    """Test window relationship detection"""
    from .window_detector import WindowDetector
    
    print("🔍 Testing Window Relationship Detection")
    print("=" * 50)
    
    detector = WindowDetector()
    relationship_detector = WindowRelationshipDetector()
    
    # Get current windows
    windows = detector.get_all_windows()
    print(f"\n📊 Found {len(windows)} windows")
    
    # Detect relationships
    relationships = relationship_detector.detect_relationships(windows)
    print(f"\n🔗 Found {len(relationships)} relationships:")
    
    for rel in relationships[:10]:  # Show first 10
        window1 = next(w for w in windows if w.window_id == rel.window1_id)
        window2 = next(w for w in windows if w.window_id == rel.window2_id)
        
        print(f"\n   Relationship: {rel.relationship_type}")
        print(f"   Window 1: {window1.app_name} - {window1.window_title}")
        print(f"   Window 2: {window2.app_name} - {window2.window_title}")
        print(f"   Confidence: {rel.confidence:.0%}")
        print(f"   Evidence: {', '.join(rel.evidence)}")
    
    # Group windows
    groups = relationship_detector.group_windows(windows, relationships)
    print(f"\n📁 Found {len(groups)} window groups:")
    
    for group in groups:
        print(f"\n   Group: {group.group_type}")
        print(f"   Windows: {len(group.windows)}")
        print(f"   Apps: {[w.app_name for w in group.windows]}")
        print(f"   Common elements: {', '.join(group.common_elements)}")
        print(f"   Confidence: {group.confidence:.0%}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_relationship_detection())