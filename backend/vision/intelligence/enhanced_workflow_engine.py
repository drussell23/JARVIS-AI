#!/usr/bin/env python3
"""
Enhanced Workflow Pattern Engine - Dynamic Learning Extension
Adds advanced pattern learning, clustering, and optimization capabilities
"""

import asyncio
import json
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from collections import defaultdict
import networkx as nx

# Import base workflow engine
from .workflow_pattern_engine import (
    WorkflowEvent, PatternType, PatternConfidence, 
    WorkflowPatternEngine, WorkflowPattern
)


class AdvancedPatternClusterer:
    """Advanced clustering for workflow patterns using multiple algorithms"""
    
    def __init__(self, min_cluster_size: int = 3):
        self.min_cluster_size = min_cluster_size
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=10)
        self.clusters = {}
        
    def cluster_sequences(self, sequences: List[List[str]], 
                         method: str = 'hybrid') -> Dict[int, List[int]]:
        """Cluster sequences using specified method"""
        if not sequences:
            return {}
            
        # Convert sequences to feature vectors
        features = self._sequences_to_features(sequences)
        
        if method == 'dbscan':
            return self._cluster_dbscan(features)
        elif method == 'hierarchical':
            return self._cluster_hierarchical(features)
        elif method == 'hybrid':
            return self._cluster_hybrid(features, sequences)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
    
    def _sequences_to_features(self, sequences: List[List[str]]) -> np.ndarray:
        """Convert sequences to numerical feature vectors"""
        # Get unique actions
        unique_actions = set()
        for seq in sequences:
            unique_actions.update(seq)
        
        action_to_idx = {action: idx for idx, action in enumerate(sorted(unique_actions))}
        
        # Create feature matrix
        max_len = max(len(seq) for seq in sequences)
        features = []
        
        for seq in sequences:
            # One-hot encode sequence
            feature_vec = np.zeros((max_len, len(unique_actions)))
            for i, action in enumerate(seq):
                if action in action_to_idx:
                    feature_vec[i, action_to_idx[action]] = 1
            
            # Add temporal features
            temporal_features = [
                len(seq) / max_len,  # Relative length
                np.mean([i / len(seq) for i in range(len(seq))]),  # Average position
                len(set(seq)) / len(seq) if seq else 0,  # Uniqueness ratio
            ]
            
            # Flatten and combine
            flat_features = np.concatenate([feature_vec.flatten(), temporal_features])
            features.append(flat_features)
        
        features = np.array(features)
        
        # Normalize and reduce dimensions
        features = self.scaler.fit_transform(features)
        if features.shape[1] > 10:
            features = self.pca.fit_transform(features)
        
        return features
    
    def _cluster_dbscan(self, features: np.ndarray) -> Dict[int, List[int]]:
        """DBSCAN clustering for density-based pattern discovery"""
        clusterer = DBSCAN(eps=0.5, min_samples=self.min_cluster_size)
        labels = clusterer.fit_predict(features)
        
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            if label != -1:  # Ignore noise
                clusters[label].append(idx)
        
        return dict(clusters)
    
    def _cluster_hierarchical(self, features: np.ndarray) -> Dict[int, List[int]]:
        """Hierarchical clustering for nested pattern discovery"""
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0,
            linkage='ward'
        )
        labels = clusterer.fit_predict(features)
        
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append(idx)
        
        # Filter small clusters
        return {k: v for k, v in clusters.items() if len(v) >= self.min_cluster_size}
    
    def _cluster_hybrid(self, features: np.ndarray, 
                       sequences: List[List[str]]) -> Dict[int, List[int]]:
        """Hybrid clustering combining multiple methods"""
        # Get clusters from both methods
        dbscan_clusters = self._cluster_dbscan(features)
        hier_clusters = self._cluster_hierarchical(features)
        
        # Merge clusters with sequence similarity check
        merged_clusters = {}
        cluster_id = 0
        
        # Start with DBSCAN clusters
        for db_cluster in dbscan_clusters.values():
            merged_clusters[cluster_id] = db_cluster.copy()
            cluster_id += 1
        
        # Add hierarchical clusters that don't overlap much
        for hier_cluster in hier_clusters.values():
            overlap_found = False
            for existing_cluster in merged_clusters.values():
                overlap = len(set(hier_cluster) & set(existing_cluster))
                if overlap > len(hier_cluster) * 0.5:  # >50% overlap
                    overlap_found = True
                    break
            
            if not overlap_found:
                merged_clusters[cluster_id] = hier_cluster
                cluster_id += 1
        
        return merged_clusters


class PatternOptimizer:
    """Optimize discovered patterns for efficiency and effectiveness"""
    
    def __init__(self):
        self.optimization_history = []
        
    def optimize_pattern(self, pattern: WorkflowPattern, 
                        context: Dict[str, Any]) -> WorkflowPattern:
        """Optimize a pattern based on context and historical performance"""
        optimized = pattern.copy()
        
        # Remove redundant steps
        optimized.action_sequence = self._remove_redundancies(pattern.action_sequence)
        
        # Reorder for efficiency
        optimized.action_sequence = self._reorder_for_efficiency(
            optimized.action_sequence, context
        )
        
        # Add parallel opportunities
        optimized.metadata['parallel_groups'] = self._identify_parallel_groups(
            optimized.action_sequence
        )
        
        # Predict optimization impact
        optimized.metadata['optimization_score'] = self._calculate_optimization_score(
            pattern, optimized
        )
        
        self.optimization_history.append({
            'original': pattern.pattern_id,
            'optimized': optimized.pattern_id,
            'timestamp': datetime.now(),
            'improvements': optimized.metadata['optimization_score']
        })
        
        return optimized
    
    def _remove_redundancies(self, sequence: List[str]) -> List[str]:
        """Remove redundant actions from sequence"""
        if not sequence:
            return sequence
        
        optimized = []
        last_action = None
        
        for action in sequence:
            # Skip immediate repetitions
            if action == last_action:
                continue
            
            # Skip known redundant patterns
            if optimized and self._is_redundant_pair(optimized[-1], action):
                continue
            
            optimized.append(action)
            last_action = action
        
        return optimized
    
    def _is_redundant_pair(self, action1: str, action2: str) -> bool:
        """Check if two actions form a redundant pair"""
        redundant_pairs = [
            ('focus_window', 'focus_window'),
            ('save_file', 'save_file'),
            ('undo', 'redo'),
            ('minimize', 'maximize'),
        ]
        
        return (action1, action2) in redundant_pairs or (action2, action1) in redundant_pairs
    
    def _reorder_for_efficiency(self, sequence: List[str], 
                              context: Dict[str, Any]) -> List[str]:
        """Reorder actions for better efficiency"""
        if len(sequence) <= 2:
            return sequence
        
        # Build dependency graph
        graph = self._build_dependency_graph(sequence)
        
        # Topological sort with efficiency heuristics
        try:
            ordered = list(nx.topological_sort(graph))
            
            # Apply efficiency heuristics
            ordered = self._apply_efficiency_heuristics(ordered, context)
            
            return ordered
        except nx.NetworkXError:
            # Cycle detected, return original
            return sequence
    
    def _build_dependency_graph(self, sequence: List[str]) -> nx.DiGraph:
        """Build dependency graph for actions"""
        graph = nx.DiGraph()
        
        # Add nodes
        for i, action in enumerate(sequence):
            graph.add_node(f"{action}_{i}", action=action, position=i)
        
        # Add dependencies
        for i in range(len(sequence) - 1):
            curr_node = f"{sequence[i]}_{i}"
            next_node = f"{sequence[i+1]}_{i+1}"
            
            # Direct sequence dependency
            graph.add_edge(curr_node, next_node, weight=1)
            
            # Add logical dependencies
            if self._has_logical_dependency(sequence[i], sequence[i+1]):
                graph.add_edge(curr_node, next_node, weight=10)
        
        return graph
    
    def _has_logical_dependency(self, action1: str, action2: str) -> bool:
        """Check if action2 logically depends on action1"""
        dependencies = {
            'open_file': ['edit_file', 'save_file', 'close_file'],
            'create_file': ['edit_file', 'save_file'],
            'compile': ['run', 'test'],
            'git_add': ['git_commit'],
            'git_commit': ['git_push'],
        }
        
        return action1 in dependencies and action2 in dependencies.get(action1, [])
    
    def _apply_efficiency_heuristics(self, sequence: List[str], 
                                   context: Dict[str, Any]) -> List[str]:
        """Apply heuristics to improve efficiency"""
        # Group similar actions together
        action_groups = defaultdict(list)
        for action in sequence:
            base_action = action.split('_')[0] if '_' in action else action
            action_groups[base_action].append(action)
        
        # Reconstruct sequence with grouped actions
        optimized = []
        for group in action_groups.values():
            optimized.extend(sorted(group))  # Keep related actions together
        
        return optimized
    
    def _identify_parallel_groups(self, sequence: List[str]) -> List[List[int]]:
        """Identify groups of actions that can be executed in parallel"""
        if len(sequence) <= 1:
            return [[0]]
        
        parallel_groups = []
        current_group = [0]
        
        for i in range(1, len(sequence)):
            # Check if current action can be parallel with previous
            can_parallel = not self._has_logical_dependency(
                sequence[i-1], sequence[i]
            )
            
            if can_parallel:
                current_group.append(i)
            else:
                if current_group:
                    parallel_groups.append(current_group)
                current_group = [i]
        
        if current_group:
            parallel_groups.append(current_group)
        
        return parallel_groups
    
    def _calculate_optimization_score(self, original: WorkflowPattern, 
                                    optimized: WorkflowPattern) -> float:
        """Calculate optimization improvement score"""
        # Length reduction
        length_improvement = 1 - (len(optimized.action_sequence) / 
                                 max(len(original.action_sequence), 1))
        
        # Parallelization opportunity
        parallel_groups = optimized.metadata.get('parallel_groups', [])
        parallelization_score = sum(len(g) - 1 for g in parallel_groups) / \
                               max(len(optimized.action_sequence), 1)
        
        # Combine scores
        return (length_improvement * 0.6 + parallelization_score * 0.4)


class NeuralPatternPredictor(nn.Module):
    """Neural network for pattern prediction and completion"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, 
                 hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        self.output = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, lengths=None):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Output projection
        output = self.output(attn_out)
        
        return output, (hidden, cell)
    
    def predict_next(self, sequence: torch.Tensor, temperature: float = 1.0) -> int:
        """Predict next action in sequence"""
        with torch.no_grad():
            output, _ = self.forward(sequence.unsqueeze(0))
            logits = output[0, -1] / temperature
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).item()


class EnhancedWorkflowEngine(WorkflowPatternEngine):
    """Enhanced workflow engine with advanced capabilities"""
    
    def __init__(self, memory_allocation: Dict[str, int] = None):
        super().__init__(memory_allocation)
        
        # Advanced components
        self.clusterer = AdvancedPatternClusterer()
        self.optimizer = PatternOptimizer()
        self.predictor = None  # Will be initialized when vocabulary is known
        
        # Performance tracking
        self.performance_metrics = defaultdict(lambda: {
            'executions': 0,
            'successes': 0,
            'failures': 0,
            'avg_duration': 0,
            'optimizations_applied': 0
        })
    
    async def learn_patterns_advanced(self, time_window: timedelta = timedelta(hours=24)):
        """Advanced pattern learning with clustering and optimization"""
        # Get recent events
        recent_events = self._get_recent_events(time_window)
        
        if len(recent_events) < 10:
            return []
        
        # Extract sequences
        sequences = self._extract_sequences(recent_events)
        
        # Cluster sequences
        clusters = self.clusterer.cluster_sequences(sequences, method='hybrid')
        
        # Form patterns from clusters
        new_patterns = []
        for cluster_id, sequence_indices in clusters.items():
            cluster_sequences = [sequences[i] for i in sequence_indices]
            pattern = self._form_pattern_from_cluster(cluster_sequences)
            
            # Optimize pattern
            pattern = self.optimizer.optimize_pattern(pattern, {
                'time_window': time_window,
                'cluster_size': len(cluster_sequences)
            })
            
            new_patterns.append(pattern)
        
        # Update pattern database
        for pattern in new_patterns:
            self.patterns[pattern.pattern_id] = pattern
        
        return new_patterns
    
    def _form_pattern_from_cluster(self, sequences: List[List[str]]) -> WorkflowPattern:
        """Form a pattern from a cluster of sequences"""
        # Find consensus sequence
        consensus = self._find_consensus_sequence(sequences)
        
        # Calculate pattern properties
        pattern = WorkflowPattern(
            pattern_id=f"pattern_{datetime.now().timestamp()}",
            pattern_type=self._infer_pattern_type(consensus),
            confidence=self._calculate_cluster_confidence(sequences),
            action_sequence=consensus,
            trigger_conditions=self._extract_trigger_conditions(sequences),
            expected_outcomes=self._extract_expected_outcomes(sequences),
            variations=sequences,
            metadata={
                'cluster_size': len(sequences),
                'formation_method': 'clustering',
                'timestamp': datetime.now()
            }
        )
        
        return pattern
    
    def _find_consensus_sequence(self, sequences: List[List[str]]) -> List[str]:
        """Find consensus sequence from multiple sequences"""
        if not sequences:
            return []
        
        if len(sequences) == 1:
            return sequences[0]
        
        # Use longest common subsequence approach
        consensus = sequences[0]
        for seq in sequences[1:]:
            consensus = self._longest_common_subsequence(consensus, seq)
        
        return consensus
    
    def _longest_common_subsequence(self, seq1: List[str], seq2: List[str]) -> List[str]:
        """Find LCS of two sequences"""
        m, n = len(seq1), len(seq2)
        dp = [[[] for _ in range(n + 1)] for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + [seq1[i-1]]
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1], key=len)
        
        return dp[m][n]
    
    def suggest_automation(self, current_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest workflow automations based on current context"""
        suggestions = []
        
        # Match current context with patterns
        matched_patterns = self._match_context_to_patterns(current_context)
        
        for pattern in matched_patterns:
            # Check if pattern would be beneficial
            benefit_score = self._calculate_automation_benefit(pattern, current_context)
            
            if benefit_score > 0.7:
                suggestions.append({
                    'pattern_id': pattern.pattern_id,
                    'description': self._generate_automation_description(pattern),
                    'benefit_score': benefit_score,
                    'estimated_time_saved': self._estimate_time_saved(pattern),
                    'actions': self._generate_automation_actions(pattern)
                })
        
        # Sort by benefit score
        suggestions.sort(key=lambda x: x['benefit_score'], reverse=True)
        
        return suggestions[:5]  # Top 5 suggestions
    
    def _calculate_automation_benefit(self, pattern: WorkflowPattern, 
                                    context: Dict[str, Any]) -> float:
        """Calculate benefit score for automating a pattern"""
        # Frequency benefit
        frequency_score = min(pattern.frequency / 100, 1.0)
        
        # Complexity benefit (more complex = more benefit)
        complexity_score = min(len(pattern.action_sequence) / 20, 1.0)
        
        # Performance history
        perf = self.performance_metrics[pattern.pattern_id]
        if perf['executions'] > 0:
            success_rate = perf['successes'] / perf['executions']
        else:
            success_rate = 0.5
        
        # Time saving potential
        time_score = min(self._estimate_time_saved(pattern) / 300, 1.0)  # 5 min max
        
        # Combine scores
        benefit = (frequency_score * 0.3 + 
                  complexity_score * 0.2 + 
                  success_rate * 0.3 + 
                  time_score * 0.2)
        
        return benefit
    
    def _estimate_time_saved(self, pattern: WorkflowPattern) -> float:
        """Estimate time saved by automating pattern (in seconds)"""
        # Base estimate: 2 seconds per action
        base_time = len(pattern.action_sequence) * 2
        
        # Add time for context switches
        switches = sum(1 for i in range(1, len(pattern.action_sequence))
                      if self._is_context_switch(pattern.action_sequence[i-1], 
                                               pattern.action_sequence[i]))
        switch_time = switches * 3
        
        # Subtract parallel execution time
        parallel_groups = pattern.metadata.get('parallel_groups', [])
        parallel_savings = sum(max(0, len(g) - 1) * 1.5 for g in parallel_groups)
        
        return max(0, base_time + switch_time - parallel_savings)
    
    def _is_context_switch(self, action1: str, action2: str) -> bool:
        """Check if two actions represent a context switch"""
        # Simple heuristic: different app or major action type change
        app1 = action1.split('_')[0] if '_' in action1 else 'default'
        app2 = action2.split('_')[0] if '_' in action2 else 'default'
        
        return app1 != app2


# Global instance management
_enhanced_engine_instance = None

def get_enhanced_workflow_engine() -> EnhancedWorkflowEngine:
    """Get or create enhanced workflow engine instance"""
    global _enhanced_engine_instance
    if _enhanced_engine_instance is None:
        _enhanced_engine_instance = EnhancedWorkflowEngine()
    return _enhanced_engine_instance