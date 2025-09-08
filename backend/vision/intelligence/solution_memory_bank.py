"""
Solution Memory Bank - Intelligent Solution Storage and Retrieval System

This module implements a sophisticated solution memory system that learns from
past problem resolutions and applies them to new situations. It uses a 
multi-language architecture for optimal performance and flexibility.

Memory Allocation: 100MB
- Solution database: 60MB
- Index structures: 20MB
- Application engine: 20MB
"""

import os
import json
import time
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import pickle
from pathlib import Path
import logging
from collections import defaultdict, Counter
import uuid
import re

# ML imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import DBSCAN
    import faiss  # For efficient similarity search
    HAS_ML = True
except ImportError:
    HAS_ML = False
    logging.warning("ML libraries not available. Using basic matching.")

# Import Rust acceleration if available
try:
    from ..jarvis_rust_core import PySolutionMatcher, PySimilarityIndex
    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    logging.info("Rust acceleration not available. Using Python implementation.")

logger = logging.getLogger(__name__)


class ProblemType(Enum):
    """Types of problems that can be solved"""
    ERROR = "error"
    PERFORMANCE = "performance"
    USABILITY = "usability"
    CONFIGURATION = "configuration"
    WORKFLOW = "workflow"
    INTEGRATION = "integration"
    UNKNOWN = "unknown"


class SolutionStatus(Enum):
    """Status of a solution"""
    VALIDATED = "validated"
    EXPERIMENTAL = "experimental"
    DEPRECATED = "deprecated"
    FAILED = "failed"


@dataclass
class ProblemSignature:
    """Signature identifying a specific problem"""
    signature_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    visual_pattern: Dict[str, Any] = field(default_factory=dict)  # Screenshot features
    error_messages: List[str] = field(default_factory=list)
    context_state: Dict[str, Any] = field(default_factory=dict)  # App state, env, etc
    symptoms: List[str] = field(default_factory=list)
    problem_type: ProblemType = ProblemType.UNKNOWN
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for similarity matching"""
        # Combine all text features
        text = " ".join([
            " ".join(self.error_messages),
            " ".join(self.symptoms),
            json.dumps(self.visual_pattern),
            json.dumps(self.context_state)
        ])
        
        # Simple hash-based vector (fallback if no ML)
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        # Convert to 256-dimensional vector
        return np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32) / 255.0
    
    def generate_hash(self) -> str:
        """Generate a hash for quick lookup"""
        key_parts = [
            str(self.problem_type.value),
            ";".join(sorted(self.error_messages)),
            ";".join(sorted(self.symptoms))
        ]
        return hashlib.md5(":".join(key_parts).encode()).hexdigest()


@dataclass
class SolutionStep:
    """Individual step in a solution"""
    action: str
    target: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    wait_condition: Optional[str] = None
    timeout: float = 30.0
    verification: Optional[str] = None


@dataclass
class SolutionDetails:
    """Complete solution information"""
    solution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action_sequence: List[SolutionStep] = field(default_factory=list)
    success_rate: float = 0.0
    average_time: float = 0.0
    side_effects: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    automated: bool = False
    confidence: float = 0.0


@dataclass
class ApplicationContext:
    """Context for applying solutions"""
    applicable_conditions: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    variations: List[Dict[str, Any]] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    app_versions: List[str] = field(default_factory=list)
    os_versions: List[str] = field(default_factory=list)


@dataclass
class LearningMetadata:
    """Tracking solution effectiveness"""
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_used: Optional[datetime] = None
    user_ratings: List[float] = field(default_factory=list)
    refinement_history: List[Dict[str, Any]] = field(default_factory=list)
    feedback_notes: List[str] = field(default_factory=list)


@dataclass
class Solution:
    """Complete solution entry"""
    problem_signature: ProblemSignature
    solution_details: SolutionDetails
    application_context: ApplicationContext
    learning_metadata: LearningMetadata = field(default_factory=LearningMetadata)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: SolutionStatus = SolutionStatus.EXPERIMENTAL
    
    def calculate_effectiveness(self) -> float:
        """Calculate solution effectiveness score"""
        if self.learning_metadata.usage_count == 0:
            return self.solution_details.confidence
        
        success_rate = (self.learning_metadata.success_count / 
                       max(self.learning_metadata.usage_count, 1))
        
        # Factor in user ratings
        avg_rating = (sum(self.learning_metadata.user_ratings) / 
                     len(self.learning_metadata.user_ratings) 
                     if self.learning_metadata.user_ratings else 0.5)
        
        # Combine metrics
        effectiveness = (
            success_rate * 0.6 +
            avg_rating * 0.3 +
            self.solution_details.confidence * 0.1
        )
        
        # Boost for frequently used solutions
        if self.learning_metadata.usage_count > 10:
            effectiveness *= 1.1
        
        return min(effectiveness, 1.0)


class SolutionMemoryBank:
    """Main solution memory and retrieval system"""
    
    def __init__(self, memory_allocation: Dict[str, int] = None):
        self.memory_allocation = memory_allocation or {
            'solution_database': 60 * 1024 * 1024,  # 60MB
            'index_structures': 20 * 1024 * 1024,   # 20MB
            'application_engine': 20 * 1024 * 1024  # 20MB
        }
        
        # Storage
        self.solutions: Dict[str, Solution] = {}
        self.problem_index: Dict[str, List[str]] = defaultdict(list)  # hash -> solution_ids
        self.type_index: Dict[ProblemType, List[str]] = defaultdict(list)
        
        # ML components
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.solution_vectors: Optional[np.ndarray] = None
        self.faiss_index: Optional[Any] = None
        
        if HAS_ML:
            self.vectorizer = TfidfVectorizer(max_features=1000)
            self.faiss_index = faiss.IndexFlatL2(256)  # 256-dim vectors
        
        # Rust acceleration
        self.rust_matcher: Optional[Any] = None
        self.rust_index: Optional[Any] = None
        
        if HAS_RUST:
            self.rust_matcher = PySolutionMatcher()
            self.rust_index = PySimilarityIndex(256)
        
        # Configuration
        self.min_confidence = float(os.getenv('SOLUTION_MIN_CONFIDENCE', '0.6'))
        self.max_solutions = int(os.getenv('SOLUTION_MAX_RESULTS', '5'))
        self.auto_apply_threshold = float(os.getenv('SOLUTION_AUTO_APPLY_THRESHOLD', '0.9'))
        
        # Persistence
        self.storage_path = Path(os.getenv('SOLUTION_STORAGE_PATH', 
                                          './vision_solutions.db'))
        self._load_solutions()
        
        # Statistics
        self.stats = {
            'captures': 0,
            'applications': 0,
            'successes': 0,
            'failures': 0,
            'refinements': 0
        }
    
    async def capture_solution(self, problem: ProblemSignature, 
                             solution_steps: List[Dict[str, Any]],
                             success: bool,
                             execution_time: float,
                             context: Dict[str, Any] = None) -> Solution:
        """Capture a new solution after problem resolution"""
        # Convert steps
        steps = []
        for step_data in solution_steps:
            step = SolutionStep(
                action=step_data.get('action', 'unknown'),
                target=step_data.get('target'),
                parameters=step_data.get('parameters', {}),
                wait_condition=step_data.get('wait_condition'),
                timeout=step_data.get('timeout', 30.0),
                verification=step_data.get('verification')
            )
            steps.append(step)
        
        # Create solution details
        details = SolutionDetails(
            action_sequence=steps,
            success_rate=1.0 if success else 0.0,
            average_time=execution_time,
            confidence=0.8 if success else 0.3
        )
        
        # Extract context
        app_context = ApplicationContext()
        if context:
            app_context.applicable_conditions = context.get('conditions', [])
            app_context.prerequisites = context.get('prerequisites', [])
            app_context.app_versions = [context.get('app_version', 'unknown')]
            app_context.os_versions = [context.get('os_version', 'unknown')]
        
        # Create solution
        solution = Solution(
            problem_signature=problem,
            solution_details=details,
            application_context=app_context,
            status=SolutionStatus.VALIDATED if success else SolutionStatus.EXPERIMENTAL
        )
        
        # Store solution
        await self.store_solution(solution)
        
        self.stats['captures'] += 1
        logger.info(f"Captured {'successful' if success else 'experimental'} solution {solution.solution_details.solution_id}")
        
        return solution
    
    async def store_solution(self, solution: Solution):
        """Store solution with indexing"""
        solution_id = solution.solution_details.solution_id
        
        # Check for similar existing solutions
        similar = await self.find_similar_solutions(solution.problem_signature, threshold=0.9)
        
        if similar:
            # Merge with existing solution
            existing_id = similar[0][0]
            existing = self.solutions[existing_id]
            await self._merge_solutions(existing, solution)
            logger.info(f"Merged solution with existing {existing_id}")
            return
        
        # Store new solution
        self.solutions[solution_id] = solution
        
        # Update indices
        problem_hash = solution.problem_signature.generate_hash()
        self.problem_index[problem_hash].append(solution_id)
        self.type_index[solution.problem_signature.problem_type].append(solution_id)
        
        # Update vector index
        if HAS_ML or HAS_RUST:
            await self._update_vector_index(solution)
        
        # Persist
        await self._save_solutions()
    
    async def find_similar_solutions(self, problem: ProblemSignature, 
                                   threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find solutions for similar problems"""
        results = []
        
        # Quick hash lookup
        problem_hash = problem.generate_hash()
        if problem_hash in self.problem_index:
            for solution_id in self.problem_index[problem_hash]:
                results.append((solution_id, 1.0))
        
        # Vector similarity search
        if (HAS_ML or HAS_RUST) and self.faiss_index and self.faiss_index.ntotal > 0:
            problem_vector = problem.to_vector()
            
            if HAS_RUST and self.rust_index:
                # Use Rust for fast search
                similar_ids = self.rust_index.search(problem_vector, self.max_solutions)
                for sid, score in similar_ids:
                    if score >= threshold:
                        results.append((sid, score))
            else:
                # Use FAISS
                k = min(self.max_solutions, self.faiss_index.ntotal)
                distances, indices = self.faiss_index.search(
                    problem_vector.reshape(1, -1), k
                )
                
                for i, dist in enumerate(distances[0]):
                    similarity = 1.0 / (1.0 + dist)  # Convert distance to similarity
                    if similarity >= threshold:
                        solution_id = list(self.solutions.keys())[indices[0][i]]
                        results.append((solution_id, similarity))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by effectiveness
        filtered_results = []
        for solution_id, similarity in results[:self.max_solutions]:
            solution = self.solutions[solution_id]
            effectiveness = solution.calculate_effectiveness()
            
            if effectiveness >= self.min_confidence:
                filtered_results.append((solution_id, similarity * effectiveness))
        
        return filtered_results
    
    async def apply_solution(self, solution_id: str, 
                           current_context: Dict[str, Any],
                           execute_callback=None) -> Dict[str, Any]:
        """Apply a solution to current problem"""
        if solution_id not in self.solutions:
            return {'success': False, 'error': 'Solution not found'}
        
        solution = self.solutions[solution_id]
        
        # Check prerequisites
        if not await self._check_prerequisites(solution, current_context):
            return {
                'success': False, 
                'error': 'Prerequisites not met',
                'missing': solution.application_context.prerequisites
            }
        
        # Update usage tracking
        solution.learning_metadata.usage_count += 1
        solution.learning_metadata.last_used = datetime.now()
        
        # Execute solution
        start_time = time.time()
        results = []
        success = True
        
        for i, step in enumerate(solution.solution_details.action_sequence):
            try:
                if execute_callback:
                    # Execute through callback
                    step_result = await execute_callback(step, current_context)
                else:
                    # Simulated execution
                    step_result = {
                        'success': True,
                        'action': step.action,
                        'message': f"Executed {step.action}"
                    }
                
                results.append(step_result)
                
                if not step_result.get('success', False):
                    success = False
                    break
                
                # Wait if needed
                if step.wait_condition:
                    await asyncio.sleep(1.0)  # Simplified wait
                
            except Exception as e:
                logger.error(f"Error executing step {i}: {e}")
                success = False
                break
        
        execution_time = time.time() - start_time
        
        # Update learning
        if success:
            solution.learning_metadata.success_count += 1
            self.stats['successes'] += 1
        else:
            solution.learning_metadata.failure_count += 1
            self.stats['failures'] += 1
        
        # Update solution metrics
        solution.solution_details.success_rate = (
            solution.learning_metadata.success_count / 
            solution.learning_metadata.usage_count
        )
        
        self.stats['applications'] += 1
        
        # Persist updates
        await self._save_solutions()
        
        return {
            'success': success,
            'solution_id': solution_id,
            'execution_time': execution_time,
            'steps_completed': len(results),
            'results': results
        }
    
    async def refine_solution(self, solution_id: str, 
                            refinements: Dict[str, Any],
                            user_feedback: Optional[str] = None,
                            rating: Optional[float] = None):
        """Refine an existing solution based on feedback"""
        if solution_id not in self.solutions:
            return
        
        solution = self.solutions[solution_id]
        
        # Record refinement
        refinement_entry = {
            'timestamp': datetime.now().isoformat(),
            'refinements': refinements,
            'previous_success_rate': solution.solution_details.success_rate
        }
        solution.learning_metadata.refinement_history.append(refinement_entry)
        
        # Apply refinements
        if 'steps' in refinements:
            # Update action sequence
            for step_update in refinements['steps']:
                step_index = step_update.get('index')
                if step_index < len(solution.solution_details.action_sequence):
                    step = solution.solution_details.action_sequence[step_index]
                    if 'action' in step_update:
                        step.action = step_update['action']
                    if 'parameters' in step_update:
                        step.parameters.update(step_update['parameters'])
        
        if 'add_steps' in refinements:
            # Add new steps
            for new_step in refinements['add_steps']:
                step = SolutionStep(**new_step)
                solution.solution_details.action_sequence.append(step)
        
        if 'remove_steps' in refinements:
            # Remove steps (in reverse order to maintain indices)
            for index in sorted(refinements['remove_steps'], reverse=True):
                if 0 <= index < len(solution.solution_details.action_sequence):
                    solution.solution_details.action_sequence.pop(index)
        
        # Update metadata
        if user_feedback:
            solution.learning_metadata.feedback_notes.append(user_feedback)
        
        if rating is not None:
            solution.learning_metadata.user_ratings.append(rating)
        
        solution.updated_at = datetime.now()
        
        # Re-evaluate status
        effectiveness = solution.calculate_effectiveness()
        if effectiveness >= 0.8 and solution.learning_metadata.usage_count >= 5:
            solution.status = SolutionStatus.VALIDATED
        
        self.stats['refinements'] += 1
        
        # Persist
        await self._save_solutions()
        
        logger.info(f"Refined solution {solution_id} with {len(refinements)} changes")
    
    async def get_solution_recommendations(self, 
                                         problem: ProblemSignature,
                                         context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get recommended solutions for a problem"""
        # Find similar solutions
        similar_solutions = await self.find_similar_solutions(problem)
        
        recommendations = []
        for solution_id, similarity in similar_solutions:
            solution = self.solutions[solution_id]
            
            # Check context compatibility
            compatibility = await self._calculate_context_compatibility(
                solution, context
            )
            
            # Calculate overall score
            effectiveness = solution.calculate_effectiveness()
            score = (similarity * 0.4 + effectiveness * 0.4 + compatibility * 0.2)
            
            # Prepare recommendation
            rec = {
                'solution_id': solution_id,
                'score': score,
                'similarity': similarity,
                'effectiveness': effectiveness,
                'compatibility': compatibility,
                'usage_count': solution.learning_metadata.usage_count,
                'success_rate': solution.solution_details.success_rate,
                'estimated_time': solution.solution_details.average_time,
                'auto_applicable': score >= self.auto_apply_threshold,
                'steps': len(solution.solution_details.action_sequence),
                'status': solution.status.value
            }
            
            recommendations.append(rec)
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations
    
    async def _merge_solutions(self, existing: Solution, new: Solution):
        """Merge a new solution with existing one"""
        # Update success metrics
        total_usage = (existing.learning_metadata.usage_count + 1)
        existing.solution_details.success_rate = (
            (existing.solution_details.success_rate * existing.learning_metadata.usage_count +
             new.solution_details.success_rate) / total_usage
        )
        
        # Update timing
        existing.solution_details.average_time = (
            (existing.solution_details.average_time * existing.learning_metadata.usage_count +
             new.solution_details.average_time) / total_usage
        )
        
        # Merge variations
        if new.solution_details.action_sequence != existing.solution_details.action_sequence:
            variation = {
                'timestamp': datetime.now().isoformat(),
                'steps': [asdict(step) for step in new.solution_details.action_sequence],
                'context': asdict(new.application_context)
            }
            existing.application_context.variations.append(variation)
        
        # Update metadata
        existing.learning_metadata.usage_count = total_usage
        if new.solution_details.success_rate > 0:
            existing.learning_metadata.success_count += 1
        else:
            existing.learning_metadata.failure_count += 1
        
        existing.updated_at = datetime.now()
    
    async def _check_prerequisites(self, solution: Solution, 
                                 context: Dict[str, Any]) -> bool:
        """Check if prerequisites are met"""
        for prereq in solution.application_context.prerequisites:
            # Simple string matching for now
            if prereq not in str(context):
                return False
        return True
    
    async def _calculate_context_compatibility(self, solution: Solution,
                                             context: Dict[str, Any]) -> float:
        """Calculate how compatible a solution is with current context"""
        score = 1.0
        
        # Check app version
        if 'app_version' in context:
            if context['app_version'] not in solution.application_context.app_versions:
                score *= 0.8  # Slight penalty for different version
        
        # Check OS version
        if 'os_version' in context:
            if context['os_version'] not in solution.application_context.os_versions:
                score *= 0.9
        
        # Check conditions
        matched_conditions = 0
        for condition in solution.application_context.applicable_conditions:
            if condition in str(context):
                matched_conditions += 1
        
        if solution.application_context.applicable_conditions:
            condition_score = (matched_conditions / 
                             len(solution.application_context.applicable_conditions))
            score *= (0.5 + 0.5 * condition_score)
        
        return score
    
    async def _update_vector_index(self, solution: Solution):
        """Update vector search index"""
        vector = solution.problem_signature.to_vector()
        
        if HAS_RUST and self.rust_index:
            self.rust_index.add_vector(
                solution.solution_details.solution_id, 
                vector
            )
        elif self.faiss_index:
            self.faiss_index.add(vector.reshape(1, -1))
    
    def _load_solutions(self):
        """Load solutions from storage"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'rb') as f:
                    data = pickle.load(f)
                    self.solutions = data.get('solutions', {})
                    self.problem_index = defaultdict(list, data.get('problem_index', {}))
                    self.type_index = defaultdict(list, data.get('type_index', {}))
                    self.stats = data.get('stats', self.stats)
                    
                logger.info(f"Loaded {len(self.solutions)} solutions from storage")
                
                # Rebuild vector index
                if HAS_ML or HAS_RUST:
                    asyncio.create_task(self._rebuild_vector_index())
                    
            except Exception as e:
                logger.error(f"Error loading solutions: {e}")
    
    async def _save_solutions(self):
        """Persist solutions to storage"""
        try:
            # Ensure directory exists
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'solutions': self.solutions,
                'problem_index': dict(self.problem_index),
                'type_index': dict(self.type_index),
                'stats': self.stats,
                'version': '1.0'
            }
            
            with open(self.storage_path, 'wb') as f:
                pickle.dump(data, f)
                
        except Exception as e:
            logger.error(f"Error saving solutions: {e}")
    
    async def _rebuild_vector_index(self):
        """Rebuild the vector search index"""
        if not (HAS_ML or HAS_RUST):
            return
        
        vectors = []
        for solution in self.solutions.values():
            vector = solution.problem_signature.to_vector()
            vectors.append(vector)
        
        if vectors:
            vectors_array = np.array(vectors)
            
            if HAS_RUST and self.rust_index:
                for i, (sol_id, solution) in enumerate(self.solutions.items()):
                    self.rust_index.add_vector(sol_id, vectors_array[i])
            elif self.faiss_index:
                self.faiss_index.add(vectors_array)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory bank statistics"""
        stats = self.stats.copy()
        stats.update({
            'total_solutions': len(self.solutions),
            'by_type': {
                ptype.value: len(solutions) 
                for ptype, solutions in self.type_index.items()
            },
            'validated_solutions': sum(
                1 for s in self.solutions.values() 
                if s.status == SolutionStatus.VALIDATED
            ),
            'average_effectiveness': np.mean([
                s.calculate_effectiveness() 
                for s in self.solutions.values()
            ]) if self.solutions else 0
        })
        return stats
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get current memory usage"""
        # Estimate memory usage
        solution_size = len(pickle.dumps(self.solutions))
        index_size = len(pickle.dumps(dict(self.problem_index))) + \
                    len(pickle.dumps(dict(self.type_index)))
        
        vector_size = 0
        if self.faiss_index and hasattr(self.faiss_index, 'ntotal'):
            vector_size = self.faiss_index.ntotal * 256 * 4  # 256 dims, float32
        
        return {
            'solution_database': solution_size,
            'index_structures': index_size + vector_size,
            'application_engine': 1024 * 1024,  # Rough estimate
            'total': solution_size + index_size + vector_size + 1024 * 1024
        }


# Singleton instance
_solution_memory_bank = None

def get_solution_memory_bank(memory_allocation: Dict[str, int] = None) -> SolutionMemoryBank:
    """Get or create the solution memory bank instance"""
    global _solution_memory_bank
    if _solution_memory_bank is None:
        _solution_memory_bank = SolutionMemoryBank(memory_allocation)
    return _solution_memory_bank