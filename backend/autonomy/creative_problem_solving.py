#!/usr/bin/env python3
"""
Creative Problem Solving Module for JARVIS

This module provides innovative solution generation and workflow optimization capabilities
using advanced AI techniques and creative problem-solving methodologies. It implements
multiple creative approaches including lateral thinking, systematic innovation, analogical
reasoning, and biomimicry to generate novel solutions to complex problems.

The module supports various problem types from workflow optimization to technical challenges,
using intelligent model selection and hybrid solution generation to maximize innovation
while maintaining practical feasibility.

Example:
    >>> from creative_problem_solving import CreativeProblemSolver, Problem, ProblemType
    >>> solver = CreativeProblemSolver(anthropic_api_key="your_key")
    >>> problem = Problem(
    ...     problem_id="workflow_001",
    ...     description="Optimize data processing pipeline",
    ...     problem_type=ProblemType.WORKFLOW_OPTIMIZATION,
    ...     constraints=["Limited memory", "Real-time processing"],
    ...     objectives=["Reduce latency", "Increase throughput"],
    ...     context={"current_latency": "500ms", "target_latency": "100ms"},
    ...     priority=0.8
    ... )
    >>> solutions = await solver.solve_problem(problem)
    >>> print(f"Generated {len(solutions)} innovative solutions")
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import json
import numpy as np
from itertools import combinations, permutations
import anthropic
import hashlib
import random

logger = logging.getLogger(__name__)

class ProblemType(Enum):
    """Types of problems the system can solve.
    
    Defines the various categories of problems that the creative problem solver
    can handle, each requiring different approaches and techniques.
    """
    WORKFLOW_OPTIMIZATION = "workflow_optimization"
    AUTOMATION_DESIGN = "automation_design"
    PRODUCTIVITY_ENHANCEMENT = "productivity_enhancement"
    TECHNICAL_CHALLENGE = "technical_challenge"
    CREATIVE_BLOCK = "creative_block"
    RESOURCE_ALLOCATION = "resource_allocation"
    DECISION_MAKING = "decision_making"
    INNOVATION_OPPORTUNITY = "innovation_opportunity"
    SYSTEM_INTEGRATION = "system_integration"
    USER_EXPERIENCE = "user_experience"

class SolutionApproach(Enum):
    """Creative approaches to problem solving.
    
    Defines the various creative methodologies and techniques that can be
    applied to generate innovative solutions to problems.
    """
    LATERAL_THINKING = "lateral_thinking"
    SYSTEMATIC_INNOVATION = "systematic_innovation"
    ANALOGICAL_REASONING = "analogical_reasoning"
    REVERSE_ENGINEERING = "reverse_engineering"
    COMBINATORIAL_CREATIVITY = "combinatorial_creativity"
    BIOMIMICRY = "biomimicry"
    FIRST_PRINCIPLES = "first_principles"
    DESIGN_THINKING = "design_thinking"
    SYSTEMS_THINKING = "systems_thinking"
    MORPHOLOGICAL_ANALYSIS = "morphological_analysis"

@dataclass
class Problem:
    """Represents a problem to be solved.
    
    Encapsulates all the information needed to understand and solve a problem,
    including context, constraints, objectives, and success criteria.
    
    Attributes:
        problem_id: Unique identifier for the problem
        description: Detailed description of the problem
        problem_type: Category of problem from ProblemType enum
        constraints: List of limitations or restrictions
        objectives: List of goals to achieve
        context: Additional contextual information as key-value pairs
        priority: Priority level from 0.0 to 1.0
        deadline: Optional deadline for solution implementation
        stakeholders: List of people or groups affected by the problem
        success_criteria: List of measurable success indicators
    """
    problem_id: str
    description: str
    problem_type: ProblemType
    constraints: List[str]
    objectives: List[str]
    context: Dict[str, Any]
    priority: float
    deadline: Optional[datetime] = None
    stakeholders: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    
    def to_prompt_context(self) -> str:
        """Convert problem to context for AI prompt.
        
        Creates a formatted string representation of the problem suitable
        for use in AI prompts and analysis.
        
        Returns:
            Formatted string containing all problem details
            
        Example:
            >>> problem = Problem(...)
            >>> context = problem.to_prompt_context()
            >>> print(context)
            Problem: Optimize data processing pipeline
            Type: workflow_optimization
            ...
        """
        return f"""
Problem: {self.description}
Type: {self.problem_type.value}
Objectives: {', '.join(self.objectives)}
Constraints: {', '.join(self.constraints)}
Context: {json.dumps(self.context, indent=2)}
Priority: {self.priority}
"""

@dataclass
class CreativeSolution:
    """A creative solution to a problem.
    
    Represents a complete solution generated through creative problem-solving
    techniques, including implementation details, scoring, and risk assessment.
    
    Attributes:
        solution_id: Unique identifier for the solution
        problem_id: ID of the problem this solution addresses
        approach: Creative approach used to generate the solution
        description: Detailed description of the solution
        implementation_steps: List of steps required to implement the solution
        innovation_score: Score from 0.0 to 1.0 indicating novelty
        feasibility_score: Score from 0.0 to 1.0 indicating practicality
        impact_score: Score from 0.0 to 1.0 indicating potential impact
        resources_required: List of resources needed for implementation
        estimated_time: Estimated time for implementation
        risks: List of potential risks and challenges
        alternatives: List of alternative approaches
        synergies: List of ways this solution connects with existing systems
    """
    solution_id: str
    problem_id: str
    approach: SolutionApproach
    description: str
    implementation_steps: List[Dict[str, Any]]
    innovation_score: float
    feasibility_score: float
    impact_score: float
    resources_required: List[str]
    estimated_time: str
    risks: List[Dict[str, Any]]
    alternatives: List[str]
    synergies: List[str]  # How it connects with existing systems
    
    def get_overall_score(self) -> float:
        """Calculate overall solution quality score.
        
        Computes a weighted average of innovation, feasibility, and impact scores
        to provide an overall quality metric for ranking solutions.
        
        Returns:
            Overall quality score from 0.0 to 1.0
            
        Example:
            >>> solution = CreativeSolution(...)
            >>> score = solution.get_overall_score()
            >>> print(f"Solution quality: {score:.2f}")
        """
        return (self.innovation_score * 0.3 + 
                self.feasibility_score * 0.4 + 
                self.impact_score * 0.3)

@dataclass
class IdeaNode:
    """Node in the idea generation graph.
    
    Represents a concept or idea in the creative problem-solving network,
    with connections to related concepts and strength indicators.
    
    Attributes:
        node_id: Unique identifier for the node
        concept: The core concept or idea
        category: Category or domain of the concept
        connections: List of IDs of connected nodes
        strength: Strength of the concept from 0.0 to 1.0
        metadata: Additional metadata about the concept
    """
    node_id: str
    concept: str
    category: str
    connections: List[str]
    strength: float
    metadata: Dict[str, Any] = field(default_factory=dict) 

class CreativeProblemSolver:
    """Advanced creative problem solving engine using AI and innovative techniques.
    
    This class implements a comprehensive creative problem-solving system that uses
    multiple AI models, creative methodologies, and learning mechanisms to generate
    innovative solutions to complex problems.
    
    The solver supports various creative approaches including lateral thinking,
    systematic innovation, analogical reasoning, biomimicry, and more. It learns
    from implementation outcomes to improve future solution quality.
    
    Attributes:
        claude: Anthropic Claude API client
        use_intelligent_selection: Whether to use intelligent model selection
        active_problems: Dictionary of currently active problems
        solution_history: List of all generated solutions
        idea_graph: Graph of interconnected ideas and concepts
        solution_patterns: Patterns learned from successful solutions
        success_metrics: Metrics tracking solution success rates
        approach_effectiveness: Effectiveness scores for each approach
        creative_techniques: Mapping of approaches to implementation functions
        analogy_database: Database of analogies for creative reasoning
        innovation_metrics: Tracking metrics for innovation performance
        
    Example:
        >>> solver = CreativeProblemSolver(anthropic_api_key="your_key")
        >>> problem = Problem(...)
        >>> solutions = await solver.solve_problem(problem)
        >>> best_solution = solutions[0]
        >>> print(f"Best solution: {best_solution.description}")
    """
    
    def __init__(self, anthropic_api_key: str, use_intelligent_selection: bool = True):
        """Initialize the creative problem solver.
        
        Args:
            anthropic_api_key: API key for Anthropic Claude
            use_intelligent_selection: Whether to use intelligent model selection
            
        Raises:
            ValueError: If API key is invalid or missing
        """
        # Anthropic API Client to work with Claude
        self.claude = anthropic.Anthropic(api_key=anthropic_api_key)
        self.use_intelligent_selection = use_intelligent_selection

        # Problem solving components
        self.active_problems: Dict[str, Problem] = {}
        self.solution_history: List[CreativeSolution] = []
        self.idea_graph: Dict[str, IdeaNode] = {}

        # Learning components
        self.solution_patterns = defaultdict(list)
        self.success_metrics = defaultdict(float)
        self.approach_effectiveness = defaultdict(lambda: 0.5)

        # Creative techniques
        self.creative_techniques = self._initialize_creative_techniques()
        self.analogy_database = self._load_analogy_database()

        # Innovation tracking
        self.innovation_metrics = {
            'solutions_generated': 0,
            'average_innovation_score': 0.0,
            'successful_implementations': 0,
            'time_saved': 0  # in hours
        }
    
    def _initialize_creative_techniques(self) -> Dict[str, Callable]:
        """Initialize creative problem-solving techniques.
        
        Creates a mapping between solution approaches and their corresponding
        implementation functions.
        
        Returns:
            Dictionary mapping SolutionApproach to implementation functions
        """
        return {
            SolutionApproach.LATERAL_THINKING: self._lateral_thinking_approach,
            SolutionApproach.SYSTEMATIC_INNOVATION: self._systematic_innovation_approach,
            SolutionApproach.ANALOGICAL_REASONING: self._analogical_reasoning_approach,
            SolutionApproach.REVERSE_ENGINEERING: self._reverse_engineering_approach,
            SolutionApproach.COMBINATORIAL_CREATIVITY: self._combinatorial_approach,
            SolutionApproach.BIOMIMICRY: self._biomimicry_approach,
            SolutionApproach.FIRST_PRINCIPLES: self._first_principles_approach,
            SolutionApproach.DESIGN_THINKING: self._design_thinking_approach,
            SolutionApproach.SYSTEMS_THINKING: self._systems_thinking_approach,
            SolutionApproach.MORPHOLOGICAL_ANALYSIS: self._morphological_analysis_approach
        }
    
    def _load_analogy_database(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load database of analogies for creative problem solving.
        
        Initializes a comprehensive database of analogies from various domains
        including nature, engineering, and human systems that can be used for
        analogical reasoning in problem solving.
        
        Returns:
            Dictionary with categories as keys and lists of analogy dictionaries as values
        """
        return {
            'nature': [
                {'system': 'ant_colony', 'principles': ['distributed_intelligence', 'pheromone_trails', 'task_specialization']},
                {'system': 'neural_network', 'principles': ['parallel_processing', 'learning', 'adaptation']},
                {'system': 'ecosystem', 'principles': ['balance', 'recycling', 'diversity_strength']},
                {'system': 'swarm', 'principles': ['emergent_behavior', 'simple_rules', 'collective_intelligence']}
            ],
            'engineering': [
                {'system': 'modular_design', 'principles': ['reusability', 'scalability', 'maintainability']},
                {'system': 'feedback_loops', 'principles': ['self_regulation', 'continuous_improvement', 'stability']},
                {'system': 'redundancy', 'principles': ['fault_tolerance', 'reliability', 'backup_systems']}
            ],
            'human_systems': [
                {'system': 'agile_methodology', 'principles': ['iteration', 'flexibility', 'collaboration']},
                {'system': 'assembly_line', 'principles': ['efficiency', 'specialization', 'workflow']},
                {'system': 'democracy', 'principles': ['distributed_decision_making', 'checks_balances', 'representation']}
            ]
        }
    
    async def solve_problem(self, problem: Problem) -> List[CreativeSolution]:
        """Generate creative solutions for a given problem.
        
        This is the main entry point for problem solving. It analyzes the problem,
        applies multiple creative approaches, generates hybrid solutions, and returns
        the top-ranked solutions.
        
        Args:
            problem: The Problem instance to solve
            
        Returns:
            List of CreativeSolution instances, ranked by quality (top 3)
            
        Raises:
            Exception: If problem analysis or solution generation fails
            
        Example:
            >>> problem = Problem(
            ...     problem_id="opt_001",
            ...     description="Reduce API response time",
            ...     problem_type=ProblemType.TECHNICAL_CHALLENGE,
            ...     constraints=["Limited budget", "No downtime"],
            ...     objectives=["50% faster responses", "Maintain reliability"],
            ...     context={"current_time": "200ms", "target_time": "100ms"},
            ...     priority=0.9
            ... )
            >>> solutions = await solver.solve_problem(problem)
            >>> print(f"Generated {len(solutions)} solutions")
        """
        self.active_problems[problem.problem_id] = problem
        solutions = []
        
        # Analyze problem deeply
        problem_analysis = await self._analyze_problem(problem)
        
        # Generate ideas using multiple approaches
        for approach in self._select_approaches(problem, problem_analysis):
            try:
                solution = await self._generate_solution(problem, approach, problem_analysis)
                if solution and solution.feasibility_score > 0.4:
                    solutions.append(solution)
                    self.solution_history.append(solution)
                    self.innovation_metrics['solutions_generated'] += 1
            except Exception as e:
                logger.error(f"Error with approach {approach}: {e}")
        
        # Cross-pollinate solutions
        if len(solutions) > 1:
            hybrid_solutions = await self._create_hybrid_solutions(solutions, problem)
            solutions.extend(hybrid_solutions)
        
        # Rank and refine solutions
        solutions = self._rank_solutions(solutions)
        solutions = await self._refine_top_solutions(solutions[:5], problem)
        
        # Update metrics
        if solutions:
            avg_innovation = sum(s.innovation_score for s in solutions) / len(solutions)
            self.innovation_metrics['average_innovation_score'] = (
                self.innovation_metrics['average_innovation_score'] * 0.9 + avg_innovation * 0.1
            )
        
        return solutions[:3]  # Return top 3 solutions
    
    async def _analyze_problem(self, problem: Problem) -> Dict[str, Any]:
        """Deep analysis of the problem using AI or local analysis.
        
        Performs comprehensive problem analysis to understand root causes,
        assumptions, stakeholder needs, and opportunities for innovation.
        Falls back to local analysis if AI analysis fails or system load is high.
        
        Args:
            problem: The Problem instance to analyze
            
        Returns:
            Dictionary containing analysis results with keys:
            - root_causes: List of underlying causes
            - assumptions: List of current assumptions
            - stakeholder_needs: List of stakeholder requirements
            - interdependencies: List of system interdependencies
            - risks: List of potential risks
            - opportunities: List of improvement opportunities
            - analogous_problems: List of similar problems
            - leverage_points: List of high-impact intervention points
            
        Example:
            >>> analysis = await solver._analyze_problem(problem)
            >>> print(f"Found {len(analysis['root_causes'])} root causes")
        """
        try:
            # LOCAL PROBLEM ANALYSIS - No Claude API calls
            import psutil
            cpu_usage = psutil.cpu_percent(interval=0.1)
            
            if cpu_usage > 25:
                logger.warning(f"CPU usage too high ({cpu_usage}%) - using local analysis")
                return self._local_problem_analysis(problem)
            
            # Use local analysis patterns based on problem type
            if problem.problem_type == ProblemType.WORKFLOW_OPTIMIZATION:
                return {
                    'root_causes': ['Process bottlenecks', 'Resource allocation inefficiency', 'Communication gaps'],
                    'assumptions': ['Current workflow is optimal', 'All steps are necessary', 'Sequential processing required'],
                    'stakeholder_needs': ['Efficiency', 'Predictability', 'Resource optimization'],
                    'interdependencies': ['Process steps', 'Team coordination', 'Resource sharing'],
                    'risks': ['Change resistance', 'Implementation complexity', 'Temporary productivity loss'],
                    'opportunities': ['Automation', 'Parallel processing', 'Standardization'],
                    'analogous_problems': ['Manufacturing optimization', 'Supply chain efficiency', 'Software development workflows'],
                    'leverage_points': ['Bottleneck processes', 'Decision points', 'Resource allocation']
                }
            elif problem.problem_type == ProblemType.TECHNICAL_CHALLENGE:
                return {
                    'root_causes': ['Technical debt', 'Resource constraints', 'Knowledge gaps'],
                    'assumptions': ['Current architecture is optimal', 'All features are necessary', 'Technical approach is correct'],
                    'stakeholder_needs': ['Reliability', 'Performance', 'Maintainability'],
                    'interdependencies': ['System components', 'External dependencies', 'Technical constraints'],
                    'risks': ['System failure', 'Performance degradation', 'Maintenance complexity'],
                    'opportunities': ['Architecture improvement', 'Technology upgrade', 'Process optimization'],
                    'analogous_problems': ['Software refactoring', 'Infrastructure scaling', 'Legacy system modernization'],
                    'leverage_points': ['Core architecture', 'Critical dependencies', 'Performance bottlenecks']
                }
            else:
                return self._generate_default_analysis(problem)
                
        except Exception as e:
            logger.error(f"Error in problem analysis: {e}")
            return self._generate_default_analysis(problem)
    
    def _local_problem_analysis(self, problem: Problem) -> Dict[str, Any]:
        """Local problem analysis without API calls.
        
        Performs intelligent problem analysis using local heuristics and
        pattern matching based on problem attributes and description.
        
        Args:
            problem: The Problem instance to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        # Generate intelligent analysis based on problem attributes
        base_analysis = {
            'root_causes': [],
            'assumptions': [],
            'stakeholder_needs': [],
            'interdependencies': [],
            'risks': [],
            'opportunities': [],
            'analogous_problems': [],
            'leverage_points': []
        }
        
        # Add contextual analysis based on problem description
        if problem.description:
            desc_lower = problem.description.lower()
            
            # Detect performance issues
            if any(word in desc_lower for word in ['slow', 'performance', 'lag', 'delay']):
                base_analysis['root_causes'].append('Performance bottlenecks')
                base_analysis['opportunities'].append('Performance optimization')
                
            # Detect resource issues
            if any(word in desc_lower for word in ['memory', 'cpu', 'resource', 'capacity']):
                base_analysis['root_causes'].append('Resource constraints')
                base_analysis['opportunities'].append('Resource optimization')
                
            # Detect user experience issues
            if any(word in desc_lower for word in ['user', 'experience', 'usability', 'interface']):
                base_analysis['stakeholder_needs'].append('Improved user experience')
                base_analysis['opportunities'].append('UX enhancement')
        
        # Add constraints as risks
        for constraint in problem.constraints:
            base_analysis['risks'].append(f'Constraint: {constraint}')
            
        # Add success criteria as stakeholder needs
        for criteria in problem.success_criteria:
            base_analysis['stakeholder_needs'].append(f'Success: {criteria}')
            
        return base_analysis
    
    def _generate_default_analysis(self, problem: Problem) -> Dict[str, Any]:
        """Generate default analysis when AI analysis fails.
        
        Provides a fallback analysis with generic but useful insights
        when more sophisticated analysis methods are unavailable.
        
        Args:
            problem: The Problem instance to analyze
            
        Returns:
            Dictionary containing default analysis results
        """
        return {
            'root_causes': ['Resource constraints', 'Process inefficiency'],
            'assumptions': ['Current approach is optimal', 'Users need all features'],
            'stakeholder_needs': ['Efficiency', 'Reliability', 'Ease of use'],
            'interdependencies': ['System components', 'User workflows'],
            'risks': ['Implementation complexity', 'User adoption'],
            'opportunities': ['Automation', 'Integration', 'Simplification'],
            'analogous_problems': ['Similar optimization challenges'],
            'leverage_points': ['Key bottlenecks', 'High-impact areas']
        }
    
    def _select_approaches(self, problem: Problem, 
                         analysis: Dict[str, Any]) -> List[SolutionApproach]:
        """Select appropriate creative approaches based on problem type.
        
        Chooses the most effective creative problem-solving approaches based on
        the problem type, analysis results, and historical effectiveness data.
        
        Args:
            problem: The Problem instance
            analysis: Problem analysis results
            
        Returns:
            List of SolutionApproach instances, ordered by expected effectiveness
            
        Example:
            >>> approaches = solver._select_approaches(problem, analysis)
            >>> print(f"Selected {len(approaches)} approaches")
        """
        approaches = []
        
        # Map problem types to effective approaches
        if problem.problem_type == ProblemType.WORKFLOW_OPTIMIZATION:
            approaches.extend([
                SolutionApproach.SYSTEMS_THINKING,
                SolutionApproach.SYSTEMATIC_INNOVATION,
                SolutionApproach.BIOMIMICRY
            ])
        elif problem.problem_type == ProblemType.TECHNICAL_CHALLENGE:
            approaches.extend([
                SolutionApproach.FIRST_PRINCIPLES,
                SolutionApproach.REVERSE_ENGINEERING,
                SolutionApproach.ANALOGICAL_REASONING
            ])
        elif problem.problem_type == ProblemType.CREATIVE_BLOCK:
            approaches.extend([
                SolutionApproach.LATERAL_THINKING,
                SolutionApproach.COMBINATORIAL_CREATIVITY,
                SolutionApproach.MORPHOLOGICAL_ANALYSIS
            ])
        else:
            # Default approaches
            approaches.extend([
                SolutionApproach.DESIGN_THINKING,
                SolutionApproach.SYSTEMATIC_INNOVATION,
                SolutionApproach.ANALOGICAL_REASONING
            ])
        
        # Prioritize based on past effectiveness
        approaches.sort(key=lambda a: self.approach_effectiveness[a], reverse=True)
        
        return approaches[:4]  # Use top 4 approaches
    
    async def _generate_solution(self, problem: Problem, 
                               approach: SolutionApproach,
                               analysis: Dict[str, Any]) -> Optional[CreativeSolution]:
        """Generate solution using specific approach.
        
        Applies a specific creative problem-solving approach to generate
        a solution for the given problem.
        
        Args:
            problem: The Problem instance to solve
            approach: The SolutionApproach to use
            analysis: Problem analysis results
            
        Returns:
            CreativeSolution instance or None if generation fails
            
        Raises:
            Exception: If the approach technique is not available
        """
        technique = self.creative_techniques.get(approach)
        if not technique:
            return None
        
        return await technique(problem, analysis)
    
    async def _lateral_thinking_with_intelligent_selection(self, problem: Problem,
                                                           analysis: Dict[str, Any]) -> Optional[CreativeSolution]:
        """Generate solution using lateral thinking with intelligent model selection.
        
        Uses the hybrid orchestrator for intelligent model selection to apply
        lateral thinking techniques for creative problem solving.
        
        Args:
            problem: The Problem instance to solve
            analysis: Problem analysis results containing assumptions to challenge
            
        Returns:
            CreativeSolution instance or None if generation fails
            
        Raises:
            ImportError: If hybrid orchestrator is not available
            Exception: If intelligent selection fails
        """
        try:
            from backend.core.hybrid_orchestrator import HybridOrchestrator

            orchestrator = HybridOrchestrator()
            if not orchestrator.is_running:
                await orchestrator.start()

            # Challenge assumptions
            assumptions = analysis.get('assumptions', [])

            # Build rich context for intelligent selection
            intelligent_context = {
                "task_type": "creative_thinking",
                "problem_details": {
                    "type": problem.problem_type.value,
                    "description": problem.description,
                    "constraints": problem.constraints,
                    "objectives": problem.objectives
                },
                "creative_approach": "lateral_thinking",
                "innovation_requirements": {
                    "break_conventions": True,
                    "challenge_assumptions": assumptions,
                    "find_unexpected_connections": True
                }
            }

            prompt = f"""Use lateral thinking to solve this problem:

{problem.to_prompt_context()}

Assumptions to challenge: {', '.join(assumptions)}

Apply lateral thinking techniques:
1. Random entry: Start from an unrelated concept
2. Provocation: Make deliberately unreasonable statements
3. Movement: Extract useful ideas from provocations
4. Concept extraction: Find underlying principles
5. Challenge boundaries: Question all constraints

Generate an innovative solution that:
- Breaks conventional thinking
- Finds unexpected connections
- Simplifies radically
- Changes the problem definition if needed

Provide: solution description, implementation steps, innovation reasoning"""

            result = await orchestrator.execute_with_intelligent_model_selection(
                query=prompt,
                intent="creative_thinking",
                required_capabilities={"creative_reasoning", "problem_solving", "innovation"},
                context=intelligent_context,
                max_tokens=1500,
                temperature=0.9,
            )

            if not result.get("success"):
                raise Exception(result.get("error", "Unknown error"))

            solution_text = result.get("text", "").strip()
            model_used = result.get("model_used", "intelligent_selection")

            logger.info(f"✨ Lateral thinking solution generated using {model_used}")

            return self._parse_solution(
                solution_text, problem, SolutionApproach.LATERAL_THINKING
            )

        except ImportError:
            logger.warning("Hybrid orchestrator not available for lateral thinking")
            raise
        except Exception as e:
            logger.error(f"Error in intelligent lateral thinking: {e}")
            raise

    async def _lateral_thinking_approach(self, problem: Problem,
                                       analysis: Dict[str, Any]) -> Optional[CreativeSolution]:
        """Generate solution using lateral thinking.
        
        Applies lateral thinking techniques to break conventional thinking patterns
        and generate innovative solutions by challenging assumptions and exploring
        unexpected connections.
        
        Args:
            problem: The Problem instance to solve
            analysis: Problem analysis results containing assumptions to challenge
            
        Returns:
            CreativeSolution instance or None if generation fails
        """
        # Use intelligent selection first with fallback
        if self.use_intelligent_selection:
            try:
                return await self._lateral_thinking_with_intelligent_selection(problem, analysis)
            except Exception as e:
                logger.warning(f"Intelligent selection failed for lateral thinking, falling back: {e}")

        # Fallback: original implementation
        try:
            # Challenge assumptions
            assumptions = analysis.get('assumptions', [])

            response = await asyncio.to_thread(
                self.claude.messages.create,
                model="claude-3-opus-20240229",
                max_tokens=1500,
                messages=[{
                    "role": "user",
                    "content": f"""Use lateral thinking to solve this problem:

{problem.to_prompt_context()}

Assumptions to challenge: {', '.join(assumptions)}

Apply lateral thinking techniques:
1. Random entry: Start from an unrelated concept
2. Provocation: Make deliberately unreasonable statements
3. Movement: Extract useful ideas from provocations
4. Concept extraction: Find underlying principles
5. Challenge boundaries: Question all constraints

Generate an innovative solution that:
- Breaks conventional thinking
- Finds unexpected connections
- Simplifies radically
- Changes the problem definition if needed

Provide: solution description, implementation steps, innovation reasoning"""
                }]
            )
            
            solution_text = response.content[0].text
            
            return self._parse_solution(
                solution_text, problem, SolutionApproach.LATERAL_THINKING
            )
            
        except Exception as e:
            logger.error(f"Error in lateral thinking: {e}")
            return None
    
    async def _systematic_innovation_approach(self, problem: Problem,
                                            analysis: Dict[str, Any]) -> Optional[CreativeSolution]:
        """Use TRIZ-like systematic innovation.
        
        Applies systematic innovation principles similar to TRIZ methodology
        to resolve contradictions and increase solution ideality.
        
        Args:
            problem: The Problem instance to solve
            analysis: Problem analysis results containing leverage points
            
        Returns:
            CreativeSolution instance or None if generation fails
        """
        try:
            response = await asyncio.to_thread(
                self.claude.messages.create,
                model="claude-3-opus-20240229",
                max_tokens=1500,
                messages=[{
                    "role": "user",
                    "content": f"""Apply systematic innovation (TRIZ principles) to solve:

{problem.to_prompt_context()}

Leverage points: {', '.join(analysis.get('leverage_points', []))}

Apply these innovation principles:
1. Segmentation - Divide into independent parts
2. Asymmetry - Change from symmetrical to asymmetrical
3. Dynamics - Make objects adaptive
4. Periodic action - Use periodic or pulsating actions
5. Continuity - Make all parts work at full load
6. Rushing through - Conduct process at high speed
7. Convert harm to benefit - Use harmful factors to achieve positive effect

Generate solution focusing on:
- Resolving contradictions