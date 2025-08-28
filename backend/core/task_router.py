"""
Task Router - Intelligence Dispatcher for JARVIS
Analyzes query complexity and routes to appropriate model tier
"""

import logging
import re
import asyncio
from typing import Dict, Any, Tuple, Optional, List
from enum import Enum
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

# Try to import NLTK, but make it optional
try:
    import nltk
    from nltk.tokenize import word_tokenize
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except:
            logger.warning("Could not download NLTK data, using fallback tokenizer")
    NLTK_AVAILABLE = True
except ImportError:
    logger.warning("NLTK not available, using fallback tokenizer")
    NLTK_AVAILABLE = False
    # Simple fallback tokenizer
    def word_tokenize(text):
        return text.split()

class TaskType(Enum):
    """Types of tasks JARVIS can handle"""
    CHAT = "chat"              # Simple conversation
    CODE = "code"              # Code generation/analysis
    ANALYSIS = "analysis"      # Data analysis, reasoning
    CREATIVE = "creative"      # Creative writing, stories
    FACTUAL = "factual"        # Fact-based Q&A
    COMPLEX = "complex"        # Multi-step reasoning
    

@dataclass
class TaskAnalysis:
    """Analysis of a task/query"""
    task_type: TaskType
    complexity: float          # 0.0 to 1.0
    estimated_tokens: int
    requires_context: bool
    confidence: float         # Confidence in analysis
    reasoning: str           # Explanation of analysis
    

class TaskRouter:
    """Routes tasks to appropriate models based on complexity analysis"""
    
    def __init__(self, model_manager, memory_controller):
        self.model_manager = model_manager
        self.memory_controller = memory_controller
        
        # Complexity indicators
        self.complexity_indicators = {
            "high": [
                r"analyze.*complex",
                r"debug.*code",
                r"explain.*detail",
                r"compare.*contrast",
                r"design.*system",
                r"architect",
                r"optimize.*performance",
                r"mathematical proof",
                r"multi.*step"
            ],
            "medium": [
                r"write.*code",
                r"create.*function",
                r"explain",
                r"summarize",
                r"translate",
                r"convert",
                r"generate.*list"
            ],
            "low": [
                r"hello",
                r"hi",
                r"thank",
                r"what.*time",
                r"simple.*question",
                r"yes.*no",
                r"define"
            ]
        }
        
        # Task type patterns
        self.task_patterns = {
            TaskType.CODE: [
                r"code", r"function", r"class", r"debug", r"program",
                r"python", r"javascript", r"java", r"implement", r"algorithm"
            ],
            TaskType.ANALYSIS: [
                r"analyze", r"compare", r"evaluate", r"assess", r"examine",
                r"investigate", r"research", r"study"
            ],
            TaskType.CREATIVE: [
                r"story", r"poem", r"creative", r"imagine", r"fiction",
                r"character", r"plot", r"narrative"
            ],
            TaskType.FACTUAL: [
                r"what is", r"who is", r"when did", r"where is", r"fact",
                r"definition", r"explain what"
            ],
            TaskType.COMPLEX: [
                r"step by step", r"plan", r"strategy", r"multiple", r"stages",
                r"phases", r"comprehensive"
            ]
        }
        
        # Performance tracking
        self.routing_stats = {
            "total_requests": 0,
            "routing_times": [],
            "model_usage": {tier.value: 0 for tier in self.model_manager.models.keys()},
            "task_types": {task_type.value: 0 for task_type in TaskType}
        }
        
    async def route_task(self, query: str, context: Optional[List[str]] = None) -> Tuple[Any, Dict[str, Any]]:
        """Route a task to the appropriate model"""
        start_time = time.time()
        
        # Analyze the task
        analysis = self.analyze_task(query, context)
        
        # Get appropriate model
        model, tier = await self.model_manager.get_model_for_task(
            analysis.task_type.value,
            analysis.complexity,
            analysis.estimated_tokens
        )
        
        # Track statistics
        routing_time = time.time() - start_time
        self._update_stats(analysis, tier, routing_time)
        
        # Prepare routing info
        routing_info = {
            "analysis": analysis,
            "model_tier": tier.value,
            "routing_time": routing_time,
            "memory_state": self.memory_controller.get_memory_stats()["current"]["pressure"]
        }
        
        logger.info(f"Routed task to {tier.value} model: {analysis.task_type.value}, "
                   f"complexity={analysis.complexity:.2f}")
        
        return model, routing_info
        
    def analyze_task(self, query: str, context: Optional[List[str]] = None) -> TaskAnalysis:
        """Analyze a task to determine type and complexity"""
        query_lower = query.lower()
        
        # Determine task type
        task_type = self._identify_task_type(query_lower)
        
        # Calculate complexity
        complexity = self._calculate_complexity(query_lower, task_type)
        
        # Estimate tokens
        estimated_tokens = self._estimate_tokens(query, context)
        
        # Check if context is required
        requires_context = self._requires_context(query_lower, task_type)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(task_type, complexity, estimated_tokens)
        
        # Calculate confidence
        confidence = self._calculate_confidence(query_lower, task_type)
        
        return TaskAnalysis(
            task_type=task_type,
            complexity=complexity,
            estimated_tokens=estimated_tokens,
            requires_context=requires_context,
            confidence=confidence,
            reasoning=reasoning
        )
        
    def _identify_task_type(self, query: str) -> TaskType:
        """Identify the type of task"""
        # Check each task type pattern
        scores = {}
        
        for task_type, patterns in self.task_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, query))
            scores[task_type] = score
            
        # Get highest scoring type
        if scores:
            best_type = max(scores, key=scores.get)
            if scores[best_type] > 0:
                return best_type
                
        # Default to CHAT
        return TaskType.CHAT
        
    def _calculate_complexity(self, query: str, task_type: TaskType) -> float:
        """Calculate task complexity (0.0 to 1.0)"""
        complexity = 0.3  # Base complexity
        
        # Check complexity indicators
        high_matches = sum(1 for pattern in self.complexity_indicators["high"] 
                          if re.search(pattern, query))
        medium_matches = sum(1 for pattern in self.complexity_indicators["medium"] 
                           if re.search(pattern, query))
        low_matches = sum(1 for pattern in self.complexity_indicators["low"] 
                         if re.search(pattern, query))
                         
        # Adjust complexity based on matches
        if high_matches > 0:
            complexity += 0.3 * high_matches
        if medium_matches > 0:
            complexity += 0.15 * medium_matches
        if low_matches > 0:
            complexity -= 0.1 * low_matches
            
        # Task type adjustments
        if task_type in [TaskType.COMPLEX, TaskType.ANALYSIS]:
            complexity += 0.2
        elif task_type == TaskType.CODE:
            complexity += 0.1
        elif task_type == TaskType.CHAT:
            complexity -= 0.1
            
        # Query length adjustment
        word_count = len(word_tokenize(query))
        if word_count > 50:
            complexity += 0.1
        elif word_count < 10:
            complexity -= 0.1
            
        # Clamp to valid range
        return max(0.0, min(1.0, complexity))
        
    def _estimate_tokens(self, query: str, context: Optional[List[str]] = None) -> int:
        """Estimate total tokens needed"""
        # Rough estimation: 1 token ≈ 4 characters
        query_tokens = len(query) // 4
        
        context_tokens = 0
        if context:
            context_tokens = sum(len(c) // 4 for c in context)
            
        # Add some buffer for response
        response_buffer = 200
        
        return query_tokens + context_tokens + response_buffer
        
    def _requires_context(self, query: str, task_type: TaskType) -> bool:
        """Determine if task requires context"""
        context_indicators = [
            r"previous", r"earlier", r"before", r"context", r"continue",
            r"based on", r"referring", r"mentioned"
        ]
        
        # Check for context indicators
        if any(re.search(pattern, query) for pattern in context_indicators):
            return True
            
        # Some task types typically need context
        if task_type in [TaskType.COMPLEX, TaskType.ANALYSIS]:
            return True
            
        return False
        
    def _calculate_confidence(self, query: str, task_type: TaskType) -> float:
        """Calculate confidence in the analysis"""
        confidence = 0.7  # Base confidence
        
        # Clear task type indicators increase confidence
        pattern_matches = sum(
            1 for pattern in self.task_patterns.get(task_type, [])
            if re.search(pattern, query)
        )
        
        if pattern_matches >= 2:
            confidence += 0.2
        elif pattern_matches == 1:
            confidence += 0.1
            
        # Very short or very long queries reduce confidence
        word_count = len(word_tokenize(query))
        if word_count < 5 or word_count > 100:
            confidence -= 0.1
            
        return max(0.0, min(1.0, confidence))
        
    def _generate_reasoning(self, task_type: TaskType, complexity: float, 
                          tokens: int) -> str:
        """Generate reasoning for the routing decision"""
        reasons = []
        
        # Task type reasoning
        reasons.append(f"Task identified as {task_type.value}")
        
        # Complexity reasoning
        if complexity > 0.7:
            reasons.append("High complexity requiring advanced model")
        elif complexity > 0.4:
            reasons.append("Moderate complexity suitable for standard model")
        else:
            reasons.append("Low complexity can be handled by lightweight model")
            
        # Token reasoning
        if tokens > 3000:
            reasons.append(f"Large context ({tokens} tokens) requires model with bigger context window")
        elif tokens > 1500:
            reasons.append(f"Moderate context size ({tokens} tokens)")
            
        return "; ".join(reasons)
        
    def _update_stats(self, analysis: TaskAnalysis, tier: Any, routing_time: float):
        """Update routing statistics"""
        self.routing_stats["total_requests"] += 1
        self.routing_stats["routing_times"].append(routing_time)
        self.routing_stats["model_usage"][tier.value] += 1
        self.routing_stats["task_types"][analysis.task_type.value] += 1
        
        # Keep only recent routing times
        if len(self.routing_stats["routing_times"]) > 1000:
            self.routing_stats["routing_times"] = self.routing_stats["routing_times"][-1000:]
            
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        avg_routing_time = (
            sum(self.routing_stats["routing_times"]) / len(self.routing_stats["routing_times"])
            if self.routing_stats["routing_times"] else 0
        )
        
        return {
            "total_requests": self.routing_stats["total_requests"],
            "average_routing_time": avg_routing_time,
            "model_usage": self.routing_stats["model_usage"],
            "task_type_distribution": self.routing_stats["task_types"]
        }
        
    def suggest_optimization(self) -> List[str]:
        """Suggest optimizations based on usage patterns"""
        suggestions = []
        stats = self.routing_stats
        
        # Check model usage patterns
        total_usage = sum(stats["model_usage"].values())
        if total_usage > 10:
            # Check if tiny model is underutilized
            tiny_percent = stats["model_usage"].get("tiny", 0) / total_usage
            if tiny_percent < 0.2:
                suggestions.append("Consider routing more simple queries to TinyLlama to save resources")
                
            # Check if advanced model is overused
            adv_percent = stats["model_usage"].get("adv", 0) / total_usage
            if adv_percent > 0.5:
                suggestions.append("High advanced model usage - consider if all tasks truly need it")
                
        # Check routing performance
        avg_time = (
            sum(stats["routing_times"]) / len(stats["routing_times"])
            if stats["routing_times"] else 0
        )
        if stats["routing_times"] and avg_time > 0.1:
            suggestions.append("Routing taking longer than optimal - consider caching decisions")
            
        return suggestions