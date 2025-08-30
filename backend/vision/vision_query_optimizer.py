"""
Vision Query Optimizer - Intelligent prompt engineering for Claude Vision API
Reduces token usage while maintaining quality of results
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

class QueryType(Enum):
    """Types of vision queries with different optimization strategies"""
    TEXT_EXTRACTION = "text"
    UI_NAVIGATION = "ui"
    ACTIVITY_MONITORING = "activity"
    UPDATE_DETECTION = "update"
    SECURITY_CHECK = "security"
    GENERAL_ANALYSIS = "general"

@dataclass
class OptimizedQuery:
    """Optimized query with metadata"""
    prompt: str
    expected_tokens: int
    focus_areas: List[str]
    response_format: str
    compression_level: str

class VisionQueryOptimizer:
    """Optimize prompts for Claude Vision API to reduce token usage"""
    
    def __init__(self):
        # Prompt templates optimized for brevity and clarity
        self.templates = {
            QueryType.TEXT_EXTRACTION: {
                "prompt": "Extract text from {region}. Format: plain text, no descriptions.",
                "tokens": 15,
                "compression": "high"
            },
            QueryType.UI_NAVIGATION: {
                "prompt": "Find {target}. Reply: location(top/bottom/left/right/center), appearance, clickable(y/n).",
                "tokens": 20,
                "compression": "medium"
            },
            QueryType.ACTIVITY_MONITORING: {
                "prompt": "Current activity: apps open, task type, distractions. Brief list.",
                "tokens": 15,
                "compression": "high"
            },
            QueryType.UPDATE_DETECTION: {
                "prompt": "Updates/notifications? JSON: {updates_found:bool, details:[{type,app,urgency}]}",
                "tokens": 25,
                "compression": "medium"
            },
            QueryType.SECURITY_CHECK: {
                "prompt": "Security issues? List: exposed_data, suspicious_content, warnings. Action if needed.",
                "tokens": 20,
                "compression": "low"
            },
            QueryType.GENERAL_ANALYSIS: {
                "prompt": "{custom_prompt}",
                "tokens": 50,
                "compression": "medium"
            }
        }
        
        # Common abbreviations to reduce token usage
        self.abbreviations = {
            "application": "app",
            "notification": "notif",
            "information": "info",
            "configuration": "config",
            "description": "desc",
            "location": "loc",
            "appearance": "appear",
            "recommendations": "recs",
            "approximately": "~",
            "including": "incl",
            "excluding": "excl"
        }
        
        # Response format optimizations
        self.response_formats = {
            "json": "JSON only, no text:",
            "list": "Bullet list:",
            "brief": "1-2 sentences:",
            "keywords": "Keywords only:"
        }
    
    def optimize_prompt(self, 
                       query_type: QueryType,
                       custom_params: Optional[Dict[str, str]] = None,
                       max_tokens: int = 50) -> OptimizedQuery:
        """
        Optimize a prompt for minimal token usage
        
        Args:
            query_type: Type of query to optimize
            custom_params: Parameters to fill in template
            max_tokens: Maximum tokens for prompt
            
        Returns:
            Optimized query object
        """
        template = self.templates[query_type]
        prompt = template["prompt"]
        
        # Fill in custom parameters
        if custom_params:
            for key, value in custom_params.items():
                # Abbreviate values
                abbreviated_value = self._abbreviate_text(value)
                prompt = prompt.replace(f"{{{key}}}", abbreviated_value)
        
        # Truncate if needed
        if self._estimate_tokens(prompt) > max_tokens:
            prompt = self._truncate_prompt(prompt, max_tokens)
        
        # Determine focus areas based on query type
        focus_areas = self._get_focus_areas(query_type)
        
        # Get optimal response format
        response_format = self._get_response_format(query_type)
        
        return OptimizedQuery(
            prompt=prompt,
            expected_tokens=template["tokens"],
            focus_areas=focus_areas,
            response_format=response_format,
            compression_level=template["compression"]
        )
    
    def batch_optimize_queries(self, queries: List[Tuple[QueryType, Dict[str, str]]]) -> str:
        """
        Optimize multiple queries into a single efficient prompt
        
        Args:
            queries: List of (query_type, params) tuples
            
        Returns:
            Single optimized prompt combining all queries
        """
        if len(queries) == 1:
            return self.optimize_prompt(queries[0][0], queries[0][1]).prompt
        
        # Group similar queries
        grouped = {}
        for query_type, params in queries:
            if query_type not in grouped:
                grouped[query_type] = []
            grouped[query_type].append(params)
        
        # Build combined prompt
        combined_parts = []
        
        for query_type, params_list in grouped.items():
            if query_type == QueryType.TEXT_EXTRACTION:
                regions = [p.get('region', 'screen') for p in params_list]
                combined_parts.append(f"Text from: {', '.join(regions)}")
            
            elif query_type == QueryType.UI_NAVIGATION:
                targets = [p.get('target', 'element') for p in params_list]
                combined_parts.append(f"Find: {', '.join(targets)} (loc+clickable)")
            
            elif query_type == QueryType.UPDATE_DETECTION:
                combined_parts.append("Check all updates/notifs")
            
            else:
                # For other types, just add the optimized prompt
                for params in params_list:
                    opt = self.optimize_prompt(query_type, params)
                    combined_parts.append(opt.prompt)
        
        # Combine with numbered list for clarity
        combined = "Analyze:\n" + "\n".join(f"{i+1}. {part}" for i, part in enumerate(combined_parts))
        
        # Add response format instruction
        combined += "\nFormat: Brief numbered responses."
        
        return combined
    
    def _abbreviate_text(self, text: str) -> str:
        """Apply abbreviations to reduce token count"""
        result = text.lower()
        for full, abbr in self.abbreviations.items():
            result = result.replace(full, abbr)
        return result
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count"""
        # Approximate: 1 token ≈ 4 characters or 0.75 words
        word_count = len(text.split())
        char_count = len(text)
        return max(int(word_count * 0.75), int(char_count / 4))
    
    def _truncate_prompt(self, prompt: str, max_tokens: int) -> str:
        """Truncate prompt to fit token limit"""
        estimated_chars = max_tokens * 4
        if len(prompt) <= estimated_chars:
            return prompt
        
        # Truncate and add ellipsis
        return prompt[:estimated_chars-3] + "..."
    
    def _get_focus_areas(self, query_type: QueryType) -> List[str]:
        """Get areas to focus on for each query type"""
        focus_map = {
            QueryType.TEXT_EXTRACTION: ["text_content", "readability"],
            QueryType.UI_NAVIGATION: ["ui_elements", "interactive_elements"],
            QueryType.ACTIVITY_MONITORING: ["active_windows", "user_actions"],
            QueryType.UPDATE_DETECTION: ["notifications", "badges", "alerts"],
            QueryType.SECURITY_CHECK: ["passwords", "warnings", "popups"],
            QueryType.GENERAL_ANALYSIS: ["overall_content", "context"]
        }
        return focus_map.get(query_type, ["overall_content"])
    
    def _get_response_format(self, query_type: QueryType) -> str:
        """Get optimal response format for query type"""
        format_map = {
            QueryType.TEXT_EXTRACTION: "plain",
            QueryType.UI_NAVIGATION: "structured",
            QueryType.ACTIVITY_MONITORING: "list",
            QueryType.UPDATE_DETECTION: "json",
            QueryType.SECURITY_CHECK: "list",
            QueryType.GENERAL_ANALYSIS: "brief"
        }
        return format_map.get(query_type, "brief")
    
    def create_contextual_prompt(self, 
                               base_query: str,
                               context: Dict[str, Any],
                               max_tokens: int = 100) -> str:
        """
        Create a contextual prompt that includes relevant system state
        
        Args:
            base_query: Base query from user
            context: System context (active apps, recent actions, etc.)
            max_tokens: Maximum tokens for prompt
            
        Returns:
            Optimized contextual prompt
        """
        # Build context string efficiently
        context_parts = []
        
        if "active_app" in context:
            context_parts.append(f"In: {context['active_app']}")
        
        if "recent_action" in context:
            context_parts.append(f"After: {context['recent_action']}")
        
        if "focus_area" in context:
            context_parts.append(f"Focus: {context['focus_area']}")
        
        # Combine context with query
        if context_parts:
            context_str = " | ".join(context_parts)
            prompt = f"[{context_str}] {base_query}"
        else:
            prompt = base_query
        
        # Abbreviate and truncate if needed
        prompt = self._abbreviate_text(prompt)
        
        if self._estimate_tokens(prompt) > max_tokens:
            prompt = self._truncate_prompt(prompt, max_tokens)
        
        return prompt
    
    def extract_structured_response(self, 
                                  response: str,
                                  query_type: QueryType) -> Dict[str, Any]:
        """
        Extract structured data from Claude's response
        
        Args:
            response: Raw response from Claude
            query_type: Type of query for parsing context
            
        Returns:
            Structured data dictionary
        """
        result = {"raw_response": response}
        
        if query_type == QueryType.TEXT_EXTRACTION:
            # For text extraction, the whole response is the text
            result["extracted_text"] = response.strip()
        
        elif query_type == QueryType.UI_NAVIGATION:
            # Parse location and clickable info
            location_match = re.search(r'(top|bottom|left|right|center)', response.lower())
            clickable_match = re.search(r'clickable[:\s]*(y|yes|n|no)', response.lower())
            
            result["location"] = location_match.group(1) if location_match else "unknown"
            result["clickable"] = clickable_match.group(1).startswith('y') if clickable_match else False
        
        elif query_type == QueryType.UPDATE_DETECTION:
            # Try to parse JSON response
            try:
                import json
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result["updates"] = json.loads(json_match.group())
            except:
                # Fallback to text parsing
                result["updates_found"] = "update" in response.lower() or "notification" in response.lower()
        
        elif query_type == QueryType.ACTIVITY_MONITORING:
            # Extract app names and activities
            apps = []
            common_apps = ["chrome", "safari", "firefox", "vscode", "terminal", "slack", "mail"]
            response_lower = response.lower()
            
            for app in common_apps:
                if app in response_lower:
                    apps.append(app.title())
            
            result["active_apps"] = apps
            result["activity_summary"] = response.split('\n')[0] if '\n' in response else response
        
        elif query_type == QueryType.SECURITY_CHECK:
            # Extract security concerns
            concerns = []
            security_keywords = ["password", "sensitive", "warning", "suspicious", "malicious"]
            
            for keyword in security_keywords:
                if keyword in response.lower():
                    concerns.append(keyword)
            
            result["security_concerns"] = concerns
            result["has_issues"] = len(concerns) > 0
        
        return result
    
    def get_token_usage_report(self, queries: List[OptimizedQuery]) -> Dict[str, Any]:
        """Generate a report on token usage for optimization tracking"""
        total_tokens = sum(q.expected_tokens for q in queries)
        
        by_type = {}
        for q in queries:
            type_name = q.response_format
            if type_name not in by_type:
                by_type[type_name] = {"count": 0, "tokens": 0}
            by_type[type_name]["count"] += 1
            by_type[type_name]["tokens"] += q.expected_tokens
        
        return {
            "total_queries": len(queries),
            "total_estimated_tokens": total_tokens,
            "average_tokens_per_query": total_tokens / len(queries) if queries else 0,
            "by_response_type": by_type,
            "optimization_suggestions": self._get_optimization_suggestions(queries)
        }
    
    def _get_optimization_suggestions(self, queries: List[OptimizedQuery]) -> List[str]:
        """Provide suggestions for further optimization"""
        suggestions = []
        
        # Check for high token usage
        high_token_queries = [q for q in queries if q.expected_tokens > 30]
        if high_token_queries:
            suggestions.append(f"Consider batching {len(high_token_queries)} high-token queries")
        
        # Check for similar queries that could be combined
        text_queries = [q for q in queries if q.response_format == "plain"]
        if len(text_queries) > 2:
            suggestions.append(f"Combine {len(text_queries)} text extraction queries")
        
        # Check compression levels
        low_compression = [q for q in queries if q.compression_level == "low"]
        if low_compression:
            suggestions.append(f"Increase compression for {len(low_compression)} queries")
        
        return suggestions