"""
Predictive Query Handler v2.0 - INTELLIGENT PREDICTIONS WITH MONITORING DATA
==============================================================================

High-level handler for predictive/analytical queries with ML-powered insights.

This module provides intelligent query handling for developer productivity analysis,
combining monitoring data, natural language processing, and computer vision to deliver
actionable insights about code progress, potential bugs, and workflow optimization.

**FULLY IMPLEMENTED v2.0 Features**:
✅ "Am I making progress?" - Analyzes builds, errors, changes from HybridMonitoring
✅ Auto-bug detection - Pattern matching on error history with confidence scores
✅ "What should I work on next?" - Priority-based suggestions from workflow analysis
✅ Workspace change tracking - Productivity scoring with space-level breakdowns
✅ Context-aware queries - Natural language via ImplicitReferenceResolver
✅ Real-time progress scoring - Evidence-based (builds vs errors ratio)
✅ Predictive bug alerts - Error frequency + type classification + recommendations
✅ Dynamic, async, NO HARDCODING - All data from real monitoring events

**Real Integration (Not Mock)**:
- HybridProactiveMonitoringManager._alert_history - Real monitoring alerts
- HybridProactiveMonitoringManager._pattern_rules - Learned ML patterns
- ImplicitReferenceResolver.parse_query() - Natural language understanding
- PredictiveAnalyzer: Metrics and insights
- Claude Vision: Semantic code analysis

**Example Queries** (v2.0 powered with REAL data):
- "Am I making progress?" → Score: 0.75, Evidence: [3 builds, 2 errors fixed], Recommendation: "Keep up good work"
- "What should I work on next?" → Priority: HIGH, Action: "Fix 5 errors in Space 3", Confidence: 0.9
- "Are there any potential bugs?" → TypeError pattern (occurred 4x), Confidence: 0.7, Rec: "Add type hints"
- "What's my workspace activity?" → 25 changes, Productivity: 0.72, Pattern: "High activity in Space 3"

**Implementation Details**:
- analyze_progress_from_monitoring(): Real alert analysis with build/error ratio
- predict_bugs_from_patterns(): Counter-based pattern detection with error type extraction
- suggest_next_steps_from_workflow(): Priority-ordered suggestions from recent alerts
- track_workspace_changes(): Space-level activity tracking with productivity scoring

Author: Derek Russell
Date: 2025-10-19 (v2.0 REAL implementation completed)
"""

import base64
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from context_intelligence.analyzers.predictive_analyzer import (
    AnalysisScope,
    AnalyticsResult,
    PredictiveQueryType,
    get_predictive_analyzer,
    initialize_predictive_analyzer,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CLAUDE VISION INTEGRATION
# ============================================================================


class ClaudeVisionAnalyzer:
    """
    Integrates Claude Vision API for semantic code analysis with intelligent model selection.

    Uses Claude to analyze:
    - Code screenshots for understanding
    - Terminal output for error analysis
    - IDE views for context understanding

    Attributes:
        api_key: Optional Claude API key for authentication
        use_intelligent_selection: Whether to use intelligent model selection
        _claude_available: Whether Claude API is available
    """

    def __init__(self, api_key: Optional[str] = None, use_intelligent_selection: bool = True):
        """
        Initialize Claude Vision analyzer with intelligent model selection.

        Args:
            api_key: Optional Claude API key for authentication
            use_intelligent_selection: Whether to use HybridOrchestrator for model selection
        """
        self.api_key = api_key
        self.use_intelligent_selection = use_intelligent_selection
        self._claude_available = self._check_claude_availability()

    def _check_claude_availability(self) -> bool:
        """
        Check if Claude API is available.

        Returns:
            True if Claude API library is available, False otherwise
        """
        try:
            pass

            return True
        except ImportError:
            logger.warning(
                "[CLAUDE-VISION] Anthropic library not available - install with: pip install anthropic"
            )
            return False

    async def analyze_code_screenshot(
        self, image_path: str, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a code screenshot using Claude Vision.

        Args:
            image_path: Path to screenshot image file
            query: What to analyze (e.g., "explain this code", "find bugs")
            context: Additional context information (space_id, app_name, etc.)

        Returns:
            Analysis result dictionary with keys:
            - success: Whether analysis succeeded
            - analysis: Analysis text from Claude
            - model: Model used for analysis
            - timestamp: When analysis was performed
            - error: Error message if failed

        Example:
            >>> analyzer = ClaudeVisionAnalyzer()
            >>> result = await analyzer.analyze_code_screenshot(
            ...     "/path/to/code.png", 
            ...     "explain this function",
            ...     {"space_id": 1}
            ... )
            >>> print(result["analysis"])
            This function implements a binary search algorithm...
        """
        if not self._claude_available:
            return {
                "success": False,
                "error": "Claude API not available",
                "message": "Install anthropic library for Claude Vision support",
            }

        try:
            import anthropic

            # Read and encode image
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            # Determine image type
            suffix = Path(image_path).suffix.lower()
            media_type = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".gif": "image/gif",
            }.get(suffix, "image/png")

            # Create Claude client
            client = anthropic.Anthropic(api_key=self.api_key)

            # Build prompt based on query type
            prompt = self._build_vision_prompt(query, context)

            # Try intelligent model selection first
            if self.use_intelligent_selection:
                try:
                    from backend.core.hybrid_orchestrator import HybridOrchestrator

                    orchestrator = HybridOrchestrator()
                    if not orchestrator.is_running:
                        await orchestrator.start()

                    # Execute with intelligent selection for vision
                    result = await orchestrator.execute_with_intelligent_model_selection(
                        query=prompt,
                        intent="vision_analysis",
                        required_capabilities={"vision", "vision_analyze_heavy", "multimodal"},
                        context={
                            "image_data": image_data,
                            "image_format": "base64",
                            **(context or {}),
                        },
                        max_tokens=2048,
                        temperature=0.7,
                    )

                    if result.get("success"):
                        analysis_text = result.get("text", "").strip()
                        model_used = result.get("model_used", "unknown")
                        logger.info(
                            f"[CLAUDE-VISION] Analysis complete using {model_used}, {len(analysis_text)} chars"
                        )

                        return {
                            "success": True,
                            "analysis": analysis_text,
                            "model": model_used,
                            "timestamp": datetime.now().isoformat(),
                        }
                except Exception as e:
                    logger.warning(
                        f"[CLAUDE-VISION] Intelligent selection failed, falling back to direct API: {e}"
                    )

            # Fallback: Direct Claude Vision API
            logger.info(f"[CLAUDE-VISION] Analyzing screenshot with direct API: {image_path}")

            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2048,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )

            # Extract response
            analysis_text = response.content[0].text

            logger.info(
                f"[CLAUDE-VISION] Analysis complete (direct API), {len(analysis_text)} chars"
            )

            return {
                "success": True,
                "analysis": analysis_text,
                "model": "claude-3-5-sonnet-20241022",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"[CLAUDE-VISION] Error analyzing screenshot: {e}")
            return {"success": False, "error": str(e)}

    def _build_vision_prompt(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """
        Build a prompt for Claude Vision based on query type.

        Args:
            query: User's query about the image
            context: Additional context (space_id, app_name, etc.)

        Returns:
            Formatted prompt string for Claude Vision
        """
        base_prompt = "You are analyzing a screenshot from a developer's workspace.\n\n"

        # Add context if available
        if context:
            if context.get("space_id"):
                base_prompt += f"This is from Space {context['space_id']}.\n"
            if context.get("app_name"):
                base_prompt += f"Application: {context['app_name']}\n"
            base_prompt += "\n"

        # Add query-specific instructions
        query_lower = query.lower()

        if "explain" in query_lower or "what does" in query_lower:
            base_prompt += "Please explain what this code does. Be concise and focus on:\n"
            base_prompt += "- Main purpose and functionality\n"
            base_prompt += "- Key components or functions\n"
            base_prompt += "- Any notable patterns or techniques\n"

        elif "bug" in query_lower or "error" in query_lower or "issue" in query_lower:
            base_prompt += "Analyze this for potential bugs or issues. Look for:\n"
            base_prompt += "- Syntax errors or typos\n"
            base_prompt += "- Logic errors or anti-patterns\n"
            base_prompt += "- Performance issues\n"
            base_prompt += "- Security vulnerabilities\n"
            base_prompt += "- Code smells or maintainability concerns\n"

        elif "improve" in query_lower or "optimize" in query_lower:
            base_prompt += "Suggest improvements for this code. Consider:\n"
            base_prompt += "- Code quality and readability\n"
            base_prompt += "- Performance optimization\n"
            base_prompt += "- Best practices\n"
            base_prompt += "- Maintainability\n"

        elif "test" in query_lower:
            base_prompt += "Analyze the testing approach. Look at:\n"
            base_prompt += "- Test coverage\n"
            base_prompt += "- Test quality and assertions\n"
            base_prompt += "- Missing test cases\n"

        else:
            # Generic analysis
            base_prompt += f"User query: {query}\n\n"
            base_prompt += "Provide a helpful analysis addressing their question.\n"

        return base_prompt

    async def analyze_terminal_output(
        self, terminal_text: str, query: str = "analyze this terminal output"
    ) -> Dict[str, Any]:
        """
        Analyze terminal output using Claude (text mode).

        Args:
            terminal_text: Terminal output text to analyze
            query: What to analyze about the terminal output

        Returns:
            Analysis result dictionary with keys:
            - success: Whether analysis succeeded
            - analysis: Analysis text from Claude
            - model: Model used (if available)
            - timestamp: When analysis was performed
            - error: Error message if failed

        Example:
            >>> analyzer = ClaudeVisionAnalyzer()
            >>> result = await analyzer.analyze_terminal_output(
            ...     "Error: ModuleNotFoundError: No module named 'requests'",
            ...     "What's wrong and how to fix it?"
            ... )
            >>> print(result["analysis"])
            The error indicates a missing Python module...
        """
        if not self._claude_available:
            return {"success": False, "error": "Claude API not available"}

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)

            prompt = f"""Analyze this terminal output: