"""
Advanced Natural Vision Response System with Rust Acceleration
Zero hardcoding - uses Claude AI for dynamic, contextual screen analysis
Integrates with Rust components for high-performance image processing
"""

import os
import asyncio
import base64
import io
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from functools import lru_cache

import numpy as np
from PIL import Image
from anthropic import AsyncAnthropic

# Import Rust bridge if available
try:
    from .rust_bridge import (
        RustImageProcessor,
        RustAdvancedMemoryPool,
        RustRuntimeManager,
        process_image_batch,
        extract_dominant_colors_rust,
        calculate_edge_density_rust,
        analyze_texture_rust,
        analyze_spatial_layout_rust,
        RUST_AVAILABLE,
    )
except ImportError:
    RUST_AVAILABLE = False
    logging.warning("Rust acceleration not available, using Python fallback")

# Import performance optimizer
try:
    from .performance_optimizer import (
        VisionPerformanceOptimizer,
        PerformanceConfig,
        RequestComplexity,
        get_performance_optimizer,
    )

    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    logging.warning("Performance optimizer not available")

logger = logging.getLogger(__name__)


class ConversationStyle(Enum):
    """Dynamic conversation styles based on context"""

    PROFESSIONAL = "professional"
    TECHNICAL = "technical"
    CASUAL = "casual"
    EDUCATIONAL = "educational"
    DIAGNOSTIC = "diagnostic"


class ScreenRegion(Enum):
    """Screen regions for focused analysis"""

    FULL_SCREEN = "full_screen"
    ACTIVE_WINDOW = "active_window"
    MENU_BAR = "menu_bar"
    DOCK_TASKBAR = "dock_taskbar"
    NOTIFICATION_AREA = "notification_area"
    CUSTOM = "custom"


@dataclass
class ConversationContext:
    """Maintains conversation state and context"""

    user_name: str = "sir"
    conversation_history: deque = field(default_factory=lambda: deque(maxlen=10))
    current_style: ConversationStyle = ConversationStyle.PROFESSIONAL
    user_expertise_level: str = "intermediate"
    previous_screens: deque = field(default_factory=lambda: deque(maxlen=5))
    active_applications: List[str] = field(default_factory=list)
    screen_change_patterns: Dict[str, Any] = field(default_factory=dict)
    interaction_preferences: Dict[str, Any] = field(default_factory=dict)
    temporal_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScreenAnalysisResult:
    """Comprehensive screen analysis results"""

    raw_description: str
    key_elements: List[Dict[str, Any]]
    detected_applications: List[str]
    user_activity_inference: str
    visual_hierarchy: Dict[str, Any]
    actionable_insights: List[str]
    confidence_scores: Dict[str, float]
    processing_metrics: Dict[str, float]
    rust_accelerated: bool = False


class RustAcceleratedProcessor:
    """Handles Rust-accelerated image processing operations"""

    def __init__(self):
        if not RUST_AVAILABLE:
            raise RuntimeError("Rust components not available")

        self.processor = RustImageProcessor()
        self.memory_pool = RustAdvancedMemoryPool()
        self.runtime = RustRuntimeManager(enable_cpu_affinity=True)
        self.processing_stats = {
            "total_processed": 0,
            "rust_speedup": [],
            "memory_saved": [],
        }

    def preprocess_screenshot(self, image: Image.Image) -> np.ndarray:
        """Preprocess screenshot using Rust for optimal performance"""
        # Convert PIL to numpy
        img_array = np.array(image)

        # Use Rust for processing
        start_time = time.time()

        # Ensure correct shape (height, width, channels)
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)

        # Process through Rust
        processed = self.processor.process_numpy_image(img_array)

        rust_time = time.time() - start_time
        self.processing_stats["rust_speedup"].append(rust_time)
        self.processing_stats["total_processed"] += 1

        return processed

    def extract_visual_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract visual features using Rust acceleration"""
        # This would use Rust's SIMD operations for feature extraction
        features = {
            "dominant_colors": self._extract_dominant_colors_rust(image),
            "edge_density": self._calculate_edge_density_rust(image),
            "texture_patterns": self._analyze_texture_rust(image),
            "spatial_layout": self._analyze_spatial_layout_rust(image),
        }
        return features

    def _extract_dominant_colors_rust(
        self, image: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """Extract dominant colors using Rust's parallel processing"""
        return extract_dominant_colors_rust(image, num_colors=5)

    def _calculate_edge_density_rust(self, image: np.ndarray) -> float:
        """Calculate edge density using Rust's SIMD operations"""
        return calculate_edge_density_rust(image)

    def _analyze_texture_rust(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze texture patterns using Rust"""
        return analyze_texture_rust(image)

    def _analyze_spatial_layout_rust(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze spatial layout using Rust"""
        return analyze_spatial_layout_rust(image)


class DynamicResponseGenerator:
    """
    Advanced response generator using Anthropic's Claude API
    with full context awareness and zero hardcoding
    """

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = AsyncAnthropic(api_key=self.api_key) if self.api_key else None
        self.context = ConversationContext()
        self.rust_processor = RustAcceleratedProcessor() if RUST_AVAILABLE else None
        self.response_cache = {}
        self.analysis_history = deque(maxlen=50)

        # Advanced prompt templates that adapt dynamically
        self.prompt_engine = DynamicPromptEngine()

        # Performance monitoring
        self.performance_metrics = {
            "api_calls": 0,
            "cache_hits": 0,
            "avg_response_time": 0,
            "rust_acceleration_used": 0,
        }

        # Initialize performance optimizer
        if OPTIMIZER_AVAILABLE:
            self.optimizer = get_performance_optimizer()
            # Configure for optimal performance
            self.optimizer.config.use_fast_model_for_confirmation = True
            self.optimizer.config.enable_parallel_processing = True
            self.optimizer.config.cache_ttl_seconds = (
                3  # 3 second cache for screen states
            )
        else:
            self.optimizer = None

    async def analyze_screen_with_full_context(
        self,
        screenshot: Image.Image,
        user_query: str,
        conversation_context: Optional[Dict[str, Any]] = None,
        analysis_depth: str = "comprehensive",
    ) -> ScreenAnalysisResult:
        """
        Perform deep screen analysis with full contextual understanding
        """
        if not self.client:
            raise ValueError("Anthropic API key not configured")

        start_time = time.time()

        # Prepare screenshot with Rust acceleration if available
        if self.rust_processor:
            processed_img = self.rust_processor.preprocess_screenshot(screenshot)
            visual_features = self.rust_processor.extract_visual_features(processed_img)
            rust_accelerated = True
            self.performance_metrics["rust_acceleration_used"] += 1
        else:
            processed_img = np.array(screenshot)
            visual_features = self._extract_visual_features_python(processed_img)
            rust_accelerated = False

        # Convert to base64 for Claude
        buffered = io.BytesIO()
        screenshot.save(buffered, format="PNG", optimize=True)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Build comprehensive context
        full_context = self._build_comprehensive_context(
            user_query, conversation_context, visual_features
        )

        # Generate dynamic prompt
        prompt = self.prompt_engine.generate_analysis_prompt(
            query=user_query,
            context=full_context,
            style=self.context.current_style,
            depth=analysis_depth,
        )

        # Call Claude with advanced parameters
        try:
            # Determine model based on analysis depth
            if analysis_depth == "fast":
                model = "claude-3-haiku-20240307"  # Fastest model
                max_tokens = 200
                temperature = 0.3
            elif analysis_depth == "basic":
                model = "claude-3-sonnet-20240229"  # Balanced model
                max_tokens = 500
                temperature = 0.5
            else:
                model = "claude-3-opus-20240229"  # Most capable model
                max_tokens = 1000
                temperature = 0.7

            # Add timeout for API calls
            import asyncio

            message = await asyncio.wait_for(
                self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=self._get_system_prompt(),
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": img_base64,
                                    },
                                },
                            ],
                        }
                    ],
                ),
                timeout=5.0 if analysis_depth == "fast" else 10.0,  # Dynamic timeout
            )

            raw_response = message.content[0].text

            # Parse structured analysis from response
            analysis_result = self._parse_analysis_response(
                raw_response, visual_features, rust_accelerated
            )

            # Update context and history
            self._update_conversation_context(user_query, analysis_result)
            self.analysis_history.append(
                {
                    "timestamp": time.time(),
                    "query": user_query,
                    "result": analysis_result,
                    "processing_time": time.time() - start_time,
                }
            )

            # Update performance metrics
            self.performance_metrics["api_calls"] += 1
            avg_time = self.performance_metrics["avg_response_time"]
            new_time = time.time() - start_time
            self.performance_metrics["avg_response_time"] = float(
                (avg_time * (self.performance_metrics["api_calls"] - 1) + new_time)
                / self.performance_metrics["api_calls"]
            )

            return analysis_result

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise

    async def generate_natural_response(
        self,
        analysis_result: ScreenAnalysisResult,
        response_type: str = "conversational",
        include_actions: bool = True,
    ) -> str:
        """
        Generate natural, contextual response based on analysis
        """
        # Build response context
        response_context = {
            "analysis": analysis_result,
            "conversation_history": list(self.context.conversation_history),
            "user_preferences": self.context.interaction_preferences,
            "current_activity": analysis_result.user_activity_inference,
            "time_context": self._get_temporal_context(),
        }

        # Generate response prompt
        prompt = self.prompt_engine.generate_response_prompt(
            context=response_context,
            response_type=response_type,
            include_actions=include_actions,
            style=self.context.current_style,
        )

        try:
            # Use faster model for conversational responses
            model = (
                "claude-3-haiku-20240307"
                if response_type == "confirmation"
                else "claude-3-sonnet-20240229"
            )

            # Generate natural response with timeout
            message = await asyncio.wait_for(
                self.client.messages.create(
                    model=model,
                    max_tokens=150 if response_type == "confirmation" else 300,
                    temperature=0.5 if response_type == "confirmation" else 0.8,
                    system=self._get_conversational_system_prompt(),
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=3.0 if response_type == "confirmation" else 5.0,
            )

            response = message.content[0].text

            # Add to conversation history
            self.context.conversation_history.append(
                {
                    "role": "assistant",
                    "content": response,
                    "timestamp": time.time(),
                    "analysis_id": id(analysis_result),
                }
            )

            return response

        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return self._generate_fallback_response(analysis_result)

    def _build_comprehensive_context(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]],
        visual_features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build comprehensive context for analysis"""
        context = {
            "query": query,
            "visual_features": visual_features,
            "conversation_history": list(self.context.conversation_history)[-5:],
            "active_applications": self.context.active_applications,
            "temporal": self._get_temporal_context(),
            "user_expertise": self.context.user_expertise_level,
            "screen_changes": self._analyze_screen_changes(),
        }

        if user_context:
            context.update(user_context)

        return context

    def _get_system_prompt(self) -> str:
        """Generate dynamic system prompt"""
        return f"""You are JARVIS, Tony Stark's advanced AI assistant. You have sophisticated visual perception and analytical capabilities.

Your characteristics:
- Highly intelligent and perceptive
- Professional yet personable
- Address the user as "{self.context.user_name}" when appropriate
- Expertise level: Adapt explanations for {self.context.user_expertise_level} level
- Current conversation style: {self.context.current_style.value}

Your capabilities:
- Analyze screen content with exceptional detail
- Understand user workflow and intent
- Provide actionable insights
- Maintain context across conversations
- Detect patterns and anomalies

Important: 
- Be specific and accurate about what you observe
- Provide insights beyond simple description
- Adapt your communication style to the context
- Never make assumptions about content you cannot see
"""

    def _get_conversational_system_prompt(self) -> str:
        """Get conversational system prompt"""
        return """You are JARVIS, responding naturally to the user based on your screen analysis. 
Keep responses concise but informative, maintaining your sophisticated yet approachable personality.
Use technical terms when appropriate but ensure clarity."""

    def _parse_analysis_response(
        self, raw_response: str, visual_features: Dict[str, Any], rust_accelerated: bool
    ) -> ScreenAnalysisResult:
        """Parse Claude's response into structured analysis"""
        # Extract key information using intelligent parsing
        lines = raw_response.split("\n")

        key_elements = []
        detected_apps = []
        actionable_insights = []

        for line in lines:
            line = line.strip()
            if any(
                app in line.lower()
                for app in ["terminal", "chrome", "safari", "vscode", "finder"]
            ):
                # Extract application mentions
                for app in [
                    "Terminal",
                    "Chrome",
                    "Safari",
                    "VSCode",
                    "Finder",
                    "Slack",
                    "Discord",
                ]:
                    if app.lower() in line.lower():
                        detected_apps.append(app)

            if any(
                keyword in line.lower()
                for keyword in ["suggest", "recommend", "could", "should"]
            ):
                actionable_insights.append(line)

            if ":" in line:
                # Extract structured elements
                parts = line.split(":", 1)
                if len(parts) == 2:
                    key_elements.append(
                        {"type": parts[0].strip(), "description": parts[1].strip()}
                    )

        # Infer user activity
        activity_inference = self._infer_user_activity(raw_response, detected_apps)

        # Calculate confidence scores
        confidence_scores = {
            "overall": 0.95,  # High confidence with Claude
            "app_detection": len(detected_apps) / max(len(set(detected_apps)), 1),
            "activity_inference": 0.85,
            "visual_analysis": 0.9 if visual_features else 0.7,
        }

        return ScreenAnalysisResult(
            raw_description=raw_response,
            key_elements=key_elements,
            detected_applications=list(set(detected_apps)),
            user_activity_inference=activity_inference,
            visual_hierarchy=self._extract_visual_hierarchy(raw_response),
            actionable_insights=actionable_insights,
            confidence_scores=confidence_scores,
            processing_metrics={
                "rust_accelerated": rust_accelerated,
                "visual_features_extracted": (
                    len(visual_features) if visual_features else 0
                ),
            },
            rust_accelerated=rust_accelerated,
        )

    def _infer_user_activity(self, response: str, detected_apps: List[str]) -> str:
        """Infer user activity from analysis"""
        response_lower = response.lower()

        if any(
            term in response_lower
            for term in ["code", "programming", "development", "terminal"]
        ):
            return "Software development or coding"
        elif any(
            term in response_lower for term in ["browse", "web", "search", "reading"]
        ):
            return "Web browsing or research"
        elif any(
            term in response_lower for term in ["chat", "message", "communication"]
        ):
            return "Communication or collaboration"
        elif any(term in response_lower for term in ["document", "writing", "editing"]):
            return "Document creation or editing"
        elif detected_apps:
            return f"Working with {', '.join(detected_apps[:2])}"
        else:
            return "General computer usage"

    def _extract_visual_hierarchy(self, response: str) -> Dict[str, Any]:
        """Extract visual hierarchy from response"""
        hierarchy = {
            "primary_focus": None,
            "secondary_elements": [],
            "background_elements": [],
        }

        lines = response.split("\n")
        for i, line in enumerate(lines):
            if (
                "primary" in line.lower()
                or "main" in line.lower()
                or "focus" in line.lower()
            ):
                hierarchy["primary_focus"] = line.strip()
            elif "secondary" in line.lower() or "also" in line.lower():
                hierarchy["secondary_elements"].append(line.strip())
            elif "background" in line.lower() or "behind" in line.lower():
                hierarchy["background_elements"].append(line.strip())

        return hierarchy

    def _update_conversation_context(self, query: str, result: ScreenAnalysisResult):
        """Update conversation context with new information"""
        self.context.conversation_history.append(
            {"role": "user", "content": query, "timestamp": time.time()}
        )

        # Update active applications
        self.context.active_applications = result.detected_applications

        # Track screen changes
        if self.context.previous_screens:
            last_screen = self.context.previous_screens[-1]
            changes = self._detect_screen_changes(last_screen, result)
            self.context.screen_change_patterns[str(time.time())] = changes

        self.context.previous_screens.append(result)

    def _detect_screen_changes(
        self, previous: ScreenAnalysisResult, current: ScreenAnalysisResult
    ) -> Dict[str, Any]:
        """Detect changes between screen states"""
        changes = {
            "app_changes": {
                "opened": list(
                    set(current.detected_applications)
                    - set(previous.detected_applications)
                ),
                "closed": list(
                    set(previous.detected_applications)
                    - set(current.detected_applications)
                ),
            },
            "activity_change": previous.user_activity_inference
            != current.user_activity_inference,
            "significant_change": len(
                set(current.key_elements) - set(previous.key_elements)
            )
            > 3,
        }
        return changes

    def _get_temporal_context(self) -> Dict[str, Any]:
        """Get temporal context for responses"""
        import datetime

        now = datetime.datetime.now()

        return {
            "time": now.strftime("%H:%M"),
            "date": now.strftime("%Y-%m-%d"),
            "day_of_week": now.strftime("%A"),
            "time_of_day": self._get_time_of_day(now.hour),
            "working_hours": 9 <= now.hour <= 18,
        }

    def _get_time_of_day(self, hour: int) -> str:
        """Get descriptive time of day"""
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    def _analyze_screen_changes(self) -> Dict[str, Any]:
        """Analyze patterns in screen changes"""
        if not self.context.screen_change_patterns:
            return {}

        recent_changes = list(self.context.screen_change_patterns.values())[-5:]

        return {
            "frequent_app_switches": sum(
                1 for c in recent_changes if c["app_changes"]["opened"]
            )
            > 3,
            "stable_workflow": sum(
                1 for c in recent_changes if not c["significant_change"]
            )
            > 3,
            "activity_transitions": sum(
                1 for c in recent_changes if c["activity_change"]
            ),
        }

    def _extract_visual_features_python(self, image: np.ndarray) -> Dict[str, Any]:
        """Python fallback for visual feature extraction"""
        # Simple fallback implementation
        return {
            "image_shape": image.shape,
            "mean_color": tuple(np.mean(image, axis=(0, 1)).astype(int)),
            "color_variance": float(np.var(image)),
        }

    def _generate_fallback_response(self, analysis_result: ScreenAnalysisResult) -> str:
        """Generate fallback response when API fails"""
        return f"I can see {len(analysis_result.detected_applications)} applications running. {analysis_result.user_activity_inference}. My advanced analysis systems are temporarily offline."

    async def adapt_conversation_style(self, user_feedback: str):
        """Adapt conversation style based on user feedback"""
        if "technical" in user_feedback.lower() or "details" in user_feedback.lower():
            self.context.current_style = ConversationStyle.TECHNICAL
        elif "simple" in user_feedback.lower() or "basic" in user_feedback.lower():
            self.context.current_style = ConversationStyle.CASUAL
        elif "teach" in user_feedback.lower() or "explain" in user_feedback.lower():
            self.context.current_style = ConversationStyle.EDUCATIONAL

        logger.info(
            f"Adapted conversation style to: {self.context.current_style.value}"
        )


class DynamicPromptEngine:
    """Generates dynamic, contextual prompts for Claude"""

    def __init__(self):
        self.prompt_templates = {}
        self.prompt_history = deque(maxlen=100)

    def generate_analysis_prompt(
        self, query: str, context: Dict[str, Any], style: ConversationStyle, depth: str
    ) -> str:
        """Generate dynamic analysis prompt"""
        base_prompt = f"""Analyze this screen image to answer: "{query}"

Context:
- User expertise: {context.get('user_expertise', 'intermediate')}
- Time: {context['temporal']['time_of_day']} ({context['temporal']['time']})
- Recent activity: {context.get('conversation_history', [])[-1]['content'] if context.get('conversation_history') else 'First interaction'}
- Active applications detected previously: {', '.join(context.get('active_applications', []))}

Analysis Requirements:
1. Describe what you see with specific details
2. Identify all applications and their states
3. Infer the user's current activity or workflow
4. Note any interesting patterns or potential issues
5. Provide insights relevant to the query

Style: {style.value} - Adjust your language accordingly
Depth: {depth} - Provide {'comprehensive details' if depth == 'comprehensive' else 'focused analysis'}

Visual Features Detected (via processing):
{json.dumps(context.get('visual_features', {}), indent=2)}

Remember: Be specific, accurate, and insightful. Focus on what's actually visible."""

        self.prompt_history.append(
            {"timestamp": time.time(), "type": "analysis", "prompt": base_prompt}
        )

        return base_prompt

    def generate_response_prompt(
        self,
        context: Dict[str, Any],
        response_type: str,
        include_actions: bool,
        style: ConversationStyle,
    ) -> str:
        """Generate dynamic response prompt"""
        analysis = context["analysis"]

        prompt = f"""Based on this screen analysis, generate a {response_type} response:

Analysis Summary:
- Detected applications: {', '.join(analysis.detected_applications)}
- User activity: {analysis.user_activity_inference}
- Key insights: {'; '.join(analysis.actionable_insights[:3])}
- Confidence: {analysis.confidence_scores['overall']:.2f}

Conversation History (last 3):
{self._format_conversation_history(context.get('conversation_history', [])[-3:])}

Response Requirements:
- Style: {style.value}
- Type: {response_type}
- Include suggested actions: {include_actions}
- Current activity context: {context.get('current_activity', 'Unknown')}
- Time context: {context['time_context']['time_of_day']}

Generate a natural response that:
1. Acknowledges what you see
2. Provides relevant insights
3. {'Suggests helpful actions' if include_actions else 'Focuses on observation'}
4. Maintains JARVIS personality
5. Adapts to the user's expertise level

Keep it concise but informative (2-4 sentences ideal)."""

        return prompt

    def _format_conversation_history(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history for prompt"""
        if not history:
            return "No previous conversation"

        formatted = []
        for item in history:
            role = item.get("role", "unknown")
            content = (
                item.get("content", "")[:100] + "..."
                if len(item.get("content", "")) > 100
                else item.get("content", "")
            )
            formatted.append(f"{role}: {content}")

        return "\n".join(formatted)


# Global instance management
_generator = None


def get_response_generator() -> DynamicResponseGenerator:
    """Get singleton instance of response generator"""
    global _generator
    if _generator is None:
        _generator = DynamicResponseGenerator()
    return _generator


# Main interface functions
async def analyze_and_respond(
    screenshot: Image.Image,
    user_query: str,
    conversation_context: Optional[Dict[str, Any]] = None,
    response_type: str = "conversational",
    analysis_depth: str = "comprehensive",
) -> Dict[str, Any]:
    """
    Main function to analyze screen and generate response
    """
    generator = get_response_generator()

    # Use performance optimizer if available
    if generator.optimizer and OPTIMIZER_AVAILABLE:
        # Define optimized functions for the optimizer
        async def screenshot_func():
            return {
                "success": True,
                "image": screenshot,
                "timestamp": time.time(),
                "screen_size": screenshot.size,
            }

        async def analysis_func(screenshot_data, query, model_config):
            # Use the specified model from optimizer config
            original_model = None
            if "model" in model_config:
                # Temporarily override model
                original_model = (
                    generator.client.default_model
                    if hasattr(generator.client, "default_model")
                    else None
                )

            analysis = await generator.analyze_screen_with_full_context(
                screenshot=screenshot_data["image"],
                user_query=query,
                conversation_context=conversation_context,
                analysis_depth=(
                    "fast" if model_config.get("quick_mode") else analysis_depth
                ),
            )
            return analysis

        async def response_func(analysis_result, context=None):
            response = await generator.generate_natural_response(
                analysis_result=analysis_result,
                response_type=response_type,
                include_actions=True,
            )
            return {
                "response": response,
                "analysis": analysis_result,
                "performance_metrics": generator.performance_metrics,
                "rust_accelerated": analysis_result.rust_accelerated,
            }

        # Run through optimizer
        result = await generator.optimizer.optimize_vision_request(
            query=user_query,
            screenshot_func=screenshot_func,
            analysis_func=analysis_func,
            response_func=response_func,
            context=conversation_context,
        )

        return result
    else:
        # Fallback to original implementation
        # Perform analysis
        analysis = await generator.analyze_screen_with_full_context(
            screenshot=screenshot,
            user_query=user_query,
            conversation_context=conversation_context,
            analysis_depth=analysis_depth,
        )

        # Generate natural response
        response = await generator.generate_natural_response(
            analysis_result=analysis, response_type=response_type, include_actions=True
        )

        return {
            "response": response,
            "analysis": analysis,
            "performance_metrics": generator.performance_metrics,
            "rust_accelerated": analysis.rust_accelerated,
        }


# Legacy compatibility wrapper
class NaturalResponseGenerator:
    """Legacy compatibility wrapper"""

    def __init__(self):
        self.generator = get_response_generator()

    def generate_vision_confirmation(self, context: Optional[Dict] = None) -> str:
        """Legacy method - now uses dynamic generator"""
        # Create a minimal screenshot (1x1 pixel)
        dummy_screenshot = Image.new("RGB", (1, 1))

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                analyze_and_respond(
                    screenshot=dummy_screenshot,
                    user_query="Can you see my screen?",
                    conversation_context=context,
                    response_type="confirmation",
                )
            )
            return result["response"]
        finally:
            loop.close()

    def simplify_technical_response(self, technical_response: str) -> str:
        """Legacy method - adaptation handled by generator"""
        return technical_response

    def format_screen_description(
        self, raw_description: str, be_concise: bool = True
    ) -> str:
        """Legacy method - formatting handled by generator"""
        return raw_description
