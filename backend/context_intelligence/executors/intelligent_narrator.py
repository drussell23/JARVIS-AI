"""
Intelligent Narrator for JARVIS Document Creation
==================================================

Advanced AI-powered narration system with:
- Dynamic, context-aware message generation using Claude
- Adaptive timing based on activity and progress
- Content analysis for relevant updates
- Anti-repetition and engagement optimization
- Zero hardcoding - fully intelligent decision making
"""

import asyncio
import logging
import time
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class NarrationContext:
    """Rich context for intelligent narration"""
    # Document info
    topic: str
    document_type: str
    format: str
    target_word_count: int
    
    # Progress tracking
    current_phase: str
    current_section: str = ""
    word_count: int = 0
    sections_completed: List[str] = field(default_factory=list)
    
    # Content analysis
    recent_content: str = ""
    key_themes: List[str] = field(default_factory=list)
    writing_velocity: float = 0.0  # words per second
    
    # Narration history
    last_narration_time: float = 0
    last_narration_hash: str = ""
    narration_count: int = 0
    recent_narrations: List[str] = field(default_factory=list)
    
    # User engagement
    session_start_time: float = field(default_factory=time.time)
    activity_level: str = "normal"  # "low", "normal", "high"
    
    def add_narration(self, message: str):
        """Track narration history"""
        self.narration_count += 1
        self.last_narration_time = time.time()
        self.last_narration_hash = hashlib.md5(message.encode()).hexdigest()
        self.recent_narrations.append(message)
        if len(self.recent_narrations) > 10:
            self.recent_narrations.pop(0)
    
    def get_session_duration(self) -> float:
        """Get duration of current session in seconds"""
        return time.time() - self.session_start_time
    
    def get_progress_percentage(self) -> float:
        """Calculate progress percentage"""
        if self.target_word_count == 0:
            return 0.0
        return min((self.word_count / self.target_word_count) * 100, 100)


class IntelligentNarrator:
    """
    AI-powered narrator with adaptive, context-aware communication
    """
    
    def __init__(self, claude_client=None):
        """Initialize intelligent narrator"""
        self._claude = claude_client
        self._context: Optional[NarrationContext] = None
        
        # Adaptive timing parameters (tuned to prevent overlap and reduce frequency)
        self.min_interval = 6.0  # Minimum 6 seconds between narrations (gives TTS time to finish)
        self.max_interval = 15.0  # Maximum seconds of silence
        self.base_interval = 8.0  # Base interval for normal activity
        
        # Intelligence thresholds
        self.significance_threshold = 0.7  # How "important" something must be to narrate (raised to be more selective)
        self.repetition_similarity_threshold = 0.7  # Avoid similar messages
        self.engagement_decay = 0.95  # Reduce frequency if user hasn't responded
        
        # Content analysis patterns
        self.milestone_patterns = {
            'structural': ['introduction', 'conclusion', 'thesis', 'argument'],
            'progress': ['halfway', 'quarter', 'third', 'milestone'],
            'quality': ['analysis', 'evidence', 'example', 'citation']
        }
        
    async def initialize(self, topic: str, doc_type: str, format_style: str, 
                        target_words: int, claude_client=None):
        """Initialize narrator with document context"""
        if claude_client:
            self._claude = claude_client
            
        self._context = NarrationContext(
            topic=topic,
            document_type=doc_type,
            format=format_style,
            target_word_count=target_words
        )
        logger.info(f"[INTELLIGENT NARRATOR] Initialized for: {topic}")
    
    async def should_narrate(self, phase: str, content_update: Optional[str] = None) -> Tuple[bool, str]:
        """
        Intelligently decide if narration should occur
        Returns: (should_narrate: bool, reason: str)
        """
        if not self._context:
            return False, "No context"
        
        # Calculate time since last narration
        time_since_last = time.time() - self._context.last_narration_time
        
        # Always narrate on first call
        if self._context.narration_count == 0:
            return True, "First narration"
        
        # Don't narrate too frequently
        if time_since_last < self.min_interval:
            return False, f"Too soon ({time_since_last:.1f}s < {self.min_interval}s)"
        
        # Force narration if too much silence
        if time_since_last > self.max_interval:
            return True, f"Max silence reached ({time_since_last:.1f}s)"
        
        # Calculate significance score
        significance = await self._calculate_significance(phase, content_update)
        
        logger.info(f"[INTELLIGENT NARRATOR] Significance: {significance:.2f} (threshold: {self.significance_threshold})")
        
        if significance >= self.significance_threshold:
            return True, f"High significance ({significance:.2f})"
        
        # Adaptive timing based on activity
        adaptive_interval = self._calculate_adaptive_interval()
        if time_since_last >= adaptive_interval:
            return True, f"Adaptive interval reached ({time_since_last:.1f}s >= {adaptive_interval:.1f}s)"
        
        return False, f"Not significant enough ({significance:.2f})"
    
    async def _calculate_significance(self, phase: str, content_update: Optional[str]) -> float:
        """
        Calculate how significant/important this moment is
        Returns: 0.0 to 1.0 score
        """
        significance = 0.0
        
        # Phase-based significance
        phase_weights = {
            'acknowledging_request': 1.0,  # Always important
            'starting_writing': 0.9,
            'writing_section': 0.7,
            'outline_complete': 0.8,
            'progress_update': 0.5,
            'document_ready': 1.0,
            'writing_complete': 1.0
        }
        significance += phase_weights.get(phase, 0.4)
        
        # Progress milestone significance (0%, 25%, 50%, 75%, 100%)
        progress = self._context.get_progress_percentage()
        milestone_distance = min(
            abs(progress - 0), abs(progress - 25), abs(progress - 50),
            abs(progress - 75), abs(progress - 100)
        )
        if milestone_distance < 5:  # Within 5% of a milestone
            significance += 0.3
        
        # Content-based significance
        if content_update:
            # Check for structural keywords
            content_lower = content_update.lower()
            for category, keywords in self.milestone_patterns.items():
                if any(kw in content_lower for kw in keywords):
                    significance += 0.2
                    break
        
        # Section change significance
        if phase == 'writing_section' and self._context.current_section:
            significance += 0.3
        
        # Normalize to 0-1 range
        return min(significance, 1.0)
    
    def _calculate_adaptive_interval(self) -> float:
        """Calculate adaptive interval based on context"""
        interval = self.base_interval
        
        # Adjust based on progress (more frequent at start and end)
        progress = self._context.get_progress_percentage()
        if progress < 10 or progress > 90:
            interval *= 0.8  # Slightly more frequent at very start/end
        elif 30 < progress < 70:
            interval *= 1.4  # Much less frequent in middle to avoid annoyance
        
        # Adjust based on writing velocity
        if self._context.writing_velocity > 30:  # Fast writing (>30 words/sec)
            interval *= 1.3  # Give more space
        elif self._context.writing_velocity < 10:  # Slow writing
            interval *= 0.9  # Check in more often
        
        return max(self.min_interval, min(interval, self.max_interval))
    
    async def generate_narration(self, phase: str, additional_context: Dict[str, Any] = None) -> str:
        """
        Generate intelligent, context-aware narration using Claude AI
        """
        if not self._context:
            return "Processing..."
        
        # Update context
        if additional_context:
            for key, value in additional_context.items():
                if hasattr(self._context, key):
                    setattr(self._context, key, value)
        
        # Build rich context prompt for Claude
        prompt = self._build_narration_prompt(phase)
        
        try:
            # Generate narration with Claude
            if self._claude:
                narration = await self._generate_with_claude(prompt)
            else:
                narration = await self._generate_fallback(phase)
            
            # Validate and refine
            narration = self._refine_narration(narration)
            
            # Check for repetition
            if self._is_too_similar_to_recent(narration):
                logger.info(f"[INTELLIGENT NARRATOR] Regenerating to avoid repetition")
                narration = await self._generate_with_variation(prompt, narration)
            
            # Track narration
            self._context.add_narration(narration)
            
            return narration
            
        except Exception as e:
            logger.error(f"[INTELLIGENT NARRATOR] Error generating narration: {e}")
            return await self._generate_fallback(phase)
    
    def _build_narration_prompt(self, phase: str) -> str:
        """Build intelligent prompt for Claude"""
        progress = self._context.get_progress_percentage()
        session_duration = self._context.get_session_duration()
        
        # Analyze what to emphasize
        emphasis = self._determine_emphasis(phase, progress)
        
        prompt = f"""You are JARVIS, Tony Stark's AI assistant. Generate a single, natural sentence (8-15 words) to update the user about document writing progress.

Context:
- Topic: {self._context.topic}
- Document type: {self._context.document_type}
- Current phase: {phase}
- Progress: {progress:.1f}% ({self._context.word_count}/{self._context.target_word_count} words)
- Current section: {self._context.current_section or 'N/A'}
- Session duration: {session_duration:.0f}s
- Emphasis: {emphasis}

Recent narrations (DON'T repeat these):
{chr(10).join(f'  - {n}' for n in self._context.recent_narrations[-3:])}

Personality guidelines:
- Sound engaged and interested in the {self._context.topic}
- Use "Sir" occasionally (20% of time) - naturally
- Vary your language - be conversational, not robotic
- Reference specific progress/sections when relevant
- Match the urgency to the phase (calm for middle, energetic for milestones)
- Be encouraging but not overly enthusiastic
- Sound like you're actively watching and understanding the content

Generate ONE natural sentence that JARVIS would say right now:"""

        return prompt
    
    def _determine_emphasis(self, phase: str, progress: float) -> str:
        """Determine what to emphasize in narration"""
        if phase in ['acknowledging_request', 'starting_writing']:
            return "Getting started, set expectations"
        elif phase == 'writing_section':
            return f"Section transition - {self._context.current_section}"
        elif progress < 20:
            return "Building foundation, early momentum"
        elif 40 < progress < 60:
            return "Steady progress, maintain engagement"
        elif progress > 80:
            return "Near completion, final push"
        elif phase == 'progress_update':
            return f"Progress milestone - {self._context.word_count} words achieved"
        else:
            return "General progress update"
    
    async def _generate_with_claude(self, prompt: str) -> str:
        """Generate narration using Claude API"""
        try:
            full_response = ""
            async for chunk in self._claude.stream_content(
                prompt,
                max_tokens=80,
                model="claude-3-5-sonnet-20241022",
                temperature=0.9  # Higher temperature for more variety
            ):
                full_response += chunk
            
            return full_response.strip().strip('"').strip()
        except Exception as e:
            logger.error(f"[INTELLIGENT NARRATOR] Claude error: {e}")
            raise
    
    async def _generate_with_variation(self, prompt: str, avoid: str) -> str:
        """Regenerate with explicit instruction to vary from previous"""
        varied_prompt = f"{prompt}\n\nIMPORTANT: Do NOT say anything similar to: \"{avoid}\"\nGenerate something COMPLETELY different:"
        
        return await self._generate_with_claude(varied_prompt)
    
    async def _generate_fallback(self, phase: str) -> str:
        """Simple fallback if Claude unavailable"""
        import random
        
        progress = self._context.get_progress_percentage()
        
        fallbacks = {
            'starting_writing': [
                f"Writing about {self._context.topic}",
                "Getting the words down",
                "Composing the content"
            ],
            'progress_update': [
                f"{self._context.word_count} words written",
                f"Progress: {progress:.0f}%",
                "Making headway"
            ],
            'writing_section': [
                f"Writing {self._context.current_section}",
                f"Developing {self._context.current_section}",
                f"Now covering {self._context.current_section}"
            ]
        }
        
        messages = fallbacks.get(phase, ["Processing..."])
        return random.choice(messages)
    
    def _refine_narration(self, narration: str) -> str:
        """Clean up and refine the narration"""
        # Remove markdown, quotes, etc.
        narration = re.sub(r'[*_~`]', '', narration)
        narration = narration.strip('"\'')
        
        # Ensure it ends properly
        if not narration[-1] in '.!?':
            narration += '.'
        
        # Capitalize first letter
        if narration:
            narration = narration[0].upper() + narration[1:]
        
        return narration
    
    def _is_too_similar_to_recent(self, narration: str) -> bool:
        """Check if narration is too similar to recent ones"""
        if not self._context.recent_narrations:
            return False
        
        # Simple similarity check (can be enhanced with NLP)
        narration_words = set(narration.lower().split())
        
        for recent in self._context.recent_narrations[-3:]:
            recent_words = set(recent.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(narration_words & recent_words)
            union = len(narration_words | recent_words)
            
            if union > 0:
                similarity = intersection / union
                if similarity > self.repetition_similarity_threshold:
                    return True
        
        return False
    
    def update_writing_metrics(self, new_word_count: int, time_delta: float):
        """Update writing velocity and other metrics"""
        if self._context:
            words_added = new_word_count - self._context.word_count
            self._context.word_count = new_word_count
            
            if time_delta > 0:
                self._context.writing_velocity = words_added / time_delta
    
    def update_content_analysis(self, recent_text: str):
        """Analyze recent content for context"""
        if self._context:
            self._context.recent_content = recent_text
            # Could add more sophisticated NLP analysis here


# Global instance
_narrator_instance: Optional[IntelligentNarrator] = None


def get_intelligent_narrator(claude_client=None) -> IntelligentNarrator:
    """Get or create global intelligent narrator"""
    global _narrator_instance
    if _narrator_instance is None:
        _narrator_instance = IntelligentNarrator(claude_client)
    elif claude_client and not _narrator_instance._claude:
        _narrator_instance._claude = claude_client
    return _narrator_instance