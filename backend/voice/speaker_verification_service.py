"""
Speaker Verification Service for JARVIS
Provides voice biometric verification for security-sensitive operations

Features:
- Speaker identification from audio
- Confidence scoring
- Primary user (owner) detection
- Integration with learning database
- Background pre-loading for instant unlock
- LangGraph-based adaptive authentication reasoning
- ChromaDB voice pattern recognition and anti-spoofing
- Langfuse authentication audit trails
- Helicone voice processing cost optimization
- Multi-factor authentication fusion
- Progressive confidence communication

Enhanced Version: 2.0.0
"""

import asyncio
import logging
import threading
import hashlib
import time
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

import numpy as np

# ============================================================================
# CRITICAL FIX: Patch torchaudio for compatibility with version 2.9.0+
# ============================================================================
# Issue: torchaudio.list_audio_backends() was removed in torchaudio 2.9.0
# This breaks SpeechBrain 1.0.3 which still calls the deprecated function
# Solution: Monkey patch torchaudio before importing SpeechBrain
# ============================================================================
try:
    import torchaudio

    # Check if list_audio_backends is missing (torchaudio >= 2.9.0)
    if not hasattr(torchaudio, 'list_audio_backends'):
        logging.getLogger(__name__).info(
            "🔧 Patching torchaudio 2.9.0+ for SpeechBrain compatibility..."
        )

        # Add dummy list_audio_backends function that returns available backends
        # In torchaudio 2.9+, the backend system was simplified - we can safely
        # return a list of known backends without actually checking
        def _list_audio_backends_fallback():
            """
            Fallback implementation for removed torchaudio.list_audio_backends()

            Returns list of potentially available backends. Since torchaudio 2.9+
            handles backend selection automatically, we just return common ones.
            """
            backends = []
            try:
                # Try to import soundfile (most common backend)
                import soundfile
                backends.append('soundfile')
            except ImportError:
                pass

            try:
                # Try to import sox_io
                import torchaudio.backend.sox_io_backend
                backends.append('sox_io')
            except (ImportError, AttributeError):
                pass

            # If no backends found, return default
            if not backends:
                backends = ['soundfile']  # Default assumption

            return backends

        # Monkey patch the missing function
        torchaudio.list_audio_backends = _list_audio_backends_fallback

        logging.getLogger(__name__).info(
            f"✅ torchaudio patched successfully - backends: {torchaudio.list_audio_backends()}"
        )

except ImportError as e:
    logging.getLogger(__name__).warning(f"Could not patch torchaudio: {e}")

# Now safe to import SpeechBrain components
from intelligence.learning_database import JARVISLearningDatabase
from voice.engines.speechbrain_engine import SpeechBrainEngine
from voice.stt_config import ModelConfig, STTEngine

logger = logging.getLogger(__name__)

# ============================================================================
# Optional Dependencies for Enhanced Features
# ============================================================================

# ChromaDB for voice pattern recognition
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.info("ChromaDB not available - voice pattern store disabled")

# Langfuse for observability
try:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logger.info("Langfuse not available - audit trails disabled")

# LangGraph for reasoning
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.info("LangGraph not available - using fallback reasoning")


# ============================================================================
# Enhanced Authentication Enums and Types
# ============================================================================

class AuthenticationPhase(str, Enum):
    """Phases of the authentication process."""
    AUDIO_CAPTURE = "audio_capture"
    VOICE_ANALYSIS = "voice_analysis"
    EMBEDDING_EXTRACTION = "embedding_extraction"
    SPEAKER_VERIFICATION = "speaker_verification"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"
    ANTI_SPOOFING = "anti_spoofing"
    MULTI_FACTOR_FUSION = "multi_factor_fusion"
    DECISION = "decision"
    COMPLETE = "complete"


class ConfidenceLevel(str, Enum):
    """Human-readable confidence levels for voice feedback."""
    EXCELLENT = "excellent"      # >90%
    GOOD = "good"               # 85-90%
    BORDERLINE = "borderline"   # 80-85%
    LOW = "low"                 # 75-80%
    FAILED = "failed"           # <75%


class ThreatType(str, Enum):
    """Types of detected threats."""
    REPLAY_ATTACK = "replay_attack"
    VOICE_CLONING = "voice_cloning"
    SYNTHETIC_VOICE = "synthetic_voice"
    ENVIRONMENTAL_ANOMALY = "environmental_anomaly"
    UNKNOWN_SPEAKER = "unknown_speaker"
    NONE = "none"


# ============================================================================
# Data Classes for Enhanced Authentication
# ============================================================================

@dataclass
class AuthenticationTrace:
    """Complete trace of an authentication attempt for Langfuse."""
    trace_id: str
    speaker_name: str
    timestamp: datetime
    phases: List[Dict[str, Any]] = field(default_factory=list)

    # Audio metrics
    audio_duration_ms: float = 0.0
    audio_snr_db: float = 0.0
    audio_quality_score: float = 0.0

    # Verification metrics
    voice_confidence: float = 0.0
    behavioral_confidence: float = 0.0
    context_confidence: float = 0.0
    fused_confidence: float = 0.0

    # Decision
    decision: str = "pending"
    threshold_used: float = 0.0

    # Security
    threat_detected: ThreatType = ThreatType.NONE
    anti_spoofing_score: float = 0.0

    # Performance
    total_duration_ms: float = 0.0
    api_cost_usd: float = 0.0

    # Context
    environment: str = "default"
    device: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "speaker_name": self.speaker_name,
            "timestamp": self.timestamp.isoformat(),
            "phases": self.phases,
            "audio": {
                "duration_ms": self.audio_duration_ms,
                "snr_db": self.audio_snr_db,
                "quality_score": self.audio_quality_score
            },
            "verification": {
                "voice_confidence": self.voice_confidence,
                "behavioral_confidence": self.behavioral_confidence,
                "context_confidence": self.context_confidence,
                "fused_confidence": self.fused_confidence
            },
            "decision": self.decision,
            "threshold_used": self.threshold_used,
            "security": {
                "threat_detected": self.threat_detected.value,
                "anti_spoofing_score": self.anti_spoofing_score
            },
            "performance": {
                "total_duration_ms": self.total_duration_ms,
                "api_cost_usd": self.api_cost_usd
            },
            "context": {
                "environment": self.environment,
                "device": self.device
            }
        }


@dataclass
class VoicePattern:
    """Voice pattern for ChromaDB storage."""
    pattern_id: str
    speaker_name: str
    pattern_type: str  # 'rhythm', 'phrase', 'environment', 'emotion'
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    success_count: int = 0
    failure_count: int = 0


@dataclass
class VoiceFeedback:
    """Voice feedback message for the user."""
    confidence_level: ConfidenceLevel
    message: str
    suggestion: Optional[str] = None
    is_final: bool = False
    speak_aloud: bool = True


# ============================================================================
# Voice Pattern Store (ChromaDB)
# ============================================================================

class VoicePatternStore:
    """
    ChromaDB-based store for voice patterns and behavioral biometrics.

    Stores:
    - Speaking rhythm patterns
    - Phrase preferences
    - Environmental signatures
    - Emotional baselines
    - Time-of-day voice variations
    """

    def __init__(self, persist_directory: Optional[str] = None):
        self.logger = logging.getLogger(f"{__name__}.VoicePatternStore")
        self._initialized = False
        self._client = None
        self._collection = None
        self.persist_directory = persist_directory or "/tmp/jarvis_voice_patterns"

    async def initialize(self):
        """Initialize ChromaDB client and collection."""
        if not CHROMADB_AVAILABLE:
            self.logger.warning("ChromaDB not available - pattern store disabled")
            return

        try:
            self._client = chromadb.Client(ChromaSettings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.persist_directory,
                anonymized_telemetry=False
            ))

            self._collection = self._client.get_or_create_collection(
                name="voice_patterns",
                metadata={"description": "JARVIS voice behavioral patterns"}
            )

            self._initialized = True
            self.logger.info(f"✅ Voice pattern store initialized with {self._collection.count()} patterns")

        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            self._initialized = False

    async def store_pattern(self, pattern: VoicePattern) -> bool:
        """Store a voice pattern in ChromaDB."""
        if not self._initialized:
            return False

        try:
            self._collection.add(
                ids=[pattern.pattern_id],
                embeddings=[pattern.embedding.tolist()],
                metadatas=[{
                    "speaker_name": pattern.speaker_name,
                    "pattern_type": pattern.pattern_type,
                    "created_at": pattern.created_at.isoformat(),
                    "success_count": pattern.success_count,
                    "failure_count": pattern.failure_count,
                    **pattern.metadata
                }]
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to store pattern: {e}")
            return False

    async def find_similar_patterns(
        self,
        embedding: np.ndarray,
        speaker_name: str,
        pattern_type: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar voice patterns for anti-spoofing and behavioral analysis."""
        if not self._initialized:
            return []

        try:
            where_filter = {"speaker_name": speaker_name}
            if pattern_type:
                where_filter["pattern_type"] = pattern_type

            results = self._collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=top_k,
                where=where_filter
            )

            patterns = []
            if results['ids'] and results['ids'][0]:
                for i, pattern_id in enumerate(results['ids'][0]):
                    patterns.append({
                        "pattern_id": pattern_id,
                        "distance": results['distances'][0][i] if results.get('distances') else 0,
                        "metadata": results['metadatas'][0][i] if results.get('metadatas') else {}
                    })
            return patterns

        except Exception as e:
            self.logger.error(f"Failed to query patterns: {e}")
            return []

    async def detect_replay_attack(
        self,
        audio_fingerprint: str,
        speaker_name: str,
        time_window_seconds: int = 300
    ) -> Tuple[bool, float]:
        """
        Detect if this exact audio has been played before (replay attack).

        Returns:
            Tuple of (is_replay, anomaly_score)
        """
        if not self._initialized:
            return False, 0.0

        try:
            # Check for exact fingerprint match in recent history
            recent_cutoff = (datetime.utcnow() - timedelta(seconds=time_window_seconds)).isoformat()

            results = self._collection.get(
                where={
                    "speaker_name": speaker_name,
                    "pattern_type": "audio_fingerprint",
                    "fingerprint": audio_fingerprint
                }
            )

            if results['ids']:
                # Found matching fingerprint - potential replay attack
                return True, 0.95

            return False, 0.0

        except Exception as e:
            self.logger.error(f"Replay detection failed: {e}")
            return False, 0.0

    async def store_audio_fingerprint(
        self,
        speaker_name: str,
        audio_fingerprint: str,
        embedding: np.ndarray
    ):
        """Store audio fingerprint for replay attack detection."""
        pattern = VoicePattern(
            pattern_id=f"fp_{audio_fingerprint[:16]}_{uuid4().hex[:8]}",
            speaker_name=speaker_name,
            pattern_type="audio_fingerprint",
            embedding=embedding,
            metadata={"fingerprint": audio_fingerprint}
        )
        await self.store_pattern(pattern)


# ============================================================================
# Authentication Audit Trail (Langfuse)
# ============================================================================

class AuthenticationAuditTrail:
    """
    Langfuse-based audit trail for authentication attempts.

    Provides complete transparency into authentication decisions.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.AuditTrail")
        self._langfuse = None
        self._initialized = False
        self._trace_cache: Dict[str, AuthenticationTrace] = {}

    async def initialize(self):
        """Initialize Langfuse client."""
        if not LANGFUSE_AVAILABLE:
            self.logger.info("Langfuse not available - using local audit trail")
            self._initialized = True  # Use local storage fallback
            return

        try:
            self._langfuse = Langfuse()
            self._initialized = True
            self.logger.info("✅ Langfuse audit trail initialized")
        except Exception as e:
            self.logger.warning(f"Langfuse initialization failed, using local: {e}")
            self._initialized = True  # Fallback to local

    def start_trace(self, speaker_name: str, environment: str = "default") -> str:
        """Start a new authentication trace."""
        trace_id = f"auth_{uuid4().hex[:16]}"
        trace = AuthenticationTrace(
            trace_id=trace_id,
            speaker_name=speaker_name,
            timestamp=datetime.utcnow(),
            environment=environment
        )
        self._trace_cache[trace_id] = trace

        if self._langfuse:
            try:
                self._langfuse.trace(
                    id=trace_id,
                    name="voice_authentication",
                    user_id=speaker_name,
                    metadata={"environment": environment}
                )
            except Exception as e:
                self.logger.debug(f"Langfuse trace failed: {e}")

        return trace_id

    def log_phase(
        self,
        trace_id: str,
        phase: AuthenticationPhase,
        duration_ms: float,
        metrics: Dict[str, Any],
        success: bool = True
    ):
        """Log an authentication phase."""
        if trace_id not in self._trace_cache:
            return

        trace = self._trace_cache[trace_id]
        phase_data = {
            "phase": phase.value,
            "duration_ms": duration_ms,
            "success": success,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        trace.phases.append(phase_data)

        if self._langfuse:
            try:
                self._langfuse.span(
                    trace_id=trace_id,
                    name=phase.value,
                    metadata=metrics
                )
            except Exception:
                pass

    def complete_trace(
        self,
        trace_id: str,
        decision: str,
        confidence: float,
        threat: ThreatType = ThreatType.NONE
    ) -> Optional[AuthenticationTrace]:
        """Complete and finalize an authentication trace."""
        if trace_id not in self._trace_cache:
            return None

        trace = self._trace_cache[trace_id]
        trace.decision = decision
        trace.fused_confidence = confidence
        trace.threat_detected = threat
        trace.total_duration_ms = sum(p.get("duration_ms", 0) for p in trace.phases)

        # Estimate API cost
        trace.api_cost_usd = self._estimate_cost(trace)

        if self._langfuse:
            try:
                self._langfuse.score(
                    trace_id=trace_id,
                    name="confidence",
                    value=confidence
                )
                self._langfuse.score(
                    trace_id=trace_id,
                    name="decision",
                    value=1.0 if decision == "authenticated" else 0.0
                )
            except Exception:
                pass

        # Log to local file as backup
        self._log_to_file(trace)

        return trace

    def _estimate_cost(self, trace: AuthenticationTrace) -> float:
        """Estimate API cost for the authentication."""
        # Base cost estimation
        cost = 0.0

        for phase in trace.phases:
            phase_name = phase.get("phase", "")
            if "embedding" in phase_name:
                cost += 0.002  # Embedding extraction
            elif "verification" in phase_name:
                cost += 0.001  # Verification

        return cost

    def _log_to_file(self, trace: AuthenticationTrace):
        """Log trace to local file for backup."""
        try:
            import os
            log_dir = "/tmp/jarvis_auth_logs"
            os.makedirs(log_dir, exist_ok=True)

            log_file = f"{log_dir}/auth_{trace.timestamp.strftime('%Y%m%d')}.jsonl"
            with open(log_file, "a") as f:
                f.write(json.dumps(trace.to_dict()) + "\n")
        except Exception as e:
            self.logger.debug(f"Local log failed: {e}")

    def get_trace(self, trace_id: str) -> Optional[AuthenticationTrace]:
        """Get a trace by ID."""
        return self._trace_cache.get(trace_id)

    def get_recent_traces(
        self,
        speaker_name: Optional[str] = None,
        limit: int = 20
    ) -> List[AuthenticationTrace]:
        """Get recent authentication traces."""
        traces = list(self._trace_cache.values())

        if speaker_name:
            traces = [t for t in traces if t.speaker_name == speaker_name]

        traces.sort(key=lambda t: t.timestamp, reverse=True)
        return traces[:limit]


# ============================================================================
# Voice Processing Cache (Helicone-style)
# ============================================================================

class VoiceProcessingCache:
    """
    Intelligent caching for voice processing to reduce costs.

    Caches:
    - Recent voice embeddings
    - Verification results
    - Audio quality analyses
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.logger = logging.getLogger(f"{__name__}.VoiceCache")
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._stats = {
            "hits": 0,
            "misses": 0,
            "cost_saved_usd": 0.0
        }

    def _generate_key(self, audio_data: bytes, operation: str) -> str:
        """Generate cache key from audio fingerprint."""
        audio_hash = hashlib.sha256(audio_data[:8000]).hexdigest()[:32]
        return f"{operation}:{audio_hash}"

    def get(self, audio_data: bytes, operation: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available and fresh."""
        key = self._generate_key(audio_data, operation)

        if key in self._cache:
            entry = self._cache[key]
            age = (datetime.utcnow() - entry["timestamp"]).total_seconds()

            if age < self._ttl_seconds:
                self._stats["hits"] += 1
                self._stats["cost_saved_usd"] += entry.get("estimated_cost", 0.002)
                self.logger.debug(f"Cache hit for {operation} (age: {age:.1f}s)")
                return entry["result"]
            else:
                del self._cache[key]

        self._stats["misses"] += 1
        return None

    def set(
        self,
        audio_data: bytes,
        operation: str,
        result: Dict[str, Any],
        estimated_cost: float = 0.002
    ):
        """Cache a processing result."""
        key = self._generate_key(audio_data, operation)

        # Evict oldest if at capacity
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["timestamp"])
            del self._cache[oldest_key]

        self._cache[key] = {
            "result": result,
            "timestamp": datetime.utcnow(),
            "estimated_cost": estimated_cost
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total * 100 if total > 0 else 0

        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate_percent": hit_rate,
            "cost_saved_usd": self._stats["cost_saved_usd"],
            "cache_size": len(self._cache)
        }

    def clear(self):
        """Clear the cache."""
        self._cache.clear()


# ============================================================================
# Voice Feedback Generator
# ============================================================================

class VoiceFeedbackGenerator:
    """
    Generates natural, conversational feedback during authentication.

    Makes authentication feel like talking to a trusted security professional.
    """

    def __init__(self, user_name: str = "Derek"):
        self.user_name = user_name
        self._feedback_templates = {
            ConfidenceLevel.EXCELLENT: [
                f"Of course, {user_name}. Unlocking for you.",
                f"Welcome back, {user_name}.",
                f"Good to hear you, {user_name}. Unlocking now."
            ],
            ConfidenceLevel.GOOD: [
                f"Good morning, {user_name}. Unlocking now.",
                f"Verified. Unlocking for you, {user_name}."
            ],
            ConfidenceLevel.BORDERLINE: [
                f"One moment... yes, verified. Unlocking for you, {user_name}.",
                f"I can confirm it's you, {user_name}. Unlocking now."
            ],
            ConfidenceLevel.LOW: [
                "I'm having a little trouble hearing you clearly. Let me try again...",
                "Your voice sounds a bit different today. Let me adjust...",
                "Give me a second - filtering out background noise..."
            ],
            ConfidenceLevel.FAILED: [
                "I'm not able to verify your voice right now. Want to try again, or use manual authentication?",
                "Voice verification didn't match. Would you like to try speaking closer to the microphone?",
                "I couldn't confirm your identity. Try speaking more clearly, or use an alternative method."
            ]
        }

        self._environmental_feedback = {
            "noisy": "Give me a second - filtering out background noise... Got it - verified despite the noise.",
            "quiet_late": "Up late again? Unlocking quietly for you.",
            "sick_voice": "Your voice sounds different - hope you're feeling okay. I can still verify it's you.",
            "new_location": "First time unlocking from this location. Let me recalibrate... Got it!"
        }

    def get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Map confidence score to human-readable level."""
        if confidence >= 0.90:
            return ConfidenceLevel.EXCELLENT
        elif confidence >= 0.85:
            return ConfidenceLevel.GOOD
        elif confidence >= 0.80:
            return ConfidenceLevel.BORDERLINE
        elif confidence >= 0.75:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.FAILED

    def generate_feedback(
        self,
        confidence: float,
        context: Optional[Dict[str, Any]] = None
    ) -> VoiceFeedback:
        """Generate appropriate voice feedback based on confidence and context."""
        import random

        level = self.get_confidence_level(confidence)
        templates = self._feedback_templates[level]
        message = random.choice(templates)

        suggestion = None
        if level == ConfidenceLevel.LOW:
            suggestion = "Try speaking a bit louder and closer to the microphone."
        elif level == ConfidenceLevel.FAILED:
            suggestion = "You can also unlock using your password or Face ID."

        # Add environmental context
        if context:
            if context.get("snr_db", 20) < 12:
                message = self._environmental_feedback["noisy"]
            elif context.get("hour", 12) >= 23 or context.get("hour", 12) <= 5:
                message = self._environmental_feedback["quiet_late"]
            elif context.get("voice_changed"):
                message = self._environmental_feedback["sick_voice"]
            elif context.get("new_location"):
                message = self._environmental_feedback["new_location"]

        return VoiceFeedback(
            confidence_level=level,
            message=message,
            suggestion=suggestion,
            is_final=(level != ConfidenceLevel.LOW),
            speak_aloud=True
        )

    def generate_security_alert(
        self,
        threat: ThreatType,
        details: Optional[Dict[str, Any]] = None
    ) -> VoiceFeedback:
        """Generate security alert feedback."""
        messages = {
            ThreatType.REPLAY_ATTACK: "Security alert: I detected characteristics consistent with a recording playback. Access denied.",
            ThreatType.VOICE_CLONING: "Security alert: Voice pattern anomaly detected. This doesn't sound like natural speech.",
            ThreatType.SYNTHETIC_VOICE: "Security alert: Synthetic voice detected. Access denied for security reasons.",
            ThreatType.UNKNOWN_SPEAKER: f"I don't recognize this voice. This Mac is voice-locked to {self.user_name} only.",
            ThreatType.ENVIRONMENTAL_ANOMALY: "Something seems off about this authentication attempt. Please try again."
        }

        return VoiceFeedback(
            confidence_level=ConfidenceLevel.FAILED,
            message=messages.get(threat, "Security concern detected. Access denied."),
            suggestion="If you're the owner, please try again with a live voice command.",
            is_final=True,
            speak_aloud=True
        )


# ============================================================================
# Multi-Factor Authentication Fusion Engine
# ============================================================================

class MultiFactorAuthFusionEngine:
    """
    Fuses multiple authentication signals for robust verification.

    Factors:
    - Voice biometric (primary)
    - Behavioral patterns
    - Contextual intelligence
    - Device proximity (Apple Watch)
    - Time-based patterns
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.MultiFactor")

        # Factor weights (should sum to 1.0)
        self.weights = {
            "voice": 0.50,      # Primary voice biometric
            "behavioral": 0.20,  # Speaking patterns, timing
            "context": 0.15,    # Location, device, time
            "proximity": 0.10,  # Apple Watch, Bluetooth devices
            "history": 0.05    # Past verification history
        }

        # Minimum thresholds per factor
        self.factor_thresholds = {
            "voice": 0.60,       # Voice must be at least 60% alone
            "overall": 0.80     # Combined must reach 80%
        }

    async def fuse_factors(
        self,
        voice_confidence: float,
        behavioral_confidence: Optional[float] = None,
        context_confidence: Optional[float] = None,
        proximity_confidence: Optional[float] = None,
        history_confidence: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Fuse multiple authentication factors into a final decision.

        Returns:
            Dict with fused_confidence, decision, and factor breakdown
        """
        factors = {
            "voice": voice_confidence
        }

        # Add available factors with defaults
        factors["behavioral"] = behavioral_confidence if behavioral_confidence is not None else voice_confidence * 0.9
        factors["context"] = context_confidence if context_confidence is not None else 0.95
        factors["proximity"] = proximity_confidence if proximity_confidence is not None else 0.90
        factors["history"] = history_confidence if history_confidence is not None else 0.85

        # Calculate weighted fusion
        fused_confidence = 0.0
        total_weight = 0.0

        for factor, confidence in factors.items():
            weight = self.weights.get(factor, 0.1)
            fused_confidence += confidence * weight
            total_weight += weight

        if total_weight > 0:
            fused_confidence = fused_confidence / total_weight

        # Check minimum voice threshold
        voice_pass = factors["voice"] >= self.factor_thresholds["voice"]
        overall_pass = fused_confidence >= self.factor_thresholds["overall"]

        # Decision logic
        if voice_pass and overall_pass:
            decision = "authenticated"
        elif not voice_pass and fused_confidence >= 0.85:
            # Voice low but other factors strong - require confirmation
            decision = "requires_confirmation"
        else:
            decision = "denied"

        return {
            "fused_confidence": fused_confidence,
            "decision": decision,
            "factors": factors,
            "weights_used": self.weights,
            "voice_threshold_met": voice_pass,
            "overall_threshold_met": overall_pass
        }

    def calculate_behavioral_confidence(
        self,
        speaker_name: str,
        verification_history: List[Dict[str, Any]],
        current_time: datetime
    ) -> float:
        """Calculate behavioral confidence based on patterns."""
        if not verification_history:
            return 0.85  # Default for new users

        # Check time-of-day pattern
        current_hour = current_time.hour
        typical_hours = [h["timestamp"].hour for h in verification_history[-20:] if h.get("verified")]

        hour_match = current_hour in typical_hours or any(abs(current_hour - h) <= 2 for h in typical_hours)
        time_confidence = 0.95 if hour_match else 0.80

        # Check recent success rate
        recent = verification_history[-10:]
        success_rate = sum(1 for h in recent if h.get("verified")) / len(recent) if recent else 0.5

        # Combine
        behavioral_confidence = (time_confidence * 0.6) + (success_rate * 0.4)

        return min(1.0, behavioral_confidence)

    def calculate_context_confidence(
        self,
        current_environment: str,
        known_environments: List[str],
        last_activity_hours: float,
        failed_attempts_24h: int
    ) -> float:
        """Calculate contextual confidence."""
        confidence = 0.95  # Start high

        # Environment check
        if current_environment not in known_environments:
            confidence -= 0.10

        # Recent activity gap
        if last_activity_hours > 24:
            confidence -= 0.05

        # Failed attempts
        if failed_attempts_24h > 0:
            confidence -= min(0.20, failed_attempts_24h * 0.05)

        return max(0.50, confidence)


class SpeakerVerificationService:
    """
    Speaker verification service for JARVIS

    Verifies speaker identity using voice biometrics
    """

    def __init__(self, learning_db: Optional[JARVISLearningDatabase] = None):
        """
        Initialize speaker verification service

        Args:
            learning_db: LearningDatabase instance (optional, will create if not provided)
        """
        self.learning_db = learning_db
        self.speechbrain_engine = None
        self.initialized = False
        self.speaker_profiles = {}  # Cache of speaker profiles
        self.verification_threshold = 0.40  # 40% confidence for verification (matches owner-aware fusion threshold)
        self.legacy_threshold = 0.40  # 40% for legacy profiles with dimension mismatch
        self.profile_quality_scores = {}  # Track profile quality (1.0 = native, <1.0 = legacy)
        self._preload_thread = None
        self._encoder_preloading = False
        self._encoder_preloaded = False
        self._shutdown_event = threading.Event()  # For clean thread shutdown
        self._preload_loop = None  # Track event loop for cleanup

        # Debug mode for detailed verification logging
        self.debug_mode = True  # Enable detailed verification debugging

        # Adaptive learning tracking
        self.verification_history = {}  # Track verification attempts per speaker
        self.learning_enabled = True
        self.min_samples_for_update = 3  # Minimum attempts before adapting threshold

        # Dynamic embedding dimension detection
        self.current_model_dimension = None  # Will be detected automatically
        self.supported_dimensions = [192, 768, 96]  # Common embedding dimensions
        self.enable_auto_migration = True  # Auto-migrate incompatible profiles

        # Hot reload configuration
        self.profile_version_cache = {}  # Track profile versions/timestamps for change detection
        self.auto_reload_enabled = True  # Enable automatic profile reloading
        self.reload_check_interval = 30  # Check for updates every 30 seconds
        self._reload_task = None  # Background task for checking updates

        # ENHANCED ADAPTIVE VERIFICATION SYSTEM
        # Dynamic confidence boosting
        self.confidence_boost_enabled = True
        self.boost_multiplier = 1.5  # Boost confidence for known good patterns
        self.min_confidence_for_boost = 0.15  # Minimum confidence before boost can apply

        # Multi-stage verification
        self.multi_stage_enabled = True
        self.stage_weights = {
            'primary': 0.6,    # Main embedding comparison
            'acoustic': 0.2,   # Acoustic features (pitch, energy)
            'temporal': 0.1,   # Temporal patterns
            'adaptive': 0.1    # Historical pattern matching
        }

        # Rolling average for adaptive embeddings
        self.rolling_embeddings = {}  # Store recent successful embeddings
        self.max_rolling_samples = 10  # Keep last 10 successful verifications
        self.rolling_weight = 0.3  # Weight for new samples in rolling average

        # Dynamic calibration mode
        self.calibration_mode = False
        self.calibration_samples = []
        self.calibration_threshold = 0.10  # Very low threshold during calibration
        self.auto_calibrate_on_failure = True  # Auto-enter calibration after repeated failures
        self.failure_count = {}  # Track consecutive failures per speaker
        self.max_failures_before_calibration = 3

        # Environmental adaptation
        self.environment_profiles = {}  # Store different environment signatures
        self.current_environment = 'default'
        self.adapt_to_environment = True

        # Confidence normalization
        self.normalize_confidence = True
        self.confidence_history_window = 20  # Use last 20 attempts for normalization
        self.confidence_stats = {}  # Store mean/std for normalization

        # CONTINUOUS LEARNING SYSTEM
        # Store all voice interactions for ML training
        self.continuous_learning_enabled = True
        self.store_all_audio = True  # Store audio samples in database
        self.min_audio_quality = 0.1  # Minimum quality to store
        self.max_stored_samples_per_day = 100  # Limit daily storage
        self.audio_storage_format = 'wav'  # Store as WAV files

        # ML-based continuous improvement
        self.ml_update_frequency = 10  # Update model every N samples
        self.incremental_learning = True  # Use incremental learning
        self.embedding_update_weight = 0.1  # Weight for new samples in embedding
        self.auto_retrain_threshold = 50  # Retrain after N new samples

        # Voice sample collection
        self.voice_sample_buffer = []  # Buffer for recent samples
        self.max_buffer_size = 20  # Keep last 20 samples in memory
        self.sample_metadata = {}  # Store metadata for each sample

        # ========================================================================
        # ENHANCED AUTHENTICATION COMPONENTS (v2.0)
        # ========================================================================

        # Voice Pattern Store (ChromaDB) for behavioral biometrics
        self.voice_pattern_store = VoicePatternStore()
        self._pattern_store_initialized = False

        # Authentication Audit Trail (Langfuse)
        self.audit_trail = AuthenticationAuditTrail()
        self._audit_trail_initialized = False

        # Voice Processing Cache (Helicone-style)
        self.processing_cache = VoiceProcessingCache(
            max_size=100,
            ttl_seconds=300  # 5 minute cache
        )

        # Voice Feedback Generator
        self.feedback_generator = VoiceFeedbackGenerator(user_name="Derek")

        # Multi-Factor Authentication Fusion
        self.multi_factor_fusion = MultiFactorAuthFusionEngine()

        # TTS callback for voice feedback (set externally)
        self.tts_callback: Optional[Callable[[str], Any]] = None

        # Enhanced security settings
        self.anti_spoofing_enabled = True
        self.replay_detection_enabled = True
        self.synthetic_voice_detection = True

        # Known environments for context
        self.known_environments = ["home", "office", "default"]

    async def initialize_fast(self):
        """
        Fast initialization with background encoder pre-loading.

        Loads profiles immediately, defers SpeechBrain loading to background.
        JARVIS starts fast (~2s), encoder ready in ~10s total.
        """
        if self.initialized:
            return

        logger.info("🔐 Initializing Speaker Verification Service (fast mode)...")

        # Initialize learning database if not provided - use singleton
        if self.learning_db is None:
            from intelligence.learning_database import get_learning_database
            self.learning_db = await get_learning_database()

        # Create SpeechBrain engine but DON'T initialize it yet (deferred to background)
        model_config = ModelConfig(
            name="speechbrain-wav2vec2",
            engine=STTEngine.SPEECHBRAIN,
            disk_size_mb=380,
            ram_required_gb=2.0,
            vram_required_gb=1.8,
            expected_accuracy=0.96,
            avg_latency_ms=150,
            supports_fine_tuning=True,
            model_path="speechbrain/asr-wav2vec2-commonvoice-en",
        )

        self.speechbrain_engine = SpeechBrainEngine(model_config)
        # DON'T call initialize() here - defer to background thread

        # Load speaker profiles from database
        await self._load_speaker_profiles()

        self.initialized = True
        logger.info(
            f"✅ Speaker Verification Service ready - {len(self.speaker_profiles)} profiles loaded (encoder loading in background)"
        )

        # Start background initialization of SpeechBrain engine
        logger.info("🔄 Loading SpeechBrain encoder in background thread...")
        self._start_background_preload()

        # Initialize enhanced components in background
        asyncio.create_task(self._initialize_enhanced_components())

        # Start background profile reload monitoring
        if self.auto_reload_enabled:
            logger.info(f"🔄 Starting profile auto-reload (check every {self.reload_check_interval}s)...")
            self._reload_task = asyncio.create_task(self._profile_reload_monitor())
            logger.info("✅ Profile hot reload enabled - updates will be detected automatically")

    def _start_background_preload(self):
        """Start background thread to initialize SpeechBrain engine and pre-load speaker encoder"""
        if self._encoder_preloading or self._encoder_preloaded:
            return

        self._encoder_preloading = True

        def preload_worker():
            """Worker function to initialize engine and pre-load encoder in background thread"""
            try:
                # Run async function in thread's event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._preload_loop = loop  # Store for cleanup

                try:
                    # First initialize the SpeechBrain engine (loads models)
                    logger.info("🔄 Background: Initializing SpeechBrain engine...")
                    loop.run_until_complete(self.speechbrain_engine.initialize())
                    logger.info("✅ Background: SpeechBrain engine initialized")

                    # Then pre-load the speaker encoder
                    logger.info("🔄 Background: Pre-loading speaker encoder...")
                    loop.run_until_complete(self.speechbrain_engine._load_speaker_encoder())
                    self._encoder_preloaded = True
                    logger.info("✅ Speaker encoder ready - voice biometric unlock now instant!")
                finally:
                    # Clean shutdown of event loop
                    try:
                        # Cancel all pending tasks
                        pending = asyncio.all_tasks(loop)
                        for task in pending:
                            task.cancel()

                        # Wait for tasks to finish cancellation
                        if pending:
                            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

                        # Close the loop
                        loop.close()
                    except Exception as cleanup_error:
                        logger.debug(f"Event loop cleanup: {cleanup_error}")
                    finally:
                        self._preload_loop = None

            except Exception as e:
                logger.error(f"Background encoder pre-loading failed: {e}", exc_info=True)
            finally:
                self._encoder_preloading = False

        self._preload_thread = threading.Thread(
            target=preload_worker,
            daemon=True,
            name="SpeakerEncoderPreloader"  # Give it a descriptive name
        )
        self._preload_thread.start()

    async def _initialize_enhanced_components(self):
        """
        Initialize enhanced authentication components (v2.0).

        Initializes:
        - ChromaDB voice pattern store
        - Langfuse audit trail
        - Multi-factor fusion engine
        """
        try:
            logger.info("🚀 Initializing enhanced authentication components...")

            # Initialize Voice Pattern Store (ChromaDB)
            if CHROMADB_AVAILABLE:
                await self.voice_pattern_store.initialize()
                self._pattern_store_initialized = True
                logger.info("✅ Voice pattern store (ChromaDB) initialized")

            # Initialize Audit Trail (Langfuse)
            await self.audit_trail.initialize()
            self._audit_trail_initialized = True
            logger.info("✅ Authentication audit trail initialized")

            # Update feedback generator with primary user name
            primary_user = None
            for name, profile in self.speaker_profiles.items():
                if profile.get("is_primary_user"):
                    primary_user = name
                    break

            if primary_user:
                self.feedback_generator.user_name = primary_user
                logger.info(f"✅ Voice feedback configured for {primary_user}")

            logger.info("🎉 Enhanced authentication components ready!")

        except Exception as e:
            logger.error(f"❌ Failed to initialize enhanced components: {e}", exc_info=True)

    async def verify_speaker_enhanced(
        self,
        audio_data: bytes,
        speaker_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enhanced speaker verification with multi-factor fusion and audit trail.

        This is the v2.0 verification method that provides:
        - Full audit trail via Langfuse
        - Multi-factor authentication fusion
        - Anti-spoofing detection
        - Progressive voice feedback
        - Intelligent caching

        Args:
            audio_data: Audio bytes (WAV format)
            speaker_name: Expected speaker name (if None, identifies from all profiles)
            context: Additional context (environment, device, etc.)

        Returns:
            Enhanced verification result with full trace
        """
        if not self.initialized:
            await self.initialize()

        context = context or {}
        start_time = time.time()

        # Start audit trace
        trace_id = self.audit_trail.start_trace(
            speaker_name=speaker_name or "unknown",
            environment=context.get("environment", self.current_environment)
        )

        try:
            # Phase 1: Check cache for recent identical audio
            cache_key = "verification"
            cached_result = self.processing_cache.get(audio_data, cache_key)
            if cached_result:
                self.audit_trail.log_phase(
                    trace_id, AuthenticationPhase.SPEAKER_VERIFICATION,
                    duration_ms=0.1,
                    metrics={"cached": True, "confidence": cached_result.get("confidence", 0)}
                )
                logger.info("🔄 Using cached verification result")
                return cached_result

            # Phase 2: Audio quality analysis
            phase_start = time.time()
            audio_quality = await self._calculate_audio_quality(audio_data)
            snr_db = self._estimate_snr(np.frombuffer(audio_data[:min(4000, len(audio_data))], dtype=np.int16).astype(np.float32) / 32768.0)

            self.audit_trail.log_phase(
                trace_id, AuthenticationPhase.AUDIO_CAPTURE,
                duration_ms=(time.time() - phase_start) * 1000,
                metrics={"quality_score": audio_quality, "snr_db": snr_db}
            )

            # Phase 3: Anti-spoofing checks
            threat_detected = ThreatType.NONE
            if self.anti_spoofing_enabled and self._pattern_store_initialized:
                phase_start = time.time()

                # Generate audio fingerprint for replay detection
                audio_fingerprint = hashlib.sha256(audio_data).hexdigest()

                if self.replay_detection_enabled:
                    is_replay, anomaly_score = await self.voice_pattern_store.detect_replay_attack(
                        audio_fingerprint,
                        speaker_name or "unknown"
                    )
                    if is_replay:
                        threat_detected = ThreatType.REPLAY_ATTACK
                        logger.warning(f"⚠️ REPLAY ATTACK DETECTED for {speaker_name}")

                self.audit_trail.log_phase(
                    trace_id, AuthenticationPhase.ANTI_SPOOFING,
                    duration_ms=(time.time() - phase_start) * 1000,
                    metrics={"threat": threat_detected.value, "fingerprint": audio_fingerprint[:16]}
                )

                if threat_detected != ThreatType.NONE:
                    # Generate security alert feedback
                    feedback = self.feedback_generator.generate_security_alert(threat_detected)
                    if self.tts_callback:
                        await self._speak_feedback(feedback)

                    self.audit_trail.complete_trace(trace_id, "denied", 0.0, threat_detected)
                    return {
                        "verified": False,
                        "confidence": 0.0,
                        "speaker_name": speaker_name,
                        "threat_detected": threat_detected.value,
                        "feedback": feedback.message,
                        "trace_id": trace_id
                    }

            # Phase 4: Core speaker verification (existing logic)
            phase_start = time.time()
            base_result = await self.verify_speaker(audio_data, speaker_name)
            voice_confidence = base_result.get("confidence", 0.0)

            self.audit_trail.log_phase(
                trace_id, AuthenticationPhase.SPEAKER_VERIFICATION,
                duration_ms=(time.time() - phase_start) * 1000,
                metrics={"confidence": voice_confidence, "verified": base_result.get("verified", False)}
            )

            # Phase 5: Multi-factor fusion
            phase_start = time.time()

            # Calculate behavioral confidence
            behavioral_confidence = self.multi_factor_fusion.calculate_behavioral_confidence(
                speaker_name or "unknown",
                self.verification_history.get(speaker_name, []),
                datetime.now()
            )

            # Calculate context confidence
            last_activity_hours = 0
            if speaker_name in self.verification_history:
                history = self.verification_history[speaker_name]
                if history:
                    last_ts = datetime.fromisoformat(history[-1].get("timestamp", datetime.now().isoformat()))
                    last_activity_hours = (datetime.now() - last_ts).total_seconds() / 3600

            context_confidence = self.multi_factor_fusion.calculate_context_confidence(
                context.get("environment", self.current_environment),
                self.known_environments,
                last_activity_hours,
                self.failure_count.get(speaker_name, 0)
            )

            # Fuse all factors
            fusion_result = await self.multi_factor_fusion.fuse_factors(
                voice_confidence=voice_confidence,
                behavioral_confidence=behavioral_confidence,
                context_confidence=context_confidence,
                proximity_confidence=context.get("proximity_confidence"),
                history_confidence=None
            )

            self.audit_trail.log_phase(
                trace_id, AuthenticationPhase.MULTI_FACTOR_FUSION,
                duration_ms=(time.time() - phase_start) * 1000,
                metrics={
                    "fused_confidence": fusion_result["fused_confidence"],
                    "factors": fusion_result["factors"],
                    "decision": fusion_result["decision"]
                }
            )

            # Phase 6: Final decision
            final_confidence = fusion_result["fused_confidence"]
            is_verified = fusion_result["decision"] == "authenticated"

            # Generate voice feedback
            feedback_context = {
                "snr_db": snr_db,
                "hour": datetime.now().hour,
                "voice_changed": voice_confidence < 0.70 and behavioral_confidence > 0.85,
                "new_location": context.get("environment") not in self.known_environments
            }
            feedback = self.feedback_generator.generate_feedback(final_confidence, feedback_context)

            if self.tts_callback:
                await self._speak_feedback(feedback)

            # Store audio fingerprint for future replay detection
            if is_verified and self._pattern_store_initialized:
                audio_fingerprint = hashlib.sha256(audio_data).hexdigest()
                try:
                    embedding = await self.speechbrain_engine.extract_speaker_embedding(audio_data)
                    await self.voice_pattern_store.store_audio_fingerprint(
                        speaker_name or "unknown",
                        audio_fingerprint,
                        embedding
                    )
                except Exception as e:
                    logger.debug(f"Failed to store fingerprint: {e}")

            # Complete trace
            decision = "authenticated" if is_verified else "denied"
            trace = self.audit_trail.complete_trace(trace_id, decision, final_confidence, threat_detected)

            # Build result
            result = {
                "verified": is_verified,
                "confidence": final_confidence,
                "voice_confidence": voice_confidence,
                "behavioral_confidence": behavioral_confidence,
                "context_confidence": context_confidence,
                "speaker_name": speaker_name or base_result.get("speaker_name"),
                "speaker_id": base_result.get("speaker_id"),
                "is_owner": base_result.get("is_owner", False),
                "security_level": base_result.get("security_level", "standard"),
                "feedback": {
                    "message": feedback.message,
                    "level": feedback.confidence_level.value,
                    "suggestion": feedback.suggestion
                },
                "trace_id": trace_id,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "cache_stats": self.processing_cache.get_stats()
            }

            # Cache successful results
            if is_verified:
                self.processing_cache.set(audio_data, cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Enhanced verification failed: {e}", exc_info=True)
            self.audit_trail.complete_trace(trace_id, "error", 0.0)
            return {
                "verified": False,
                "confidence": 0.0,
                "speaker_name": speaker_name,
                "error": str(e),
                "trace_id": trace_id
            }

    async def _speak_feedback(self, feedback: VoiceFeedback):
        """Speak voice feedback via TTS callback."""
        if self.tts_callback and feedback.speak_aloud:
            try:
                if asyncio.iscoroutinefunction(self.tts_callback):
                    await self.tts_callback(feedback.message)
                else:
                    self.tts_callback(feedback.message)
            except Exception as e:
                logger.debug(f"TTS feedback failed: {e}")

    def set_tts_callback(self, callback: Callable[[str], Any]):
        """Set the TTS callback for voice feedback."""
        self.tts_callback = callback
        logger.info("✅ TTS callback configured for voice feedback")

    def get_authentication_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed authentication trace for debugging/display."""
        trace = self.audit_trail.get_trace(trace_id)
        if trace:
            return trace.to_dict()
        return None

    def get_recent_authentications(
        self,
        speaker_name: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get recent authentication attempts with full traces."""
        traces = self.audit_trail.get_recent_traces(speaker_name, limit)
        return [t.to_dict() for t in traces]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get voice processing cache statistics."""
        return self.processing_cache.get_stats()

    async def initialize(self, preload_encoder: bool = True):
        """
        Initialize service and load speaker profiles

        Args:
            preload_encoder: If True, pre-loads ECAPA-TDNN encoder during initialization
                           for instant unlock (adds ~10s to startup, but unlock is instant)
        """
        if self.initialized:
            return

        logger.info("🔐 Initializing Speaker Verification Service...")

        # Initialize learning database if not provided - use singleton
        if self.learning_db is None:
            from intelligence.learning_database import get_learning_database
            self.learning_db = await get_learning_database()

        # Initialize SpeechBrain engine for embeddings
        model_config = ModelConfig(
            name="speechbrain-wav2vec2",
            engine=STTEngine.SPEECHBRAIN,
            disk_size_mb=380,
            ram_required_gb=2.0,
            vram_required_gb=1.8,
            expected_accuracy=0.96,
            avg_latency_ms=150,
            supports_fine_tuning=True,
            model_path="speechbrain/asr-wav2vec2-commonvoice-en",
        )

        self.speechbrain_engine = SpeechBrainEngine(model_config)
        await self.speechbrain_engine.initialize()

        # Load speaker profiles from database
        await self._load_speaker_profiles()

        # Pre-load speaker encoder for instant unlock (if enabled)
        if preload_encoder:
            logger.info("🔄 Pre-loading speaker encoder (ECAPA-TDNN) for instant unlock...")
            await self.speechbrain_engine._load_speaker_encoder()
            logger.info("✅ Speaker encoder pre-loaded - unlock will be instant!")

        self.initialized = True
        logger.info(
            f"✅ Speaker Verification Service ready ({len(self.speaker_profiles)} profiles loaded)"
        )

    async def _detect_current_model_dimension(self):
        """
        Detect the current model's embedding dimension dynamically.
        Makes system adaptive to any model without hardcoding dimensions.
        """
        if self.current_model_dimension is not None:
            return self.current_model_dimension

        try:
            logger.info("🔍 Detecting current model embedding dimension...")

            # Create realistic test audio (pink noise for better model response)
            # Pink noise has more speech-like frequency distribution than white noise
            duration = 1.0  # 1 second
            sample_rate = 16000
            num_samples = int(duration * sample_rate)

            # Generate pink noise (1/f spectrum)
            white_noise = np.random.randn(num_samples).astype(np.float32)
            # Apply simple 1/f filter
            fft = np.fft.rfft(white_noise)
            frequencies = np.fft.rfftfreq(num_samples, 1/sample_rate)
            # Avoid division by zero
            pink_filter = 1 / np.sqrt(frequencies + 1)
            fft_filtered = fft * pink_filter
            pink_noise = np.fft.irfft(fft_filtered, num_samples).astype(np.float32)

            # Normalize to prevent clipping
            pink_noise = pink_noise / np.max(np.abs(pink_noise)) * 0.3

            # Convert to bytes
            test_audio_bytes = (pink_noise * 32767).astype(np.int16).tobytes()

            # Extract test embedding
            test_embedding = await self.speechbrain_engine.extract_speaker_embedding(test_audio_bytes)

            # Handle 2D embeddings (batch dimension)
            if test_embedding.ndim == 2:
                # Shape is (1, dim) - get the actual dimension
                self.current_model_dimension = test_embedding.shape[1]
                logger.info(f"🔍 Detected 2D embedding shape: {test_embedding.shape}, using dimension: {self.current_model_dimension}D")
            else:
                # Shape is (dim,)
                self.current_model_dimension = test_embedding.shape[0]
                logger.info(f"🔍 Detected 1D embedding shape: {test_embedding.shape}, dimension: {self.current_model_dimension}D")

            logger.info(f"✅ Current model dimension: {self.current_model_dimension}D")

            # Validate dimension is reasonable
            if self.current_model_dimension < 10:
                logger.warning(f"⚠️  Detected dimension ({self.current_model_dimension}D) seems too small, using fallback")
                self.current_model_dimension = 192

            return self.current_model_dimension

        except Exception as e:
            logger.error(f"❌ Failed to detect model dimension: {e}", exc_info=True)
            self.current_model_dimension = 192  # Fallback
            logger.warning(f"⚠️  Using fallback dimension: {self.current_model_dimension}D")
            return self.current_model_dimension

    async def _migrate_embedding_dimension(self, embedding: np.ndarray, target_dim: int) -> np.ndarray:
        """
        Intelligently migrate embedding from one dimension to another.

        Uses adaptive techniques:
        - Upsampling: PCA + learned projection + interpolation
        - Downsampling: PCA for dimensionality reduction
        - Zero-padding: Simple extension for small differences

        Args:
            embedding: Source embedding
            target_dim: Target dimension

        Returns:
            Migrated embedding with target dimension
        """
        source_dim = embedding.shape[0]

        if source_dim == target_dim:
            return embedding

        logger.info(f"🔄 Migrating embedding: {source_dim}D → {target_dim}D")

        try:
            # Case 1: Upsample (e.g., 96D → 192D or 768D → 192D is actually downsample)
            if target_dim > source_dim:
                # Method: Interpolation + learned pattern repetition
                ratio = target_dim / source_dim

                if ratio == int(ratio):
                    # Perfect multiple - replicate with variation
                    migrated = np.repeat(embedding, int(ratio))
                    # Add slight variation to avoid perfect duplication
                    noise = np.random.randn(target_dim) * 0.01 * np.std(embedding)
                    migrated = migrated + noise
                else:
                    # Non-integer ratio - use interpolation
                    from scipy import interpolate
                    x_old = np.linspace(0, 1, source_dim)
                    x_new = np.linspace(0, 1, target_dim)
                    f = interpolate.interp1d(x_old, embedding, kind='cubic', fill_value='extrapolate')
                    migrated = f(x_new)

            # Case 2: Downsample (e.g., 768D → 192D or 96D → 192D is actually upsample)
            else:
                # Method: PCA or averaging
                ratio = source_dim / target_dim

                if ratio == int(ratio):
                    # Perfect divisor - use averaging
                    migrated = embedding.reshape(target_dim, int(ratio)).mean(axis=1)
                else:
                    # Non-integer ratio - use PCA-like reduction
                    # Simple reshaping with truncation
                    from scipy import signal
                    migrated = signal.resample(embedding, target_dim)

            # Normalize to maintain magnitude
            migrated = migrated.astype(np.float64)
            original_norm = np.linalg.norm(embedding)
            migrated_norm = np.linalg.norm(migrated)
            if migrated_norm > 0:
                migrated = migrated * (original_norm / migrated_norm)

            logger.info(f"✅ Migration complete: shape={migrated.shape}, norm={np.linalg.norm(migrated):.4f}")
            return migrated

        except Exception as e:
            logger.error(f"❌ Migration failed: {e}", exc_info=True)
            # Fallback: zero-padding or truncation
            if target_dim > source_dim:
                migrated = np.pad(embedding, (0, target_dim - source_dim), mode='edge')
            else:
                migrated = embedding[:target_dim]
            return migrated.astype(np.float64)

    async def _reconstruct_embedding_from_samples(self, speaker_name: str, speaker_id: int) -> np.ndarray:
        """
        Reconstruct a proper embedding using original audio samples from database.

        This solves the re-enrollment problem by using existing voice data
        to generate fresh embeddings with the current model.

        Args:
            speaker_name: Name of speaker
            speaker_id: Database ID of speaker

        Returns:
            New embedding with current model dimension
        """
        try:
            logger.info(f"🔄 Attempting to reconstruct embedding from audio samples for {speaker_name}")

            if not self.learning_db:
                logger.warning("No database connection for sample reconstruction")
                return None

            # Try to get original audio samples from database
            samples = await self.learning_db.get_voice_samples_for_speaker(speaker_id)

            if not samples or len(samples) == 0:
                logger.info(f"No audio samples found for {speaker_name}, trying alternate methods...")
                return None

            logger.info(f"Found {len(samples)} audio samples for {speaker_name}")

            # Extract embeddings from each sample using current model
            embeddings = []
            samples_with_audio = [s for s in samples[:10] if s.get("audio_data")]

            if not samples_with_audio:
                logger.warning(
                    f"No audio_data found in voice samples for {speaker_name}. "
                    f"This is expected for profiles created before audio storage was enabled. "
                    f"Will use fallback migration methods (padding/truncation)."
                )
                return None

            logger.info(f"Found {len(samples_with_audio)} samples with audio data")

            for sample in samples_with_audio:
                try:
                    audio_data = sample.get("audio_data")
                    if audio_data:
                        # Extract embedding with current model
                        embedding = await self.speechbrain_engine.extract_speaker_embedding(audio_data)
                        if embedding.shape[0] == self.current_model_dimension:
                            embeddings.append(embedding)
                            logger.info(f"  ✓ Extracted {embedding.shape[0]}D embedding from sample")
                except Exception as e:
                    logger.debug(f"Failed to extract from sample: {e}")
                    continue

            if len(embeddings) == 0:
                logger.warning(f"Could not extract any valid embeddings for {speaker_name}")
                return None

            # Average the embeddings for a robust representation
            avg_embedding = np.mean(embeddings, axis=0)
            logger.info(f"✅ Reconstructed {avg_embedding.shape[0]}D embedding from {len(embeddings)} samples")

            return avg_embedding

        except Exception as e:
            logger.error(f"Failed to reconstruct embedding for {speaker_name}: {type(e).__name__}: {e}", exc_info=True)
            return None

    async def _create_multi_model_profile(self, profile: dict, speaker_name: str) -> dict:
        """
        Create a universal profile that works across different models.

        Stores multiple embeddings for different model dimensions,
        allowing seamless model switching without re-enrollment.

        Args:
            profile: Original profile dict
            speaker_name: Name of speaker

        Returns:
            Enhanced profile with multi-model support
        """
        try:
            logger.info(f"🌐 Creating multi-model profile for {speaker_name}")

            # Store embeddings for multiple dimensions
            multi_embeddings = {}

            # Get original embedding
            embedding_bytes = profile.get("voiceprint_embedding")
            if embedding_bytes:
                # Fix: Use float32 as embeddings are stored as float32
                original_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                original_dim = original_embedding.shape[0]
                multi_embeddings[original_dim] = original_embedding

            # Try to reconstruct for current model
            speaker_id = profile.get("speaker_id")
            if speaker_id and self.current_model_dimension not in multi_embeddings:
                reconstructed = await self._reconstruct_embedding_from_samples(speaker_name, speaker_id)
                if reconstructed is not None:
                    multi_embeddings[self.current_model_dimension] = reconstructed

            # If we still don't have the right dimension, use smart migration
            if self.current_model_dimension not in multi_embeddings:
                # Use cross-model transfer learning
                best_source_dim = min(multi_embeddings.keys(),
                                     key=lambda x: abs(x - self.current_model_dimension))
                source_embedding = multi_embeddings[best_source_dim]

                # Apply intelligent migration with cross-model compensation
                migrated = await self._cross_model_migration(
                    source_embedding,
                    source_dim=best_source_dim,
                    target_dim=self.current_model_dimension
                )
                multi_embeddings[self.current_model_dimension] = migrated

            # Update profile with multi-model support
            profile["multi_embeddings"] = multi_embeddings
            profile["supported_dimensions"] = list(multi_embeddings.keys())

            # Use the embedding for current model
            if self.current_model_dimension in multi_embeddings:
                profile["voiceprint_embedding"] = multi_embeddings[self.current_model_dimension].tobytes()
                profile["embedding_dimension"] = self.current_model_dimension
                logger.info(f"✅ Multi-model profile ready with dimensions: {profile['supported_dimensions']}")

            return profile

        except Exception as e:
            logger.error(f"Failed to create multi-model profile: {e}")
            return profile

    async def _cross_model_migration(self, embedding: np.ndarray, source_dim: int, target_dim: int) -> np.ndarray:
        """
        Advanced cross-model migration using transfer learning principles.

        This handles the case where embeddings come from fundamentally different models,
        not just different dimensions of the same model.

        Args:
            embedding: Source embedding
            source_dim: Source dimension
            target_dim: Target dimension

        Returns:
            Migrated embedding optimized for cross-model compatibility
        """
        logger.info(f"🔀 Cross-model migration: {source_dim}D → {target_dim}D")

        # Apply model-specific transformations
        if source_dim == 1536 and target_dim == 192:
            # Large model (1536D) to ECAPA-TDNN (192D)
            # Use advanced dimension reduction preserving speaker characteristics

            # Method 1: Intelligent downsampling with feature preservation
            # Divide into 8 segments (1536/192 = 8)
            segment_size = source_dim // target_dim  # 8

            # Extract key features from each segment
            features = []
            for i in range(target_dim):
                segment = embedding[i*segment_size:(i+1)*segment_size]
                # Take weighted combination: mean + variance information
                feature = np.mean(segment) * 0.7 + np.std(segment) * 0.3
                features.append(feature)

            features = np.array(features)

            # Preserve energy distribution
            orig_norm = np.linalg.norm(embedding)
            features = features / (np.linalg.norm(features) + 1e-10) * orig_norm

            logger.info(f"✅ Reduced 1536D → 192D preserving speaker characteristics")
            return features

        elif source_dim == 768 and target_dim == 192:
            # Likely transformer (768D) to ECAPA-TDNN (192D)
            # Use PCA-like dimensionality reduction with emphasis on speaker-discriminative features

            # Reshape to blocks and take statistics
            num_blocks = 4
            block_size = source_dim // num_blocks
            blocks = embedding.reshape(num_blocks, block_size)

            # Extract features from each block
            features = []
            for block in blocks:
                features.extend([
                    np.mean(block),
                    np.std(block),
                    np.max(block),
                    np.min(block),
                    np.median(block)
                ])

            # Pad or truncate to target dimension
            features = np.array(features)
            if len(features) < target_dim:
                # Pad with computed statistics
                padding = target_dim - len(features)
                features = np.pad(features, (0, padding), mode='wrap')
            else:
                features = features[:target_dim]

            # Normalize to maintain speaker characteristics
            features = features / (np.linalg.norm(features) + 1e-10) * np.linalg.norm(embedding)
            return features

        elif source_dim == 96 and target_dim == 192:
            # Likely smaller model to ECAPA-TDNN
            # Use harmonic expansion to preserve speaker patterns

            # Duplicate and add harmonics
            base = np.repeat(embedding, 2)  # 96 → 192

            # Add frequency-domain variations
            fft = np.fft.rfft(embedding)
            harmonics = np.fft.irfft(fft * 1.1, target_dim)  # Slight frequency shift

            # Blend base and harmonics
            migrated = 0.8 * base + 0.2 * harmonics

            # Normalize
            migrated = migrated / (np.linalg.norm(migrated) + 1e-10) * np.linalg.norm(embedding)
            return migrated

        else:
            # Generic cross-model migration
            return await self._migrate_embedding_dimension(embedding, target_dim)

    async def _auto_migrate_profile(self, profile: dict, speaker_name: str) -> dict:
        """
        Automatically migrate profile using smart reconstruction.

        Tries in order:
        1. Reconstruct from original audio samples
        2. Create multi-model profile
        3. Cross-model migration
        4. Simple dimension migration

        Args:
            profile: Profile dict with embedding
            speaker_name: Name of speaker

        Returns:
            Updated profile dict with migrated embedding
        """
        try:
            embedding_bytes = profile.get("voiceprint_embedding")
            if not embedding_bytes:
                return profile

            # Fix: Use float32 as embeddings are stored as float32
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            source_dim = embedding.shape[0]

            if source_dim == self.current_model_dimension:
                return profile  # Already correct dimension

            logger.info(
                f"🔄 Smart migration for {speaker_name}: "
                f"{source_dim}D → {self.current_model_dimension}D"
            )

            # Method 1: Try to reconstruct from audio samples (best)
            speaker_id = profile.get("speaker_id")
            if speaker_id:
                reconstructed = await self._reconstruct_embedding_from_samples(speaker_name, speaker_id)
                if reconstructed is not None:
                    profile["voiceprint_embedding"] = reconstructed.tobytes()
                    profile["embedding_dimension"] = self.current_model_dimension
                    profile["migration_method"] = "audio_reconstruction"
                    logger.info(f"✅ {speaker_name} reconstructed from audio samples")
                    return profile

            # Method 2: Try multi-model profile (good)
            enhanced_profile = await self._create_multi_model_profile(profile, speaker_name)
            if enhanced_profile.get("multi_embeddings"):
                logger.info(f"✅ {speaker_name} using multi-model profile")
                return enhanced_profile

            # Method 3: Use cross-model migration (acceptable)
            migrated_embedding = await self._cross_model_migration(
                embedding,
                source_dim=source_dim,
                target_dim=self.current_model_dimension
            )

            # Update profile with migrated embedding
            profile["voiceprint_embedding"] = migrated_embedding.tobytes()
            profile["embedding_dimension"] = self.current_model_dimension
            profile["migration_method"] = "cross_model"
            profile["original_dimension"] = source_dim

            # Update database asynchronously
            asyncio.create_task(self._update_profile_in_database(profile, speaker_name))

            logger.info(f"✅ {speaker_name} migrated via cross-model transfer")
            return profile

        except Exception as e:
            logger.error(f"❌ Smart migration failed for {speaker_name}: {e}", exc_info=True)
            return profile

    async def _update_profile_in_database(self, profile: dict, speaker_name: str):
        """
        Update migrated profile in database (async background task).

        Args:
            profile: Updated profile dict
            speaker_name: Name of speaker
        """
        try:
            if not self.learning_db:
                return

            speaker_id = profile.get("speaker_id")
            if not speaker_id:
                return

            logger.info(f"💾 Updating {speaker_name} profile in database...")

            # Update the profile in database
            await self.learning_db.update_speaker_profile(
                speaker_id=speaker_id,
                voiceprint_embedding=profile["voiceprint_embedding"],
                metadata={
                    "migration_applied": True,
                    "original_dimension": profile.get("original_dimension"),
                    "current_dimension": profile.get("embedding_dimension"),
                    "migrated_at": datetime.now().isoformat()
                }
            )

            logger.info(f"✅ {speaker_name} profile updated in database")

        except Exception as e:
            logger.warning(f"⚠️  Failed to update {speaker_name} in database: {e}")
            # Non-critical - migration is already in memory

    def _select_best_profile_for_speaker(self, profiles_for_speaker: list) -> dict:
        """
        Intelligently select best profile when duplicates exist.

        Prioritizes:
        1. Native dimension matching current model (100 pts)
        2. Higher total_samples (up to 50 pts)
        3. Primary user status (30 pts)
        4. Security level (up to 20 pts)
        """
        if len(profiles_for_speaker) == 1:
            return profiles_for_speaker[0]

        logger.info(f"🔍 Found {len(profiles_for_speaker)} profiles, selecting best...")

        scored_profiles = []
        for profile in profiles_for_speaker:
            score = 0
            embedding_bytes = profile.get("voiceprint_embedding")
            if not embedding_bytes:
                continue

            try:
                # Always use float32 - embeddings are stored as float32
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                dimension = embedding.shape[0]

                # Dimension match - highest priority
                if dimension == self.current_model_dimension:
                    score += 100
                    logger.info(f"   ✓ {profile.get('speaker_name')} matches dimension ({dimension}D)")
                else:
                    score -= 50
                    logger.info(f"   ✗ {profile.get('speaker_name')} dimension mismatch ({dimension}D vs {self.current_model_dimension}D)")

                # Total samples
                total_samples = profile.get("total_samples", 0)
                score += min(50, total_samples // 2)

                # Primary user
                if profile.get("is_primary_user", False):
                    score += 30

                # Security level
                security = profile.get("security_level", "standard")
                if security == "admin":
                    score += 20
                elif security == "high":
                    score += 15

                scored_profiles.append((score, profile, dimension))

            except Exception as e:
                logger.warning(f"⚠️  Error scoring profile: {e}")
                continue

        if not scored_profiles:
            return profiles_for_speaker[0]

        scored_profiles.sort(key=lambda x: x[0], reverse=True)
        best_score, best_profile, best_dim = scored_profiles[0]

        logger.info(
            f"✅ Selected: {best_profile.get('speaker_name')} "
            f"(score: {best_score}, {best_dim}D, {best_profile.get('total_samples', 0)} samples)"
        )

        return best_profile

    async def _load_speaker_profiles(self):
        """
        Load speaker profiles with intelligent duplicate handling.

        Enhanced features:
        1. Auto-detects current model dimension
        2. Groups profiles by speaker name
        3. Selects best profile per speaker
        4. Validates embeddings
        5. Sets adaptive thresholds
        """
        loaded_count = 0
        skipped_count = 0

        try:
            logger.info("🔄 Loading speaker profiles from database...")

            # Use singleton to get the shared database instance
            from intelligence.learning_database import get_learning_database
            self.learning_db = await get_learning_database()

            # Verify database connection
            if not self.learning_db or not self.learning_db._initialized:
                logger.error("❌ Learning database not initialized")
                raise RuntimeError("Learning database not initialized")

            # Detect current model dimension
            await self._detect_current_model_dimension()

            profiles = await self.learning_db.get_all_speaker_profiles()
            logger.info(f"📊 Found {len(profiles)} profile(s) in database")

            # Group profiles by speaker name
            profiles_by_speaker = {}
            for profile in profiles:
                speaker_name = profile.get("speaker_name")
                if speaker_name:
                    if speaker_name not in profiles_by_speaker:
                        profiles_by_speaker[speaker_name] = []
                    profiles_by_speaker[speaker_name].append(profile)

            logger.info(f"📊 Found {len(profiles_by_speaker)} unique speaker(s)")

            # Process each speaker
            for speaker_name, speaker_profiles in profiles_by_speaker.items():
                try:
                    # Select best profile if duplicates
                    if len(speaker_profiles) > 1:
                        logger.info(f"⚠️  {len(speaker_profiles)} profiles for {speaker_name}, selecting best...")
                        profile = self._select_best_profile_for_speaker(speaker_profiles)
                    else:
                        profile = speaker_profiles[0]

                    # Auto-migrate profile if dimension mismatch and auto-migration enabled
                    if self.enable_auto_migration:
                        embedding_bytes = profile.get("voiceprint_embedding")
                        if embedding_bytes:
                            # Fix: Use float32 as embeddings are stored as float32
                            test_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                            logger.info(f"🔍 DEBUG: {speaker_name} - Current: {test_embedding.shape[0]}D, Model: {self.current_model_dimension}D")
                            if test_embedding.shape[0] != self.current_model_dimension:
                                logger.warning(f"🔄 DIMENSION MISMATCH: {speaker_name} has {test_embedding.shape[0]}D but model expects {self.current_model_dimension}D")
                                logger.info(f"🔄 Starting auto-migration for {speaker_name}...")
                                profile = await self._auto_migrate_profile(profile, speaker_name)
                                logger.info(f"✅ Auto-migration completed for {speaker_name}")
                            else:
                                logger.info(f"✅ {speaker_name} embedding dimension matches model ({self.current_model_dimension}D)")

                    # Process the selected profile
                    speaker_id = profile.get("speaker_id")

                    # Validate required fields
                    if not speaker_id or not speaker_name:
                        logger.warning(f"⚠️ Skipping invalid profile: missing speaker_id or speaker_name")
                        skipped_count += 1
                        continue

                    # Deserialize embedding
                    embedding_bytes = profile.get("voiceprint_embedding")
                    if not embedding_bytes:
                        logger.warning(f"⚠️ Speaker profile {speaker_name} has no embedding - skipping")
                        skipped_count += 1
                        continue

                    # Validate embedding data
                    try:
                        # Always use float32 - embeddings are stored as float32
                        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    except Exception as deserialize_error:
                        logger.error(f"❌ Failed to deserialize embedding for {speaker_name}: {deserialize_error}")
                        skipped_count += 1
                        continue

                    # Validate embedding dimension
                    if embedding.shape[0] == 0:
                        logger.warning(f"⚠️ Speaker profile {speaker_name} has empty embedding - skipping")
                        skipped_count += 1
                        continue

                    # Assess profile quality - NOW DYNAMIC!
                    is_native = embedding.shape[0] == self.current_model_dimension
                    total_samples = profile.get("total_samples", 0)

                    # Determine quality and threshold
                    if is_native and total_samples >= 100:
                        quality = "excellent"
                        threshold = self.verification_threshold
                    elif is_native and total_samples >= 50:
                        quality = "good"
                        threshold = self.verification_threshold
                    elif total_samples >= 50:
                        quality = "fair"
                        threshold = self.legacy_threshold
                    else:
                        quality = "legacy"
                        threshold = self.legacy_threshold

                    if self.debug_mode:
                        logger.info(f"  🔍 DEBUG Profile '{speaker_name}':")
                        logger.info(f"     - Embedding dimension: {embedding.shape[0]} (Model: {self.current_model_dimension})")
                        logger.info(f"     - Is native model: {is_native}")
                        logger.info(f"     - Total samples: {total_samples}")
                        logger.info(f"     - Quality rating: {quality}")
                        logger.info(f"     - Assigned threshold: {threshold:.2%}")
                        logger.info(f"     - Created: {profile.get('created_at', 'unknown')}")

                    # Store profile with comprehensive acoustic features
                    self.speaker_profiles[speaker_name] = {
                        "speaker_id": speaker_id,
                        "embedding": embedding,
                        "embedding_dimension": profile.get("embedding_dimension", embedding.shape[0]),
                        "confidence": profile.get("recognition_confidence", 0.0),
                        "is_primary_user": profile.get("is_primary_user", False),
                        "security_level": profile.get("security_level", "standard"),
                        "total_samples": total_samples,
                        "is_native": is_native,
                        "quality": quality,
                        "threshold": threshold,

                        # 🔬 Acoustic biometric features
                        "acoustic_features": {
                            # Pitch
                            "pitch_mean_hz": profile.get("pitch_mean_hz"),
                            "pitch_std_hz": profile.get("pitch_std_hz"),
                            "pitch_range_hz": profile.get("pitch_range_hz"),
                            "pitch_min_hz": profile.get("pitch_min_hz"),
                            "pitch_max_hz": profile.get("pitch_max_hz"),

                            # Formants
                            "formant_f1_hz": profile.get("formant_f1_hz"),
                            "formant_f1_std": profile.get("formant_f1_std"),
                            "formant_f2_hz": profile.get("formant_f2_hz"),
                            "formant_f2_std": profile.get("formant_f2_std"),
                            "formant_f3_hz": profile.get("formant_f3_hz"),
                            "formant_f3_std": profile.get("formant_f3_std"),
                            "formant_f4_hz": profile.get("formant_f4_hz"),
                            "formant_f4_std": profile.get("formant_f4_std"),

                            # Spectral
                            "spectral_centroid_hz": profile.get("spectral_centroid_hz"),
                            "spectral_centroid_std": profile.get("spectral_centroid_std"),
                            "spectral_rolloff_hz": profile.get("spectral_rolloff_hz"),
                            "spectral_rolloff_std": profile.get("spectral_rolloff_std"),
                            "spectral_flux": profile.get("spectral_flux"),
                            "spectral_flux_std": profile.get("spectral_flux_std"),
                            "spectral_entropy": profile.get("spectral_entropy"),
                            "spectral_entropy_std": profile.get("spectral_entropy_std"),
                            "spectral_flatness": profile.get("spectral_flatness"),
                            "spectral_bandwidth_hz": profile.get("spectral_bandwidth_hz"),

                            # Temporal
                            "speaking_rate_wpm": profile.get("speaking_rate_wpm"),
                            "speaking_rate_std": profile.get("speaking_rate_std"),
                            "pause_ratio": profile.get("pause_ratio"),
                            "pause_ratio_std": profile.get("pause_ratio_std"),
                            "syllable_rate": profile.get("syllable_rate"),
                            "articulation_rate": profile.get("articulation_rate"),

                            # Energy
                            "energy_mean": profile.get("energy_mean"),
                            "energy_std": profile.get("energy_std"),
                            "energy_dynamic_range_db": profile.get("energy_dynamic_range_db"),

                            # Voice quality
                            "jitter_percent": profile.get("jitter_percent"),
                            "jitter_std": profile.get("jitter_std"),
                            "shimmer_percent": profile.get("shimmer_percent"),
                            "shimmer_std": profile.get("shimmer_std"),
                            "harmonic_to_noise_ratio_db": profile.get("harmonic_to_noise_ratio_db"),
                            "hnr_std": profile.get("hnr_std"),

                            # Statistical
                            "feature_covariance_matrix": profile.get("feature_covariance_matrix"),
                            "feature_statistics": profile.get("feature_statistics"),
                        },

                        # Quality metrics
                        "enrollment_quality_score": profile.get("enrollment_quality_score"),
                        "feature_extraction_version": profile.get("feature_extraction_version"),
                    }

                    self.profile_quality_scores[speaker_name] = {
                        "is_native": is_native,
                        "quality": quality,
                        "threshold": threshold,
                        "samples": total_samples,
                    }

                    # Validate acoustic features (BEAST MODE check)
                    acoustic_features = self.speaker_profiles[speaker_name]["acoustic_features"]
                    has_acoustic_features = any(
                        v is not None for v in acoustic_features.values()
                    )

                    if has_acoustic_features:
                        logger.info(
                            f"✅ Loaded: {speaker_name} "
                            f"(ID: {speaker_id}, Primary: {profile.get('is_primary_user', False)}, "
                            f"{embedding.shape[0]}D, Quality: {quality}, "
                            f"Threshold: {threshold*100:.0f}%, Samples: {total_samples}) "
                            f"🔬 BEAST MODE"
                        )
                    else:
                        logger.warning(
                            f"⚠️  Loaded: {speaker_name} "
                            f"(ID: {speaker_id}, {embedding.shape[0]}D, Samples: {total_samples}) "
                            f"- NO ACOUSTIC FEATURES (basic mode only)"
                        )
                        logger.info(
                            f"   💡 To enable BEAST MODE for {speaker_name}, run: "
                            f"python3 backend/quick_voice_enhancement.py"
                        )

                    loaded_count += 1

                except Exception as profile_error:
                    logger.error(f"❌ Error loading profile {speaker_name}: {profile_error}")
                    skipped_count += 1
                    continue

            # If no profiles found, provide diagnostics
            if len(profiles) == 0:
                logger.warning("⚠️ No speaker profiles found in database!")
                logger.info("💡 To create a speaker profile, use voice commands like:")
                logger.info("   - 'Learn my voice as Derek'")
                logger.info("   - 'Create speaker profile for Derek'")

            # Summary
            logger.info(
                f"✅ Speaker profile loading complete: {loaded_count} loaded, {skipped_count} skipped"
            )

            if loaded_count == 0 and len(profiles) == 0:
                logger.warning(
                    "⚠️ No speaker profiles available - voice authentication will operate in enrollment mode"
                )
            elif loaded_count == 0 and len(profiles) > 0:
                logger.error(
                    f"❌ Found {len(profiles)} profiles in database but failed to load any - check logs above for errors"
                )

        except Exception as e:
            logger.error(f"❌ Failed to load speaker profiles: {e}", exc_info=True)
            logger.warning("⚠️ Continuing with 0 profiles - voice verification will fail until profiles are loaded")
            logger.info("💡 Troubleshooting steps:")
            logger.info("   1. Check database connection and credentials")
            logger.info("   2. Verify speaker_profiles table exists and has correct schema")
            logger.info("   3. Run database migrations if needed")
            logger.info("   4. Check Cloud SQL proxy is running (if using Cloud SQL)")

    async def verify_speaker(self, audio_data: bytes, speaker_name: Optional[str] = None) -> dict:
        """
        Verify speaker from audio with adaptive learning

        Args:
            audio_data: Audio bytes (WAV format)
            speaker_name: Expected speaker name (if None, identifies from all profiles)

        Returns:
            Verification result dict with:
                - verified: bool
                - confidence: float
                - speaker_name: str
                - is_owner: bool
                - security_level: str
                - adaptive_threshold: float (current dynamic threshold)
        """
        if not self.initialized:
            await self.initialize()

        # Debug audio data
        logger.info(f"🎤 AUDIO DEBUG: Received {len(audio_data) if audio_data else 0} bytes of audio")
        if audio_data and len(audio_data) > 0:
            # Check if audio is not silent
            import numpy as np
            # JARVIS sends int16 PCM audio, not float32
            try:
                # Try int16 first (JARVIS format)
                audio_array = np.frombuffer(audio_data[:min(2000, len(audio_data))], dtype=np.int16)
                audio_array = audio_array.astype(np.float32) / 32768.0  # Convert to float32 normalized
                logger.info(f"🎤 AUDIO DEBUG: Detected int16 PCM format")
            except:
                # Fallback to float32 if that fails
                audio_array = np.frombuffer(audio_data[:min(1000, len(audio_data))], dtype=np.float32, count=-1)
                logger.info(f"🎤 AUDIO DEBUG: Using float32 format")

            if len(audio_array) > 0:
                audio_energy = np.mean(np.abs(audio_array))
                logger.info(f"🎤 AUDIO DEBUG: Energy level = {audio_energy:.6f}")
                if audio_energy < 0.0001:
                    logger.warning("⚠️ AUDIO DEBUG: Audio appears to be silent!")
        else:
            logger.error("❌ AUDIO DEBUG: No audio data received!")

        # Convert audio to float32 for processing
        if audio_data and len(audio_data) > 0:
            try:
                # Convert int16 PCM to float32 for the engine
                audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                audio_data = audio_float32.tobytes()
                logger.info(f"🎤 AUDIO DEBUG: Converted {len(audio_int16)} int16 samples to float32")
            except Exception as e:
                logger.info(f"🎤 AUDIO DEBUG: Keeping original format: {e}")

        try:
            # If speaker name provided, verify against that profile
            if speaker_name and speaker_name in self.speaker_profiles:
                profile = self.speaker_profiles[speaker_name]
                known_embedding = profile["embedding"]

                # DEBUG: Log embedding dimensions
                logger.info(f"🔍 DEBUG: Verifying {speaker_name}")
                logger.info(f"🔍 DEBUG: Stored embedding shape: {known_embedding.shape if hasattr(known_embedding, 'shape') else len(known_embedding)}")
                logger.info(f"🔍 DEBUG: Stored embedding dimension in profile: {profile.get('embedding_dimension', 'unknown')}")

                # Get adaptive threshold based on history
                adaptive_threshold = await self._get_adaptive_threshold(speaker_name, profile)
                logger.info(f"🔍 DEBUG: Adaptive threshold: {adaptive_threshold:.2%}")

                # Check if we should enter calibration mode
                if await self._should_enter_calibration(speaker_name):
                    logger.info(f"🔄 Entering calibration mode for {speaker_name}")
                    self.calibration_mode = True
                    adaptive_threshold = self.calibration_threshold

                # Get base verification result
                if self.debug_mode:
                    logger.info(f"🎤 VERIFICATION DEBUG: Starting verification for {speaker_name}")
                    logger.info(f"  📊 Audio data size: {len(audio_data)} bytes")
                    logger.info(f"  📊 Profile has {profile.get('total_samples', 0)} training samples")

                    # BEEFED UP: Robust quality score handling
                    quality_info = self.profile_quality_scores.get(speaker_name, {"quality": 1.0}) # Default to 1.0 if not found in dict or missing key 
                    quality_score = quality_info.get("quality", 1.0) if isinstance(quality_info, dict) else quality_info # Default to 1.0 if not found in dict or missing key 

                    # Convert to float with robust error handling
                    try:
                        # Convert to float with robust error handling (incase it's a string) or default to 1.0
                        quality_score_float = float(quality_score) if quality_score is not None else 1.0
                    except (ValueError, TypeError):
                        logger.warning(f"  ⚠️ Invalid quality score type: {type(quality_score)}, defaulting to 1.0")
                        quality_score_float = 1.0 # Default to 1.0 if quality score is invalid

                    logger.info(f"  📊 Profile quality score: {quality_score_float:.2f}")
                    logger.info(f"  📊 Profile created: {profile.get('created_at', 'unknown')}")
                    logger.info(f"  📊 Profile embedding dim: {profile.get('embedding_dimension', 'unknown')}")

                is_verified, confidence = await self.speechbrain_engine.verify_speaker(
                    audio_data, known_embedding, threshold=adaptive_threshold,
                    speaker_name=speaker_name, transcription="",
                    enrolled_profile=profile  # Pass full profile with acoustic features
                )

                if self.debug_mode:
                    logger.info(f"  🔍 Base confidence: {confidence:.2%} ({confidence:.4f} raw)")
                    logger.info(f"  🔍 Threshold used: {adaptive_threshold:.2%} ({adaptive_threshold:.4f} raw)")
                    logger.info(f"  🔍 Initial verification: {'PASS' if is_verified else 'FAIL'}")
                    logger.info(f"  🔍 Confidence vs Threshold: {confidence:.4f} {'≥' if confidence >= adaptive_threshold else '<'} {adaptive_threshold:.4f}")

                # Apply multi-stage verification if enabled
                if self.multi_stage_enabled and confidence > 0.05:
                    original_confidence = confidence
                    confidence = await self._apply_multi_stage_verification(
                        confidence, audio_data, speaker_name, profile
                    )
                    if self.debug_mode:
                        logger.info(f"  🔄 Multi-stage verification: {original_confidence:.2%} → {confidence:.2%}")
                        logger.info(f"     Change: {(confidence - original_confidence):.2%} ({'↑' if confidence > original_confidence else '↓'})")

                # Apply confidence boosting if applicable
                if self.confidence_boost_enabled:
                    original_confidence = confidence
                    confidence = await self._apply_confidence_boost(
                        confidence, speaker_name, profile
                    )
                    if self.debug_mode and confidence != original_confidence:
                        logger.info(f"  ⬆️ Confidence boost applied: {original_confidence:.2%} → {confidence:.2%}")
                        logger.info(f"     Boost factor: {confidence/original_confidence:.2f}x")

                # Update verification decision based on boosted confidence
                is_verified = confidence >= adaptive_threshold
                if self.debug_mode:
                    logger.info(f"  📍 Final verification decision: {'✅ PASS' if is_verified else '❌ FAIL'}")
                    logger.info(f"  📍 Final confidence: {confidence:.2%} vs threshold: {adaptive_threshold:.2%}")

                # Handle calibration mode
                if self.calibration_mode and confidence > 0.10:
                    await self._handle_calibration_sample(
                        audio_data, speaker_name, confidence
                    )

                logger.info(f"🔍 DEBUG: Final result - Confidence: {confidence:.2%}, Verified: {is_verified}")

                # Store voice sample for continuous learning
                if self.continuous_learning_enabled and self.store_all_audio:
                    await self._store_voice_sample_async(
                        speaker_name=speaker_name,
                        audio_data=audio_data,
                        confidence=confidence,
                        verified=is_verified,
                        command="unlock_screen",
                        environment_type=self.current_environment
                    )

                # Learn from this attempt
                await self._record_verification_attempt(speaker_name, confidence, is_verified)

                # 🧠 Update Voice Memory Agent
                try:
                    from agents.voice_memory_agent import get_voice_memory_agent
                    voice_agent = await get_voice_memory_agent()
                    await voice_agent.record_interaction(speaker_name, confidence, is_verified)
                except Exception as e:
                    logger.debug(f"Voice memory agent update skipped: {e}")

                result = {
                    "verified": is_verified,
                    "confidence": confidence,
                    "speaker_name": speaker_name,
                    "speaker_id": profile["speaker_id"],
                    "is_owner": profile["is_primary_user"],
                    "security_level": profile["security_level"],
                    "adaptive_threshold": adaptive_threshold,
                }

                # If confidence is very low, suggest re-enrollment
                if confidence < 0.10:
                    result["suggestion"] = "Voice profile may need re-enrollment"
                    logger.warning(f"⚠️ Very low confidence ({confidence:.2%}) for {speaker_name} - consider re-enrollment")

                return result

            # Otherwise, identify speaker from all profiles
            best_match = None
            best_confidence = 0.0

            for profile_name, profile in self.speaker_profiles.items():
                known_embedding = profile["embedding"]
                profile_threshold = profile.get("threshold", self.verification_threshold)
                is_verified, confidence = await self.speechbrain_engine.verify_speaker(
                    audio_data, known_embedding, threshold=profile_threshold,
                    speaker_name=profile_name, transcription="",
                    enrolled_profile=profile  # Pass full profile with acoustic features
                )

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = {
                        "verified": is_verified,
                        "confidence": confidence,
                        "speaker_name": profile_name,
                        "speaker_id": profile["speaker_id"],
                        "is_owner": profile["is_primary_user"],
                        "security_level": profile["security_level"],
                    }

            if best_match:
                logger.info(
                    f"🎤 Speaker identified: {best_match['speaker_name']} "
                    f"(confidence: {best_match['confidence']:.1%}, "
                    f"owner: {best_match['is_owner']})"
                )
                return best_match

            # No match found - but check if we have any primary user profile
            # This ensures we at least know who the owner is even if verification failed
            primary_profile = None
            for profile_name, profile in self.speaker_profiles.items():
                if profile.get("is_primary_user", False):
                    primary_profile = profile_name
                    break

            logger.warning(f"⚠️ No speaker match found. Primary user: {primary_profile}, Best confidence: {best_confidence:.2%}")

            return {
                "verified": False,
                "confidence": best_confidence if best_confidence else 0.0,
                "speaker_name": "unknown",
                "speaker_id": None,
                "is_owner": False,
                "security_level": "none",
                "primary_user": primary_profile,  # Include who the actual owner is
                "requires_enrollment": len(self.speaker_profiles) == 0
            }

        except Exception as e:
            logger.error(f"Speaker verification error: {e}", exc_info=True)
            return {
                "verified": False,
                "confidence": 0.0,
                "speaker_name": "error",
                "speaker_id": None,
                "is_owner": False,
                "security_level": "none",
                "error": str(e),
            }

    async def is_owner(self, audio_data: bytes) -> tuple[bool, float]:
        """
        Check if audio is from the device owner (Derek J. Russell)

        Args:
            audio_data: Audio bytes

        Returns:
            Tuple of (is_owner, confidence)
        """
        result = await self.verify_speaker(audio_data)
        return result["is_owner"], result["confidence"]

    async def get_speaker_name(self, audio_data: bytes) -> str:
        """
        Get speaker name from audio

        Args:
            audio_data: Audio bytes

        Returns:
            Speaker name or "unknown"
        """
        result = await self.verify_speaker(audio_data)
        return result["speaker_name"]

    async def refresh_profiles(self):
        """Reload speaker profiles from database"""
        logger.info("🔄 Refreshing speaker profiles...")
        self.speaker_profiles.clear()
        await self._load_speaker_profiles()

    async def _get_adaptive_threshold(self, speaker_name: str, profile: dict) -> float:
        """
        Calculate adaptive threshold based on verification history

        Args:
            speaker_name: Name of speaker
            profile: Speaker profile dict

        Returns:
            Adaptive threshold value
        """
        base_threshold = profile.get("threshold", self.verification_threshold)

        # If no history, use base threshold
        if speaker_name not in self.verification_history:
            return base_threshold

        attempts = self.verification_history[speaker_name]

        # Need minimum samples for adaptation
        if len(attempts) < self.min_samples_for_update:
            return base_threshold

        # Calculate average confidence from recent attempts
        recent_attempts = attempts[-10:]  # Last 10 attempts
        successful_confidences = [a["confidence"] for a in recent_attempts if a.get("verified", False)]

        if not successful_confidences:
            # No successful attempts recently - lower threshold progressively
            avg_confidence = sum(a["confidence"] for a in recent_attempts) / len(recent_attempts)
            if avg_confidence > 0.10:
                # There's some similarity, progressively lower threshold
                # Factor in number of consecutive failures
                failure_factor = min(self.failure_count.get(speaker_name, 0) * 0.05, 0.2)
                adaptive_threshold = max(0.20, avg_confidence * (0.9 - failure_factor))
                logger.info(f"📊 Adaptive threshold for {speaker_name}: {adaptive_threshold:.2%} (lowered from {base_threshold:.2%}, failures: {self.failure_count.get(speaker_name, 0)})")
                return adaptive_threshold
            return base_threshold

        # Calculate statistics
        avg_confidence = sum(successful_confidences) / len(successful_confidences)
        min_confidence = min(successful_confidences)

        # Set threshold slightly below minimum successful confidence
        # This allows for natural variation in voice
        adaptive_threshold = max(0.25, min(base_threshold, min_confidence * 0.90))

        logger.info(
            f"📊 Adaptive threshold for {speaker_name}: {adaptive_threshold:.2%} "
            f"(base: {base_threshold:.2%}, avg: {avg_confidence:.2%}, min: {min_confidence:.2%})"
        )

        return adaptive_threshold

    async def _record_verification_attempt(self, speaker_name: str, confidence: float, verified: bool):
        """
        Record verification attempt for adaptive learning

        Args:
            speaker_name: Name of speaker
            confidence: Confidence score
            verified: Whether verification succeeded
        """
        if not self.learning_enabled:
            return

        # Initialize history for this speaker
        if speaker_name not in self.verification_history:
            self.verification_history[speaker_name] = []

        # Track failure count for calibration
        if not verified:
            self.failure_count[speaker_name] = self.failure_count.get(speaker_name, 0) + 1
            logger.info(f"❌ Verification failed for {speaker_name} (failure #{self.failure_count[speaker_name]})")
        else:
            self.failure_count[speaker_name] = 0  # Reset on success

        # Record attempt in simplified format
        attempt = {
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence,
            "verified": verified,
        }

        self.verification_history[speaker_name].append(attempt)

        # Keep only recent attempts (last 50)
        if len(self.verification_history[speaker_name]) > 50:
            self.verification_history[speaker_name] = self.verification_history[speaker_name][-50:]

        # Update confidence statistics for normalization
        if speaker_name not in self.confidence_stats:
            self.confidence_stats[speaker_name] = {"scores": []}

        self.confidence_stats[speaker_name]["scores"].append(confidence)
        # Keep last N scores for statistics
        if len(self.confidence_stats[speaker_name]["scores"]) > self.confidence_history_window:
            self.confidence_stats[speaker_name]["scores"] = \
                self.confidence_stats[speaker_name]["scores"][-self.confidence_history_window:]

        # Log learning progress
        total_attempts = len(self.verification_history[speaker_name])
        if total_attempts % 5 == 0:
            recent_attempts = self.verification_history[speaker_name][-10:]
            success_rate = sum(1 for a in recent_attempts if a['verified']) / len(recent_attempts) * 100
            logger.info(
                f"📚 Learning progress for {speaker_name}: "
                f"{total_attempts} total attempts, "
                f"{success_rate:.1f}% recent success rate"
            )

    async def _check_profile_updates(self) -> dict:
        """
        Check if any speaker profiles have been updated in the database.

        Returns:
            dict: Mapping of speaker_name -> has_updates (bool)
        """
        try:
            if not self.learning_db:
                return {}

            updates = {}

            # Query database for current profile timestamps/versions
            async with self.learning_db.db.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT speaker_name, speaker_id, last_updated, total_samples,
                           enrollment_quality_score, feature_extraction_version
                    FROM speaker_profiles
                    """
                )
                profiles = await cursor.fetchall()

                for profile in profiles:
                    speaker_name = profile['speaker_name'] if isinstance(profile, dict) else profile[0]
                    speaker_id = profile['speaker_id'] if isinstance(profile, dict) else profile[1]
                    updated_at = profile['last_updated'] if isinstance(profile, dict) else profile[2]
                    total_samples = profile['total_samples'] if isinstance(profile, dict) else profile[3]
                    quality_score = profile['enrollment_quality_score'] if isinstance(profile, dict) else profile[4]
                    feature_version = profile['feature_extraction_version'] if isinstance(profile, dict) else profile[5]

                    # Create version fingerprint
                    current_fingerprint = {
                        'updated_at': str(updated_at) if updated_at else None,
                        'total_samples': total_samples,
                        'quality_score': quality_score,
                        'feature_version': feature_version,
                    }

                    # Check if we've seen this profile before
                    if speaker_name not in self.profile_version_cache:
                        # New profile detected
                        self.profile_version_cache[speaker_name] = current_fingerprint
                        updates[speaker_name] = True
                    else:
                        # Check if profile changed
                        cached_fingerprint = self.profile_version_cache[speaker_name]
                        has_changed = (
                            cached_fingerprint['updated_at'] != current_fingerprint['updated_at'] or
                            cached_fingerprint['total_samples'] != current_fingerprint['total_samples'] or
                            cached_fingerprint['quality_score'] != current_fingerprint['quality_score'] or
                            cached_fingerprint['feature_version'] != current_fingerprint['feature_version']
                        )

                        if has_changed:
                            logger.info(f"🔄 Detected update for profile '{speaker_name}'")
                            logger.debug(f"   Old: {cached_fingerprint}")
                            logger.debug(f"   New: {current_fingerprint}")
                            self.profile_version_cache[speaker_name] = current_fingerprint
                            updates[speaker_name] = True
                        else:
                            updates[speaker_name] = False

            return updates

        except Exception as e:
            logger.error(f"❌ Error checking profile updates: {e}", exc_info=True)
            return {}

    async def _profile_reload_monitor(self):
        """
        Background task that monitors for profile updates and reloads automatically.
        Runs continuously until service shutdown.
        """
        logger.info("🔄 Profile reload monitor started")

        try:
            while not self._shutdown_event.is_set():
                try:
                    # Check for updates
                    updates = await self._check_profile_updates()

                    # If any profiles updated, reload all
                    if any(updates.values()):
                        updated_profiles = [name for name, has_update in updates.items() if has_update]
                        logger.info(f"🔄 Reloading profiles due to updates: {', '.join(updated_profiles)}")
                        await self.refresh_profiles()
                        logger.info("✅ Profiles reloaded successfully with latest data from database")

                    # Wait before next check
                    await asyncio.sleep(self.reload_check_interval)

                except asyncio.CancelledError:
                    logger.info("🛑 Profile reload monitor cancelled")
                    break
                except Exception as e:
                    logger.error(f"❌ Error in profile reload monitor: {e}", exc_info=True)
                    # Continue monitoring even after error
                    await asyncio.sleep(self.reload_check_interval)

        except Exception as e:
            logger.error(f"❌ Profile reload monitor crashed: {e}", exc_info=True)
        finally:
            logger.info("🛑 Profile reload monitor stopped")

    async def manual_reload_profiles(self) -> dict:
        """
        Manually trigger profile reload (for API endpoint).

        Returns:
            dict: Status information about the reload
        """
        try:
            logger.info("🔄 Manual profile reload triggered")
            profiles_before = len(self.speaker_profiles)

            await self.refresh_profiles()

            profiles_after = len(self.speaker_profiles)

            return {
                "success": True,
                "message": "Profiles reloaded successfully",
                "profiles_before": profiles_before,
                "profiles_after": profiles_after,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"❌ Manual profile reload failed: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Reload failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
            }

    async def _should_enter_calibration(self, speaker_name: str) -> bool:
        """Check if we should enter calibration mode for this speaker"""
        if not self.auto_calibrate_on_failure:
            return False

        failures = self.failure_count.get(speaker_name, 0)
        return failures >= self.max_failures_before_calibration

    async def _apply_multi_stage_verification(
        self, base_confidence: float, audio_data: bytes,
        speaker_name: str, profile: dict
    ) -> float:
        """Apply multi-stage verification with weighted scoring"""
        scores = {'primary': base_confidence}

        # Add acoustic feature scoring (simplified for now)
        acoustic_score = base_confidence * 1.1  # Boost by 10% for acoustic match
        scores['acoustic'] = min(acoustic_score, 1.0)

        # Add temporal pattern scoring
        if speaker_name in self.verification_history:
            recent_scores = [h['confidence'] for h in self.verification_history[speaker_name][-5:]]
            if recent_scores:
                temporal_score = np.mean(recent_scores) * 1.2
                scores['temporal'] = min(temporal_score, 1.0)

        # Add adaptive scoring based on rolling embeddings
        if speaker_name in self.rolling_embeddings:
            adaptive_score = base_confidence * 1.15
            scores['adaptive'] = min(adaptive_score, 1.0)

        # Calculate weighted average
        total_confidence = 0
        total_weight = 0
        for stage, weight in self.stage_weights.items():
            if stage in scores:
                total_confidence += scores[stage] * weight
                total_weight += weight

        if total_weight > 0:
            final_confidence = total_confidence / total_weight
            logger.info(f"🔄 Multi-stage verification: {base_confidence:.2%} -> {final_confidence:.2%}")
            return final_confidence

        return base_confidence

    async def _apply_confidence_boost(
        self, confidence: float, speaker_name: str, profile: dict
    ) -> float:
        """Apply confidence boosting based on patterns"""
        if confidence < self.min_confidence_for_boost:
            return confidence

        # Check if this speaker has good history
        if speaker_name in self.verification_history:
            history = self.verification_history[speaker_name]
            if len(history) >= 3:
                recent_success_rate = sum(1 for h in history[-10:] if h.get('verified', False)) / min(len(history), 10)
                if recent_success_rate > 0.5:
                    # Apply boost
                    boosted = confidence * self.boost_multiplier
                    boosted = min(boosted, 0.95)  # Cap at 95%
                    logger.info(f"🚀 Confidence boost applied: {confidence:.2%} -> {boosted:.2%}")
                    return boosted

        # Apply environmental boost if in known environment
        if self.current_environment in self.environment_profiles:
            env_boost = 1.2
            boosted = confidence * env_boost
            boosted = min(boosted, 0.95)
            if boosted > confidence:
                logger.info(f"🌍 Environment boost: {confidence:.2%} -> {boosted:.2%}")
                return boosted

        return confidence

    async def _handle_calibration_sample(
        self, audio_data: bytes, speaker_name: str, confidence: float
    ):
        """Handle a calibration sample"""
        logger.info(f"📝 Recording calibration sample for {speaker_name} (confidence: {confidence:.2%})")

        # Extract embedding from this sample
        try:
            new_embedding = await self.speechbrain_engine.extract_speaker_embedding(audio_data)

            # Add to calibration samples
            self.calibration_samples.append({
                'speaker': speaker_name,
                'embedding': new_embedding,
                'confidence': confidence,
                'timestamp': datetime.now()
            })

            # If we have enough samples, update the profile
            speaker_samples = [s for s in self.calibration_samples if s['speaker'] == speaker_name]
            if len(speaker_samples) >= 3:
                await self._update_profile_from_calibration(speaker_name, speaker_samples)
                # Reset calibration mode
                self.calibration_mode = False
                self.calibration_samples = []
                self.failure_count[speaker_name] = 0
                logger.info(f"✅ Calibration complete for {speaker_name}")
        except Exception as e:
            logger.error(f"Calibration sample error: {e}")

    async def _update_profile_from_calibration(self, speaker_name: str, samples: list):
        """Update speaker profile from calibration samples"""
        logger.info(f"🔄 Updating profile for {speaker_name} from {len(samples)} calibration samples")

        # Average the embeddings
        embeddings = [s['embedding'] for s in samples]
        avg_embedding = np.mean(embeddings, axis=0)

        # Update in database
        if self.learning_db:
            try:
                await self.learning_db.update_speaker_embedding(
                    speaker_name=speaker_name,
                    embedding=avg_embedding,
                    metadata={
                        'calibration_update': True,
                        'samples_used': len(samples),
                        'update_time': datetime.now().isoformat()
                    }
                )

                # Update local cache
                if speaker_name in self.speaker_profiles:
                    self.speaker_profiles[speaker_name]['embedding'] = avg_embedding
                    # Adjust threshold based on calibration confidence
                    avg_confidence = np.mean([s['confidence'] for s in samples])
                    new_threshold = max(0.25, avg_confidence * 0.8)  # 80% of average confidence
                    self.speaker_profiles[speaker_name]['threshold'] = new_threshold
                    logger.info(f"✅ Profile updated with new threshold: {new_threshold:.2%}")
            except Exception as e:
                logger.error(f"Failed to update profile: {e}")

    async def _store_voice_sample_async(
        self, speaker_name: str, audio_data: bytes,
        confidence: float, verified: bool,
        command: Optional[str] = None,
        environment_type: Optional[str] = None
    ):
        """
        Asynchronously store voice sample for continuous learning
        """
        try:
            # Extract embedding for storage
            embedding = None
            if self.speechbrain_engine:
                try:
                    embedding = await self.speechbrain_engine.extract_speaker_embedding(audio_data)
                except Exception as e:
                    logger.warning(f"Could not extract embedding: {e}")

            # Calculate audio quality
            quality_score = await self._calculate_audio_quality(audio_data)

            # Store in database
            sample_id = await self.learning_db.store_voice_sample(
                speaker_name=speaker_name,
                audio_data=audio_data,
                embedding=embedding,
                confidence=confidence,
                verified=verified,
                command=command,
                environment_type=environment_type,
                quality_score=quality_score,
                metadata={
                    'threshold_used': self.speaker_profiles.get(speaker_name, {}).get('threshold', self.verification_threshold),
                    'calibration_mode': self.calibration_mode,
                    'failure_count': self.failure_count.get(speaker_name, 0),
                    'timestamp': datetime.now().isoformat()
                }
            )

            # Add to memory buffer for quick access
            if len(self.voice_sample_buffer) >= self.max_buffer_size:
                self.voice_sample_buffer.pop(0)  # Remove oldest

            self.voice_sample_buffer.append({
                'sample_id': sample_id,
                'speaker_name': speaker_name,
                'confidence': confidence,
                'verified': verified,
                'embedding': embedding,
                'timestamp': datetime.now()
            })

            logger.info(f"📀 Stored voice sample #{sample_id} for {speaker_name} (conf: {confidence:.2%}, verified: {verified})")

            # Update rolling embeddings if successful
            if verified and embedding is not None:
                await self._update_rolling_embeddings(speaker_name, embedding)

        except Exception as e:
            logger.error(f"Failed to store voice sample: {e}")

    async def _calculate_audio_quality(self, audio_data: bytes) -> float:
        """
        Calculate audio quality score (0-1)
        """
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Calculate metrics
            energy = np.mean(np.abs(audio_array))
            snr = self._estimate_snr(audio_array)

            # Simple quality score
            quality = min(1.0, energy * 10) * min(1.0, snr / 20)
            return max(0.1, quality)  # Minimum quality 0.1

        except Exception:
            return 0.5  # Default quality

    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate signal-to-noise ratio"""
        try:
            # Simple SNR estimation
            signal_power = np.mean(audio ** 2)
            noise_floor = np.percentile(np.abs(audio), 10) ** 2

            if noise_floor > 0:
                snr_db = 10 * np.log10(signal_power / noise_floor)
                return max(0, min(30, snr_db))  # Clamp between 0-30 dB
            return 15.0  # Default SNR

        except Exception:
            return 10.0

    async def _update_rolling_embeddings(self, speaker_name: str, new_embedding: np.ndarray):
        """
        Update rolling embeddings for adaptive learning
        """
        if speaker_name not in self.rolling_embeddings:
            self.rolling_embeddings[speaker_name] = []

        embeddings_list = self.rolling_embeddings[speaker_name]
        embeddings_list.append(new_embedding)

        # Keep only recent embeddings
        if len(embeddings_list) > self.max_rolling_samples:
            embeddings_list.pop(0)

        # Update profile with rolling average if enough samples
        if len(embeddings_list) >= 5:
            # Compute weighted average (recent samples have more weight)
            weights = np.linspace(0.5, 1.0, len(embeddings_list))
            weights = weights / weights.sum()

            rolling_avg = np.average(embeddings_list, axis=0, weights=weights)

            # Blend with current embedding
            if speaker_name in self.speaker_profiles:
                current_embedding = self.speaker_profiles[speaker_name]['embedding']
                updated_embedding = (
                    (1 - self.rolling_weight) * current_embedding +
                    self.rolling_weight * rolling_avg
                )

                # Update in memory (not database yet)
                self.speaker_profiles[speaker_name]['rolling_embedding'] = updated_embedding
                logger.info(f"📊 Updated rolling embedding for {speaker_name} ({len(embeddings_list)} samples)")

    async def perform_rag_similarity_search(
        self, speaker_name: str, current_embedding: np.ndarray, top_k: int = 5
    ) -> List[Dict]:
        """
        RAG: Retrieve similar voice patterns for better verification
        """
        try:
            # Get recent successful verifications from buffer
            similar_samples = []

            for sample in self.voice_sample_buffer:
                if (sample['speaker_name'] == speaker_name and
                    sample['verified'] and
                    sample.get('embedding') is not None):

                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(current_embedding, sample['embedding'])
                    similar_samples.append({
                        'sample_id': sample['sample_id'],
                        'similarity': similarity,
                        'confidence': sample['confidence'],
                        'timestamp': sample['timestamp']
                    })

            # Sort by similarity and return top-k
            similar_samples.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_samples[:top_k]

        except Exception as e:
            logger.error(f"RAG similarity search failed: {e}")
            return []

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            a_norm = a / np.linalg.norm(a)
            b_norm = b / np.linalg.norm(b)
            return float(np.dot(a_norm, b_norm))
        except Exception:
            return 0.0

    async def apply_human_feedback(
        self, verification_id: int, correct: bool, notes: Optional[str] = None
    ):
        """
        Apply RLHF: Human feedback on verification result

        Args:
            verification_id: ID of the verification attempt
            correct: Whether the verification was correct
            notes: Optional feedback notes
        """
        try:
            # Find the sample in buffer
            sample = None
            for s in self.voice_sample_buffer:
                if s.get('sample_id') == verification_id:
                    sample = s
                    break

            if not sample:
                logger.warning(f"Sample {verification_id} not found in buffer")
                return

            # Calculate feedback score
            feedback_score = 1.0 if correct else 0.0

            # Apply RLHF to database
            await self.learning_db.apply_rlhf_feedback(
                sample_id=verification_id,
                feedback_score=feedback_score,
                feedback_notes=notes
            )

            # Update local statistics
            speaker_name = sample['speaker_name']
            if speaker_name not in self.sample_metadata:
                self.sample_metadata[speaker_name] = {'feedback_count': 0, 'correct_count': 0}

            self.sample_metadata[speaker_name]['feedback_count'] += 1
            if correct:
                self.sample_metadata[speaker_name]['correct_count'] += 1

            logger.info(f"✅ Applied human feedback for {speaker_name} (correct: {correct})")

            # Trigger retraining if enough feedback
            feedback_count = self.sample_metadata[speaker_name]['feedback_count']
            if feedback_count >= 10 and feedback_count % 5 == 0:
                await self._trigger_retraining(speaker_name)

        except Exception as e:
            logger.error(f"Failed to apply human feedback: {e}")

    async def _trigger_retraining(self, speaker_name: str):
        """
        Trigger model retraining with accumulated samples
        """
        try:
            logger.info(f"🔄 Triggering retraining for {speaker_name}")

            # Get recent samples with feedback
            samples = await self.learning_db.get_voice_samples_for_training(
                speaker_name=speaker_name,
                limit=30,
                min_confidence=0.05  # Include low confidence samples for learning
            )

            if len(samples) >= 10:
                # Perform incremental learning
                result = await self.learning_db.perform_incremental_learning(
                    speaker_name=speaker_name,
                    new_samples=samples
                )

                if result.get('success'):
                    # Reload profile
                    await self._load_speaker_profiles()
                    logger.info(f"✅ Retraining complete for {speaker_name}: {result}")
                else:
                    logger.error(f"Retraining failed: {result.get('error')}")

        except Exception as e:
            logger.error(f"Retraining trigger failed: {e}")

    async def enable_calibration_mode(self, speaker_name: str = None):
        """Manually enable calibration mode"""
        self.calibration_mode = True
        self.calibration_samples = []
        if speaker_name:
            self.failure_count[speaker_name] = 0
        logger.info(f"🎯 Calibration mode enabled{f' for {speaker_name}' if speaker_name else ''}")
        return {"status": "calibration_enabled", "speaker": speaker_name}

    async def cleanup(self):
        """Cleanup resources and terminate background threads"""
        logger.info("🧹 Cleaning up Speaker Verification Service...")

        # Signal shutdown to background threads
        self._shutdown_event.set()

        # Cancel profile reload monitor task
        if self._reload_task and not self._reload_task.done():
            logger.debug("   Cancelling profile reload monitor...")
            self._reload_task.cancel()
            try:
                await asyncio.wait_for(self._reload_task, timeout=2.0)
                logger.debug("   ✅ Profile reload monitor cancelled")
            except (asyncio.CancelledError, asyncio.TimeoutError):
                logger.debug("   ✅ Profile reload monitor terminated")
            except Exception as e:
                logger.warning(f"   ⚠ Profile reload monitor cleanup error: {e}")

        # Wait for preload thread to complete (with timeout)
        if self._preload_thread and self._preload_thread.is_alive():
            logger.debug("   Waiting for background preload thread to finish...")
            self._preload_thread.join(timeout=2.0)

            if self._preload_thread.is_alive():
                logger.warning("   Preload thread did not exit cleanly - marking as daemon")
                # Ensure it's daemon so it doesn't block shutdown
                self._preload_thread.daemon = True
            else:
                logger.debug("   ✅ Preload thread terminated cleanly")

            self._preload_thread = None

        # Clean up event loop if still running
        if self._preload_loop and not self._preload_loop.is_closed():
            try:
                logger.debug("   Closing background event loop...")
                self._preload_loop.stop()
                self._preload_loop.close()
                self._preload_loop = None
            except Exception as e:
                logger.debug(f"   Event loop cleanup error: {e}")

        # Clean up learning database (closes background tasks and threads)
        if self.learning_db:
            try:
                logger.debug("   Closing learning database...")
                from intelligence.learning_database import close_learning_database
                await close_learning_database()
                logger.debug("   ✅ Learning database closed")
            except Exception as e:
                logger.warning(f"   ⚠ Learning database cleanup error: {e}")

        # Clean up SpeechBrain engine
        if self.speechbrain_engine:
            try:
                await self.speechbrain_engine.cleanup()
                logger.debug("   ✅ SpeechBrain engine cleaned up")
            except Exception as e:
                logger.warning(f"   ⚠ SpeechBrain cleanup error: {e}")

        # Clear caches
        self.speaker_profiles.clear()
        self.profile_quality_scores.clear()

        # Reset state
        self.initialized = False
        self._encoder_preloaded = False
        self._encoder_preloading = False
        self._preload_thread = None
        self.learning_db = None

        logger.info("✅ Speaker Verification Service cleaned up")


# Global singleton instance
_speaker_verification_service: Optional[SpeakerVerificationService] = None


async def get_speaker_verification_service(
    learning_db: Optional[JARVISLearningDatabase] = None,
) -> SpeakerVerificationService:
    """
    Get global speaker verification service instance

    First checks for pre-loaded service from start_system.py,
    then falls back to creating new instance if needed.

    Args:
        learning_db: LearningDatabase instance (optional)

    Returns:
        SpeakerVerificationService instance
    """
    global _speaker_verification_service, _global_speaker_service

    # First check if there's a pre-loaded service from start_system.py
    if _global_speaker_service is not None:
        logger.info("✅ Using pre-loaded speaker verification service")
        _speaker_verification_service = _global_speaker_service
        return _global_speaker_service

    # Otherwise use the singleton pattern
    if _speaker_verification_service is None:
        logger.info("🔐 Creating new speaker verification service...")
        _speaker_verification_service = SpeakerVerificationService(learning_db)
        # Use fast initialization to avoid blocking (encoder loads in background)
        await _speaker_verification_service.initialize_fast()

    return _speaker_verification_service


async def reset_speaker_verification_service():
    """Reset service (for testing)"""
    global _speaker_verification_service
    if _speaker_verification_service:
        await _speaker_verification_service.cleanup()
    _speaker_verification_service = None
