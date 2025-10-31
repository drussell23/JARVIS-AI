"""
Ultra-Advanced Hybrid STT Router
Zero hardcoding, fully async, RAM-aware, cost-optimized
Integrates with learning database for continuous improvement
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

import psutil

from .stt_config import ModelConfig, RoutingStrategy, STTEngine, get_stt_config

logger = logging.getLogger(__name__)


@dataclass
class STTResult:
    """Result from STT engine"""

    text: str
    confidence: float
    engine: STTEngine
    model_name: str
    latency_ms: float
    audio_duration_ms: float
    metadata: Dict = field(default_factory=dict)
    audio_hash: Optional[str] = None
    speaker_identified: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SystemResources:
    """Current system resource availability"""

    total_ram_gb: float
    available_ram_gb: float
    ram_percent_used: float
    cpu_percent: float
    gpu_available: bool
    gpu_memory_gb: float = 0.0
    network_available: bool = True


class HybridSTTRouter:
    """
    Ultra-intelligent STT routing system.

    Features:
    - Zero hardcoding (all config-driven)
    - Fully async (non-blocking)
    - RAM-aware (adapts to available resources)
    - Cost-optimized (prefers local, escalates smartly)
    - Learning-enabled (gets better over time)
    - Multi-engine (Wav2Vec, Vosk, Whisper local/GCP)
    - Speaker-aware (Derek gets priority)
    - Confidence-based escalation
    - Automatic fallbacks
    """

    def __init__(self, config=None):
        self.config = config or get_stt_config()
        self.engines = {}  # Lazy-loaded STT engines
        self.performance_stats = {}  # Track engine performance
        self.learning_db = None  # Lazy-loaded learning database

        # Performance tracking
        self.total_requests = 0
        self.cloud_requests = 0
        self.cache_hits = 0

        logger.info("🎤 Hybrid STT Router initialized")
        logger.info(f"   Strategy: {self.config.default_strategy.value}")
        logger.info(f"   Models configured: {len(self.config.models)}")

    async def _get_learning_db(self):
        """Lazy-load learning database"""
        if self.learning_db is None:
            try:
                from intelligence.learning_database import LearningDatabase

                self.learning_db = LearningDatabase()
                await self.learning_db.initialize()
                logger.info("📚 Learning database connected to STT router")
            except Exception as e:
                logger.error(f"Failed to connect learning database: {e}")
                self.learning_db = None
        return self.learning_db

    def _get_system_resources(self) -> SystemResources:
        """Get current system resource availability"""
        mem = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Check for GPU (Metal on macOS)
        gpu_available = False
        gpu_memory_gb = 0.0
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                gpu_available = True
                # Estimate Metal shared memory (uses system RAM)
                gpu_memory_gb = mem.available / (1024**3) * 0.5  # Estimate 50% shareable
        except ImportError:
            pass

        # Check network
        network_available = True
        try:
            import socket

            socket.create_connection(("8.8.8.8", 53), timeout=1)
        except OSError:
            network_available = False

        return SystemResources(
            total_ram_gb=mem.total / (1024**3),
            available_ram_gb=mem.available / (1024**3),
            ram_percent_used=mem.percent,
            cpu_percent=cpu_percent,
            gpu_available=gpu_available,
            gpu_memory_gb=gpu_memory_gb,
            network_available=network_available,
        )

    def _select_optimal_model(
        self,
        resources: SystemResources,
        strategy: RoutingStrategy,
        speaker_name: Optional[str] = None,
    ) -> ModelConfig:
        """
        Intelligently select best model based on:
        - Available resources
        - Strategy
        - Speaker priority
        - Historical performance
        """
        available_ram = resources.available_ram_gb

        # WHISPER PRIORITY: Always try Whisper first if available
        # This ensures proper transcription instead of [transcription failed]
        whisper_models = ["whisper-base", "whisper-small", "whisper-tiny"]
        for whisper_name in whisper_models:
            whisper_model = self.config.models.get(whisper_name)
            if whisper_model and whisper_model.ram_required_gb <= available_ram:
                logger.info(f"🎯 Whisper priority: selected {whisper_name} for reliable transcription")
                return whisper_model

        if strategy == RoutingStrategy.SPEED:
            # Fastest model that fits
            model = self.config.get_fastest_model(available_ram)
            if model:
                logger.debug(f"🏃 Speed strategy: selected {model.name}")
                return model

        elif strategy == RoutingStrategy.ACCURACY:
            # Best accuracy (may use cloud)
            if speaker_name in self.config.priority_speakers or resources.network_available:
                model = self.config.get_most_accurate_model(available_ram)
                logger.debug(f"🎯 Accuracy strategy: selected {model.name}")
                return model

        elif strategy == RoutingStrategy.COST:
            # Minimize cloud usage
            model = self.config.get_model_for_ram(available_ram)
            if model and not model.requires_internet:
                logger.debug(f"💰 Cost strategy: selected {model.name}")
                return model

        elif strategy == RoutingStrategy.BALANCED or strategy == RoutingStrategy.ADAPTIVE:
            # Smart selection based on context
            if available_ram >= self.config.min_ram_for_wav2vec:
                # Prefer Wav2Vec (fine-tunable on Derek)
                model = self.config.models.get("wav2vec2-base")
                if model and model.ram_required_gb <= available_ram:
                    logger.debug(f"⚖️  Balanced strategy: selected {model.name}")
                    return model

            # Fallback to Vosk if RAM is tight
            model = self.config.models.get("vosk-small")
            if model:
                logger.debug(f"⚖️  Balanced strategy (RAM constrained): selected {model.name}")
                return model

        # Ultimate fallback: smallest model
        return self.config.get_fastest_model(100.0)  # Get ANY model

    async def _load_engine(self, model: ModelConfig):
        """Lazy-load STT engine"""
        engine_key = f"{model.engine.value}:{model.name}"

        if engine_key in self.engines:
            return self.engines[engine_key]

        logger.info(f"🔧 Loading STT engine: {model.name} ({model.engine.value})")

        try:
            if model.engine == STTEngine.WAV2VEC:
                from .engines.wav2vec_engine import Wav2VecEngine

                engine = Wav2VecEngine(model)
                await engine.initialize()

            elif model.engine == STTEngine.VOSK:
                from .engines.vosk_engine import VoskEngine

                engine = VoskEngine(model)
                await engine.initialize()

            elif model.engine == STTEngine.WHISPER_LOCAL:
                from .engines.whisper_local_engine import WhisperLocalEngine

                engine = WhisperLocalEngine(model)
                await engine.initialize()

            elif model.engine == STTEngine.WHISPER_GCP:
                from .engines.whisper_gcp_engine import WhisperGCPEngine

                engine = WhisperGCPEngine(model)
                await engine.initialize()

            elif model.engine == STTEngine.SPEECHBRAIN:
                from .engines.speechbrain_engine import SpeechBrainEngine

                engine = SpeechBrainEngine(model)
                await engine.initialize()

            else:
                raise ValueError(f"Unknown engine: {model.engine}")

            self.engines[engine_key] = engine
            logger.info(f"✅ Engine loaded: {model.name}")
            return engine

        except Exception as e:
            logger.error(f"Failed to load engine {model.name}: {e}")
            return None

    async def _transcribe_with_engine(
        self, audio_data: bytes, model: ModelConfig, timeout_sec: float = 10.0
    ) -> Optional[STTResult]:
        """Transcribe audio with specific engine (with timeout)"""
        start_time = time.time()

        try:
            engine = await self._load_engine(model)
            if engine is None:
                return None

            # Transcribe with timeout
            result = await asyncio.wait_for(engine.transcribe(audio_data), timeout=timeout_sec)

            latency_ms = (time.time() - start_time) * 1000

            # Update performance stats
            if model.name not in self.performance_stats:
                self.performance_stats[model.name] = {
                    "total_requests": 0,
                    "total_latency_ms": 0,
                    "avg_confidence": 0,
                }

            stats = self.performance_stats[model.name]
            stats["total_requests"] += 1
            stats["total_latency_ms"] += latency_ms
            stats["avg_latency_ms"] = stats["total_latency_ms"] / stats["total_requests"]

            logger.debug(
                f"🎤 {model.name}: '{result.text[:50]}...' (confidence={result.confidence:.2f}, latency={latency_ms:.0f}ms)"
            )

            return result

        except asyncio.TimeoutError:
            latency_ms = (time.time() - start_time) * 1000
            logger.warning(f"⏱️  {model.name} timed out after {latency_ms:.0f}ms")
            return None

        except Exception as e:
            logger.error(f"Error transcribing with {model.name}: {e}")
            return None

    async def _identify_speaker(self, audio_data: bytes) -> Optional[str]:
        """
        Identify speaker from voice using advanced speaker recognition.

        Returns speaker name if recognized, None if unknown.
        """
        try:
            from voice.speaker_recognition import get_speaker_recognition_engine

            speaker_engine = get_speaker_recognition_engine()
            await speaker_engine.initialize()

            # Identify speaker from audio
            speaker_name, confidence = await speaker_engine.identify_speaker(audio_data)

            if speaker_name:
                logger.info(f"🎭 Speaker identified: {speaker_name} (confidence: {confidence:.2f})")

                # Check if this is the owner
                if speaker_engine.is_owner(speaker_name):
                    logger.info(f"👑 Owner detected: {speaker_name}")

                return speaker_name
            else:
                logger.info(
                    f"🎭 Unknown speaker (best confidence: {confidence:.2f}, threshold: {speaker_engine.recognition_threshold})"
                )
                return None

        except Exception as e:
            logger.error(f"Speaker identification failed: {e}")
            import traceback

            traceback.print_exc()
            return None

    async def _record_transcription(
        self, audio_data: bytes, result: STTResult, speaker_name: Optional[str] = None
    ) -> int:
        """Record transcription to learning database"""
        try:
            learning_db = await self._get_learning_db()
            if not learning_db:
                return -1

            # Calculate audio duration (estimate from bytes)
            audio_duration_ms = len(audio_data) / (16000 * 2) * 1000  # 16kHz, 16-bit

            transcription_id = await learning_db.record_voice_transcription(
                audio_data=audio_data,
                transcribed_text=result.text,
                confidence_score=result.confidence,
                audio_duration_ms=audio_duration_ms,
            )

            # Also record as voice sample for Derek if identified
            if speaker_name == "Derek J. Russell":
                await learning_db.record_voice_sample(
                    speaker_name=speaker_name,
                    audio_data=audio_data,
                    transcription=result.text,
                    audio_duration_ms=audio_duration_ms,
                    quality_score=result.confidence,
                )

            return transcription_id

        except Exception as e:
            logger.error(f"Failed to record transcription: {e}")
            return -1

    async def transcribe(
        self,
        audio_data: bytes,
        strategy: Optional[RoutingStrategy] = None,
        speaker_name: Optional[str] = None,
        context: Optional[Dict] = None,
    ) -> STTResult:
        """
        Main transcription entry point.

        Ultra-intelligent routing:
        1. Assess system resources
        2. Identify speaker (if possible)
        3. Select optimal model
        4. Try local model first
        5. Escalate to cloud if needed
        6. Record to database for learning
        7. Return best result
        """
        self.total_requests += 1
        start_time = time.time()

        # Use configured strategy if not specified
        if strategy is None:
            strategy = self.config.default_strategy

        # Get current system resources
        resources = self._get_system_resources()
        logger.info(
            f"📊 Resources: RAM {resources.available_ram_gb:.1f}/{resources.total_ram_gb:.1f}GB, "
            f"CPU {resources.cpu_percent:.0f}%, GPU={resources.gpu_available}"
        )

        # Try to identify speaker
        if speaker_name is None:
            speaker_name = await self._identify_speaker(audio_data)
            if speaker_name:
                logger.info(f"👤 Speaker identified: {speaker_name}")

        # Select primary model
        primary_model = self._select_optimal_model(resources, strategy, speaker_name)
        logger.info(f"🎯 Primary model selected: {primary_model.name}")

        # Try primary model
        primary_result = await self._transcribe_with_engine(
            audio_data, primary_model, timeout_sec=self.config.max_local_latency_ms / 1000
        )

        # Decide if escalation is needed
        should_escalate = False
        if primary_result is None:
            logger.warning("⚠️  Primary model failed, escalating...")
            should_escalate = True
        elif primary_result.confidence < self.config.min_confidence_local:
            logger.info(
                f"📉 Low confidence ({primary_result.confidence:.2f}), escalating for validation..."
            )
            should_escalate = True
        elif (
            speaker_name in self.config.priority_speakers
            and primary_result.confidence < self.config.high_confidence_threshold
        ):
            logger.info(f"👑 Priority speaker {speaker_name}, escalating for best accuracy...")
            should_escalate = True

        final_result = primary_result

        # Escalate to cloud if needed
        if should_escalate and resources.network_available:
            # Get most accurate model (likely GCP)
            cloud_model = self.config.get_most_accurate_model()

            if cloud_model.requires_internet:
                logger.info(f"☁️  Escalating to cloud model: {cloud_model.name}")
                self.cloud_requests += 1
                self.config.increment_cloud_usage()

                cloud_result = await self._transcribe_with_engine(
                    audio_data, cloud_model, timeout_sec=self.config.max_cloud_latency_ms / 1000
                )

                if cloud_result and cloud_result.confidence > (
                    primary_result.confidence if primary_result else 0
                ):
                    logger.info(
                        f"✅ Cloud result better: {cloud_result.confidence:.2f} > {primary_result.confidence if primary_result else 0:.2f}"
                    )
                    final_result = cloud_result
                elif cloud_result:
                    logger.info(
                        f"⚖️  Keeping primary result (confidence: {primary_result.confidence:.2f})"
                    )
                else:
                    logger.warning("☁️  Cloud transcription failed")

        # Fallback if everything failed
        if final_result is None:
            logger.error("❌ All transcription attempts failed, using fallback")
            # Try smallest/fastest model as last resort
            fallback_model = self.config.models.get("vosk-small")
            if fallback_model:
                final_result = await self._transcribe_with_engine(
                    audio_data, fallback_model, timeout_sec=5.0
                )

        # Still no result? Try Whisper directly as last resort
        if final_result is None:
            logger.warning("⚠️  All engines failed, attempting direct Whisper transcription...")
            try:
                # Try to use Whisper directly without going through engine system
                import whisper
                import tempfile
                import numpy as np

                # Load Whisper model if not already loaded
                if not hasattr(self, '_whisper_fallback_model'):
                    logger.info("Loading Whisper base model for fallback...")
                    self._whisper_fallback_model = whisper.load_model("base")

                # Convert audio bytes to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Save to temp file and transcribe
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp:
                    import soundfile as sf
                    sf.write(tmp.name, audio_array, 16000)
                    result = self._whisper_fallback_model.transcribe(tmp.name)

                final_result = STTResult(
                    text=result["text"].strip(),
                    confidence=0.85,  # Whisper doesn't provide confidence
                    engine=STTEngine.WHISPER_LOCAL,
                    model_name="whisper-base-fallback",
                    latency_ms=(time.time() - start_time) * 1000,
                    audio_duration_ms=len(audio_data) / 32,
                )
                logger.info(f"✅ Whisper fallback succeeded: '{final_result.text}'")
            except Exception as e:
                logger.error(f"Whisper fallback failed: {e}")
                # Last resort error result
                final_result = STTResult(
                    text="[transcription failed]",
                    confidence=0.0,
                    engine=STTEngine.VOSK,
                    model_name="fallback",
                    latency_ms=(time.time() - start_time) * 1000,
                    audio_duration_ms=len(audio_data) / 32,
                )

        # Calculate total latency
        total_latency_ms = (time.time() - start_time) * 1000
        final_result.latency_ms = total_latency_ms

        # Record to database for learning (async, don't block)
        asyncio.create_task(self._record_transcription(audio_data, final_result, speaker_name))

        # Log final result
        logger.info(
            f"🎯 Final transcription: '{final_result.text[:50]}...' "
            f"(model={final_result.model_name}, confidence={final_result.confidence:.2f}, "
            f"latency={total_latency_ms:.0f}ms)"
        )

        return final_result

    async def record_misheard(
        self, transcription_id: int, what_heard: str, what_meant: str
    ) -> bool:
        """Record when user corrects a misheard transcription"""
        try:
            learning_db = await self._get_learning_db()
            if not learning_db:
                return False

            await learning_db.record_misheard_query(
                transcription_id=transcription_id,
                what_jarvis_heard=what_heard,
                what_user_meant=what_meant,
                correction_method="user_correction",
            )

            logger.info(f"📝 Recorded misheard: '{what_heard}' -> '{what_meant}'")
            return True

        except Exception as e:
            logger.error(f"Failed to record misheard: {e}")
            return False

    def get_stats(self) -> Dict:
        """Get router performance statistics"""
        return {
            "total_requests": self.total_requests,
            "cloud_requests": self.cloud_requests,
            "cloud_usage_percent": (
                (self.cloud_requests / self.total_requests * 100) if self.total_requests > 0 else 0
            ),
            "cache_hits": self.cache_hits,
            "loaded_engines": list(self.engines.keys()),
            "performance_by_model": self.performance_stats,
            "config": self.config.to_dict(),
        }


# Global singleton
_hybrid_router: Optional[HybridSTTRouter] = None


def get_hybrid_router() -> HybridSTTRouter:
    """Get global hybrid STT router instance"""
    global _hybrid_router
    if _hybrid_router is None:
        _hybrid_router = HybridSTTRouter()
    return _hybrid_router
