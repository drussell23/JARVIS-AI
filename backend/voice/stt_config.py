"""
Dynamic Speech-to-Text Configuration System
Zero hardcoding - all values loaded from environment/config
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class STTEngine(Enum):
    """Available STT engines"""

    WAV2VEC = "wav2vec2"
    VOSK = "vosk"
    WHISPER_LOCAL = "whisper_local"
    WHISPER_GCP = "whisper_gcp"
    SPEECHBRAIN = "speechbrain"
    BROWSER_API = "browser_api"


class RoutingStrategy(Enum):
    """STT routing strategies"""

    SPEED = "speed"  # Always local
    ACCURACY = "accuracy"  # Always best model
    BALANCED = "balanced"  # Smart routing
    COST = "cost"  # Minimize cloud usage
    ADAPTIVE = "adaptive"  # Learn from usage patterns


@dataclass
class ModelConfig:
    """Configuration for an STT model"""

    name: str
    engine: STTEngine
    disk_size_mb: float
    ram_required_gb: float
    vram_required_gb: float = 0.0
    expected_accuracy: float = 0.0
    avg_latency_ms: float = 0.0
    cost_per_minute: float = 0.0
    requires_gpu: bool = False
    requires_internet: bool = False
    supports_fine_tuning: bool = False
    supports_streaming: bool = False
    model_path: Optional[str] = None
    download_url: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class STTConfig:
    """Master STT configuration - loaded from environment/file"""

    # Routing strategy
    default_strategy: RoutingStrategy = RoutingStrategy.BALANCED
    fallback_strategy: RoutingStrategy = RoutingStrategy.SPEED

    # RAM thresholds (in GB)
    min_ram_for_wav2vec: float = 2.0
    min_ram_for_whisper_small: float = 3.0
    min_ram_for_whisper_medium: float = 6.0

    # Confidence thresholds
    min_confidence_local: float = 0.75  # Below this, escalate to cloud
    min_confidence_acceptable: float = 0.60  # Below this, flag for review
    high_confidence_threshold: float = 0.90  # Above this, skip validation

    # Performance thresholds
    max_local_latency_ms: float = 500.0  # Switch to faster model if exceeded
    max_cloud_latency_ms: float = 2000.0  # Timeout for cloud requests

    # Cost optimization
    max_daily_cloud_requests: int = 1000  # Rate limit cloud usage
    cloud_request_count: int = 0  # Current count (resets daily)

    # Priority users (get best models)
    priority_speakers: List[str] = field(default_factory=lambda: ["Derek J. Russell"])

    # Model configurations
    models: Dict[str, ModelConfig] = field(default_factory=dict)

    # Paths
    models_dir: Path = field(default_factory=lambda: Path.home() / ".jarvis" / "models" / "stt")
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".jarvis" / "cache" / "stt")

    # GCP Configuration
    gcp_project_id: Optional[str] = None
    gcp_region: str = "us-central1"
    gcp_vm_enabled: bool = True

    # Learning parameters
    enable_learning: bool = True
    min_samples_for_fine_tuning: int = 100
    auto_fine_tune: bool = True

    # Advanced features
    enable_speaker_diarization: bool = True
    enable_punctuation: bool = True
    enable_timestamps: bool = True

    def __post_init__(self):
        """Load configuration from environment and ensure directories exist"""
        self._load_from_environment()
        self._initialize_model_configs()
        self._ensure_directories()

    def _load_from_environment(self):
        """Load config from environment variables"""
        # Routing strategy
        strategy = os.getenv("JARVIS_STT_STRATEGY", "balanced")
        try:
            self.default_strategy = RoutingStrategy(strategy)
        except ValueError:
            pass

        # RAM thresholds
        self.min_ram_for_wav2vec = float(os.getenv("JARVIS_MIN_RAM_WAV2VEC", "2.0"))
        self.min_ram_for_whisper_small = float(os.getenv("JARVIS_MIN_RAM_WHISPER_SMALL", "3.0"))

        # Confidence thresholds
        self.min_confidence_local = float(os.getenv("JARVIS_MIN_CONFIDENCE_LOCAL", "0.75"))
        self.min_confidence_acceptable = float(
            os.getenv("JARVIS_MIN_CONFIDENCE_ACCEPTABLE", "0.60")
        )

        # GCP config
        self.gcp_project_id = os.getenv("GCP_PROJECT_ID")
        self.gcp_region = os.getenv("GCP_REGION", "us-central1")
        self.gcp_vm_enabled = os.getenv("JARVIS_GCP_STT_ENABLED", "true").lower() == "true"

        # Learning
        self.enable_learning = os.getenv("JARVIS_STT_LEARNING", "true").lower() == "true"
        self.auto_fine_tune = os.getenv("JARVIS_AUTO_FINE_TUNE", "true").lower() == "true"

    def _initialize_model_configs(self):
        """Initialize model configurations with best practices"""
        self.models = {
            # Wav2Vec 2.0 models (local, fine-tunable)
            "wav2vec2-base": ModelConfig(
                name="wav2vec2-base",
                engine=STTEngine.WAV2VEC,
                disk_size_mb=360,
                ram_required_gb=1.5,
                vram_required_gb=1.5,
                expected_accuracy=0.93,
                avg_latency_ms=150,
                cost_per_minute=0.0,
                requires_gpu=True,
                supports_fine_tuning=True,
                supports_streaming=True,
                model_path="facebook/wav2vec2-base-960h",
            ),
            "wav2vec2-large": ModelConfig(
                name="wav2vec2-large",
                engine=STTEngine.WAV2VEC,
                disk_size_mb=1200,
                ram_required_gb=4.0,
                vram_required_gb=4.0,
                expected_accuracy=0.95,
                avg_latency_ms=300,
                cost_per_minute=0.0,
                requires_gpu=True,
                supports_fine_tuning=True,
                supports_streaming=True,
                model_path="facebook/wav2vec2-large-960h-lv60-self",
            ),
            # Vosk models (local, fast)
            "vosk-small": ModelConfig(
                name="vosk-small",
                engine=STTEngine.VOSK,
                disk_size_mb=40,
                ram_required_gb=0.5,
                expected_accuracy=0.88,
                avg_latency_ms=50,
                cost_per_minute=0.0,
                requires_gpu=False,
                supports_streaming=True,
                model_path="vosk-model-small-en-us-0.15",
                download_url="https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
            ),
            "vosk-large": ModelConfig(
                name="vosk-large",
                engine=STTEngine.VOSK,
                disk_size_mb=1800,
                ram_required_gb=2.0,
                expected_accuracy=0.92,
                avg_latency_ms=100,
                cost_per_minute=0.0,
                requires_gpu=False,
                supports_streaming=True,
                model_path="vosk-model-en-us-0.22",
                download_url="https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",
            ),
            # Whisper models (local/GCP)
            "whisper-tiny": ModelConfig(
                name="whisper-tiny",
                engine=STTEngine.WHISPER_LOCAL,
                disk_size_mb=75,
                ram_required_gb=1.0,
                vram_required_gb=1.0,
                expected_accuracy=0.76,
                avg_latency_ms=200,
                cost_per_minute=0.0,
                requires_gpu=False,
                model_path="tiny",
            ),
            "whisper-base": ModelConfig(
                name="whisper-base",
                engine=STTEngine.WHISPER_LOCAL,
                disk_size_mb=142,
                ram_required_gb=1.5,
                vram_required_gb=1.5,
                expected_accuracy=0.85,
                avg_latency_ms=250,
                cost_per_minute=0.0,
                requires_gpu=False,
                model_path="base",
            ),
            "whisper-small": ModelConfig(
                name="whisper-small",
                engine=STTEngine.WHISPER_LOCAL,
                disk_size_mb=466,
                ram_required_gb=2.0,
                vram_required_gb=2.0,
                expected_accuracy=0.91,
                avg_latency_ms=300,
                cost_per_minute=0.0,
                requires_gpu=False,
                model_path="small",
            ),
            "whisper-medium": ModelConfig(
                name="whisper-medium",
                engine=STTEngine.WHISPER_LOCAL,
                disk_size_mb=1500,
                ram_required_gb=5.0,
                vram_required_gb=5.0,
                expected_accuracy=0.95,
                avg_latency_ms=500,
                cost_per_minute=0.0,
                requires_gpu=True,
                model_path="medium",
            ),
            "whisper-large-v3-gcp": ModelConfig(
                name="whisper-large-v3-gcp",
                engine=STTEngine.WHISPER_GCP,
                disk_size_mb=3000,
                ram_required_gb=11.0,
                vram_required_gb=11.0,
                expected_accuracy=0.99,
                avg_latency_ms=500,
                cost_per_minute=0.006,  # $0.006/min on GCP
                requires_gpu=True,
                requires_internet=True,
            ),
            # SpeechBrain models (local, adaptive, noise-robust)
            "speechbrain-asr-crdnn": ModelConfig(
                name="speechbrain-asr-crdnn",
                engine=STTEngine.SPEECHBRAIN,
                disk_size_mb=450,
                ram_required_gb=2.5,
                vram_required_gb=2.0,
                expected_accuracy=0.94,
                avg_latency_ms=180,
                cost_per_minute=0.0,
                requires_gpu=False,  # Works on CPU but faster on GPU
                supports_fine_tuning=True,
                supports_streaming=True,
                model_path="speechbrain/asr-crdnn-rnnlm-librispeech",
                metadata={
                    "description": "CRDNN with RNN language model - robust to noise",
                    "specialization": "command_words",
                    "noise_robustness": "high",
                },
            ),
            "speechbrain-wav2vec2": ModelConfig(
                name="speechbrain-wav2vec2",
                engine=STTEngine.SPEECHBRAIN,
                disk_size_mb=380,
                ram_required_gb=2.0,
                vram_required_gb=1.8,
                expected_accuracy=0.96,
                avg_latency_ms=150,
                cost_per_minute=0.0,
                requires_gpu=False,
                supports_fine_tuning=True,
                supports_streaming=True,
                model_path="speechbrain/asr-wav2vec2-commonvoice-en",
                metadata={
                    "description": "Wav2Vec2-based for high accuracy",
                    "specialization": "general",
                    "noise_robustness": "medium",
                },
            ),
            "speechbrain-transformer": ModelConfig(
                name="speechbrain-transformer",
                engine=STTEngine.SPEECHBRAIN,
                disk_size_mb=520,
                ram_required_gb=3.0,
                vram_required_gb=2.5,
                expected_accuracy=0.97,
                avg_latency_ms=220,
                cost_per_minute=0.0,
                requires_gpu=False,
                supports_fine_tuning=True,
                supports_streaming=False,
                model_path="speechbrain/asr-transformer-transformerlm-librispeech",
                metadata={
                    "description": "Transformer-based for long-form speech",
                    "specialization": "long_form",
                    "noise_robustness": "medium",
                },
            ),
        }

    def _ensure_directories(self):
        """Ensure required directories exist"""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_model_for_ram(
        self, available_ram_gb: float, engine: Optional[STTEngine] = None
    ) -> Optional[ModelConfig]:
        """Get best model that fits in available RAM"""
        compatible_models = [
            model
            for model in self.models.values()
            if model.ram_required_gb <= available_ram_gb
            and (engine is None or model.engine == engine)
        ]

        if not compatible_models:
            return None

        # Sort by accuracy (descending)
        compatible_models.sort(key=lambda m: m.expected_accuracy, reverse=True)
        return compatible_models[0]

    def get_fastest_model(self, available_ram_gb: float) -> Optional[ModelConfig]:
        """Get fastest model that fits in RAM"""
        compatible_models = [
            model
            for model in self.models.values()
            if model.ram_required_gb <= available_ram_gb and not model.requires_internet
        ]

        if not compatible_models:
            return None

        # Sort by latency (ascending)
        compatible_models.sort(key=lambda m: m.avg_latency_ms)
        return compatible_models[0]

    def get_most_accurate_model(self, available_ram_gb: float = 100.0) -> ModelConfig:
        """Get most accurate model (may be on GCP)"""
        all_models = list(self.models.values())
        all_models.sort(key=lambda m: m.expected_accuracy, reverse=True)

        # Find first model that fits
        for model in all_models:
            if model.ram_required_gb <= available_ram_gb or model.requires_internet:
                return model

        # Fallback to any model
        return all_models[0]

    def should_use_cloud(
        self,
        local_confidence: float,
        speaker_name: Optional[str] = None,
        available_ram_gb: float = 16.0,
    ) -> bool:
        """Determine if cloud STT should be used"""
        # Priority speakers always get cloud if needed
        if (
            speaker_name in self.priority_speakers
            and local_confidence < self.high_confidence_threshold
        ):
            return True

        # Low confidence always escalates to cloud
        if local_confidence < self.min_confidence_local:
            return True

        # Check cloud request limits
        if self.cloud_request_count >= self.max_daily_cloud_requests:
            return False

        # If local RAM is very tight, use cloud
        if available_ram_gb < self.min_ram_for_wav2vec:
            return True

        return False

    def increment_cloud_usage(self):
        """Track cloud usage"""
        self.cloud_request_count += 1

    def to_dict(self) -> Dict:
        """Serialize config to dictionary"""
        return {
            "default_strategy": self.default_strategy.value,
            "min_confidence_local": self.min_confidence_local,
            "models": {name: vars(model) for name, model in self.models.items()},
            "cloud_usage": {
                "count": self.cloud_request_count,
                "limit": self.max_daily_cloud_requests,
            },
        }

    def save(self, path: Optional[Path] = None):
        """Save configuration to file"""
        if path is None:
            path = self.models_dir / "config.json"

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "STTConfig":
        """Load configuration from file"""
        config = cls()

        if path is None:
            path = config.models_dir / "config.json"

        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)

            # Apply loaded values
            if "default_strategy" in data:
                config.default_strategy = RoutingStrategy(data["default_strategy"])
            if "min_confidence_local" in data:
                config.min_confidence_local = data["min_confidence_local"]
            if "cloud_usage" in data:
                config.cloud_request_count = data["cloud_usage"].get("count", 0)

        return config


# Global singleton instance
_stt_config: Optional[STTConfig] = None


def get_stt_config() -> STTConfig:
    """Get global STT configuration instance"""
    global _stt_config
    if _stt_config is None:
        _stt_config = STTConfig.load()
    return _stt_config


def reset_stt_config():
    """Reset configuration (for testing)"""
    global _stt_config
    _stt_config = None
