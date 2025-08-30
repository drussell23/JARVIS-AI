"""
Picovoice Porcupine Integration for efficient wake word detection
Optimized for low-latency, on-device processing
"""

import os
import numpy as np
import logging
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass
import asyncio
from collections import deque

try:
    import pvporcupine
    PICOVOICE_AVAILABLE = True
except ImportError:
    PICOVOICE_AVAILABLE = False
    print("Picovoice Porcupine not available. Install with: pip install pvporcupine")

try:
    from .config import VOICE_CONFIG
except ImportError:
    from config import VOICE_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class PicovoiceConfig:
    """Configuration for Picovoice integration"""
    access_key: Optional[str] = None
    keyword_paths: List[str] = None
    keywords: List[str] = None  # Built-in keywords like "jarvis"
    sensitivities: List[float] = None
    model_path: Optional[str] = None
    require_endpoint: bool = True

class PicovoiceWakeWordDetector:
    """
    High-performance wake word detection using Picovoice Porcupine
    - Runs entirely on-device with minimal CPU/RAM usage
    - Sub-50ms latency
    - Works in noisy environments
    """
    
    def __init__(self, config: PicovoiceConfig = None):
        if not PICOVOICE_AVAILABLE:
            raise ImportError("Picovoice Porcupine is not installed")
            
        self.config = config or PicovoiceConfig()
        
        # Get access key from config or environment
        self.access_key = self.config.access_key or VOICE_CONFIG.picovoice_access_key
        if not self.access_key:
            raise ValueError("Picovoice access key required. Set PICOVOICE_ACCESS_KEY env variable")
        
        # Default sensitivity for each keyword
        self.default_sensitivity = 0.5  # Middle ground
        
        # Initialize Porcupine
        self.porcupine = None
        self._init_porcupine()
        
        # Audio processing
        self.frame_length = 512  # Porcupine's required frame length
        self.sample_rate = 16000  # Porcupine's required sample rate
        self.audio_buffer = deque(maxlen=self.frame_length * 2)
        
        # Detection callback
        self.detection_callback: Optional[Callable] = None
        
        # Performance metrics
        self.total_frames_processed = 0
        self.detections = 0
        
    def _init_porcupine(self):
        """Initialize Porcupine with keywords"""
        try:
            # Prepare keywords
            keywords = self.config.keywords or ["jarvis"]
            keyword_paths = self.config.keyword_paths or []
            
            # Prepare sensitivities
            num_keywords = len(keywords) + len(keyword_paths)
            sensitivities = self.config.sensitivities
            if not sensitivities:
                sensitivities = [self.default_sensitivity] * num_keywords
            elif len(sensitivities) < num_keywords:
                # Pad with default
                sensitivities.extend([self.default_sensitivity] * (num_keywords - len(sensitivities)))
            
            # Create Porcupine instance
            create_params = {
                "access_key": self.access_key,
                "keywords": keywords if keywords else None,
                "keyword_paths": keyword_paths if keyword_paths else None,
                "sensitivities": sensitivities[:num_keywords]
            }
            
            if self.config.model_path:
                create_params["model_path"] = self.config.model_path
                
            self.porcupine = pvporcupine.create(**create_params)
            
            logger.info(f"Picovoice Porcupine initialized with keywords: {keywords}")
            logger.info(f"Frame length: {self.porcupine.frame_length}")
            logger.info(f"Sample rate: {self.porcupine.sample_rate}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Porcupine: {e}")
            raise
    
    def process_audio(self, audio_data: np.ndarray) -> Optional[int]:
        """
        Process audio data for wake word detection
        Returns: keyword index if detected, None otherwise
        """
        if not self.porcupine:
            return None
        
        # Ensure audio is in the correct format
        if audio_data.dtype != np.int16:
            # Convert float to int16
            if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                audio_data = (audio_data * 32767).astype(np.int16)
            else:
                audio_data = audio_data.astype(np.int16)
        
        # Process in frames
        keyword_index = None
        
        # Add to buffer
        self.audio_buffer.extend(audio_data)
        
        # Process complete frames
        while len(self.audio_buffer) >= self.frame_length:
            # Extract frame
            frame = np.array(list(self.audio_buffer)[:self.frame_length])
            
            # Remove processed samples
            for _ in range(self.frame_length):
                self.audio_buffer.popleft()
            
            # Process frame
            try:
                result = self.porcupine.process(frame)
                self.total_frames_processed += 1
                
                if result >= 0:
                    keyword_index = result
                    self.detections += 1
                    logger.info(f"Wake word detected! Keyword index: {result}")
                    
                    # Call callback if set
                    if self.detection_callback:
                        self.detection_callback(result)
                    
                    break  # Return on first detection
                    
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
        
        return keyword_index
    
    async def process_audio_async(self, audio_data: np.ndarray) -> Optional[int]:
        """Async wrapper for process_audio"""
        return await asyncio.to_thread(self.process_audio, audio_data)
    
    def update_sensitivity(self, keyword_index: int, sensitivity: float):
        """Update sensitivity for a specific keyword (requires reinit)"""
        if 0 <= keyword_index < len(self.config.sensitivities or []):
            if not self.config.sensitivities:
                self.config.sensitivities = [self.default_sensitivity] * self.porcupine.num_keywords
            
            self.config.sensitivities[keyword_index] = max(0.0, min(1.0, sensitivity))
            
            # Reinitialize with new sensitivity
            self.cleanup()
            self._init_porcupine()
            
            logger.info(f"Updated sensitivity for keyword {keyword_index} to {sensitivity}")
    
    def get_metrics(self) -> dict:
        """Get performance metrics"""
        return {
            "total_frames_processed": self.total_frames_processed,
            "total_detections": self.detections,
            "detection_rate": self.detections / self.total_frames_processed if self.total_frames_processed > 0 else 0,
            "frame_length": self.frame_length,
            "sample_rate": self.sample_rate
        }
    
    def cleanup(self):
        """Clean up Porcupine resources"""
        if self.porcupine:
            self.porcupine.delete()
            self.porcupine = None
            logger.info("Porcupine cleaned up")

class HybridWakeWordDetector:
    """
    Combines Picovoice for initial detection with ML for verification
    Best of both worlds: low latency + high accuracy
    """
    
    def __init__(self, ml_detector, picovoice_config: PicovoiceConfig = None):
        self.ml_detector = ml_detector
        self.picovoice_detector = None
        
        # Try to initialize Picovoice
        if PICOVOICE_AVAILABLE and VOICE_CONFIG.use_picovoice:
            try:
                self.picovoice_detector = PicovoiceWakeWordDetector(picovoice_config)
                logger.info("Hybrid mode: Picovoice + ML verification enabled")
            except Exception as e:
                logger.warning(f"Failed to init Picovoice, falling back to ML only: {e}")
        
        # Buffer for verification
        self.verification_buffer = deque(maxlen=16000 * 2)  # 2 seconds
        
    async def detect_wake_word(self, audio_data: np.ndarray, user_id: str = "default") -> Tuple[bool, float, Optional[str]]:
        """
        Hybrid detection: Picovoice for speed, ML for accuracy
        """
        # Add to verification buffer
        self.verification_buffer.extend(audio_data)
        
        # First, try Picovoice for fast detection
        if self.picovoice_detector:
            keyword_idx = self.picovoice_detector.process_audio(audio_data)
            
            if keyword_idx is not None:
                # Picovoice detected something - verify with ML
                logger.info("Picovoice triggered, verifying with ML...")
                
                # Use buffered audio for verification
                verify_audio = np.array(list(self.verification_buffer))
                
                # Run ML verification
                ml_result = await self.ml_detector.detect_wake_word(verify_audio, user_id)
                
                if ml_result[0]:  # ML confirms
                    logger.info(f"ML confirmed wake word with confidence {ml_result[1]:.2f}")
                    return ml_result
                else:
                    logger.info("ML rejected Picovoice detection as false positive")
                    # Could adjust Picovoice sensitivity here
                    return (False, 0.3, "Picovoice detected but ML rejected")
        
        # Fall back to pure ML detection
        return await self.ml_detector.detect_wake_word(audio_data, user_id)
    
    def cleanup(self):
        """Clean up resources"""
        if self.picovoice_detector:
            self.picovoice_detector.cleanup()

# Example usage
async def test_picovoice():
    """Test Picovoice integration"""
    if not PICOVOICE_AVAILABLE:
        print("Picovoice not available")
        return
    
    # Create config
    config = PicovoiceConfig(
        keywords=["jarvis"],
        sensitivities=[0.5]
    )
    
    # Create detector
    detector = PicovoiceWakeWordDetector(config)
    
    # Simulate audio processing
    print("Picovoice wake word detector ready")
    print(f"Say one of: {config.keywords}")
    
    # In real usage, this would be audio from microphone
    # For testing, generate some dummy audio
    sample_rate = detector.sample_rate
    duration = 1.0
    
    for i in range(5):
        # Generate dummy audio
        audio = np.random.randn(int(sample_rate * duration)) * 0.1
        audio = (audio * 32767).astype(np.int16)
        
        # Process
        result = await detector.process_audio_async(audio)
        if result is not None:
            print(f"Wake word detected: keyword index {result}")
        else:
            print("No wake word detected")
        
        await asyncio.sleep(0.5)
    
    # Show metrics
    print("\nMetrics:", detector.get_metrics())
    
    # Cleanup
    detector.cleanup()

if __name__ == "__main__":
    asyncio.run(test_picovoice())