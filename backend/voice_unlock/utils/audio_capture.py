"""
Audio Capture Utilities
======================

Provides cross-platform audio capture with real-time monitoring
and quality assessment for voice enrollment and verification.
"""

import numpy as np
import pyaudio
import threading
import queue
from typing import Optional, Callable, Tuple
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Audio capture configuration"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    format: int = pyaudio.paInt16
    
    @property
    def bytes_per_sample(self) -> int:
        return pyaudio.get_sample_size(self.format)


class AudioCapture:
    """Real-time audio capture with quality monitoring"""
    
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.capture_thread = None
        self.audio_queue = queue.Queue()
        self.is_capturing = False
        self.callbacks = []
        
        # Quality monitoring
        self.noise_floor = None
        self.calibrated = False
        
    def add_callback(self, callback: Callable[[np.ndarray], None]):
        """Add callback for audio chunks (useful for visualization)"""
        self.callbacks.append(callback)
        
    def calibrate_noise_floor(self, duration: float = 1.0) -> float:
        """Calibrate noise floor for better voice detection"""
        logger.info("Calibrating noise floor...")
        
        # Capture ambient noise
        noise_samples = self.capture_audio(duration, silent=True)
        
        # Calculate noise statistics
        self.noise_floor = np.sqrt(np.mean(noise_samples ** 2))
        self.calibrated = True
        
        logger.info(f"Noise floor calibrated: {self.noise_floor:.4f}")
        return self.noise_floor
        
    def capture_audio(self, duration: float, silent: bool = False) -> np.ndarray:
        """Capture audio for specified duration"""
        if not silent:
            logger.info(f"Starting audio capture for {duration} seconds")
            
        frames = []
        
        try:
            self.stream = self.audio.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size
            )
            
            total_chunks = int(self.config.sample_rate * duration / self.config.chunk_size)
            
            for _ in range(total_chunks):
                data = self.stream.read(self.config.chunk_size, exception_on_overflow=False)
                frames.append(data)
                
                # Convert to numpy for callbacks
                audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Trigger callbacks (for visualization)
                for callback in self.callbacks:
                    callback(audio_chunk)
                    
        except Exception as e:
            logger.error(f"Audio capture error: {e}")
            raise
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                
        # Convert to numpy array
        audio_data = b''.join(frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        return audio_array
        
    def capture_with_vad(self, max_duration: float = 10.0, 
                        silence_threshold: float = 0.02,
                        silence_duration: float = 1.5) -> Tuple[np.ndarray, bool]:
        """Capture audio with voice activity detection"""
        logger.info("Starting VAD-based capture")
        
        # Calibrate if not done
        if not self.calibrated:
            self.calibrate_noise_floor()
            
        frames = []
        is_speaking = False
        silence_start = None
        capture_start = time.time()
        
        try:
            self.stream = self.audio.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size
            )
            
            while True:
                # Check max duration
                if time.time() - capture_start > max_duration:
                    logger.warning("Max capture duration reached")
                    break
                    
                # Read chunk
                data = self.stream.read(self.config.chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Calculate energy
                energy = np.sqrt(np.mean(audio_chunk ** 2))
                
                # Adjust threshold based on noise floor
                adaptive_threshold = max(silence_threshold, self.noise_floor * 3)
                
                # Voice activity detection
                if energy > adaptive_threshold:
                    is_speaking = True
                    silence_start = None
                    frames.append(data)
                    
                    # Trigger callbacks
                    for callback in self.callbacks:
                        callback(audio_chunk)
                        
                elif is_speaking:
                    # Speaking stopped, start silence timer
                    frames.append(data)
                    
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > silence_duration:
                        logger.info("Silence detected, stopping capture")
                        break
                        
        except Exception as e:
            logger.error(f"VAD capture error: {e}")
            raise
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                
        # Convert to numpy array
        if frames:
            audio_data = b''.join(frames)
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            return audio_array, True
        else:
            return np.array([]), False
            
    def start_continuous_capture(self):
        """Start continuous audio capture in background thread"""
        if self.is_capturing:
            logger.warning("Capture already in progress")
            return
            
        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.start()
        logger.info("Started continuous capture")
        
    def stop_continuous_capture(self):
        """Stop continuous audio capture"""
        if not self.is_capturing:
            return
            
        self.is_capturing = False
        if self.capture_thread:
            self.capture_thread.join()
        logger.info("Stopped continuous capture")
        
    def _capture_loop(self):
        """Background capture loop"""
        try:
            self.stream = self.audio.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size
            )
            
            while self.is_capturing:
                data = self.stream.read(self.config.chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Add to queue
                self.audio_queue.put(audio_chunk)
                
                # Trigger callbacks
                for callback in self.callbacks:
                    callback(audio_chunk)
                    
        except Exception as e:
            logger.error(f"Capture loop error: {e}")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                
    def get_audio_chunk(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get audio chunk from queue"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def list_devices(self) -> list:
        """List available audio input devices"""
        devices = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': info['name'],
                    'channels': info['maxInputChannels'],
                    'sample_rate': info['defaultSampleRate']
                })
        return devices
        
    def __del__(self):
        """Cleanup resources"""
        self.stop_continuous_capture()
        if self.audio:
            self.audio.terminate()


class AudioVisualizer:
    """Real-time audio visualization for enrollment UI"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.history_size = 100
        self.energy_history = []
        self.pitch_history = []
        
    def update(self, audio_chunk: np.ndarray) -> dict:
        """Update visualization data"""
        # Calculate energy
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        self.energy_history.append(energy)
        if len(self.energy_history) > self.history_size:
            self.energy_history.pop(0)
            
        # Simple pitch estimation (zero-crossing rate)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_chunk)))) / 2
        pitch_estimate = zero_crossings * self.sample_rate / (2 * len(audio_chunk))
        self.pitch_history.append(pitch_estimate)
        if len(self.pitch_history) > self.history_size:
            self.pitch_history.pop(0)
            
        # Calculate spectrum for visualization
        spectrum = np.abs(np.fft.rfft(audio_chunk * np.hanning(len(audio_chunk))))
        spectrum_db = 20 * np.log10(spectrum + 1e-10)
        
        return {
            'energy': energy,
            'energy_history': self.energy_history,
            'pitch_estimate': pitch_estimate,
            'pitch_history': self.pitch_history,
            'spectrum': spectrum_db[:50],  # First 50 bins for visualization
            'waveform': audio_chunk[::10]  # Downsample for display
        }


def test_audio_capture():
    """Test audio capture functionality"""
    print("Testing audio capture...")
    
    # Create capture instance
    capture = AudioCapture()
    
    # List devices
    print("\nAvailable audio devices:")
    for device in capture.list_devices():
        print(f"  {device['index']}: {device['name']} ({device['channels']} channels)")
        
    # Test calibration
    print("\n1. Calibrating noise floor (please be quiet)...")
    noise_level = capture.calibrate_noise_floor()
    print(f"   Noise floor: {noise_level:.4f}")
    
    # Test fixed duration capture
    print("\n2. Testing 3-second capture (please speak)...")
    audio = capture.capture_audio(3.0)
    print(f"   Captured {len(audio)/capture.config.sample_rate:.2f} seconds of audio")
    print(f"   Energy level: {np.sqrt(np.mean(audio**2)):.4f}")
    
    # Test VAD capture
    print("\n3. Testing VAD capture (speak, then pause for 1.5 seconds)...")
    audio_vad, detected = capture.capture_with_vad(max_duration=10.0)
    if detected:
        print(f"   Captured {len(audio_vad)/capture.config.sample_rate:.2f} seconds with VAD")
    else:
        print("   No speech detected")
        
    print("\nAudio capture test complete!")


if __name__ == "__main__":
    test_audio_capture()