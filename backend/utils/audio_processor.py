import numpy as np
import scipy.signal as signal
import scipy.fft as fft
from typing import Union, Tuple, Optional, List, Dict, Callable
import wave
import struct
import threading
import queue
import time
from dataclasses import dataclass
from enum import Enum
import noisereduce as nr
import librosa
import soundfile as sf
from collections import deque
import asyncio
import websockets
import json
import base64

class AudioFormat(Enum):
    """Audio format specifications"""
    PCM_16 = "pcm16"
    PCM_32 = "pcm32"
    FLOAT32 = "float32"
    
    
class ProcessingMode(Enum):
    """Audio processing modes"""
    REALTIME = "realtime"
    BATCH = "batch"
    STREAMING = "streaming"

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    format: AudioFormat = AudioFormat.PCM_16
    
    # Noise reduction
    noise_reduction: bool = True
    noise_threshold: float = 0.02
    
    # Voice Activity Detection
    vad_enabled: bool = True
    vad_threshold: float = 0.01
    vad_min_duration: float = 0.3  # seconds
    
    # Audio enhancement
    normalize: bool = True
    compress_dynamics: bool = True
    remove_silence: bool = True
    
    # Streaming
    buffer_size: int = 10  # seconds
    latency_target: float = 0.1  # seconds

@dataclass
class AudioMetrics:
    """Audio analysis metrics"""
    rms_level: float
    peak_level: float
    noise_floor: float
    snr: float  # Signal-to-Noise Ratio
    is_speech: bool
    frequency_centroid: float
    zero_crossing_rate: float

class AudioStreamProcessor:
    """Real-time audio stream processing with advanced features"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.is_running = False
        
        # Buffers
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.audio_buffer = deque(maxlen=int(config.sample_rate * config.buffer_size))
        
        # Noise profile
        self.noise_profile = None
        self.calibration_samples = []
        
        # VAD state
        self.vad_state = False
        self.speech_buffer = []
        self.silence_duration = 0
        
        # Processing threads
        self.processing_thread = None
        self.streaming_thread = None
        
        # Callbacks
        self.speech_callback = None
        self.metrics_callback = None
        
    def start(self):
        """Start audio processing"""
        self.is_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio_stream)
        self.processing_thread.start()
        
        print("Audio processor started")
        
    def stop(self):
        """Stop audio processing"""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join()
            
        print("Audio processor stopped")
        
    def process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Process a single audio chunk"""
        # Convert to float32 if needed
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32) / 32768.0
            
        # Apply processing chain
        processed = audio_chunk
        
        if self.config.noise_reduction and self.noise_profile is not None:
            processed = self._reduce_noise(processed)
            
        if self.config.normalize:
            processed = self._normalize_audio(processed)
            
        if self.config.compress_dynamics:
            processed = self._compress_dynamics(processed)
            
        # Compute metrics
        metrics = self._compute_metrics(processed)
        
        # Voice activity detection
        if self.config.vad_enabled:
            is_speech = self._detect_voice_activity(processed, metrics)
            metrics.is_speech = is_speech
            
        # Call metrics callback if set
        if self.metrics_callback:
            self.metrics_callback(metrics)
            
        return processed
        
    def _process_audio_stream(self):
        """Main processing loop"""
        while self.is_running:
            try:
                # Get audio chunk from input queue
                if not self.input_queue.empty():
                    audio_chunk = self.input_queue.get(timeout=0.1)
                    
                    # Process chunk
                    processed = self.process_chunk(audio_chunk)
                    
                    # Add to output queue
                    self.output_queue.put(processed)
                    
                    # Add to buffer
                    self.audio_buffer.extend(processed)
                    
                else:
                    time.sleep(0.01)  # Small delay to prevent busy waiting
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                
    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise reduction"""
        try:
            # Use spectral subtraction
            if self.noise_profile is None:
                return audio
                
            # Apply noise reduction
            reduced = nr.reduce_noise(
                y=audio,
                sr=self.config.sample_rate,
                stationary=True,
                prop_decrease=0.8
            )
            
            return reduced
            
        except Exception as e:
            print(f"Noise reduction error: {e}")
            return audio
            
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio levels"""
        # Find peak
        peak = np.max(np.abs(audio))
        
        if peak > 0:
            # Normalize to 0.9 to avoid clipping
            normalized = audio * (0.9 / peak)
            return normalized
        
        return audio
        
    def _compress_dynamics(self, audio: np.ndarray, ratio: float = 4.0, threshold: float = 0.5) -> np.ndarray:
        """Apply dynamic range compression"""
        # Simple compressor
        compressed = np.copy(audio)
        
        # Find samples above threshold
        above_threshold = np.abs(compressed) > threshold
        
        # Apply compression
        compressed[above_threshold] = threshold + (compressed[above_threshold] - threshold) / ratio
        
        return compressed
        
    def _compute_metrics(self, audio: np.ndarray) -> AudioMetrics:
        """Compute audio metrics"""
        # RMS level
        rms = np.sqrt(np.mean(audio**2))
        
        # Peak level
        peak = np.max(np.abs(audio))
        
        # Estimate noise floor (lower 10th percentile)
        noise_floor = np.percentile(np.abs(audio), 10)
        
        # SNR
        if noise_floor > 0:
            snr = 20 * np.log10(rms / noise_floor)
        else:
            snr = 40.0  # Default high SNR
            
        # Frequency centroid
        fft_data = np.fft.rfft(audio)
        magnitude = np.abs(fft_data)
        frequencies = np.fft.rfftfreq(len(audio), 1/self.config.sample_rate)
        
        if np.sum(magnitude) > 0:
            frequency_centroid = np.sum(frequencies * magnitude) / np.sum(magnitude)
        else:
            frequency_centroid = 0
            
        # Zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
        
        return AudioMetrics(
            rms_level=float(rms),
            peak_level=float(peak),
            noise_floor=float(noise_floor),
            snr=float(snr),
            is_speech=False,  # Will be set by VAD
            frequency_centroid=float(frequency_centroid),
            zero_crossing_rate=float(zero_crossings)
        )
        
    def _detect_voice_activity(self, audio: np.ndarray, metrics: AudioMetrics) -> bool:
        """Detect voice activity in audio"""
        # Simple energy-based VAD with frequency analysis
        
        # Energy threshold
        if metrics.rms_level < self.config.vad_threshold:
            return False
            
        # Frequency range check (human speech is typically 80-8000 Hz)
        if metrics.frequency_centroid < 80 or metrics.frequency_centroid > 8000:
            return False
            
        # Zero crossing rate check (speech has moderate ZCR)
        if metrics.zero_crossing_rate < 0.02 or metrics.zero_crossing_rate > 0.5:
            return False
            
        # SNR check
        if metrics.snr < 10:  # 10 dB minimum SNR
            return False
            
        return True
        
    def calibrate_noise(self, duration: float = 1.0):
        """Calibrate noise profile from ambient sound"""
        print(f"Calibrating noise profile for {duration} seconds...")
        
        self.calibration_samples = []
        calibration_start = time.time()
        
        while time.time() - calibration_start < duration:
            if not self.input_queue.empty():
                chunk = self.input_queue.get()
                self.calibration_samples.append(chunk)
                
        if self.calibration_samples:
            # Compute noise profile
            noise_sample = np.concatenate(self.calibration_samples)
            self.noise_profile = self._compute_noise_profile(noise_sample)
            print("Noise calibration complete")
        else:
            print("No audio samples for calibration")
            
    def _compute_noise_profile(self, noise_sample: np.ndarray) -> Dict:
        """Compute noise profile from sample"""
        # Convert to float32
        if noise_sample.dtype != np.float32:
            noise_sample = noise_sample.astype(np.float32) / 32768.0
            
        # Compute spectral characteristics
        fft_noise = np.fft.rfft(noise_sample)
        magnitude = np.abs(fft_noise)
        
        return {
            "magnitude_mean": np.mean(magnitude),
            "magnitude_std": np.std(magnitude),
            "rms": np.sqrt(np.mean(noise_sample**2))
        }
        
    def add_audio(self, audio_data: Union[bytes, np.ndarray]):
        """Add audio data to processing queue"""
        if isinstance(audio_data, bytes):
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
        else:
            audio_array = audio_data
            
        self.input_queue.put(audio_array)
        
    def get_processed_audio(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get processed audio from output queue"""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

class VoiceActivityDetector:
    """Advanced Voice Activity Detection"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        
        # State machine
        self.state = "silence"  # silence, speech, trailing
        self.speech_start = None
        self.silence_start = None
        
        # Buffers
        self.energy_buffer = deque(maxlen=50)  # 50 chunks history
        self.speech_buffer = []
        
        # Thresholds
        self.energy_threshold = config.vad_threshold
        self.min_speech_duration = config.vad_min_duration
        self.max_silence_duration = 0.5  # seconds
        
        # Callbacks
        self.on_speech_start = None
        self.on_speech_end = None
        
    def process(self, audio_chunk: np.ndarray, metrics: AudioMetrics) -> Tuple[str, Optional[np.ndarray]]:
        """
        Process audio chunk for VAD
        
        Returns:
            Tuple of (state, speech_audio) where speech_audio is complete speech segment
        """
        # Add energy to buffer
        self.energy_buffer.append(metrics.rms_level)
        
        # Compute adaptive threshold
        if len(self.energy_buffer) > 10:
            energy_mean = np.mean(list(self.energy_buffer))
            energy_std = np.std(list(self.energy_buffer))
            adaptive_threshold = energy_mean + 2 * energy_std
            self.energy_threshold = max(self.config.vad_threshold, adaptive_threshold)
            
        # State machine
        current_time = time.time()
        
        if self.state == "silence":
            if metrics.is_speech and metrics.rms_level > self.energy_threshold:
                # Transition to speech
                self.state = "speech"
                self.speech_start = current_time
                self.speech_buffer = [audio_chunk]
                
                if self.on_speech_start:
                    self.on_speech_start()
                    
        elif self.state == "speech":
            self.speech_buffer.append(audio_chunk)
            
            if not metrics.is_speech or metrics.rms_level < self.energy_threshold:
                # Transition to trailing silence
                self.state = "trailing"
                self.silence_start = current_time
                
        elif self.state == "trailing":
            self.speech_buffer.append(audio_chunk)
            
            if metrics.is_speech and metrics.rms_level > self.energy_threshold:
                # Back to speech
                self.state = "speech"
                self.silence_start = None
            elif current_time - self.silence_start > self.max_silence_duration:
                # End of speech
                self.state = "silence"
                
                # Check minimum duration
                speech_duration = current_time - self.speech_start
                if speech_duration >= self.min_speech_duration:
                    # Valid speech segment
                    speech_audio = np.concatenate(self.speech_buffer)
                    
                    if self.on_speech_end:
                        self.on_speech_end(speech_audio)
                        
                    self.speech_buffer = []
                    return "speech_end", speech_audio
                else:
                    # Too short, discard
                    self.speech_buffer = []
                    
        return self.state, None

class AudioFeedbackGenerator:
    """Generate audio feedback and confirmations"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
        # Pre-generate common feedback sounds
        self.feedback_sounds = {
            "beep": self._generate_beep(440, 0.1),  # A4 note, 100ms
            "double_beep": self._generate_double_beep(440, 0.1),
            "success": self._generate_success_sound(),
            "error": self._generate_error_sound(),
            "listening": self._generate_listening_sound(),
            "processing": self._generate_processing_sound()
        }
        
    def _generate_beep(self, frequency: float, duration: float) -> np.ndarray:
        """Generate a simple beep"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Generate sine wave
        beep = np.sin(2 * np.pi * frequency * t)
        
        # Apply envelope to avoid clicks
        envelope = np.ones_like(beep)
        fade_samples = int(0.01 * self.sample_rate)  # 10ms fade
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        
        return (beep * envelope * 0.5).astype(np.float32)
        
    def _generate_double_beep(self, frequency: float, duration: float) -> np.ndarray:
        """Generate a double beep"""
        beep = self._generate_beep(frequency, duration)
        silence = np.zeros(int(self.sample_rate * 0.05))  # 50ms silence
        
        return np.concatenate([beep, silence, beep])
        
    def _generate_success_sound(self) -> np.ndarray:
        """Generate a pleasant success sound (ascending notes)"""
        notes = [440, 554, 659]  # A4, C#5, E5 (A major chord)
        sound = []
        
        for freq in notes:
            note = self._generate_beep(freq, 0.15)
            sound.append(note)
            
        return np.concatenate(sound)
        
    def _generate_error_sound(self) -> np.ndarray:
        """Generate an error sound (descending notes)"""
        notes = [440, 415, 392]  # A4, G#4, G4
        sound = []
        
        for freq in notes:
            note = self._generate_beep(freq, 0.1)
            sound.append(note)
            
        return np.concatenate(sound)
        
    def _generate_listening_sound(self) -> np.ndarray:
        """Generate a soft listening indicator"""
        # Soft ascending sweep
        duration = 0.3
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Frequency sweep from 200 to 800 Hz
        frequency = np.linspace(200, 800, len(t))
        phase = 2 * np.pi * np.cumsum(frequency) / self.sample_rate
        
        sweep = np.sin(phase) * 0.3
        
        # Apply envelope
        envelope = np.ones_like(sweep)
        fade_samples = int(0.05 * self.sample_rate)
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        
        return (sweep * envelope).astype(np.float32)
        
    def _generate_processing_sound(self) -> np.ndarray:
        """Generate a processing/thinking sound"""
        # Modulated tone
        duration = 0.5
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Carrier frequency
        carrier = 440
        
        # Modulation
        mod_freq = 4  # 4 Hz modulation
        modulation = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
        
        # Generate modulated tone
        tone = np.sin(2 * np.pi * carrier * t) * modulation * 0.3
        
        return tone.astype(np.float32)
        
    def get_feedback(self, feedback_type: str) -> Optional[np.ndarray]:
        """Get feedback sound by type"""
        return self.feedback_sounds.get(feedback_type)
        
    def create_custom_feedback(self, frequency: float, duration: float, pattern: str = "single") -> np.ndarray:
        """Create custom feedback sound"""
        if pattern == "single":
            return self._generate_beep(frequency, duration)
        elif pattern == "double":
            return self._generate_double_beep(frequency, duration)
        elif pattern == "triple":
            beep = self._generate_beep(frequency, duration)
            silence = np.zeros(int(self.sample_rate * 0.05))
            return np.concatenate([beep, silence, beep, silence, beep])
        else:
            return self._generate_beep(frequency, duration)

class StreamingAudioProcessor:
    """WebSocket-based streaming audio processor"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.processor = AudioStreamProcessor(config)
        self.vad = VoiceActivityDetector(config)
        self.feedback = AudioFeedbackGenerator(config.sample_rate)
        
        # WebSocket clients
        self.clients = set()
        
        # Processing state
        self.is_processing = False
        
    async def start_server(self, host: str = "localhost", port: int = 8765):
        """Start WebSocket server for audio streaming"""
        async with websockets.serve(self.handle_client, host, port):
            print(f"Audio streaming server started on ws://{host}:{port}")
            await asyncio.Future()  # Run forever
            
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection"""
        self.clients.add(websocket)
        print(f"Client connected: {websocket.remote_address}")
        
        try:
            # Send connection confirmation
            await websocket.send(json.dumps({
                "type": "connected",
                "config": {
                    "sample_rate": self.config.sample_rate,
                    "channels": self.config.channels,
                    "chunk_size": self.config.chunk_size
                }
            }))
            
            # Send initial feedback sound
            feedback_audio = self.feedback.get_feedback("listening")
            await self._send_audio(websocket, feedback_audio, "feedback")
            
            async for message in websocket:
                await self._process_message(websocket, message)
                
        except websockets.ConnectionClosed:
            print(f"Client disconnected: {websocket.remote_address}")
        finally:
            self.clients.remove(websocket)
            
    async def _process_message(self, websocket, message):
        """Process incoming WebSocket message"""
        try:
            # Parse message
            if isinstance(message, bytes):
                # Raw audio data
                await self._process_audio_data(websocket, message)
            else:
                # JSON message
                data = json.loads(message)
                message_type = data.get("type")
                
                if message_type == "audio":
                    # Base64 encoded audio
                    audio_data = base64.b64decode(data["data"])
                    await self._process_audio_data(websocket, audio_data)
                    
                elif message_type == "config":
                    # Update configuration
                    await self._update_config(websocket, data["config"])
                    
                elif message_type == "command":
                    # Process command
                    await self._process_command(websocket, data["command"])
                    
        except Exception as e:
            print(f"Message processing error: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": str(e)
            }))
            
    async def _process_audio_data(self, websocket, audio_data: bytes):
        """Process incoming audio data"""
        # Convert to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Process audio
        processed = self.processor.process_chunk(audio_array)
        
        # Get metrics
        metrics = self.processor._compute_metrics(processed)
        
        # Voice activity detection
        vad_state, speech_segment = self.vad.process(processed, metrics)
        
        # Send metrics to client
        await websocket.send(json.dumps({
            "type": "metrics",
            "data": {
                "rms": metrics.rms_level,
                "peak": metrics.peak_level,
                "snr": metrics.snr,
                "is_speech": metrics.is_speech,
                "vad_state": vad_state
            }
        }))
        
        # Send processed audio back
        await self._send_audio(websocket, processed, "processed")
        
        # If complete speech segment detected
        if speech_segment is not None:
            await websocket.send(json.dumps({
                "type": "speech_segment",
                "duration": len(speech_segment) / self.config.sample_rate
            }))
            
            # Send feedback
            feedback_audio = self.feedback.get_feedback("success")
            await self._send_audio(websocket, feedback_audio, "feedback")
            
    async def _send_audio(self, websocket, audio: np.ndarray, audio_type: str):
        """Send audio data to client"""
        # Convert to int16
        if audio.dtype == np.float32:
            audio_int16 = (audio * 32767).astype(np.int16)
        else:
            audio_int16 = audio
            
        # Send as base64
        audio_bytes = audio_int16.tobytes()
        audio_base64 = base64.b64encode(audio_bytes).decode()
        
        await websocket.send(json.dumps({
            "type": "audio",
            "audio_type": audio_type,
            "data": audio_base64,
            "sample_rate": self.config.sample_rate,
            "samples": len(audio)
        }))
        
    async def _update_config(self, websocket, config_update: Dict):
        """Update audio configuration"""
        # Update relevant config fields
        if "noise_reduction" in config_update:
            self.config.noise_reduction = config_update["noise_reduction"]
        if "vad_threshold" in config_update:
            self.config.vad_threshold = config_update["vad_threshold"]
            
        await websocket.send(json.dumps({
            "type": "config_updated",
            "config": {
                "noise_reduction": self.config.noise_reduction,
                "vad_threshold": self.config.vad_threshold
            }
        }))
        
    async def _process_command(self, websocket, command: str):
        """Process control command"""
        if command == "calibrate_noise":
            # Start noise calibration
            await websocket.send(json.dumps({
                "type": "calibration_start",
                "message": "Please remain quiet for noise calibration"
            }))
            
            # Send feedback
            feedback_audio = self.feedback.get_feedback("beep")
            await self._send_audio(websocket, feedback_audio, "feedback")
            
            # Calibrate (simplified for async)
            await asyncio.sleep(1.0)
            
            await websocket.send(json.dumps({
                "type": "calibration_complete",
                "message": "Noise calibration complete"
            }))
            
        elif command == "test_feedback":
            # Test all feedback sounds
            for feedback_type in ["beep", "double_beep", "success", "error", "listening", "processing"]:
                audio = self.feedback.get_feedback(feedback_type)
                await self._send_audio(websocket, audio, "feedback")
                await asyncio.sleep(0.5)

# Example usage and testing
if __name__ == "__main__":
    # Test audio processor
    config = AudioConfig(
        sample_rate=16000,
        chunk_size=1024,
        noise_reduction=True,
        vad_enabled=True
    )
    
    # Create processor
    processor = AudioStreamProcessor(config)
    
    # Test feedback generator
    feedback = AudioFeedbackGenerator()
    
    print("Audio processing module ready")
    print("Features:")
    print("- Real-time audio streaming")
    print("- Noise reduction")
    print("- Voice activity detection") 
    print("- Audio feedback generation")
    
    # Run streaming server
    async def main():
        streaming = StreamingAudioProcessor(config)
        await streaming.start_server()
        
    # asyncio.run(main())