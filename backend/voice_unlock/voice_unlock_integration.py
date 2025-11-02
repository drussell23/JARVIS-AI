"""
Voice Unlock System Integration
==============================

Integrates the optimized ML system with the existing voice unlock components,
providing a unified interface for JARVIS.
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import json
import hashlib
import asyncio
import gc
from concurrent.futures import ThreadPoolExecutor

# ML optimization components
from .ml import VoiceUnlockMLSystem, get_ml_manager, get_monitor
from .ml.optimized_voice_auth import OptimizedVoiceAuthenticator

# Resource management for 30% target
try:
    from ...resource_manager import get_resource_manager
    RESOURCE_MANAGER_AVAILABLE = True
except ImportError:
    RESOURCE_MANAGER_AVAILABLE = False

# Core components
from .utils.audio_capture import AudioCapture
# from .core.voice_commands import VoiceCommandProcessor  # TODO: Create this module

# Proximity authentication
# from .proximity_voice_auth.python.proximity_authenticator import ProximityAuthenticator  # TODO: Implement

# Apple Watch proximity
from .apple_watch_proximity import AppleWatchProximityDetector

# Configuration
from .config import get_config

logger = logging.getLogger(__name__)


class VoiceUnlockSystem:
    """
    Main integration class for the voice unlock system with ML optimization
    """
    
    def __init__(self):
        self.config = get_config()
        
        # Initialize ML system with optimization
        self.ml_system = VoiceUnlockMLSystem()
        
        # Audio components (lazy loaded)
        self._audio_manager = None
        self._command_processor = None
        self._proximity_auth = None
        self._apple_watch_detector = None
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # System state
        self.is_active = False
        self.is_locked = True
        self.current_user = None
        
        # Performance tracking
        self.last_auth_time = None
        self.auth_history = []
        
        logger.info("Voice Unlock System initialized with ML optimization")
        
    @property
    def audio_manager(self):
        """Lazy load audio manager"""
        if self._audio_manager is None:
            self._audio_manager = AudioCapture()
        return self._audio_manager
        
    @property
    def command_processor(self):
        """Lazy load command processor"""
        if self._command_processor is None:
            # self._command_processor = VoiceCommandProcessor()  # TODO: Implement
            return None
        return self._command_processor
        
    @property
    def proximity_auth(self):
        """Lazy load proximity authenticator"""
        if self._proximity_auth is None:
            # self._proximity_auth = ProximityAuthenticator()  # TODO: Implement
            return None
        return self._proximity_auth
        
    @property
    def apple_watch_detector(self):
        """Lazy load Apple Watch proximity detector"""
        if self._apple_watch_detector is None:
            self._apple_watch_detector = AppleWatchProximityDetector({
                'unlock_distance': 3.0,  # 3 meters (~10 feet)
                'lock_distance': 10.0,   # 10 meters (~33 feet)
                'require_unlocked_watch': True
            })
        return self._apple_watch_detector
        
    async def start(self):
        """Start the voice unlock system"""
        logger.info("Starting Voice Unlock System...")
        
        # Start audio monitoring in background
        self.is_active = True
        
        # Start proximity detection
        if self.config.system.integration_mode in ['screensaver', 'both']:
            await self._start_proximity_monitoring()
            
        # Start Apple Watch detection
        await self._start_apple_watch_monitoring()
            
        # Pre-register known users for lazy loading
        self._preregister_users()
        
        logger.info("Voice Unlock System started")
        
    def _preregister_users(self):
        """Pre-register all known users for optimal lazy loading"""
        try:
            # Get list of enrolled users
            users_file = Path(self.config.security.storage_path).expanduser() / 'enrolled_users.json'
            
            if users_file.exists():
                with open(users_file, 'r') as f:
                    users = json.load(f)
                    
                # Predict which users are likely to authenticate based on time patterns
                current_hour = datetime.now().hour
                
                for user_id, user_data in users.items():
                    # Calculate likelihood based on usage patterns
                    likelihood = self._calculate_user_likelihood(user_id, user_data, current_hour)
                    
                    if likelihood > 0.5:
                        # Pre-register for lazy loading
                        logger.debug(f"Pre-registering user {user_id} (likelihood: {likelihood:.2f})")
                        
        except Exception as e:
            logger.error(f"Failed to preregister users: {e}")
            
    def _calculate_user_likelihood(self, user_id: str, user_data: Dict, current_hour: int) -> float:
        """Calculate likelihood of user authenticating based on patterns"""
        # Simple time-based prediction (can be enhanced)
        common_hours = user_data.get('common_hours', [9, 10, 11, 14, 15, 16, 17])
        
        if current_hour in common_hours:
            return 0.8
        elif abs(current_hour - 12) < 4:  # Business hours
            return 0.6
        else:
            return 0.3
            
    async def _start_proximity_monitoring(self):
        """Start proximity-based authentication monitoring"""
        loop = asyncio.get_event_loop()
        
        def proximity_callback(distance: float, device_id: str):
            """Handle proximity events"""
            if distance < self.config.system.unlock_distance and self.is_locked:
                # Trigger authentication
                loop.create_task(self._handle_proximity_unlock(device_id))
                
        # Start proximity monitoring in background
        await loop.run_in_executor(
            self.executor,
            self.proximity_auth.start_monitoring,
            proximity_callback
        )
        
    async def _start_apple_watch_monitoring(self):
        """Start Apple Watch proximity monitoring"""
        logger.info("Starting Apple Watch proximity monitoring...")
        
        # Define callback for proximity events
        def watch_proximity_callback(distance: float, device_id: str):
            """Handle Apple Watch proximity events"""
            logger.debug(f"Apple Watch proximity: {distance:.1f}m")
            
            # Store watch proximity status
            self.apple_watch_nearby = distance <= 3.0  # Within unlock distance
            
        # Define callback for lock events
        def watch_lock_callback(device_id: str):
            """Handle Apple Watch out of range"""
            logger.info("Apple Watch out of range - triggering lock")
            self.apple_watch_nearby = False
            
            # Lock the system if configured
            if self.config.system.auto_lock_on_distance:
                asyncio.create_task(self.lock_system())
                
        # Add callbacks
        self.apple_watch_detector.add_proximity_callback(watch_proximity_callback)
        self.apple_watch_detector.add_lock_callback(watch_lock_callback)
        
        # Start scanning in background
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            self.apple_watch_detector.start_scanning
        )
        
        # Track watch status
        self.apple_watch_nearby = False
        
    async def _handle_proximity_unlock(self, device_id: str):
        """Handle proximity-based unlock request"""
        logger.info(f"Proximity unlock triggered by device: {device_id}")
        
        # Check if device is authorized
        if not self._is_device_authorized(device_id):
            logger.warning(f"Unauthorized device: {device_id}")
            return
            
        # Start listening for voice authentication
        await self.authenticate_with_voice()
        
    def _is_device_authorized(self, device_id: str) -> bool:
        """Check if device is authorized for proximity unlock"""
        authorized_devices_file = Path(self.config.security.storage_path).expanduser() / 'authorized_devices.json'
        
        if authorized_devices_file.exists():
            with open(authorized_devices_file, 'r') as f:
                devices = json.load(f)
                return device_id in devices
                
        return False
        
    async def authenticate_proximity_voice(self, timeout: float = 10.0) -> Tuple[bool, Optional[str]]:
        """
        Authenticate using proximity + voice for 30% memory target.
        Ultra-optimized for minimal memory usage.
        """
        logger.info("Starting proximity + voice authentication (30% memory mode)")
        
        # Step 1: Check Apple Watch proximity
        if self.apple_watch_detector:
            proximity_result = await self._check_apple_watch_proximity()
            if not proximity_result['is_nearby']:
                logger.warning("Apple Watch not detected nearby")
                return False, "No Apple Watch detected"
                
            logger.info(f"Apple Watch detected: {proximity_result['distance_category']}")
            
        # Step 2: Request resources from resource manager
        if RESOURCE_MANAGER_AVAILABLE:
            rm = get_resource_manager()
            if not rm.request_voice_unlock_resources():
                logger.error("Resource manager denied voice unlock")
                return False, "Insufficient resources"
                
        # Step 3: Use ML manager for ultra-optimized auth
        ml_manager = get_ml_manager()
        
        # Prepare system (ultra-aggressive cleanup)
        if not ml_manager.prepare_for_voice_unlock():
            logger.error("Failed to prepare ML system")
            return False, "System preparation failed"
            
        try:
            # Capture voice with minimal memory
            audio_data, detected = await self._record_authentication_audio(timeout)
            if not detected:
                return False, "No voice detected"
                
            # Ultra-fast model load
            model = ml_manager.load_voice_model_fast()
            if model is None:
                return False, "Model load failed"
                
            # Process with minimal memory footprint
            result = self._authenticate_voice_ultra(audio_data, model)
            
            return result['authenticated'], result.get('user_id')
            
        finally:
            # Always clean up immediately for 30% target
            ml_manager._emergency_unload_all()
            if RESOURCE_MANAGER_AVAILABLE:
                rm.voice_unlock_pending = False
            gc.collect()
            
    async def authenticate_with_voice(self, timeout: float = 10.0, 
                                    require_watch: bool = True) -> Dict[str, Any]:
        """
        Perform voice authentication with ML optimization and Apple Watch proximity
        
        Args:
            timeout: Maximum time to wait for voice input
            require_watch: Whether Apple Watch proximity is required
            
        Returns:
            Authentication result dictionary
        """
        result = {
            'authenticated': False,
            'user_id': None,
            'confidence': 0.0,
            'processing_time': 0.0,
            'method': 'voice+watch' if require_watch else 'voice',
            'watch_nearby': False,
            'watch_distance': None
        }
        
        try:
            # Check Apple Watch proximity if required
            if require_watch:
                watch_status = self.apple_watch_detector.get_status()
                result['watch_nearby'] = watch_status['watch_nearby']
                result['watch_distance'] = watch_status['watch_distance']
                
                if not watch_status['watch_nearby']:
                    result['error'] = "Apple Watch not detected nearby"
                    logger.warning("Authentication failed: Apple Watch not in range")
                    return result
                    
                logger.info(f"Apple Watch detected at {watch_status['watch_distance']:.1f}m")
            
            # Start recording
            logger.info("Listening for voice authentication...")
            
            # Record audio with timeout
            audio_data = await self._record_authentication_audio(timeout)
            
            if audio_data is None:
                result['error'] = "No voice detected"
                return result
                
            # Process voice command if detected
            command_result = await self._process_voice_command(audio_data)
            
            if command_result['has_command']:
                command = command_result['command']
                
                # Handle different command types
                if command.command_type == 'unlock':
                    # Extract user ID from command or identify from voice
                    user_id = command.user_name or command_result.get('user_id')
                    if not user_id:
                        # Try to identify user from voice alone
                        user_id = await self._identify_user_from_voice(audio_data)
                        
                    if user_id:
                        # Authenticate the identified user
                        auth_result = self.ml_system.authenticate_user(
                            user_id, 
                            audio_data,
                            self.config.audio.sample_rate
                        )
                        
                        result.update(auth_result)
                        result['user_id'] = user_id
                        result['command'] = command
                        
                        # Handle successful authentication
                        if result['authenticated']:
                            await self._handle_successful_auth(user_id, result)
                            
                        # Generate JARVIS response
                        response = self.command_processor.jarvis_handler.generate_response(
                            command, result
                        )
                        await self._speak_response(response)
                    else:
                        result['error'] = "Could not identify user"
                        response = self.command_processor.jarvis_handler.generate_response(
                            command, result
                        )
                        await self._speak_response(response)
                        
                elif command.command_type == 'lock':
                    # Handle lock command
                    await self.lock_system()
                    lock_result = {'success': True}
                    response = self.command_processor.jarvis_handler.generate_response(
                        command, lock_result
                    )
                    await self._speak_response(response)
                    
                elif command.command_type == 'status':
                    # Handle status command
                    status = self.get_status()
                    response = self.command_processor.jarvis_handler.generate_response(
                        command, status
                    )
                    await self._speak_response(response)

                elif command.command_type == 'security_test':
                    # Handle voice security testing
                    await self._speak_response("Initiating voice security test. This will take a moment.")
                    logger.info("🔒 Running voice security test...")

                    try:
                        from .voice_security_tester import VoiceSecurityTester, PlaybackConfig, AudioBackend

                        # Enable audio playback for voice-triggered tests
                        playback_config = PlaybackConfig(
                            enabled=True,
                            verbose=True,
                            backend=AudioBackend.AUTO,
                            volume=0.5,
                            announce_profile=True,
                            pause_after_playback=0.5
                        )

                        # Use standard test mode (8 diverse profiles)
                        test_config = {
                            'test_mode': 'standard',
                            'authorized_user': self.authorized_user if hasattr(self, 'authorized_user') else 'Derek',
                        }

                        tester = VoiceSecurityTester(config=test_config, playback_config=playback_config)
                        report = await tester.run_security_tests()

                        # Save the report
                        await tester.save_report(report)

                        # Generate voice response based on results
                        if report.is_secure:
                            response = (
                                f"Voice security test complete. {report.summary['passed']} of {report.summary['total']} tests passed. "
                                f"Your voice authentication is secure. No unauthorized voices were accepted."
                            )
                        else:
                            breaches = report.summary.get('security_breaches', 0)
                            false_rejects = report.summary.get('false_rejections', 0)
                            response = (
                                f"Voice security test complete. Warning: {breaches} security breaches detected. "
                                f"{false_rejects} false rejections occurred. Please review the security report."
                            )

                        await self._speak_response(response)

                        # Update result
                        result['security_test'] = {
                            'success': True,
                            'is_secure': report.is_secure,
                            'summary': report.summary,
                            'report_file': str(report.report_file) if hasattr(report, 'report_file') else None
                        }

                        logger.info(f"🔒 Security test completed: {'SECURE' if report.is_secure else 'VULNERABLE'}")

                    except Exception as e:
                        logger.error(f"Security test failed: {e}")
                        await self._speak_response(f"Security test failed: {str(e)}")
                        result['error'] = f"Security test failed: {str(e)}"

                result['command_type'] = command.command_type
                result['raw_command'] = command.raw_text
            else:
                # No command detected
                if command_result.get('transcription'):
                    logger.info(f"No command in: {command_result['transcription']}")
                result['error'] = "No valid command detected"
                        
        except Exception as e:
            logger.error(f"Voice authentication error: {e}")
            result['error'] = str(e)
            
        return result
        
    async def _record_authentication_audio(self, timeout: float) -> Optional[np.ndarray]:
        """Record audio for authentication"""
        loop = asyncio.get_event_loop()
        
        # Use thread pool for blocking audio operations
        audio_data, detected = await loop.run_in_executor(
            self.executor,
            self.audio_manager.capture_with_vad,
            timeout
        )
        
        return audio_data
        
    async def _check_apple_watch_proximity(self) -> Dict[str, Any]:
        """Check Apple Watch proximity asynchronously"""
        loop = asyncio.get_event_loop()
        
        # Run proximity check in thread pool
        result = await loop.run_in_executor(
            self.executor,
            self.apple_watch_detector.check_proximity
        )
        
        return result
        
    def _authenticate_voice_ultra(self, audio_data: np.ndarray, model: Any) -> Dict[str, Any]:
        """Ultra-optimized voice authentication for 30% memory target"""
        try:
            # Extract features with minimal memory
            # Downsample if needed to reduce memory
            if len(audio_data) > 16000 * 5:  # More than 5 seconds
                audio_data = audio_data[:16000 * 5]  # Truncate
                
            # Simple feature extraction (minimal memory)
            features = self._extract_minimal_features(audio_data)
            
            # Run inference
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba([features])[0]
                confidence = float(proba[1])
                authenticated = confidence > 0.8
            else:
                prediction = model.predict([features])[0]
                authenticated = bool(prediction)
                confidence = 0.9 if authenticated else 0.2
                
            return {
                'authenticated': authenticated,
                'confidence': confidence,
                'user_id': 'default_user' if authenticated else None
            }
            
        except Exception as e:
            logger.error(f"Ultra auth failed: {e}")
            return {
                'authenticated': False,
                'confidence': 0.0,
                'user_id': None
            }
            
    def _extract_minimal_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract minimal features for ultra-low memory"""
        # Very simple feature extraction
        # In production, this would use proper voice features
        
        # Normalize
        audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)
        
        # Simple features
        features = []
        
        # Energy
        features.append(np.mean(audio_data ** 2))
        
        # Zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_data)))) / (2 * len(audio_data))
        features.append(zero_crossings)
        
        # Simple spectral features (minimal FFT)
        fft_size = 512  # Small FFT for low memory
        if len(audio_data) > fft_size:
            segment = audio_data[:fft_size]
            spectrum = np.abs(np.fft.rfft(segment * np.hanning(fft_size)))
            
            # Spectral centroid
            freqs = np.fft.rfftfreq(fft_size, 1/16000)
            centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)
            features.append(centroid)
            
            # Spectral rolloff
            cumsum = np.cumsum(spectrum)
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            if len(rolloff_idx) > 0:
                features.append(freqs[rolloff_idx[0]])
            else:
                features.append(0.0)
                
        else:
            features.extend([0.0, 0.0])
            
        return np.array(features, dtype=np.float32)
        
    async def _process_voice_command(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Process voice command from audio"""
        loop = asyncio.get_event_loop()
        
        # Process in thread pool
        result = await loop.run_in_executor(
            self.executor,
            self.command_processor.process_audio,
            audio_data,
            self.config.audio.sample_rate
        )
        
        return result
        
    async def _identify_user_from_voice(self, audio_data: np.ndarray) -> Optional[str]:
        """Identify user from voice characteristics alone"""
        # This would require a separate speaker identification model
        # For now, return None (requires explicit user identification in command)
        return None
        
    async def _handle_successful_auth(self, user_id: str, auth_result: Dict[str, Any]):
        """Handle successful authentication"""
        self.current_user = user_id
        self.is_locked = False
        self.last_auth_time = datetime.now()
        
        # Record authentication event
        self.auth_history.append({
            'user_id': user_id,
            'timestamp': self.last_auth_time,
            'confidence': auth_result['confidence'],
            'method': auth_result.get('method', 'voice')
        })
        
        # Trigger unlock action
        if self.config.system.integration_mode in ['screensaver', 'both']:
            await self._unlock_screen()
            
        # JARVIS response
        if self.config.system.jarvis_responses:
            response = self.config.system.custom_responses.get(
                'success', 
                f"Welcome back, {user_id}"
            )
            await self._speak_response(response)
            
        logger.info(f"Successfully authenticated user: {user_id}")
        
    async def _unlock_screen(self):
        """Unlock the macOS screen"""
        # This would integrate with the screen lock manager
        logger.info("Unlocking screen...")
        
        # Use AppleScript or system APIs to unlock
        # For now, just log
        
    async def _speak_response(self, text: str):
        """Speak response using JARVIS voice"""
        loop = asyncio.get_event_loop()
        
        # Use TTS in thread pool
        await loop.run_in_executor(
            self.executor,
            self._speak_tts,
            text
        )
        
    def _speak_tts(self, text: str):
        """Text-to-speech implementation"""
        # This would use the JARVIS TTS system
        logger.info(f"JARVIS: {text}")
        
    async def enroll_user(self, user_id: str, audio_samples: List[np.ndarray]) -> Dict[str, Any]:
        """Enroll a new user with voice samples"""
        # Use ML system for enrollment
        result = self.ml_system.enroll_user(user_id, audio_samples)
        
        if result['success']:
            # Update enrolled users list
            self._update_enrolled_users(user_id)
            
            # Speak confirmation
            if self.config.system.jarvis_responses:
                await self._speak_response(f"Voice profile created for {user_id}")
                
        return result
        
    def _update_enrolled_users(self, user_id: str):
        """Update list of enrolled users"""
        users_file = Path(self.config.security.storage_path).expanduser() / 'enrolled_users.json'
        users_file.parent.mkdir(parents=True, exist_ok=True)
        
        users = {}
        if users_file.exists():
            with open(users_file, 'r') as f:
                users = json.load(f)
                
        users[user_id] = {
            'enrolled_at': datetime.now().isoformat(),
            'common_hours': [9, 10, 11, 14, 15, 16, 17]  # Default
        }
        
        with open(users_file, 'w') as f:
            json.dump(users, f, indent=2)
            
    async def lock_system(self):
        """Lock the system"""
        self.is_locked = True
        self.current_user = None
        
        # Clear sensitive data from memory
        self.ml_system._cleanup_resources()
        
        logger.info("System locked")
        
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        ml_status = self.ml_system.get_performance_report()
        
        return {
            'is_active': self.is_active,
            'is_locked': self.is_locked,
            'current_user': self.current_user,
            'last_auth_time': self.last_auth_time.isoformat() if self.last_auth_time else None,
            'ml_status': ml_status['system_health'],
            'recent_authentications': len([
                a for a in self.auth_history 
                if (datetime.now() - a['timestamp']).seconds < 3600
            ])
        }
        
    async def stop(self):
        """Stop the voice unlock system"""
        logger.info("Stopping Voice Unlock System...")
        
        self.is_active = False
        
        # Stop proximity monitoring
        if self._proximity_auth:
            self.proximity_auth.stop_monitoring()
            
        # Stop Apple Watch detection
        if self._apple_watch_detector:
            self.apple_watch_detector.stop_scanning()
            
        # Cleanup ML system
        self.ml_system.cleanup()
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        logger.info("Voice Unlock System stopped")
        
    def __enter__(self):
        """Context manager entry"""
        asyncio.run(self.start())
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        asyncio.run(self.stop())


# Convenience functions for integration
async def create_voice_unlock_system() -> VoiceUnlockSystem:
    """Create and start voice unlock system"""
    system = VoiceUnlockSystem()
    await system.start()
    return system


def test_voice_unlock():
    """Test the integrated voice unlock system"""
    import sounddevice as sd
    
    async def run_test():
        # Create system
        system = await create_voice_unlock_system()
        
        try:
            # Show status
            status = system.get_status()
            print(f"System Status: {json.dumps(status, indent=2)}")
            
            # Test enrollment
            print("\nTesting enrollment...")
            print("Please say the enrollment phrase 3 times")
            
            samples = []
            for i in range(3):
                print(f"\nRecording sample {i+1}/3 (3 seconds)...")
                audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype='float32')
                sd.wait()
                samples.append(audio.flatten())
                
            result = await system.enroll_user("test_user", samples)
            print(f"Enrollment result: {json.dumps(result, indent=2)}")
            
            # Test authentication
            print("\nTesting authentication...")
            print("Please say your authentication phrase")
            
            auth_result = await system.authenticate_with_voice(timeout=10.0)
            print(f"Authentication result: {json.dumps(auth_result, indent=2)}")
            
            # Final status
            final_status = system.get_status()
            print(f"\nFinal Status: {json.dumps(final_status, indent=2)}")
            
        finally:
            # Cleanup
            await system.stop()
            
    # Run test
    asyncio.run(run_test())


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_voice_unlock()