"""
Dual-Factor Authentication Engine
=================================

Combines Apple Watch proximity and voice biometrics for secure authentication.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import sys
import time

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from voice_biometrics.voice_authenticator import VoiceAuthenticator
from .security_logger import SecurityLogger

logger = logging.getLogger(__name__)


class AuthStatus(Enum):
    """Authentication status codes"""
    SUCCESS = "success"
    PROXIMITY_FAILED = "proximity_failed"
    VOICE_FAILED = "voice_failed"
    LIVENESS_FAILED = "liveness_failed"
    COMBINED_FAILED = "combined_failed"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class AuthResult:
    """Authentication result"""
    status: AuthStatus
    success: bool
    proximity_score: float
    voice_score: float
    combined_score: float
    reason: str
    threat_level: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class DualFactorAuthEngine:
    """
    Dual-factor authentication engine combining proximity and voice.
    """
    
    def __init__(self, config_path: Path = None):
        self.config = self._load_config(config_path)
        
        # Authentication components
        self.voice_authenticator = None
        self.security_logger = SecurityLogger()
        
        # File-based IPC for Swift communication
        self.request_file = "/tmp/jarvis_proximity_request.json"
        self.response_file = "/tmp/jarvis_proximity_response.json"
        
        # Authentication state
        self.auth_attempts = {}
        self.lockout_until = {}
        
        # Thresholds from config
        self.proximity_threshold = self.config['proximity']['min_confidence']
        self.voice_threshold = self.config['voice']['min_confidence']
        self.combined_threshold = self.config['security']['combined_threshold']
        self.max_attempts = self.config['security']['max_attempts']
        self.lockout_duration = self.config['security']['lockout_duration']
    
    def _load_config(self, config_path: Path = None) -> Dict:
        """Load authentication configuration."""
        if config_path is None:
            config_path = Path(__file__).parent / "auth_config.json"
        
        default_config = {
            "proximity": {
                "min_confidence": 80.0,
                "detection_range": 3.0,
                "update_frequency": 2
            },
            "voice": {
                "min_confidence": 85.0,
                "min_samples": 3,
                "sample_duration": 3.0,
                "liveness_required": True
            },
            "security": {
                "combined_threshold": 90.0,
                "max_attempts": 3,
                "lockout_duration": 300,
                "proximity_weight": 0.4,
                "voice_weight": 0.6
            }
        }
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                # Merge with defaults
                for key in default_config:
                    if key in loaded_config:
                        default_config[key].update(loaded_config[key])
                return default_config
        
        # Save default config
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    async def initialize(self, user_id: str):
        """Initialize authentication engine for a user."""
        try:
            # Initialize voice authenticator
            self.voice_authenticator = VoiceAuthenticator(user_id)
            
            # Clear any existing IPC files
            for file in [self.request_file, self.response_file]:
                if Path(file).exists():
                    Path(file).unlink()
            
            logger.info(f"Dual-factor auth engine initialized for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize auth engine: {e}")
            raise
    
    async def authenticate(self, voice_data: bytes, sample_rate: int) -> AuthResult:
        """
        Perform dual-factor authentication.
        
        Args:
            voice_data: Raw audio data for voice authentication
            sample_rate: Audio sample rate
            
        Returns:
            AuthResult with authentication decision
        """
        user_id = self.voice_authenticator.user_id
        
        # Check for lockout
        if user_id in self.lockout_until:
            if datetime.now() < self.lockout_until[user_id]:
                remaining = (self.lockout_until[user_id] - datetime.now()).seconds
                return AuthResult(
                    status=AuthStatus.ERROR,
                    success=False,
                    proximity_score=0,
                    voice_score=0,
                    combined_score=0,
                    reason=f"Account locked for {remaining} seconds"
                )
            else:
                # Lockout expired
                del self.lockout_until[user_id]
                self.auth_attempts[user_id] = 0
        
        try:
            # Get proximity score from Swift service
            proximity_score = await self._get_proximity_score()
            
            # Check proximity threshold
            if proximity_score < self.proximity_threshold:
                result = AuthResult(
                    status=AuthStatus.PROXIMITY_FAILED,
                    success=False,
                    proximity_score=proximity_score,
                    voice_score=0,
                    combined_score=proximity_score * self.config['security']['proximity_weight'],
                    reason="Apple Watch not in proximity"
                )
                await self._handle_failed_attempt(user_id, result)
                return result
            
            # Perform voice authentication
            voice_result = self.voice_authenticator.authenticate(
                voice_data, sample_rate
            )
            
            voice_score = voice_result['confidence']
            
            # Check for liveness failure
            if voice_result.get('threat_type') == 'replay_attack':
                result = AuthResult(
                    status=AuthStatus.LIVENESS_FAILED,
                    success=False,
                    proximity_score=proximity_score,
                    voice_score=voice_score,
                    combined_score=0,
                    reason=voice_result['reason'],
                    threat_level='high'
                )
                await self._handle_security_threat(user_id, result)
                return result
            
            # Check voice threshold
            if voice_score < self.voice_threshold:
                result = AuthResult(
                    status=AuthStatus.VOICE_FAILED,
                    success=False,
                    proximity_score=proximity_score,
                    voice_score=voice_score,
                    combined_score=self._calculate_combined_score(
                        proximity_score, voice_score
                    ),
                    reason="Voice authentication failed"
                )
                await self._handle_failed_attempt(user_id, result)
                return result
            
            # Calculate combined score
            combined_score = self._calculate_combined_score(
                proximity_score, voice_score
            )
            
            # Make final authentication decision
            if combined_score >= self.combined_threshold:
                # Success!
                self.auth_attempts[user_id] = 0
                
                result = AuthResult(
                    status=AuthStatus.SUCCESS,
                    success=True,
                    proximity_score=proximity_score,
                    voice_score=voice_score,
                    combined_score=combined_score,
                    reason="Authentication successful"
                )
                
                # Log successful authentication
                await self.security_logger.log_event(
                    "authentication_success",
                    user_id=user_id,
                    proximity_score=proximity_score,
                    voice_score=voice_score,
                    combined_score=combined_score
                )
                
                return result
            else:
                # Combined score too low
                result = AuthResult(
                    status=AuthStatus.COMBINED_FAILED,
                    success=False,
                    proximity_score=proximity_score,
                    voice_score=voice_score,
                    combined_score=combined_score,
                    reason=f"Combined score {combined_score:.1f}% below threshold"
                )
                await self._handle_failed_attempt(user_id, result)
                return result
                
        except asyncio.TimeoutError:
            return AuthResult(
                status=AuthStatus.TIMEOUT,
                success=False,
                proximity_score=0,
                voice_score=0,
                combined_score=0,
                reason="Authentication timeout"
            )
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return AuthResult(
                status=AuthStatus.ERROR,
                success=False,
                proximity_score=0,
                voice_score=0,
                combined_score=0,
                reason=str(e)
            )
    
    def _calculate_combined_score(self, proximity_score: float, 
                                voice_score: float) -> float:
        """Calculate weighted combined score."""
        proximity_weight = self.config['security']['proximity_weight']
        voice_weight = self.config['security']['voice_weight']
        
        return (proximity_score * proximity_weight + 
                voice_score * voice_weight)
    
    async def _get_proximity_score(self) -> float:
        """Get proximity score from Swift service via file-based IPC."""
        try:
            # Clear any existing response
            if Path(self.response_file).exists():
                Path(self.response_file).unlink()
            
            # Send request for proximity status
            request = {
                'command': 'get_proximity',
                'timestamp': time.time()
            }
            
            with open(self.request_file, 'w') as f:
                json.dump(request, f)
            
            # Wait for response with timeout
            timeout = 1.0
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if Path(self.response_file).exists():
                    try:
                        with open(self.response_file, 'r') as f:
                            response = json.load(f)
                        
                        if response.get('status') == 'success' and 'data' in response:
                            return float(response['data'].get('confidence', 0.0))
                        else:
                            return 0.0
                    except json.JSONDecodeError:
                        await asyncio.sleep(0.01)
                        continue
                await asyncio.sleep(0.01)
            
            logger.warning("Proximity request timeout")
            return 0.0
                
        except Exception as e:
            logger.error(f"Error getting proximity score: {e}")
            return 0.0
    
    async def _handle_failed_attempt(self, user_id: str, result: AuthResult):
        """Handle failed authentication attempt."""
        # Increment attempt counter
        if user_id not in self.auth_attempts:
            self.auth_attempts[user_id] = 0
        
        self.auth_attempts[user_id] += 1
        
        # Log failed attempt
        await self.security_logger.log_event(
            "authentication_failed",
            user_id=user_id,
            status=result.status.value,
            reason=result.reason,
            attempt_number=self.auth_attempts[user_id]
        )
        
        # Check for lockout
        if self.auth_attempts[user_id] >= self.max_attempts:
            lockout_time = datetime.now() + timedelta(
                seconds=self.lockout_duration
            )
            self.lockout_until[user_id] = lockout_time
            
            await self.security_logger.log_event(
                "account_locked",
                user_id=user_id,
                lockout_until=lockout_time.isoformat(),
                reason=f"Too many failed attempts ({self.max_attempts})"
            )
    
    async def _handle_security_threat(self, user_id: str, result: AuthResult):
        """Handle potential security threat."""
        # Log security threat
        await self.security_logger.log_event(
            "security_threat",
            user_id=user_id,
            threat_level=result.threat_level,
            threat_type=result.status.value,
            reason=result.reason
        )
        
        # Immediate lockout for high threats
        if result.threat_level == 'high':
            lockout_time = datetime.now() + timedelta(
                seconds=self.lockout_duration * 2  # Double lockout for threats
            )
            self.lockout_until[user_id] = lockout_time
    
    async def enroll_voice(self, user_id: str, voice_data: bytes, 
                         sample_rate: int) -> Dict:
        """Enroll voice sample for user."""
        if not self.voice_authenticator:
            self.voice_authenticator = VoiceAuthenticator(user_id)
        
        result = self.voice_authenticator.enroll_voice(voice_data, sample_rate)
        
        # Log enrollment
        await self.security_logger.log_event(
            "voice_enrollment",
            user_id=user_id,
            success=result['success'],
            samples_collected=result.get('samples_collected', 0)
        )
        
        return result
    
    def get_status(self) -> Dict:
        """Get authentication engine status."""
        return {
            'initialized': self.voice_authenticator is not None,
            'proximity_connected': True,  # Using file-based IPC
            'voice_model_ready': (
                self.voice_authenticator.model is not None 
                if self.voice_authenticator else False
            ),
            'active_lockouts': len(self.lockout_until),
            'config': {
                'proximity_threshold': self.proximity_threshold,
                'voice_threshold': self.voice_threshold,
                'combined_threshold': self.combined_threshold
            }
        }
    
    async def shutdown(self):
        """Shutdown authentication engine."""
        # Clean up IPC files
        for file in [self.request_file, self.response_file]:
            if Path(file).exists():
                Path(file).unlink()
        
        logger.info("Dual-factor auth engine shutdown complete")