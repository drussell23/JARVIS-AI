"""
Liveness Detector
=================

Detects replay attacks and ensures voice input is from a live person.
"""

import numpy as np
import librosa
import logging
from typing import Tuple, Dict
from scipy import signal
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class LivenessDetector:
    """
    Detects whether audio input is from a live person or a replay/synthetic attack.
    """
    
    def __init__(self):
        # Detection thresholds
        self.min_duration = 0.5  # Minimum duration in seconds
        self.max_duration = 10.0  # Maximum duration in seconds
        
        # Environmental noise thresholds
        self.min_snr = 10.0  # Minimum signal-to-noise ratio in dB
        self.max_noise_level = 0.3  # Maximum background noise level
        
        # Audio quality thresholds
        self.min_bandwidth = 3000  # Minimum bandwidth in Hz
        self.max_clipping_ratio = 0.01  # Maximum clipped samples ratio
        
    def check_liveness(self, audio: np.ndarray, sr: int) -> float:
        """
        Perform comprehensive liveness detection.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Liveness score (0-100)
        """
        scores = {}
        
        # Basic audio quality checks
        scores['duration'] = self._check_duration(audio, sr)
        scores['clipping'] = self._check_clipping(audio)
        scores['bandwidth'] = self._check_bandwidth(audio, sr)
        
        # Environmental consistency checks
        scores['noise_consistency'] = self._check_noise_consistency(audio, sr)
        scores['snr'] = self._check_snr(audio, sr)
        
        # Replay attack detection
        scores['microphone_pattern'] = self._check_microphone_pattern(audio, sr)
        scores['channel_consistency'] = self._check_channel_consistency(audio)
        
        # Synthetic speech detection
        scores['naturalness'] = self._check_voice_naturalness(audio, sr)
        scores['temporal_dynamics'] = self._check_temporal_dynamics(audio, sr)
        
        # Calculate weighted average
        weights = {
            'duration': 0.1,
            'clipping': 0.1,
            'bandwidth': 0.1,
            'noise_consistency': 0.15,
            'snr': 0.1,
            'microphone_pattern': 0.15,
            'channel_consistency': 0.1,
            'naturalness': 0.15,
            'temporal_dynamics': 0.15
        }
        
        total_score = sum(scores[key] * weights[key] for key in scores)
        
        logger.debug(f"Liveness scores: {scores}")
        logger.info(f"Total liveness score: {total_score:.1f}%")
        
        return total_score
    
    def _check_duration(self, audio: np.ndarray, sr: int) -> float:
        """Check if audio duration is within expected range."""
        duration = len(audio) / sr
        
        if duration < self.min_duration:
            return 0.0  # Too short
        elif duration > self.max_duration:
            return 50.0  # Suspiciously long
        else:
            return 100.0
    
    def _check_clipping(self, audio: np.ndarray) -> float:
        """Check for audio clipping (indicates poor quality or manipulation)."""
        max_val = np.max(np.abs(audio))
        if max_val == 0:
            return 0.0
        
        # Normalize
        audio_norm = audio / max_val
        
        # Count clipped samples
        clipped = np.sum(np.abs(audio_norm) > 0.99)
        clipping_ratio = clipped / len(audio)
        
        if clipping_ratio > self.max_clipping_ratio:
            return 50.0  # Too much clipping
        else:
            return 100.0 * (1 - clipping_ratio / self.max_clipping_ratio)
    
    def _check_bandwidth(self, audio: np.ndarray, sr: int) -> float:
        """Check if audio has sufficient bandwidth."""
        # Compute spectrum
        freqs = np.fft.rfftfreq(len(audio), 1/sr)
        magnitude = np.abs(np.fft.rfft(audio))
        
        # Find effective bandwidth
        total_energy = np.sum(magnitude**2)
        cumsum_energy = np.cumsum(magnitude**2)
        
        # Find frequency containing 95% of energy
        idx_95 = np.where(cumsum_energy >= 0.95 * total_energy)[0]
        
        if len(idx_95) > 0:
            bandwidth = freqs[idx_95[0]]
            
            if bandwidth < self.min_bandwidth:
                return 50.0 * (bandwidth / self.min_bandwidth)
            else:
                return 100.0
        else:
            return 0.0
    
    def _check_noise_consistency(self, audio: np.ndarray, sr: int) -> float:
        """Check for consistent background noise (real recordings have it)."""
        # Estimate noise from quiet parts
        energy = librosa.feature.rms(y=audio, hop_length=512)[0]
        
        # Find quiet segments (bottom 20%)
        threshold = np.percentile(energy, 20)
        quiet_segments = energy < threshold
        
        if np.sum(quiet_segments) < 10:
            return 50.0  # No quiet segments found
        
        # Check noise variance in quiet segments
        noise_segments = []
        for i, is_quiet in enumerate(quiet_segments):
            if is_quiet:
                start = i * 512
                end = min((i + 1) * 512, len(audio))
                noise_segments.append(audio[start:end])
        
        if noise_segments:
            noise_vars = [np.var(seg) for seg in noise_segments]
            noise_consistency = 1.0 - (np.std(noise_vars) / (np.mean(noise_vars) + 1e-10))
            return 100.0 * noise_consistency
        else:
            return 50.0
    
    def _check_snr(self, audio: np.ndarray, sr: int) -> float:
        """Check signal-to-noise ratio."""
        # Estimate signal and noise power
        energy = librosa.feature.rms(y=audio, hop_length=512)[0]
        
        signal_power = np.percentile(energy, 80) ** 2
        noise_power = np.percentile(energy, 20) ** 2
        
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
            
            if snr_db < self.min_snr:
                return 50.0 * (snr_db / self.min_snr)
            else:
                return 100.0
        else:
            return 100.0  # No noise detected (could be synthetic)
    
    def _check_microphone_pattern(self, audio: np.ndarray, sr: int) -> float:
        """Check for microphone-specific patterns."""
        # Real microphones have specific frequency responses and noise patterns
        
        # Check for DC offset (common in real recordings)
        dc_offset = np.mean(audio)
        has_dc_offset = abs(dc_offset) > 0.001
        
        # Check for low-frequency rumble (common in real recordings)
        b, a = signal.butter(4, 50 / (sr/2), 'low')
        low_freq = signal.filtfilt(b, a, audio)
        low_freq_energy = np.sum(low_freq**2) / np.sum(audio**2)
        
        # Check for high-frequency roll-off (natural in real recordings)
        b, a = signal.butter(4, 4000 / (sr/2), 'high')
        high_freq = signal.filtfilt(b, a, audio)
        high_freq_energy = np.sum(high_freq**2) / np.sum(audio**2)
        
        score = 0.0
        if has_dc_offset:
            score += 30.0
        if 0.0001 < low_freq_energy < 0.1:
            score += 35.0
        if high_freq_energy < 0.3:
            score += 35.0
            
        return score
    
    def _check_channel_consistency(self, audio: np.ndarray) -> float:
        """Check for channel consistency (mono audio expected)."""
        # For mono audio, this checks internal consistency
        # Real recordings have natural variations
        
        # Split into segments and check correlation
        n_segments = 10
        segment_len = len(audio) // n_segments
        
        correlations = []
        for i in range(n_segments - 1):
            seg1 = audio[i * segment_len:(i + 1) * segment_len]
            seg2 = audio[(i + 1) * segment_len:(i + 2) * segment_len]
            
            if len(seg1) == len(seg2) and len(seg1) > 0:
                corr = np.corrcoef(seg1, seg2)[0, 1]
                correlations.append(corr)
        
        if correlations:
            # Real audio has moderate correlation between segments
            avg_corr = np.mean(correlations)
            if 0.3 < avg_corr < 0.9:
                return 100.0
            else:
                return 50.0
        else:
            return 50.0
    
    def _check_voice_naturalness(self, audio: np.ndarray, sr: int) -> float:
        """Check for natural voice characteristics."""
        # Extract voice-specific features
        
        # Check for formant structure
        try:
            # Simple formant detection using LPC
            lpc_order = 16
            a = librosa.lpc(audio, order=lpc_order)
            roots = np.roots(a)
            roots = [r for r in roots if np.imag(r) >= 0]
            
            freqs = np.arctan2(np.imag(roots), np.real(roots)) * (sr / (2 * np.pi))
            freqs = sorted([f for f in freqs if 200 < f < 4000])  # Voice frequency range
            
            # Check for at least 3 formants
            if len(freqs) >= 3:
                # Check formant spacing (should be reasonable)
                spacings = np.diff(freqs[:3])
                if all(200 < s < 2000 for s in spacings):
                    formant_score = 100.0
                else:
                    formant_score = 50.0
            else:
                formant_score = 0.0
                
        except:
            formant_score = 50.0
        
        # Check for pitch variations (natural speech has them)
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        if np.sum(voiced_flag) > 10:
            f0_voiced = f0[voiced_flag]
            pitch_var = np.std(f0_voiced) / (np.mean(f0_voiced) + 1e-10)
            
            if 0.05 < pitch_var < 0.3:  # Natural pitch variation range
                pitch_score = 100.0
            else:
                pitch_score = 50.0
        else:
            pitch_score = 50.0
        
        return 0.6 * formant_score + 0.4 * pitch_score
    
    def _check_temporal_dynamics(self, audio: np.ndarray, sr: int) -> float:
        """Check for natural temporal dynamics in speech."""
        # Compute envelope
        envelope = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512)).mean(axis=0)
        
        # Check for natural speech rhythm
        # Real speech has pauses and variations
        
        # Compute local variations
        variations = np.diff(envelope)
        variation_ratio = np.std(variations) / (np.mean(envelope) + 1e-10)
        
        # Check for silence/pause patterns
        silence_threshold = 0.1 * np.max(envelope)
        silent_frames = envelope < silence_threshold
        
        # Count transitions (speech bursts)
        transitions = np.diff(silent_frames.astype(int))
        n_bursts = np.sum(transitions == -1)  # Silent to speech transitions
        
        # Natural speech has multiple bursts
        burst_score = min(100.0, n_bursts * 20)  # Expect ~5 bursts
        
        # Natural speech has good variation
        if 0.5 < variation_ratio < 2.0:
            variation_score = 100.0
        else:
            variation_score = 50.0
        
        return 0.5 * burst_score + 0.5 * variation_score
    
    def detect_replay_attack(self, audio: np.ndarray, sr: int, 
                           reference_audio: np.ndarray = None) -> Dict:
        """
        Specific replay attack detection.
        
        Args:
            audio: Audio to check
            sr: Sample rate
            reference_audio: Optional reference for comparison
            
        Returns:
            Detection results
        """
        results = {
            'is_replay': False,
            'confidence': 0.0,
            'indicators': []
        }
        
        # Check for recording artifacts
        # 1. Double compression artifacts
        spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
        if np.mean(spectral_flatness) > 0.8:
            results['indicators'].append('high_spectral_flatness')
            results['confidence'] += 30.0
        
        # 2. Speaker/playback frequency response
        magnitude = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/sr)
        
        # Check for unnatural frequency cutoffs
        idx_8k = np.argmin(np.abs(freqs - 8000))
        high_freq_ratio = np.sum(magnitude[idx_8k:]) / np.sum(magnitude)
        
        if high_freq_ratio < 0.01:  # Suspicious cutoff
            results['indicators'].append('frequency_cutoff')
            results['confidence'] += 40.0
        
        # 3. Phase consistency check
        if reference_audio is not None and len(reference_audio) == len(audio):
            phase_corr = np.corrcoef(
                np.angle(np.fft.rfft(audio)),
                np.angle(np.fft.rfft(reference_audio))
            )[0, 1]
            
            if abs(phase_corr) > 0.9:  # Too similar
                results['indicators'].append('phase_similarity')
                results['confidence'] += 30.0
        
        results['is_replay'] = results['confidence'] >= 50.0
        
        return results