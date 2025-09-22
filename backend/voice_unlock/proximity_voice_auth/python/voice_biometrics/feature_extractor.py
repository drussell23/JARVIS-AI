"""
Voice Feature Extractor
======================

Extracts biometric features from voice samples for authentication.
"""

import numpy as np
import librosa
import logging
from typing import Dict, Tuple, Optional
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class VoiceFeatureExtractor:
    """
    Extracts various acoustic features for voice biometric authentication.
    """
    
    def __init__(self):
        # Feature extraction parameters
        self.n_mfcc = 13
        self.n_mels = 128
        self.hop_length = 512
        self.n_fft = 2048
        
    def extract_all_features(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Extract comprehensive feature set from audio.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Preprocess audio
        audio = self._preprocess_audio(audio, sr)
        
        # Extract different feature types
        features.update(self.extract_mfcc_features(audio, sr))
        features.update(self.extract_spectral_features(audio, sr))
        features.update(self.extract_prosodic_features(audio, sr))
        features.update(self.extract_temporal_features(audio, sr))
        features.update(self.extract_voice_quality_features(audio, sr))
        
        return features
    
    def _preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Preprocess audio signal."""
        # Remove silence
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Normalize
        audio = librosa.util.normalize(audio)
        
        # Apply pre-emphasis filter
        audio = librosa.effects.preemphasis(audio)
        
        return audio
    
    def extract_mfcc_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract MFCC-based features."""
        features = {}
        
        # Compute MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio, sr=sr, 
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Statistical features
        features['mfcc'] = mfccs.mean(axis=1).tolist()
        features['mfcc_std'] = mfccs.std(axis=1).tolist()
        features['mfcc_delta'] = librosa.feature.delta(mfccs).mean(axis=1).tolist()
        features['mfcc_delta2'] = librosa.feature.delta(mfccs, order=2).mean(axis=1).tolist()
        
        return features
    
    def extract_spectral_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract spectral features."""
        features = {}
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=sr, hop_length=self.hop_length
        )
        features['spectral_centroid'] = float(spectral_centroid.mean())
        features['spectral_centroid_std'] = float(spectral_centroid.std())
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=sr, hop_length=self.hop_length
        )
        features['spectral_bandwidth'] = float(spectral_bandwidth.mean())
        features['spectral_bandwidth_std'] = float(spectral_bandwidth.std())
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=sr, hop_length=self.hop_length
        )
        features['spectral_rolloff'] = float(spectral_rolloff.mean())
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features['zero_crossing_rate'] = float(zcr.mean())
        features['zero_crossing_rate_std'] = float(zcr.std())
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio, sr=sr, hop_length=self.hop_length
        )
        features['spectral_contrast'] = spectral_contrast.mean(axis=1).tolist()
        
        return features
    
    def extract_prosodic_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract prosodic (pitch and energy) features."""
        features = {}
        
        # Fundamental frequency (F0) estimation
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # Remove unvoiced segments
        f0_voiced = f0[voiced_flag]
        
        if len(f0_voiced) > 0:
            features['pitch_mean'] = float(np.mean(f0_voiced))
            features['pitch_std'] = float(np.std(f0_voiced))
            features['pitch_min'] = float(np.min(f0_voiced))
            features['pitch_max'] = float(np.max(f0_voiced))
            features['pitch_range'] = features['pitch_max'] - features['pitch_min']
        else:
            # Default values if no voiced segments
            features['pitch_mean'] = 0.0
            features['pitch_std'] = 0.0
            features['pitch_min'] = 0.0
            features['pitch_max'] = 0.0
            features['pitch_range'] = 0.0
        
        # Energy features
        energy = librosa.feature.rms(y=audio, hop_length=self.hop_length)
        features['energy_mean'] = float(energy.mean())
        features['energy_std'] = float(energy.std())
        features['energy_max'] = float(energy.max())
        
        return features
    
    def extract_temporal_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract temporal features."""
        features = {}
        
        # Tempo and beat tracking
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        features['tempo'] = float(tempo)
        
        # Onset detection
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        features['onset_rate'] = float(len(librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr
        )) / (len(audio) / sr))
        
        # Speech rate approximation (syllable rate)
        # Using onset detection as proxy for syllables
        features['speech_rate'] = features['onset_rate']
        
        return features
    
    def extract_voice_quality_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract voice quality features."""
        features = {}
        
        # Jitter (pitch perturbation)
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        if np.sum(voiced_flag) > 1:
            f0_voiced = f0[voiced_flag]
            period_lengths = sr / f0_voiced[:-1]
            period_diffs = np.abs(np.diff(period_lengths))
            features['jitter'] = float(np.mean(period_diffs) / np.mean(period_lengths))
        else:
            features['jitter'] = 0.0
        
        # Shimmer (amplitude perturbation)
        amplitude_env = np.abs(librosa.stft(audio))
        amplitude_diffs = np.abs(np.diff(amplitude_env.mean(axis=0)))
        features['shimmer'] = float(
            np.mean(amplitude_diffs) / np.mean(amplitude_env)
        )
        
        # Harmonics-to-Noise Ratio (HNR) approximation
        # Using spectral features as proxy
        spectral_flatness = librosa.feature.spectral_flatness(y=audio)
        features['hnr_proxy'] = float(1.0 - spectral_flatness.mean())
        
        return features
    
    def extract_formants(self, audio: np.ndarray, sr: int, n_formants: int = 3) -> Dict:
        """
        Extract formant frequencies using LPC analysis.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            n_formants: Number of formants to extract
            
        Returns:
            Dictionary with formant frequencies
        """
        features = {}
        
        # LPC analysis
        order = 2 + n_formants * 2
        a = librosa.lpc(audio, order=order)
        
        # Find roots of polynomial
        roots = np.roots(a)
        roots = [r for r in roots if np.imag(r) >= 0]
        
        # Convert to frequencies
        freqs = np.arctan2(np.imag(roots), np.real(roots)) * (sr / (2 * np.pi))
        freqs = sorted([f for f in freqs if f > 0])
        
        # Extract formants
        for i in range(min(n_formants, len(freqs))):
            features[f'formant_f{i+1}'] = float(freqs[i])
        
        return features
    
    def compute_similarity(self, features1: Dict, features2: Dict) -> float:
        """
        Compute similarity between two feature sets.
        
        Args:
            features1: First feature set
            features2: Second feature set
            
        Returns:
            Similarity score (0-1)
        """
        # Extract comparable features
        common_keys = set(features1.keys()) & set(features2.keys())
        
        if not common_keys:
            return 0.0
        
        similarities = []
        
        for key in common_keys:
            val1 = features1[key]
            val2 = features2[key]
            
            if isinstance(val1, list) and isinstance(val2, list):
                # For vector features, use cosine similarity
                if len(val1) == len(val2):
                    val1 = np.array(val1)
                    val2 = np.array(val2)
                    
                    norm1 = np.linalg.norm(val1)
                    norm2 = np.linalg.norm(val2)
                    
                    if norm1 > 0 and norm2 > 0:
                        similarity = np.dot(val1, val2) / (norm1 * norm2)
                        similarities.append(similarity)
            else:
                # For scalar features, use normalized difference
                if val1 > 0 and val2 > 0:
                    similarity = 1 - abs(val1 - val2) / max(val1, val2)
                    similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0