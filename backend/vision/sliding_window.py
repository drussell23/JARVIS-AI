"""
Python implementation of sliding window for frame analysis.
Used as fallback when Rust implementation is unavailable.
"""

import time
import hashlib
from collections import deque
from typing import Dict, Any, List, Optional, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SlidingWindow:
    """
    Python implementation of sliding window for duplicate frame detection
    and temporal analysis.
    """
    
    def __init__(self, window_size: int = 30, overlap_threshold: float = 0.9):
        """
        Initialize sliding window.
        
        Args:
            window_size: Number of frames to keep in window
            overlap_threshold: Similarity threshold for duplicate detection
        """
        self.window_size = window_size
        self.overlap_threshold = overlap_threshold
        self.frames = deque(maxlen=window_size)
        self.frame_hashes = deque(maxlen=window_size)
        self.stats = {
            'total_frames': 0,
            'duplicates_found': 0,
            'average_process_time': 0.0
        }
        
        logger.info(f"Python sliding window initialized: size={window_size}, threshold={overlap_threshold}")
        
    def add_frame(self, frame_data: Union[Dict[str, Any], bytes], timestamp: float) -> Dict[str, Any]:
        """
        Add a frame to the sliding window and check for duplicates.
        
        Args:
            frame_data: Frame data (dict with 'data' key or raw bytes)
            timestamp: Frame timestamp
            
        Returns:
            Dict with analysis results
        """
        start_time = time.time()
        
        # Extract frame bytes
        if isinstance(frame_data, dict):
            import base64
            data = frame_data.get('data', b'')
            if isinstance(data, str):
                data = base64.b64decode(data)
            width = frame_data.get('width', 0)
            height = frame_data.get('height', 0)
        else:
            data = frame_data
            width = height = 0
            
        # Compute frame hash
        frame_hash = self._compute_hash(data)
        
        # Check for duplicates in window
        is_duplicate = False
        confidence = 0.0
        
        if frame_hash in self.frame_hashes:
            is_duplicate = True
            confidence = 1.0
            self.stats['duplicates_found'] += 1
        else:
            # Check for similar frames (simplified check)
            for i, existing_hash in enumerate(self.frame_hashes):
                similarity = self._compute_similarity(frame_hash, existing_hash)
                if similarity >= self.overlap_threshold:
                    is_duplicate = True
                    confidence = similarity
                    self.stats['duplicates_found'] += 1
                    break
                    
        # Add to window
        self.frames.append({
            'data': data,
            'timestamp': timestamp,
            'width': width,
            'height': height,
            'hash': frame_hash
        })
        self.frame_hashes.append(frame_hash)
        
        # Update stats
        self.stats['total_frames'] += 1
        process_time = time.time() - start_time
        self.stats['average_process_time'] = (
            (self.stats['average_process_time'] * (self.stats['total_frames'] - 1) + process_time) 
            / self.stats['total_frames']
        )
        
        return {
            'is_duplicate': is_duplicate,
            'confidence': confidence,
            'window_size': len(self.frames),
            'process_time': process_time,
            'frame_hash': frame_hash
        }
        
    def _compute_hash(self, data: bytes) -> str:
        """Compute hash of frame data."""
        # Use sampling for faster hashing of large frames
        if len(data) > 10000:
            # Sample every 10th byte
            sampled = data[::10]
        else:
            sampled = data
            
        return hashlib.sha256(sampled).hexdigest()
        
    def _compute_similarity(self, hash1: str, hash2: str) -> float:
        """
        Compute similarity between two hashes.
        Simple implementation - in practice would use perceptual hashing.
        """
        # Simple character-based similarity
        matches = sum(1 for a, b in zip(hash1, hash2) if a == b)
        return matches / len(hash1)
        
    def get_window_frames(self) -> List[Dict[str, Any]]:
        """Get current frames in the window."""
        return list(self.frames)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get window statistics."""
        return {
            **self.stats,
            'current_window_size': len(self.frames),
            'duplicate_rate': (
                self.stats['duplicates_found'] / max(1, self.stats['total_frames'])
            )
        }
        
    def clear(self):
        """Clear the window."""
        self.frames.clear()
        self.frame_hashes.clear()
        
    def process_frame(self, frame_data: bytes, timestamp: float) -> Dict[str, Any]:
        """
        Rust-compatible interface for processing frames.
        """
        return self.add_frame(frame_data, timestamp)
        
    def __repr__(self):
        stats = self.get_stats()
        return (f"SlidingWindow(size={self.window_size}, "
                f"frames={len(self.frames)}, "
                f"duplicates={stats['duplicate_rate']:.1%})")