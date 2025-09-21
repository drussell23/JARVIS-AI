"""
Python implementation of bloom filter for fallback when Rust is unavailable.
"""

import hashlib
import math
import numpy as np
from typing import Union, List
import logging

logger = logging.getLogger(__name__)

class PythonBloomFilter:
    """
    Python implementation of a bloom filter for duplicate detection.
    Uses multiple hash functions to minimize false positive rate.
    """
    
    def __init__(self, size_mb: float = 10.0, num_hashes: int = 7):
        """
        Initialize bloom filter.
        
        Args:
            size_mb: Size in megabytes
            num_hashes: Number of hash functions to use
        """
        self.size_bits = int(size_mb * 1024 * 1024 * 8)  # Convert MB to bits
        self.size_bytes = self.size_bits // 8
        self.num_hashes = num_hashes
        
        # Use numpy for efficient bit operations
        self.bit_array = np.zeros(self.size_bytes, dtype=np.uint8)
        
        # Track statistics
        self.items_added = 0
        self.false_positive_rate = 0.0
        
        logger.info(f"Python bloom filter initialized: {size_mb}MB, {num_hashes} hashes")
        
    def _hash_functions(self, item: bytes) -> List[int]:
        """Generate multiple hash values for an item."""
        hashes = []
        
        # Use different hash algorithms for diversity
        algorithms = ['md5', 'sha1', 'sha256', 'sha384', 'sha512', 'sha3_256', 'blake2b']
        
        for i in range(self.num_hashes):
            # Create hash with salt
            hasher = hashlib.new(algorithms[i % len(algorithms)])
            hasher.update(item)
            hasher.update(str(i).encode())
            
            # Get hash value and map to bit position
            hash_bytes = hasher.digest()
            hash_int = int.from_bytes(hash_bytes[:8], 'big')
            bit_pos = hash_int % self.size_bits
            
            hashes.append(bit_pos)
            
        return hashes
        
    def add(self, item: Union[str, bytes]):
        """Add an item to the bloom filter."""
        if isinstance(item, str):
            item = item.encode('utf-8')
            
        positions = self._hash_functions(item)
        
        for pos in positions:
            byte_idx = pos // 8
            bit_idx = pos % 8
            # Ensure we don't go out of bounds
            if byte_idx < len(self.bit_array):
                self.bit_array[byte_idx] |= (1 << bit_idx)
            
        self.items_added += 1
        self._update_false_positive_rate()
        
    def contains(self, item: Union[str, bytes]) -> bool:
        """Check if an item might be in the set."""
        if isinstance(item, str):
            item = item.encode('utf-8')
            
        positions = self._hash_functions(item)
        
        for pos in positions:
            byte_idx = pos // 8
            bit_idx = pos % 8
            # Check bounds
            if byte_idx >= len(self.bit_array):
                return False
            if not (self.bit_array[byte_idx] & (1 << bit_idx)):
                return False
                
        return True
        
    def clear(self):
        """Clear the bloom filter."""
        self.bit_array.fill(0)
        self.items_added = 0
        self.false_positive_rate = 0.0
        
    def get_saturation(self) -> float:
        """Get the saturation level (0.0 to 1.0)."""
        # Count set bits
        set_bits = 0
        for byte in self.bit_array:
            set_bits += bin(byte).count('1')
            
        return set_bits / self.size_bits
        
    def _update_false_positive_rate(self):
        """Update estimated false positive rate."""
        if self.items_added == 0:
            self.false_positive_rate = 0.0
            return
            
        # Use bloom filter formula
        # p ≈ (1 - e^(-kn/m))^k
        # where k = num hashes, n = items added, m = size in bits
        k = self.num_hashes
        n = self.items_added
        m = self.size_bits
        
        self.false_positive_rate = math.pow(1 - math.exp(-k * n / m), k)
        
    def export_state(self) -> dict:
        """Export current state for migration."""
        return {
            'bit_array': self.bit_array.tobytes(),
            'size_mb': self.size_bits / (1024 * 1024 * 8),
            'num_hashes': self.num_hashes,
            'items_added': self.items_added
        }
        
    def import_state(self, state: dict):
        """Import state from another bloom filter."""
        if 'bit_array' in state:
            # Create a writable copy
            self.bit_array = np.frombuffer(state['bit_array'], dtype=np.uint8).copy()
            self.items_added = state.get('items_added', 0)
            self._update_false_positive_rate()
            
    def __repr__(self):
        return (f"PythonBloomFilter(size={self.size_bits//(1024*1024*8)}MB, "
                f"items={self.items_added}, saturation={self.get_saturation():.2%}, "
                f"fp_rate={self.false_positive_rate:.4%})")