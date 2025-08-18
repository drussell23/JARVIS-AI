#!/usr/bin/env python3
"""
Fix import issues with TensorFlow and transformers
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_tensorflow_imports():
    """Fix TensorFlow import issues by setting environment variables"""
    # Disable TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # Try to fix TensorFlow data attribute issue
    try:
        import tensorflow as tf
        if not hasattr(tf, 'data'):
            # Create a mock data module to prevent import errors
            class MockData:
                class Dataset:
                    @staticmethod
                    def from_tensor_slices(*args, **kwargs):
                        return None
            tf.data = MockData()
            logger.info("Applied TensorFlow data module fix")
    except ImportError:
        logger.warning("TensorFlow not installed - some features may be limited")
    except Exception as e:
        logger.warning(f"Could not fix TensorFlow imports: {e}")

def disable_tensorflow_in_transformers():
    """Disable TensorFlow backend in transformers"""
    os.environ['USE_TORCH'] = '1'
    os.environ['USE_TF'] = '0'
    logger.info("Configured transformers to use PyTorch backend only")

if __name__ == "__main__":
    print("Fixing import issues...")
    fix_tensorflow_imports()
    disable_tensorflow_in_transformers()
    print("Import fixes applied!")