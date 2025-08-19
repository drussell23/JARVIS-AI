#!/usr/bin/env python3
"""
Fallback screen capture method using screencapture command
"""

import subprocess
import tempfile
import os
from PIL import Image
import numpy as np


def capture_screen_fallback():
    """
    Capture screen using macOS screencapture command as fallback
    This often works when Quartz fails due to permissions
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        # Use screencapture command (usually has better permissions)
        result = subprocess.run(
            ["screencapture", "-x", "-C", tmp_path], capture_output=True, text=True
        )

        if result.returncode == 0 and os.path.exists(tmp_path):
            # Load image
            image = Image.open(tmp_path)
            # Convert to numpy array
            img_array = np.array(image)

            # Clean up temp file
            os.unlink(tmp_path)

            # Convert RGBA to RGB if needed
            if img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]

            return img_array
        else:
            return None

    except Exception as e:
        print(f"Fallback capture failed: {e}")
        return None
