#!/usr/bin/env python3
"""Start JARVIS backend with import fixes"""

import sys
from unittest.mock import MagicMock

# Mock problematic imports before they're loaded
sys.modules['pyautogui'] = MagicMock()
sys.modules['mouseinfo'] = MagicMock()
sys.modules['rubicon'] = MagicMock()
sys.modules['rubicon.objc'] = MagicMock()

# Now import and run the main module
import main

print("✅ JARVIS backend started with import fixes")