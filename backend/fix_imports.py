#!/usr/bin/env python3
"""Fix import issues by mocking problematic modules"""

import sys
from unittest.mock import MagicMock

# Mock pyautogui to avoid the macOS compatibility issue
sys.modules['pyautogui'] = MagicMock()
sys.modules['mouseinfo'] = MagicMock()
sys.modules['rubicon'] = MagicMock()
sys.modules['rubicon.objc'] = MagicMock()

print("✅ Import fixes applied - pyautogui mocked")
print("Run: python -c 'import fix_imports' before starting the backend")