# Yabai Space Detector - Important Notes

## ⚠️ Auto-Formatter Issues

The file `yabai_space_detector.py` has **specific indentation requirements** that auto-formatters (Black, autopep8) tend to break, causing syntax errors.

### Known Problematic Lines:
- **Line ~70**: `else:` statement indentation
- **Line ~169**: `return []` statement indentation
- **Line ~207**: `return {}` block indentation

### Protection Measures in Place:

#### ✅ Primary Protection (VS Code/Cursor)
Your `.vscode/settings.json` (local, not committed) should have:
```json
{
  "black-formatter.args": [
    "--extend-exclude=backend/vision/yabai_space_detector.py"
  ],
  "[python][**/backend/vision/yabai_space_detector.py]": {
    "editor.formatOnSave": false
  }
}
```

#### ✅ Secondary Protection (Project-wide)
- **`pyproject.toml`**: Black/isort exclusions
- **`setup.cfg`**: Flake8/autopep8 exclusions  
- **`.editorconfig`**: Editor-agnostic rules

### Before Committing:

Always verify syntax after any changes:
```bash
python -m py_compile backend/vision/yabai_space_detector.py
```

If you get syntax errors, the indentation was likely changed by a formatter.

### Manual Formatting:

If you MUST format this file:
1. **Make changes carefully**
2. **Run py_compile to verify**
3. **Fix any indentation errors**
4. **Test imports**: `python -c "from vision.yabai_space_detector import YabaiSpaceDetector"`

### Why This File?

This file has nested try/except blocks and conditional returns that confuse auto-formatters. The patterns are:
```python
if condition:
    return value
else:
    return other_value
```

Auto-formatters sometimes incorrectly indent the `else:` or the return statements.

## History

- The Objective-C space detector was removed due to segfaults
- Yabai is now the primary space detection method
- This file has had repeated indentation issues from formatters
- Exclusion configs added Oct 2025 to prevent future issues
