"""
Code Analysis Adapter for Follow-Up System
Analyzes code editor windows.
"""
import logging
from typing import Dict, Any, Optional
import re

logger = logging.getLogger(__name__)


async def analyze_code_window(window_id: str, snapshot_id: str) -> Optional[Dict[str, Any]]:
    """
    Analyze a code editor window.

    Args:
        window_id: Editor window identifier
        snapshot_id: Screenshot snapshot ID

    Returns:
        Dict with file_path, language, diagnostics, etc.
    """
    try:
        from backend.vision.adapters.ocr import ocr_text_from_snapshot

        # Extract text
        text = await ocr_text_from_snapshot(snapshot_id)

        if not text:
            logger.warning(f"[CODE] No text extracted from {snapshot_id}")
            return None

        # Detect language
        language = _detect_language(text)

        # Extract file path (if visible in window title/tab)
        file_path = _extract_file_path(text)

        # Detect diagnostics/errors
        diagnostics = _extract_diagnostics(text, language)

        analysis = {
            "file_path": file_path or "Unknown file",
            "language": language,
            "diagnostics": diagnostics,
            "line_count": len(text.split('\n')),
            "has_errors": len([d for d in diagnostics if d['severity'] == 'error']) > 0,
            "has_warnings": len([d for d in diagnostics if d['severity'] == 'warning']) > 0,
        }

        logger.info(
            f"[CODE] Analyzed code window: {language}, "
            f"{len(diagnostics)} diagnostic(s), "
            f"errors={analysis['has_errors']}"
        )

        return analysis

    except Exception as e:
        logger.error(f"[CODE] Failed to analyze code window: {e}", exc_info=True)
        return None


def _detect_language(text: str) -> str:
    """Detect programming language from code text."""
    # Language-specific patterns
    language_indicators = {
        'python': [
            r'^\s*def\s+\w+\(', r'^\s*class\s+\w+:',
            r'^\s*import\s+\w+', r'^\s*from\s+\w+\s+import',
            r'if __name__ == ["\']__main__["\']',
        ],
        'javascript': [
            r'^\s*function\s+\w+\(', r'^\s*const\s+\w+\s*=',
            r'^\s*let\s+\w+\s*=', r'^\s*var\s+\w+\s*=',
            r'require\(["\']', r'import.*from\s+["\']',
        ],
        'typescript': [
            r':\s*\w+\s*=', r'interface\s+\w+',
            r'type\s+\w+\s*=', r'as\s+\w+',
        ],
        'java': [
            r'^\s*public\s+class\s+\w+', r'^\s*private\s+\w+\s+\w+\(',
            r'^\s*public\s+static\s+void\s+main',
        ],
        'c': [
            r'^\s*#include\s*<', r'^\s*int\s+main\(',
            r'^\s*void\s+\w+\(', r'printf\(',
        ],
        'cpp': [
            r'^\s*#include\s*<', r'std::',
            r'^\s*template\s*<', r'cout\s*<<',
        ],
        'go': [
            r'^\s*func\s+\w+\(', r'^\s*package\s+\w+',
            r'^\s*import\s+\(', r'^\s*type\s+\w+\s+struct',
        ],
        'rust': [
            r'^\s*fn\s+\w+\(', r'^\s*pub\s+fn\s+\w+\(',
            r'^\s*use\s+\w+::', r'let\s+mut\s+\w+',
        ],
        'ruby': [
            r'^\s*def\s+\w+', r'^\s*class\s+\w+',
            r'^\s*require\s+["\']', r'^\s*end\s*$',
        ],
    }

    # Count matches for each language
    scores = {}
    for lang, patterns in language_indicators.items():
        score = sum(1 for p in patterns if re.search(p, text, re.MULTILINE))
        if score > 0:
            scores[lang] = score

    if scores:
        detected = max(scores, key=scores.get)
        logger.debug(f"[CODE] Detected language: {detected} (score={scores[detected]})")
        return detected

    return "unknown"


def _extract_file_path(text: str) -> Optional[str]:
    """Extract file path from code window text."""
    # Look for common file path patterns
    path_patterns = [
        r'([/~][\w\-./]+\.[\w]+)',  # Unix-style path
        r'([A-Z]:\\[\w\-\\]+\.[\w]+)',  # Windows path
        r'(\w+/[\w\-/]+\.[\w]+)',  # Relative path
    ]

    for pattern in path_patterns:
        match = re.search(pattern, text)
        if match:
            path = match.group(1)
            # Filter out unlikely paths
            if not any(noise in path for noise in ['http', 'www', '@']):
                logger.debug(f"[CODE] Extracted file path: {path}")
                return path

    return None


def _extract_diagnostics(text: str, language: str) -> list[Dict[str, Any]]:
    """Extract diagnostics (errors/warnings) from code editor."""
    diagnostics = []

    # Common diagnostic patterns (VS Code, IntelliJ, etc.)
    diagnostic_patterns = [
        # VS Code format: "file.py:10:5 - error: message"
        (r'(\w+\.[\w]+):(\d+):(\d+)\s*-\s*(error|warning):\s*(.+)', 'vscode'),

        # IntelliJ format
        (r'(\w+\.[\w]+):(\d+):\s*(error|warning):\s*(.+)', 'intellij'),

        # Generic format
        (r'(error|warning):\s*line\s*(\d+):\s*(.+)', 'generic'),

        # Python traceback
        (r'File\s+"(.+)",\s*line\s*(\d+)', 'python_traceback'),
    ]

    for pattern, source in diagnostic_patterns:
        matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)

        for match in matches:
            if source == 'vscode':
                diagnostic = {
                    "file": match.group(1),
                    "line": int(match.group(2)),
                    "column": int(match.group(3)),
                    "severity": match.group(4).lower(),
                    "message": match.group(5).strip(),
                }
            elif source == 'intellij':
                diagnostic = {
                    "file": match.group(1),
                    "line": int(match.group(2)),
                    "column": None,
                    "severity": match.group(3).lower(),
                    "message": match.group(4).strip(),
                }
            elif source == 'generic':
                diagnostic = {
                    "file": None,
                    "line": int(match.group(2)),
                    "column": None,
                    "severity": match.group(1).lower(),
                    "message": match.group(3).strip(),
                }
            elif source == 'python_traceback':
                diagnostic = {
                    "file": match.group(1),
                    "line": int(match.group(2)),
                    "column": None,
                    "severity": "error",
                    "message": "Exception in code",
                }

            diagnostics.append(diagnostic)

    logger.debug(f"[CODE] Extracted {len(diagnostics)} diagnostic(s)")
    return diagnostics[:10]  # Limit to 10
