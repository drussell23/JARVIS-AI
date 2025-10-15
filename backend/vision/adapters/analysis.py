"""
Error Analysis Adapter for Follow-Up System
Extracts and categorizes errors from terminal text.
"""
import re
import logging
from typing import List

logger = logging.getLogger(__name__)

# Common error patterns (regex)
ERROR_PATTERNS = [
    # Python errors
    (r'(?:Traceback.*?\n)?(\w+Error:.*?)(?:\n|$)', 'python'),
    (r'(ModuleNotFoundError:.*?)(?:\n|$)', 'python_import'),
    (r'(ImportError:.*?)(?:\n|$)', 'python_import'),
    (r'(FileNotFoundError:.*?)(?:\n|$)', 'python_file'),
    (r'(PermissionError:.*?)(?:\n|$)', 'python_permission'),
    (r'(SyntaxError:.*?)(?:\n|$)', 'python_syntax'),
    (r'(IndentationError:.*?)(?:\n|$)', 'python_syntax'),
    (r'(KeyError:.*?)(?:\n|$)', 'python_runtime'),
    (r'(ValueError:.*?)(?:\n|$)', 'python_runtime'),
    (r'(TypeError:.*?)(?:\n|$)', 'python_runtime'),
    (r'(AttributeError:.*?)(?:\n|$)', 'python_runtime'),

    # JavaScript/Node errors
    (r'(Error:.*at.*?)(?:\n|$)', 'javascript'),
    (r'(ReferenceError:.*?)(?:\n|$)', 'javascript'),
    (r'(TypeError:.*?)(?:\n|$)', 'javascript'),
    (r'(SyntaxError:.*?)(?:\n|$)', 'javascript'),
    (r'(Cannot find module.*?)(?:\n|$)', 'node_module'),

    # Shell/Command errors
    (r'(command not found:.*?)(?:\n|$)', 'shell_command'),
    (r'(bash:.*command not found)', 'shell_command'),
    (r'(zsh:.*command not found)', 'shell_command'),
    (r'(No such file or directory)', 'shell_file'),
    (r'(Permission denied)', 'shell_permission'),

    # Compilation errors
    (r'(error:.*?)(?:\n|$)', 'compilation'),
    (r'(fatal error:.*?)(?:\n|$)', 'compilation'),
    (r'(\w+\.c:\d+:\d+:.*error:.*?)(?:\n|$)', 'c_compilation'),
    (r'(\w+\.cpp:\d+:\d+:.*error:.*?)(?:\n|$)', 'cpp_compilation'),

    # Package manager errors
    (r'(npm ERR!.*?)(?:\n|$)', 'npm'),
    (r'(pip install.*?error)', 'pip'),
    (r'(Could not find a version that satisfies.*?)(?:\n|$)', 'pip_version'),

    # Git errors
    (r'(fatal:.*?)(?:\n|$)', 'git'),
    (r'(error:.*?failed to push)', 'git_push'),

    # Generic error indicators
    (r'(FAILED.*?)(?:\n|$)', 'generic'),
    (r'(ERROR.*?)(?:\n|$)', 'generic'),
    (r'(\[ERROR\].*?)(?:\n|$)', 'generic'),
]


def extract_errors(text: str) -> List[str]:
    """
    Extract error messages from terminal/console text.

    Args:
        text: Raw text from terminal

    Returns:
        List of detected error messages
    """
    if not text:
        return []

    errors = []
    seen_errors = set()  # Deduplicate

    for pattern, error_type in ERROR_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)

        for match in matches:
            error_text = match.group(1).strip()

            # Skip if we've seen this exact error
            if error_text in seen_errors:
                continue

            seen_errors.add(error_text)
            errors.append(error_text)

            logger.debug(f"[ANALYSIS] Detected {error_type} error: {error_text[:80]}...")

    # If no errors found via patterns, look for common keywords
    if not errors:
        error_keywords = ['error', 'failed', 'exception', 'traceback', 'fatal']

        for keyword in error_keywords:
            if keyword in text.lower():
                # Extract a snippet around the keyword
                idx = text.lower().find(keyword)
                start = max(0, idx - 50)
                end = min(len(text), idx + 150)
                snippet = text[start:end].strip()

                if snippet and snippet not in seen_errors:
                    errors.append(snippet)
                    seen_errors.add(snippet)
                    break

    logger.info(f"[ANALYSIS] Extracted {len(errors)} error(s) from text")
    return errors


def suggest_fix(error: str) -> str:
    """
    Suggest a fix for common errors.

    Args:
        error: Error message

    Returns:
        Human-readable fix suggestion
    """
    error_lower = error.lower()

    # Python module/import errors
    if 'modulenotfounderror' in error_lower or 'no module named' in error_lower:
        # Extract module name
        match = re.search(r"module named ['\"](.+?)['\"]", error, re.IGNORECASE)
        if match:
            module_name = match.group(1)
            return f"Try installing the module: `pip install {module_name}` (or `pip3 install {module_name}`)."
        return "Try installing the missing module with pip or pip3."

    # Python file not found
    if 'filenotfounderror' in error_lower:
        return "Check if the file path is correct and the file exists."

    # Permission errors
    if 'permission' in error_lower and 'denied' in error_lower:
        return "Try running with appropriate permissions (e.g., `sudo` for system files, or fix file permissions)."

    # Command not found
    if 'command not found' in error_lower:
        match = re.search(r"command not found:\s*(\S+)", error, re.IGNORECASE)
        if match:
            command = match.group(1)
            return f"The command `{command}` is not installed or not in your PATH. Try installing it first."
        return "The command is not installed or not in your PATH."

    # Syntax errors
    if 'syntaxerror' in error_lower or 'indentationerror' in error_lower:
        return "Review the syntax in the indicated file and line. Check for typos, missing colons, or incorrect indentation."

    # NPM errors
    if 'npm err' in error_lower:
        return "Try clearing npm cache (`npm cache clean --force`) and reinstalling dependencies (`npm install`)."

    # Git errors
    if error_lower.startswith('fatal:') or 'git' in error_lower:
        if 'push' in error_lower or 'pull' in error_lower:
            return "Check your network connection and remote repository settings. You may need to pull changes first."
        return "Review your git configuration and repository status."

    # Generic fallback
    return "Review the error details above and check the documentation for the relevant tool or library."


async def summarize_terminal_state(text: str) -> str:
    """
    Generate a brief summary of terminal state.

    Args:
        text: Terminal text content

    Returns:
        Human-readable summary
    """
    if not text:
        return "Terminal appears to be empty."

    lines = text.strip().split('\n')
    line_count = len(lines)

    # Check for errors
    errors = extract_errors(text)
    if errors:
        return f"Terminal shows {len(errors)} error(s). Last {min(10, line_count)} lines of output available."

    # Check for success indicators
    success_patterns = [
        r'success',
        r'completed',
        r'done',
        r'\[OK\]',
        r'✓',
        r'passing',
    ]

    has_success = any(re.search(pattern, text, re.IGNORECASE) for pattern in success_patterns)

    if has_success:
        return f"Terminal shows successful output ({line_count} lines). Last command appears to have completed successfully."

    # Check for running processes
    if re.search(r'(running|listening|serving|watching)', text, re.IGNORECASE):
        return f"Terminal shows a running process ({line_count} lines of output)."

    # Generic summary
    return f"Terminal contains {line_count} lines of output. No obvious errors detected."
