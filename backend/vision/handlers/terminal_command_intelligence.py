"""
Terminal Command Intelligence with Safety Tiers
Enhances terminal follow-up with intelligent command suggestions and safety classification.

Features:
- Detects commands from terminal OCR
- Classifies command safety (GREEN/YELLOW/RED)
- Suggests safer alternatives for dangerous commands
- Provides dry-run options when available
- Never auto-executes destructive commands
"""
from __future__ import annotations

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TerminalCommandContext:
    """Context extracted from terminal analysis."""
    last_command: Optional[str] = None
    command_output: Optional[str] = None
    errors: List[str] = None
    current_directory: Optional[str] = None
    shell_type: Optional[str] = None  # bash, zsh, fish, etc.

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class CommandSuggestion:
    """Suggested command with safety analysis."""
    command: str
    purpose: str
    safety_tier: str  # green, yellow, red
    requires_confirmation: bool
    dry_run_command: Optional[str] = None
    estimated_impact: str = ""  # "No impact", "Installs packages", "Deletes files", etc.
    confidence: float = 0.8


class TerminalCommandIntelligence:
    """
    Intelligent terminal command analysis and suggestion system.

    Integrates with:
    - CommandSafetyClassifier for risk assessment
    - Error pattern detection
    - Context-aware command suggestions
    """

    def __init__(self):
        """Initialize terminal command intelligence."""
        from backend.system_control.command_safety import get_command_classifier

        self.safety_classifier = get_command_classifier()

        # Common error -> fix command mappings
        self.error_fix_patterns = [
            # Python errors
            (
                r"ModuleNotFoundError: No module named ['\"]?(\w+)['\"]?",
                lambda m: CommandSuggestion(
                    command=f"pip install {m.group(1)}",
                    purpose=f"Install missing Python module '{m.group(1)}'",
                    safety_tier="yellow",
                    requires_confirmation=True,
                    estimated_impact=f"Installs Python package '{m.group(1)}'",
                ),
            ),
            (
                r"ImportError: cannot import name '(\w+)'",
                lambda m: CommandSuggestion(
                    command="pip install --upgrade -r requirements.txt",
                    purpose="Update dependencies that may have changed",
                    safety_tier="yellow",
                    requires_confirmation=True,
                    estimated_impact="Updates all Python packages in requirements.txt",
                ),
            ),
            (
                r"SyntaxError: .*line (\d+)",
                lambda m: CommandSuggestion(
                    command=f"# Check syntax error on line {m.group(1)}",
                    purpose=f"Review code at line {m.group(1)} for syntax issues",
                    safety_tier="green",
                    requires_confirmation=False,
                    estimated_impact="No impact - informational only",
                ),
            ),

            # npm/Node errors
            (
                r"Cannot find module '([\w\-@/]+)'",
                lambda m: CommandSuggestion(
                    command=f"npm install {m.group(1)}",
                    purpose=f"Install missing npm module '{m.group(1)}'",
                    safety_tier="yellow",
                    requires_confirmation=True,
                    estimated_impact=f"Installs npm package '{m.group(1)}'",
                ),
            ),
            (
                r"ENOENT: no such file or directory.*package\.json",
                lambda m: CommandSuggestion(
                    command="npm init -y",
                    purpose="Initialize npm project with package.json",
                    safety_tier="yellow",
                    requires_confirmation=True,
                    estimated_impact="Creates package.json in current directory",
                ),
            ),

            # Git errors
            (
                r"fatal: not a git repository",
                lambda m: CommandSuggestion(
                    command="git init",
                    purpose="Initialize git repository in current directory",
                    safety_tier="yellow",
                    requires_confirmation=True,
                    estimated_impact="Creates .git directory",
                ),
            ),
            (
                r"Your branch is behind.*by (\d+) commit",
                lambda m: CommandSuggestion(
                    command="git pull",
                    purpose=f"Pull {m.group(1)} commit(s) from remote",
                    safety_tier="yellow",
                    requires_confirmation=True,
                    estimated_impact="Updates local branch with remote changes",
                ),
            ),
            (
                r"Changes not staged for commit",
                lambda m: CommandSuggestion(
                    command="git add .",
                    purpose="Stage all changes for commit",
                    safety_tier="yellow",
                    requires_confirmation=True,
                    estimated_impact="Stages modified files (reversible with git reset)",
                ),
            ),

            # Permission errors
            (
                r"Permission denied.*(\S+)",
                lambda m: CommandSuggestion(
                    command=f"chmod +x {m.group(1)}",
                    purpose=f"Make '{m.group(1)}' executable",
                    safety_tier="yellow",
                    requires_confirmation=True,
                    estimated_impact=f"Adds execute permission to {m.group(1)}",
                ),
            ),

            # Command not found
            (
                r"command not found: (\w+)",
                lambda m: CommandSuggestion(
                    command=f"# Install '{m.group(1)}' using your package manager",
                    purpose=f"Command '{m.group(1)}' not found - may need installation",
                    safety_tier="green",
                    requires_confirmation=False,
                    estimated_impact="No impact - informational only",
                ),
            ),

            # Port already in use
            (
                r"Address already in use.*:(\d+)",
                lambda m: CommandSuggestion(
                    command=f"lsof -ti :{m.group(1)} | xargs kill",
                    purpose=f"Kill process using port {m.group(1)}",
                    safety_tier="yellow",
                    requires_confirmation=True,
                    estimated_impact=f"Terminates process listening on port {m.group(1)}",
                ),
            ),

            # Docker errors
            (
                r"Cannot connect to the Docker daemon",
                lambda m: CommandSuggestion(
                    command="# Start Docker Desktop or run: sudo systemctl start docker",
                    purpose="Docker daemon not running",
                    safety_tier="green",
                    requires_confirmation=False,
                    estimated_impact="No impact - informational only",
                ),
            ),
        ]

        logger.info("[TERMINAL-CMD-INTEL] Initialized with command safety classification")

    async def analyze_terminal_context(self, ocr_text: str) -> TerminalCommandContext:
        """
        Extract context from terminal OCR text.

        Args:
            ocr_text: Raw OCR text from terminal window

        Returns:
            TerminalCommandContext with extracted information
        """
        lines = ocr_text.strip().split('\n')

        # Detect last command (look for common prompt patterns)
        last_command = self._extract_last_command(lines)

        # Extract command output (everything after last prompt)
        command_output = self._extract_command_output(lines)

        # Detect errors
        errors = self._extract_errors(ocr_text)

        # Detect current directory
        current_dir = self._extract_current_directory(lines)

        # Detect shell type
        shell_type = self._detect_shell_type(lines)

        return TerminalCommandContext(
            last_command=last_command,
            command_output=command_output,
            errors=errors,
            current_directory=current_dir,
            shell_type=shell_type,
        )

    def _extract_last_command(self, lines: List[str]) -> Optional[str]:
        """Extract the last command executed from terminal lines."""
        # Look for common prompt patterns
        prompt_patterns = [
            r'[$%#]\s+(.+?)$',  # bash/zsh standard prompts
            r'>>>\s+(.+?)$',    # Python REPL
            r'>\s+(.+?)$',      # Windows cmd
        ]

        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]
            for pattern in prompt_patterns:
                match = re.search(pattern, line)
                if match:
                    cmd = match.group(1).strip()
                    if cmd and not cmd.startswith('#'):  # Skip comments
                        return cmd

        return None

    def _extract_command_output(self, lines: List[str]) -> Optional[str]:
        """Extract command output (text after last prompt)."""
        # Find last prompt line
        last_prompt_idx = -1
        for i in range(len(lines) - 1, -1, -1):
            if any(p in lines[i] for p in ['$', '%', '#', '>>>', '>']):
                last_prompt_idx = i
                break

        if last_prompt_idx >= 0 and last_prompt_idx < len(lines) - 1:
            output_lines = lines[last_prompt_idx + 1:]
            return '\n'.join(output_lines).strip()

        return None

    def _extract_errors(self, ocr_text: str) -> List[str]:
        """Extract error messages from terminal text."""
        errors = []

        # Error patterns - preserve full error name (e.g., ModuleNotFoundError)
        error_indicators = [
            r'\w*Error:?\s+.+',      # Catches ModuleNotFoundError, ImportError, etc.
            r'\w*Exception:?\s+.+',  # Catches all exception types
            r'ERROR:?\s+(.+)',
            r'Traceback.*',
            r'fatal:?\s+(.+)',
            r'Failed:?\s+(.+)',
            r'\[ERROR\]\s+(.+)',
        ]

        for pattern in error_indicators:
            matches = re.finditer(pattern, ocr_text, re.IGNORECASE)
            for match in matches:
                error = match.group(0).strip()
                if error and error not in errors:
                    errors.append(error)

        return errors

    def _extract_current_directory(self, lines: List[str]) -> Optional[str]:
        """Extract current working directory from prompt."""
        # Look for common directory indicators in prompts
        for line in reversed(lines[-10:]):  # Check last 10 lines
            # Match patterns like "user@host:/path/to/dir $"
            match = re.search(r'[~\/][\w\/\-\.]+(?=\s*[$%#>])', line)
            if match:
                return match.group(0)

        return None

    def _detect_shell_type(self, lines: List[str]) -> Optional[str]:
        """Detect shell type from terminal indicators."""
        text = '\n'.join(lines[-20:])  # Check last 20 lines

        if 'zsh' in text.lower():
            return 'zsh'
        elif 'bash' in text.lower():
            return 'bash'
        elif 'fish' in text.lower():
            return 'fish'
        elif '>>>' in text:
            return 'python'
        elif 'node>' in text.lower():
            return 'node'

        return 'unknown'

    async def suggest_fix_commands(
        self,
        context: TerminalCommandContext,
    ) -> List[CommandSuggestion]:
        """
        Suggest commands to fix detected errors.

        Args:
            context: Terminal context with errors

        Returns:
            List of CommandSuggestion objects
        """
        suggestions = []

        for error in context.errors:
            # Try to match error patterns
            for pattern, suggestion_func in self.error_fix_patterns:
                match = re.search(pattern, error, re.IGNORECASE)
                if match:
                    try:
                        suggestion = suggestion_func(match)

                        # Classify command safety
                        classification = self.safety_classifier.classify(suggestion.command)

                        # Update suggestion with safety info
                        suggestion.safety_tier = classification.tier.value
                        suggestion.requires_confirmation = classification.requires_confirmation
                        suggestion.dry_run_command = (
                            classification.suggested_alternative
                            if classification.dry_run_available
                            else None
                        )

                        suggestions.append(suggestion)
                        break  # Only match first pattern per error

                    except Exception as e:
                        logger.error(f"[TERMINAL-CMD-INTEL] Error creating suggestion: {e}")

        return suggestions

    async def classify_command(self, command: str) -> Dict[str, Any]:
        """
        Classify a command by safety tier.

        Args:
            command: Shell command to classify

        Returns:
            Dictionary with classification details
        """
        classification = self.safety_classifier.classify(command)

        return {
            'command': command,
            'tier': classification.tier.value,
            'tier_color': self._get_tier_color(classification.tier.value),
            'is_safe': classification.is_safe,
            'is_destructive': classification.is_destructive,
            'requires_confirmation': classification.requires_confirmation,
            'is_reversible': classification.is_reversible,
            'confidence': classification.confidence,
            'risk_categories': [r.value for r in classification.risk_categories],
            'reasoning': classification.reasoning,
            'safer_alternative': classification.suggested_alternative,
            'dry_run_available': classification.dry_run_available,
        }

    def _get_tier_color(self, tier: str) -> str:
        """Get color for safety tier (for UI display)."""
        colors = {
            'green': '#00C853',   # Material Green A700
            'yellow': '#FFC400',  # Material Amber A700
            'red': '#D50000',     # Material Red A700
            'unknown': '#9E9E9E', # Material Grey
        }
        return colors.get(tier, colors['unknown'])

    async def format_suggestion_for_user(
        self,
        suggestion: CommandSuggestion,
        include_safety_warning: bool = True,
    ) -> str:
        """
        Format command suggestion for user display.

        Args:
            suggestion: CommandSuggestion to format
            include_safety_warning: Include safety tier information

        Returns:
            Formatted string for user
        """
        # Safety tier emoji
        tier_emoji = {
            'green': '✅',
            'yellow': '⚠️',
            'red': '🛑',
        }
        emoji = tier_emoji.get(suggestion.safety_tier, '❓')

        # Build message
        parts = [
            f"{emoji} **{suggestion.purpose}**",
            f"",
            f"```bash",
            f"{suggestion.command}",
            f"```",
        ]

        if include_safety_warning:
            if suggestion.safety_tier == 'red':
                parts.append("")
                parts.append(f"⚠️ **Warning:** This command is potentially destructive!")
                parts.append(f"Impact: {suggestion.estimated_impact}")

            elif suggestion.safety_tier == 'yellow':
                parts.append("")
                parts.append(f"📝 Impact: {suggestion.estimated_impact}")

        if suggestion.dry_run_command:
            parts.append("")
            parts.append(f"💡 **Dry-run option:**")
            parts.append(f"```bash")
            parts.append(f"{suggestion.dry_run_command}")
            parts.append(f"```")

        return '\n'.join(parts)


# Global instance
_global_terminal_intelligence: Optional[TerminalCommandIntelligence] = None


def get_terminal_intelligence() -> TerminalCommandIntelligence:
    """Get or create global terminal command intelligence."""
    global _global_terminal_intelligence

    if _global_terminal_intelligence is None:
        _global_terminal_intelligence = TerminalCommandIntelligence()

    return _global_terminal_intelligence
