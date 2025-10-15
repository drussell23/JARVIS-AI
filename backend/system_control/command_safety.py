"""
Command Safety Tier Classification System
Classifies shell commands by risk level and determines execution requirements.

Philosophy:
- Never auto-execute destructive commands
- Require confirmation for file system changes
- Allow safe read-only commands freely
- Track commands that can be reverted
- Learn from user overrides

Safety Tiers:
- GREEN (Safe): Read-only, no side effects, auto-executable
- YELLOW (Caution): Modifies state, requires confirmation once
- RED (Dangerous): Irreversible/destructive, always confirm
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
import shlex

logger = logging.getLogger(__name__)


class SafetyTier(str, Enum):
    """Command safety classification tiers."""
    GREEN = "green"      # Safe, auto-executable
    YELLOW = "yellow"    # Caution, confirm once
    RED = "red"          # Dangerous, always confirm
    UNKNOWN = "unknown"  # Unclassified, default to caution


class RiskCategory(str, Enum):
    """Categories of command risks."""
    DATA_LOSS = "data_loss"                      # rm, dd, format
    SYSTEM_MODIFICATION = "system_modification"  # chmod, chown, sudo
    NETWORK_EXPOSURE = "network_exposure"        # curl, wget with pipes
    PROCESS_CONTROL = "process_control"          # kill, pkill
    FILE_MODIFICATION = "file_modification"      # mv, cp, write operations
    PACKAGE_MANAGEMENT = "package_management"    # npm, pip, brew install
    VERSION_CONTROL = "version_control"          # git push, git reset
    DATABASE_OPERATION = "database_operation"    # DROP, DELETE, TRUNCATE
    SAFE_READ = "safe_read"                      # ls, cat, grep
    SAFE_NAVIGATION = "safe_navigation"          # cd, pwd


@dataclass
class CommandClassification:
    """Result of command safety classification."""
    command: str
    tier: SafetyTier
    risk_categories: List[RiskCategory]
    requires_confirmation: bool
    is_reversible: bool
    confidence: float  # 0.0-1.0
    reasoning: str
    suggested_alternative: Optional[str] = None
    dry_run_available: bool = False

    @property
    def is_safe(self) -> bool:
        """Quick check if command is safe to execute."""
        return self.tier == SafetyTier.GREEN

    @property
    def is_destructive(self) -> bool:
        """Check if command is potentially destructive."""
        return (
            self.tier == SafetyTier.RED
            or RiskCategory.DATA_LOSS in self.risk_categories
        )


class CommandSafetyClassifier:
    """
    Classifies shell commands by safety tier.

    Uses pattern matching, command parsing, and heuristics to determine
    if a command is safe to execute automatically or requires user confirmation.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize command safety classifier.

        Args:
            config_path: Optional path to custom safety rules JSON
        """
        # GREEN tier: Safe, read-only commands
        self.green_commands: Set[str] = {
            # File viewing
            'ls', 'cat', 'less', 'more', 'head', 'tail', 'file', 'stat',
            'wc', 'du', 'df', 'tree',

            # Text processing
            'grep', 'egrep', 'fgrep', 'sed', 'awk', 'cut', 'sort', 'uniq',
            'tr', 'diff', 'comm', 'cmp',

            # Navigation
            'cd', 'pwd', 'pushd', 'popd', 'dirs',

            # Process inspection
            'ps', 'top', 'htop', 'pgrep', 'jobs', 'fg', 'bg',

            # System info
            'uname', 'hostname', 'whoami', 'id', 'groups', 'uptime', 'date',
            'cal', 'which', 'whereis', 'whatis', 'man', 'info', 'env',

            # Git read-only
            'git status', 'git log', 'git diff', 'git show', 'git branch',
            'git remote', 'git config --get', 'git ls-files', 'git blame',

            # Network inspection
            'ping', 'netstat', 'ifconfig', 'ip addr', 'nslookup', 'dig',
            'traceroute', 'mtr',

            # Python/Node inspection
            'python --version', 'python -m pip list', 'node --version',
            'npm list', 'pip list', 'pip show',

            # Docker inspection
            'docker ps', 'docker images', 'docker logs', 'docker inspect',

            # Other safe utilities
            'echo', 'printf', 'sleep', 'true', 'false', 'yes', 'seq',
            'basename', 'dirname', 'realpath', 'readlink',
        }

        # YELLOW tier: Modify state but generally safe with confirmation
        self.yellow_commands: Set[str] = {
            # Package management
            'npm install', 'npm update', 'npm ci',
            'pip install', 'pip install --upgrade',
            'brew install', 'brew upgrade', 'brew update',
            'apt install', 'apt update', 'apt upgrade',
            'yum install', 'yum update',

            # Git operations
            'git add', 'git commit', 'git pull', 'git fetch', 'git merge',
            'git checkout', 'git switch', 'git stash', 'git cherry-pick',

            # File operations (non-destructive)
            'cp', 'mv', 'mkdir', 'touch', 'ln',

            # Build operations
            'make', 'cmake', 'cargo build', 'go build', 'npm run build',
            'python setup.py install',

            # Testing
            'pytest', 'npm test', 'cargo test', 'go test', 'jest',

            # Docker operations
            'docker build', 'docker run', 'docker start', 'docker stop',

            # Process control (non-critical)
            'kill', 'killall', 'pkill',
        }

        # RED tier: Destructive/dangerous commands
        self.red_commands: Set[str] = {
            # Data deletion
            'rm', 'rm -f', 'rm -rf', 'rmdir', 'unlink', 'shred',

            # Disk operations
            'dd', 'fdisk', 'mkfs', 'parted', 'gparted', 'format',

            # System modifications
            'chmod', 'chown', 'chgrp', 'usermod', 'groupmod',
            'systemctl', 'service', 'launchctl',

            # Network operations with risk
            'curl | sh', 'wget | sh', 'curl | bash', 'wget | bash',
            'scp', 'sftp', 'rsync --delete',

            # Git destructive
            'git push --force', 'git push -f', 'git reset --hard',
            'git clean -fd', 'git rebase -i',

            # Database operations
            'DROP TABLE', 'DROP DATABASE', 'TRUNCATE', 'DELETE FROM',
            'ALTER TABLE', 'UPDATE', 'mysql', 'psql',

            # System commands
            'sudo', 'su', 'doas', 'shutdown', 'reboot', 'init',
            'halt', 'poweroff',

            # Package removal
            'npm uninstall', 'pip uninstall', 'brew uninstall',
            'apt remove', 'apt purge', 'yum remove',

            # Docker destructive
            'docker rm', 'docker rmi', 'docker system prune',
        }

        # Destructive patterns (regex)
        self.destructive_patterns: List[Tuple[re.Pattern, str]] = [
            (re.compile(r'\brm\s+-[rf]+'), "rm with -rf flags"),
            (re.compile(r'\|.*\b(sh|bash|zsh|fish)\b'), "pipe to shell"),
            (re.compile(r'>\s*/dev/(sd[a-z]|disk\d+)'), "write to disk device"),
            (re.compile(r'sudo\s+rm'), "sudo rm"),
            (re.compile(r'dd\s+.*of='), "dd output to file/device"),
            (re.compile(r'mkfs\.\w+'), "filesystem creation"),
            (re.compile(r'chmod\s+777'), "chmod 777"),
            (re.compile(r'--force\b'), "force flag"),
            (re.compile(r'DROP\s+(TABLE|DATABASE)\b', re.IGNORECASE), "SQL DROP"),
            (re.compile(r'DELETE\s+FROM\b', re.IGNORECASE), "SQL DELETE"),
            (re.compile(r'TRUNCATE\b', re.IGNORECASE), "SQL TRUNCATE"),
            (re.compile(r'git\s+push.*--force'), "git force push"),
            (re.compile(r'git\s+reset\s+--hard'), "git hard reset"),
            (re.compile(r'npm\s+install\s+-g'), "npm global install"),
            (re.compile(r':\(\)\{.*:\|:&\};:'), "fork bomb pattern"),
        ]

        # Commands with dry-run support
        self.dry_run_supported: Dict[str, str] = {
            'rm': 'rm -i',  # Interactive mode
            'rsync': 'rsync --dry-run',
            'apt': 'apt --dry-run',
            'npm': 'npm --dry-run',
            'pip': 'pip install --dry-run',
            'ansible-playbook': 'ansible-playbook --check',
            'terraform': 'terraform plan',
        }

        # Reversible operations (have undo mechanisms)
        self.reversible_commands: Set[str] = {
            'git add', 'git commit', 'git stash', 'git checkout',
            'mv', 'cp', 'mkdir', 'touch',
            'npm install', 'pip install',
        }

        logger.info(
            f"[COMMAND-SAFETY] Initialized with "
            f"{len(self.green_commands)} green, "
            f"{len(self.yellow_commands)} yellow, "
            f"{len(self.red_commands)} red commands"
        )

    def classify(self, command: str) -> CommandClassification:
        """
        Classify a command by safety tier.

        Args:
            command: Shell command to classify

        Returns:
            CommandClassification with tier, risks, and recommendations
        """
        command = command.strip()
        if not command:
            return CommandClassification(
                command="",
                tier=SafetyTier.UNKNOWN,
                risk_categories=[],
                requires_confirmation=True,
                is_reversible=False,
                confidence=1.0,
                reasoning="Empty command",
            )

        # Parse command to extract base command
        base_cmd = self._extract_base_command(command)
        full_cmd = self._extract_full_command(command)

        # Check for destructive patterns first
        destructive_match = self._check_destructive_patterns(command)
        if destructive_match:
            return CommandClassification(
                command=command,
                tier=SafetyTier.RED,
                risk_categories=[RiskCategory.DATA_LOSS],
                requires_confirmation=True,
                is_reversible=False,
                confidence=0.95,
                reasoning=f"Destructive pattern detected: {destructive_match}",
                dry_run_available=self._supports_dry_run(base_cmd),
            )

        # Check tier classifications
        if self._matches_command_set(full_cmd, self.green_commands):
            return CommandClassification(
                command=command,
                tier=SafetyTier.GREEN,
                risk_categories=[RiskCategory.SAFE_READ],
                requires_confirmation=False,
                is_reversible=True,
                confidence=0.9,
                reasoning="Safe read-only command",
            )

        if self._matches_command_set(full_cmd, self.red_commands):
            risk_cats = self._determine_risk_categories(command, base_cmd)
            return CommandClassification(
                command=command,
                tier=SafetyTier.RED,
                risk_categories=risk_cats,
                requires_confirmation=True,
                is_reversible=False,
                confidence=0.9,
                reasoning="Dangerous command requiring confirmation",
                suggested_alternative=self._suggest_safer_alternative(command),
                dry_run_available=self._supports_dry_run(base_cmd),
            )

        if self._matches_command_set(full_cmd, self.yellow_commands):
            is_reversible = full_cmd in self.reversible_commands
            risk_cats = self._determine_risk_categories(command, base_cmd)
            return CommandClassification(
                command=command,
                tier=SafetyTier.YELLOW,
                risk_categories=risk_cats,
                requires_confirmation=True,
                is_reversible=is_reversible,
                confidence=0.85,
                reasoning="Modifies state, requires user confirmation",
                dry_run_available=self._supports_dry_run(base_cmd),
            )

        # Unknown command - default to YELLOW (cautious)
        return CommandClassification(
            command=command,
            tier=SafetyTier.YELLOW,
            risk_categories=[RiskCategory.SYSTEM_MODIFICATION],
            requires_confirmation=True,
            is_reversible=False,
            confidence=0.5,
            reasoning="Unknown command, defaulting to caution",
        )

    def classify_batch(self, commands: List[str]) -> List[CommandClassification]:
        """
        Classify multiple commands.

        Args:
            commands: List of commands to classify

        Returns:
            List of classifications
        """
        return [self.classify(cmd) for cmd in commands]

    def _extract_base_command(self, command: str) -> str:
        """Extract base command name (e.g., 'git' from 'git push')."""
        try:
            # Split by pipes, semicolons, etc. and take first command
            first_cmd = re.split(r'[|;&]', command)[0].strip()

            # Parse with shlex to handle quotes
            tokens = shlex.split(first_cmd)
            if not tokens:
                return ""

            # Skip environment variables (VAR=value cmd)
            for token in tokens:
                if '=' not in token or token.startswith('-'):
                    return token.split('/')[-1]  # Handle /usr/bin/cmd

            return tokens[0].split('/')[-1]

        except Exception as e:
            logger.warning(f"[COMMAND-SAFETY] Failed to parse command '{command}': {e}")
            return command.split()[0] if command.split() else ""

    def _extract_full_command(self, command: str) -> str:
        """Extract full command with subcommands (e.g., 'git push' from 'git push origin')."""
        try:
            tokens = shlex.split(command.split('|')[0].split(';')[0].strip())
            if not tokens:
                return ""

            # For commands like 'git push', return both parts
            base = self._extract_base_command(command)

            # Common multi-part commands
            if base in ['git', 'docker', 'npm', 'pip', 'brew', 'apt', 'yum', 'cargo', 'go']:
                # Find first non-flag token after base command
                for i, token in enumerate(tokens):
                    if token == base and i + 1 < len(tokens):
                        next_token = tokens[i + 1]
                        if not next_token.startswith('-'):
                            return f"{base} {next_token}"

            return base

        except Exception:
            return self._extract_base_command(command)

    def _matches_command_set(self, full_cmd: str, command_set: Set[str]) -> bool:
        """Check if command matches any command in set."""
        # Exact match
        if full_cmd in command_set:
            return True

        # Check if any command in set starts with our full_cmd
        for known_cmd in command_set:
            if known_cmd.startswith(full_cmd):
                return True

        return False

    def _check_destructive_patterns(self, command: str) -> Optional[str]:
        """Check if command matches destructive patterns."""
        for pattern, description in self.destructive_patterns:
            if pattern.search(command):
                return description
        return None

    def _determine_risk_categories(self, command: str, base_cmd: str) -> List[RiskCategory]:
        """Determine risk categories for command."""
        risks = []

        # Data loss risks
        if any(word in command for word in ['rm', 'dd', 'shred', 'DELETE', 'DROP', 'TRUNCATE']):
            risks.append(RiskCategory.DATA_LOSS)

        # System modification
        if any(word in command for word in ['chmod', 'chown', 'sudo', 'systemctl']):
            risks.append(RiskCategory.SYSTEM_MODIFICATION)

        # Network exposure
        if re.search(r'\|\s*(sh|bash)', command) and any(cmd in command for cmd in ['curl', 'wget']):
            risks.append(RiskCategory.NETWORK_EXPOSURE)

        # Process control
        if base_cmd in ['kill', 'killall', 'pkill']:
            risks.append(RiskCategory.PROCESS_CONTROL)

        # File modification
        if base_cmd in ['mv', 'cp', 'touch', 'mkdir']:
            risks.append(RiskCategory.FILE_MODIFICATION)

        # Package management
        if 'install' in command or 'uninstall' in command or 'upgrade' in command:
            risks.append(RiskCategory.PACKAGE_MANAGEMENT)

        # Version control
        if command.startswith('git'):
            risks.append(RiskCategory.VERSION_CONTROL)

        # Database
        if any(op in command.upper() for op in ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'UPDATE']):
            risks.append(RiskCategory.DATABASE_OPERATION)

        # Default to safe read if no risks identified
        if not risks:
            risks.append(RiskCategory.SAFE_READ)

        return risks

    def _supports_dry_run(self, base_cmd: str) -> bool:
        """Check if command supports dry-run mode."""
        return base_cmd in self.dry_run_supported

    def _suggest_safer_alternative(self, command: str) -> Optional[str]:
        """Suggest safer alternative for dangerous command."""
        base_cmd = self._extract_base_command(command)

        # Suggest dry-run if available
        if base_cmd in self.dry_run_supported:
            return self.dry_run_supported[base_cmd]

        # Specific suggestions
        if 'rm -rf' in command:
            return command.replace('rm -rf', 'rm -i')

        if 'git push --force' in command:
            return command.replace('--force', '--force-with-lease')

        if 'chmod 777' in command:
            return command.replace('777', '755')

        return None

    def add_custom_rule(
        self,
        command_pattern: str,
        tier: SafetyTier,
        is_reversible: bool = False,
    ) -> None:
        """
        Add custom safety rule (for user-specific workflows).

        Args:
            command_pattern: Command or pattern to classify
            tier: Safety tier to assign
            is_reversible: Whether operation can be undone
        """
        if tier == SafetyTier.GREEN:
            self.green_commands.add(command_pattern)
        elif tier == SafetyTier.YELLOW:
            self.yellow_commands.add(command_pattern)
        elif tier == SafetyTier.RED:
            self.red_commands.add(command_pattern)

        if is_reversible:
            self.reversible_commands.add(command_pattern)

        logger.info(f"[COMMAND-SAFETY] Added custom rule: '{command_pattern}' -> {tier.value}")


# Global instance
_global_classifier: Optional[CommandSafetyClassifier] = None


def get_command_classifier() -> CommandSafetyClassifier:
    """Get or create global command safety classifier."""
    global _global_classifier

    if _global_classifier is None:
        _global_classifier = CommandSafetyClassifier()

    return _global_classifier
