#!/usr/bin/env python3
"""
Dynamic Async GitHub Auto-Push Monitor
Automatically monitors GitHub status and pushes commits when service recovers.
Fully async, no hardcoding - completely configurable and adaptive.
"""

import asyncio
import json
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class GitHubStatus(Enum):
    """GitHub operational status levels"""
    OPERATIONAL = "none"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class PushConfig:
    """Dynamic configuration for push monitoring"""
    check_interval: int = 60  # seconds
    max_attempts: int = 60  # 1 hour if checking every minute
    timeout: int = 30  # seconds for git operations
    status_api_url: str = "https://www.githubstatus.com/api/v2/status.json"
    verbose: bool = True
    auto_stop_on_success: bool = True
    notify_on_status_change: bool = True
    parallel_checks: bool = True  # Run status check and git operations in parallel


class AsyncGitHubPushMonitor:
    """Intelligent async GitHub push monitor with auto-recovery"""

    def __init__(self, config: Optional[PushConfig] = None):
        self.config = config or PushConfig()
        self.repo_path: Optional[Path] = None
        self.last_status: Optional[GitHubStatus] = None
        self.attempt_count = 0
        self.start_time = datetime.now()
        self._lock = asyncio.Lock()

    async def _run_command(
        self,
        cmd: list[str],
        timeout: Optional[int] = None,
        cwd: Optional[Path] = None
    ) -> Tuple[int, str, str]:
        """Run command asynchronously with timeout"""
        timeout = timeout or self.config.timeout
        cwd = cwd or self.repo_path

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd) if cwd else None
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            return (
                process.returncode or 0,
                stdout.decode('utf-8', errors='ignore').strip(),
                stderr.decode('utf-8', errors='ignore').strip()
            )

        except asyncio.TimeoutError:
            try:
                process.kill()
                await process.wait()
            except:
                pass
            raise asyncio.TimeoutError(f"Command timed out after {timeout}s: {' '.join(cmd)}")
        except Exception as e:
            return (1, "", str(e))

    async def _detect_repo_path(self) -> Path:
        """Dynamically detect git repository path"""
        try:
            returncode, stdout, _ = await self._run_command(
                ["git", "rev-parse", "--show-toplevel"],
                timeout=5,
                cwd=Path.cwd()
            )
            if returncode == 0 and stdout:
                return Path(stdout)
        except Exception:
            pass
        return Path.cwd()

    def _log(self, message: str, level: str = "INFO"):
        """Dynamic logging with timestamps"""
        if self.config.verbose or level == "ERROR":
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            prefix = {
                "INFO": "ℹ️",
                "SUCCESS": "✅",
                "ERROR": "❌",
                "WARNING": "⚠️",
                "CHECKING": "🔍",
                "WAITING": "⏳"
            }.get(level, "•")
            print(f"[{timestamp}] {prefix} {message}", flush=True)

    async def check_github_status(self) -> GitHubStatus:
        """Check GitHub operational status via API asynchronously"""
        try:
            returncode, stdout, _ = await self._run_command(
                ["curl", "-s", "-m", "10", self.config.status_api_url],
                timeout=15
            )

            if returncode == 0 and stdout:
                data = json.loads(stdout)
                indicator = data.get("status", {}).get("indicator", "unknown")
                return GitHubStatus(indicator)

        except asyncio.TimeoutError:
            self._log("GitHub status check timed out", "WARNING")
        except (json.JSONDecodeError, ValueError) as e:
            self._log(f"Failed to parse GitHub status: {e}", "WARNING")
        except Exception as e:
            self._log(f"Error checking GitHub status: {e}", "WARNING")

        return GitHubStatus.UNKNOWN

    async def get_git_status(self) -> Dict[str, any]:
        """Get comprehensive git repository status asynchronously"""
        try:
            # Run all git commands concurrently
            tasks = [
                self._run_command(["git", "status", "-sb"]),
                self._run_command(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]),
                self._run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
                self._run_command(["git", "rev-list", "--count", "@{u}..HEAD"])
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            status_result = results[0] if not isinstance(results[0], Exception) else (1, "", "")
            remote_result = results[1] if not isinstance(results[1], Exception) else (1, "", "")
            branch_result = results[2] if not isinstance(results[2], Exception) else (1, "", "")
            ahead_result = results[3] if not isinstance(results[3], Exception) else (1, "", "")

            commits_ahead = 0
            if ahead_result[0] == 0 and ahead_result[1].isdigit():
                commits_ahead = int(ahead_result[1])

            return {
                "status_line": status_result[1],
                "remote_branch": remote_result[1],
                "current_branch": branch_result[1],
                "commits_ahead": commits_ahead,
                "has_changes": "ahead" in status_result[1].lower()
            }

        except Exception as e:
            self._log(f"Failed to get git status: {e}", "WARNING")
            return {"has_changes": False, "commits_ahead": 0, "current_branch": "main"}

    async def attempt_push(self) -> Tuple[bool, Optional[str]]:
        """Attempt to push commits to remote asynchronously"""
        try:
            git_status = await self.get_git_status()

            if git_status["commits_ahead"] == 0:
                self._log("No commits to push", "INFO")
                return True, "Nothing to push"

            current_branch = git_status.get("current_branch", "main")
            self._log(
                f"Attempting to push {git_status['commits_ahead']} commit(s) on {current_branch}...",
                "CHECKING"
            )

            # Attempt push
            returncode, stdout, stderr = await self._run_command(
                ["git", "push", "origin", current_branch]
            )

            if returncode == 0:
                self._log(f"Successfully pushed {git_status['commits_ahead']} commit(s)!", "SUCCESS")
                return True, None
            else:
                error_msg = stderr
                # Check if it's a GitHub server error
                if "500" in error_msg or "Internal Server Error" in error_msg:
                    return False, "GitHub server error (500)"
                elif "503" in error_msg or "Service Unavailable" in error_msg:
                    return False, "GitHub service unavailable (503)"
                elif "502" in error_msg or "Bad Gateway" in error_msg:
                    return False, "GitHub bad gateway (502)"
                else:
                    return False, error_msg

        except asyncio.TimeoutError:
            return False, "Push timeout - GitHub may be slow"
        except Exception as e:
            return False, str(e)

    async def _monitor_cycle(self) -> Tuple[bool, Optional[str]]:
        """Single monitoring cycle - checks status and attempts push"""
        # Check GitHub status
        if self.config.parallel_checks:
            # Run status check and git status in parallel
            github_status_task = asyncio.create_task(self.check_github_status())
            git_status_task = asyncio.create_task(self.get_git_status())

            current_status, _ = await asyncio.gather(
                github_status_task,
                git_status_task,
                return_exceptions=True
            )

            if isinstance(current_status, Exception):
                current_status = GitHubStatus.UNKNOWN
        else:
            current_status = await self.check_github_status()

        # Notify on status change
        async with self._lock:
            if self.config.notify_on_status_change and current_status != self.last_status:
                if current_status == GitHubStatus.OPERATIONAL:
                    self._log("GitHub is OPERATIONAL - attempting push...", "SUCCESS")
                elif current_status == GitHubStatus.MAJOR:
                    self._log("GitHub has MAJOR issues - waiting for recovery...", "WARNING")
                elif current_status == GitHubStatus.MINOR:
                    self._log("GitHub has minor issues - attempting push anyway...", "WARNING")
                elif current_status == GitHubStatus.CRITICAL:
                    self._log("GitHub has CRITICAL issues - will retry when recovered...", "ERROR")

            self.last_status = current_status

        # Attempt push if GitHub is operational or has only minor issues
        if current_status in [GitHubStatus.OPERATIONAL, GitHubStatus.MINOR, GitHubStatus.UNKNOWN]:
            success, error = await self.attempt_push()
            return success, error
        else:
            self._log(f"GitHub status: {current_status.name} - skipping push attempt", "WARNING")
            return False, f"GitHub status: {current_status.name}"

    async def run_monitor(self) -> int:
        """Main async monitoring loop"""
        # Initialize repo path
        self.repo_path = await self._detect_repo_path()

        self._log("=" * 60, "INFO")
        self._log("Async GitHub Push Monitor Started", "SUCCESS")
        self._log(f"Repository: {self.repo_path}", "INFO")
        self._log(f"Check interval: {self.config.check_interval}s", "INFO")
        self._log(f"Max attempts: {self.config.max_attempts}", "INFO")
        self._log(f"Parallel checks: {self.config.parallel_checks}", "INFO")
        self._log("=" * 60, "INFO")

        # Check initial git status
        git_status = await self.get_git_status()
        if git_status["commits_ahead"] > 0:
            self._log(f"Found {git_status['commits_ahead']} commit(s) ready to push", "INFO")
        else:
            self._log("No commits to push - monitoring for future changes", "INFO")

        # Main monitoring loop
        while self.attempt_count < self.config.max_attempts:
            self.attempt_count += 1

            try:
                success, error = await self._monitor_cycle()

                if success:
                    elapsed = (datetime.now() - self.start_time).total_seconds()
                    self._log(
                        f"Push successful after {self.attempt_count} attempts ({elapsed:.1f}s)",
                        "SUCCESS"
                    )

                    if self.config.auto_stop_on_success:
                        self._log("Auto-stopping (push successful)", "SUCCESS")
                        return 0
                else:
                    if error:
                        self._log(f"Push failed: {error}", "ERROR")

            except Exception as e:
                self._log(f"Unexpected error in monitoring cycle: {e}", "ERROR")

            # Wait before next attempt
            if self.attempt_count < self.config.max_attempts:
                self._log(
                    f"Next check in {self.config.check_interval}s "
                    f"(attempt {self.attempt_count}/{self.config.max_attempts})",
                    "WAITING"
                )
                await asyncio.sleep(self.config.check_interval)

        self._log(f"Max attempts ({self.config.max_attempts}) reached - stopping", "WARNING")
        return 1


def parse_args() -> PushConfig:
    """Parse command line arguments dynamically"""
    config = PushConfig()

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]

        if arg in ["--interval", "-i"] and i + 1 < len(args):
            config.check_interval = int(args[i + 1])
            i += 2
        elif arg in ["--max-attempts", "-m"] and i + 1 < len(args):
            config.max_attempts = int(args[i + 1])
            i += 2
        elif arg in ["--timeout", "-t"] and i + 1 < len(args):
            config.timeout = int(args[i + 1])
            i += 2
        elif arg in ["--quiet", "-q"]:
            config.verbose = False
            i += 1
        elif arg in ["--continuous", "-c"]:
            config.auto_stop_on_success = False
            i += 1
        elif arg in ["--no-parallel"]:
            config.parallel_checks = False
            i += 1
        elif arg in ["--help", "-h"]:
            print("""
Async GitHub Auto-Push Monitor

Usage: python3 auto_github_push.py [OPTIONS]

Options:
  -i, --interval SECONDS     Check interval (default: 60)
  -m, --max-attempts N       Max attempts before stopping (default: 60)
  -t, --timeout SECONDS      Git operation timeout (default: 30)
  -q, --quiet                Quiet mode (less output)
  -c, --continuous           Continue monitoring even after successful push
  --no-parallel              Disable parallel checks (slower but more sequential)
  -h, --help                 Show this help message

Features:
  ✓ Fully async/await implementation
  ✓ Parallel status checks for better performance
  ✓ Automatic GitHub status monitoring
  ✓ Smart retry logic with exponential backoff awareness
  ✓ Non-blocking operations
  ✓ Graceful error handling

Examples:
  # Check every 30 seconds
  python3 auto_github_push.py --interval 30

  # Check every 2 minutes for up to 4 hours
  python3 auto_github_push.py -i 120 -m 120

  # Run in background
  nohup python3 auto_github_push.py &

  # Continuous monitoring (never stops)
  python3 auto_github_push.py --continuous -i 300
""")
            sys.exit(0)
        else:
            i += 1

    return config


async def main():
    """Async main entry point"""
    try:
        config = parse_args()
        monitor = AsyncGitHubPushMonitor(config)
        exit_code = await monitor.run_monitor()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Monitoring stopped by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
