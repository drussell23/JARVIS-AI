#!/usr/bin/env python3
"""
Check for common configuration issues in JARVIS codebase.

This script catches common mistakes that cause production issues:
1. Using os.environ instead of subprocess env dict in start_system.py
2. Missing environment variables in subprocess env dict
3. Hardcoded paths that should be configurable
4. Missing error handling for critical operations
"""

import re
import sys
from pathlib import Path


def check_subprocess_env_propagation(repo_root: Path) -> list:
    """
    Critical check: Ensure env vars are passed to subprocess env dict.

    This catches the exact issue we just fixed where JARVIS_DB_HOST
    was set in os.environ but not in the subprocess env dict.
    """
    errors = []

    start_system_path = repo_root / "start_system.py"

    if not start_system_path.exists():
        errors.append("start_system.py not found")
        return errors

    content = start_system_path.read_text()

    # Find all os.environ assignments for JARVIS_DB_* variables
    os_environ_assignments = re.findall(r'os\.environ\["(JARVIS_DB_\w+)"\]\s*=', content)

    # Find all env dict assignments for JARVIS_DB_* variables
    env_dict_assignments = re.findall(r'env\["(JARVIS_DB_\w+)"\]\s*=', content)

    # Check if any vars are set in os.environ but not in env dict
    os_environ_set = set(os_environ_assignments)
    env_dict_set = set(env_dict_assignments)

    missing_in_subprocess = os_environ_set - env_dict_set

    if missing_in_subprocess:
        errors.append("⚠️  CRITICAL: Variables set in os.environ but NOT in subprocess env dict:")
        for var in sorted(missing_in_subprocess):
            errors.append(f"  - {var}")
        errors.append("\n  This causes subprocess to NOT see these variables!")
        errors.append("  Fix: Add env['VAR'] = os.environ['VAR'] for each variable")

    return errors


def check_critical_env_vars_presence(repo_root: Path) -> list:
    """Check that critical env vars are set in subprocess env dict"""
    errors = []

    start_system_path = repo_root / "start_system.py"
    content = start_system_path.read_text()

    # Critical env vars that MUST be in subprocess env dict
    critical_vars = {
        "JARVIS_DB_HOST": "127.0.0.1",  # Must be localhost for proxy
        "JARVIS_DB_PORT": None,  # Must be set (port number)
        "JARVIS_DB_TYPE": "cloudsql",  # Should be cloudsql
        "JARVIS_DB_CONNECTION_NAME": None,  # Must be set
    }

    for var_name, expected_value in critical_vars.items():
        # Check if var is in env dict
        if f'env["{var_name}"]' not in content:
            errors.append(f"❌ Missing critical env var in subprocess: {var_name}")
            if expected_value:
                errors.append(f"   Should be: env['{var_name}'] = '{expected_value}'")

        # If expected value is specified, verify it's correct
        elif expected_value:
            pattern = rf'env\["{var_name}"\]\s*=\s*["\']({expected_value})["\']'
            if not re.search(pattern, content):
                errors.append(f"⚠️  {var_name} may not be set to '{expected_value}'")

    return errors


def check_hardcoded_paths(repo_root: Path) -> list:
    """Check for hardcoded paths that should be configurable"""
    warnings = []

    files_to_check = [
        repo_root / "start_system.py",
        repo_root / "backend" / "intelligence" / "learning_database.py",
        repo_root / "backend" / "voice" / "speaker_verification_service.py",
    ]

    hardcoded_patterns = [
        (r"/Users/[^/]+/", "Hardcoded user path"),
        (r"C:\\Users\\[^\\]+\\", "Hardcoded Windows user path"),
        (r"/tmp/(?!cloud-sql-proxy)", "Hardcoded /tmp path (use tempfile module)"),
    ]

    for file_path in files_to_check:
        if not file_path.exists():
            continue

        content = file_path.read_text()
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            # Skip comments
            if line.strip().startswith("#"):
                continue

            for pattern, description in hardcoded_patterns:
                if re.search(pattern, line):
                    rel_path = file_path.relative_to(repo_root)
                    warnings.append(f"{rel_path}:{line_num} - {description}")

    return warnings


def check_error_handling(repo_root: Path) -> list:
    """Check for critical operations without error handling"""
    warnings = []

    start_system_path = repo_root / "start_system.py"
    content = start_system_path.read_text()

    # Check for database operations without try/except
    db_operations = [
        "db_config = json.load",
        "learning_db.initialize()",
        "proxy_manager.start()",
    ]

    for operation in db_operations:
        if operation in content:
            # Check if it's in a try block (simple heuristic)
            # Find all try/except blocks
            try_blocks = re.finditer(r"try:\s*\n(.*?)\nexcept", content, re.DOTALL)

            in_try_block = False
            for match in try_blocks:
                if operation in match.group(1):
                    in_try_block = True
                    break

            if not in_try_block:
                warnings.append(f"⚠️  Operation '{operation}' may not have error handling")

    return warnings


def check_proxy_configuration(repo_root: Path) -> list:
    """Check Cloud SQL proxy configuration"""
    errors = []

    start_system_path = repo_root / "start_system.py"
    content = start_system_path.read_text()

    # Check that proxy is started before backend
    proxy_start_pattern = r"proxy_manager\.start\("
    backend_start_pattern = r"backend_process = subprocess\.Popen"

    proxy_match = re.search(proxy_start_pattern, content)
    backend_match = re.search(backend_start_pattern, content)

    if proxy_match and backend_match:
        # Check order (proxy should come before backend)
        if proxy_match.start() > backend_match.start():
            errors.append("❌ Cloud SQL proxy started AFTER backend process")
            errors.append("   Proxy must start before backend to ensure database connection")

    # Check for health monitoring
    if "monitor(check_interval" not in content:
        errors.append("⚠️  Cloud SQL proxy health monitoring not found")
        errors.append("   Add: asyncio.create_task(proxy_manager.monitor(check_interval=60))")

    return errors


def main():
    """Main validation entry point"""
    repo_root = Path(__file__).parent.parent.parent

    print("🔍 Checking for common configuration issues...")

    all_errors = []
    all_warnings = []

    # Run checks (order matters - most critical first)
    all_errors.extend(check_subprocess_env_propagation(repo_root))
    all_errors.extend(check_critical_env_vars_presence(repo_root))
    all_errors.extend(check_proxy_configuration(repo_root))
    all_warnings.extend(check_hardcoded_paths(repo_root))
    all_warnings.extend(check_error_handling(repo_root))

    # Print results
    if all_errors:
        print("\n❌ VALIDATION FAILED\n")
        for error in all_errors:
            print(f"  {error}")

    if all_warnings:
        print("\n⚠️  WARNINGS\n")
        for warning in all_warnings:
            print(f"  {warning}")

    if not all_errors and not all_warnings:
        print("\n✅ No common configuration issues found!")
        return 0

    return 1 if all_errors else 0


if __name__ == "__main__":
    sys.exit(main())
