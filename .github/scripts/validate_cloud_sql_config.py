#!/usr/bin/env python3
"""
Validate Cloud SQL configuration in codebase.

Ensures that:
1. No hardcoded IP addresses are used
2. All connections go through proxy (127.0.0.1)
3. Environment variables are used consistently
"""

import re
import sys
from pathlib import Path


def check_hardcoded_ips(repo_root: Path) -> list:
    """Check for hardcoded Cloud SQL IP addresses"""
    errors = []
    backend_dir = repo_root / "backend"

    # Known Cloud SQL private IP pattern
    cloud_sql_ip_pattern = r"(?:34\.46\.152\.27|10\.\d+\.\d+\.\d+):\d+"

    # Exclude virtual environment directories
    python_files = [
        f
        for f in backend_dir.glob("**/*.py")
        if "venv" not in f.parts and "site-packages" not in f.parts
    ]
    python_files.append(repo_root / "start_system.py")

    for file_path in python_files:
        try:
            content = file_path.read_text()
            matches = re.findall(cloud_sql_ip_pattern, content)

            if matches:
                rel_path = file_path.relative_to(repo_root)
                errors.append(
                    f"Hardcoded Cloud SQL IP found in {rel_path}: {', '.join(set(matches))}"
                )
                errors.append(f"  → Use JARVIS_DB_HOST environment variable instead")

        except Exception:
            continue

    return errors


def check_proxy_usage(repo_root: Path) -> list:
    """Ensure Cloud SQL connections use localhost proxy"""
    warnings = []
    backend_dir = repo_root / "backend"

    # Look for database connection patterns
    connection_files = [
        backend_dir / "intelligence" / "cloud_database_adapter.py",
        backend_dir / "intelligence" / "learning_database.py",
    ]

    for file_path in connection_files:
        if not file_path.exists():
            continue

        content = file_path.read_text()

        # Check if it uses JARVIS_DB_HOST env var
        if "JARVIS_DB_HOST" not in content:
            rel_path = file_path.relative_to(repo_root)
            warnings.append(f"{rel_path} may not be using JARVIS_DB_HOST environment variable")

        # Check for direct connections (not through proxy)
        if re.search(r'host\s*=\s*["\'](?!127\.0\.0\.1|localhost)', content):
            rel_path = file_path.relative_to(repo_root)
            warnings.append(f"{rel_path} may be connecting directly instead of through proxy")

    return warnings


def check_env_var_consistency(repo_root: Path) -> list:
    """Check that Cloud SQL env vars are used consistently"""
    errors = []

    # Required env vars
    required_vars = [
        "JARVIS_DB_TYPE",
        "JARVIS_DB_CONNECTION_NAME",
        "JARVIS_DB_HOST",
        "JARVIS_DB_PORT",
        "JARVIS_DB_PASSWORD",
    ]

    start_system = repo_root / "start_system.py"
    content = start_system.read_text()

    missing_vars = []
    for var in required_vars:
        # Check if var is set in env dict
        if f'env["{var}"]' not in content:
            missing_vars.append(var)

    if missing_vars:
        errors.append(f"start_system.py missing env var assignments: {', '.join(missing_vars)}")
        errors.append("These must be set in the 'env' dict before starting backend subprocess")

    return errors


def main():
    """Main validation entry point"""
    repo_root = Path(__file__).parent.parent.parent

    print("🔍 Validating Cloud SQL configuration...")

    all_errors = []
    all_warnings = []

    # Run checks
    all_errors.extend(check_hardcoded_ips(repo_root))
    all_warnings.extend(check_proxy_usage(repo_root))
    all_errors.extend(check_env_var_consistency(repo_root))

    # Print results
    if all_errors:
        print("\n❌ VALIDATION FAILED\n")
        for error in all_errors:
            print(f"  ❌ {error}")

    if all_warnings:
        print("\n⚠️  WARNINGS\n")
        for warning in all_warnings:
            print(f"  ⚠️  {warning}")

    if not all_errors and not all_warnings:
        print("\n✅ All Cloud SQL configuration checks passed!")
        return 0

    return 1 if all_errors else 0


if __name__ == "__main__":
    sys.exit(main())
