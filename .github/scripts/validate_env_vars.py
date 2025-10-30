#!/usr/bin/env python3
"""
Validate environment variable usage across the codebase.

Ensures that:
1. Environment variables are accessed via os.getenv() with defaults
2. Required env vars are documented
3. No env vars are accessed without proper error handling
"""

import re
import sys
from pathlib import Path


def check_env_var_access(repo_root: Path) -> list:
    """Check that environment variables are accessed safely"""
    errors = []
    warnings = []

    backend_dir = repo_root / "backend"
    python_files = list(backend_dir.glob("**/*.py"))
    python_files.append(repo_root / "start_system.py")

    # Pattern for unsafe env var access (os.environ["VAR"] without try/except)
    unsafe_pattern = r'os\.environ\["([^"]+)"\]'

    for file_path in python_files:
        try:
            content = file_path.read_text()
            lines = content.split("\n")

            for line_num, line in enumerate(lines, 1):
                # Check for unsafe access
                matches = re.findall(unsafe_pattern, line)

                for var_name in matches:
                    # Skip if it's an assignment (setting the var)
                    if re.search(rf'os\.environ\["{var_name}"\]\s*=', line):
                        continue

                    # Check if it's inside a try/except block (simple check)
                    # Look back a few lines for 'try:'
                    try_found = False
                    for i in range(max(0, line_num - 10), line_num):
                        if "try:" in lines[i]:
                            try_found = True
                            break

                    if not try_found:
                        rel_path = file_path.relative_to(repo_root)
                        warnings.append(
                            f"{rel_path}:{line_num} - Unsafe env var access: {var_name}"
                        )
                        warnings.append(f"  → Consider using: os.getenv('{var_name}', 'default')")

        except Exception:
            continue

    return errors, warnings


def check_required_env_vars_documented(repo_root: Path) -> list:
    """Check that required env vars are documented"""
    errors = []

    # Known required env vars
    required_vars = [
        "JARVIS_DB_TYPE",
        "JARVIS_DB_CONNECTION_NAME",
        "JARVIS_DB_HOST",
        "JARVIS_DB_PORT",
        "JARVIS_DB_PASSWORD",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GCP_PROJECT_ID",
        "GCP_VM_ENABLED",
    ]

    # Check if there's a .env.example or similar documentation
    docs_files = [
        repo_root / ".env.example",
        repo_root / "ENV_VARS.md",
        repo_root / "README.md",
    ]

    documented_vars = set()

    for doc_file in docs_files:
        if doc_file.exists():
            content = doc_file.read_text()
            for var in required_vars:
                if var in content:
                    documented_vars.add(var)

    undocumented = set(required_vars) - documented_vars

    if undocumented:
        errors.append(f"Required env vars not documented: {', '.join(sorted(undocumented))}")
        errors.append("Create .env.example or ENV_VARS.md to document these variables")

    return errors


def main():
    """Main validation entry point"""
    repo_root = Path(__file__).parent.parent.parent

    print("🔍 Validating environment variable usage...")

    all_errors = []
    all_warnings = []

    # Run checks
    errors, warnings = check_env_var_access(repo_root)
    all_errors.extend(errors)
    all_warnings.extend(warnings)

    all_errors.extend(check_required_env_vars_documented(repo_root))

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
        print("\n✅ All environment variable checks passed!")
        return 0

    return 1 if all_errors else 0


if __name__ == "__main__":
    sys.exit(main())
