#!/usr/bin/env python3
"""
Validate voice biometric system setup.

Ensures that:
1. Speaker verification service uses fast initialization
2. Speaker encoder is pre-loaded during startup
3. Voice profiles load from Cloud SQL (not SQLite fallback)
4. No blocking initialization in get_speaker_verification_service()
"""

import sys
from pathlib import Path


def check_fast_initialization(repo_root: Path) -> list:
    """Check that speaker verification uses fast initialization"""
    errors = []

    speaker_service_path = repo_root / "backend" / "voice" / "speaker_verification_service.py"

    if not speaker_service_path.exists():
        errors.append("Speaker verification service file not found")
        return errors

    content = speaker_service_path.read_text()

    # Check for initialize_fast() method
    if "async def initialize_fast(self):" not in content:
        errors.append("initialize_fast() method not found in SpeakerVerificationService")
        errors.append("This method is required for background encoder pre-loading")

    # Check that get_speaker_verification_service() uses initialize_fast()
    if "await _speaker_verification_service.initialize_fast()" not in content:
        # Check if it's using blocking initialize() instead
        if "await _speaker_verification_service.initialize()" in content:
            errors.append("get_speaker_verification_service() uses blocking initialize()")
            errors.append("This causes 60+ second delay on first unlock")
            errors.append("Change to: await _speaker_verification_service.initialize_fast()")

    # Check for background pre-loading thread
    if "_start_background_preload" not in content:
        errors.append("Background pre-loading mechanism not found")
        errors.append("Speaker encoder should pre-load in background thread")

    return errors


def check_cloud_sql_usage(repo_root: Path) -> list:
    """Check that voice profiles load from Cloud SQL"""
    warnings = []

    learning_db_path = repo_root / "backend" / "intelligence" / "learning_database.py"

    if not learning_db_path.exists():
        warnings.append("learning_database.py not found")
        return warnings

    content = learning_db_path.read_text()

    # Check for Cloud SQL connection
    if "cloud_sql" not in content.lower() and "cloudsql" not in content.lower():
        warnings.append("learning_database.py may not support Cloud SQL connections")

    # Check for SQLite fallback
    if "sqlite" in content.lower():
        # This is OK as long as it's a fallback, not the primary
        if "fallback" not in content.lower():
            warnings.append("SQLite should only be used as fallback, not primary database")

    return warnings


def check_pre_loading_in_startup(repo_root: Path) -> list:
    """Check that start_system.py pre-loads speaker service"""
    warnings = []

    start_system_path = repo_root / "start_system.py"

    if not start_system_path.exists():
        warnings.append("start_system.py not found")
        return warnings

    content = start_system_path.read_text()

    # Check for speaker service pre-loading
    if "speaker_verification_service" not in content.lower():
        warnings.append("start_system.py may not pre-load speaker verification service")
        warnings.append("Pre-loading during startup ensures instant unlock")

    return warnings


def main():
    """Main validation entry point"""
    repo_root = Path(__file__).parent.parent.parent

    print("🔍 Validating voice biometric setup...")

    all_errors = []
    all_warnings = []

    # Run checks
    all_errors.extend(check_fast_initialization(repo_root))
    all_warnings.extend(check_cloud_sql_usage(repo_root))
    all_warnings.extend(check_pre_loading_in_startup(repo_root))

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
        print("\n✅ All voice biometric checks passed!")
        return 0

    return 1 if all_errors else 0


if __name__ == "__main__":
    sys.exit(main())
