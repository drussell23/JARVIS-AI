"""
Centralized Secret Management for JARVIS
Supports multiple backends with automatic fallback:
  1. GCP Secret Manager (production)
  2. macOS Keychain (local development)
  3. Environment variables (CI/CD fallback)
"""

from google.cloud import secretmanager
from functools import lru_cache
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SecretManager:
    """Centralized secret management for JARVIS with multi-backend support"""

    def __init__(self, project_id: str = None):
        """
        Initialize SecretManager

        Args:
            project_id: GCP project ID (defaults to env var or jarvis-473803)
        """
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID", "jarvis-473803")

        # Try to initialize GCP client (may fail in local dev without credentials)
        self.gcp_client = None
        try:
            self.gcp_client = secretmanager.SecretManagerServiceClient()
            logger.info("✅ GCP Secret Manager client initialized")
        except Exception as e:
            logger.debug(f"GCP Secret Manager not available: {e}")
            logger.info("⚠️  Using fallback secret sources (Keychain/Environment)")

    @lru_cache(maxsize=32)
    def get_secret(self, secret_id: str, version: str = "latest") -> Optional[str]:
        """
        Retrieve secret from available backend with automatic fallback chain:
        1. GCP Secret Manager (production)
        2. macOS Keychain (local development)
        3. Environment variable (CI/CD)

        Args:
            secret_id: Secret identifier (e.g., 'anthropic-api-key')
            version: Secret version (default: 'latest')

        Returns:
            Secret value or None if not found
        """
        # Try GCP Secret Manager first
        if self.gcp_client:
            try:
                name = f"projects/{self.project_id}/secrets/{secret_id}/versions/{version}"
                response = self.gcp_client.access_secret_version(request={"name": name})
                secret_value = response.payload.data.decode("UTF-8")
                logger.info(f"✅ Retrieved '{secret_id}' from GCP Secret Manager")
                return secret_value
            except Exception as e:
                logger.debug(f"Failed to get '{secret_id}' from GCP: {e}")

        # Try macOS Keychain (local development)
        try:
            import keyring
            keychain_value = keyring.get_password("JARVIS", secret_id)
            if keychain_value:
                logger.info(f"✅ Retrieved '{secret_id}' from macOS Keychain")
                return keychain_value
        except ImportError:
            logger.debug("keyring module not available")
        except Exception as e:
            logger.debug(f"Failed to get '{secret_id}' from Keychain: {e}")

        # Final fallback: environment variable
        env_var = secret_id.upper().replace("-", "_")
        env_value = os.getenv(env_var)
        if env_value:
            logger.info(f"✅ Retrieved '{secret_id}' from environment variable")
            return env_value

        # Not found anywhere
        logger.error(f"❌ Secret '{secret_id}' not found in any backend")
        return None

    def get_anthropic_key(self) -> Optional[str]:
        """Get Anthropic API key"""
        return self.get_secret("anthropic-api-key")

    def get_db_password(self) -> Optional[str]:
        """Get database password"""
        return self.get_secret("jarvis-db-password")

    def get_picovoice_key(self) -> Optional[str]:
        """Get Picovoice access key"""
        return self.get_secret("picovoice-access-key")

    def rotate_secret(self, secret_id: str, new_value: str) -> bool:
        """
        Add new secret version (rotation)

        Args:
            secret_id: Secret identifier
            new_value: New secret value

        Returns:
            True if successful, False otherwise
        """
        if not self.gcp_client:
            logger.error("❌ GCP Secret Manager not available for rotation")
            return False

        try:
            parent = f"projects/{self.project_id}/secrets/{secret_id}"
            payload = {"data": new_value.encode("UTF-8")}

            self.gcp_client.add_secret_version(
                request={"parent": parent, "payload": payload}
            )

            # Clear cache to force fresh retrieval
            self.get_secret.cache_clear()
            logger.info(f"✅ Rotated secret '{secret_id}'")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to rotate secret '{secret_id}': {e}")
            return False

    def list_secrets(self) -> list:
        """List all available secrets"""
        if not self.gcp_client:
            logger.warning("GCP Secret Manager not available")
            return []

        try:
            parent = f"projects/{self.project_id}"
            secrets = self.gcp_client.list_secrets(request={"parent": parent})
            secret_names = [secret.name.split("/")[-1] for secret in secrets]
            logger.info(f"Found {len(secret_names)} secrets in GCP Secret Manager")
            return secret_names
        except Exception as e:
            logger.error(f"Failed to list secrets: {e}")
            return []


# Global singleton instance
_secret_manager = None


def get_secret_manager() -> SecretManager:
    """
    Get or create global SecretManager instance (singleton pattern)

    Returns:
        SecretManager instance
    """
    global _secret_manager
    if _secret_manager is None:
        _secret_manager = SecretManager()
    return _secret_manager


# Convenience functions for direct access
def get_anthropic_key() -> Optional[str]:
    """Quick access to Anthropic API key"""
    return get_secret_manager().get_anthropic_key()


def get_db_password() -> Optional[str]:
    """Quick access to database password"""
    return get_secret_manager().get_db_password()


def get_picovoice_key() -> Optional[str]:
    """Quick access to Picovoice access key"""
    return get_secret_manager().get_picovoice_key()


if __name__ == "__main__":
    # Test script
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("🔐 JARVIS Secret Manager Test\n")

    mgr = get_secret_manager()

    # List available secrets
    print("📋 Available secrets:")
    secrets = mgr.list_secrets()
    for secret in secrets:
        print(f"  - {secret}")

    print("\n🔍 Testing secret retrieval:\n")

    # Test Anthropic key
    anthropic_key = mgr.get_anthropic_key()
    if anthropic_key:
        print(f"✅ Anthropic API Key: {anthropic_key[:20]}...{anthropic_key[-10:]}")
    else:
        print("❌ Anthropic API Key: Not found")

    # Test DB password
    db_password = mgr.get_db_password()
    if db_password:
        print(f"✅ DB Password: {'*' * len(db_password)} (hidden)")
    else:
        print("❌ DB Password: Not found")

    # Test Picovoice key
    picovoice_key = mgr.get_picovoice_key()
    if picovoice_key:
        print(f"✅ Picovoice Key: {picovoice_key[:15]}...{picovoice_key[-5:]}")
    else:
        print("❌ Picovoice Key: Not found")

    print("\n✅ Secret Manager test complete!")
