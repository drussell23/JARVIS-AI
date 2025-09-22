"""
macOS Keychain Service for Voice Unlock
======================================

Secure storage of voiceprints and authentication data
using native macOS Keychain with zero hardcoding.
"""

import keyring
from keyring.backends import macOS
import json
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime
import numpy as np
from pathlib import Path

from ..config import get_config
from ..core.voiceprint import Voiceprint, VoiceFeatures

logger = logging.getLogger(__name__)


class KeychainService:
    """Secure storage service using macOS Keychain"""
    
    # Dynamic service name from config
    SERVICE_NAME_PREFIX = "ai.jarvis.voice-unlock"
    
    def __init__(self):
        self.config = get_config()
        
        # Use native macOS keychain
        try:
            keyring.set_keyring(macOS.Keyring())
        except Exception as e:
            logger.warning(f"Failed to set macOS keyring, using default: {e}")
            
        # Dynamic encryption key management
        self._init_encryption()
        
        # Service names (no hardcoding)
        self.service_names = {
            'voiceprints': f"{self.SERVICE_NAME_PREFIX}.voiceprints",
            'config': f"{self.SERVICE_NAME_PREFIX}.config",
            'auth_state': f"{self.SERVICE_NAME_PREFIX}.auth-state",
            'audit': f"{self.SERVICE_NAME_PREFIX}.audit"
        }
        
    def _init_encryption(self):
        """Initialize encryption with dynamic key generation"""
        
        # Get or generate master key
        master_key = self._get_or_create_master_key()
        
        # Derive encryption key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._get_device_salt(),
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key))
        self.cipher = Fernet(key)
        
    def _get_or_create_master_key(self) -> bytes:
        """Get or create master encryption key"""
        
        key_service = f"{self.SERVICE_NAME_PREFIX}.master"
        key_account = "encryption-key"
        
        try:
            # Try to get existing key
            key_b64 = keyring.get_password(key_service, key_account)
            if key_b64:
                return base64.b64decode(key_b64)
                
        except Exception as e:
            logger.info(f"No existing master key found: {e}")
            
        # Generate new key
        master_key = os.urandom(32)
        key_b64 = base64.b64encode(master_key).decode()
        
        # Store in keychain
        try:
            keyring.set_password(key_service, key_account, key_b64)
            logger.info("Created new master encryption key")
        except Exception as e:
            logger.error(f"Failed to store master key: {e}")
            
        return master_key
        
    def _get_device_salt(self) -> bytes:
        """Generate device-specific salt"""
        
        # Combine multiple device identifiers
        import platform
        import hashlib
        
        device_info = f"{platform.node()}-{platform.machine()}-{platform.processor()}"
        return hashlib.sha256(device_info.encode()).digest()[:16]
        
    def store_voiceprint(self, voiceprint: Voiceprint) -> bool:
        """Store voiceprint securely in keychain"""
        
        try:
            # Serialize voiceprint
            voiceprint_data = self._serialize_voiceprint(voiceprint)
            
            # Encrypt if configured
            if self.config.security.encrypt_voiceprints:
                voiceprint_data = self._encrypt_data(voiceprint_data)
                
            # Store in keychain
            service = self.service_names['voiceprints']
            account = voiceprint.user_id
            
            keyring.set_password(service, account, voiceprint_data)
            
            # Store metadata separately for quick access
            metadata = {
                'user_id': voiceprint.user_id,
                'created_at': voiceprint.created_at.isoformat(),
                'updated_at': voiceprint.updated_at.isoformat(),
                'sample_count': voiceprint.sample_count,
                'quality_score': voiceprint.metadata.get('quality_score', 0)
            }
            
            self._store_metadata(voiceprint.user_id, metadata)
            
            logger.info(f"Stored voiceprint for user {voiceprint.user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store voiceprint: {e}")
            return False
            
    def load_voiceprint(self, user_id: str) -> Optional[Voiceprint]:
        """Load voiceprint from keychain"""
        
        try:
            service = self.service_names['voiceprints']
            voiceprint_data = keyring.get_password(service, user_id)
            
            if not voiceprint_data:
                return None
                
            # Decrypt if needed
            if self.config.security.encrypt_voiceprints:
                voiceprint_data = self._decrypt_data(voiceprint_data)
                
            # Deserialize
            voiceprint = self._deserialize_voiceprint(voiceprint_data)
            
            logger.info(f"Loaded voiceprint for user {user_id}")
            return voiceprint
            
        except Exception as e:
            logger.error(f"Failed to load voiceprint: {e}")
            return None
            
    def delete_voiceprint(self, user_id: str) -> bool:
        """Delete voiceprint from keychain"""
        
        try:
            service = self.service_names['voiceprints']
            keyring.delete_password(service, user_id)
            
            # Delete metadata
            self._delete_metadata(user_id)
            
            logger.info(f"Deleted voiceprint for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete voiceprint: {e}")
            return False
            
    def list_voiceprints(self) -> List[Dict[str, Any]]:
        """List all stored voiceprints (metadata only)"""
        
        try:
            # Get all metadata entries
            metadata_list = []
            
            # Note: keyring doesn't provide list functionality
            # We maintain a separate index for this
            index = self._load_index()
            
            for user_id in index.get('users', []):
                metadata = self._load_metadata(user_id)
                if metadata:
                    metadata_list.append(metadata)
                    
            return metadata_list
            
        except Exception as e:
            logger.error(f"Failed to list voiceprints: {e}")
            return []
            
    def store_auth_state(self, user_id: str, state: Dict[str, Any]) -> bool:
        """Store user authentication state"""
        
        try:
            service = self.service_names['auth_state']
            state_json = json.dumps(state, default=str)
            
            if self.config.security.encrypt_voiceprints:
                state_json = self._encrypt_data(state_json)
                
            keyring.set_password(service, user_id, state_json)
            return True
            
        except Exception as e:
            logger.error(f"Failed to store auth state: {e}")
            return False
            
    def load_auth_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Load user authentication state"""
        
        try:
            service = self.service_names['auth_state']
            state_json = keyring.get_password(service, user_id)
            
            if not state_json:
                return None
                
            if self.config.security.encrypt_voiceprints:
                state_json = self._decrypt_data(state_json)
                
            return json.loads(state_json)
            
        except Exception as e:
            logger.error(f"Failed to load auth state: {e}")
            return None
            
    def store_config_override(self, key: str, value: Any) -> bool:
        """Store configuration override in keychain"""
        
        try:
            service = self.service_names['config']
            value_json = json.dumps(value)
            keyring.set_password(service, key, value_json)
            return True
            
        except Exception as e:
            logger.error(f"Failed to store config override: {e}")
            return False
            
    def load_config_override(self, key: str) -> Optional[Any]:
        """Load configuration override from keychain"""
        
        try:
            service = self.service_names['config']
            value_json = keyring.get_password(service, key)
            
            if value_json:
                return json.loads(value_json)
            return None
            
        except Exception as e:
            logger.error(f"Failed to load config override: {e}")
            return None
            
    def audit_log(self, event: str, details: Dict[str, Any]):
        """Add entry to secure audit log"""
        
        if not self.config.security.audit_enabled:
            return
            
        try:
            # Create audit entry
            entry = {
                'timestamp': datetime.now().isoformat(),
                'event': event,
                'details': details
            }
            
            # Anonymize if configured
            if self.config.security.anonymize_logs and 'user_id' in details:
                import hashlib
                details['user_id'] = hashlib.sha256(
                    details['user_id'].encode()
                ).hexdigest()[:8]
                
            # Get existing audit log
            service = self.service_names['audit']
            audit_key = datetime.now().strftime("%Y-%m-%d")
            
            existing = keyring.get_password(service, audit_key)
            if existing:
                if self.config.security.encrypt_voiceprints:
                    existing = self._decrypt_data(existing)
                entries = json.loads(existing)
            else:
                entries = []
                
            entries.append(entry)
            
            # Limit size (keep last 1000 entries per day)
            if len(entries) > 1000:
                entries = entries[-1000:]
                
            # Store back
            audit_json = json.dumps(entries)
            if self.config.security.encrypt_voiceprints:
                audit_json = self._encrypt_data(audit_json)
                
            keyring.set_password(service, audit_key, audit_json)
            
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
            
    def _encrypt_data(self, data: str) -> str:
        """Encrypt data using Fernet"""
        return self.cipher.encrypt(data.encode()).decode()
        
    def _decrypt_data(self, data: str) -> str:
        """Decrypt data using Fernet"""
        return self.cipher.decrypt(data.encode()).decode()
        
    def _serialize_voiceprint(self, voiceprint: Voiceprint) -> str:
        """Serialize voiceprint to JSON"""
        
        # Convert numpy arrays to lists
        data = {
            'user_id': voiceprint.user_id,
            'created_at': voiceprint.created_at.isoformat(),
            'updated_at': voiceprint.updated_at.isoformat(),
            'template_vector': voiceprint.template_vector.tolist(),
            'variance_vector': voiceprint.variance_vector.tolist(),
            'metadata': voiceprint.metadata,
            'enrollment_samples': []
        }
        
        # Serialize each sample's features
        for sample in voiceprint.enrollment_samples:
            sample_data = {
                'mfcc_features': sample.mfcc_features.tolist(),
                'pitch_contour': sample.pitch_contour.tolist(),
                'spectral_centroid': float(sample.spectral_centroid),
                'zero_crossing_rate': float(sample.zero_crossing_rate),
                'energy_profile': sample.energy_profile.tolist(),
                'formants': sample.formants
            }
            data['enrollment_samples'].append(sample_data)
            
        return json.dumps(data)
        
    def _deserialize_voiceprint(self, data: str) -> Voiceprint:
        """Deserialize voiceprint from JSON"""
        
        data_dict = json.loads(data)
        
        # Reconstruct enrollment samples
        enrollment_samples = []
        for sample_data in data_dict['enrollment_samples']:
            features = VoiceFeatures(
                mfcc_features=np.array(sample_data['mfcc_features']),
                pitch_contour=np.array(sample_data['pitch_contour']),
                spectral_centroid=sample_data['spectral_centroid'],
                zero_crossing_rate=sample_data['zero_crossing_rate'],
                energy_profile=np.array(sample_data['energy_profile']),
                formants=sample_data['formants']
            )
            enrollment_samples.append(features)
            
        # Reconstruct voiceprint
        voiceprint = Voiceprint(
            user_id=data_dict['user_id'],
            created_at=datetime.fromisoformat(data_dict['created_at']),
            updated_at=datetime.fromisoformat(data_dict['updated_at']),
            enrollment_samples=enrollment_samples,
            template_vector=np.array(data_dict['template_vector']),
            variance_vector=np.array(data_dict['variance_vector']),
            metadata=data_dict['metadata']
        )
        
        return voiceprint
        
    def _store_metadata(self, user_id: str, metadata: Dict[str, Any]):
        """Store user metadata for quick access"""
        
        # Update index
        index = self._load_index()
        if 'users' not in index:
            index['users'] = []
            
        if user_id not in index['users']:
            index['users'].append(user_id)
            self._save_index(index)
            
        # Store metadata
        service = f"{self.service_names['voiceprints']}.metadata"
        metadata_json = json.dumps(metadata)
        keyring.set_password(service, user_id, metadata_json)
        
    def _load_metadata(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Load user metadata"""
        
        try:
            service = f"{self.service_names['voiceprints']}.metadata"
            metadata_json = keyring.get_password(service, user_id)
            
            if metadata_json:
                return json.loads(metadata_json)
            return None
            
        except Exception:
            return None
            
    def _delete_metadata(self, user_id: str):
        """Delete user metadata"""
        
        # Update index
        index = self._load_index()
        if 'users' in index and user_id in index['users']:
            index['users'].remove(user_id)
            self._save_index(index)
            
        # Delete metadata
        try:
            service = f"{self.service_names['voiceprints']}.metadata"
            keyring.delete_password(service, user_id)
        except Exception:
            pass
            
    def _load_index(self) -> Dict[str, Any]:
        """Load user index"""
        
        try:
            service = f"{self.service_names['voiceprints']}.index"
            index_json = keyring.get_password(service, "index")
            
            if index_json:
                return json.loads(index_json)
        except Exception:
            pass
            
        return {'users': []}
        
    def _save_index(self, index: Dict[str, Any]):
        """Save user index"""
        
        try:
            service = f"{self.service_names['voiceprints']}.index"
            index_json = json.dumps(index)
            keyring.set_password(service, "index", index_json)
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            
    def export_backup(self, backup_path: Path, password: Optional[str] = None) -> bool:
        """Export encrypted backup of all voice data"""
        
        try:
            backup_data = {
                'version': '1.0',
                'created_at': datetime.now().isoformat(),
                'voiceprints': {},
                'auth_states': {},
                'config_overrides': {}
            }
            
            # Export all voiceprints
            index = self._load_index()
            for user_id in index.get('users', []):
                voiceprint = self.load_voiceprint(user_id)
                if voiceprint:
                    backup_data['voiceprints'][user_id] = self._serialize_voiceprint(voiceprint)
                    
                auth_state = self.load_auth_state(user_id)
                if auth_state:
                    backup_data['auth_states'][user_id] = auth_state
                    
            # Encrypt backup if password provided
            backup_json = json.dumps(backup_data)
            
            if password:
                # Derive key from password
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b'jarvis-voice-backup',  # Static salt for backups
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
                cipher = Fernet(key)
                backup_data_encrypted = cipher.encrypt(backup_json.encode())
                
                with open(backup_path, 'wb') as f:
                    f.write(backup_data_encrypted)
            else:
                with open(backup_path, 'w') as f:
                    json.dump(backup_data, f, indent=2)
                    
            logger.info(f"Exported backup to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export backup: {e}")
            return False
            
    def import_backup(self, backup_path: Path, password: Optional[str] = None) -> bool:
        """Import backup of voice data"""
        
        try:
            if password:
                # Decrypt backup
                with open(backup_path, 'rb') as f:
                    encrypted_data = f.read()
                    
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b'jarvis-voice-backup',
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
                cipher = Fernet(key)
                backup_json = cipher.decrypt(encrypted_data).decode()
                backup_data = json.loads(backup_json)
            else:
                with open(backup_path, 'r') as f:
                    backup_data = json.load(f)
                    
            # Import voiceprints
            for user_id, voiceprint_data in backup_data.get('voiceprints', {}).items():
                voiceprint = self._deserialize_voiceprint(voiceprint_data)
                self.store_voiceprint(voiceprint)
                
            # Import auth states
            for user_id, auth_state in backup_data.get('auth_states', {}).items():
                self.store_auth_state(user_id, auth_state)
                
            logger.info(f"Imported backup from {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import backup: {e}")
            return False