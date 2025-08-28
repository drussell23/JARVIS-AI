#!/usr/bin/env python3
"""
Migrate Vision System to Lazy Loading
Replaces eager model loading with lazy initialization
"""

import os
import sys
from pathlib import Path
import shutil
from datetime import datetime

def create_backup(file_path: Path):
    """Create backup of file before modification"""
    backup_name = f"{file_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_path.suffix}"
    backup_path = file_path.parent / backup_name
    shutil.copy2(file_path, backup_path)
    print(f"✓ Created backup: {backup_path}")
    return backup_path

def migrate_dynamic_vision_engine():
    """Update dynamic_vision_engine.py to use lazy loading"""
    file_path = Path("dynamic_vision_engine.py")
    if not file_path.exists():
        print("❌ dynamic_vision_engine.py not found")
        return False
        
    print(f"\n📝 Migrating {file_path}...")
    
    # Create wrapper that imports lazy engine
    wrapper_content = '''#!/usr/bin/env python3
"""
Dynamic Vision Engine - Wrapper for Lazy Loading
This module wraps the lazy vision engine for backward compatibility
"""

import logging
from .lazy_vision_engine import (
    get_lazy_vision_engine,
    initialize_vision_engine_models,
    VisionCapability,
    VisionIntent
)

logger = logging.getLogger(__name__)

# Create a wrapper class that delegates to lazy engine
class DynamicVisionEngine:
    """Wrapper for lazy vision engine - maintains compatibility"""
    
    def __init__(self):
        self._engine = get_lazy_vision_engine()
        logger.info("Dynamic Vision Engine initialized with lazy loading")
        
    def __getattr__(self, name):
        """Delegate all attributes to lazy engine"""
        return getattr(self._engine, name)
        
    async def initialize(self):
        """Initialize models - for compatibility"""
        if not self._engine._models_loaded:
            await initialize_vision_engine_models()
            
    def get_model_info(self):
        """Get model loading status"""
        return self._engine.get_model_info()

# For backward compatibility
def get_vision_engine():
    """Get the global vision engine instance"""
    return DynamicVisionEngine()
'''
    
    # Backup original file
    backup_path = create_backup(file_path)
    
    # Write new content
    with open(file_path, 'w') as f:
        f.write(wrapper_content)
    
    print(f"✓ Migrated to lazy loading wrapper")
    return True

def migrate_vision_system_v2():
    """Update vision_system_v2.py to use lazy initialization"""
    file_path = Path("vision_system_v2.py")
    if not file_path.exists():
        print("❌ vision_system_v2.py not found")
        return False
        
    print(f"\n📝 Migrating {file_path}...")
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Create backup
    backup_path = create_backup(file_path)
    
    # Replace eager model loading patterns
    replacements = [
        # Replace model loading in __init__
        (
            "self.ml_router = MLIntentClassifier()",
            "self._ml_router = None  # Lazy loaded"
        ),
        (
            "self.semantic_engine = SemanticUnderstandingEngine()",
            "self._semantic_engine = None  # Lazy loaded"
        ),
        (
            "self.transformer_router = TransformerCommandRouter()",
            "self._transformer_router = None  # Lazy loaded"
        ),
        (
            "self.neural_router = NeuralCommandRouter()",
            "self._neural_router = None  # Lazy loaded"
        ),
        # Add property for lazy access
        (
            "class VisionSystemV2:",
            '''class VisionSystemV2:
    @property
    def ml_router(self):
        if self._ml_router is None:
            from .ml_intent_classifier import MLIntentClassifier
            self._ml_router = MLIntentClassifier()
        return self._ml_router
        
    @property
    def semantic_engine(self):
        if self._semantic_engine is None:
            from .semantic_understanding_engine import SemanticUnderstandingEngine
            self._semantic_engine = SemanticUnderstandingEngine()
        return self._semantic_engine
        
    @property
    def transformer_router(self):
        if self._transformer_router is None:
            try:
                from .transformer_command_router import TransformerCommandRouter
                self._transformer_router = TransformerCommandRouter()
            except ImportError:
                self._transformer_router = None
        return self._transformer_router
        
    @property
    def neural_router(self):
        if self._neural_router is None:
            try:
                from .neural_command_router import NeuralCommandRouter
                self._neural_router = NeuralCommandRouter()
            except ImportError:
                self._neural_router = None
        return self._neural_router
    '''
        )
    ]
    
    # Apply replacements
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"✓ Replaced: {old[:50]}...")
    
    # Write updated content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Migrated to lazy loading")
    return True

def update_ml_model_loader():
    """Update ML model loader to include vision models"""
    loader_path = Path("../ml_model_loader.py")
    if not loader_path.exists():
        print("❌ ml_model_loader.py not found")
        return False
        
    print(f"\n📝 Updating ML model loader...")
    
    with open(loader_path, 'r') as f:
        content = f.read()
    
    # Add vision model loading
    vision_loader_code = '''
    
    # Add vision models to base directories
    if 'vision' not in self.base_dirs:
        self.base_dirs.append('vision')
        
    # Register vision engine for parallel loading
    try:
        from vision.lazy_vision_engine import initialize_vision_engine_models
        
        # Create a special model info for vision engine
        vision_model = ModelInfo(
            name="vision.lazy_vision_engine",
            module_path="vision.lazy_vision_engine",
            init_function="initialize_vision_engine_models",
            priority=10  # High priority
        )
        
        self.models[vision_model.name] = vision_model
        logger.info("Registered lazy vision engine for parallel loading")
    except ImportError:
        logger.warning("Could not import lazy vision engine")
'''
    
    # Find where to insert (after __init__ method)
    if "def __init__" in content and vision_loader_code not in content:
        # Find the end of __init__
        init_start = content.find("def __init__")
        init_end = content.find("\n    def", init_start + 1)
        
        if init_end > 0:
            # Insert before next method
            content = content[:init_end] + vision_loader_code + content[init_end:]
            
            with open(loader_path, 'w') as f:
                f.write(content)
            
            print("✓ Added vision models to ML loader")
            return True
            
    print("⚠️  ML loader already updated or couldn't find insertion point")
    return True

def create_vision_init_wrapper():
    """Create __init__.py wrapper for vision module"""
    init_path = Path("__init__.py")
    
    print(f"\n📝 Creating vision module wrapper...")
    
    wrapper_content = '''#!/usr/bin/env python3
"""
Vision Module - Lazy Loading Wrapper
All models are loaded on-demand or through parallel loading
"""

import logging

logger = logging.getLogger(__name__)

# Import core components with lazy loading
try:
    from .lazy_vision_engine import get_lazy_vision_engine, initialize_vision_engine_models
    from .dynamic_vision_engine import DynamicVisionEngine
    
    # Create default instance
    vision_engine = get_lazy_vision_engine()
    
    logger.info("Vision module initialized with lazy loading")
except ImportError as e:
    logger.warning(f"Could not import lazy vision components: {e}")
    vision_engine = None

# Export main interfaces
__all__ = [
    'vision_engine',
    'get_lazy_vision_engine',
    'initialize_vision_engine_models',
    'DynamicVisionEngine'
]
'''
    
    # Backup if exists
    if init_path.exists():
        backup_path = create_backup(init_path)
    
    # Write new content
    with open(init_path, 'w') as f:
        f.write(wrapper_content)
    
    print("✓ Created lazy loading wrapper")
    return True

def main():
    """Run the migration"""
    print("🚀 Starting Vision System Lazy Loading Migration...")
    print("=" * 60)
    
    # Change to vision directory
    vision_dir = Path(__file__).parent
    os.chdir(vision_dir)
    
    # Run migrations
    success = True
    
    # 1. Create lazy loading wrapper
    success &= create_vision_init_wrapper()
    
    # 2. Migrate dynamic vision engine
    success &= migrate_dynamic_vision_engine()
    
    # 3. Migrate vision system v2
    success &= migrate_vision_system_v2()
    
    # 4. Update ML model loader
    success &= update_ml_model_loader()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Migration completed successfully!")
        print("\nNext steps:")
        print("1. Test the backend startup: python main.py")
        print("2. Verify model loading: check logs for parallel loading")
        print("3. Test vision commands to ensure functionality")
    else:
        print("❌ Migration had some issues. Check the logs above.")
        print("\nYou can restore from backups if needed.")

if __name__ == "__main__":
    main()