"""
Emergency patch to prevent model loader crashes
Limits model discovery to prevent loading 197 models at once
"""

import logging
from typing import Dict, Set

logger = logging.getLogger(__name__)

# Models to explicitly exclude from auto-discovery
EXCLUDED_MODELS = {
    # Pydantic base models that shouldn't be instantiated
    "BaseModel", "BaseSettings", "BaseConfig",
    
    # Request/Response models
    "Request", "Response", "Config", "Settings",
    
    # Test and demo models
    "TestModel", "DemoModel", "MockModel",
    
    # Abstract base classes
    "AbstractModel", "BaseHandler", "BaseEngine",
}

# Only load these critical models initially
CRITICAL_MODELS_WHITELIST = {
    "vision_core": "vision.vision_system_v2.VisionSystemV2",
    "claude_vision": "vision.workspace_analyzer.WorkspaceAnalyzer", 
    "simple_chatbot": "chatbots.simple_chatbot.SimpleChatbot",
}

def should_skip_model(model_name: str, class_name: str) -> bool:
    """Check if a model should be skipped during discovery"""
    
    # Skip if class name contains excluded terms
    for excluded in EXCLUDED_MODELS:
        if excluded in class_name:
            return True
    
    # Skip data models and configs
    skip_patterns = [
        "Config", "Settings", "Options", "Params",
        "Request", "Response", "Schema", "DTO",
        "Base", "Abstract", "Interface", "Protocol",
        "Test", "Mock", "Demo", "Example"
    ]
    
    for pattern in skip_patterns:
        if pattern in class_name:
            return True
    
    # Skip if it's just a data container (Pydantic model with no methods)
    if model_name.endswith("_model") and not any(
        keyword in class_name.lower() 
        for keyword in ["system", "engine", "manager", "handler"]
    ):
        return True
    
    return False

def patch_model_discovery():
    """Monkey-patch the model discovery to limit loaded models"""
    try:
        from utils.progressive_model_loader import DynamicModelDiscovery, ModelInfo
        
        original_discover = DynamicModelDiscovery.discover_models
        original_is_model = DynamicModelDiscovery._is_model_class
        
        # Track discovered count
        discovered_count = 0
        max_discovery = 20  # Limit total discovered models
        
        def patched_is_model_class(self, obj, name: str) -> bool:
            """Patched version with stricter filtering"""
            # First apply original logic
            is_model = original_is_model(self, obj, name)
            if not is_model:
                return False
                
            # Additional filtering
            return not should_skip_model("", name)
        
        def patched_discover_models(self) -> Dict[str, ModelInfo]:
            """Patched discovery that limits models"""
            nonlocal discovered_count
            
            logger.info("🛡️ Model discovery patch active - limiting auto-discovery")
            
            # Start with critical models only
            self.discovered_models = {}
            
            # Add whitelisted models first
            for key, path_and_class in CRITICAL_MODELS_WHITELIST.items():
                parts = path_and_class.rsplit(".", 1)
                if len(parts) == 2:
                    module_path, class_name = parts
                    self.discovered_models[key] = ModelInfo(
                        name=key,
                        module_path=module_path,
                        class_name=class_name,
                        category="critical",
                        priority=1,
                        lazy=False
                    )
                    discovered_count += 1
            
            logger.info(f"✅ Added {len(CRITICAL_MODELS_WHITELIST)} critical models")
            
            # Carefully discover additional models with limit
            original_result = original_discover(self)
            
            # Filter and limit discovered models
            for name, model_info in original_result.items():
                if discovered_count >= max_discovery:
                    logger.info(f"🛑 Reached discovery limit ({max_discovery} models)")
                    break
                    
                if name not in self.discovered_models:
                    # Apply additional filtering
                    if not should_skip_model(name, model_info.class_name):
                        self.discovered_models[name] = model_info
                        discovered_count += 1
            
            logger.info(f"📦 Limited discovery to {len(self.discovered_models)} models (was 197)")
            return self.discovered_models
        
        # Apply patches
        DynamicModelDiscovery.discover_models = patched_discover_models  
        DynamicModelDiscovery._is_model_class = patched_is_model_class
        
        logger.info("✅ Model discovery patch applied successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to apply model discovery patch: {e}")

# Auto-apply patch when imported
patch_model_discovery()