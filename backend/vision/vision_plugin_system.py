#!/usr/bin/env python3
"""
Vision Plugin System - Extensible Vision Provider Architecture
Allows dynamic registration of vision providers without modifying core code
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Protocol, Callable, Type, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import inspect
import importlib.util
from pathlib import Path
import json
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class VisionProvider(Protocol):
    """Protocol for vision providers - defines the interface"""
    
    @property
    def name(self) -> str:
        """Provider name"""
        ...
        
    @property
    def capabilities(self) -> List[str]:
        """List of capabilities this provider offers"""
        ...
        
    async def execute(self, capability: str, **kwargs) -> Any:
        """Execute a capability"""
        ...
        
    def get_confidence(self, capability: str) -> float:
        """Get confidence score for a capability"""
        ...

@dataclass
class VisionPlugin:
    """Registered vision plugin"""
    name: str
    provider: VisionProvider
    capabilities: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_stats: Dict[str, float] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 0
    last_used: Optional[datetime] = None

class BaseVisionProvider(ABC):
    """Base class for vision providers"""
    
    def __init__(self, name: str):
        self._name = name
        self._capabilities = []
        self._confidence_scores = {}
        self._initialize()
        
    @property
    def name(self) -> str:
        return self._name
        
    @property 
    def capabilities(self) -> List[str]:
        return self._capabilities
        
    @abstractmethod
    def _initialize(self):
        """Initialize the provider and discover capabilities"""
        pass
        
    @abstractmethod
    async def execute(self, capability: str, **kwargs) -> Any:
        """Execute a capability"""
        pass
        
    def get_confidence(self, capability: str) -> float:
        """Get confidence score for a capability"""
        return self._confidence_scores.get(capability, 0.5)
        
    def register_capability(self, name: str, confidence: float = 0.8):
        """Register a capability"""
        self._capabilities.append(name)
        self._confidence_scores[name] = confidence

class VisionPluginSystem:
    """
    Plugin system for vision providers
    Allows dynamic registration and intelligent routing
    """
    
    def __init__(self):
        self.plugins: Dict[str, VisionPlugin] = {}
        self.capability_map: Dict[str, List[str]] = {}  # capability -> [provider_names]
        self.execution_history = []
        self.plugin_directory = Path("backend/vision/plugins")
        
        # ML components for intelligent routing
        self.routing_scores = {}
        self.performance_data = {}
        
        # Initialize system
        self._initialize_system()
        
    def _initialize_system(self):
        """Initialize the plugin system"""
        # Create plugin directory
        self.plugin_directory.mkdir(exist_ok=True)
        
        # Load built-in providers
        self._load_builtin_providers()
        
        # Discover and load external plugins
        self._discover_plugins()
        
        # Load performance data
        self._load_performance_data()
        
        logger.info(f"Plugin system initialized with {len(self.plugins)} providers")
        
    def _load_builtin_providers(self):
        """Load built-in vision providers"""
        builtin_providers = [
            ('ClaudeVisionProvider', self._create_claude_provider),
            ('ScreenCaptureProvider', self._create_screen_capture_provider),
            ('IntelligentVisionProvider', self._create_intelligent_provider),
            ('WorkspaceProvider', self._create_workspace_provider)
        ]
        
        for name, creator in builtin_providers:
            try:
                provider = creator()
                if provider:
                    self.register_provider(provider)
            except Exception as e:
                logger.debug(f"Could not load {name}: {e}")
                
    def _create_claude_provider(self) -> Optional[VisionProvider]:
        """Create Claude vision provider"""
        try:
            import os
            
            class ClaudeVisionProvider(BaseVisionProvider):
                def _initialize(self):
                    self.api_key = os.getenv("ANTHROPIC_API_KEY")
                    if self.api_key:
                        self.register_capability("intelligent_analysis", 0.95)
                        self.register_capability("describe_with_context", 0.95)
                        self.register_capability("answer_vision_questions", 0.9)
                        
                async def execute(self, capability: str, **kwargs) -> Any:
                    if capability == "intelligent_analysis":
                        from vision.claude_vision_analyzer import ClaudeVisionAnalyzer
                        analyzer = ClaudeVisionAnalyzer(self.api_key)
                        return await analyzer.analyze_screen(kwargs.get('image'), kwargs.get('query'))
                    # Add more capability handlers
                    return f"Executed {capability} with Claude"
                    
            return ClaudeVisionProvider("Claude Vision")
        except:
            return None
            
    def _create_screen_capture_provider(self) -> Optional[VisionProvider]:
        """Create screen capture provider"""
        try:
            class ScreenCaptureProvider(BaseVisionProvider):
                def _initialize(self):
                    self.register_capability("capture_screen", 0.99)
                    self.register_capability("capture_window", 0.95)
                    self.register_capability("capture_region", 0.9)
                    
                async def execute(self, capability: str, **kwargs) -> Any:
                    if capability == "capture_screen":
                        from vision.screen_capture_fallback import capture_screen_direct
                        return capture_screen_direct()
                    # Add more capability handlers
                    return f"Captured via {capability}"
                    
            return ScreenCaptureProvider("Screen Capture")
        except:
            return None
            
    def _create_intelligent_provider(self) -> Optional[VisionProvider]:
        """Create intelligent vision provider"""
        try:
            class IntelligentProvider(BaseVisionProvider):
                def _initialize(self):
                    try:
                        from vision.intelligent_vision_integration import IntelligentJARVISVision
                        self.vision = IntelligentJARVISVision()
                        self.register_capability("intelligent_command", 0.9)
                        self.register_capability("contextual_analysis", 0.85)
                    except:
                        pass
                        
                async def execute(self, capability: str, **kwargs) -> Any:
                    if hasattr(self, 'vision'):
                        if capability == "intelligent_command":
                            return await self.vision.handle_intelligent_command(kwargs.get('command', ''))
                    return "Intelligent vision not available"
                    
            return IntelligentProvider("Intelligent Vision")
        except:
            return None
            
    def _create_workspace_provider(self) -> Optional[VisionProvider]:
        """Create workspace vision provider"""
        try:
            class WorkspaceProvider(BaseVisionProvider):
                def _initialize(self):
                    try:
                        from vision.jarvis_workspace_integration import JARVISWorkspaceIntelligence
                        self.workspace = JARVISWorkspaceIntelligence()
                        self.register_capability("workspace_analysis", 0.9)
                        self.register_capability("multi_window_analysis", 0.85)
                    except:
                        pass
                        
                async def execute(self, capability: str, **kwargs) -> Any:
                    if hasattr(self, 'workspace'):
                        if capability == "workspace_analysis":
                            return await self.workspace.handle_workspace_command(kwargs.get('command', ''))
                    return "Workspace vision not available"
                    
            return WorkspaceProvider("Workspace Vision")
        except:
            return None
            
    def _discover_plugins(self):
        """Discover and load external plugins"""
        plugin_files = self.plugin_directory.glob("*.py")
        
        for plugin_file in plugin_files:
            if plugin_file.name.startswith("_"):
                continue
                
            try:
                # Load the plugin module
                spec = importlib.util.spec_from_file_location(
                    f"vision_plugin_{plugin_file.stem}",
                    plugin_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Look for VisionProvider implementations
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseVisionProvider) and 
                        obj is not BaseVisionProvider):
                        # Create instance and register
                        provider = obj(name)
                        self.register_provider(provider)
                        logger.info(f"Loaded plugin: {name} from {plugin_file.name}")
                        
            except Exception as e:
                logger.error(f"Error loading plugin {plugin_file}: {e}")
                
    def register_provider(self, provider: VisionProvider, priority: int = 0):
        """Register a vision provider"""
        plugin = VisionPlugin(
            name=provider.name,
            provider=provider,
            capabilities=provider.capabilities,
            priority=priority
        )
        
        self.plugins[provider.name] = plugin
        
        # Update capability map
        for capability in provider.capabilities:
            if capability not in self.capability_map:
                self.capability_map[capability] = []
            self.capability_map[capability].append(provider.name)
            
        logger.info(f"Registered provider: {provider.name} with {len(provider.capabilities)} capabilities")
        
    def unregister_provider(self, name: str):
        """Unregister a provider"""
        if name in self.plugins:
            plugin = self.plugins[name]
            # Remove from capability map
            for capability in plugin.capabilities:
                if capability in self.capability_map:
                    self.capability_map[capability].remove(name)
                    if not self.capability_map[capability]:
                        del self.capability_map[capability]
                        
            del self.plugins[name]
            logger.info(f"Unregistered provider: {name}")
            
    async def execute_capability(self, capability: str, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute a capability using the best available provider
        Returns result and metadata
        """
        # Find providers for this capability
        providers = self.capability_map.get(capability, [])
        
        if not providers:
            # Try to find similar capabilities
            similar = self._find_similar_capabilities(capability)
            if similar:
                return None, {
                    'error': f"Capability '{capability}' not found",
                    'suggestions': similar
                }
            return None, {'error': f"No provider for capability '{capability}'"}
            
        # Score providers
        provider_scores = self._score_providers_for_capability(capability, providers, kwargs)
        
        # Try providers in order of score
        errors = []
        for provider_name, score in sorted(provider_scores.items(), key=lambda x: x[1], reverse=True):
            plugin = self.plugins[provider_name]
            
            if not plugin.enabled:
                continue
                
            try:
                # Execute
                start_time = datetime.now()
                result = await plugin.provider.execute(capability, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Update stats
                plugin.last_used = datetime.now()
                self._update_performance_stats(provider_name, capability, True, execution_time)
                
                # Record execution
                self.execution_history.append({
                    'capability': capability,
                    'provider': provider_name,
                    'success': True,
                    'execution_time': execution_time,
                    'timestamp': datetime.now()
                })
                
                return result, {
                    'provider': provider_name,
                    'confidence': score,
                    'execution_time': execution_time
                }
                
            except Exception as e:
                errors.append(f"{provider_name}: {str(e)}")
                self._update_performance_stats(provider_name, capability, False, 0)
                
        # All providers failed
        return None, {
            'error': "All providers failed",
            'errors': errors,
            'providers_tried': list(provider_scores.keys())
        }
        
    def _score_providers_for_capability(self, capability: str, providers: List[str], kwargs: Dict) -> Dict[str, float]:
        """Score providers for a capability"""
        scores = {}
        
        for provider_name in providers:
            plugin = self.plugins[provider_name]
            
            # Base confidence from provider
            score = plugin.provider.get_confidence(capability)
            
            # Adjust based on performance history
            perf_key = f"{provider_name}:{capability}"
            if perf_key in self.performance_data:
                perf = self.performance_data[perf_key]
                success_rate = perf['successes'] / (perf['successes'] + perf['failures']) if perf['failures'] + perf['successes'] > 0 else 0
                avg_time = perf['total_time'] / perf['successes'] if perf['successes'] > 0 else 1.0
                
                # Boost score based on success rate
                score *= (0.5 + 0.5 * success_rate)
                
                # Slight penalty for slow providers
                if avg_time > 2.0:
                    score *= 0.9
                    
            # Boost recently used providers slightly (momentum)
            if plugin.last_used:
                recency = (datetime.now() - plugin.last_used).total_seconds()
                if recency < 300:  # Used in last 5 minutes
                    score *= 1.1
                    
            # Apply priority
            score *= (1 + plugin.priority * 0.1)
            
            scores[provider_name] = min(score, 1.0)
            
        return scores
        
    def _find_similar_capabilities(self, query: str) -> List[str]:
        """Find similar capabilities"""
        suggestions = []
        query_lower = query.lower()
        
        for capability in self.capability_map:
            if query_lower in capability.lower() or capability.lower() in query_lower:
                suggestions.append(capability)
            elif any(word in capability.lower() for word in query_lower.split('_')):
                suggestions.append(capability)
                
        return suggestions[:5]
        
    def _update_performance_stats(self, provider: str, capability: str, success: bool, execution_time: float):
        """Update performance statistics"""
        key = f"{provider}:{capability}"
        
        if key not in self.performance_data:
            self.performance_data[key] = {
                'successes': 0,
                'failures': 0,
                'total_time': 0.0
            }
            
        if success:
            self.performance_data[key]['successes'] += 1
            self.performance_data[key]['total_time'] += execution_time
        else:
            self.performance_data[key]['failures'] += 1
            
        # Periodically save
        if len(self.execution_history) % 10 == 0:
            self._save_performance_data()
            
    def _save_performance_data(self):
        """Save performance data"""
        save_path = Path("backend/data/vision_plugin_performance.json")
        save_path.parent.mkdir(exist_ok=True)
        
        try:
            with open(save_path, 'w') as f:
                json.dump(self.performance_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
            
    def _load_performance_data(self):
        """Load performance data"""
        save_path = Path("backend/data/vision_plugin_performance.json")
        
        if save_path.exists():
            try:
                with open(save_path, 'r') as f:
                    self.performance_data = json.load(f)
            except Exception as e:
                logger.error(f"Error loading performance data: {e}")
                self.performance_data = {}
        else:
            self.performance_data = {}
            
    def get_provider_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a provider"""
        if name not in self.plugins:
            return None
            
        plugin = self.plugins[name]
        return {
            'name': plugin.name,
            'capabilities': plugin.capabilities,
            'enabled': plugin.enabled,
            'priority': plugin.priority,
            'last_used': plugin.last_used.isoformat() if plugin.last_used else None,
            'metadata': plugin.metadata
        }
        
    def list_capabilities(self) -> Dict[str, List[str]]:
        """List all capabilities and their providers"""
        return dict(self.capability_map)
        
    def enable_provider(self, name: str):
        """Enable a provider"""
        if name in self.plugins:
            self.plugins[name].enabled = True
            
    def disable_provider(self, name: str):
        """Disable a provider"""
        if name in self.plugins:
            self.plugins[name].enabled = False
            
    def set_provider_priority(self, name: str, priority: int):
        """Set provider priority"""
        if name in self.plugins:
            self.plugins[name].priority = priority

# Singleton instance
_plugin_system = None

def get_vision_plugin_system() -> VisionPluginSystem:
    """Get singleton plugin system"""
    global _plugin_system
    if _plugin_system is None:
        _plugin_system = VisionPluginSystem()
    return _plugin_system