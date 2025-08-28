#!/usr/bin/env python3
"""
Dynamic ML Model Loader with Parallel Initialization
Automatically discovers and loads all ML models in parallel with detailed progress tracking
"""

import asyncio
import time
import logging
import importlib
import inspect
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import traceback
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    """Status of model loading"""
    PENDING = "pending"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class ModelInfo:
    """Information about a model to be loaded"""
    name: str
    module_path: str
    class_name: Optional[str] = None
    init_function: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    status: ModelStatus = ModelStatus.PENDING
    load_time: float = 0.0
    error: Optional[str] = None
    instance: Optional[Any] = None
    priority: int = 0  # Higher priority loads first
    
class DynamicModelLoader:
    """Dynamically discovers and loads ML models in parallel"""
    
    def __init__(self, 
                 base_dirs: Optional[List[str]] = None,
                 max_workers: Optional[int] = None,
                 progress_callback: Optional[Callable] = None):
        """
        Initialize the dynamic model loader
        
        Args:
            base_dirs: List of directories to search for models
            max_workers: Maximum number of parallel workers (defaults to CPU count)
            progress_callback: Callback for progress updates
        """
        self.base_dirs = base_dirs or ['vision', 'voice', 'audio', 'autonomy']
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        self.progress_callback = progress_callback
        
        self.models: Dict[str, ModelInfo] = {}
        self.load_order: List[str] = []
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Model patterns to discover (fully dynamic)
        self.model_patterns = [
            # Pattern: (file_pattern, class_pattern, priority)
            ('*_model.py', '*Model', 10),
            ('*_system.py', '*System', 8),
            ('*_engine.py', '*Engine', 7),
            ('*_classifier.py', '*Classifier', 9),
            ('*_router.py', '*Router', 6),
            ('*_manager.py', '*Manager', 5),
            ('*_handler.py', '*Handler', 4),
            ('*_processor.py', '*Processor', 3),
            ('ml_*.py', None, 8),  # ML modules without specific class
        ]
        
    async def discover_models(self) -> Dict[str, ModelInfo]:
        """Dynamically discover all ML models in the codebase"""
        logger.info(f"🔍 Discovering ML models in directories: {self.base_dirs}")
        
        discovered = 0
        for base_dir in self.base_dirs:
            base_path = Path(base_dir)
            if not base_path.exists():
                continue
                
            # Recursively find all Python files
            for py_file in base_path.rglob('*.py'):
                # Skip __pycache__ and test files
                if '__pycache__' in str(py_file) or 'test_' in py_file.name:
                    continue
                    
                # Check against patterns
                for pattern, class_pattern, priority in self.model_patterns:
                    if self._matches_pattern(py_file.name, pattern):
                        models = await self._extract_models_from_file(
                            py_file, class_pattern, priority
                        )
                        for model in models:
                            self.models[model.name] = model
                            discovered += 1
                        break
                        
        logger.info(f"✅ Discovered {discovered} ML models/components")
        
        # Add vision engine for parallel loading
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
            discovered += 1
        except ImportError:
            logger.warning("Could not import lazy vision engine")
        
        # Resolve dependencies
        await self._resolve_dependencies()
        
        return self.models

    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches pattern (supports wildcards)"""
        import fnmatch
        return fnmatch.fnmatch(filename, pattern)
        
    async def _extract_models_from_file(self, 
                                      py_file: Path, 
                                      class_pattern: Optional[str],
                                      priority: int) -> List[ModelInfo]:
        """Extract model information from a Python file"""
        models = []
        
        try:
            # Get module path relative to current directory
            # Try to make it relative to the current working directory
            try:
                rel_path = py_file.relative_to(Path.cwd())
            except ValueError:
                # If that fails, try to make the path relative to backend
                backend_path = Path(__file__).parent
                try:
                    rel_path = py_file.relative_to(backend_path)
                except ValueError:
                    # If still fails, use the absolute path
                    rel_path = py_file
                    
            module_path = str(rel_path).replace('/', '.').replace('.py', '')
            
            # Try to load module metadata without executing
            with open(py_file, 'r') as f:
                content = f.read()
                
            # Look for ML-related imports (indicates ML model)
            ml_indicators = [
                'torch', 'tensorflow', 'keras', 'sklearn',
                'transformers', 'sentence_transformers', 'numpy',
                'cv2', 'PIL', 'faiss', 'joblib'
            ]
            
            has_ml = any(indicator in content for indicator in ml_indicators)
            if not has_ml:
                return models
                
            # Extract class names and functions
            import ast
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                model_info = None
                
                # Find classes
                if isinstance(node, ast.ClassDef):
                    if class_pattern is None or self._matches_pattern(node.name, class_pattern):
                        # Check if it has ML-related methods
                        ml_methods = ['fit', 'predict', 'forward', 'train', 'load_model',
                                     'process', 'analyze', 'detect', 'classify']
                        has_ml_method = any(
                            isinstance(item, ast.FunctionDef) and item.name in ml_methods
                            for item in node.body
                        )
                        
                        if has_ml_method or class_pattern:
                            model_info = ModelInfo(
                                name=f"{module_path}.{node.name}",
                                module_path=module_path,
                                class_name=node.name,
                                priority=priority
                            )
                            
                # Find standalone initialization functions
                elif isinstance(node, ast.FunctionDef):
                    if node.name.startswith('init_') or node.name.startswith('load_'):
                        model_info = ModelInfo(
                            name=f"{module_path}.{node.name}",
                            module_path=module_path,
                            init_function=node.name,
                            priority=priority - 1  # Functions have slightly lower priority
                        )
                        
                if model_info:
                    # Extract dependencies from imports
                    model_info.dependencies = self._extract_dependencies(content, module_path)
                    models.append(model_info)
                    
        except Exception as e:
            logger.debug(f"Could not extract models from {py_file}: {e}")
            
        return models
        
    def _extract_dependencies(self, content: str, module_path: str) -> List[str]:
        """Extract model dependencies from imports"""
        dependencies = []
        
        # Look for relative imports that might be other models
        import_patterns = [
            r'from \. import (\w+)',
            r'from \.\.(\w+) import',
            r'from (\w+) import \w*[Mm]odel',
            r'from (\w+) import \w*[Ss]ystem',
        ]
        
        import re
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            dependencies.extend(matches)
            
        return dependencies
        
    async def _resolve_dependencies(self):
        """Resolve dependencies and determine load order"""
        # Perform topological sort for dependency resolution
        from collections import defaultdict, deque
        
        # Build dependency graph
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        for name, model in self.models.items():
            in_degree[name] = len(model.dependencies)
            for dep in model.dependencies:
                # Find matching model
                for model_name in self.models:
                    if dep in model_name:
                        graph[model_name].append(name)
                        
        # Topological sort with priority consideration
        queue = deque()
        
        # Start with models that have no dependencies
        for name, degree in in_degree.items():
            if degree == 0:
                queue.append((self.models[name].priority, name))
                
        # Sort by priority
        queue = deque(sorted(queue, reverse=True))
        
        while queue:
            _, current = queue.popleft()
            self.load_order.append(current)
            
            # Reduce in-degree for dependent models
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append((self.models[neighbor].priority, neighbor))
                    queue = deque(sorted(queue, reverse=True))
                    
        # Add any remaining models (circular dependencies)
        for name in self.models:
            if name not in self.load_order:
                self.load_order.append(name)
                
    async def load_models_parallel(self) -> Dict[str, Any]:
        """Load all discovered models in parallel"""
        logger.info(f"🚀 Starting parallel model loading with {self.max_workers} workers")
        
        start_time = time.time()
        loaded_models = {}
        
        # Group models by priority for batch loading
        priority_groups = {}
        for name in self.load_order:
            model = self.models[name]
            priority = model.priority
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(name)
            
        # Load models in priority batches
        for priority in sorted(priority_groups.keys(), reverse=True):
            batch = priority_groups[priority]
            logger.info(f"📦 Loading priority {priority} models: {len(batch)} models")
            
            # Create tasks for parallel loading
            tasks = []
            for model_name in batch:
                task = asyncio.create_task(self._load_model_async(model_name))
                tasks.append((model_name, task))
                
            # Wait for batch to complete
            for model_name, task in tasks:
                try:
                    result = await task
                    if result:
                        loaded_models[model_name] = result
                except Exception as e:
                    logger.error(f"Failed to load {model_name}: {e}")
                    
            # Progress update
            if self.progress_callback:
                loaded = sum(1 for m in self.models.values() if m.status == ModelStatus.LOADED)
                total = len(self.models)
                # Check if callback is async
                if asyncio.iscoroutinefunction(self.progress_callback):
                    await self.progress_callback(loaded, total)
                else:
                    self.progress_callback(loaded, total)
                
        total_time = time.time() - start_time
        
        # Log summary
        loaded = sum(1 for m in self.models.values() if m.status == ModelStatus.LOADED)
        failed = sum(1 for m in self.models.values() if m.status == ModelStatus.FAILED)
        
        logger.info(f"""
🎉 Model Loading Complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Successfully loaded: {loaded} models
❌ Failed: {failed} models  
⏱️  Total time: {total_time:.2f} seconds
⚡ Average time per model: {total_time/max(loaded, 1):.2f} seconds
🔧 Parallel workers used: {self.max_workers}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """)
        
        # Log individual model times
        logger.info("📊 Model Loading Times:")
        for name, model in sorted(self.models.items(), key=lambda x: x[1].load_time, reverse=True):
            if model.status == ModelStatus.LOADED:
                logger.info(f"  • {name}: {model.load_time:.2f}s ✅")
            elif model.status == ModelStatus.FAILED:
                logger.info(f"  • {name}: FAILED ❌ - {model.error}")
                
        return loaded_models
        
    async def _load_model_async(self, model_name: str) -> Optional[Any]:
        """Load a single model asynchronously"""
        model = self.models[model_name]
        model.status = ModelStatus.LOADING
        start_time = time.time()
        
        logger.info(f"⏳ Loading {model_name}...")
        
        try:
            # Run the actual loading in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            instance = await loop.run_in_executor(
                self.executor,
                self._load_model_sync,
                model
            )
            
            # Check if this is an async init that needs to be called
            if isinstance(instance, dict) and 'async_init' in instance:
                async_init = instance['async_init']
                logger.info(f"Running async initialization for {model_name}")
                # Pass the executor to the async init function if it accepts it
                try:
                    import inspect
                    sig = inspect.signature(async_init)
                    if 'executor' in sig.parameters:
                        instance = await async_init(executor=self.executor)
                    else:
                        instance = await async_init()
                except Exception as e:
                    logger.warning(f"Async init failed for {model_name}: {e}")
                    instance = None
            
            model.instance = instance
            model.status = ModelStatus.LOADED
            model.load_time = time.time() - start_time
            
            logger.info(f"✅ Loaded {model_name} in {model.load_time:.2f}s")
            return instance
            
        except Exception as e:
            model.status = ModelStatus.FAILED
            model.error = str(e)
            model.load_time = time.time() - start_time
            
            logger.error(f"❌ Failed to load {model_name}: {e}")
            logger.debug(traceback.format_exc())
            return None
            
    def _load_model_sync(self, model: ModelInfo) -> Any:
        """Synchronously load a model (runs in thread pool)"""
        try:
            # Import the module
            module = importlib.import_module(model.module_path)
            
            # Load based on type
            if model.class_name:
                # Instantiate class
                cls = getattr(module, model.class_name)
                
                # Check if it needs special initialization
                if hasattr(cls, 'load') and callable(getattr(cls, 'load')):
                    instance = cls.load()
                elif hasattr(cls, 'from_pretrained') and callable(getattr(cls, 'from_pretrained')):
                    # For transformers models
                    instance = cls.from_pretrained(cls.__name__.lower())
                else:
                    # Standard initialization
                    instance = cls()
                    
                return instance
                
            elif model.init_function:
                # Call initialization function
                init_func = getattr(module, model.init_function)
                # Check if it's an async function
                if asyncio.iscoroutinefunction(init_func):
                    # For async init functions, we need to handle them specially
                    # Since we're in a sync context, we'll return a placeholder
                    return {'async_init': init_func, 'model_name': model.name}
                else:
                    return init_func()
                
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")
            
    def get_status(self) -> Dict[str, Any]:
        """Get current loading status"""
        status = {
            'total': len(self.models),
            'loaded': sum(1 for m in self.models.values() if m.status == ModelStatus.LOADED),
            'failed': sum(1 for m in self.models.values() if m.status == ModelStatus.FAILED),
            'pending': sum(1 for m in self.models.values() if m.status == ModelStatus.PENDING),
            'loading': sum(1 for m in self.models.values() if m.status == ModelStatus.LOADING),
            'models': {}
        }
        
        for name, model in self.models.items():
            status['models'][name] = {
                'status': model.status.value,
                'load_time': model.load_time,
                'error': model.error
            }
            
        return status
        

# Global loader instance
_loader_instance: Optional[DynamicModelLoader] = None

async def initialize_models(progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Initialize all ML models in parallel with progress tracking"""
    global _loader_instance
    
    if _loader_instance is None:
        _loader_instance = DynamicModelLoader(progress_callback=progress_callback)
        
    # Discover models
    await _loader_instance.discover_models()
    
    # Load in parallel
    loaded_models = await _loader_instance.load_models_parallel()
    
    return loaded_models
    
def get_loader_status() -> Dict[str, Any]:
    """Get current model loading status"""
    if _loader_instance:
        return _loader_instance.get_status()
    return {'error': 'Loader not initialized'}


# Example usage
if __name__ == "__main__":
    async def progress_update(loaded: int, total: int):
        print(f"Progress: {loaded}/{total} models loaded ({loaded/total*100:.1f}%)")
        
    async def main():
        models = await initialize_models(progress_callback=progress_update)
        print(f"\nLoaded {len(models)} models successfully!")
        
    asyncio.run(main())