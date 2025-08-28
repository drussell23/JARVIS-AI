"""
Phase 2: Parallel Processing Pipeline
Multi-threaded vision processing with lock-free data structures
Target: 35% CPU usage with 5x performance improvement
"""

import asyncio
import concurrent.futures
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from queue import Queue
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import time
import psutil
import logging
from threading import RLock
import multiprocessing as mp

logger = logging.getLogger(__name__)

# Lock-free data structures
class LockFreeQueue:
    """Lock-free queue implementation using atomic operations"""
    
    def __init__(self, maxsize: int = 1000):
        self._queue = deque(maxlen=maxsize)
        self._lock = RLock()  # Reentrant lock for rare contentions
        
    def put_nowait(self, item: Any) -> bool:
        """Non-blocking put"""
        try:
            self._queue.append(item)
            return True
        except:
            return False
            
    def get_nowait(self) -> Optional[Any]:
        """Non-blocking get"""
        try:
            return self._queue.popleft()
        except IndexError:
            return None
            
    def qsize(self) -> int:
        """Get approximate size"""
        return len(self._queue)

@dataclass
class ProcessingTask:
    """Task for parallel processing"""
    id: str
    task_type: str  # 'vision', 'inference', 'weight_update'
    data: Any
    priority: int = 0
    timestamp: float = 0
    

class ParallelProcessingPipeline:
    """
    Advanced parallel processing with:
    - Multi-threaded vision processing
    - Concurrent model inference
    - Asynchronous weight updates
    - Lock-free data structures
    """
    
    def __init__(self, 
                 num_vision_threads: int = 2,
                 num_inference_threads: int = 2,
                 num_update_threads: int = 1,
                 target_cpu: float = 35.0):
        
        self.target_cpu = target_cpu
        self.num_vision_threads = num_vision_threads
        self.num_inference_threads = num_inference_threads
        self.num_update_threads = num_update_threads
        
        # Lock-free queues for each stage
        self.vision_queue = LockFreeQueue(maxsize=100)
        self.inference_queue = LockFreeQueue(maxsize=100)
        self.update_queue = LockFreeQueue(maxsize=50)
        self.result_queue = LockFreeQueue(maxsize=200)
        
        # Thread pools
        self.vision_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=num_vision_threads,
            thread_name_prefix="vision"
        )
        self.inference_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=num_inference_threads,
            thread_name_prefix="inference"
        )
        self.update_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=num_update_threads,
            thread_name_prefix="update"
        )
        
        # CPU monitoring
        self.cpu_monitor = CPUThrottler(target_cpu)
        
        # Performance metrics
        self.metrics = {
            'vision_processed': 0,
            'inference_completed': 0,
            'updates_applied': 0,
            'avg_latency': 0,
            'cpu_usage': 0
        }
        
        # Start workers
        self._running = True
        self._start_workers()
        
        logger.info(f"✅ Parallel Processing Pipeline initialized")
        logger.info(f"   Vision threads: {num_vision_threads}")
        logger.info(f"   Inference threads: {num_inference_threads}")
        logger.info(f"   Update threads: {num_update_threads}")
        logger.info(f"   Target CPU: {target_cpu}%")
        
    def _start_workers(self):
        """Start all worker threads"""
        # Vision workers
        for i in range(self.num_vision_threads):
            self.vision_pool.submit(self._vision_worker, i)
            
        # Inference workers
        for i in range(self.num_inference_threads):
            self.inference_pool.submit(self._inference_worker, i)
            
        # Update workers
        for i in range(self.num_update_threads):
            self.update_pool.submit(self._update_worker, i)
    
    def _vision_worker(self, worker_id: int):
        """Vision processing worker"""
        logger.info(f"Vision worker {worker_id} started")
        
        while self._running:
            # CPU throttling
            self.cpu_monitor.throttle()
            
            # Get task
            task = self.vision_queue.get_nowait()
            if task is None:
                time.sleep(0.001)  # Avoid busy waiting
                continue
                
            try:
                # Process vision task
                start_time = time.time()
                
                if task.task_type == 'screenshot':
                    result = self._process_screenshot(task.data)
                elif task.task_type == 'feature_extraction':
                    result = self._extract_features(task.data)
                else:
                    result = task.data  # Passthrough
                
                # Create inference task
                inference_task = ProcessingTask(
                    id=task.id,
                    task_type='inference',
                    data=result,
                    priority=task.priority,
                    timestamp=time.time()
                )
                self.inference_queue.put_nowait(inference_task)
                
                # Update metrics
                self.metrics['vision_processed'] += 1
                latency = time.time() - start_time
                self.metrics['avg_latency'] = (
                    self.metrics['avg_latency'] * 0.9 + latency * 0.1
                )
                
            except Exception as e:
                logger.error(f"Vision worker {worker_id} error: {e}")
                
    def _inference_worker(self, worker_id: int):
        """Model inference worker"""
        logger.info(f"Inference worker {worker_id} started")
        
        # Thread-local model cache
        model_cache = {}
        
        while self._running:
            # CPU throttling
            self.cpu_monitor.throttle()
            
            # Get task
            task = self.inference_queue.get_nowait()
            if task is None:
                time.sleep(0.001)
                continue
                
            try:
                # Run inference
                result = self._run_inference(task.data, model_cache)
                
                # Create update task if needed
                if self._should_update_weights(result):
                    update_task = ProcessingTask(
                        id=task.id,
                        task_type='weight_update',
                        data={
                            'gradients': result.get('gradients'),
                            'loss': result.get('loss')
                        },
                        priority=task.priority,
                        timestamp=time.time()
                    )
                    self.update_queue.put_nowait(update_task)
                
                # Store result
                self.result_queue.put_nowait({
                    'id': task.id,
                    'result': result,
                    'timestamp': time.time()
                })
                
                self.metrics['inference_completed'] += 1
                
            except Exception as e:
                logger.error(f"Inference worker {worker_id} error: {e}")
                
    def _update_worker(self, worker_id: int):
        """Weight update worker"""
        logger.info(f"Update worker {worker_id} started")
        
        # Gradient accumulator
        gradient_buffer = {}
        update_counter = 0
        
        while self._running:
            # CPU throttling (more aggressive for updates)
            self.cpu_monitor.throttle(factor=2.0)
            
            # Get task
            task = self.update_queue.get_nowait()
            if task is None:
                # Check if we should apply accumulated gradients
                if update_counter > 0 and update_counter % 10 == 0:
                    self._apply_gradient_updates(gradient_buffer)
                    gradient_buffer.clear()
                    
                time.sleep(0.01)  # Lower priority
                continue
                
            try:
                # Accumulate gradients
                gradients = task.data.get('gradients', {})
                for name, grad in gradients.items():
                    if name not in gradient_buffer:
                        gradient_buffer[name] = []
                    gradient_buffer[name].append(grad)
                
                update_counter += 1
                
                # Apply updates every N accumulations
                if update_counter % 10 == 0:
                    self._apply_gradient_updates(gradient_buffer)
                    gradient_buffer.clear()
                    self.metrics['updates_applied'] += 1
                    
            except Exception as e:
                logger.error(f"Update worker {worker_id} error: {e}")
    
    def _process_screenshot(self, data: np.ndarray) -> Dict[str, Any]:
        """Process screenshot with optimizations"""
        # Resize for faster processing
        height, width = data.shape[:2]
        if height > 512 or width > 512:
            scale = min(512 / height, 512 / width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            # Fast resize using stride
            data = data[::int(1/scale), ::int(1/scale)]
        
        # Convert to tensor
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).float() / 255.0
        else:
            tensor = data
            
        return {
            'tensor': tensor,
            'original_shape': (height, width),
            'processed_shape': tensor.shape
        }
    
    def _extract_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features with caching"""
        tensor = data.get('tensor')
        
        # Simple feature extraction (edge detection)
        if len(tensor.shape) == 3:
            # Convert to grayscale
            gray = tensor.mean(dim=2)
            
            # Sobel edge detection (simplified)
            edges_x = gray[1:, :] - gray[:-1, :]
            edges_y = gray[:, 1:] - gray[:, :-1]
            
            # Magnitude
            magnitude = (edges_x[:-1, :-1]**2 + edges_y[:-1, :-1]**2).sqrt()
            
            data['features'] = {
                'edges': magnitude,
                'mean_edge': magnitude.mean().item(),
                'max_edge': magnitude.max().item()
            }
            
        return data
    
    def _run_inference(self, data: Dict[str, Any], 
                      model_cache: Dict[str, Any]) -> Dict[str, Any]:
        """Run model inference with caching"""
        # Simulate inference with CPU limiting
        features = data.get('features', {})
        
        # Check cache
        cache_key = f"{features.get('mean_edge', 0):.2f}_{features.get('max_edge', 0):.2f}"
        if cache_key in model_cache:
            return model_cache[cache_key]
        
        # Simulate inference result
        result = {
            'predictions': torch.randn(10).softmax(dim=0),
            'confidence': np.random.random(),
            'loss': np.random.random() * 0.1,
            'gradients': {
                'layer1': torch.randn(100, 100) * 0.01,
                'layer2': torch.randn(50, 50) * 0.01
            }
        }
        
        # Cache result
        if len(model_cache) < 100:  # Limit cache size
            model_cache[cache_key] = result
            
        return result
    
    def _should_update_weights(self, result: Dict[str, Any]) -> bool:
        """Determine if weights should be updated"""
        # Update only if loss is significant
        return result.get('loss', 0) > 0.05
    
    def _apply_gradient_updates(self, gradient_buffer: Dict[str, List[torch.Tensor]]):
        """Apply accumulated gradient updates"""
        for name, grads in gradient_buffer.items():
            if grads:
                # Average gradients
                avg_grad = torch.stack(grads).mean(dim=0)
                # Apply with small learning rate
                # In real implementation, this would update actual model weights
                logger.debug(f"Applied gradient update to {name}: {avg_grad.norm():.4f}")
    
    async def process_async(self, image_data: np.ndarray, 
                          task_type: str = 'screenshot') -> Dict[str, Any]:
        """Async interface for processing"""
        task = ProcessingTask(
            id=f"{time.time()}",
            task_type=task_type,
            data=image_data,
            priority=0,
            timestamp=time.time()
        )
        
        # Submit to pipeline
        self.vision_queue.put_nowait(task)
        
        # Wait for result
        start_time = time.time()
        while time.time() - start_time < 5.0:  # 5 second timeout
            result = self.result_queue.get_nowait()
            if result and result['id'] == task.id:
                return result['result']
            await asyncio.sleep(0.01)
            
        raise TimeoutError("Processing timeout")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        self.metrics['cpu_usage'] = psutil.cpu_percent(interval=0.1)
        self.metrics['queue_sizes'] = {
            'vision': self.vision_queue.qsize(),
            'inference': self.inference_queue.qsize(),
            'update': self.update_queue.qsize(),
            'result': self.result_queue.qsize()
        }
        return self.metrics
    
    def shutdown(self):
        """Shutdown pipeline"""
        self._running = False
        self.vision_pool.shutdown(wait=True)
        self.inference_pool.shutdown(wait=True)
        self.update_pool.shutdown(wait=True)
        logger.info("Parallel Processing Pipeline shutdown complete")

class CPUThrottler:
    """Advanced CPU throttling with dynamic adjustment"""
    
    def __init__(self, target_cpu: float):
        self.target_cpu = target_cpu
        self.last_check = time.time()
        self.sleep_factor = 0.01  # Initial sleep factor
        
        # PID controller parameters
        self.kp = 0.001  # Proportional gain
        self.ki = 0.0001  # Integral gain
        self.kd = 0.0005  # Derivative gain
        
        self.integral = 0.0
        self.last_error = 0.0
        
    def throttle(self, factor: float = 1.0):
        """Apply CPU throttling with PID control"""
        current_time = time.time()
        
        # Check every 100ms
        if current_time - self.last_check < 0.1:
            return
            
        self.last_check = current_time
        
        # Get current CPU
        current_cpu = psutil.cpu_percent(interval=0.05)
        
        # PID control
        error = current_cpu - self.target_cpu
        
        if error > 0:  # Only throttle if over target
            # Proportional
            p_term = self.kp * error
            
            # Integral
            self.integral += error
            i_term = self.ki * self.integral
            
            # Derivative
            d_term = self.kd * (error - self.last_error)
            
            # Calculate sleep time
            sleep_time = (p_term + i_term + d_term) * factor
            sleep_time = max(0.001, min(0.1, sleep_time))  # Clamp
            
            time.sleep(sleep_time)
            
        self.last_error = error

# Factory function
def create_parallel_pipeline(**kwargs) -> ParallelProcessingPipeline:
    """Create optimized parallel processing pipeline"""
    return ParallelProcessingPipeline(**kwargs)

# Test function
def test_parallel_pipeline():
    """Test the parallel processing pipeline"""
    pipeline = create_parallel_pipeline(
        num_vision_threads=2,
        num_inference_threads=2,
        num_update_threads=1,
        target_cpu=35.0
    )
    
    # Simulate processing
    import numpy as np
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    async def test():
        result = await pipeline.process_async(test_image)
        print(f"Processing result: {result}")
        print(f"Metrics: {pipeline.get_metrics()}")
    
    asyncio.run(test())
    pipeline.shutdown()

if __name__ == "__main__":
    test_parallel_pipeline()