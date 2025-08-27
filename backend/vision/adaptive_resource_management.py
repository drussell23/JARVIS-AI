"""
Phase 3: Adaptive Resource Management
ML-based workload prediction and preemptive resource allocation
Dynamic frequency scaling and thermal awareness
Target: Further reduce CPU to 25%
"""

import torch
import torch.nn as nn
import numpy as np
import psutil
import platform
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import threading
from enum import Enum
import subprocess
import os

logger = logging.getLogger(__name__)


class WorkloadLevel(Enum):
    """Workload intensity levels"""
    IDLE = 0
    LIGHT = 1
    MODERATE = 2
    HEAVY = 3
    CRITICAL = 4


@dataclass
class SystemMetrics:
    """Current system metrics"""
    cpu_percent: float
    memory_percent: float
    temperature: float
    io_wait: float
    process_count: int
    timestamp: float
    
    # Derived metrics
    cpu_trend: float = 0.0
    memory_pressure: float = 0.0
    thermal_state: str = "normal"


@dataclass
class ResourcePrediction:
    """ML model prediction output"""
    next_cpu: float  # Predicted CPU in next window
    next_memory: float
    workload_level: WorkloadLevel
    confidence: float
    recommended_threads: int
    recommended_frequency: float


class WorkloadPredictor(nn.Module):
    """LSTM-based workload prediction model"""
    
    def __init__(self, input_size: int = 5, hidden_size: int = 32, 
                 sequence_length: int = 10):
        super().__init__()
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        
        # LSTM for time series prediction
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Output layers
        self.cpu_predictor = nn.Linear(hidden_size, 1)
        self.memory_predictor = nn.Linear(hidden_size, 1)
        self.workload_classifier = nn.Linear(hidden_size, 5)  # 5 workload levels
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # x shape: (batch, sequence_length, features)
        lstm_out, _ = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Predictions
        cpu_pred = torch.sigmoid(self.cpu_predictor(last_hidden)) * 100  # 0-100%
        mem_pred = torch.sigmoid(self.memory_predictor(last_hidden)) * 100
        workload_logits = self.workload_classifier(last_hidden)
        
        return cpu_pred, mem_pred, workload_logits


class ThermalManager:
    """Manage thermal throttling and frequency scaling"""
    
    def __init__(self):
        self.platform = platform.system()
        self.cpu_count = psutil.cpu_count()
        self.base_frequency = self._get_base_frequency()
        
        # Thermal thresholds
        self.thermal_thresholds = {
            'normal': 60,    # °C
            'warm': 70,
            'hot': 80,
            'critical': 90
        }
        
        logger.info(f"Thermal Manager initialized for {self.platform}")
        logger.info(f"CPU count: {self.cpu_count}, Base freq: {self.base_frequency}MHz")
    
    def _get_base_frequency(self) -> float:
        """Get CPU base frequency"""
        try:
            if self.platform == "Darwin":  # macOS
                # Use sysctl to get CPU frequency
                result = subprocess.run(['sysctl', '-n', 'hw.cpufrequency_max'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    return float(result.stdout.strip()) / 1e6  # Convert to MHz
            elif self.platform == "Linux":
                with open('/sys/devices/system/cpu/cpu0/cpufreq/base_frequency', 'r') as f:
                    return float(f.read().strip()) / 1000  # KHz to MHz
        except:
            pass
        return 2000.0  # Default 2GHz
    
    def get_temperature(self) -> float:
        """Get current CPU temperature"""
        try:
            if self.platform == "Darwin":
                # macOS temperature monitoring (requires osx-cpu-temp)
                try:
                    result = subprocess.run(['osx-cpu-temp'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        temp_str = result.stdout.strip()
                        # Parse temperature from output
                        temp = float(temp_str.split('°')[0])
                        return temp
                except:
                    # Fallback: use powermetrics (requires sudo)
                    pass
            elif self.platform == "Linux":
                # Linux thermal zones
                temps = []
                for zone in os.listdir('/sys/class/thermal/'):
                    if zone.startswith('thermal_zone'):
                        try:
                            with open(f'/sys/class/thermal/{zone}/temp', 'r') as f:
                                temp = float(f.read().strip()) / 1000  # mC to C
                                temps.append(temp)
                        except:
                            continue
                if temps:
                    return max(temps)
            
            # Use psutil if available
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    for entry in entries:
                        if entry.label and 'core' in entry.label.lower():
                            return entry.current
        except Exception as e:
            logger.debug(f"Temperature reading failed: {e}")
        
        # Return estimated temperature based on CPU usage
        return 50.0 + psutil.cpu_percent() * 0.4
    
    def get_thermal_state(self, temperature: float) -> str:
        """Determine thermal state"""
        if temperature < self.thermal_thresholds['normal']:
            return 'normal'
        elif temperature < self.thermal_thresholds['warm']:
            return 'warm'
        elif temperature < self.thermal_thresholds['hot']:
            return 'hot'
        elif temperature < self.thermal_thresholds['critical']:
            return 'critical'
        else:
            return 'emergency'
    
    def set_cpu_frequency(self, frequency_mhz: float):
        """Set CPU frequency scaling"""
        try:
            if self.platform == "Linux":
                # Linux cpufreq governor
                for cpu in range(self.cpu_count):
                    # Set scaling governor to userspace
                    with open(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor', 'w') as f:
                        f.write('userspace\n')
                    # Set frequency
                    freq_khz = int(frequency_mhz * 1000)
                    with open(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_setspeed', 'w') as f:
                        f.write(f'{freq_khz}\n')
                logger.info(f"Set CPU frequency to {frequency_mhz}MHz")
            elif self.platform == "Darwin":
                # macOS doesn't allow direct frequency control
                # Use process priority instead
                nice_value = int((self.base_frequency - frequency_mhz) / 100)
                os.nice(nice_value)
                logger.info(f"Adjusted process priority (nice={nice_value})")
        except Exception as e:
            logger.debug(f"Frequency scaling not available: {e}")
    
    def apply_thermal_throttling(self, thermal_state: str) -> float:
        """Apply thermal throttling based on state"""
        throttle_factors = {
            'normal': 1.0,
            'warm': 0.9,
            'hot': 0.7,
            'critical': 0.5,
            'emergency': 0.3
        }
        
        factor = throttle_factors.get(thermal_state, 1.0)
        target_freq = self.base_frequency * factor
        
        self.set_cpu_frequency(target_freq)
        return factor


class AdaptiveResourceManager:
    """
    Main resource management system with:
    - ML-based workload prediction
    - Preemptive resource allocation
    - Dynamic frequency scaling
    - Thermal throttling awareness
    """
    
    def __init__(self, target_cpu: float = 25.0):
        self.target_cpu = target_cpu
        
        # Initialize components
        self.predictor = WorkloadPredictor()
        self.thermal_manager = ThermalManager()
        
        # Metrics history
        self.metrics_history = deque(maxlen=100)
        self.prediction_history = deque(maxlen=50)
        
        # Resource allocation
        self.current_threads = psutil.cpu_count() // 2
        self.max_threads = psutil.cpu_count()
        self.min_threads = 1
        
        # Control loop
        self._running = True
        self._lock = threading.RLock()
        self._monitor_thread = None
        
        # Pretrained weights (simulated)
        self._load_pretrained_weights()
        
        logger.info("✅ Adaptive Resource Manager initialized")
        logger.info(f"   Target CPU: {target_cpu}%")
        logger.info(f"   Thread range: {self.min_threads}-{self.max_threads}")
    
    def _load_pretrained_weights(self):
        """Load pretrained model weights (simulated)"""
        # In production, load from checkpoint
        # For now, use random initialized weights
        self.predictor.eval()
    
    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        temperature = self.thermal_manager.get_temperature()
        
        # Calculate IO wait (approximation)
        io_counters = psutil.disk_io_counters()
        io_wait = (io_counters.read_time + io_counters.write_time) / 1000 / 60  # Convert to percentage
        
        # CPU trend
        cpu_trend = 0.0
        if len(self.metrics_history) > 5:
            recent_cpu = [m.cpu_percent for m in list(self.metrics_history)[-5:]]
            cpu_trend = (recent_cpu[-1] - np.mean(recent_cpu[:-1])) / max(1, np.mean(recent_cpu[:-1]))
        
        metrics = SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            temperature=temperature,
            io_wait=min(io_wait, 100.0),
            process_count=len(psutil.pids()),
            timestamp=time.time(),
            cpu_trend=cpu_trend,
            memory_pressure=memory.percent / 100 * (1 + cpu_trend),
            thermal_state=self.thermal_manager.get_thermal_state(temperature)
        )
        
        with self._lock:
            self.metrics_history.append(metrics)
        
        return metrics
    
    def predict_workload(self, lookback_window: int = 10) -> ResourcePrediction:
        """Predict future workload using ML model"""
        with self._lock:
            if len(self.metrics_history) < lookback_window:
                # Not enough history, return conservative prediction
                return ResourcePrediction(
                    next_cpu=self.target_cpu,
                    next_memory=50.0,
                    workload_level=WorkloadLevel.MODERATE,
                    confidence=0.5,
                    recommended_threads=self.current_threads,
                    recommended_frequency=self.thermal_manager.base_frequency * 0.8
                )
            
            # Prepare input features
            recent_metrics = list(self.metrics_history)[-lookback_window:]
            features = []
            
            for m in recent_metrics:
                features.append([
                    m.cpu_percent / 100,
                    m.memory_percent / 100,
                    m.temperature / 100,
                    m.io_wait / 100,
                    m.process_count / 1000
                ])
            
            # Convert to tensor
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            
            # Predict
            with torch.no_grad():
                cpu_pred, mem_pred, workload_logits = self.predictor(x)
                
                # Get predictions
                next_cpu = cpu_pred.item()
                next_memory = mem_pred.item()
                workload_probs = torch.softmax(workload_logits, dim=-1)
                workload_level = WorkloadLevel(workload_probs.argmax().item())
                confidence = workload_probs.max().item()
            
            # Calculate recommended resources
            cpu_ratio = next_cpu / self.target_cpu
            if cpu_ratio > 1.5:
                recommended_threads = min(self.max_threads, self.current_threads + 2)
                recommended_frequency = self.thermal_manager.base_frequency
            elif cpu_ratio > 1.2:
                recommended_threads = min(self.max_threads, self.current_threads + 1)
                recommended_frequency = self.thermal_manager.base_frequency * 0.9
            elif cpu_ratio < 0.5:
                recommended_threads = max(self.min_threads, self.current_threads - 1)
                recommended_frequency = self.thermal_manager.base_frequency * 0.6
            else:
                recommended_threads = self.current_threads
                recommended_frequency = self.thermal_manager.base_frequency * 0.8
            
            prediction = ResourcePrediction(
                next_cpu=next_cpu,
                next_memory=next_memory,
                workload_level=workload_level,
                confidence=confidence,
                recommended_threads=recommended_threads,
                recommended_frequency=recommended_frequency
            )
            
            self.prediction_history.append(prediction)
            return prediction
    
    def allocate_resources(self, prediction: ResourcePrediction):
        """Allocate resources based on prediction"""
        logger.info(f"Resource allocation - Predicted CPU: {prediction.next_cpu:.1f}%, "
                   f"Workload: {prediction.workload_level.name}")
        
        # Update thread count
        if prediction.recommended_threads != self.current_threads:
            self.current_threads = prediction.recommended_threads
            logger.info(f"Adjusted thread count to {self.current_threads}")
            
            # Notify thread pools (in real implementation)
            # self.thread_pool.resize(self.current_threads)
        
        # Apply frequency scaling
        current_metrics = self.metrics_history[-1] if self.metrics_history else None
        if current_metrics:
            # Apply thermal throttling
            thermal_factor = self.thermal_manager.apply_thermal_throttling(
                current_metrics.thermal_state
            )
            
            # Adjust frequency based on prediction and thermal state
            target_frequency = prediction.recommended_frequency * thermal_factor
            self.thermal_manager.set_cpu_frequency(target_frequency)
    
    def start_monitoring(self):
        """Start resource monitoring loop"""
        def monitor_loop():
            logger.info("Starting adaptive resource monitoring")
            
            while self._running:
                try:
                    # Collect metrics
                    metrics = self.collect_metrics()
                    
                    # Predict workload
                    prediction = self.predict_workload()
                    
                    # Allocate resources preemptively
                    self.allocate_resources(prediction)
                    
                    # Log status
                    if len(self.metrics_history) % 10 == 0:
                        logger.info(f"Current: CPU={metrics.cpu_percent:.1f}%, "
                                   f"Temp={metrics.temperature:.1f}°C, "
                                   f"Predicted: CPU={prediction.next_cpu:.1f}%, "
                                   f"Threads={self.current_threads}")
                    
                    # Adaptive sleep based on workload
                    sleep_time = 0.5 if prediction.workload_level == WorkloadLevel.IDLE else 0.1
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(1)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring loop"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get current resource statistics"""
        current_metrics = self.metrics_history[-1] if self.metrics_history else None
        latest_prediction = self.prediction_history[-1] if self.prediction_history else None
        
        stats = {
            'current_cpu': current_metrics.cpu_percent if current_metrics else 0,
            'target_cpu': self.target_cpu,
            'current_threads': self.current_threads,
            'temperature': current_metrics.temperature if current_metrics else 0,
            'thermal_state': current_metrics.thermal_state if current_metrics else 'unknown',
            'predicted_cpu': latest_prediction.next_cpu if latest_prediction else 0,
            'workload_level': latest_prediction.workload_level.name if latest_prediction else 'UNKNOWN',
            'prediction_confidence': latest_prediction.confidence if latest_prediction else 0
        }
        
        return stats


# Test function
def test_adaptive_resource_management():
    """Test adaptive resource management"""
    logger.info("=" * 60)
    logger.info("TESTING ADAPTIVE RESOURCE MANAGEMENT")
    logger.info("=" * 60)
    
    manager = AdaptiveResourceManager(target_cpu=25.0)
    manager.start_monitoring()
    
    # Simulate workload changes
    import random
    
    logger.info("\nSimulating variable workload...")
    for i in range(20):
        # Simulate CPU load
        load_level = random.choice([0.1, 0.3, 0.5, 0.7, 0.9])
        threads = []
        
        # Create CPU load
        def cpu_burn(duration):
            end_time = time.time() + duration
            while time.time() < end_time:
                _ = sum(i**2 for i in range(1000))
        
        for _ in range(int(psutil.cpu_count() * load_level)):
            t = threading.Thread(target=cpu_burn, args=(0.5,))
            t.start()
            threads.append(t)
        
        # Wait and collect stats
        time.sleep(1)
        stats = manager.get_resource_stats()
        
        logger.info(f"\nIteration {i+1}:")
        logger.info(f"  Current CPU: {stats['current_cpu']:.1f}%")
        logger.info(f"  Predicted CPU: {stats['predicted_cpu']:.1f}%")
        logger.info(f"  Workload: {stats['workload_level']}")
        logger.info(f"  Threads: {stats['current_threads']}")
        logger.info(f"  Temperature: {stats['temperature']:.1f}°C ({stats['thermal_state']})")
        
        # Wait for threads to finish
        for t in threads:
            t.join()
    
    manager.stop_monitoring()
    logger.info("\n✅ Adaptive resource management test complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_adaptive_resource_management()