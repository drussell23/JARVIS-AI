"""
Video Stream Capture Module for JARVIS Vision System
Real-time video capture with memory-safe processing and sliding window support
Designed for 16GB macOS systems with intelligent memory management
"""

import asyncio
import time
import os
import gc
import logging
import queue
import threading
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import weakref

import numpy as np
import psutil
from PIL import Image

# Try to import macOS-specific modules
try:
    import AVFoundation
    import CoreMedia
    from Quartz import CoreVideo
    from Cocoa import NSObject
    import objc
    from Foundation import NSRunLoop
    import libdispatch
    MACOS_CAPTURE_AVAILABLE = True
except ImportError as e:
    MACOS_CAPTURE_AVAILABLE = False
    logging.warning(f"macOS capture frameworks not available - will use fallback: {e}")

# Try to import alternative capture methods
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class VideoStreamConfig:
    """Configuration for video stream capture - NO HARDCODING"""
    # Stream settings
    target_fps: int = field(default_factory=lambda: int(os.getenv('VIDEO_STREAM_FPS', '30')))
    resolution: str = field(default_factory=lambda: os.getenv('VIDEO_STREAM_RESOLUTION', '1920x1080'))
    capture_display_id: int = field(default_factory=lambda: int(os.getenv('VIDEO_STREAM_DISPLAY_ID', '0')))
    
    # Memory management
    max_frame_buffer_size: int = field(default_factory=lambda: int(os.getenv('VIDEO_STREAM_BUFFER_SIZE', '10')))
    memory_limit_mb: int = field(default_factory=lambda: int(os.getenv('VIDEO_STREAM_MEMORY_LIMIT_MB', '800')))  # 800MB for video
    frame_memory_threshold_mb: int = field(default_factory=lambda: int(os.getenv('VIDEO_STREAM_FRAME_THRESHOLD_MB', '50')))
    
    # Processing settings
    enable_sliding_window: bool = field(default_factory=lambda: os.getenv('VIDEO_STREAM_SLIDING_WINDOW', 'true').lower() == 'true')
    sliding_window_size: str = field(default_factory=lambda: os.getenv('VIDEO_STREAM_WINDOW_SIZE', '640x480'))
    sliding_window_overlap: float = field(default_factory=lambda: float(os.getenv('VIDEO_STREAM_WINDOW_OVERLAP', '0.2')))
    max_windows_per_frame: int = field(default_factory=lambda: int(os.getenv('VIDEO_STREAM_MAX_WINDOWS', '4')))
    
    # Analysis settings
    analyze_every_n_frames: int = field(default_factory=lambda: int(os.getenv('VIDEO_STREAM_ANALYZE_INTERVAL', '30')))  # Analyze every 1 second at 30fps
    enable_motion_detection: bool = field(default_factory=lambda: os.getenv('VIDEO_STREAM_MOTION_DETECTION', 'true').lower() == 'true')
    motion_threshold: float = field(default_factory=lambda: float(os.getenv('VIDEO_STREAM_MOTION_THRESHOLD', '0.1')))
    
    # Adaptive quality
    enable_adaptive_quality: bool = field(default_factory=lambda: os.getenv('VIDEO_STREAM_ADAPTIVE', 'true').lower() == 'true')
    min_fps: int = field(default_factory=lambda: int(os.getenv('VIDEO_STREAM_MIN_FPS', '10')))
    min_resolution: str = field(default_factory=lambda: os.getenv('VIDEO_STREAM_MIN_RES', '960x540'))

@dataclass
class FrameMetrics:
    """Metrics for frame processing"""
    timestamp: float
    frame_number: int
    processing_time: float
    memory_usage_mb: float
    motion_score: float
    windows_processed: int
    analysis_triggered: bool

class MemorySafeFrameBuffer:
    """Thread-safe circular buffer for video frames with memory management"""
    
    def __init__(self, max_frames: int, memory_limit_mb: int):
        self.max_frames = max_frames
        self.memory_limit_mb = memory_limit_mb
        self.frames = []
        self.lock = threading.Lock()
        self.total_memory_bytes = 0
        self.frame_counter = 0
        
    def add_frame(self, frame: np.ndarray) -> bool:
        """Add frame with memory checking"""
        frame_size = frame.nbytes
        
        with self.lock:
            # Check memory before adding
            while (self.total_memory_bytes + frame_size > self.memory_limit_mb * 1024 * 1024 or
                   len(self.frames) >= self.max_frames) and self.frames:
                # Remove oldest frame
                old_frame = self.frames.pop(0)
                self.total_memory_bytes -= old_frame['size']
            
            # Add new frame
            self.frames.append({
                'data': frame,
                'timestamp': time.time(),
                'frame_number': self.frame_counter,
                'size': frame_size
            })
            self.total_memory_bytes += frame_size
            self.frame_counter += 1
            
            return True
    
    def get_latest_frame(self) -> Optional[Dict[str, Any]]:
        """Get most recent frame"""
        with self.lock:
            return self.frames[-1] if self.frames else None
    
    def get_frames_for_analysis(self, count: int = 1) -> List[Dict[str, Any]]:
        """Get multiple frames for analysis"""
        with self.lock:
            return self.frames[-count:] if len(self.frames) >= count else self.frames.copy()
    
    def clear(self):
        """Clear all frames"""
        with self.lock:
            self.frames.clear()
            self.total_memory_bytes = 0
            gc.collect()

class MacOSVideoCapture:
    """Native macOS video capture using AVFoundation"""
    
    def __init__(self, config: VideoStreamConfig):
        self.config = config
        self.session = None
        self.output = None
        self.is_running = False
        self.frame_callback = None
        
        if not MACOS_CAPTURE_AVAILABLE:
            raise ImportError("macOS capture frameworks not available")
    
    def start_capture(self, frame_callback: Callable):
        """Start video capture session"""
        logger.info("MacOSVideoCapture.start_capture called")
        self.frame_callback = frame_callback
        
        try:
            # Create capture session
            logger.info("[MACOS] Creating AVCaptureSession...")
            self.session = AVFoundation.AVCaptureSession.alloc().init()
            logger.info("Created AVCaptureSession")
            
            # Configure session
            logger.info(f"[MACOS] Setting resolution preset for {self.config.resolution}")
            if self.config.resolution == '1920x1080':
                self.session.setSessionPreset_(AVFoundation.AVCaptureSessionPreset1920x1080)
            elif self.config.resolution == '1280x720':
                self.session.setSessionPreset_(AVFoundation.AVCaptureSessionPreset1280x720)
            else:
                self.session.setSessionPreset_(AVFoundation.AVCaptureSessionPreset640x480)
            
            # Create screen input
            display_id = self.config.capture_display_id
            logger.info(f"[MACOS] Creating screen input for display ID {display_id}")
            screen_input = AVFoundation.AVCaptureScreenInput.alloc().initWithDisplayID_(display_id)
        except Exception as e:
            logger.error(f"[MACOS] Error in start_capture: {e}", exc_info=True)
            raise
        
        if screen_input:
            # Configure capture settings
            screen_input.setMinFrameDuration_(CoreMedia.CMTimeMake(1, self.config.target_fps))
            screen_input.setCapturesCursor_(False)
            screen_input.setCapturesMouseClicks_(False)
            
            # Add input to session
            if self.session.canAddInput_(screen_input):
                self.session.addInput_(screen_input)
            
            # Create output
            self.output = AVFoundation.AVCaptureVideoDataOutput.alloc().init()
            self.output.setAlwaysDiscardsLateVideoFrames_(True)
            
            # Configure output pixel format
            self.output.setVideoSettings_({
                CoreVideo.kCVPixelBufferPixelFormatTypeKey: CoreVideo.kCVPixelFormatType_32BGRA
            })
            
            # Set delegate (frame callback)
            delegate = VideoFrameDelegate.alloc().initWithCallback_(self._handle_frame)
            # Create a serial dispatch queue for video processing
            queue = libdispatch.dispatch_queue_create(b"com.jarvis.videoqueue", None)
            self.output.setSampleBufferDelegate_queue_(delegate, queue)
            
            # Add output to session
            if self.session.canAddOutput_(self.output):
                self.session.addOutput_(self.output)
            
            # Start capture
            logger.info("Starting capture session...")
            self.session.startRunning()
            self.is_running = True
            logger.info("Started macOS video capture - purple indicator should be visible")
        else:
            logger.error("Failed to create screen input")
            raise RuntimeError("Failed to create screen input")
    
    def _handle_frame(self, sample_buffer):
        """Handle captured frame"""
        if self.frame_callback:
            # Convert CMSampleBuffer to numpy array
            image_buffer = CoreMedia.CMSampleBufferGetImageBuffer(sample_buffer)
            if image_buffer:
                CoreVideo.CVPixelBufferLockBaseAddress(image_buffer, 0)
                
                base_address = CoreVideo.CVPixelBufferGetBaseAddress(image_buffer)
                bytes_per_row = CoreVideo.CVPixelBufferGetBytesPerRow(image_buffer)
                height = CoreVideo.CVPixelBufferGetHeight(image_buffer)
                width = CoreVideo.CVPixelBufferGetWidth(image_buffer)
                
                # Create numpy array from pixel data
                frame = np.frombuffer(base_address, dtype=np.uint8)
                frame = frame.reshape((height, bytes_per_row // 4, 4))
                frame = frame[:, :width, :3]  # Remove alpha channel
                frame = frame[:, :, ::-1]  # BGRA to RGB
                
                CoreVideo.CVPixelBufferUnlockBaseAddress(image_buffer, 0)
                
                # Call the callback
                self.frame_callback(frame)
    
    def stop_capture(self):
        """Stop video capture"""
        if self.session and self.is_running:
            self.session.stopRunning()
            self.is_running = False
            logger.info("Stopped macOS video capture")

# Only define VideoFrameDelegate if macOS frameworks are available
if MACOS_CAPTURE_AVAILABLE:
    class VideoFrameDelegate(NSObject):
        """Delegate for handling video frames"""
        
        def initWithCallback_(self, callback):
            self = objc.super(VideoFrameDelegate, self).init()
            if self:
                self.callback = callback
            return self
        
        def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sample_buffer, connection):
            """Handle frame capture"""
            self.callback(sample_buffer)
else:
    # Placeholder class when macOS frameworks aren't available
    class VideoFrameDelegate:
        """Placeholder delegate for non-macOS systems"""
        
        def __init__(self):
            raise NotImplementedError("VideoFrameDelegate requires macOS frameworks")

class VideoStreamCapture:
    """Main video stream capture manager with memory-safe processing"""
    
    def __init__(self, vision_analyzer, config: Optional[VideoStreamConfig] = None):
        self.vision_analyzer = vision_analyzer
        self.config = config or VideoStreamConfig()
        
        # State
        self.is_capturing = False
        self.capture_thread = None
        self.process_thread = None
        
        # Frame buffer
        self.frame_buffer = MemorySafeFrameBuffer(
            self.config.max_frame_buffer_size,
            self.config.memory_limit_mb
        )
        
        # Metrics
        self.metrics = []
        self.frames_processed = 0
        self.frames_analyzed = 0
        
        # Motion detection
        self.previous_frame = None
        self.motion_regions = []
        
        # Callbacks
        self.event_callbacks = {
            'frame_captured': weakref.WeakSet(),
            'frame_analyzed': weakref.WeakSet(),
            'motion_detected': weakref.WeakSet(),
            'memory_warning': weakref.WeakSet()
        }
        
        # Platform-specific capture
        self.capture_impl = None
        
        logger.info(f"Video Stream Capture initialized with config: {self.config}")
    
    async def start_streaming(self) -> bool:
        """Start video stream capture"""
        logger.info("[VIDEO] start_streaming called")
        
        if self.is_capturing:
            logger.warning("Video capture already running")
            return False
        
        try:
            # Check memory before starting
            logger.info(f"[VIDEO] Checking memory availability...")
            if not self._check_memory_available():
                logger.error("Insufficient memory for video streaming")
                return False
            logger.info(f"[VIDEO] Memory check passed")
            
            # Initialize capture implementation
            if MACOS_CAPTURE_AVAILABLE:
                logger.info("Using native macOS video capture")
                logger.info(f"[VIDEO] MACOS_CAPTURE_AVAILABLE = {MACOS_CAPTURE_AVAILABLE}")
                logger.info(f"[VIDEO] Creating MacOSVideoCapture instance...")
                self.capture_impl = MacOSVideoCapture(self.config)
                logger.info(f"[VIDEO] Starting capture with callback...")
                self.capture_impl.start_capture(self._on_frame_captured)
                logger.info("MacOS video capture started successfully")
            elif CV2_AVAILABLE:
                # Fallback to OpenCV
                logger.info(f"[VIDEO] Using OpenCV fallback (CV2_AVAILABLE = {CV2_AVAILABLE})")
                self._start_cv2_capture()
            else:
                # Final fallback to screenshot loop
                logger.info("[VIDEO] Using screenshot loop fallback")
                self._start_screenshot_loop()
            
            self.is_capturing = True
            
            # Start processing thread
            self.process_thread = threading.Thread(target=self._process_frames_loop)
            self.process_thread.daemon = True
            self.process_thread.start()
            
            logger.info("Video streaming started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start video streaming: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            return False
    
    def _on_frame_captured(self, frame: np.ndarray):
        """Handle captured frame"""
        try:
            # Add to buffer
            self.frame_buffer.add_frame(frame)
            self.frames_processed += 1
            
            # Trigger callback
            asyncio.create_task(self._trigger_event('frame_captured', {
                'frame_number': self.frames_processed,
                'timestamp': time.time()
            }))
        except Exception as e:
            logger.error(f"Error in _on_frame_captured: {e}", exc_info=True)
    
    def _process_frames_loop(self):
        """Process frames in separate thread"""
        frame_count = 0
        
        while self.is_capturing:
            try:
                # Get latest frame
                frame_data = self.frame_buffer.get_latest_frame()
                if not frame_data:
                    time.sleep(0.033)  # ~30fps
                    continue
                
                frame = frame_data['data']
                frame_count += 1
                
                # Check if we should analyze this frame
                should_analyze = False
                
                # 1. Periodic analysis
                if frame_count % self.config.analyze_every_n_frames == 0:
                    should_analyze = True
                
                # 2. Motion-triggered analysis
                if self.config.enable_motion_detection:
                    motion_score = self._detect_motion(frame)
                    if motion_score > self.config.motion_threshold:
                        should_analyze = True
                        asyncio.create_task(self._trigger_event('motion_detected', {
                            'motion_score': motion_score,
                            'regions': self.motion_regions
                        }))
                
                # 3. Analyze frame if needed
                if should_analyze:
                    asyncio.create_task(self._analyze_frame(frame, frame_data['frame_number']))
                
                # Adaptive quality adjustment
                if self.config.enable_adaptive_quality:
                    self._adjust_quality_based_on_memory()
                
                # Small delay to prevent CPU overload
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
    
    async def _analyze_frame(self, frame: np.ndarray, frame_number: int):
        """Analyze frame using Claude Vision API with sliding window if needed"""
        start_time = time.time()
        
        try:
            # Check if we should use sliding window
            if self.config.enable_sliding_window and self._should_use_sliding_window(frame):
                results = await self._analyze_with_sliding_window(frame)
            else:
                # Full frame analysis
                results = await self._analyze_full_frame(frame)
            
            self.frames_analyzed += 1
            
            # Record metrics
            metrics = FrameMetrics(
                timestamp=time.time(),
                frame_number=frame_number,
                processing_time=time.time() - start_time,
                memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
                motion_score=0.0,  # Set by motion detection
                windows_processed=results.get('windows_processed', 1),
                analysis_triggered=True
            )
            self.metrics.append(metrics)
            
            # Trigger callback
            await self._trigger_event('frame_analyzed', {
                'frame_number': frame_number,
                'results': results,
                'metrics': metrics
            })
            
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
    
    def _should_use_sliding_window(self, frame: np.ndarray) -> bool:
        """Determine if sliding window should be used"""
        height, width = frame.shape[:2]
        total_pixels = height * width
        available_mb = psutil.virtual_memory().available / 1024 / 1024
        
        # Use sliding window if frame is large or memory is low
        return (total_pixels > 2_000_000 or  # >2MP
                available_mb < 3000 or  # <3GB available
                psutil.Process().memory_info().rss / 1024 / 1024 > self.config.frame_memory_threshold_mb)
    
    async def _analyze_with_sliding_window(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze frame using sliding window approach"""
        # Parse window size
        w_width, w_height = map(int, self.config.sliding_window_size.split('x'))
        
        # Generate windows with priority on motion regions
        windows = self._generate_priority_windows(frame, w_width, w_height)
        
        # Analyze each window
        window_results = []
        for i, window in enumerate(windows[:self.config.max_windows_per_frame]):
            x, y, w, h = window['bounds']
            window_frame = frame[y:y+h, x:x+w]
            
            # Quick analysis per window
            result = await self.vision_analyzer.analyze_screenshot(
                window_frame,
                f"Analyze this region of the screen (window {i+1}/{len(windows)})",
                custom_config={'max_tokens': 300}  # Smaller response per window
            )
            
            window_results.append({
                'window': window,
                'analysis': result[0] if isinstance(result, tuple) else result
            })
        
        # Combine results
        return {
            'windows_processed': len(window_results),
            'results': window_results,
            'method': 'sliding_window'
        }
    
    async def _analyze_full_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze full frame"""
        # Optionally downsample if memory is tight
        if psutil.virtual_memory().available / 1024 / 1024 < 2000:
            # Downsample to half resolution
            frame = frame[::2, ::2]
        
        result = await self.vision_analyzer.analyze_screenshot(
            frame,
            "Analyze the current screen content"
        )
        
        return {
            'windows_processed': 1,
            'results': result[0] if isinstance(result, tuple) else result,
            'method': 'full_frame'
        }
    
    def _generate_priority_windows(self, frame: np.ndarray, w_width: int, w_height: int) -> List[Dict]:
        """Generate windows with priority on areas with motion or importance"""
        height, width = frame.shape[:2]
        windows = []
        
        # If we have motion regions, prioritize those
        if self.motion_regions:
            for region in self.motion_regions:
                windows.append({
                    'bounds': region,
                    'priority': 1.0,
                    'type': 'motion'
                })
        
        # Add regular grid windows
        overlap = self.config.sliding_window_overlap
        step_x = int(w_width * (1 - overlap))
        step_y = int(w_height * (1 - overlap))
        
        for y in range(0, height - w_height + 1, step_y):
            for x in range(0, width - w_width + 1, step_x):
                # Calculate priority based on position (center gets higher priority)
                center_x = x + w_width // 2
                center_y = y + w_height // 2
                dx = (center_x - width // 2) / width
                dy = (center_y - height // 2) / height
                distance = np.sqrt(dx**2 + dy**2)
                priority = 1.0 - min(distance, 1.0)
                
                windows.append({
                    'bounds': (x, y, w_width, w_height),
                    'priority': priority * 0.5,  # Regular windows have lower priority
                    'type': 'grid'
                })
        
        # Sort by priority
        windows.sort(key=lambda w: w['priority'], reverse=True)
        
        return windows
    
    def _detect_motion(self, frame: np.ndarray) -> float:
        """Detect motion between frames"""
        if self.previous_frame is None:
            self.previous_frame = frame
            return 0.0
        
        try:
            # Simple motion detection using frame difference
            diff = np.abs(frame.astype(np.float32) - self.previous_frame.astype(np.float32))
            motion_score = np.mean(diff) / 255.0
            
            # Find regions with significant motion
            if motion_score > self.config.motion_threshold:
                # Simple grid-based motion regions
                grid_size = 128
                h, w = frame.shape[:2]
                self.motion_regions = []
                
                for y in range(0, h, grid_size):
                    for x in range(0, w, grid_size):
                        region_diff = diff[y:min(y+grid_size, h), x:min(x+grid_size, w)]
                        if np.mean(region_diff) > self.config.motion_threshold * 255:
                            self.motion_regions.append((x, y, grid_size, grid_size))
            
            self.previous_frame = frame
            return motion_score
            
        except Exception as e:
            logger.error(f"Motion detection failed: {e}")
            return 0.0
    
    def _check_memory_available(self) -> bool:
        """Check if enough memory is available"""
        available_mb = psutil.virtual_memory().available / 1024 / 1024
        process_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Need at least 1.5x the memory limit available
        return available_mb > self.config.memory_limit_mb * 1.5
    
    def _adjust_quality_based_on_memory(self):
        """Dynamically adjust capture quality based on memory"""
        available_mb = psutil.virtual_memory().available / 1024 / 1024
        process_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
        if available_mb < 2000 or process_mb > self.config.memory_limit_mb * 0.8:
            # Reduce quality
            if hasattr(self.capture_impl, 'config'):
                self.capture_impl.config.target_fps = max(
                    self.config.min_fps,
                    self.capture_impl.config.target_fps - 5
                )
                logger.warning(f"Reduced capture FPS to {self.capture_impl.config.target_fps}")
    
    async def stop_streaming(self):
        """Stop video streaming"""
        self.is_capturing = False
        
        # Stop capture
        if self.capture_impl:
            if hasattr(self.capture_impl, 'stop_capture'):
                self.capture_impl.stop_capture()
        
        # Wait for threads
        if self.process_thread:
            self.process_thread.join(timeout=2.0)
        
        # Clear buffers
        self.frame_buffer.clear()
        
        logger.info("Video streaming stopped")
    
    async def _trigger_event(self, event_type: str, data: Dict[str, Any]):
        """Trigger event callbacks"""
        if event_type in self.event_callbacks:
            callbacks = list(self.event_callbacks[event_type])
            for callback in callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"Error in callback for {event_type}: {e}")
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register event callback"""
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].add(callback)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get streaming metrics"""
        recent_metrics = self.metrics[-10:] if self.metrics else []
        
        return {
            'is_capturing': self.is_capturing,
            'frames_processed': self.frames_processed,
            'frames_analyzed': self.frames_analyzed,
            'buffer_size': len(self.frame_buffer.frames),
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'recent_analysis': recent_metrics,
            'capture_method': 'macos_native' if MACOS_CAPTURE_AVAILABLE else 'fallback'
        }
    
    # Fallback methods
    def _start_cv2_capture(self):
        """Start capture using OpenCV"""
        logger.info("Starting OpenCV video capture (fallback)")
        # Implementation would go here
        pass
    
    def _start_screenshot_loop(self):
        """Start capture using screenshot loop"""
        logger.info("Starting screenshot loop capture (fallback)")
        
        def screenshot_loop():
            while self.is_capturing:
                try:
                    # Use vision analyzer's capture method
                    screenshot = asyncio.run(self.vision_analyzer.capture_screen())
                    if screenshot:
                        frame = np.array(screenshot)
                        self._on_frame_captured(frame)
                    
                    # Sleep to achieve target FPS
                    time.sleep(1.0 / self.config.target_fps)
                    
                except Exception as e:
                    logger.error(f"Screenshot capture error: {e}")
        
        self.capture_thread = threading.Thread(target=screenshot_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()