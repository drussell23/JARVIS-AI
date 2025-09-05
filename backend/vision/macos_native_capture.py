"""
Direct macOS native capture using PyObjC
This will show the purple recording indicator
"""

import logging
import threading
import time
from typing import Optional, Callable

try:
    import AVFoundation
    import CoreMedia
    from Quartz import CoreVideo
    from Cocoa import NSObject, NSRunLoop, NSDefaultRunLoopMode, NSDate
    from Foundation import NSThread, NSAutoreleasePool
    import objc
    import libdispatch
    MACOS_AVAILABLE = True
except ImportError:
    MACOS_AVAILABLE = False

logger = logging.getLogger(__name__)

class MacOSNativeCapture:
    """Direct macOS screen capture that shows purple indicator"""
    
    def __init__(self):
        if not MACOS_AVAILABLE:
            raise ImportError("macOS frameworks not available")
            
        self.session = None
        self.screen_input = None
        self.video_output = None
        self.is_running = False
        self.capture_thread = None
        self.delegate = None
        self._frame_callback = None
        
    def start_capture(self, frame_callback: Optional[Callable] = None) -> bool:
        """Start screen capture - this will show the purple indicator"""
        if self.is_running:
            logger.warning("Capture already running")
            return True
            
        logger.info("[NATIVE] Starting macOS native capture...")
        self._frame_callback = frame_callback
        
        try:
            # Create capture session
            self.session = AVFoundation.AVCaptureSession.alloc().init()
            logger.info("[NATIVE] Created AVCaptureSession")
            
            # Set quality preset
            self.session.setSessionPreset_(AVFoundation.AVCaptureSessionPresetHigh)
            
            # Create screen input for main display
            display_id = 0  # Main display
            self.screen_input = AVFoundation.AVCaptureScreenInput.alloc().initWithDisplayID_(display_id)
            
            if not self.screen_input:
                logger.error("[NATIVE] Failed to create screen input")
                return False
                
            logger.info("[NATIVE] Created screen input")
            
            # Configure screen input
            self.screen_input.setMinFrameDuration_(CoreMedia.CMTimeMake(1, 30))  # 30 FPS
            self.screen_input.setCapturesCursor_(True)
            self.screen_input.setCapturesMouseClicks_(True)
            
            # Add input to session
            if self.session.canAddInput_(self.screen_input):
                self.session.addInput_(self.screen_input)
                logger.info("[NATIVE] Added screen input to session")
            else:
                logger.error("[NATIVE] Cannot add screen input to session")
                return False
            
            # Create video data output
            self.video_output = AVFoundation.AVCaptureVideoDataOutput.alloc().init()
            self.video_output.setAlwaysDiscardsLateVideoFrames_(True)
            
            # Set pixel format
            self.video_output.setVideoSettings_({
                str(CoreVideo.kCVPixelBufferPixelFormatTypeKey): CoreVideo.kCVPixelFormatType_32BGRA
            })
            
            # Create delegate for handling frames
            if frame_callback:
                self.delegate = CaptureDelegate.alloc().initWithCallback_(self._handle_frame)
                
                # Create dispatch queue for video processing
                from Foundation import DISPATCH_QUEUE_SERIAL
                import libdispatch
                queue = libdispatch.dispatch_queue_create(b"com.jarvis.videocapture", DISPATCH_QUEUE_SERIAL)
                
                self.video_output.setSampleBufferDelegate_queue_(self.delegate, queue)
            
            # Add output to session
            if self.session.canAddOutput_(self.video_output):
                self.session.addOutput_(self.video_output)
                logger.info("[NATIVE] Added video output to session")
            else:
                logger.error("[NATIVE] Cannot add video output to session")
                return False
            
            # Start the session in a separate thread to keep it running
            self.capture_thread = threading.Thread(target=self._run_capture_loop, daemon=True)
            self.capture_thread.start()
            
            # Wait a bit longer for the session to fully initialize
            time.sleep(1.0)
            
            if self.is_running:
                logger.info("[NATIVE] ✅ Capture started successfully - purple indicator should be visible!")
                return True
            else:
                logger.error("[NATIVE] Failed to start capture session")
                return False
                
        except Exception as e:
            logger.error(f"[NATIVE] Error starting capture: {e}", exc_info=True)
            self.cleanup()
            return False
    
    def _run_capture_loop(self):
        """Run the capture session with proper run loop to maintain purple indicator"""
        try:
            logger.info("[NATIVE] Starting capture session...")
            
            # Create autorelease pool for this thread
            pool = NSAutoreleasePool.alloc().init()
            
            # Start the capture session
            self.session.startRunning()
            self.is_running = True
            logger.info("[NATIVE] Session started, entering run loop...")
            
            # Get current run loop and add an input source to keep it alive
            run_loop = NSRunLoop.currentRunLoop()
            
            # Create a port to keep the run loop active
            from Foundation import NSPort
            port = NSPort.port()
            run_loop.addPort_forMode_(port, NSDefaultRunLoopMode)
            
            # Monitor loop iterations
            loop_count = 0
            last_check = time.time()
            
            # Run the run loop indefinitely until stopped
            while self.is_running:
                # Check session status periodically
                current_time = time.time()
                if current_time - last_check > 5.0:  # Check every 5 seconds
                    loop_count += 1
                    is_running = self.session.isRunning() if self.session else False
                    logger.info(f"[NATIVE] Status check #{loop_count}: Session running = {is_running}")
                    
                    if not is_running and self.is_running:
                        logger.warning("[NATIVE] Session stopped unexpectedly, attempting restart...")
                        self.session.startRunning()
                    
                    last_check = current_time
                
                # Run the run loop for a short interval
                # This processes events and maintains the purple indicator
                run_loop.runMode_beforeDate_(NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(0.1))
            
            # Clean up
            run_loop.removePort_forMode_(port, NSDefaultRunLoopMode)
            pool.drain()
            
        except Exception as e:
            logger.error(f"[NATIVE] Error in capture loop: {e}", exc_info=True)
        finally:
            if self.session and self.session.isRunning():
                self.session.stopRunning()
            self.is_running = False
            logger.info("[NATIVE] Capture loop ended")
    
    def _handle_frame(self, sample_buffer):
        """Handle captured frame from delegate"""
        if self._frame_callback:
            try:
                # Convert CMSampleBuffer to data if needed
                self._frame_callback(sample_buffer)
            except Exception as e:
                logger.error(f"[NATIVE] Error in frame callback: {e}")
    
    def stop_capture(self):
        """Stop capture - purple indicator will disappear"""
        logger.info("[NATIVE] Stopping capture...")
        
        self.is_running = False
        
        if self.session and self.session.isRunning():
            self.session.stopRunning()
            logger.info("[NATIVE] Session stopped")
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
            
        self.cleanup()
        logger.info("[NATIVE] ✅ Capture stopped - purple indicator should disappear")
    
    def cleanup(self):
        """Clean up resources"""
        if self.session:
            if self.screen_input:
                self.session.removeInput_(self.screen_input)
            if self.video_output:
                self.session.removeOutput_(self.video_output)
                
        self.session = None
        self.screen_input = None
        self.video_output = None
        self.delegate = None
        self.is_running = False
    
    def is_capturing(self) -> bool:
        """Check if currently capturing"""
        return self.is_running and self.session and self.session.isRunning()


# Delegate class for handling video frames
if MACOS_AVAILABLE:
    class CaptureDelegate(NSObject):
        """Delegate for handling captured video frames"""
        
        def initWithCallback_(self, callback):
            self = objc.super(CaptureDelegate, self).init()
            if self:
                self.callback = callback
            return self
        
        def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sample_buffer, connection):
            """AVCaptureVideoDataOutputSampleBufferDelegate method"""
            if self.callback:
                self.callback(sample_buffer)


# Global instance for easy access
_native_capture = None

def get_native_capture():
    """Get or create the native capture instance"""
    global _native_capture
    if _native_capture is None:
        _native_capture = MacOSNativeCapture()
    return _native_capture

def start_native_capture(frame_callback=None) -> bool:
    """Start native capture with purple indicator"""
    capture = get_native_capture()
    return capture.start_capture(frame_callback)

def stop_native_capture():
    """Stop native capture"""
    capture = get_native_capture()
    capture.stop_capture()

def is_native_capturing() -> bool:
    """Check if native capture is running"""
    capture = get_native_capture()
    return capture.is_capturing()