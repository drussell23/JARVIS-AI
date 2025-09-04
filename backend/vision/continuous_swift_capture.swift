#!/usr/bin/env swift
//
// ContinuousSwiftCapture.swift
// Continuous screen capture that shows purple indicator until stopped
//

import Foundation
import AVFoundation
import CoreGraphics
import Darwin

class ContinuousCapture {
    private static var session: AVCaptureSession?
    private static var isCapturing = false
    
    static func startCapture() -> Bool {
        guard !isCapturing else {
            print("[SWIFT] Capture already running")
            return true
        }
        
        print("[SWIFT] Creating capture session...")
        
        // Create session
        session = AVCaptureSession()
        guard let session = session else { return false }
        
        session.sessionPreset = .hd1920x1080
        
        // Create screen input for main display
        guard let screenInput = AVCaptureScreenInput(displayID: CGMainDisplayID()) else {
            print("[SWIFT] ERROR: Failed to create screen input")
            return false
        }
        
        print("[SWIFT] Configuring screen input...")
        screenInput.minFrameDuration = CMTimeMake(value: 1, timescale: 30)
        screenInput.capturesCursor = true
        
        // Add input to session
        if session.canAddInput(screenInput) {
            session.addInput(screenInput)
            print("[SWIFT] Added screen input to session")
        } else {
            print("[SWIFT] ERROR: Cannot add screen input")
            return false
        }
        
        // Create simple output (required for session to run)
        let output = AVCaptureVideoDataOutput()
        if session.canAddOutput(output) {
            session.addOutput(output)
            print("[SWIFT] Added video output")
        }
        
        // START THE SESSION - THIS SHOWS THE PURPLE INDICATOR
        print("[SWIFT] Starting capture session...")
        session.startRunning()
        
        // Check if running
        if session.isRunning {
            isCapturing = true
            print("[SWIFT] ✅ SUCCESS: Capture session is running!")
            print("[SWIFT] 🟣 PURPLE INDICATOR SHOULD BE VISIBLE NOW!")
            print("[SWIFT] Session will continue running until stopped")
            return true
        } else {
            print("[SWIFT] ERROR: Session failed to start")
            return false
        }
    }
    
    static func stopCapture() {
        guard let session = session, isCapturing else {
            print("[SWIFT] No active capture session")
            return
        }
        
        print("[SWIFT] Stopping capture session...")
        session.stopRunning()
        self.session = nil
        isCapturing = false
        print("[SWIFT] ✅ Session stopped - purple indicator should disappear")
    }
    
    static func isRunning() -> Bool {
        return session?.isRunning ?? false
    }
}

// Handle termination signal
signal(SIGTERM) { _ in
    print("\n[SWIFT] Received termination signal")
    ContinuousCapture.stopCapture()
    exit(0)
}

signal(SIGINT) { _ in
    print("\n[SWIFT] Received interrupt signal")
    ContinuousCapture.stopCapture()
    exit(0)
}

// Main execution
if CommandLine.arguments.count > 1 {
    let command = CommandLine.arguments[1]
    
    switch command {
    case "--start":
        if ContinuousCapture.startCapture() {
            print("[SWIFT] Capture running. Process will continue until terminated.")
            // Keep the process alive
            RunLoop.main.run()
        } else {
            print("[SWIFT] Failed to start capture")
            exit(1)
        }
        
    case "--stop":
        ContinuousCapture.stopCapture()
        
    case "--status":
        let running = ContinuousCapture.isRunning()
        print("[SWIFT] Capture is \(running ? "running" : "not running")")
        
    default:
        print("Usage: swift continuous_swift_capture.swift [--start|--stop|--status]")
    }
} else {
    print("Usage: swift continuous_swift_capture.swift [--start|--stop|--status]")
}