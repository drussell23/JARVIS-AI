#!/usr/bin/env swift
//
// SimpleSwiftCapture.swift
// Direct screen capture that shows purple indicator immediately
//

import Foundation
import AVFoundation
import CoreGraphics

class SimpleCapture {
    static func startCapture() -> Bool {
        print("[SWIFT] Creating capture session...")
        
        // Create session
        let session = AVCaptureSession()
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
            print("[SWIFT] ✅ SUCCESS: Capture session is running!")
            print("[SWIFT] 🟣 PURPLE INDICATOR SHOULD BE VISIBLE NOW!")
            
            // Keep the session alive for monitoring
            print("[SWIFT] Keeping session alive for 30 seconds...")
            Thread.sleep(forTimeInterval: 30.0)
            
            // Stop the session
            print("[SWIFT] Stopping capture session...")
            session.stopRunning()
            print("[SWIFT] ✅ Session stopped - purple indicator should disappear")
            
            return true
        } else {
            print("[SWIFT] ERROR: Session failed to start")
            return false
        }
    }
}

// Main execution
if CommandLine.arguments.contains("--capture") {
    print("\n🟣 SIMPLE SWIFT CAPTURE TEST")
    print(String(repeating: "=", count: 60))
    print("This will show the purple recording indicator for 30 seconds\n")
    
    let success = SimpleCapture.startCapture()
    
    if success {
        print("\n✅ Test completed successfully!")
    } else {
        print("\n❌ Test failed!")
        exit(1)
    }
} else {
    print("Usage: swift simple_swift_capture.swift --capture")
}