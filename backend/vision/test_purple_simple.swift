#!/usr/bin/env swift
//
// Simple test to verify purple indicator stays on
//

import Foundation
import AVFoundation

print("🟣 Simple Purple Indicator Test")
print("================================")

// Create session
let session = AVCaptureSession()
session.sessionPreset = .hd1920x1080

// Create screen input
guard let screenInput = AVCaptureScreenInput(displayID: CGMainDisplayID()) else {
    print("❌ Failed to create screen input")
    exit(1)
}

// Configure
screenInput.minFrameDuration = CMTimeMake(value: 1, timescale: 30)

// Add input
if session.canAddInput(screenInput) {
    session.addInput(screenInput)
    
    // Add output (required)
    let output = AVCaptureVideoDataOutput()
    if session.canAddOutput(output) {
        session.addOutput(output)
    }
    
    // Start session
    print("Starting capture session...")
    session.startRunning()
    
    if session.isRunning {
        print("✅ Session started successfully!")
        print("🟣 PURPLE INDICATOR SHOULD BE VISIBLE NOW!")
        print("Press Ctrl+C to stop...\n")
        
        // Keep alive with checks
        var counter = 0
        while true {
            Thread.sleep(forTimeInterval: 5.0)
            counter += 1
            
            let isRunning = session.isRunning
            print("Check #\(counter): Session running = \(isRunning)")
            
            if !isRunning {
                print("⚠️ Session stopped unexpectedly!")
                break
            }
        }
    } else {
        print("❌ Failed to start session")
    }
} else {
    print("❌ Cannot add input to session")
}