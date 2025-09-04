#!/usr/bin/env swift
//
// InfinitePurpleCapture.swift
// Maintains purple indicator indefinitely until killed
//

import Foundation
import AVFoundation
import CoreGraphics

// Global session that persists
var globalSession: AVCaptureSession?
var keepRunning = true

// Signal handlers
signal(SIGTERM) { _ in
    print("\n[CAPTURE] Received termination signal")
    keepRunning = false
    if let session = globalSession {
        session.stopRunning()
    }
    exit(0)
}

signal(SIGINT) { _ in
    print("\n[CAPTURE] Received interrupt signal")
    keepRunning = false
    if let session = globalSession {
        session.stopRunning()
    }
    exit(0)
}

func startInfiniteCapture() -> Bool {
    print("[CAPTURE] Creating infinite capture session...")
    
    // Create session
    globalSession = AVCaptureSession()
    guard let session = globalSession else {
        print("[CAPTURE] Failed to create session")
        return false
    }
    
    // Configure session
    session.beginConfiguration()
    session.sessionPreset = .hd1920x1080
    
    // Create screen input
    guard let screenInput = AVCaptureScreenInput(displayID: CGMainDisplayID()) else {
        print("[CAPTURE] Failed to create screen input")
        return false
    }
    
    // Configure input
    screenInput.minFrameDuration = CMTimeMake(value: 1, timescale: 30)
    screenInput.capturesCursor = true
    screenInput.capturesMouseClicks = false
    
    // Add input
    if session.canAddInput(screenInput) {
        session.addInput(screenInput)
        print("[CAPTURE] Added screen input")
    } else {
        print("[CAPTURE] Cannot add screen input")
        return false
    }
    
    // Create video output (required for session to run)
    let output = AVCaptureVideoDataOutput()
    output.alwaysDiscardsLateVideoFrames = true
    
    if session.canAddOutput(output) {
        session.addOutput(output)
        print("[CAPTURE] Added video output")
    }
    
    // Commit configuration
    session.commitConfiguration()
    
    // Start the session
    print("[CAPTURE] Starting capture session...")
    session.startRunning()
    
    // Wait a moment for session to stabilize
    Thread.sleep(forTimeInterval: 0.5)
    
    if session.isRunning {
        print("[CAPTURE] ✅ SUCCESS! Session is running")
        print("[CAPTURE] 🟣 PURPLE INDICATOR IS NOW VISIBLE")
        print("[CAPTURE] Session will run INDEFINITELY until process is killed")
        print("[CAPTURE] Current status: \(session.isRunning ? "RUNNING" : "STOPPED")")
        return true
    } else {
        print("[CAPTURE] ❌ Failed to start session")
        return false
    }
}

// Main execution
if CommandLine.arguments.contains("--start") {
    print("\n🟣 INFINITE PURPLE CAPTURE")
    print("=" * 60)
    print("This will keep the purple indicator on INDEFINITELY")
    print("The process must be killed to stop it\n")
    
    if startInfiniteCapture() {
        print("\n[CAPTURE] Entering infinite loop...")
        print("[CAPTURE] Purple indicator will stay on until process is terminated")
        
        // Monitor session health
        var checkCount = 0
        while keepRunning {
            Thread.sleep(forTimeInterval: 10.0) // Check every 10 seconds
            checkCount += 1
            
            if let session = globalSession {
                if session.isRunning {
                    print("[CAPTURE] Health check #\(checkCount): Session running ✓")
                } else {
                    print("[CAPTURE] ⚠️ Session stopped! Restarting...")
                    session.startRunning()
                    if session.isRunning {
                        print("[CAPTURE] ✅ Session restarted successfully")
                    } else {
                        print("[CAPTURE] ❌ Failed to restart session")
                        break
                    }
                }
            } else {
                print("[CAPTURE] ❌ Session is nil!")
                break
            }
        }
        
        print("[CAPTURE] Loop ended, cleaning up...")
        if let session = globalSession {
            session.stopRunning()
        }
    } else {
        print("[CAPTURE] Failed to start infinite capture")
        exit(1)
    }
} else {
    print("Usage: swift infinite_purple_capture.swift --start")
}

// String extension
extension String {
    static func *(left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
}