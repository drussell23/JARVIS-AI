#!/usr/bin/env swift
//
// PersistentCapture.swift
// Maintains AVCaptureSession with purple indicator
//

import Foundation
import AVFoundation
import CoreGraphics
import AppKit

class PersistentCaptureManager: NSObject {
    private var captureSession: AVCaptureSession?
    private var videoOutput: AVCaptureVideoDataOutput?
    private let sessionQueue = DispatchQueue(label: "com.jarvis.capture.session")
    
    override init() {
        super.init()
        checkPermissions()
    }
    
    func checkPermissions() {
        if #available(macOS 10.15, *) {
            let status = CGPreflightScreenCaptureAccess()
            if !status {
                print("⚠️ Screen recording permission not granted")
                print("Please grant screen recording permission in System Preferences > Security & Privacy > Screen Recording")
            } else {
                print("✅ Screen recording permission granted")
            }
        }
    }
    
    func startCapture() -> Bool {
        print("[CAPTURE] Creating capture session...")
        
        captureSession = AVCaptureSession()
        guard let session = captureSession else { return false }
        
        // Configure session
        session.beginConfiguration()
        session.sessionPreset = .hd1920x1080
        
        // Create screen input
        guard let screenInput = AVCaptureScreenInput(displayID: CGMainDisplayID()) else {
            print("[CAPTURE] Failed to create screen input")
            return false
        }
        
        // Configure screen input
        screenInput.minFrameDuration = CMTimeMake(value: 1, timescale: 30)
        screenInput.capturesCursor = true
        screenInput.capturesMouseClicks = false
        
        // Add input to session
        if session.canAddInput(screenInput) {
            session.addInput(screenInput)
            print("[CAPTURE] Added screen input")
        } else {
            print("[CAPTURE] Cannot add screen input")
            return false
        }
        
        // Create video data output
        videoOutput = AVCaptureVideoDataOutput()
        guard let output = videoOutput else { return false }
        
        // Configure output
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: sessionQueue)
        
        // Add output to session
        if session.canAddOutput(output) {
            session.addOutput(output)
            print("[CAPTURE] Added video output")
        } else {
            print("[CAPTURE] Cannot add video output")
            return false
        }
        
        // Commit configuration
        session.commitConfiguration()
        
        // Start the session on a background queue
        sessionQueue.async {
            print("[CAPTURE] Starting capture session...")
            session.startRunning()
            
            if session.isRunning {
                print("[CAPTURE] ✅ Session started successfully!")
                print("[CAPTURE] 🟣 PURPLE INDICATOR SHOULD BE VISIBLE!")
            } else {
                print("[CAPTURE] ❌ Failed to start session")
            }
        }
        
        return true
    }
    
    func stopCapture() {
        sessionQueue.async {
            if let session = self.captureSession {
                print("[CAPTURE] Stopping capture session...")
                session.stopRunning()
                print("[CAPTURE] ✅ Session stopped")
            }
            self.captureSession = nil
            self.videoOutput = nil
        }
    }
    
    func isRunning() -> Bool {
        return captureSession?.isRunning ?? false
    }
}

// Implement delegate to keep session active
extension PersistentCaptureManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // We don't need to process frames, just having this keeps the session active
    }
}

// Main execution
let manager = PersistentCaptureManager()

if CommandLine.arguments.contains("--start") {
    print("\n🟣 PERSISTENT CAPTURE TEST")
    print("=" * 60)
    
    if manager.startCapture() {
        print("\nCapture started. Monitoring session status...")
        print("Press Ctrl+C to stop\n")
        
        // Monitor session status
        var checkCount = 0
        while true {
            Thread.sleep(forTimeInterval: 5.0)
            checkCount += 1
            
            let isRunning = manager.isRunning()
            print("Status check #\(checkCount): Session running = \(isRunning)")
            
            if !isRunning {
                print("⚠️ Session stopped unexpectedly!")
                // Try to restart
                print("Attempting to restart...")
                _ = manager.startCapture()
            }
        }
    } else {
        print("❌ Failed to start capture")
        exit(1)
    }
} else {
    print("Usage: swift persistent_capture.swift --start")
}

// Helper extension
extension String {
    static func *(left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
}