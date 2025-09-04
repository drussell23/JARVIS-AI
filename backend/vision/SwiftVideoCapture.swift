#!/usr/bin/env swift
//
// SwiftVideoCapture.swift
// Native macOS screen recording with proper permission handling
//

import Foundation
import AVFoundation
import CoreGraphics
import CoreMedia
import CoreVideo
import AppKit

// Configuration structure for video capture
struct VideoCaptureConfig: Codable {
    let displayID: CGDirectDisplayID
    let fps: Int
    let resolution: String
    let outputPath: String?
    
    static func fromEnvironment() -> VideoCaptureConfig {
        let displayID = CGDirectDisplayID(ProcessInfo.processInfo.environment["VIDEO_CAPTURE_DISPLAY_ID"] ?? "0") ?? 0
        let fps = Int(ProcessInfo.processInfo.environment["VIDEO_CAPTURE_FPS"] ?? "30") ?? 30
        let resolution = ProcessInfo.processInfo.environment["VIDEO_CAPTURE_RESOLUTION"] ?? "1920x1080"
        let outputPath = ProcessInfo.processInfo.environment["VIDEO_CAPTURE_OUTPUT_PATH"]
        
        return VideoCaptureConfig(
            displayID: displayID,
            fps: fps,
            resolution: resolution,
            outputPath: outputPath
        )
    }
}

// Response structure for communication with Python
struct CaptureResponse: Codable {
    let success: Bool
    let message: String
    let error: String?
    let permissionStatus: String?
    let isCapturing: Bool
    let framesCaptured: Int
}

// Main video capture class
@objc class SwiftVideoCapture: NSObject {
    private var captureSession: AVCaptureSession?
    private var screenInput: AVCaptureScreenInput?
    private var videoOutput: AVCaptureVideoDataOutput?
    private let config: VideoCaptureConfig
    private var frameCount: Int = 0
    private var isCapturing: Bool = false
    private let outputQueue = DispatchQueue(label: "com.jarvis.videocapture", qos: .userInitiated)
    
    init(config: VideoCaptureConfig) {
        self.config = config
        super.init()
    }
    
    // Check screen recording permission status
    func checkPermissionStatus() -> String {
        if #available(macOS 10.15, *) {
            // For macOS 10.15+, check if we can create a screen input
            if let _ = AVCaptureScreenInput(displayID: config.displayID) {
                return "authorized"
            } else {
                return "denied"
            }
        } else {
            // For older macOS versions
            return "authorized"
        }
    }
    
    // Request screen recording permission
    func requestPermission(completion: @escaping (Bool) -> Void) {
        if #available(macOS 10.15, *) {
            // Trigger permission dialog by attempting to create screen input
            DispatchQueue.main.async {
                let _ = AVCaptureScreenInput(displayID: self.config.displayID)
                
                // Check permission after a delay
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                    let status = self.checkPermissionStatus()
                    completion(status == "authorized")
                }
            }
        } else {
            completion(true)
        }
    }
    
    // Start video capture
    func startVideoCapture() -> CaptureResponse {
        let permissionStatus = checkPermissionStatus()
        
        guard permissionStatus == "authorized" else {
            return CaptureResponse(
                success: false,
                message: "Screen recording permission not granted",
                error: "Please grant screen recording permission in System Preferences > Security & Privacy > Privacy > Screen Recording",
                permissionStatus: permissionStatus,
                isCapturing: false,
                framesCaptured: frameCount
            )
        }
        
        // Create capture session
        captureSession = AVCaptureSession()
        
        guard let session = captureSession else {
            return CaptureResponse(
                success: false,
                message: "Failed to create capture session",
                error: "Could not initialize AVCaptureSession",
                permissionStatus: permissionStatus,
                isCapturing: false,
                framesCaptured: frameCount
            )
        }
        
        // Configure session preset based on resolution
        switch config.resolution {
        case "3840x2160":
            if session.canSetSessionPreset(.hd4K3840x2160) {
                session.sessionPreset = .hd4K3840x2160
            }
        case "1920x1080":
            if session.canSetSessionPreset(.hd1920x1080) {
                session.sessionPreset = .hd1920x1080
            }
        case "1280x720":
            if session.canSetSessionPreset(.hd1280x720) {
                session.sessionPreset = .hd1280x720
            }
        default:
            session.sessionPreset = .high
        }
        
        // Create screen input
        guard let screenInput = AVCaptureScreenInput(displayID: config.displayID) else {
            return CaptureResponse(
                success: false,
                message: "Failed to create screen input",
                error: "Could not create AVCaptureScreenInput for display \(config.displayID)",
                permissionStatus: permissionStatus,
                isCapturing: false,
                framesCaptured: frameCount
            )
        }
        
        self.screenInput = screenInput
        
        // Configure screen input
        screenInput.minFrameDuration = CMTimeMake(value: 1, timescale: Int32(config.fps))
        screenInput.capturesCursor = true
        screenInput.capturesMouseClicks = true
        
        // Add input to session
        if session.canAddInput(screenInput) {
            session.addInput(screenInput)
        } else {
            return CaptureResponse(
                success: false,
                message: "Failed to add input to session",
                error: "Session cannot accept screen input",
                permissionStatus: permissionStatus,
                isCapturing: false,
                framesCaptured: frameCount
            )
        }
        
        // Create video output
        videoOutput = AVCaptureVideoDataOutput()
        guard let output = videoOutput else {
            return CaptureResponse(
                success: false,
                message: "Failed to create video output",
                error: "Could not create AVCaptureVideoDataOutput",
                permissionStatus: permissionStatus,
                isCapturing: false,
                framesCaptured: frameCount
            )
        }
        
        // Configure output
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
        ]
        output.alwaysDiscardsLateVideoFrames = true
        
        // Set delegate
        output.setSampleBufferDelegate(self, queue: outputQueue)
        
        // Add output to session
        if session.canAddOutput(output) {
            session.addOutput(output)
        } else {
            return CaptureResponse(
                success: false,
                message: "Failed to add output to session",
                error: "Session cannot accept video output",
                permissionStatus: permissionStatus,
                isCapturing: false,
                framesCaptured: frameCount
            )
        }
        
        // Start capture
        session.startRunning()
        isCapturing = true
        
        // Log to confirm session is running
        print("DEBUG: AVCaptureSession.isRunning = \(session.isRunning)")
        print("DEBUG: Screen recording should show purple indicator now")
        
        return CaptureResponse(
            success: true,
            message: "Video capture started successfully",
            error: nil,
            permissionStatus: permissionStatus,
            isCapturing: true,
            framesCaptured: frameCount
        )
    }
    
    // Stop video capture
    func stopVideoCapture() -> CaptureResponse {
        if let session = captureSession {
            session.stopRunning()
            
            // Clean up
            if let input = screenInput {
                session.removeInput(input)
            }
            if let output = videoOutput {
                session.removeOutput(output)
            }
        }
        
        captureSession = nil
        screenInput = nil
        videoOutput = nil
        isCapturing = false
        
        return CaptureResponse(
            success: true,
            message: "Video capture stopped",
            error: nil,
            permissionStatus: checkPermissionStatus(),
            isCapturing: false,
            framesCaptured: frameCount
        )
    }
    
    // Get current status
    func getStatus() -> CaptureResponse {
        return CaptureResponse(
            success: true,
            message: isCapturing ? "Capturing" : "Not capturing",
            error: nil,
            permissionStatus: checkPermissionStatus(),
            isCapturing: isCapturing,
            framesCaptured: frameCount
        )
    }
}

// Extension for video frame handling
extension SwiftVideoCapture: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        frameCount += 1
        
        // If output path is specified, we could save frames here
        // For now, just count frames for monitoring
        
        // Send frame count update every 30 frames (approximately once per second at 30fps)
        if frameCount % 30 == 0 {
            let status = getStatus()
            if let jsonData = try? JSONEncoder().encode(status),
               let jsonString = String(data: jsonData, encoding: .utf8) {
                print("STATUS_UPDATE: \(jsonString)")
            }
        }
    }
}

// Command-line interface
class SwiftVideoCapturesCLI {
    static func main() {
        let arguments = CommandLine.arguments
        
        guard arguments.count > 1 else {
            printUsage()
            return
        }
        
        let command = arguments[1]
        let config = VideoCaptureConfig.fromEnvironment()
        let capture = SwiftVideoCapture(config: config)
        
        var response: CaptureResponse
        
        switch command {
        case "start":
            response = capture.startVideoCapture()
            
        case "stop":
            response = capture.stopVideoCapture()
            
        case "status":
            response = capture.getStatus()
            
        case "check-permission":
            let status = capture.checkPermissionStatus()
            response = CaptureResponse(
                success: true,
                message: "Permission status: \(status)",
                error: nil,
                permissionStatus: status,
                isCapturing: false,
                framesCaptured: 0
            )
            
        case "request-permission":
            var permissionGranted = false
            let semaphore = DispatchSemaphore(value: 0)
            
            capture.requestPermission { granted in
                permissionGranted = granted
                semaphore.signal()
            }
            
            semaphore.wait()
            
            response = CaptureResponse(
                success: permissionGranted,
                message: permissionGranted ? "Permission granted" : "Permission denied",
                error: permissionGranted ? nil : "User denied screen recording permission",
                permissionStatus: capture.checkPermissionStatus(),
                isCapturing: false,
                framesCaptured: 0
            )
            
        default:
            response = CaptureResponse(
                success: false,
                message: "Unknown command: \(command)",
                error: "Valid commands: start, stop, status, check-permission, request-permission",
                permissionStatus: nil,
                isCapturing: false,
                framesCaptured: 0
            )
        }
        
        // Output JSON response
        if let jsonData = try? JSONEncoder().encode(response),
           let jsonString = String(data: jsonData, encoding: .utf8) {
            print(jsonString)
        } else {
            print("{\"success\": false, \"error\": \"Failed to encode response\"}")
        }
    }
    
    static func printUsage() {
        print("Usage: swift SwiftVideoCapture.swift <command>")
        print("Commands:")
        print("  start              - Start video capture")
        print("  stop               - Stop video capture")
        print("  status             - Get current capture status")
        print("  check-permission   - Check screen recording permission")
        print("  request-permission - Request screen recording permission")
        print("\nEnvironment variables:")
        print("  VIDEO_CAPTURE_DISPLAY_ID   - Display ID to capture (default: 0)")
        print("  VIDEO_CAPTURE_FPS          - Frames per second (default: 30)")
        print("  VIDEO_CAPTURE_RESOLUTION   - Resolution (default: 1920x1080)")
        print("  VIDEO_CAPTURE_OUTPUT_PATH  - Output path for frames (optional)")
    }
}

// Run CLI
SwiftVideoCapturesCLI.main()