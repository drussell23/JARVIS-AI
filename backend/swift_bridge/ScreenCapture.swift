#!/usr/bin/swift
//
// ScreenCapture.swift
// Native macOS screen capture for JARVIS
// Provides efficient, low-latency screen capture with minimal overhead
//

import Foundation
import AppKit
import CoreGraphics
import Vision

@objc public class SwiftScreenCapture: NSObject {
    private var displayStream: CGDisplayStream?
    private let queue = DispatchQueue(label: "com.jarvis.screencapture", qos: .userInteractive)
    private var screenUpdateHandler: ((Data) -> Void)?
    
    // Performance optimizations
    private let compressionQuality: CGFloat = 0.8
    private var lastCaptureTime: Date = Date()
    private let minimumCaptureInterval: TimeInterval = 0.1 // Max 10fps
    
    @objc public override init() {
        super.init()
    }
    
    // MARK: - Public Interface
    
    @objc public func startContinuousCapture(callback: @escaping (Data) -> Void) -> Bool {
        screenUpdateHandler = callback
        
        guard let displayID = CGMainDisplayID() as CGDirectDisplayID? else {
            print("Failed to get main display ID")
            return false
        }
        
        let displayBounds = CGDisplayBounds(displayID)
        
        // Configure display stream for efficient capture
        let properties: [CFString: Any] = [
            CGDisplayStream.preserveAspectRatio: kCFBooleanTrue as Any,
            CGDisplayStream.minimumFrameTime: 0.1, // Limit to 10fps
            CGDisplayStream.showCursor: kCFBooleanTrue as Any
        ]
        
        displayStream = CGDisplayStream(
            dispatchQueueDisplay: displayID,
            outputWidth: Int(displayBounds.width),
            outputHeight: Int(displayBounds.height),
            pixelFormat: Int32(kCVPixelFormatType_32BGRA),
            properties: properties as CFDictionary,
            queue: queue,
            handler: { [weak self] status, displayTime, frameSurface, updateRef in
                self?.handleFrame(status: status, frameSurface: frameSurface)
            }
        )
        
        guard let stream = displayStream else {
            print("Failed to create display stream")
            return false
        }
        
        let result = stream.start()
        if result == .success {
            print("Started continuous screen capture")
            return true
        } else {
            print("Failed to start display stream: \(result)")
            return false
        }
    }
    
    @objc public func stopContinuousCapture() {
        displayStream?.stop()
        displayStream = nil
        screenUpdateHandler = nil
        print("Stopped continuous screen capture")
    }
    
    @objc public func captureScreen() -> Data? {
        // Single screen capture
        guard let image = CGDisplayCreateImage(CGMainDisplayID()) else {
            return nil
        }
        
        let bitmapRep = NSBitmapImageRep(cgImage: image)
        guard let jpegData = bitmapRep.representation(
            using: .jpeg,
            properties: [.compressionFactor: compressionQuality]
        ) else {
            return nil
        }
        
        return jpegData
    }
    
    @objc public func captureWindow(windowID: Int) -> Data? {
        // Capture specific window
        guard let windowImage = CGWindowListCreateImage(
            .null,
            .optionIncludingWindow,
            CGWindowID(windowID),
            [.boundsIgnoreFraming, .bestResolution]
        ) else {
            return nil
        }
        
        let bitmapRep = NSBitmapImageRep(cgImage: windowImage)
        guard let jpegData = bitmapRep.representation(
            using: .jpeg,
            properties: [.compressionFactor: compressionQuality]
        ) else {
            return nil
        }
        
        return jpegData
    }
    
    @objc public func getActiveApplication() -> String {
        if let activeApp = NSWorkspace.shared.frontmostApplication {
            return activeApp.localizedName ?? "Unknown"
        }
        return "Unknown"
    }
    
    @objc public func getWindowList() -> [[String: Any]] {
        var windows: [[String: Any]] = []
        
        let windowList = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [[String: Any]] ?? []
        
        for window in windowList {
            if let windowTitle = window[kCGWindowName as String] as? String,
               let ownerName = window[kCGWindowOwnerName as String] as? String,
               let windowID = window[kCGWindowNumber as String] as? Int,
               let bounds = window[kCGWindowBounds as String] as? [String: CGFloat] {
                
                windows.append([
                    "id": windowID,
                    "title": windowTitle,
                    "app": ownerName,
                    "bounds": bounds
                ])
            }
        }
        
        return windows
    }
    
    // MARK: - Private Methods
    
    private func handleFrame(status: CGDisplayStreamFrameStatus, frameSurface: IOSurface?) {
        guard status == .frameComplete,
              let surface = frameSurface else {
            return
        }
        
        // Rate limiting
        let now = Date()
        if now.timeIntervalSince(lastCaptureTime) < minimumCaptureInterval {
            return
        }
        lastCaptureTime = now
        
        // Convert IOSurface to JPEG data
        autoreleasepool {
            let ciImage = CIImage(ioSurface: surface)
            let context = CIContext(options: [.useSoftwareRenderer: false])
            
            if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
                let bitmapRep = NSBitmapImageRep(cgImage: cgImage)
                if let jpegData = bitmapRep.representation(
                    using: .jpeg,
                    properties: [.compressionFactor: compressionQuality]
                ) {
                    screenUpdateHandler?(jpegData)
                }
            }
        }
    }
    
    // MARK: - Vision Framework Integration
    
    @objc public func detectText(in imageData: Data) -> [[String: Any]] {
        guard let image = NSImage(data: imageData),
              let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            return []
        }
        
        var detectedText: [[String: Any]] = []
        
        let request = VNRecognizeTextRequest { request, error in
            guard let observations = request.results as? [VNRecognizedTextObservation] else { return }
            
            for observation in observations {
                if let topCandidate = observation.topCandidates(1).first {
                    detectedText.append([
                        "text": topCandidate.string,
                        "confidence": topCandidate.confidence,
                        "bounds": [
                            "x": observation.boundingBox.origin.x,
                            "y": observation.boundingBox.origin.y,
                            "width": observation.boundingBox.width,
                            "height": observation.boundingBox.height
                        ]
                    ])
                }
            }
        }
        
        request.recognitionLevel = .accurate
        
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try? handler.perform([request])
        
        return detectedText
    }
}

// MARK: - Python Bridge Functions

@_cdecl("swift_create_screen_capture")
public func createScreenCapture() -> UnsafeMutableRawPointer {
    let capture = SwiftScreenCapture()
    return Unmanaged.passRetained(capture).toOpaque()
}

@_cdecl("swift_start_continuous_capture")
public func startContinuousCapture(
    capturePtr: UnsafeMutableRawPointer,
    callback: @escaping @convention(c) (UnsafePointer<UInt8>, Int) -> Void
) -> Bool {
    let capture = Unmanaged<SwiftScreenCapture>.fromOpaque(capturePtr).takeUnretainedValue()
    
    return capture.startContinuousCapture { data in
        data.withUnsafeBytes { bytes in
            if let baseAddress = bytes.baseAddress {
                callback(baseAddress.assumingMemoryBound(to: UInt8.self), data.count)
            }
        }
    }
}

@_cdecl("swift_capture_screen")
public func captureScreen(capturePtr: UnsafeMutableRawPointer) -> UnsafeMutablePointer<UInt8>? {
    let capture = Unmanaged<SwiftScreenCapture>.fromOpaque(capturePtr).takeUnretainedValue()
    
    guard let data = capture.captureScreen() else { return nil }
    
    let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: data.count)
    data.copyBytes(to: buffer, count: data.count)
    return buffer
}

@_cdecl("swift_get_active_app")
public func getActiveApp(capturePtr: UnsafeMutableRawPointer) -> UnsafePointer<CChar>? {
    let capture = Unmanaged<SwiftScreenCapture>.fromOpaque(capturePtr).takeUnretainedValue()
    let appName = capture.getActiveApplication()
    return (appName as NSString).utf8String
}

@_cdecl("swift_release_screen_capture")
public func releaseScreenCapture(capturePtr: UnsafeMutableRawPointer) {
    Unmanaged<SwiftScreenCapture>.fromOpaque(capturePtr).release()
}