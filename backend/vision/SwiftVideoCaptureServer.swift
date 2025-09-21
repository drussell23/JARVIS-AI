#!/usr/bin/env swift
//
// SwiftVideoCaptureServer.swift
// Persistent server for macOS screen recording with purple indicator
//

import Foundation
import AVFoundation
import CoreGraphics
import CoreMedia
import CoreVideo
import AppKit

// Global capture instance
var globalCapture: SwiftVideoCapture?

// Keep the capture session alive
class PersistentVideoCapture: NSObject {
    var captureSession: AVCaptureSession?
    var isRunning = false
    
    @objc func startCapture() -> Bool {
        guard captureSession == nil else {
            print("Capture already running")
            return true
        }
        
        // Create and configure session
        captureSession = AVCaptureSession()
        guard let session = captureSession else { return false }
        
        // Set quality
        session.sessionPreset = .hd1920x1080
        
        // Create screen input
        guard let screenInput = AVCaptureScreenInput(displayID: CGMainDisplayID()) else {
            print("Failed to create screen input")
            return false
        }
        
        // Configure input
        screenInput.minFrameDuration = CMTimeMake(value: 1, timescale: 30)
        screenInput.capturesCursor = true
        screenInput.capturesMouseClicks = true
        
        // Add input
        if session.canAddInput(screenInput) {
            session.addInput(screenInput)
        } else {
            print("Cannot add screen input")
            return false
        }
        
        // Create output
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.alwaysDiscardsLateVideoFrames = true
        
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }
        
        // Start running - this triggers the purple indicator
        session.startRunning()
        isRunning = true
        
        print("✅ Capture started - purple indicator should be visible")
        return true
    }
    
    @objc func stopCapture() {
        if let session = captureSession {
            session.stopRunning()
            captureSession = nil
            isRunning = false
            print("✅ Capture stopped - purple indicator should disappear")
        }
    }
    
    @objc func isCapturing() -> Bool {
        return captureSession?.isRunning ?? false
    }
}

// Simple HTTP server for commands
class VideoCaptureServer {
    let capture = PersistentVideoCapture()
    
    func run() {
        print("🎥 Swift Video Capture Server Started")
        print("Listening on port 9876...")
        
        // Create a simple TCP server
        let server = ServerSocket(port: 9876)
        
        while true {
            if let client = server.accept() {
                let request = client.readLine()
                
                var response = ""
                
                switch request {
                case "START":
                    let success = capture.startCapture()
                    response = success ? "OK:STARTED" : "ERROR:FAILED"
                    
                case "STOP":
                    capture.stopCapture()
                    response = "OK:STOPPED"
                    
                case "STATUS":
                    let isCapturing = capture.isCapturing()
                    response = "OK:CAPTURING=\(isCapturing)"
                    
                case "PING":
                    response = "OK:PONG"
                    
                default:
                    response = "ERROR:UNKNOWN_COMMAND"
                }
                
                client.write(response)
                client.close()
            }
        }
    }
}

// Simple socket wrapper
class ServerSocket {
    private var socketfd: Int32 = -1
    
    init(port: Int) {
        // Create socket
        socketfd = socket(AF_INET, SOCK_STREAM, 0)
        
        // Allow reuse
        var yes: Int32 = 1
        setsockopt(socketfd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
        
        // Bind
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = in_port_t(port).bigEndian
        addr.sin_addr.s_addr = INADDR_ANY.bigEndian
        
        withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPtr in
                bind(socketfd, sockaddrPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        
        // Listen
        listen(socketfd, 5)
    }
    
    func accept() -> ClientSocket? {
        var clientAddr = sockaddr_in()
        var clientAddrLen = socklen_t(MemoryLayout<sockaddr_in>.size)
        
        let clientfd = withUnsafeMutablePointer(to: &clientAddr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPtr in
                Darwin.accept(socketfd, sockaddrPtr, &clientAddrLen)
            }
        }
        
        return clientfd >= 0 ? ClientSocket(fd: clientfd) : nil
    }
}

class ClientSocket {
    private let fd: Int32
    
    init(fd: Int32) {
        self.fd = fd
    }
    
    func readLine() -> String? {
        var buffer = [UInt8](repeating: 0, count: 1024)
        let bytesRead = recv(fd, &buffer, buffer.count, 0)
        
        if bytesRead > 0 {
            return String(bytes: buffer[0..<bytesRead], encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return nil
    }
    
    func write(_ message: String) {
        if let data = message.data(using: .utf8) {
            data.withUnsafeBytes { bytes in
                send(fd, bytes.baseAddress, data.count, 0)
            }
        }
    }
    
    func close() {
        Darwin.close(fd)
    }
}

// For command line usage
if CommandLine.arguments.count > 1 {
    let command = CommandLine.arguments[1]
    
    // Try to connect to running server
    let socketfd = socket(AF_INET, SOCK_STREAM, 0)
    
    var addr = sockaddr_in()
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = in_port_t(9876).bigEndian
    addr.sin_addr.s_addr = inet_addr("127.0.0.1")
    
    let connected = withUnsafePointer(to: &addr) { ptr in
        ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPtr in
            connect(socketfd, sockaddrPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
        }
    } == 0
    
    if connected {
        // Send command to server
        send(socketfd, command, command.count, 0)
        
        // Read response
        var buffer = [UInt8](repeating: 0, count: 1024)
        let bytesRead = recv(socketfd, &buffer, buffer.count, 0)
        
        if bytesRead > 0,
           let response = String(bytes: buffer[0..<bytesRead], encoding: .utf8) {
            print(response)
        }
        
        Darwin.close(socketfd)
    } else if command == "server" {
        // Start server
        let server = VideoCaptureServer()
        server.run()
    } else {
        print("ERROR:SERVER_NOT_RUNNING")
        print("Start server with: swift SwiftVideoCaptureServer.swift server")
    }
} else {
    print("Usage:")
    print("  Start server: swift SwiftVideoCaptureServer.swift server")
    print("  Send command: swift SwiftVideoCaptureServer.swift [START|STOP|STATUS|PING]")
}