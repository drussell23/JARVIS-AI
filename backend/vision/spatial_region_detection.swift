//
//  SpatialRegionDetection.swift
//  JARVIS Vision Spatial Intelligence
//
//  Purpose: Native macOS region detection and UI element recognition
//  Integrates with Quadtree system for intelligent spatial analysis
//

import Foundation
import Cocoa
import Vision
import CoreGraphics
import Accelerate

/// Region of interest with metadata
struct RegionOfInterest {
    let bounds: CGRect
    let type: RegionType
    let importance: Float
    let uiElements: [UIElementInfo]
    let timestamp: Date
}

/// Types of regions detected
enum RegionType: String {
    case window = "window"
    case dialog = "dialog"
    case menu = "menu"
    case toolbar = "toolbar"
    case content = "content"
    case sidebar = "sidebar"
    case statusBar = "status_bar"
    case notification = "notification"
    case unknown = "unknown"
}

/// UI Element information
struct UIElementInfo {
    let type: String
    let bounds: CGRect
    let label: String?
    let isInteractive: Bool
}

/// Native spatial region detection for macOS
@objc class SpatialRegionDetection: NSObject {
    private let visionQueue = DispatchQueue(label: "com.jarvis.vision.spatial", qos: .userInitiated)
    private var regionCache: [String: [RegionOfInterest]] = [:]
    private let cacheTimeout: TimeInterval = 5.0
    
    // Accessibility API permissions check
    private var hasAccessibilityPermission: Bool {
        return AXIsProcessTrusted()
    }
    
    override init() {
        super.init()
        setupObservers()
    }
    
    /// Setup system observers for window changes
    private func setupObservers() {
        if hasAccessibilityPermission {
            // Monitor window events
            NSWorkspace.shared.notificationCenter.addObserver(
                self,
                selector: #selector(windowDidChange),
                name: NSWorkspace.activeSpaceDidChangeNotification,
                object: nil
            )
        }
    }
    
    @objc private func windowDidChange(_ notification: Notification) {
        // Invalidate cache on window changes
        regionCache.removeAll()
    }
    
    /// Detect all regions in current screen
    @objc func detectScreenRegions() -> [[String: Any]] {
        var regions: [[String: Any]] = []
        
        // Get all visible windows
        if let windowList = CGWindowListCopyWindowInfo([.optionOnScreenOnly], kCGNullWindowID) as? [[String: Any]] {
            
            for window in windowList {
                guard let bounds = window[kCGWindowBounds as String] as? [String: Any],
                      let x = bounds["X"] as? CGFloat,
                      let y = bounds["Y"] as? CGFloat,
                      let width = bounds["Width"] as? CGFloat,
                      let height = bounds["Height"] as? CGFloat,
                      width > 50, height > 50 else { continue }
                
                let windowBounds = CGRect(x: x, y: y, width: width, height: height)
                
                // Determine window type
                let layer = window[kCGWindowLayer as String] as? Int ?? 0
                let alpha = window[kCGWindowAlpha as String] as? CGFloat ?? 1.0
                let ownerName = window[kCGWindowOwnerName as String] as? String ?? ""
                
                let regionType = determineRegionType(
                    layer: layer,
                    alpha: alpha,
                    ownerName: ownerName,
                    bounds: windowBounds
                )
                
                // Calculate importance based on window properties
                let importance = calculateWindowImportance(
                    type: regionType,
                    bounds: windowBounds,
                    layer: layer,
                    alpha: alpha
                )
                
                regions.append([
                    "x": Int(x),
                    "y": Int(y),
                    "width": Int(width),
                    "height": Int(height),
                    "type": regionType.rawValue,
                    "importance": importance,
                    "owner": ownerName
                ])
            }
        }
        
        // Sort by importance
        regions.sort { (r1, r2) -> Bool in
            let imp1 = r1["importance"] as? Float ?? 0
            let imp2 = r2["importance"] as? Float ?? 0
            return imp1 > imp2
        }
        
        return regions
    }
    
    /// Detect UI elements within a region using Vision framework
    @objc func detectUIElementsInRegion(x: Int, y: Int, width: Int, height: Int,
                                       imageData: Data) -> [[String: Any]] {
        var elements: [[String: Any]] = []
        let semaphore = DispatchSemaphore(value: 0)
        
        // Create CGImage from data
        guard let cgImage = createCGImage(from: imageData, width: width, height: height) else {
            return elements
        }
        
        visionQueue.async { [weak self] in
            self?.performVisionAnalysis(
                on: cgImage,
                regionBounds: CGRect(x: 0, y: 0, width: width, height: height)
            ) { detectedElements in
                elements = detectedElements
                semaphore.signal()
            }
        }
        
        semaphore.wait()
        return elements
    }
    
    /// Perform Vision framework analysis
    private func performVisionAnalysis(on image: CGImage, 
                                     regionBounds: CGRect,
                                     completion: @escaping ([[String: Any]]) -> Void) {
        var elements: [[String: Any]] = []
        
        // Text detection request
        let textRequest = VNDetectTextRectanglesRequest { request, error in
            if let observations = request.results as? [VNTextObservation] {
                for observation in observations {
                    let bounds = self.convertBounds(observation.boundingBox, in: regionBounds)
                    elements.append([
                        "type": "text",
                        "x": Int(bounds.origin.x),
                        "y": Int(bounds.origin.y),
                        "width": Int(bounds.width),
                        "height": Int(bounds.height),
                        "confidence": observation.confidence
                    ])
                }
            }
        }
        textRequest.reportCharacterBoxes = true
        
        // Rectangle detection for UI elements
        let rectRequest = VNDetectRectanglesRequest { request, error in
            if let observations = request.results as? [VNRectangleObservation] {
                for observation in observations {
                    let bounds = self.convertBounds(observation.boundingBox, in: regionBounds)
                    
                    // Filter out full region rectangles
                    if bounds.width < regionBounds.width * 0.9 {
                        elements.append([
                            "type": "ui_element",
                            "x": Int(bounds.origin.x),
                            "y": Int(bounds.origin.y),
                            "width": Int(bounds.width),
                            "height": Int(bounds.height),
                            "confidence": observation.confidence
                        ])
                    }
                }
            }
        }
        
        // Barcode detection (for QR codes, etc)
        let barcodeRequest = VNDetectBarcodesRequest { request, error in
            if let observations = request.results as? [VNBarcodeObservation] {
                for observation in observations {
                    let bounds = self.convertBounds(observation.boundingBox, in: regionBounds)
                    elements.append([
                        "type": "barcode",
                        "subtype": observation.symbology.rawValue,
                        "x": Int(bounds.origin.x),
                        "y": Int(bounds.origin.y),
                        "width": Int(bounds.width),
                        "height": Int(bounds.height),
                        "confidence": observation.confidence
                    ])
                }
            }
        }
        
        // Create request handler
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        
        do {
            try handler.perform([textRequest, rectRequest, barcodeRequest])
        } catch {
            print("Vision analysis error: \(error)")
        }
        
        completion(elements)
    }
    
    /// Calculate region importance using Accelerate framework
    @objc func calculateRegionImportance(imageData: Data, width: Int, height: Int) -> Float {
        let pixelCount = width * height
        
        // Convert to grayscale for edge detection
        var grayPixels = [UInt8](repeating: 0, count: pixelCount)
        imageData.withUnsafeBytes { ptr in
            let rgbPtr = ptr.bindMemory(to: UInt8.self)
            
            // Simple RGB to grayscale conversion
            for i in 0..<pixelCount {
                let r = Float(rgbPtr[i * 3])
                let g = Float(rgbPtr[i * 3 + 1])
                let b = Float(rgbPtr[i * 3 + 2])
                grayPixels[i] = UInt8(0.299 * r + 0.587 * g + 0.114 * b)
            }
        }
        
        // Calculate edge density using Accelerate
        var edgeMagnitude = [Float](repeating: 0, count: pixelCount)
        
        // Sobel edge detection
        grayPixels.withUnsafeBufferPointer { grayBuffer in
            edgeMagnitude.withUnsafeMutableBufferPointer { edgeBuffer in
                // Simplified edge detection using differences
                for y in 1..<height-1 {
                    for x in 1..<width-1 {
                        let idx = y * width + x
                        
                        let dx = abs(Int(grayBuffer[idx + 1]) - Int(grayBuffer[idx - 1]))
                        let dy = abs(Int(grayBuffer[idx + width]) - Int(grayBuffer[idx - width]))
                        
                        edgeBuffer[idx] = Float(dx + dy) / 510.0
                    }
                }
            }
        }
        
        // Calculate statistics using Accelerate
        var mean: Float = 0
        var variance: Float = 0
        
        vDSP_meanv(edgeMagnitude, 1, &mean, vDSP_Length(pixelCount))
        vDSP_vvar(edgeMagnitude, 1, &mean, &variance, vDSP_Length(pixelCount))
        
        // Calculate importance score
        let edgeDensity = mean
        let complexity = sqrtf(variance)
        
        // Center bias
        let centerWeight: Float = 1.0 // Could be calculated based on position
        
        // Combine factors
        let importance = (0.4 * edgeDensity + 0.3 * complexity + 0.3 * centerWeight)
        
        return min(max(importance, 0.0), 1.0)
    }
    
    /// Find focused UI element
    @objc func findFocusedElement() -> [String: Any]? {
        guard hasAccessibilityPermission else {
            return ["error": "No accessibility permission"]
        }
        
        if let focusedApp = NSWorkspace.shared.frontmostApplication,
           let focusedElement = NSAccessibility.element(at: NSEvent.mouseLocation,
                                                        in: focusedApp) {
            
            var elementInfo: [String: Any] = [:]
            
            // Get element properties
            if let role = focusedElement.accessibilityRole() {
                elementInfo["role"] = String(describing: role)
            }
            
            if let title = focusedElement.accessibilityTitle() {
                elementInfo["title"] = title
            }
            
            if let frame = focusedElement.accessibilityFrame() {
                elementInfo["x"] = Int(frame.origin.x)
                elementInfo["y"] = Int(frame.origin.y)
                elementInfo["width"] = Int(frame.width)
                elementInfo["height"] = Int(frame.height)
            }
            
            elementInfo["app"] = focusedApp.localizedName ?? "Unknown"
            
            return elementInfo
        }
        
        return nil
    }
    
    /// Optimize regions for processing
    @objc func optimizeRegionsForBatch(regions: [[String: Any]], 
                                      maxBatchSize: Int) -> [[[String: Any]]] {
        var batches: [[[String: Any]]] = []
        var currentBatch: [[String: Any]] = []
        
        // Sort regions by importance and spatial locality
        let sortedRegions = regions.sorted { r1, r2 in
            let imp1 = r1["importance"] as? Float ?? 0
            let imp2 = r2["importance"] as? Float ?? 0
            
            if abs(imp1 - imp2) > 0.1 {
                return imp1 > imp2
            }
            
            // Sort by spatial proximity for cache efficiency
            let x1 = r1["x"] as? Int ?? 0
            let y1 = r1["y"] as? Int ?? 0
            let x2 = r2["x"] as? Int ?? 0
            let y2 = r2["y"] as? Int ?? 0
            
            return (x1 + y1) < (x2 + y2)
        }
        
        // Create batches
        for region in sortedRegions {
            currentBatch.append(region)
            
            if currentBatch.count >= maxBatchSize {
                batches.append(currentBatch)
                currentBatch = []
            }
        }
        
        if !currentBatch.isEmpty {
            batches.append(currentBatch)
        }
        
        return batches
    }
    
    // MARK: - Helper Methods
    
    private func determineRegionType(layer: Int, alpha: CGFloat, 
                                   ownerName: String, bounds: CGRect) -> RegionType {
        // Menu bar
        if layer > 20 || bounds.origin.y < 30 {
            return .statusBar
        }
        
        // Notification
        if ownerName.contains("NotificationCenter") {
            return .notification
        }
        
        // Dialog (small windows with high layer)
        if layer > 0 && bounds.width < 600 && bounds.height < 400 {
            return .dialog
        }
        
        // Menu
        if alpha < 1.0 && layer > 0 {
            return .menu
        }
        
        // Toolbar (wide and short)
        if bounds.width / bounds.height > 5 {
            return .toolbar
        }
        
        // Sidebar (tall and narrow)
        if bounds.height / bounds.width > 3 {
            return .sidebar
        }
        
        // Default to window/content
        return layer == 0 ? .window : .content
    }
    
    private func calculateWindowImportance(type: RegionType, bounds: CGRect,
                                         layer: Int, alpha: CGFloat) -> Float {
        var importance: Float = 0.5
        
        // Type-based importance
        switch type {
        case .notification:
            importance = 0.9
        case .dialog:
            importance = 0.85
        case .menu:
            importance = 0.8
        case .window:
            importance = 0.7
        case .toolbar:
            importance = 0.6
        case .sidebar:
            importance = 0.5
        case .content:
            importance = 0.6
        case .statusBar:
            importance = 0.4
        case .unknown:
            importance = 0.3
        }
        
        // Adjust for layer (higher layer = more important)
        importance += Float(layer) * 0.01
        
        // Adjust for alpha (more opaque = more important)
        importance *= Float(alpha)
        
        // Center bias
        if let screen = NSScreen.main {
            let screenBounds = screen.frame
            let centerX = screenBounds.midX
            let centerY = screenBounds.midY
            
            let distanceFromCenter = hypot(
                bounds.midX - centerX,
                bounds.midY - centerY
            )
            
            let maxDistance = hypot(screenBounds.width / 2, screenBounds.height / 2)
            let centerBias = 1.0 - Float(distanceFromCenter / maxDistance)
            
            importance = importance * 0.8 + centerBias * 0.2
        }
        
        return min(max(importance, 0.0), 1.0)
    }
    
    private func convertBounds(_ normalizedBounds: CGRect, in regionBounds: CGRect) -> CGRect {
        return CGRect(
            x: normalizedBounds.origin.x * regionBounds.width,
            y: (1 - normalizedBounds.origin.y - normalizedBounds.height) * regionBounds.height,
            width: normalizedBounds.width * regionBounds.width,
            height: normalizedBounds.height * regionBounds.height
        )
    }
    
    private func createCGImage(from data: Data, width: Int, height: Int) -> CGImage? {
        let bitsPerComponent = 8
        let bitsPerPixel = 32
        let bytesPerRow = width * 4
        
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }
        
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bitsPerPixel: bitsPerPixel,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )
    }
}

/// Bridge to Python
@objc class SpatialRegionBridge: NSObject {
    private let detector = SpatialRegionDetection()
    
    @objc func detectRegions() -> [[String: Any]] {
        return detector.detectScreenRegions()
    }
    
    @objc func detectUIElements(x: Int, y: Int, width: Int, height: Int, 
                               imageData: Data) -> [[String: Any]] {
        return detector.detectUIElementsInRegion(x: x, y: y, width: width, 
                                               height: height, imageData: imageData)
    }
    
    @objc func calculateImportance(imageData: Data, width: Int, height: Int) -> Float {
        return detector.calculateRegionImportance(imageData: imageData, 
                                                width: width, height: height)
    }
    
    @objc func getFocusedElement() -> [String: Any]? {
        return detector.findFocusedElement()
    }
    
    @objc func optimizeBatches(regions: [[String: Any]], 
                              maxSize: Int) -> [[[String: Any]]] {
        return detector.optimizeRegionsForBatch(regions: regions, 
                                              maxBatchSize: maxSize)
    }
}