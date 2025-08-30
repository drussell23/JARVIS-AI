import Foundation
import CoreGraphics
import CoreImage
import Vision
import Metal
import MetalPerformanceShaders

/// High-performance vision processor using Metal and CoreML
/// Optimized for Apple Silicon with minimal memory usage
@available(macOS 11.0, *)
public class VisionProcessor {
    
    // MARK: - Properties
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let context: CIContext
    private var textureCache: CVMetalTextureCache?
    
    // Vision request handlers
    private lazy var faceDetector = VNDetectFaceRectanglesRequest()
    private lazy var textDetector = VNRecognizeTextRequest()
    private lazy var objectDetector = VNDetectRectanglesRequest()
    
    // Dynamic configuration
    private var config: VisionConfig
    
    // Performance tracking
    private var processingTimes: [Double] = []
    private let maxSamples = 100
    
    // MARK: - Types
    public struct VisionConfig: Codable {
        var maxImageSize: CGSize = CGSize(width: 1920, height: 1080)
        var compressionQuality: Float = 0.8
        var enableFaceDetection: Bool = true
        var enableTextRecognition: Bool = true
        var enableObjectDetection: Bool = true
        var cacheSize: Int = 50
        var processingTimeout: Double = 1.0
    }
    
    public struct VisionResult: Codable {
        let faces: [CGRect]
        let text: [TextObservation]
        let objects: [CGRect]
        let processingTime: Double
        let memoryUsed: Int
        let timestamp: Double
    }
    
    public struct TextObservation: Codable {
        let text: String
        let confidence: Float
        let boundingBox: CGRect
    }
    
    // MARK: - Initialization
    public init(config: VisionConfig? = nil) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw VisionError.metalNotAvailable
        }
        
        guard let queue = device.makeCommandQueue() else {
            throw VisionError.commandQueueCreationFailed
        }
        
        self.device = device
        self.commandQueue = queue
        self.context = CIContext(mtlDevice: device)
        self.config = config ?? VisionConfig()
        
        // Create texture cache for efficient CPU-GPU transfer
        var cache: CVMetalTextureCache?
        let result = CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        guard result == kCVReturnSuccess, let textureCache = cache else {
            throw VisionError.textureCacheCreationFailed
        }
        self.textureCache = textureCache
        
        // Configure Vision requests for performance
        configureVisionRequests()
    }
    
    // MARK: - Public Methods
    
    /// Process screen capture with Metal acceleration
    public func processScreenCapture(_ cgImage: CGImage) async throws -> VisionResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Resize if needed to respect memory constraints
        let resizedImage = try resizeImageIfNeeded(cgImage)
        
        // Convert to CIImage for Metal processing
        let ciImage = CIImage(cgImage: resizedImage)
        
        // Process in parallel using Vision framework
        async let faces = detectFaces(in: ciImage)
        async let text = recognizeText(in: ciImage)
        async let objects = detectObjects(in: ciImage)
        
        // Wait for all results
        let (faceResults, textResults, objectResults) = try await (faces, text, objects)
        
        let processingTime = CFAbsoluteTimeGetCurrent() - startTime
        
        // Track performance
        updatePerformanceMetrics(processingTime)
        
        return VisionResult(
            faces: faceResults,
            text: textResults,
            objects: objectResults,
            processingTime: processingTime,
            memoryUsed: getCurrentMemoryUsage(),
            timestamp: startTime
        )
    }
    
    /// Compress image with quality preservation
    public func compressImage(_ cgImage: CGImage, quality: Float? = nil) -> Data? {
        let compressionQuality = quality ?? config.compressionQuality
        
        // Use Metal for efficient compression
        let ciImage = CIImage(cgImage: cgImage)
        
        // Apply compression filter
        let filter = CIFilter(name: "CIPhotoEffectProcess")
        filter?.setValue(ciImage, forKey: kCIInputImageKey)
        
        guard let outputImage = filter?.outputImage else { return nil }
        
        // Render with compression using Data
        guard let cgImage = context.createCGImage(outputImage, from: outputImage.extent) else { return nil }
        guard let data = CFDataCreateMutable(nil, 0) else { return nil }
        guard let destination = CGImageDestinationCreateWithData(data, kUTTypeJPEG, 1, nil) else { return nil }
        
        let options: [CFString: Any] = [
            kCGImageDestinationLossyCompressionQuality: compressionQuality
        ]
        CGImageDestinationAddImage(destination, cgImage, options as CFDictionary)
        
        guard CGImageDestinationFinalize(destination) else { return nil }
        
        return data as Data
    }
    
    /// Extract text regions for focused OCR
    public func extractTextRegions(_ cgImage: CGImage) async throws -> [CGImage] {
        let ciImage = CIImage(cgImage: cgImage)
        let textRegions = try await recognizeText(in: ciImage)
        
        var extractedImages: [CGImage] = []
        
        for region in textRegions {
            // Convert normalized coordinates to pixel coordinates
            let rect = CGRect(
                x: region.boundingBox.origin.x * CGFloat(cgImage.width),
                y: region.boundingBox.origin.y * CGFloat(cgImage.height),
                width: region.boundingBox.width * CGFloat(cgImage.width),
                height: region.boundingBox.height * CGFloat(cgImage.height)
            )
            
            if let croppedImage = cgImage.cropping(to: rect) {
                extractedImages.append(croppedImage)
            }
        }
        
        return extractedImages
    }
    
    // MARK: - Private Methods
    
    private func configureVisionRequests() {
        // Configure for speed over accuracy where appropriate
        faceDetector.preferBackgroundProcessing = false
        
        textDetector.recognitionLevel = .fast
        textDetector.usesLanguageCorrection = false
        textDetector.minimumTextHeight = 0.05
        
        objectDetector.maximumObservations = 10
    }
    
    private func resizeImageIfNeeded(_ image: CGImage) throws -> CGImage {
        let imageSize = CGSize(width: image.width, height: image.height)
        
        // Check if resize is needed
        if imageSize.width <= config.maxImageSize.width &&
           imageSize.height <= config.maxImageSize.height {
            return image
        }
        
        // Calculate scale factor
        let widthScale = config.maxImageSize.width / imageSize.width
        let heightScale = config.maxImageSize.height / imageSize.height
        let scale = min(widthScale, heightScale)
        
        let newSize = CGSize(
            width: imageSize.width * scale,
            height: imageSize.height * scale
        )
        
        // Use Metal for efficient resizing
        guard let resized = resizeWithMetal(image, to: newSize) else {
            throw VisionError.resizeFailed
        }
        
        return resized
    }
    
    private func resizeWithMetal(_ image: CGImage, to size: CGSize) -> CGImage? {
        let ciImage = CIImage(cgImage: image)
        
        let scaleX = size.width / CGFloat(image.width)
        let scaleY = size.height / CGFloat(image.height)
        
        let transform = CGAffineTransform(scaleX: scaleX, y: scaleY)
        let scaledImage = ciImage.transformed(by: transform)
        
        guard let cgImage = context.createCGImage(scaledImage, from: scaledImage.extent) else {
            return nil
        }
        
        return cgImage
    }
    
    private func detectFaces(in image: CIImage) async throws -> [CGRect] {
        guard config.enableFaceDetection else { return [] }
        
        return try await withCheckedThrowingContinuation { continuation in
            let handler = VNImageRequestHandler(ciImage: image, options: [:])
            
            do {
                try handler.perform([faceDetector])
                
                let faces = faceDetector.results?.compactMap { observation in
                    observation.boundingBox
                } ?? []
                
                continuation.resume(returning: faces)
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
    
    private func recognizeText(in image: CIImage) async throws -> [TextObservation] {
        guard config.enableTextRecognition else { return [] }
        
        return try await withCheckedThrowingContinuation { continuation in
            let handler = VNImageRequestHandler(ciImage: image, options: [:])
            
            do {
                try handler.perform([textDetector])
                
                let observations = textDetector.results?.compactMap { observation -> TextObservation? in
                    guard let topCandidate = observation.topCandidates(1).first else { return nil }
                    
                    return TextObservation(
                        text: topCandidate.string,
                        confidence: observation.confidence,
                        boundingBox: observation.boundingBox
                    )
                } ?? []
                
                continuation.resume(returning: observations)
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
    
    private func detectObjects(in image: CIImage) async throws -> [CGRect] {
        guard config.enableObjectDetection else { return [] }
        
        return try await withCheckedThrowingContinuation { continuation in
            let handler = VNImageRequestHandler(ciImage: image, options: [:])
            
            do {
                try handler.perform([objectDetector])
                
                let objects = objectDetector.results?.compactMap { observation in
                    observation.boundingBox
                } ?? []
                
                continuation.resume(returning: objects)
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
    
    private func updatePerformanceMetrics(_ time: Double) {
        processingTimes.append(time)
        if processingTimes.count > maxSamples {
            processingTimes.removeFirst()
        }
    }
    
    private func getCurrentMemoryUsage() -> Int {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        return result == KERN_SUCCESS ? Int(info.resident_size / 1024 / 1024) : 0
    }
}

// MARK: - Error Types
public enum VisionError: Error {
    case metalNotAvailable
    case commandQueueCreationFailed
    case textureCacheCreationFailed
    case resizeFailed
    case processingTimeout
}

// MARK: - C Interface for Python
@available(macOS 11.0, *)
@_cdecl("vision_processor_create")
public func vision_processor_create() -> UnsafeMutableRawPointer? {
    do {
        let processor = try VisionProcessor()
        return Unmanaged.passRetained(processor).toOpaque()
    } catch {
        return nil
    }
}

@available(macOS 11.0, *)
@_cdecl("vision_processor_process_image")
public func vision_processor_process_image(
    processor: UnsafeMutableRawPointer,
    imageData: UnsafePointer<UInt8>,
    imageSize: Int,
    resultCallback: @escaping (UnsafePointer<CChar>) -> Void
) {
    let visionProcessor = Unmanaged<VisionProcessor>.fromOpaque(processor).takeUnretainedValue()
    
    let data = Data(bytes: imageData, count: imageSize)
    guard let cgImage = CGImage(jpegDataProviderSource: CGDataProvider(data: data as CFData)!,
                                decode: nil,
                                shouldInterpolate: true,
                                intent: .defaultIntent) else {
        resultCallback("{\"error\": \"Invalid image data\"}")
        return
    }
    
    Task {
        do {
            let result = try await visionProcessor.processScreenCapture(cgImage)
            let encoder = JSONEncoder()
            let jsonData = try encoder.encode(result)
            let jsonString = String(data: jsonData, encoding: .utf8) ?? "{}"
            
            jsonString.withCString { cString in
                resultCallback(cString)
            }
        } catch {
            resultCallback("{\"error\": \"\(error)\"}")
        }
    }
}

@available(macOS 11.0, *)
@_cdecl("vision_processor_destroy")
public func vision_processor_destroy(processor: UnsafeMutableRawPointer) {
    Unmanaged<VisionProcessor>.fromOpaque(processor).release()
}