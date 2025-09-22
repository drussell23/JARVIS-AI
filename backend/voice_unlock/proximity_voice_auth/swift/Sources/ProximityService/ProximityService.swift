import Foundation
import JARVISProximityAuth

@main
struct ProximityService {
    static func main() async {
        print("🚀 JARVIS Proximity Service Starting...")
        
        let detector = AppleWatchDetector()
        let bridge = HTTPBridge(port: 5555)
        
        // Start detector
        detector.startDetection()
        
        // Simple HTTP server for IPC
        await startHTTPServer(detector: detector)
    }
    
    static func startHTTPServer(detector: AppleWatchDetector) async {
        // For now, use a polling mechanism with file-based IPC
        print("📡 Starting IPC service on port 5555...")
        
        let requestFile = "/tmp/jarvis_proximity_request.json"
        let responseFile = "/tmp/jarvis_proximity_response.json"
        
        while true {
            // Check for requests
            if let requestData = try? Data(contentsOf: URL(fileURLWithPath: requestFile)) {
                do {
                    let request = try JSONDecoder().decode(ProximityRequest.self, from: requestData)
                    
                    // Process request
                    let response = processRequest(request, detector: detector)
                    
                    // Write response
                    let responseData = try JSONEncoder().encode(response)
                    try responseData.write(to: URL(fileURLWithPath: responseFile))
                    
                    // Clear request file
                    try? FileManager.default.removeItem(atPath: requestFile)
                    
                } catch {
                    print("❌ Error processing request: \(error)")
                }
            }
            
            // Small delay to prevent busy waiting
            try? await Task.sleep(nanoseconds: 100_000_000) // 0.1 seconds
        }
    }
    
    static func processRequest(_ request: ProximityRequest, detector: AppleWatchDetector) -> ProximityResponse {
        switch request.command {
        case "get_status":
            return ProximityResponse(
                command: "status",
                status: "success",
                data: ProximityResponseData(running: true)
            )
            
        case "get_proximity":
            let status = detector.getProximityStatus()
            return ProximityResponse(
                command: "proximity",
                status: "success",
                data: ProximityResponseData(
                    isNearby: status.isNearby,
                    confidence: status.confidence,
                    distance: status.distance,
                    deviceCount: detector.getDetectedDevices().count
                )
            )
            
        default:
            return ProximityResponse(
                command: request.command,
                status: "error",
                data: ProximityResponseData(error: "Unknown command")
            )
        }
    }
}

// Request/Response models
struct ProximityRequest: Codable {
    let command: String
    let timestamp: Double
}

struct ProximityResponseData: Codable {
    var isNearby: Bool?
    var confidence: Double?
    var distance: Double?
    var deviceCount: Int?
    var running: Bool?
    var error: String?
}

struct ProximityResponse: Codable {
    let command: String
    let status: String
    let data: ProximityResponseData
}