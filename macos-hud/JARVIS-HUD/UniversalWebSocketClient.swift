import Foundation
import Combine

/// Universal WebSocket Client for JARVIS
/// Supports dynamic configuration, health checks, capability negotiation, and state synchronization
/// Works with any backend WebSocket server following the JARVIS protocol v2.0
class UniversalWebSocketClient: NSObject, ObservableObject {

    // MARK: - Configuration

    struct Configuration: Codable {
        let version: String
        struct WebSocketConfig: Codable {
            let url: String
            let ready: Bool
            let healthCheckRequired: Bool
            struct ReconnectConfig: Codable {
                let enabled: Bool
                let maxAttempts: Int
                let baseDelayMs: Int
                let maxDelayMs: Int
                let jitter: Bool

                enum CodingKeys: String, CodingKey {
                    case enabled
                    case maxAttempts = "max_attempts"
                    case baseDelayMs = "base_delay_ms"
                    case maxDelayMs = "max_delay_ms"
                    case jitter
                }
            }
            let reconnect: ReconnectConfig

            enum CodingKeys: String, CodingKey {
                case url, ready
                case healthCheckRequired = "health_check_required"
                case reconnect
            }
        }
        struct HTTPConfig: Codable {
            let baseUrl: String
            let healthEndpoint: String
            let configEndpoint: String

            enum CodingKeys: String, CodingKey {
                case baseUrl = "base_url"
                case healthEndpoint = "health_endpoint"
                case configEndpoint = "config_endpoint"
            }
        }
        let websocket: WebSocketConfig
        let http: HTTPConfig
        let capabilities: [String]
    }

    struct HealthCheckResponse: Codable {
        let status: String
        let websocketReady: Bool
        let version: String
        let uptime: Double
        let capabilities: [String]

        enum CodingKeys: String, CodingKey {
            case status
            case websocketReady = "websocket_ready"
            case version, uptime, capabilities
        }
    }

    struct WelcomeMessage: Codable {
        let type: String
        let message: String
        let serverVersion: String
        let protocolVersion: String
        let capabilities: [String]
        let bufferedMessagesCount: Int

        enum CodingKeys: String, CodingKey {
            case type, message
            case serverVersion = "server_version"
            case protocolVersion = "protocol_version"
            case capabilities
            case bufferedMessagesCount = "buffered_messages_count"
        }
    }

    // MARK: - Published Properties

    @Published var connectionState: ConnectionState = .disconnected
    @Published var serverVersion: String = "unknown"
    @Published var serverCapabilities: [String] = []
    @Published var lastError: String?

    enum ConnectionState {
        case disconnected
        case fetchingConfig
        case healthCheck
        case connecting
        case connected
        case reconnecting(attempt: Int)
        case error(String)
    }

    // MARK: - Private Properties

    private var websocketTask: URLSessionWebSocketTask?
    private var configuration: Configuration?
    private var reconnectAttempt = 0
    private var reconnectTimer: Timer?
    private var messageHandler: ((String) -> Void)?

    private let clientId: String
    private let clientType: String
    private let clientCapabilities: [String]
    private let defaultBaseURL: String

    // MARK: - Initialization

    init(
        clientId: String = UUID().uuidString,
        clientType: String = "hud",
        capabilities: [String] = ["hud_client", "voice", "vision"],
        baseURL: String = "http://localhost:8010"
    ) {
        self.clientId = clientId
        self.clientType = clientType
        self.clientCapabilities = capabilities
        self.defaultBaseURL = baseURL
        super.init()
    }

    // MARK: - Public API

    /// Connect to the WebSocket server with full protocol flow
    func connect(messageHandler: @escaping (String) -> Void) {
        self.messageHandler = messageHandler

        Task {
            await performConnectionFlow()
        }
    }

    /// Disconnect from WebSocket
    func disconnect() {
        reconnectTimer?.invalidate()
        reconnectTimer = nil

        websocketTask?.cancel(with: .goingAway, reason: nil)
        websocketTask = nil

        DispatchQueue.main.async {
            self.connectionState = .disconnected
        }
    }

    /// Send a message to the server
    func send(_ message: [String: Any]) {
        guard let websocketTask = websocketTask else {
            print("❌ Cannot send message: WebSocket not connected")
            return
        }

        do {
            let jsonData = try JSONSerialization.data(withJSONObject: message)
            let jsonString = String(data: jsonData, encoding: .utf8) ?? ""
            let message = URLSessionWebSocketTask.Message.string(jsonString)

            websocketTask.send(message) { error in
                if let error = error {
                    print("❌ WebSocket send error: \(error)")
                }
            }
        } catch {
            print("❌ Failed to serialize message: \(error)")
        }
    }

    // MARK: - Connection Flow

    private func performConnectionFlow() async {
        print("🚀 Starting Universal WebSocket Connection Flow...")

        // Step 1: Fetch configuration
        await fetchConfiguration()

        guard let config = configuration else {
            await setError("Failed to fetch configuration")
            return
        }

        // Step 2: Health check (if required)
        if config.websocket.healthCheckRequired {
            let healthy = await performHealthCheck(baseURL: config.http.baseUrl)
            if !healthy {
                await setError("Health check failed - backend not ready")
                await scheduleReconnect()
                return
            }
        }

        // Step 3: Connect to WebSocket
        await connectToWebSocket(url: config.websocket.url)
    }

    private func fetchConfiguration() async {
        await MainActor.run {
            connectionState = .fetchingConfig
        }

        print("📡 Fetching configuration from backend...")

        let configURL = URL(string: "\(defaultBaseURL)/api/config")!

        do {
            let (data, response) = try await URLSession.shared.data(from: configURL)

            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                print("❌ Config endpoint returned non-200 status")
                // Fallback to defaults
                await useFallbackConfiguration()
                return
            }

            let decoder = JSONDecoder()
            let config = try decoder.decode(Configuration.self, from: data)

            self.configuration = config
            print("✅ Configuration fetched successfully")
            print("   WebSocket URL: \(config.websocket.url)")
            print("   Server capabilities: \(config.capabilities.joined(separator: ", "))")

        } catch {
            print("⚠️  Failed to fetch config: \(error)")
            print("   Using fallback configuration...")
            await useFallbackConfiguration()
        }
    }

    private func useFallbackConfiguration() async {
        // Create fallback config with defaults
        let fallbackConfig = Configuration(
            version: "2.0.0",
            websocket: Configuration.WebSocketConfig(
                url: "ws://localhost:8010/ws",
                ready: true,
                healthCheckRequired: true,
                reconnect: Configuration.WebSocketConfig.ReconnectConfig(
                    enabled: true,
                    maxAttempts: 999,
                    baseDelayMs: 500,
                    maxDelayMs: 15000,
                    jitter: true
                )
            ),
            http: Configuration.HTTPConfig(
                baseUrl: defaultBaseURL,
                healthEndpoint: "/health",
                configEndpoint: "/api/config"
            ),
            capabilities: ["voice", "vision", "commands"]
        )

        self.configuration = fallbackConfig
        print("✅ Fallback configuration loaded")
    }

    private func performHealthCheck(baseURL: String) async -> Bool {
        await MainActor.run {
            connectionState = .healthCheck
        }

        print("🏥 Performing health check...")

        let healthURL = URL(string: "\(baseURL)/health")!

        do {
            let (data, response) = try await URLSession.shared.data(from: healthURL)

            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                print("❌ Health check failed: non-200 status")
                return false
            }

            let decoder = JSONDecoder()
            let health = try decoder.decode(HealthCheckResponse.self, from: data)

            print("✅ Health check passed")
            print("   Status: \(health.status)")
            print("   WebSocket Ready: \(health.websocketReady)")
            print("   Version: \(health.version)")
            print("   Uptime: \(Int(health.uptime))s")

            await MainActor.run {
                serverVersion = health.version
                serverCapabilities = health.capabilities
            }

            return health.websocketReady && (health.status == "healthy" || health.status == "degraded")

        } catch {
            print("❌ Health check error: \(error)")
            return false
        }
    }

    private func connectToWebSocket(url: String) async {
        await MainActor.run {
            connectionState = .connecting
        }

        print("🔌 Connecting to WebSocket: \(url)")

        guard let websocketURL = URL(string: url) else {
            await setError("Invalid WebSocket URL")
            return
        }

        let session = URLSession(configuration: .default)
        let task = session.webSocketTask(with: websocketURL)

        self.websocketTask = task
        task.resume()

        // Send capability negotiation handshake
        await sendHandshake()

        // Start receiving messages
        await receiveMessages()

        await MainActor.run {
            connectionState = .connected
            reconnectAttempt = 0
        }

        print("✅ WebSocket connected successfully")
    }

    private func sendHandshake() async {
        let handshake: [String: Any] = [
            "type": "client_connect",
            "client_id": clientId,
            "client_type": clientType,
            "version": "2.0.0",
            "capabilities": clientCapabilities,
            "request_state": true
        ]

        send(handshake)
        print("📤 Sent capability negotiation handshake")
    }

    private func receiveMessages() async {
        guard let task = websocketTask else { return }

        do {
            let message = try await task.receive()

            switch message {
            case .string(let text):
                handleMessage(text)

            case .data(let data):
                if let text = String(data: data, encoding: .utf8) {
                    handleMessage(text)
                }

            @unknown default:
                print("⚠️  Unknown message type")
            }

            // Continue receiving
            await receiveMessages()

        } catch {
            print("❌ WebSocket receive error: \(error)")
            await handleDisconnection()
        }
    }

    private func handleMessage(_ text: String) {
        // Parse message to detect welcome
        if let data = text.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let type = json["type"] as? String,
           type == "welcome" {

            // Extract server info from welcome message
            if let decoder = try? JSONDecoder(),
               let welcomeData = try? JSONDecoder().decode(WelcomeMessage.self, from: data) {
                DispatchQueue.main.async {
                    self.serverVersion = welcomeData.serverVersion
                    self.serverCapabilities = welcomeData.capabilities
                }
                print("📨 Received welcome message")
                print("   Server version: \(welcomeData.serverVersion)")
                print("   Protocol version: \(welcomeData.protocolVersion)")
                print("   Buffered messages: \(welcomeData.bufferedMessagesCount)")
            }
        }

        // Forward to message handler
        messageHandler?(text)
    }

    private func handleDisconnection() async {
        print("🔌 WebSocket disconnected")

        websocketTask = nil

        await scheduleReconnect()
    }

    private func scheduleReconnect() async {
        guard let config = configuration,
              config.websocket.reconnect.enabled else {
            await setError("Reconnection disabled")
            return
        }

        reconnectAttempt += 1

        guard reconnectAttempt <= config.websocket.reconnect.maxAttempts else {
            await setError("Max reconnection attempts reached")
            return
        }

        await MainActor.run {
            connectionState = .reconnecting(attempt: reconnectAttempt)
        }

        // Calculate backoff delay
        let baseDelay = Double(config.websocket.reconnect.baseDelayMs) / 1000.0
        let maxDelay = Double(config.websocket.reconnect.maxDelayMs) / 1000.0

        var delay = min(baseDelay * pow(2.0, Double(reconnectAttempt - 1)), maxDelay)

        // Add jitter
        if config.websocket.reconnect.jitter {
            let jitter = Double.random(in: -0.1...0.1) * delay
            delay += jitter
        }

        print("🔄 Reconnecting in \(String(format: "%.1f", delay))s (attempt \(reconnectAttempt)/\(config.websocket.reconnect.maxAttempts))")

        try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))

        await performConnectionFlow()
    }

    private func setError(_ message: String) async {
        await MainActor.run {
            connectionState = .error(message)
            lastError = message
        }
        print("❌ \(message)")
    }
}
