//
//  PythonBridge.swift
//  JARVIS-HUD
//
//  Bridge to communicate with Python FastAPI backend
//  Handles WebSocket connections and HTTP API calls
//

import Foundation
import Combine

// MARK: - File Logger for Debugging

/// File-based logger since GUI apps don't show console output in terminal
class FileLogger {
    static let shared = FileLogger()
    private let logFileURL: URL
    private let fileHandle: FileHandle?

    init() {
        // Log to /tmp/jarvis_hud.log for easy inspection
        logFileURL = URL(fileURLWithPath: "/tmp/jarvis_hud.log")

        // Create or truncate log file
        if !FileManager.default.fileExists(atPath: logFileURL.path) {
            FileManager.default.createFile(atPath: logFileURL.path, contents: nil)
        }

        fileHandle = try? FileHandle(forWritingTo: logFileURL)
        fileHandle?.truncateFile(atOffset: 0)  // Clear previous logs

        log("🚀 JARVIS HUD Logger initialized at \(logFileURL.path)")
        log("📅 Session start: \(Date())")
        log(String(repeating: "=", count: 80))
    }

    func log(_ message: String) {
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let logLine = "[\(timestamp)] \(message)\n"

        // Write to console
        print(message)

        // Write to file
        if let data = logLine.data(using: .utf8) {
            fileHandle?.write(data)
        }
    }

    deinit {
        try? fileHandle?.close()
    }
}

/// WebSocket connection to Python backend
/// Enhanced with UniversalWebSocketClient for robust connection management
class PythonBridge: ObservableObject {

    // MARK: - Published State

    @Published var connectionStatus: ConnectionStatus = .disconnected
    @Published var transcriptMessages: [TranscriptMessage] = []
    @Published var hudState: HUDState = .offline
    @Published var loadingProgress: Int = 0
    @Published var loadingMessage: String = "Initializing..."
    @Published var loadingComplete: Bool = false  // Signals backend is ready

    // Enhanced connection state from Universal Client
    @Published var serverVersion: String = "unknown"
    @Published var serverCapabilities: [String] = []
    @Published var detailedConnectionState: String = "Disconnected"

    // MARK: - Configuration

    let websocketURL: URL  // Made public for AppState logging (legacy)
    let apiBaseURL: URL    // Made public for AppState logging
    private var webSocketTask: URLSessionWebSocketTask?  // Legacy - kept for compatibility
    private var cancellables = Set<AnyCancellable>()
    private var reconnectTimer: Timer?
    private var isManuallyDisconnected = false

    // Robust connection management (legacy)
    private var reconnectAttempts = 0
    private let maxReconnectAttempts = 999  // Try indefinitely - HUD launches before backend is ready (startup loading screen)
    private var connectionHealthTimer: Timer?
    private var lastMessageTime: Date?

    // File logger for debugging (GUI apps don't show console in terminal)
    private let logger = FileLogger.shared

    // 🚀 UNIVERSAL CLIENT - Advanced connection management
    private var universalClient: UniversalWebSocketClient?
    private var useUniversalClient = true  // Feature flag - can toggle to legacy

    // Voice output (TODO: Add VoiceManager.swift to Xcode project)
    // private var voiceManager: VoiceManager?

    // MARK: - Initialization

    init() {
        logger.log("🔧 PythonBridge.init() STARTED")
        logger.log("   Initializing WebSocket bridge to Python backend...")

        // Dynamic backend configuration from environment (set by Python launcher)
        // UNIFIED WEBSOCKET: Use /ws endpoint (same as web-app) instead of /ws/hud
        let wsURL = ProcessInfo.processInfo.environment["JARVIS_BACKEND_WS"] ?? "ws://localhost:8010/ws"
        let httpURL = ProcessInfo.processInfo.environment["JARVIS_BACKEND_HTTP"] ?? "http://localhost:8010"

        logger.log("📍 Environment Variables Check:")
        logger.log("   JARVIS_BACKEND_WS: \(ProcessInfo.processInfo.environment["JARVIS_BACKEND_WS"] ?? "NOT SET (using default)")")
        logger.log("   JARVIS_BACKEND_HTTP: \(ProcessInfo.processInfo.environment["JARVIS_BACKEND_HTTP"] ?? "NOT SET (using default)")")

        guard let wsURLParsed = URL(string: wsURL) else {
            logger.log("❌ CRITICAL: Failed to parse WebSocket URL: \(wsURL)")
            fatalError("Invalid WebSocket URL")
        }
        guard let httpURLParsed = URL(string: httpURL) else {
            logger.log("❌ CRITICAL: Failed to parse HTTP URL: \(httpURL)")
            fatalError("Invalid HTTP URL")
        }

        self.websocketURL = wsURLParsed
        self.apiBaseURL = httpURLParsed

        logger.log("✅ URLs parsed successfully:")
        logger.log("   WebSocket: \(wsURL) [UNIFIED ENDPOINT]")
        logger.log("   HTTP API:  \(httpURL)")
        logger.log("   Max reconnect attempts: \(maxReconnectAttempts)")

        // Initialize voice manager (TODO: Add VoiceManager.swift to Xcode project)
        // self.voiceManager = VoiceManager(apiBaseURL: self.apiBaseURL)

        // 🚀 Initialize Universal WebSocket Client
        if useUniversalClient {
            logger.log("🚀 Initializing UniversalWebSocketClient...")
            initializeUniversalClient()
        }

        logger.log("🔧 PythonBridge.init() COMPLETED")
    }

    // MARK: - Universal Client Integration

    private func initializeUniversalClient() {
        universalClient = UniversalWebSocketClient(
            clientId: "macos-hud-\(UUID().uuidString)",
            clientType: "hud",
            capabilities: ["hud_client", "voice", "vision", "commands"],
            baseURL: apiBaseURL.absoluteString
        )

        logger.log("✅ UniversalWebSocketClient created")
        logger.log("   Client Type: hud")
        logger.log("   Capabilities: hud_client, voice, vision, commands")
        logger.log("   Base URL: \(apiBaseURL.absoluteString)")

        // Set up state change observer
        universalClient?.$connectionState
            .receive(on: DispatchQueue.main)
            .sink { [weak self] (state: UniversalWebSocketClient.ConnectionState) in
                self?.handleUniversalClientStateChange(state)
            }
            .store(in: &cancellables)

        // Set up server version observer
        universalClient?.$serverVersion
            .receive(on: DispatchQueue.main)
            .sink { [weak self] (version: String) in
                self?.serverVersion = version
                self?.logger.log("📊 Server version updated: \(version)")
            }
            .store(in: &cancellables)

        // Set up server capabilities observer
        universalClient?.$serverCapabilities
            .receive(on: DispatchQueue.main)
            .sink { [weak self] (capabilities: [String]) in
                self?.serverCapabilities = capabilities
                self?.logger.log("📊 Server capabilities updated: \(capabilities.joined(separator: ", "))")
            }
            .store(in: &cancellables)

        logger.log("✅ UniversalWebSocketClient state observers configured")
    }

    private func handleUniversalClientStateChange(_ state: UniversalWebSocketClient.ConnectionState) {
        logger.log("🔄 Universal Client State Change: \(state)")

        switch state {
        case .disconnected:
            connectionStatus = .disconnected
            detailedConnectionState = "Disconnected"

        case .fetchingConfig:
            connectionStatus = .connecting
            detailedConnectionState = "Fetching Configuration..."

        case .healthCheck:
            connectionStatus = .connecting
            detailedConnectionState = "Health Check..."

        case .connecting:
            connectionStatus = .connecting
            detailedConnectionState = "Connecting to WebSocket..."

        case .connected:
            connectionStatus = .connected
            detailedConnectionState = "Connected"
            logger.log("✅ Universal Client CONNECTED!")
            // Reset reconnect attempts on successful connection
            reconnectAttempts = 0

        case .reconnecting(let attempt):
            connectionStatus = .connecting
            detailedConnectionState = "Reconnecting (Attempt \(attempt))..."
            logger.log("🔄 Reconnecting: attempt \(attempt)")

        case .error(let message):
            connectionStatus = .error
            detailedConnectionState = "Error: \(message)"
            logger.log("❌ Connection Error: \(message)")
        }
    }

    // MARK: - Connection Management

    /// Connect to Python backend via WebSocket with robust retry logic
    /// Enhanced with Universal Client for advanced connection management
    func connect() {
        logger.log(String(repeating: "=", count: 80))
        logger.log("🔌 WEBSOCKET CONNECTION ATTEMPT STARTED")
        logger.log("   Function: PythonBridge.connect()")
        logger.log("   Thread: \(Thread.current)")

        isManuallyDisconnected = false

        // 🚀 Use Universal Client if enabled
        if useUniversalClient, let client = universalClient {
            logger.log("🚀 Using UniversalWebSocketClient for connection")
            logger.log("   Advanced features: Health Check, Config Discovery, Auto-Reconnect")

            // Set up message handler
            client.connect { [weak self] message in
                self?.handleUniversalClientMessage(message)
            }

            return
        }

        // Legacy connection method (fallback)
        logger.log("⚠️  Using legacy connection method")
        connectionStatus = .connecting
        logger.log("   Status set to: .connecting")

        logger.log("📊 Connection Details:")
        logger.log("   Attempt: \(reconnectAttempts + 1)/\(maxReconnectAttempts)")
        logger.log("   Target URL: \(websocketURL)")
        logger.log("   URL Scheme: \(websocketURL.scheme ?? "nil")")
        logger.log("   URL Host: \(websocketURL.host ?? "nil")")
        logger.log("   URL Port: \(websocketURL.port ?? -1)")
        logger.log("   URL Path: \(websocketURL.path)")
        logger.log("   Endpoint: /ws (unified WebSocket)")
        logger.log("   Expected Backend: FastAPI server on port 8010")

        logger.log("🔧 Creating URLSession and WebSocket task...")
        let session = URLSession(configuration: .default)
        logger.log("   ✓ URLSession created with default config")

        webSocketTask = session.webSocketTask(with: websocketURL)
        logger.log("   ✓ WebSocket task created")

        // Check task state before resume
        if let task = webSocketTask {
            logger.log("   Task state before resume: \(task.state.rawValue)")
            logger.log("   Task description: \(task.taskDescription ?? "nil")")
        } else {
            logger.log("   ❌ ERROR: webSocketTask is nil after creation!")
        }

        // Resume the task
        logger.log("🚀 Resuming WebSocket task...")
        webSocketTask?.resume()

        if let task = webSocketTask {
            logger.log("   Task state after resume: \(task.state.rawValue)")
            logger.log("   ✓ WebSocket task resumed successfully")
        }

        logger.log("📞 Starting message receiver...")
        receiveMessage()

        // Send initial connection message with delay
        logger.log("⏲️ Scheduling handshake message (0.5s delay)...")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            self.logger.log("📤 Sending HUD handshake message NOW...")
            self.sendConnectionMessage()
        }

        // Start connection health monitoring
        logger.log("💓 Starting connection health monitor...")
        startConnectionHealthCheck()

        logger.log("⏳ Waiting for WebSocket connection to establish...")
        logger.log(String(repeating: "=", count: 80))
    }

    /// Disconnect from Python backend
    func disconnect() {
        isManuallyDisconnected = true
        reconnectTimer?.invalidate()
        reconnectTimer = nil

        // Disconnect Universal Client if enabled
        if useUniversalClient {
            universalClient?.disconnect()
            logger.log("🔌 Universal Client disconnected")
        }

        // Legacy disconnection
        webSocketTask?.cancel(with: .goingAway, reason: nil)
        connectionStatus = .disconnected
        logger.log("🔌 Disconnected from backend")
    }

    // MARK: - Universal Client Message Handling

    private func handleUniversalClientMessage(_ message: String) {
        logger.log("📨 Received message from Universal Client")

        // Delegate to existing JSON message handler
        handleJSONMessage(message)
    }

    // Send message via Universal Client or legacy WebSocket
    private func sendViaUniversalClient(_ message: [String: Any]) {
        if useUniversalClient, let client = universalClient {
            client.send(message)
            logger.log("📤 Sent message via Universal Client")
        } else {
            // Legacy send method would go here
            logger.log("⚠️  Universal Client not available for sending")
        }
    }

    /// Automatically reconnect with exponential backoff
    private func scheduleReconnect() {
        guard !isManuallyDisconnected else { return }
        guard reconnectAttempts < maxReconnectAttempts else {
            logger.log("❌ Max reconnection attempts reached (\(maxReconnectAttempts)). Backend may not be ready yet.")
            DispatchQueue.main.async {
                self.connectionStatus = .error
            }
            return
        }

        reconnectAttempts += 1

        // Exponential backoff: 0.5s, 1s, 2s, 4s, 8s, max 15s
        // After many attempts, back off to 15s to avoid spamming logs
        let baseDelay: Double = 0.5
        let delay = min(baseDelay * pow(2.0, Double(reconnectAttempts - 1)), 15.0)

        // Only log every 10th attempt after the first 20 to avoid log spam
        if reconnectAttempts <= 20 || reconnectAttempts % 10 == 0 {
            logger.log("🔄 Reconnect attempt \(reconnectAttempts)/\(maxReconnectAttempts) in \(String(format: "%.1f", delay))s...")
        }

        // CRITICAL FIX: Schedule timer on main thread's RunLoop
        // Timers created on background threads won't fire unless their RunLoop is running
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }

            self.reconnectTimer?.invalidate()

            self.reconnectTimer = Timer.scheduledTimer(withTimeInterval: delay, repeats: false) { [weak self] _ in
                self?.logger.log("⏰ Reconnect timer fired - attempting connection...")
                self?.connect()
            }

            self.logger.log("   ✓ Reconnect timer scheduled on main RunLoop (will fire in \(String(format: "%.1f", delay))s)")
        }
    }

    /// Monitor connection health and reconnect if stale
    private func startConnectionHealthCheck() {
        connectionHealthTimer?.invalidate()

        connectionHealthTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
            guard let self = self else { return }

            // If we haven't received any messages in 30 seconds, consider connection dead
            if let lastMessage = self.lastMessageTime {
                let timeSinceLastMessage = Date().timeIntervalSince(lastMessage)
                if timeSinceLastMessage > 30.0 && self.connectionStatus == .connected {
                    print("⚠️ Connection appears stale (no messages for \(Int(timeSinceLastMessage))s), reconnecting...")
                    self.webSocketTask?.cancel(with: .goingAway, reason: nil)
                    self.scheduleReconnect()
                }
            }
        }
    }

    /// Send initial connection message to backend
    /// Uses unified WebSocket protocol with HUD-specific handshake
    private func sendConnectionMessage() {
        logger.log("📤 sendConnectionMessage() called")

        let clientID = "macos-hud-\(UUID().uuidString)"
        let message = [
            "type": "hud_connect",  // HUD-specific handler in unified WebSocket
            "client_id": clientID,
            "version": "2.0.0"
        ]

        logger.log("   Handshake payload:")
        logger.log("     type: hud_connect")
        logger.log("     client_id: \(clientID)")
        logger.log("     version: 2.0.0")

        logger.log("   Sending to unified WebSocket endpoint /ws...")
        sendMessage(message)
    }

    /// Send message to backend
    private func sendMessage(_ dict: [String: Any]) {
        logger.log("📨 sendMessage() called")

        guard let data = try? JSONSerialization.data(withJSONObject: dict) else {
            logger.log("   ❌ ERROR: Failed to serialize JSON")
            return
        }

        guard let string = String(data: data, encoding: .utf8) else {
            logger.log("   ❌ ERROR: Failed to convert JSON data to UTF-8 string")
            return
        }

        logger.log("   ✓ Message serialized: \(string.prefix(100))...")

        guard let task = webSocketTask else {
            logger.log("   ❌ ERROR: webSocketTask is nil! Cannot send message.")
            return
        }

        logger.log("   ✓ WebSocket task exists (state: \(task.state.rawValue))")
        logger.log("   📤 Sending message via WebSocket.send()...")

        task.send(.string(string)) { [weak self] error in
            if let error = error {
                self?.logger.log("   ❌ SEND ERROR: \(error)")
                self?.logger.log("      Error description: \(error.localizedDescription)")
                let nsError = error as NSError
                self?.logger.log("      Domain: \(nsError.domain)")
                self?.logger.log("      Code: \(nsError.code)")
                self?.logger.log("      UserInfo: \(nsError.userInfo)")
            } else {
                self?.logger.log("   ✅ Message sent successfully!")
            }
        }
    }

    /// Receive messages from WebSocket
    private func receiveMessage() {
        logger.log("📥 receiveMessage() called - Setting up WebSocket message listener")

        guard let task = webSocketTask else {
            logger.log("   ❌ ERROR: webSocketTask is nil in receiveMessage()!")
            return
        }

        logger.log("   ✓ WebSocket task exists (state: \(task.state.rawValue))")
        logger.log("   🎧 Calling task.receive() to listen for messages...")

        task.receive { [weak self] result in
            guard let self = self else { return }

            switch result {
            case .success(let message):
                self.logger.log("✅ [WebSocket] MESSAGE RECEIVED!")
                self.logger.log("   Receive timestamp: \(Date())")

                DispatchQueue.main.async {
                    self.connectionStatus = .connected
                    self.lastMessageTime = Date()
                    // Reset reconnect attempts on successful message
                    self.reconnectAttempts = 0
                    self.logger.log("   ✓ Connection status set to: .connected")
                    self.logger.log("   ✓ Reconnect attempts reset to 0")
                }

                self.logger.log("🔄 Processing received message...")
                self.handleMessage(message)

                self.logger.log("📥 Setting up NEXT message receiver (continuous listening)...")
                self.receiveMessage() // Continue receiving

            case .failure(let error):
                self.logger.log("❌ [WebSocket] RECEIVE ERROR!")
                self.logger.log("   Error: \(error)")
                self.logger.log("   Localized: \(error.localizedDescription)")

                // Check for specific error types
                let nsError = error as NSError
                self.logger.log("   Domain: \(nsError.domain)")
                self.logger.log("   Code: \(nsError.code)")
                self.logger.log("   UserInfo: \(nsError.userInfo)")

                // Log common WebSocket error codes
                if nsError.domain == "NSPOSIXErrorDomain" {
                    self.logger.log("   → POSIX error (likely connection refused or timeout)")
                } else if nsError.domain == "NSURLErrorDomain" {
                    self.logger.log("   → URL error (likely network/DNS issue)")
                    if nsError.code == -1004 {
                        self.logger.log("   → Error -1004: Could not connect to server")
                    }
                }

                DispatchQueue.main.async {
                    self.connectionStatus = .error
                    self.logger.log("   ✓ Connection status set to: .error")
                }

                self.logger.log("🔄 Scheduling reconnection attempt...")
                self.scheduleReconnect()
            }
        }

        logger.log("   ✓ Message listener setup complete (waiting for data...)")
    }

    /// Handle incoming WebSocket message
    private func handleMessage(_ message: URLSessionWebSocketTask.Message) {
        switch message {
        case .string(let text):
            handleJSONMessage(text)
        case .data(let data):
            if let text = String(data: data, encoding: .utf8) {
                handleJSONMessage(text)
            }
        @unknown default:
            break
        }
    }

    /// Parse and handle JSON message from Python
    private func handleJSONMessage(_ jsonString: String) {
        print("📨 [WebSocket] Received raw message: \(jsonString.prefix(200))...")

        guard let data = jsonString.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            print("❌ [WebSocket] Failed to parse JSON message")
            return
        }

        print("✅ [WebSocket] JSON parsed successfully")
        print("   Message type: \(json["type"] ?? "unknown")")

        DispatchQueue.main.async {
            // Handle different message types from Python backend
            if let type = json["type"] as? String {
                print("🔀 [WebSocket] Routing message type: \(type)")

                switch type {
                case "welcome":
                    print("👋 [WebSocket] Received welcome message!")
                    if let message = json["message"] as? String {
                        print("   Welcome: \(message)")
                    }
                case "transcript":
                    print("📝 [WebSocket] Handling transcript")
                    self.handleTranscript(json)
                case "state":
                    print("🔄 [WebSocket] Handling state update")
                    self.handleStateUpdate(json)
                case "status":
                    print("📊 [WebSocket] Handling status update")
                    self.handleStatusUpdate(json)
                case "loading_progress", "hud_progress":
                    print("⏳ [WebSocket] Handling loading progress")
                    self.handleLoadingProgress(json)
                case "loading_complete", "hud_loading_complete":
                    print("✅ [WebSocket] Handling loading complete")
                    self.handleLoadingComplete(json)
                case "system_restart":
                    print("🔄 [WebSocket] Backend restarting - resetting HUD to loading screen")
                    self.handleSystemRestart(json)
                case "command_response", "response":
                    print("💬 [WebSocket] Handling command response")
                    self.handleCommandResponse(json)
                case "pong":
                    print("🏓 [WebSocket] Received pong")
                default:
                    print("⚠️  [WebSocket] Unknown message type: \(type)")
                    break
                }
            } else {
                print("❌ [WebSocket] Message missing 'type' field")
            }
        }
    }

    // MARK: - Message Handlers

    private func handleTranscript(_ json: [String: Any]) {
        guard let speaker = json["speaker"] as? String,
              let text = json["text"] as? String else {
            return
        }

        let message = TranscriptMessage(
            speaker: speaker,
            text: text,
            timestamp: Date()
        )
        transcriptMessages.append(message)
    }

    private func handleStateUpdate(_ json: [String: Any]) {
        guard let state = json["state"] as? String else { return }

        switch state.lowercased() {
        case "offline":
            hudState = .offline
        case "listening":
            hudState = .listening
        case "processing":
            hudState = .processing
        case "speaking":
            hudState = .speaking
        case "idle":
            hudState = .idle
        default:
            break
        }
    }

    private func handleStatusUpdate(_ json: [String: Any]) {
        // Handle status updates from Python
    }

    private func handleLoadingProgress(_ json: [String: Any]) {
        guard let progress = json["progress"] as? Int,
              let message = json["message"] as? String else {
            return
        }

        loadingProgress = progress
        loadingMessage = message

        print("📊 Loading Progress: \(progress)% - \(message)")
    }

    private func handleLoadingComplete(_ json: [String: Any]) {
        guard let success = json["success"] as? Bool else {
            return
        }

        loadingComplete = true
        loadingProgress = 100

        print("✅ Backend Loading Complete! Success: \(success)")
    }

    private func handleSystemRestart(_ json: [String: Any]) {
        print("🔄 System restart detected - resetting HUD to loading state")

        // Reset to loading screen state
        loadingComplete = false
        loadingProgress = 0
        loadingMessage = "Restarting JARVIS..."

        // Clear transcript for fresh start
        transcriptMessages.removeAll()

        // Set HUD state to offline/loading
        hudState = .offline

        print("   ✓ HUD reset complete - ready for fresh progress updates")
    }

    private func handleCommandResponse(_ json: [String: Any]) {
        // Extract response text
        guard let responseText = json["response"] as? String ?? json["text"] as? String else {
            return
        }

        // Check if we should speak the response
        let shouldSpeak = json["speak"] as? Bool ?? false

        print("📨 Received response: \(responseText.prefix(100))...")
        print("   Should speak: \(shouldSpeak)")

        // Add to transcript
        let transcriptMsg = TranscriptMessage(
            speaker: "JARVIS",
            text: responseText,
            timestamp: Date()
        )
        transcriptMessages.append(transcriptMsg)

        // Speak if requested (TODO: Add VoiceManager.swift to Xcode project)
        if shouldSpeak {
            print("🎤 Speaking response via TTS...")
            // voiceManager?.speak(responseText)
        }
    }

    // MARK: - HTTP API Calls

    /// Send command to Python backend via HTTP
    func sendCommand(_ command: String) async throws {
        var request = URLRequest(url: apiBaseURL.appendingPathComponent("/command"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body = ["command": command]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (_, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw NetworkError.requestFailed
        }
    }

    /// Get system status from Python backend
    func getSystemStatus() async throws -> [String: Any] {
        let request = URLRequest(url: apiBaseURL.appendingPathComponent("/status"))
        let (data, _) = try await URLSession.shared.data(for: request)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NetworkError.invalidResponse
        }
        return json
    }
}

// MARK: - Supporting Types

enum ConnectionStatus {
    case connected
    case disconnected
    case connecting
    case error
}

enum NetworkError: Error {
    case requestFailed
    case invalidResponse
    case connectionLost
}
