//
//  PythonBridge.swift
//  JARVIS-HUD
//
//  Bridge to communicate with Python FastAPI backend
//  Handles WebSocket connections and HTTP API calls
//

import Foundation
import Combine

/// WebSocket connection to Python backend
class PythonBridge: ObservableObject {

    // MARK: - Published State

    @Published var connectionStatus: ConnectionStatus = .disconnected
    @Published var transcriptMessages: [TranscriptMessage] = []
    @Published var hudState: HUDState = .offline
    @Published var loadingProgress: Int = 0
    @Published var loadingMessage: String = "Initializing..."
    @Published var loadingComplete: Bool = false  // Signals backend is ready

    // MARK: - Configuration

    private let websocketURL: URL
    private let apiBaseURL: URL
    private var webSocketTask: URLSessionWebSocketTask?
    private var cancellables = Set<AnyCancellable>()
    private var reconnectTimer: Timer?
    private var isManuallyDisconnected = false

    // Robust connection management
    private var reconnectAttempts = 0
    private let maxReconnectAttempts = 60  // Try for ~5 minutes with exponential backoff (backend can take time to start)
    private var connectionHealthTimer: Timer?
    private var lastMessageTime: Date?

    // MARK: - Initialization

    init() {
        // Dynamic backend configuration from environment (set by Python launcher)
        // UNIFIED WEBSOCKET: Use /ws endpoint (same as web-app) instead of /ws/hud
        let wsURL = ProcessInfo.processInfo.environment["JARVIS_BACKEND_WS"] ?? "ws://localhost:8010/ws"
        let httpURL = ProcessInfo.processInfo.environment["JARVIS_BACKEND_HTTP"] ?? "http://localhost:8010"

        self.websocketURL = URL(string: wsURL)!
        self.apiBaseURL = URL(string: httpURL)!

        print("🔧 Backend Configuration:")
        print("   WebSocket: \(wsURL) [UNIFIED ENDPOINT]")
        print("   HTTP API:  \(httpURL)")
        print("   Max reconnect attempts: \(maxReconnectAttempts)")
    }

    // MARK: - Connection Management

    /// Connect to Python backend via WebSocket with robust retry logic
    func connect() {
        isManuallyDisconnected = false
        connectionStatus = .connecting

        print("🔌 Attempting connection \(reconnectAttempts + 1)/\(maxReconnectAttempts) to backend at \(websocketURL)...")

        let session = URLSession(configuration: .default)
        webSocketTask = session.webSocketTask(with: websocketURL)

        webSocketTask?.resume()
        receiveMessage()

        // Send initial connection message with delay
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            self.sendConnectionMessage()
        }

        // Start connection health monitoring
        startConnectionHealthCheck()
    }

    /// Disconnect from Python backend
    func disconnect() {
        isManuallyDisconnected = true
        reconnectTimer?.invalidate()
        reconnectTimer = nil

        webSocketTask?.cancel(with: .goingAway, reason: nil)
        connectionStatus = .disconnected
        print("🔌 Disconnected from backend")
    }

    /// Automatically reconnect with exponential backoff
    private func scheduleReconnect() {
        guard !isManuallyDisconnected else { return }
        guard reconnectAttempts < maxReconnectAttempts else {
            print("❌ Max reconnection attempts reached (\(maxReconnectAttempts)). Backend may not be ready yet.")
            connectionStatus = .error
            return
        }

        reconnectAttempts += 1

        // Exponential backoff: 0.5s, 1s, 2s, 4s, 8s, max 10s
        let baseDelay: Double = 0.5
        let delay = min(baseDelay * pow(2.0, Double(reconnectAttempts - 1)), 10.0)

        print("🔄 Reconnect attempt \(reconnectAttempts)/\(maxReconnectAttempts) in \(String(format: "%.1f", delay))s...")
        reconnectTimer?.invalidate()

        reconnectTimer = Timer.scheduledTimer(withTimeInterval: delay, repeats: false) { [weak self] _ in
            self?.connect()
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
        let message = [
            "type": "hud_connect",  // HUD-specific handler in unified WebSocket
            "client_id": "macos-hud-\(UUID().uuidString)",
            "version": "2.0.0"
        ]

        print("📤 Sending HUD connection handshake to unified WebSocket endpoint...")
        sendMessage(message)
    }

    /// Send message to backend
    private func sendMessage(_ dict: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: dict),
              let string = String(data: data, encoding: .utf8) else {
            return
        }

        webSocketTask?.send(.string(string)) { error in
            if let error = error {
                print("❌ Send error: \(error)")
            }
        }
    }

    /// Receive messages from WebSocket
    private func receiveMessage() {
        webSocketTask?.receive { [weak self] result in
            switch result {
            case .success(let message):
                DispatchQueue.main.async {
                    self?.connectionStatus = .connected
                    self?.lastMessageTime = Date()
                    // Reset reconnect attempts on successful message
                    self?.reconnectAttempts = 0
                }
                self?.handleMessage(message)
                self?.receiveMessage() // Continue receiving

            case .failure(let error):
                print("❌ WebSocket error: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    self?.connectionStatus = .error
                }
                self?.scheduleReconnect()
            }
        }
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
        guard let data = jsonString.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return
        }

        DispatchQueue.main.async {
            // Handle different message types from Python backend
            if let type = json["type"] as? String {
                switch type {
                case "transcript":
                    self.handleTranscript(json)
                case "state":
                    self.handleStateUpdate(json)
                case "status":
                    self.handleStatusUpdate(json)
                case "loading_progress":
                    self.handleLoadingProgress(json)
                case "loading_complete":
                    self.handleLoadingComplete(json)
                default:
                    break
                }
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
