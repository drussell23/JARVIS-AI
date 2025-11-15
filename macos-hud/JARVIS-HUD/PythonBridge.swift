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

    // MARK: - Configuration

    private let websocketURL: URL
    private let apiBaseURL: URL
    private var webSocketTask: URLSessionWebSocketTask?
    private var cancellables = Set<AnyCancellable>()
    private var reconnectTimer: Timer?
    private var isManuallyDisconnected = false

    // MARK: - Initialization

    init() {
        // Dynamic backend configuration from environment (set by Python launcher)
        let wsURL = ProcessInfo.processInfo.environment["JARVIS_BACKEND_WS"] ?? "ws://localhost:8000/ws/hud"
        let httpURL = ProcessInfo.processInfo.environment["JARVIS_BACKEND_HTTP"] ?? "http://localhost:8000"

        self.websocketURL = URL(string: wsURL)!
        self.apiBaseURL = URL(string: httpURL)!

        print("🔧 Backend Configuration:")
        print("   WebSocket: \(wsURL)")
        print("   HTTP API:  \(httpURL)")
    }

    // MARK: - Connection Management

    /// Connect to Python backend via WebSocket with auto-retry
    func connect() {
        isManuallyDisconnected = false
        print("🔌 Connecting to backend at \(websocketURL)...")

        let session = URLSession(configuration: .default)
        webSocketTask = session.webSocketTask(with: websocketURL)

        webSocketTask?.resume()
        receiveMessage()

        // Send initial connection message
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            self.sendConnectionMessage()
        }
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

    /// Automatically reconnect after disconnect (unless manual)
    private func scheduleReconnect() {
        guard !isManuallyDisconnected else { return }

        print("🔄 Scheduling reconnect in 3 seconds...")
        reconnectTimer?.invalidate()

        reconnectTimer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: false) { [weak self] _ in
            self?.connect()
        }
    }

    /// Send initial connection message to backend
    private func sendConnectionMessage() {
        let message = [
            "type": "connect",
            "client": "macos-hud",
            "version": "1.0.0"
        ]

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
