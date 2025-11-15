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

    // MARK: - Configuration

    private let websocketURL: URL
    private let apiBaseURL: URL
    private var webSocketTask: URLSessionWebSocketTask?
    private var cancellables = Set<AnyCancellable>()

    // MARK: - Initialization

    init(
        websocketURL: String = "ws://localhost:8000/ws",
        apiBaseURL: String = "http://localhost:8000"
    ) {
        self.websocketURL = URL(string: websocketURL)!
        self.apiBaseURL = URL(string: apiBaseURL)!
    }

    // MARK: - Connection Management

    /// Connect to Python backend via WebSocket
    func connect() {
        let session = URLSession(configuration: .default)
        webSocketTask = session.webSocketTask(with: websocketURL)
        webSocketTask?.resume()
        connectionStatus = .connected
        receiveMessage()
    }

    /// Disconnect from Python backend
    func disconnect() {
        webSocketTask?.cancel(with: .goingAway, reason: nil)
        connectionStatus = .disconnected
    }

    /// Receive messages from WebSocket
    private func receiveMessage() {
        webSocketTask?.receive { [weak self] result in
            switch result {
            case .success(let message):
                self?.handleMessage(message)
                self?.receiveMessage() // Continue receiving
            case .failure(let error):
                print("WebSocket error: \(error)")
                self?.connectionStatus = .error
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
