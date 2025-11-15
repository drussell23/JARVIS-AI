//
//  VoiceManager.swift
//  JARVIS-HUD
//
//  Manages TTS audio playback for JARVIS voice responses
//  Downloads audio from backend TTS API and plays using AVAudioPlayer
//

import Foundation
import AVFoundation
import Combine

/// Manages JARVIS voice output via backend TTS
class VoiceManager: NSObject, ObservableObject {

    // MARK: - Published State

    @Published var isSpeaking: Bool = false
    @Published var currentlySpeaking: String = ""

    // MARK: - Configuration

    private let apiBaseURL: URL
    private var audioPlayer: AVAudioPlayer?
    private var speechQueue: [(text: String, completion: (() -> Void)?)] = []
    private var isProcessingQueue = false

    // MARK: - Initialization

    init(apiBaseURL: URL) {
        self.apiBaseURL = apiBaseURL
        super.init()

        // Configure audio session for playback
        configureAudioSession()
    }

    // MARK: - Audio Session Configuration

    private func configureAudioSession() {
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playback, mode: .default)
            try audioSession.setActive(true)
            print("[VoiceManager] ✅ Audio session configured for playback")
        } catch {
            print("[VoiceManager] ❌ Failed to configure audio session: \(error)")
        }
    }

    // MARK: - Public API

    /// Speak text using backend TTS (queued)
    /// - Parameters:
    ///   - text: Text to speak
    ///   - completion: Optional completion handler called when speech finishes
    func speak(_ text: String, completion: (() -> Void)? = nil) {
        print("[VoiceManager] 📝 Adding to queue: \(text.prefix(50))...")

        // Add to queue
        speechQueue.append((text: text, completion: completion))

        // Start processing if not already processing
        if !isProcessingQueue {
            processNextInQueue()
        }
    }

    /// Stop current speech and clear queue
    func stopSpeaking() {
        print("[VoiceManager] 🛑 Stopping speech and clearing queue")

        // Stop current playback
        audioPlayer?.stop()
        audioPlayer = nil

        // Clear queue
        speechQueue.removeAll()

        // Update state
        DispatchQueue.main.async {
            self.isSpeaking = false
            self.currentlySpeaking = ""
        }

        isProcessingQueue = false
    }

    // MARK: - Private Methods

    private func processNextInQueue() {
        guard !speechQueue.isEmpty else {
            isProcessingQueue = false
            return
        }

        isProcessingQueue = true
        let (text, completion) = speechQueue.removeFirst()

        // Update state
        DispatchQueue.main.async {
            self.currentlySpeaking = text
        }

        // Download and play audio
        downloadAndPlayAudio(text: text) { [weak self] success in
            // Call completion handler
            completion?()

            // Process next in queue
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                self?.processNextInQueue()
            }
        }
    }

    private func downloadAndPlayAudio(text: String, completion: @escaping (Bool) -> Void) {
        print("[VoiceManager] 🎤 Requesting TTS for: \(text.prefix(50))...")

        // Use GET endpoint for simplicity (supports text up to ~2000 chars)
        guard let encodedText = text.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) else {
            print("[VoiceManager] ❌ Failed to encode text")
            completion(false)
            return
        }

        let audioURL = apiBaseURL.appendingPathComponent("audio/speak/\(encodedText)")
        print("[VoiceManager] 📡 Requesting audio from: \(audioURL)")

        // Download audio data
        let task = URLSession.shared.dataTask(with: audioURL) { [weak self] data, response, error in
            guard let self = self else { return }

            if let error = error {
                print("[VoiceManager] ❌ Download error: \(error.localizedDescription)")
                completion(false)
                return
            }

            guard let data = data else {
                print("[VoiceManager] ❌ No audio data received")
                completion(false)
                return
            }

            print("[VoiceManager] ✅ Downloaded \(data.count) bytes of audio")

            // Play audio
            self.playAudio(data: data, completion: completion)
        }

        task.resume()
    }

    private func playAudio(data: Data, completion: @escaping (Bool) -> Void) {
        do {
            // Create audio player from data
            audioPlayer = try AVAudioPlayer(data: data)
            audioPlayer?.delegate = self

            // Update state
            DispatchQueue.main.async {
                self.isSpeaking = true
            }

            print("[VoiceManager] 🔊 Starting audio playback...")

            // Play audio
            if audioPlayer?.play() == true {
                print("[VoiceManager] ✅ Audio playback started")
            } else {
                print("[VoiceManager] ❌ Failed to start audio playback")
                DispatchQueue.main.async {
                    self.isSpeaking = false
                }
                completion(false)
            }

        } catch {
            print("[VoiceManager] ❌ Audio player creation failed: \(error)")
            DispatchQueue.main.async {
                self.isSpeaking = false
            }
            completion(false)
        }
    }
}

// MARK: - AVAudioPlayerDelegate

extension VoiceManager: AVAudioPlayerDelegate {

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        print("[VoiceManager] 🎵 Audio playback finished (success: \(flag))")

        DispatchQueue.main.async {
            self.isSpeaking = false
            self.currentlySpeaking = ""
        }
    }

    func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer, error: Error?) {
        print("[VoiceManager] ❌ Audio decode error: \(error?.localizedDescription ?? "unknown")")

        DispatchQueue.main.async {
            self.isSpeaking = false
            self.currentlySpeaking = ""
        }
    }
}
