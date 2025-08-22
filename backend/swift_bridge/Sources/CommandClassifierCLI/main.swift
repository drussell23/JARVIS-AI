/**
 * Command-line interface for testing the JARVIS Command Classifier
 */

import Foundation
import CommandClassifier

// MARK: - CLI Tool

@main
struct CommandClassifierCLI {
    static func main() {
        let classifier = CommandClassifier()
        
        // Check command line arguments
        let args = CommandLine.arguments
        
        if args.count > 1 {
            // Process single command from arguments
            let command = args[1...].joined(separator: " ")
            let result = classifier.classifyCommand(command)
            print(result)
        } else {
            // Interactive mode
            runInteractiveMode(classifier: classifier)
        }
    }
    
    static func runInteractiveMode(classifier: CommandClassifier) {
        print("🤖 JARVIS Command Classifier - Interactive Mode")
        print("Type commands to see how they're classified. Type 'quit' to exit.")
        print("Type 'stats' to see learning statistics.")
        print("Type 'learn <command> <type>' to teach the classifier.")
        print("-" * 50)
        
        while true {
            print("\n> ", terminator: "")
            
            guard let input = readLine()?.trimmingCharacters(in: .whitespacesAndNewlines) else {
                continue
            }
            
            if input.lowercased() == "quit" || input.lowercased() == "exit" {
                print("👋 Goodbye!")
                break
            }
            
            if input.lowercased() == "stats" {
                print(classifier.getClassificationStats())
                continue
            }
            
            if input.lowercased().starts(with: "learn ") {
                handleLearnCommand(input, classifier: classifier)
                continue
            }
            
            // Classify the command
            let startTime = Date()
            let result = classifier.classifyCommand(input)
            let elapsedTime = Date().timeIntervalSince(startTime) * 1000
            
            // Pretty print the result
            if let data = result.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                
                print("\n📊 Classification Result:")
                print("   Type: \(json["type"] ?? "unknown")")
                print("   Intent: \(json["intent"] ?? "unknown")")
                print("   Confidence: \(String(format: "%.2f", json["confidence"] as? Double ?? 0.0))")
                
                if let entities = json["entities"] as? [String: String], !entities.isEmpty {
                    print("   Entities:")
                    for (key, value) in entities {
                        print("      - \(key): \(value)")
                    }
                }
                
                print("   Reasoning: \(json["reasoning"] ?? "No reasoning provided")")
                print("   Time: \(String(format: "%.2f", elapsedTime))ms")
            } else {
                print("Error parsing result: \(result)")
            }
        }
    }
    
    static func handleLearnCommand(_ input: String, classifier: CommandClassifier) {
        let parts = input.components(separatedBy: " ")
        guard parts.count >= 3 else {
            print("❌ Usage: learn <command> <type>")
            print("   Types: system, vision, hybrid, conversation")
            return
        }
        
        let command = parts[1..<parts.count-1].joined(separator: " ")
        let type = parts.last!
        
        classifier.learnFromFeedback(command, type, true)
        print("✅ Learned: '\(command)' is a '\(type)' command")
    }
}

// Helper extension
extension String {
    static func * (left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
}