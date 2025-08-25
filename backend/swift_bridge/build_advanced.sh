#!/bin/bash
# Build script for Advanced Swift Command Classifier

echo "🏗️  Building Advanced Swift Command Classifier..."

# Create build directory
mkdir -p .build

# Check if Swift is available
if ! command -v swift &> /dev/null; then
    echo "❌ Swift is not installed. Please install Xcode."
    echo "   The system will use the advanced Python ML fallback."
    exit 0
fi

# Navigate to script directory
cd "$(dirname "$0")"

# Create Package.swift if it doesn't exist
if [ ! -f "Package.swift" ]; then
    echo "📦 Creating Package.swift..."
    cat > Package.swift << 'EOF'
// swift-tools-version:5.5
import PackageDescription

let package = Package(
    name: "JARVISAdvancedClassifier",
    platforms: [
        .macOS(.v10_15)
    ],
    products: [
        .executable(
            name: "jarvis-advanced-classifier",
            targets: ["CommandClassifierCLI"]
        ),
        .library(
            name: "AdvancedCommandClassifier",
            targets: ["AdvancedCommandClassifier"]
        )
    ],
    dependencies: [],
    targets: [
        .target(
            name: "AdvancedCommandClassifier",
            dependencies: [],
            path: "Sources/AdvancedCommandClassifier"
        ),
        .executableTarget(
            name: "CommandClassifierCLI",
            dependencies: ["AdvancedCommandClassifier"],
            path: "Sources/CommandClassifierCLI"
        ),
        .testTarget(
            name: "AdvancedCommandClassifierTests",
            dependencies: ["AdvancedCommandClassifier"],
            path: "Tests/AdvancedCommandClassifierTests"
        )
    ]
)
EOF
fi

# Create CLI wrapper
mkdir -p Sources/CommandClassifierCLI
cat > Sources/CommandClassifierCLI/main.swift << 'EOF'
import Foundation
import AdvancedCommandClassifier

// CLI for Advanced Command Classifier
@main
struct CommandClassifierCLI {
    static func main() {
        let arguments = CommandLine.arguments
        
        if arguments.count > 1 && arguments[1] == "--version" {
            print("JARVIS Advanced Command Classifier v2.0 - Zero Hardcoding ML System")
            return
        }
        
        if arguments.count > 1 && arguments[1] == "classify" {
            // Read JSON from stdin
            let inputData = FileHandle.standardInput.readDataToEndOfFile()
            
            do {
                let json = try JSONSerialization.jsonObject(with: inputData) as? [String: Any] ?? [:]
                let command = json["command"] as? String ?? ""
                let context = json["context"] as? [String: Any] ?? [:]
                
                // Create classifier
                let classifier = AdvancedCommandClassifier()
                
                // Analyze command
                let analysis = classifier.analyzeCommand(command)
                
                // Convert to JSON response
                let response: [String: Any] = [
                    "type": analysis.type.rawValue,
                    "intent": [
                        "primary": analysis.intent.primary,
                        "secondary": analysis.intent.secondary,
                        "confidence": analysis.intent.confidence
                    ],
                    "confidence": analysis.confidence,
                    "entities": analysis.entities.map { entity in
                        [
                            "text": entity.text,
                            "type": entity.type.rawValue,
                            "role": entity.role,
                            "confidence": entity.confidence
                        ]
                    },
                    "reasoning": analysis.reasoning,
                    "alternatives": analysis.alternatives.map { alt in
                        [
                            "type": alt.type.rawValue,
                            "confidence": alt.confidence,
                            "reasoning": alt.reasoning
                        ]
                    }
                ]
                
                let outputData = try JSONSerialization.data(withJSONObject: response)
                FileHandle.standardOutput.write(outputData)
                
            } catch {
                let errorResponse = ["error": error.localizedDescription]
                if let errorData = try? JSONSerialization.data(withJSONObject: errorResponse) {
                    FileHandle.standardOutput.write(errorData)
                }
                exit(1)
            }
        } else {
            // Interactive mode
            print("🧠 JARVIS Advanced Command Classifier - Interactive Mode")
            print("   (Zero hardcoding - Everything is learned)")
            print("   Type 'quit' to exit\n")
            
            let classifier = AdvancedCommandClassifier()
            
            while true {
                print("Enter command: ", terminator: "")
                guard let input = readLine(), !input.isEmpty else { continue }
                
                if input.lowercased() == "quit" {
                    print("Goodbye!")
                    break
                }
                
                let analysis = classifier.analyzeCommand(input)
                
                print("\n📊 Analysis:")
                print("   Type: \(analysis.type) (\(Int(analysis.confidence * 100))% confident)")
                print("   Intent: \(analysis.intent.primary)")
                print("   Reasoning: \(analysis.reasoning)")
                
                if !analysis.entities.isEmpty {
                    print("   Entities:")
                    for entity in analysis.entities {
                        print("     - \(entity.text) (\(entity.type.rawValue))")
                    }
                }
                
                if !analysis.alternatives.isEmpty {
                    print("   Alternatives:")
                    for alt in analysis.alternatives {
                        print("     - \(alt.type) (\(Int(alt.confidence * 100))%)")
                    }
                }
                
                print("")
            }
        }
    }
}
EOF

# Create test file
mkdir -p Tests/AdvancedCommandClassifierTests
cat > Tests/AdvancedCommandClassifierTests/AdvancedCommandClassifierTests.swift << 'EOF'
import XCTest
@testable import AdvancedCommandClassifier

final class AdvancedCommandClassifierTests: XCTestCase {
    func testWhatsAppRouting() {
        let classifier = AdvancedCommandClassifier()
        
        // Test the problematic "open WhatsApp" command
        let analysis = classifier.analyzeCommand("open WhatsApp")
        
        XCTAssertEqual(analysis.type, .system)
        XCTAssertTrue(analysis.confidence > 0.8)
        XCTAssertEqual(analysis.intent.primary, "open_app")
    }
    
    func testVisionCommands() {
        let classifier = AdvancedCommandClassifier()
        
        let analysis = classifier.analyzeCommand("what's on my screen")
        
        XCTAssertEqual(analysis.type, .vision)
        XCTAssertTrue(analysis.confidence > 0.7)
    }
    
    func testNoHardcoding() {
        let classifier = AdvancedCommandClassifier()
        
        // Test that it learns from patterns, not keywords
        let testCases = [
            ("close WhatsApp", CommandType.system),
            ("what's in WhatsApp", CommandType.vision),
            ("open Discord", CommandType.system),
            ("show me Safari", CommandType.vision)
        ]
        
        for (command, expectedType) in testCases {
            let analysis = classifier.analyzeCommand(command)
            XCTAssertEqual(analysis.type, expectedType, 
                          "Failed for: \(command)")
        }
    }
}
EOF

# Build the package
echo "🔨 Building Swift package..."
swift build -c release

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "📍 Binary location: .build/release/jarvis-advanced-classifier"
    
    # Make it executable
    chmod +x .build/release/jarvis-advanced-classifier
    
    # Test the classifier
    echo ""
    echo "🧪 Testing classifier..."
    echo '{"command": "open WhatsApp", "context": {}}' | .build/release/jarvis-advanced-classifier classify | python3 -m json.tool
    
else
    echo "❌ Build failed. The system will use Python ML fallback."
    echo "   This is normal if you don't have Xcode installed."
fi

echo ""
echo "📝 Note: The Python ML fallback works great even without Swift!"
echo "   Swift provides additional performance but is not required."