import XCTest
@testable import CommandClassifier

final class CommandClassifierTests: XCTestCase {
    
    var classifier: CommandClassifier!
    
    override func setUp() {
        super.setUp()
        classifier = CommandClassifier()
    }
    
    // MARK: - System Command Tests
    
    func testCloseAppCommands() {
        let commands = [
            "close whatsapp",
            "quit discord",
            "exit safari",
            "terminate chrome",
            "please close spotify"
        ]
        
        for command in commands {
            let result = parseResult(classifier.classifyCommand(command))
            XCTAssertEqual(result.type, "system", "'\(command)' should be system")
            XCTAssertTrue(result.confidence > 0.6, "Confidence should be high")
        }
    }
    
    func testOpenAppCommands() {
        let commands = [
            "open terminal",
            "launch safari",
            "start vscode",
            "run photoshop"
        ]
        
        for command in commands {
            let result = parseResult(classifier.classifyCommand(command))
            XCTAssertEqual(result.type, "system")
            XCTAssertEqual(result.intent, "open_app")
        }
    }
    
    // MARK: - Vision Command Tests
    
    func testVisionQueries() {
        let commands = [
            "what's on my screen",
            "what applications are open",
            "show me my notifications",
            "describe my workspace",
            "what am i working on",
            "do i have any messages",
            "what can you see"
        ]
        
        for command in commands {
            let result = parseResult(classifier.classifyCommand(command))
            XCTAssertEqual(result.type, "vision", "'\(command)' should be vision")
        }
    }
    
    // MARK: - Edge Cases
    
    func testAmbiguousCommands() {
        // These could be interpreted multiple ways
        let testCases: [(command: String, expectedType: String)] = [
            ("show me whatsapp", "vision"),  // Asking to see, not open
            ("where is discord", "vision"),   // Looking for location
            ("i want to close whatsapp", "system"),  // Intent to close
            ("can you open safari", "system"),  // Polite request to open
        ]
        
        for testCase in testCases {
            let result = parseResult(classifier.classifyCommand(testCase.command))
            XCTAssertEqual(result.type, testCase.expectedType,
                          "'\(testCase.command)' should be \(testCase.expectedType)")
        }
    }
    
    // MARK: - Learning Tests
    
    func testLearningFromFeedback() {
        let command = "handle whatsapp"  // Ambiguous command
        
        // First classification might be uncertain
        let firstResult = parseResult(classifier.classifyCommand(command))
        let firstConfidence = firstResult.confidence
        
        // Teach it that this is a system command
        classifier.learnFromFeedback(command, "system", true)
        
        // Second classification should be more confident
        let secondResult = parseResult(classifier.classifyCommand(command))
        XCTAssertEqual(secondResult.type, "system")
        XCTAssertGreaterThan(secondResult.confidence, firstConfidence)
    }
    
    // MARK: - Entity Extraction Tests
    
    func testEntityExtraction() {
        let testCases: [(command: String, expectedApp: String)] = [
            ("close WhatsApp", "WhatsApp"),
            ("open Visual Studio Code", "Visual"),  // First capitalized word
            ("quit discord", "discord"),
            ("launch Microsoft Teams", "Microsoft")
        ]
        
        for testCase in testCases {
            let result = parseResult(classifier.classifyCommand(testCase.command))
            XCTAssertEqual(result.entities["app"], testCase.expectedApp)
        }
    }
    
    // MARK: - Performance Tests
    
    func testClassificationPerformance() {
        measure {
            for _ in 0..<100 {
                _ = classifier.classifyCommand("close whatsapp")
            }
        }
    }
    
    // MARK: - Helper Methods
    
    private func parseResult(_ jsonString: String) -> TestResult {
        guard let data = jsonString.data(using: .utf8),
              let json = try? JSONDecoder().decode(TestResult.self, from: data) else {
            XCTFail("Failed to parse result: \(jsonString)")
            return TestResult()
        }
        return json
    }
}

// Helper struct for parsing results
struct TestResult: Codable {
    let type: String
    let intent: String
    let confidence: Double
    let entities: [String: String]
    let reasoning: String
    
    init() {
        self.type = "unknown"
        self.intent = "unknown"
        self.confidence = 0.0
        self.entities = [:]
        self.reasoning = ""
    }
}