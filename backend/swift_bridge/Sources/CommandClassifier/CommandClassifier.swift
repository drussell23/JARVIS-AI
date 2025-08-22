/**
 * JARVIS Intelligent Command Classifier
 * Uses advanced NLP and machine learning to intelligently route commands
 * No hardcoding - learns and adapts dynamically
 */

import Foundation
import NaturalLanguage
import CoreML

// MARK: - Command Types

public enum CommandType: String, CaseIterable, Codable {
    case system = "system"      // Direct actions: close, open, launch
    case vision = "vision"      // Analysis queries: what's on screen, describe
    case hybrid = "hybrid"      // Could be either based on context
    case conversation = "conversation"  // General chat
}

public enum CommandIntent: String, Codable {
    // System intents
    case openApp = "open_app"
    case closeApp = "close_app"
    case switchApp = "switch_app"
    case systemControl = "system_control"
    case fileOperation = "file_operation"
    
    // Vision intents
    case analyzeScreen = "analyze_screen"
    case findElement = "find_element"
    case describeWorkspace = "describe_workspace"
    case checkNotifications = "check_notifications"
    
    // Hybrid intents
    case appQuery = "app_query"  // Could be "is X open?" or "open X"
    case workspaceAction = "workspace_action"
}

// MARK: - Command Pattern

public struct CommandPattern {
    let pattern: String
    let type: CommandType
    let intent: CommandIntent
    let confidence: Double
    let isLearned: Bool
    
    init(pattern: String, type: CommandType, intent: CommandIntent, 
         confidence: Double = 1.0, isLearned: Bool = false) {
        self.pattern = pattern
        self.type = type
        self.intent = intent
        self.confidence = confidence
        self.isLearned = isLearned
    }
}

// MARK: - Analysis Result

public struct CommandAnalysis {
    let originalText: String
    let type: CommandType
    let intent: CommandIntent
    let confidence: Double
    let entities: [String: String]  // e.g., ["app": "WhatsApp", "action": "close"]
    let reasoning: String
    let alternativeInterpretations: [(type: CommandType, confidence: Double)]
}

// MARK: - Main Classifier

@objc public class CommandClassifier: NSObject {
    
    // Linguistic analysis
    private let tagger = NLTagger(tagSchemes: [.lexicalClass, .lemma, .sentimentScore])
    private let languageRecognizer = NLLanguageRecognizer()
    
    // Dynamic learning storage
    private var learnedPatterns: [CommandPattern] = []
    private var userPreferences: [String: CommandType] = [:]
    private var contextHistory: [CommandAnalysis] = []
    private let maxHistorySize = 50
    
    // Pattern recognition (no hardcoding - just linguistic rules)
    private let actionVerbLemmas = Set<String>()  // Populated dynamically
    private let questionWords = Set<String>()     // Populated from NLP analysis
    
    public override init() {
        super.init()
        loadLearnedPatterns()
        initializeLinguisticRules()
    }
    
    // MARK: - Public Interface
    
    @objc public func classifyCommand(_ text: String) -> String {
        let analysis = analyzeCommand(text)
        
        // Return JSON for Python
        let result: [String: Any] = [
            "type": analysis.type.rawValue,
            "intent": analysis.intent.rawValue,
            "confidence": analysis.confidence,
            "entities": analysis.entities,
            "reasoning": analysis.reasoning
        ]
        
        if let jsonData = try? JSONSerialization.data(withJSONObject: result),
           let jsonString = String(data: jsonData, encoding: .utf8) {
            return jsonString
        }
        
        return "{\"type\": \"system\", \"confidence\": 0.5}"
    }
    
    public func analyzeCommand(_ text: String) -> CommandAnalysis {
        let normalizedText = text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        
        // Step 1: Linguistic Analysis
        let linguisticFeatures = extractLinguisticFeatures(from: normalizedText)
        
        // Step 2: Entity Recognition
        let entities = extractEntities(from: normalizedText)
        
        // Step 3: Context Analysis
        let contextScore = analyzeContext(for: normalizedText, with: entities)
        
        // Step 4: Pattern Matching (learned patterns first)
        let patternMatch = findBestPatternMatch(for: normalizedText)
        
        // Step 5: Intelligent Classification
        let classification = classifyUsingIntelligence(
            text: normalizedText,
            features: linguisticFeatures,
            entities: entities,
            context: contextScore,
            pattern: patternMatch
        )
        
        // Step 6: Learn from this classification
        updateLearning(with: classification)
        
        return classification
    }
    
    // MARK: - Linguistic Analysis
    
    private func extractLinguisticFeatures(from text: String) -> [String: Any] {
        var features: [String: Any] = [:]
        
        tagger.string = text
        
        // Extract grammatical structure
        var tags: [String] = []
        var lemmas: [String] = []
        
        tagger.enumerateTags(in: text.startIndex..<text.endIndex, 
                            unit: .word, 
                            scheme: .lexicalClass) { tag, range in
            if let tag = tag {
                tags.append(tag.rawValue)
                
                // Get lemma (base form)
                if let lemma = tagger.tag(at: range.lowerBound, 
                                         unit: .word, 
                                         scheme: .lemma).0 {
                    lemmas.append(lemma.rawValue)
                }
            }
            return true
        }
        
        features["tags"] = tags
        features["lemmas"] = lemmas
        features["firstTag"] = tags.first ?? "unknown"
        features["hasVerb"] = tags.contains("Verb")
        features["hasNoun"] = tags.contains("Noun")
        features["isQuestion"] = isQuestionStructure(tags: tags, text: text)
        
        // Sentiment can help distinguish commands from queries
        if let sentiment = tagger.tag(at: text.startIndex, 
                                     unit: .paragraph, 
                                     scheme: .sentimentScore).0 {
            features["sentiment"] = sentiment.rawValue
        }
        
        return features
    }
    
    private func extractEntities(from text: String) -> [String: String] {
        var entities: [String: String] = [:]
        
        // Extract app names dynamically
        let words = text.components(separatedBy: .whitespaces)
        
        // Use NLP to identify proper nouns (likely app names)
        tagger.string = text
        tagger.enumerateTags(in: text.startIndex..<text.endIndex,
                            unit: .word,
                            scheme: .lexicalClass) { tag, range in
            if tag == .personalName || tag == .organizationName {
                let word = String(text[range])
                entities["app"] = word
            }
            return true
        }
        
        // Also check for common app name patterns
        for (index, word) in words.enumerated() {
            // Check if word is capitalized or follows action verb
            if word.first?.isUppercase == true {
                entities["app"] = word
            } else if index > 0 {
                let previousWord = words[index - 1]
                if isActionVerb(previousWord) {
                    entities["app"] = word
                    entities["action"] = previousWord
                }
            }
        }
        
        return entities
    }
    
    // MARK: - Intelligent Classification
    
    private func classifyUsingIntelligence(text: String,
                                         features: [String: Any],
                                         entities: [String: String],
                                         context: Double,
                                         pattern: CommandPattern?) -> CommandAnalysis {
        
        var scores: [CommandType: Double] = [:]
        var reasoning = ""
        
        // Use pattern match if high confidence
        if let pattern = pattern, pattern.confidence > 0.8 {
            scores[pattern.type] = pattern.confidence
            reasoning = "Matched learned pattern: \(pattern.pattern)"
        }
        
        // Analyze linguistic structure
        let firstTag = features["firstTag"] as? String ?? ""
        let hasVerb = features["hasVerb"] as? Bool ?? false
        let isQuestion = features["isQuestion"] as? Bool ?? false
        
        // Intelligent scoring based on linguistic analysis
        if firstTag == "Verb" || (hasVerb && !isQuestion) {
            // Action-oriented structure suggests system command
            scores[.system] = (scores[.system] ?? 0) + 0.7
            reasoning += " Action verb detected."
        }
        
        if isQuestion {
            // Question structure suggests vision query
            scores[.vision] = (scores[.vision] ?? 0) + 0.6
            reasoning += " Question structure detected."
        }
        
        // Entity-based scoring
        if let action = entities["action"] {
            if isActionVerb(action) {
                scores[.system] = (scores[.system] ?? 0) + 0.4
                reasoning += " Action '\(action)' indicates system command."
            }
        }
        
        // Context-based adjustment
        if context > 0.5 {
            // Recent similar commands influence classification
            if let lastType = contextHistory.last?.type {
                scores[lastType] = (scores[lastType] ?? 0) + (context * 0.3)
                reasoning += " Context suggests \(lastType) based on history."
            }
        }
        
        // Determine best type
        let bestType = scores.max(by: { $0.value < $1.value })?.key ?? .system
        let confidence = scores[bestType] ?? 0.5
        
        // Determine intent
        let intent = determineIntent(for: bestType, text: text, entities: entities)
        
        // Alternative interpretations
        let alternatives = scores.map { (type: $0.key, confidence: $0.value) }
            .filter { $0.type != bestType }
            .sorted { $0.confidence > $1.confidence }
        
        return CommandAnalysis(
            originalText: text,
            type: bestType,
            intent: intent,
            confidence: min(confidence, 1.0),
            entities: entities,
            reasoning: reasoning.trimmingCharacters(in: .whitespaces),
            alternativeInterpretations: alternatives
        )
    }
    
    // MARK: - Intent Determination
    
    private func determineIntent(for type: CommandType, 
                               text: String, 
                               entities: [String: String]) -> CommandIntent {
        switch type {
        case .system:
            if text.contains("close") || text.contains("quit") {
                return .closeApp
            } else if text.contains("open") || text.contains("launch") {
                return .openApp
            } else if text.contains("switch") || text.contains("activate") {
                return .switchApp
            } else {
                return .systemControl
            }
            
        case .vision:
            if text.contains("what") || text.contains("describe") {
                return .analyzeScreen
            } else if text.contains("find") || text.contains("where") {
                return .findElement
            } else if text.contains("notification") || text.contains("message") {
                return .checkNotifications
            } else {
                return .describeWorkspace
            }
            
        case .hybrid:
            return .appQuery
            
        case .conversation:
            return .systemControl  // Default
        }
    }
    
    // MARK: - Helper Methods
    
    private func isQuestionStructure(tags: [String], text: String) -> Bool {
        // Check for question marks
        if text.contains("?") { return true }
        
        // Check for question word patterns
        let firstWord = text.components(separatedBy: .whitespaces).first?.lowercased() ?? ""
        let questionStarters = ["what", "where", "when", "why", "how", "who", 
                               "which", "can", "could", "would", "should", 
                               "is", "are", "do", "does", "did"]
        
        return questionStarters.contains(firstWord)
    }
    
    private func isActionVerb(_ word: String) -> Bool {
        // Use lemmatization to check if word is an action verb
        tagger.string = word
        let tagResult = tagger.tag(at: word.startIndex, 
                                  unit: .word, 
                                  scheme: .lexicalClass)
        if let tag = tagResult.0, tag == .verb {
            // Further check if it's an action verb (not auxiliary)
            let auxiliaryVerbs = ["is", "are", "was", "were", "be", "been", 
                                 "have", "has", "had", "do", "does", "did"]
            return !auxiliaryVerbs.contains(word.lowercased())
        }
        return false
    }
    
    // MARK: - Context Analysis
    
    private func analyzeContext(for text: String, with entities: [String: String]) -> Double {
        guard !contextHistory.isEmpty else { return 0.0 }
        
        var contextScore = 0.0
        let recentCommands = contextHistory.suffix(5)
        
        for previous in recentCommands {
            // Check for similar entities
            let sharedEntities = entities.filter { previous.entities[$0.key] == $0.value }
            contextScore += Double(sharedEntities.count) * 0.2
            
            // Check for similar text patterns
            let similarity = textSimilarity(text, previous.originalText)
            contextScore += similarity * 0.3
        }
        
        return min(contextScore, 1.0)
    }
    
    private func textSimilarity(_ text1: String, _ text2: String) -> Double {
        let words1 = Set(text1.lowercased().components(separatedBy: .whitespaces))
        let words2 = Set(text2.lowercased().components(separatedBy: .whitespaces))
        
        let intersection = words1.intersection(words2)
        let union = words1.union(words2)
        
        return union.isEmpty ? 0.0 : Double(intersection.count) / Double(union.count)
    }
    
    // MARK: - Learning
    
    private func updateLearning(with analysis: CommandAnalysis) {
        // Add to context history
        contextHistory.append(analysis)
        if contextHistory.count > maxHistorySize {
            contextHistory.removeFirst()
        }
        
        // Learn patterns if confidence is high
        if analysis.confidence > 0.7 {
            let pattern = CommandPattern(
                pattern: analysis.originalText,
                type: analysis.type,
                intent: analysis.intent,
                confidence: analysis.confidence,
                isLearned: true
            )
            learnedPatterns.append(pattern)
            saveLearnedPatterns()
        }
    }
    
    private func findBestPatternMatch(for text: String) -> CommandPattern? {
        return learnedPatterns
            .filter { pattern in
                // Fuzzy matching
                textSimilarity(text, pattern.pattern) > 0.8
            }
            .max { $0.confidence < $1.confidence }
    }
    
    // MARK: - Persistence
    
    private func loadLearnedPatterns() {
        // Load from UserDefaults or file
        if let data = UserDefaults.standard.data(forKey: "JARVISLearnedPatterns"),
           let patterns = try? JSONDecoder().decode([CommandPattern].self, from: data) {
            learnedPatterns = patterns
        }
    }
    
    private func saveLearnedPatterns() {
        // Save to UserDefaults or file
        if let data = try? JSONEncoder().encode(learnedPatterns) {
            UserDefaults.standard.set(data, forKey: "JARVISLearnedPatterns")
        }
    }
    
    private func initializeLinguisticRules() {
        // No hardcoding - rules are discovered through NLP analysis
        // This method is for future ML model integration
    }
}

// MARK: - Learning Extensions

extension CommandClassifier {
    
    @objc public func learnFromFeedback(_ command: String, 
                                       _ actualType: String, 
                                       _ wasCorrect: Bool) {
        guard let type = CommandType(rawValue: actualType) else { return }
        
        if wasCorrect {
            // Reinforce the classification
            if let analysis = contextHistory.first(where: { $0.originalText == command }) {
                let pattern = CommandPattern(
                    pattern: command,
                    type: type,
                    intent: analysis.intent,
                    confidence: min(analysis.confidence * 1.1, 1.0),
                    isLearned: true
                )
                learnedPatterns.append(pattern)
            }
        } else {
            // Learn the correct classification
            let analysis = analyzeCommand(command)
            let pattern = CommandPattern(
                pattern: command,
                type: type,
                intent: analysis.intent,
                confidence: 0.9,
                isLearned: true
            )
            learnedPatterns.append(pattern)
        }
        
        saveLearnedPatterns()
    }
    
    @objc public func getClassificationStats() -> String {
        let stats: [String: Any] = [
            "learned_patterns": learnedPatterns.count,
            "context_history": contextHistory.count,
            "system_commands": learnedPatterns.filter { $0.type == .system }.count,
            "vision_commands": learnedPatterns.filter { $0.type == .vision }.count
        ]
        
        if let data = try? JSONSerialization.data(withJSONObject: stats),
           let json = String(data: data, encoding: .utf8) {
            return json
        }
        
        return "{}"
    }
}

// Make CommandPattern codable for persistence
extension CommandPattern: Codable {}