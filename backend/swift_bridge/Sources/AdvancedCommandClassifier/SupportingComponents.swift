import Foundation

// MARK: - Context Manager

public class ContextManager {
    private var commandHistory: [String] = []
    private var sessionStartTime: Date
    private var currentApplications: [String] = []
    private var userInteractionPatterns: [InteractionPattern] = []
    
    struct InteractionPattern {
        let timestamp: Date
        let command: String
        let response: String
        let success: Bool
    }
    
    public init() {
        self.sessionStartTime = Date()
    }
    
    public func getCurrentContext() -> ContextualInformation {
        let now = Date()
        let calendar = Calendar.current
        
        return ContextualInformation(
            previousCommands: Array(commandHistory.suffix(5)),
            currentApplications: getCurrentRunningApplications(),
            userState: inferUserState(),
            temporalContext: TemporalContext(
                timeOfDay: now,
                dayOfWeek: calendar.component(.weekday, from: now),
                isWorkingHours: isWorkingHours(now),
                sessionDuration: now.timeIntervalSince(sessionStartTime)
            ),
            environmentalFactors: gatherEnvironmentalFactors()
        )
    }
    
    // TODO: Implement this function to add a command to the command history
    public func addCommand(_ command: String) {
        commandHistory.append(command)
        
        // Keep history manageable
        if commandHistory.count > 100 {
            commandHistory = Array(commandHistory.suffix(50))
        }
    }
    
    // TODO: Implement this function to get the current running applications
    private func getCurrentRunningApplications() -> [String] {
        // This would interface with the system to get actual running apps
        // For now, returning stored value
        return currentApplications
    }
    
    private func inferUserState() -> UserState {
        // Analyze recent interactions to infer state
        let recentPatterns = userInteractionPatterns.suffix(10)
        
        let avgResponseTime = calculateAverageResponseTime(recentPatterns)
        let errorRate = calculateErrorRate(recentPatterns)
        let complexity = calculateInteractionComplexity(recentPatterns)
        
        return UserState(
            workingPattern: inferWorkingPattern(complexity),
            cognitiveLoad: min(1.0, avgResponseTime / 5.0),
            frustrationLevel: min(1.0, errorRate),
            expertise: max(0.0, 1.0 - errorRate)
        )
    }
    
    private func inferWorkingPattern(_ complexity: Double) -> WorkingPattern {
        if complexity > 0.7 {
            return .automating
        } else if complexity > 0.5 {
            return .multitasking
        } else if complexity > 0.3 {
            return .focused
        } else {
            return .exploring
        }
    }
    
    private func calculateAverageResponseTime(_ patterns: ArraySlice<InteractionPattern>) -> Double {
        guard patterns.count > 1 else { return 1.0 }
        
        var totalTime = 0.0
        for i in 1..<patterns.count {
            totalTime += patterns[patterns.startIndex + i].timestamp.timeIntervalSince(
                patterns[patterns.startIndex + i - 1].timestamp
            )
        }
        
        return totalTime / Double(patterns.count - 1)
    }
    
    // TODO: Implement this function to calculate the error rate of the command
    private func calculateErrorRate(_ patterns: ArraySlice<InteractionPattern>) -> Double {
        guard !patterns.isEmpty else { return 0.0 }
        
        let failures = patterns.filter { !$0.success }.count
        return Double(failures) / Double(patterns.count)
    }
    
    //
    private func calculateInteractionComplexity(_ patterns: ArraySlice<InteractionPattern>) -> Double {
        guard !patterns.isEmpty else { return 0.0 }
        
        let avgCommandLength = patterns.map { Double($0.command.count) }.reduce(0, +) / Double(patterns.count)
        return min(1.0, avgCommandLength / 100.0)
    }
    
    private func isWorkingHours(_ date: Date) -> Bool {
        let hour = Calendar.current.component(.hour, from: date)
        let weekday = Calendar.current.component(.weekday, from: date)
        
        // Monday-Friday, 9 AM - 6 PM
        return weekday >= 2 && weekday <= 6 && hour >= 9 && hour < 18
    }
    
    private func gatherEnvironmentalFactors() -> [String: Any] {
        return [
            "sessionDuration": Date().timeIntervalSince(sessionStartTime),
            "commandCount": commandHistory.count,
            "timeZone": TimeZone.current.identifier
        ]
    }
}

// MARK: - Pattern Recognizer

public class PatternRecognizer {
    private var recognitionPatterns: [RecognitionPattern] = []
    
    struct RecognitionPattern {
        let id: UUID
        let features: [String: Double]
        let weight: Double
        let lastUsed: Date
    }
    
    public func recognize(
        text: String,
        features: LinguisticFeatures,
        entities: [Entity],
        context: ContextualInformation
    ) -> [CommandPattern] {
        
        var patterns: [CommandPattern] = []
        
        // Create pattern from current input
        let currentPattern = CommandPattern(
            structure: features.structure,
            entities: entities,
            sentiment: features.sentiment,
            complexity: features.complexity
        )
        
        patterns.append(currentPattern)
        
        // Find similar learned patterns
        let similarPatterns = findSimilarPatterns(to: currentPattern, features: features)
        patterns.append(contentsOf: similarPatterns)
        
        return patterns
    }
    
    public func retrain(with patterns: [LearnedPattern]) {
        // Update recognition patterns based on learned data
        recognitionPatterns = patterns.map { pattern in
            RecognitionPattern(
                id: pattern.id,
                features: pattern.features,
                weight: pattern.confidence * pattern.successRate,
                lastUsed: pattern.timestamp
            )
        }
    }
    
    private func findSimilarPatterns(
        to pattern: CommandPattern,
        features: LinguisticFeatures
    ) -> [CommandPattern] {
        
        // This would use more sophisticated similarity matching
        // For now, returning empty array to avoid hardcoding
        return []
    }
}

// MARK: - Confidence Calculator

public class ConfidenceCalculator {
    private var thresholds: ConfidenceThresholds
    
    struct ConfidenceThresholds {
        var highConfidence: Double = 0.8
        var mediumConfidence: Double = 0.5
        var uncertaintyThreshold: Double = 0.3
    }
    
    public init() {
        self.thresholds = ConfidenceThresholds()
    }
    
    public func calculate(topScore: Double, allScores: [Double]) -> Double {
        guard !allScores.isEmpty else { return 0.0 }
        
        // Calculate score distribution metrics
        let sortedScores = allScores.sorted(by: >)
        let secondBest = sortedScores.count > 1 ? sortedScores[1] : 0.0
        
        // Margin between top two scores
        let margin = topScore - secondBest
        
        // Score concentration (how much the top score dominates)
        let totalScore = allScores.reduce(0, +)
        let concentration = totalScore > 0 ? topScore / totalScore : 0
        
        // Combine factors for final confidence
        let confidence = (topScore * 0.5) + (margin * 0.3) + (concentration * 0.2)
        
        return min(1.0, max(0.0, confidence))
    }
    
    public func updateThresholds(basedOn metrics: AccuracyMetrics) {
        // Dynamically adjust thresholds based on system performance
        if metrics.currentAccuracy > 0.9 {
            // System is performing well, can be more confident
            thresholds.highConfidence = max(0.7, thresholds.highConfidence - 0.05)
            thresholds.uncertaintyThreshold = max(0.2, thresholds.uncertaintyThreshold - 0.05)
        } else if metrics.currentAccuracy < 0.7 {
            // System needs to be more conservative
            thresholds.highConfidence = min(0.9, thresholds.highConfidence + 0.05)
            thresholds.uncertaintyThreshold = min(0.4, thresholds.uncertaintyThreshold + 0.05)
        }
    }
}

// MARK: - Entity Extractor

public class EntityExtractor {
    
    public func extract(
        from text: String,
        features: LinguisticFeatures
    ) -> [Entity] {
        
        var entities: [Entity] = []
        
        // Start with NLP-identified entities
        entities.append(contentsOf: features.entities)
        
        // Extract additional entities based on patterns
        let additionalEntities = extractAdditionalEntities(
            text: text,
            tokens: features.structure.tokens,
            partsOfSpeech: features.structure.partsOfSpeech
        )
        
        entities.append(contentsOf: additionalEntities)
        
        // Deduplicate and merge similar entities
        entities = deduplicateEntities(entities)
        
        return entities
    }
    
    private func extractAdditionalEntities(
        text: String,
        tokens: [String],
        partsOfSpeech: [String]
    ) -> [Entity] {
        
        var entities: [Entity] = []
        
        // Look for action-object patterns
        for i in 0..<tokens.count {
            if i < partsOfSpeech.count && partsOfSpeech[i].contains("Verb") {
                // Found a verb, look for its object
                if i + 1 < tokens.count && i + 1 < partsOfSpeech.count {
                    let nextToken = tokens[i + 1]
                    let nextPOS = partsOfSpeech[i + 1]
                    
                    if nextPOS.contains("Noun") || nextToken.first?.isUppercase == true {
                        entities.append(Entity(
                            text: nextToken,
                            type: determineEntityType(token: nextToken, pos: nextPOS),
                            role: "action_target",
                            confidence: 0.8
                        ))
                    }
                }
                
                // The verb itself is an action entity
                entities.append(Entity(
                    text: tokens[i],
                    type: .action,
                    role: "primary_action",
                    confidence: 0.9
                ))
            }
        }
        
        return entities
    }
    
    private func determineEntityType(token: String, pos: String) -> EntityType {
        // Intelligent type determination without hardcoding
        if token.first?.isUppercase == true && token != "I" {
            return .application  // Likely an app name
        } else if pos.contains("Verb") {
            return .action
        } else if pos.contains("Noun") {
            return .object
        } else {
            return .learned
        }
    }
    
    private func deduplicateEntities(_ entities: [Entity]) -> [Entity] {
        var seen = Set<String>()
        var deduped: [Entity] = []
        
        for entity in entities {
            if !seen.contains(entity.text.lowercased()) {
                seen.insert(entity.text.lowercased())
                deduped.append(entity)
            }
        }
        
        return deduped
    }
}

// MARK: - Feature Extractor

public class FeatureExtractor {
    
    public func extractFeatures(
        from patterns: [CommandPattern],
        context: ContextualInformation
    ) -> [Double] {
        
        var features: [Double] = []
        
        // Pattern-based features
        if let pattern = patterns.first {
            features.append(contentsOf: extractPatternFeatures(pattern))
        } else {
            features.append(contentsOf: Array(repeating: 0.0, count: 20))
        }
        
        // Context-based features
        features.append(contentsOf: extractContextFeatures(context))
        
        // Ensure consistent feature vector size
        while features.count < 50 {
            features.append(0.0)
        }
        
        return Array(features.prefix(50))
    }
    
    public func extractPatternFeatures(text: String) -> [String: Double] {
        var features: [String: Double] = [:]
        
        let words = text.split(separator: " ")
        
        // Length features
        features["word_count"] = Double(words.count)
        features["char_count"] = Double(text.count)
        features["avg_word_length"] = words.isEmpty ? 0 : Double(text.count) / Double(words.count)
        
        // Structure features
        features["has_question_mark"] = text.contains("?") ? 1.0 : 0.0
        features["has_exclamation"] = text.contains("!") ? 1.0 : 0.0
        features["starts_with_capital"] = text.first?.isUppercase == true ? 1.0 : 0.0
        
        // Word type features
        features["has_verb"] = words.contains { isLikelyVerb($0) } ? 1.0 : 0.0
        features["has_noun"] = words.contains { isLikelyNoun($0) } ? 1.0 : 0.0
        
        return features
    }
    
    private func extractPatternFeatures(_ pattern: CommandPattern) -> [Double] {
        var features: [Double] = []
        
        // Structure features
        features.append(Double(pattern.structure.tokens.count))
        features.append(pattern.complexity)
        features.append(pattern.sentiment)
        features.append(Double(pattern.entities.count))
        
        // Sentence type features (one-hot encoding)
        switch pattern.structure.sentenceType {
        case .imperative:
            features.append(contentsOf: [1, 0, 0, 0])
        case .interrogative:
            features.append(contentsOf: [0, 1, 0, 0])
        case .declarative:
            features.append(contentsOf: [0, 0, 1, 0])
        case .exclamatory:
            features.append(contentsOf: [0, 0, 0, 1])
        }
        
        // Entity type distribution
        let actionCount = pattern.entities.filter { $0.type == .action }.count
        let objectCount = pattern.entities.filter { $0.type == .object }.count
        let appCount = pattern.entities.filter { $0.type == .application }.count
        
        features.append(Double(actionCount))
        features.append(Double(objectCount))
        features.append(Double(appCount))
        
        // Dependency features
        features.append(Double(pattern.structure.dependencies.count))
        
        // POS distribution
        let verbCount = pattern.structure.partsOfSpeech.filter { $0.contains("Verb") }.count
        let nounCount = pattern.structure.partsOfSpeech.filter { $0.contains("Noun") }.count
        
        features.append(Double(verbCount))
        features.append(Double(nounCount))
        
        return features
    }
    
    private func extractContextFeatures(_ context: ContextualInformation) -> [Double] {
        var features: [Double] = []
        
        // Previous command features
        features.append(Double(context.previousCommands.count))
        
        // User state features
        features.append(context.userState.cognitiveLoad)
        features.append(context.userState.frustrationLevel)
        features.append(context.userState.expertise)
        
        // Working pattern (one-hot encoding)
        switch context.userState.workingPattern {
        case .focused:
            features.append(contentsOf: [1, 0, 0, 0])
        case .multitasking:
            features.append(contentsOf: [0, 1, 0, 0])
        case .exploring:
            features.append(contentsOf: [0, 0, 1, 0])
        case .automating:
            features.append(contentsOf: [0, 0, 0, 1])
        }
        
        // Temporal features
        features.append(context.temporalContext.isWorkingHours ? 1.0 : 0.0)
        features.append(Double(context.temporalContext.dayOfWeek) / 7.0)
        features.append(min(1.0, context.temporalContext.sessionDuration / 3600.0))
        
        // Application context
        features.append(Double(context.currentApplications.count))
        
        return features
    }
    
    private func isLikelyVerb(_ word: String) -> Bool {
        // Basic heuristic - would be enhanced with proper POS tagging
        let verbEndings = ["ing", "ed", "s"]
        let lowercased = word.lowercased()
        
        return verbEndings.contains { lowercased.hasSuffix($0) }
    }
    
    private func isLikelyNoun(_ word: String) -> Bool {
        // Basic heuristic
        return word.first?.isUppercase == true || word.count > 3
    }
}

// MARK: - Persistence Manager

public class PersistenceManager {
    private let documentsDirectory: URL
    private let patternsFile = "learned_patterns.json"
    private let intentsFile = "intent_patterns.json"
    private let networkFile = "neural_network.json"
    
    public init() {
        self.documentsDirectory = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first!.appendingPathComponent("JARVISClassifier")
        
        // Create directory if needed
        try? FileManager.default.createDirectory(
            at: documentsDirectory,
            withIntermediateDirectories: true
        )
    }
    
    public func savePatterns(_ patterns: [LearnedPattern]) {
        // Implementation would serialize patterns to JSON
        // For now, just creating the file structure
        let url = documentsDirectory.appendingPathComponent(patternsFile)
        // Save logic here
    }
    
    public func loadPatterns() -> [LearnedPattern]? {
        // Implementation would deserialize patterns from JSON
        return nil
    }
    
    public func saveIntentPatterns(_ patterns: [String: IntentPattern]) {
        let url = documentsDirectory.appendingPathComponent(intentsFile)
        // Save logic here
    }
    
    public func loadIntentPatterns() -> [String: IntentPattern]? {
        return nil
    }
    
    public func saveNeuralNetwork(_ network: Any) {
        let url = documentsDirectory.appendingPathComponent(networkFile)
        // Save logic here
    }
    
    public func loadNeuralNetwork() -> Any? {
        return nil
    }
}