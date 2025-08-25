import Foundation
import CoreML
import CreateML

/// Self-Learning Engine - The Brain that Gets Smarter with Every Interaction
public class LearningEngine {
    
    // MARK: - Types
    
    public struct LearnedPattern {
        let id: UUID
        let pattern: String
        let features: [String: Double]
        let classification: CommandType
        let confidence: Double
        let timestamp: Date
        let reinforcementCount: Int
        let successRate: Double
    }
    
    public struct IntentPattern {
        let intent: String
        let linguisticMarkers: [String]
        let contextualFactors: [String: Double]
        let confidence: Double
        let learned: Date
    }
    
    private struct NeuralNetwork {
        var weights: [[Double]]
        var biases: [Double]
        let learningRate: Double
        
        mutating func train(input: [Double], expected: [Double]) {
            // Simplified neural network training
            let output = forward(input: input)
            let error = zip(expected, output).map { $0 - $1 }
            
            // Backpropagation
            for i in 0..<weights.count {
                for j in 0..<weights[i].count {
                    weights[i][j] += learningRate * error[i] * input[j]
                }
                biases[i] += learningRate * error[i]
            }
        }
        
        func forward(input: [Double]) -> [Double] {
            var output = [Double]()
            for i in 0..<weights.count {
                var sum = biases[i]
                for j in 0..<input.count {
                    sum += weights[i][j] * input[j]
                }
                output.append(sigmoid(sum))
            }
            return output
        }
        
        private func sigmoid(_ x: Double) -> Double {
            return 1.0 / (1.0 + exp(-x))
        }
    }
    
    // MARK: - Properties
    
    private var learnedPatterns: [LearnedPattern] = []
    private var intentPatterns: [String: IntentPattern] = [:]
    private var neuralNetwork: NeuralNetwork
    private var accuracyHistory: [(date: Date, accuracy: Double)] = []
    private let persistenceManager: PersistenceManager
    private let featureExtractor: FeatureExtractor
    
    public var totalPatternsLearned: Int {
        return learnedPatterns.count
    }
    
    // MARK: - Initialization
    
    public init() {
        // Initialize neural network with random weights
        self.neuralNetwork = NeuralNetwork(
            weights: Array(repeating: Array(repeating: 0, count: 50), count: 10).map { _ in
                Array(repeating: 0, count: 50).map { _ in Double.random(in: -1...1) }
            },
            biases: Array(repeating: 0, count: 10).map { _ in Double.random(in: -1...1) },
            learningRate: 0.1
        )
        
        self.persistenceManager = PersistenceManager()
        self.featureExtractor = FeatureExtractor()
        
        // Load existing patterns
        loadLearnedPatterns()
    }
    
    // MARK: - Public Methods
    
    public func loadLearnedPatterns() {
        if let patterns = persistenceManager.loadPatterns() {
            self.learnedPatterns = patterns
        }
        
        if let intents = persistenceManager.loadIntentPatterns() {
            self.intentPatterns = intents
        }
        
        if let network = persistenceManager.loadNeuralNetwork() {
            self.neuralNetwork = network
        }
    }
    
    public func calculateIntentProbabilities(
        patterns: [CommandPattern],
        context: ContextualInformation
    ) -> [String: Double] {
        
        var probabilities: [String: Double] = [:]
        
        // Extract features from patterns
        let features = featureExtractor.extractFeatures(from: patterns, context: context)
        
        // Use neural network for prediction
        let predictions = neuralNetwork.forward(input: features)
        
        // Map predictions to learned intents
        for (index, intent) in Array(intentPatterns.keys).enumerated() {
            if index < predictions.count {
                probabilities[intent] = predictions[index]
            }
        }
        
        // Discover new intents if confidence is low
        let maxConfidence = probabilities.values.max() ?? 0
        if maxConfidence < 0.3 {
            let newIntent = discoverNewIntent(from: patterns)
            probabilities[newIntent] = 0.5 // Medium confidence for new discoveries
        }
        
        return probabilities
    }
    
    public func calculateTypeScore(
        type: CommandType,
        patterns: [CommandPattern],
        intentProbabilities: [String: Double],
        context: ContextualInformation
    ) -> Double {
        
        var score = 0.0
        var weightSum = 0.0
        
        // Pattern matching score (no hardcoding)
        for pattern in learnedPatterns where pattern.classification == type {
            let similarity = calculateSimilarity(
                pattern: pattern,
                currentPatterns: patterns,
                context: context
            )
            
            let weight = pattern.confidence * pattern.successRate
            score += similarity * weight
            weightSum += weight
        }
        
        // Intent alignment score
        for (intent, probability) in intentProbabilities {
            if isIntentRelevantForType(intent: intent, type: type) {
                score += probability * 0.3
                weightSum += 0.3
            }
        }
        
        // Context relevance score
        let contextScore = calculateContextRelevance(type: type, context: context)
        score += contextScore * 0.2
        weightSum += 0.2
        
        return weightSum > 0 ? score / weightSum : 0
    }
    
    public func learn(
        from text: String,
        classification: CommandAnalysis,
        context: ContextualInformation
    ) {
        // Create new pattern
        let pattern = LearnedPattern(
            id: UUID(),
            pattern: text,
            features: featureExtractor.extractPatternFeatures(text: text),
            classification: classification.type,
            confidence: classification.confidence,
            timestamp: Date(),
            reinforcementCount: 1,
            successRate: classification.confidence
        )
        
        // Add to learned patterns
        learnedPatterns.append(pattern)
        
        // Update neural network
        let input = featureExtractor.extractFeatures(
            from: [createCommandPattern(from: text)],
            context: context
        )
        let expected = createExpectedOutput(for: classification.type)
        neuralNetwork.train(input: input, expected: expected)
        
        // Update intent patterns
        if let existingIntent = intentPatterns[classification.intent.primary] {
            // Reinforce existing intent
            intentPatterns[classification.intent.primary] = IntentPattern(
                intent: existingIntent.intent,
                linguisticMarkers: existingIntent.linguisticMarkers,
                contextualFactors: existingIntent.contextualFactors,
                confidence: min(1.0, existingIntent.confidence + 0.05),
                learned: existingIntent.learned
            )
        } else {
            // Learn new intent
            intentPatterns[classification.intent.primary] = IntentPattern(
                intent: classification.intent.primary,
                linguisticMarkers: extractLinguisticMarkers(from: text),
                contextualFactors: [:],
                confidence: 0.5,
                learned: Date()
            )
        }
        
        // Persist learning
        persistenceManager.savePatterns(learnedPatterns)
        persistenceManager.saveIntentPatterns(intentPatterns)
        persistenceManager.saveNeuralNetwork(neuralNetwork)
        
        // Update accuracy tracking
        updateAccuracyMetrics()
    }
    
    public func updateFromFeedback(_ feedback: UserFeedback) {
        // Find and update relevant patterns
        for i in 0..<learnedPatterns.count {
            if learnedPatterns[i].pattern.lowercased() == feedback.originalCommand.lowercased() {
                // Update success rate
                let newSuccessRate = feedback.userRating
                learnedPatterns[i] = LearnedPattern(
                    id: learnedPatterns[i].id,
                    pattern: learnedPatterns[i].pattern,
                    features: learnedPatterns[i].features,
                    classification: feedback.shouldBe,
                    confidence: max(learnedPatterns[i].confidence, feedback.userRating),
                    timestamp: learnedPatterns[i].timestamp,
                    reinforcementCount: learnedPatterns[i].reinforcementCount + 1,
                    successRate: newSuccessRate
                )
            }
        }
        
        // Create corrected pattern if not found
        if !learnedPatterns.contains(where: { $0.pattern.lowercased() == feedback.originalCommand.lowercased() }) {
            let pattern = LearnedPattern(
                id: UUID(),
                pattern: feedback.originalCommand,
                features: featureExtractor.extractPatternFeatures(text: feedback.originalCommand),
                classification: feedback.shouldBe,
                confidence: feedback.userRating,
                timestamp: Date(),
                reinforcementCount: 1,
                successRate: feedback.userRating
            )
            learnedPatterns.append(pattern)
        }
        
        // Retrain neural network with corrected data
        let context = ContextualInformation(
            previousCommands: [],
            currentApplications: [],
            userState: UserState(
                workingPattern: .focused,
                cognitiveLoad: 0.5,
                frustrationLevel: 0.2,
                expertise: 0.7
            ),
            temporalContext: TemporalContext(
                timeOfDay: Date(),
                dayOfWeek: Calendar.current.component(.weekday, from: Date()),
                isWorkingHours: true,
                sessionDuration: 0
            ),
            environmentalFactors: [:]
        )
        
        let input = featureExtractor.extractFeatures(
            from: [createCommandPattern(from: feedback.originalCommand)],
            context: context
        )
        let expected = createExpectedOutput(for: feedback.shouldBe)
        neuralNetwork.train(input: input, expected: expected)
        
        // Persist updates
        persistenceManager.savePatterns(learnedPatterns)
        persistenceManager.saveNeuralNetwork(neuralNetwork)
    }
    
    public func isIntentRelevantForType(intent: String, type: CommandType) -> Bool {
        // Learn associations between intents and types
        let relevantPatterns = learnedPatterns.filter { 
            $0.classification == type && 
            $0.pattern.lowercased().contains(intent.lowercased())
        }
        
        // If we have learned patterns, use them
        if !relevantPatterns.isEmpty {
            let avgConfidence = relevantPatterns.map { $0.confidence }.reduce(0, +) / Double(relevantPatterns.count)
            return avgConfidence > 0.6
        }
        
        // Otherwise, learn from context
        return false
    }
    
    public func getAccuracyMetrics() -> AccuracyMetrics {
        let recentAccuracy = accuracyHistory.suffix(100).map { $0.accuracy }.reduce(0, +) / Double(min(100, accuracyHistory.count))
        let improvement = calculateAccuracyImprovement()
        
        return AccuracyMetrics(
            currentAccuracy: recentAccuracy,
            improvement: improvement,
            totalSamples: accuracyHistory.count
        )
    }
    
    public func getAccuracyImprovement() -> Double {
        return calculateAccuracyImprovement()
    }
    
    public func getCommonErrors() -> [(from: CommandType, to: CommandType, count: Int)] {
        // Analyze patterns with low success rates
        var errors: [String: Int] = [:]
        
        for pattern in learnedPatterns where pattern.successRate < 0.5 {
            let key = "\(pattern.classification)"
            errors[key, default: 0] += 1
        }
        
        return errors.map { (CommandType.unknown, CommandType.unknown, $0.value) }
    }
    
    public func getAdaptationRate() -> Double {
        // Calculate how quickly the system is learning
        let recentPatterns = learnedPatterns.filter { 
            $0.timestamp.timeIntervalSinceNow > -86400 // Last 24 hours
        }
        
        return Double(recentPatterns.count) / max(1, Double(learnedPatterns.count))
    }
    
    public func getUpdatedPatterns() -> [LearnedPattern] {
        return learnedPatterns
    }
    
    // MARK: - Private Methods
    
    private func calculateSimilarity(
        pattern: LearnedPattern,
        currentPatterns: [CommandPattern],
        context: ContextualInformation
    ) -> Double {
        
        // Feature-based similarity
        guard let currentPattern = currentPatterns.first else { return 0 }
        
        let currentFeatures = featureExtractor.extractPatternFeatures(
            text: currentPattern.entities.map { $0.text }.joined(separator: " ")
        )
        
        var similarity = 0.0
        var featureCount = 0.0
        
        for (key, value) in pattern.features {
            if let currentValue = currentFeatures[key] {
                similarity += 1.0 - abs(value - currentValue)
                featureCount += 1
            }
        }
        
        return featureCount > 0 ? similarity / featureCount : 0
    }
    
    private func calculateContextRelevance(
        type: CommandType,
        context: ContextualInformation
    ) -> Double {
        
        // Learn context relevance from patterns
        let contextualPatterns = learnedPatterns.filter { pattern in
            pattern.classification == type &&
            pattern.timestamp.timeIntervalSinceNow > -3600 // Recent patterns
        }
        
        if contextualPatterns.isEmpty {
            return 0.5 // Neutral relevance for unknown contexts
        }
        
        let avgSuccess = contextualPatterns.map { $0.successRate }.reduce(0, +) / Double(contextualPatterns.count)
        return avgSuccess
    }
    
    private func discoverNewIntent(from patterns: [CommandPattern]) -> String {
        // Generate intent name from linguistic patterns
        guard let pattern = patterns.first else { return "unknown_intent" }
        
        let verbs = pattern.structure.tokens.enumerated()
            .filter { pattern.structure.partsOfSpeech[$0.offset].contains("VERB") }
            .map { $0.element }
        
        let nouns = pattern.structure.tokens.enumerated()
            .filter { pattern.structure.partsOfSpeech[$0.offset].contains("NOUN") }
            .map { $0.element }
        
        if let verb = verbs.first, let noun = nouns.first {
            return "\(verb)_\(noun)"
        } else if let verb = verbs.first {
            return "\(verb)_action"
        } else if let noun = nouns.first {
            return "query_\(noun)"
        }
        
        return "discovered_intent_\(UUID().uuidString.prefix(8))"
    }
    
    private func createCommandPattern(from text: String) -> CommandPattern {
        let tokens = text.split(separator: " ").map { String($0) }
        
        return CommandPattern(
            structure: LinguisticStructure(
                tokens: tokens,
                partsOfSpeech: Array(repeating: "UNKNOWN", count: tokens.count),
                dependencies: [],
                sentenceType: .imperative
            ),
            entities: [],
            sentiment: 0,
            complexity: Double(tokens.count) / 10.0
        )
    }
    
    private func createExpectedOutput(for type: CommandType) -> [Double] {
        var output = Array(repeating: 0.0, count: CommandType.allCases.count)
        if let index = CommandType.allCases.firstIndex(of: type) {
            output[index] = 1.0
        }
        return output
    }
    
    private func extractLinguisticMarkers(from text: String) -> [String] {
        return text.split(separator: " ").map { String($0) }
    }
    
    private func updateAccuracyMetrics() {
        let recentPatterns = learnedPatterns.suffix(100)
        let accuracy = recentPatterns.map { $0.successRate }.reduce(0, +) / Double(recentPatterns.count)
        
        accuracyHistory.append((date: Date(), accuracy: accuracy))
        
        // Keep only recent history
        if accuracyHistory.count > 1000 {
            accuracyHistory = Array(accuracyHistory.suffix(1000))
        }
    }
    
    private func calculateAccuracyImprovement() -> Double {
        guard accuracyHistory.count >= 2 else { return 0 }
        
        let oldAccuracy = accuracyHistory.prefix(10).map { $0.accuracy }.reduce(0, +) / 10.0
        let newAccuracy = accuracyHistory.suffix(10).map { $0.accuracy }.reduce(0, +) / 10.0
        
        return newAccuracy - oldAccuracy
    }
}

// MARK: - Supporting Types

public struct AccuracyMetrics {
    public let currentAccuracy: Double
    public let improvement: Double
    public let totalSamples: Int
}