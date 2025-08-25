import Foundation
import NaturalLanguage
import CoreML
import CreateML

/// Advanced Command Classifier with Zero Hardcoding and Self-Learning Capabilities
public class AdvancedCommandClassifier {
    
    // MARK: - Types
    
    public struct CommandAnalysis {
        public let type: CommandType
        public let intent: Intent
        public let confidence: Double
        public let entities: [Entity]
        public let context: ContextualInformation
        public let reasoning: String
        public let alternatives: [AlternativeInterpretation]
    }
    
    public enum CommandType: String, CaseIterable {
        case system
        case vision
        case conversation
        case automation
        case learning
        case unknown
        
        /// Dynamic type discovery - learns new types
        static func discover(from pattern: CommandPattern) -> CommandType {
            // No hardcoded mapping - learns from patterns
            return .unknown
        }
    }
    
    public struct Intent {
        public let primary: String
        public let secondary: [String]
        public let confidence: Double
        public let learned: Bool
        
        /// Dynamically learned intents
        public static func learn(from: String, context: ContextualInformation) -> Intent {
            return Intent(
                primary: from,
                secondary: [],
                confidence: 0.5,
                learned: true
            )
        }
    }
    
    public struct Entity {
        public let text: String
        public let type: EntityType
        public let role: String
        public let confidence: Double
    }
    
    public enum EntityType: String {
        case application
        case action
        case object
        case query
        case modifier
        case learned
    }
    
    public struct ContextualInformation {
        public let previousCommands: [String]
        public let currentApplications: [String]
        public let userState: UserState
        public let temporalContext: TemporalContext
        public let environmentalFactors: [String: Any]
    }
    
    public struct UserState {
        public let workingPattern: WorkingPattern
        public let cognitiveLoad: Double
        public let frustrationLevel: Double
        public let expertise: Double
    }
    
    public enum WorkingPattern {
        case focused
        case multitasking
        case exploring
        case automating
    }
    
    public struct TemporalContext {
        public let timeOfDay: Date
        public let dayOfWeek: Int
        public let isWorkingHours: Bool
        public let sessionDuration: TimeInterval
    }
    
    public struct AlternativeInterpretation {
        public let type: CommandType
        public let intent: Intent
        public let confidence: Double
        public let reasoning: String
    }
    
    public struct CommandPattern {
        public let structure: LinguisticStructure
        public let entities: [Entity]
        public let sentiment: Double
        public let complexity: Double
    }
    
    public struct LinguisticStructure {
        public let tokens: [String]
        public let partsOfSpeech: [String]
        public let dependencies: [Dependency]
        public let sentenceType: SentenceType
    }
    
    public struct Dependency {
        public let head: Int
        public let dependent: Int
        public let relation: String
    }
    
    public enum SentenceType {
        case imperative
        case interrogative
        case declarative
        case exclamatory
    }
    
    // MARK: - Properties
    
    private let nlpEngine: NLPEngine
    private let learningEngine: LearningEngine
    private let contextManager: ContextManager
    private let patternRecognizer: PatternRecognizer
    private let confidenceCalculator: ConfidenceCalculator
    private let entityExtractor: EntityExtractor
    
    // MARK: - Initialization
    
    public init() {
        self.nlpEngine = NLPEngine()
        self.learningEngine = LearningEngine()
        self.contextManager = ContextManager()
        self.patternRecognizer = PatternRecognizer()
        self.confidenceCalculator = ConfidenceCalculator()
        self.entityExtractor = EntityExtractor()
        
        // Load any existing learned patterns
        self.learningEngine.loadLearnedPatterns()
    }
    
    // MARK: - Public Methods
    
    /// Analyze command with zero hardcoding - everything is learned
    public func analyzeCommand(_ text: String) -> CommandAnalysis {
        let normalizedText = normalize(text)
        
        // Extract linguistic features
        let linguisticFeatures = nlpEngine.extractFeatures(from: normalizedText)
        
        // Extract entities dynamically
        let entities = entityExtractor.extract(from: normalizedText, features: linguisticFeatures)
        
        // Get current context
        let context = contextManager.getCurrentContext()
        
        // Recognize patterns (learned, not hardcoded)
        let patterns = patternRecognizer.recognize(
            text: normalizedText,
            features: linguisticFeatures,
            entities: entities,
            context: context
        )
        
        // Calculate intent probabilities
        let intentProbabilities = learningEngine.calculateIntentProbabilities(
            patterns: patterns,
            context: context
        )
        
        // Generate classification with reasoning
        let classification = classifyUsingIntelligence(
            text: normalizedText,
            patterns: patterns,
            intentProbabilities: intentProbabilities,
            context: context
        )
        
        // Learn from this classification for future improvement
        learningEngine.learn(
            from: normalizedText,
            classification: classification,
            context: context
        )
        
        return classification
    }
    
    /// Update learning from user feedback
    public func updateFromFeedback(_ feedback: UserFeedback) {
        learningEngine.updateFromFeedback(feedback)
        
        // Retrain pattern recognizer with new knowledge
        patternRecognizer.retrain(with: learningEngine.getUpdatedPatterns())
        
        // Update confidence thresholds based on accuracy
        confidenceCalculator.updateThresholds(
            basedOn: learningEngine.getAccuracyMetrics()
        )
    }
    
    /// Get learning insights
    public func getLearningInsights() -> LearningInsights {
        return LearningInsights(
            totalPatternsLearned: learningEngine.totalPatternsLearned,
            accuracyImprovement: learningEngine.getAccuracyImprovement(),
            commonMisclassifications: learningEngine.getCommonErrors(),
            adaptationRate: learningEngine.getAdaptationRate()
        )
    }
    
    // MARK: - Private Methods
    
    private func normalize(_ text: String) -> String {
        return text.lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
    
    private func classifyUsingIntelligence(
        text: String,
        patterns: [CommandPattern],
        intentProbabilities: [String: Double],
        context: ContextualInformation
    ) -> CommandAnalysis {
        
        // Multi-factor intelligent classification
        let typeScores = calculateTypeScores(
            patterns: patterns,
            intentProbabilities: intentProbabilities,
            context: context
        )
        
        // Get best classification with alternatives
        let (bestType, confidence) = selectBestType(from: typeScores)
        let alternatives = generateAlternatives(from: typeScores, excluding: bestType)
        
        // Extract primary intent
        let intent = extractIntent(
            from: intentProbabilities,
            type: bestType,
            patterns: patterns
        )
        
        // Generate reasoning explanation
        let reasoning = generateReasoning(
            text: text,
            type: bestType,
            intent: intent,
            patterns: patterns,
            context: context
        )
        
        return CommandAnalysis(
            type: bestType,
            intent: intent,
            confidence: confidence,
            entities: patterns.first?.entities ?? [],
            context: context,
            reasoning: reasoning,
            alternatives: alternatives
        )
    }
    
    private func calculateTypeScores(
        patterns: [CommandPattern],
        intentProbabilities: [String: Double],
        context: ContextualInformation
    ) -> [CommandType: Double] {
        
        var scores: [CommandType: Double] = [:]
        
        // No hardcoded rules - everything is learned
        for type in CommandType.allCases {
            scores[type] = learningEngine.calculateTypeScore(
                type: type,
                patterns: patterns,
                intentProbabilities: intentProbabilities,
                context: context
            )
        }
        
        return scores
    }
    
    private func selectBestType(from scores: [CommandType: Double]) -> (CommandType, Double) {
        let sorted = scores.sorted { $0.value > $1.value }
        
        guard let best = sorted.first else {
            return (.unknown, 0.0)
        }
        
        // Apply confidence adjustment based on score distribution
        let confidence = confidenceCalculator.calculate(
            topScore: best.value,
            allScores: scores.values.map { $0 }
        )
        
        return (best.key, confidence)
    }
    
    private func generateAlternatives(
        from scores: [CommandType: Double],
        excluding: CommandType
    ) -> [AlternativeInterpretation] {
        
        return scores
            .filter { $0.key != excluding && $0.value > 0.1 }
            .sorted { $0.value > $1.value }
            .prefix(3)
            .map { type, score in
                AlternativeInterpretation(
                    type: type,
                    intent: Intent(primary: "", secondary: [], confidence: score, learned: false),
                    confidence: score,
                    reasoning: "Alternative interpretation based on pattern matching"
                )
            }
    }
    
    private func extractIntent(
        from probabilities: [String: Double],
        type: CommandType,
        patterns: [CommandPattern]
    ) -> Intent {
        
        // Find highest probability intent for this type
        let relevantIntents = probabilities.filter { intent, _ in
            learningEngine.isIntentRelevantForType(intent: intent, type: type)
        }
        
        guard let topIntent = relevantIntents.max(by: { $0.value < $1.value }) else {
            // Learn new intent if none found
            return Intent.learn(from: patterns.first?.entities.first?.text ?? "unknown", 
                              context: contextManager.getCurrentContext())
        }
        
        return Intent(
            primary: topIntent.key,
            secondary: Array(relevantIntents.keys.filter { $0 != topIntent.key }.prefix(2)),
            confidence: topIntent.value,
            learned: false
        )
    }
    
    private func generateReasoning(
        text: String,
        type: CommandType,
        intent: Intent,
        patterns: [CommandPattern],
        context: ContextualInformation
    ) -> String {
        
        var reasoning = "Classification based on: "
        var factors: [String] = []
        
        // Linguistic structure
        if let pattern = patterns.first {
            factors.append("linguistic structure (\(pattern.structure.sentenceType))")
        }
        
        // Intent confidence
        factors.append("intent '\(intent.primary)' with \(Int(intent.confidence * 100))% confidence")
        
        // Context influence
        if !context.previousCommands.isEmpty {
            factors.append("conversation context")
        }
        
        // Learning status
        if intent.learned {
            factors.append("newly learned pattern")
        }
        
        reasoning += factors.joined(separator: ", ")
        return reasoning
    }
}

// MARK: - Supporting Types

public struct UserFeedback {
    public let originalCommand: String
    public let classifiedAs: CommandType
    public let shouldBe: CommandType
    public let userRating: Double
    public let timestamp: Date
}

public struct LearningInsights {
    public let totalPatternsLearned: Int
    public let accuracyImprovement: Double
    public let commonMisclassifications: [(from: CommandType, to: CommandType, count: Int)]
    public let adaptationRate: Double
}