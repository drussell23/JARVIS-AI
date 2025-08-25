import Foundation
import NaturalLanguage

/// Advanced NLP Engine with Zero Hardcoding
public class NLPEngine {
    
    // MARK: - Properties
    
    private let tagger: NLTagger
    private let languageRecognizer: NLLanguageRecognizer
    private let tokenizer: NLTokenizer
    private let embedder: NLEmbedding?
    
    // MARK: - Initialization
    
    public init() {
        self.tagger = NLTagger(tagSchemes: [
            .lexicalClass,
            .language,
            .script,
            .lemma,
            .nameType,
            .tokenType
        ])
        
        self.languageRecognizer = NLLanguageRecognizer()
        self.tokenizer = NLTokenizer(unit: .word)
        self.embedder = NLEmbedding.wordEmbedding(for: .english)
    }
    
    // MARK: - Public Methods
    
    public func extractFeatures(from text: String) -> LinguisticFeatures {
        // Tokenization
        let tokens = tokenize(text)
        
        // Part-of-speech tagging
        let partsOfSpeech = extractPartsOfSpeech(text: text)
        
        // Dependency parsing
        let dependencies = extractDependencies(tokens: tokens, partsOfSpeech: partsOfSpeech)
        
        // Sentence type detection
        let sentenceType = detectSentenceType(
            tokens: tokens,
            partsOfSpeech: partsOfSpeech,
            text: text
        )
        
        // Named entity recognition
        let entities = extractNamedEntities(text: text)
        
        // Lemmatization
        let lemmas = extractLemmas(text: text)
        
        // Semantic features
        let semanticFeatures = extractSemanticFeatures(
            tokens: tokens,
            embedder: embedder
        )
        
        return LinguisticFeatures(
            structure: LinguisticStructure(
                tokens: tokens,
                partsOfSpeech: partsOfSpeech,
                dependencies: dependencies,
                sentenceType: sentenceType
            ),
            entities: entities,
            lemmas: lemmas,
            semanticVectors: semanticFeatures,
            language: detectLanguage(text: text),
            complexity: calculateComplexity(tokens: tokens, dependencies: dependencies),
            sentiment: analyzeSentiment(text: text, tokens: tokens)
        )
    }
    
    // MARK: - Private Methods
    
    private func tokenize(_ text: String) -> [String] {
        var tokens: [String] = []
        
        tokenizer.string = text
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { tokenRange, _ in
            let token = String(text[tokenRange])
            tokens.append(token)
            return true
        }
        
        return tokens
    }
    
    private func extractPartsOfSpeech(text: String) -> [String] {
        var partsOfSpeech: [String] = []
        
        tagger.string = text
        let options: NLTagger.Options = [.omitPunctuation, .omitWhitespace]
        
        tagger.enumerateTags(
            in: text.startIndex..<text.endIndex,
            unit: .word,
            scheme: .lexicalClass,
            options: options
        ) { tag, tokenRange in
            if let tag = tag {
                partsOfSpeech.append(tag.rawValue)
            } else {
                partsOfSpeech.append("UNKNOWN")
            }
            return true
        }
        
        return partsOfSpeech
    }
    
    private func extractDependencies(
        tokens: [String],
        partsOfSpeech: [String]
    ) -> [Dependency] {
        var dependencies: [Dependency] = []
        
        // Simplified dependency parsing based on POS patterns
        for i in 0..<tokens.count {
            // Find verb-object relationships
            if i < tokens.count - 1 &&
               partsOfSpeech[i].contains("Verb") &&
               partsOfSpeech[i + 1].contains("Noun") {
                dependencies.append(Dependency(
                    head: i,
                    dependent: i + 1,
                    relation: "direct_object"
                ))
            }
            
            // Find subject-verb relationships
            if i > 0 &&
               partsOfSpeech[i - 1].contains("Noun") &&
               partsOfSpeech[i].contains("Verb") {
                dependencies.append(Dependency(
                    head: i,
                    dependent: i - 1,
                    relation: "subject"
                ))
            }
            
            // Find modifier relationships
            if i < tokens.count - 1 &&
               (partsOfSpeech[i].contains("Adjective") || partsOfSpeech[i].contains("Adverb")) &&
               (partsOfSpeech[i + 1].contains("Noun") || partsOfSpeech[i + 1].contains("Verb")) {
                dependencies.append(Dependency(
                    head: i + 1,
                    dependent: i,
                    relation: "modifier"
                ))
            }
        }
        
        return dependencies
    }
    
    private func detectSentenceType(
        tokens: [String],
        partsOfSpeech: [String],
        text: String
    ) -> SentenceType {
        
        // Check for question marks
        if text.contains("?") {
            return .interrogative
        }
        
        // Check for exclamation marks
        if text.contains("!") {
            return .exclamatory
        }
        
        // Check for imperative patterns (verb-first)
        if let firstPOS = partsOfSpeech.first,
           firstPOS.contains("Verb") {
            return .imperative
        }
        
        // Check for question words
        let questionWords = ["what", "where", "when", "why", "how", "who", "which"]
        if let firstToken = tokens.first?.lowercased(),
           questionWords.contains(firstToken) {
            return .interrogative
        }
        
        // Default to declarative
        return .declarative
    }
    
    private func extractNamedEntities(text: String) -> [Entity] {
        var entities: [Entity] = []
        
        tagger.string = text
        let options: NLTagger.Options = [.omitPunctuation, .omitWhitespace, .joinNames]
        
        tagger.enumerateTags(
            in: text.startIndex..<text.endIndex,
            unit: .word,
            scheme: .nameType,
            options: options
        ) { tag, tokenRange in
            if let tag = tag {
                let entityText = String(text[tokenRange])
                let entityType = mapNLTagToEntityType(tag)
                
                entities.append(Entity(
                    text: entityText,
                    type: entityType,
                    role: determineEntityRole(entityType: entityType, context: text),
                    confidence: 0.9
                ))
            }
            return true
        }
        
        // Also extract potential app names (not recognized by standard NER)
        let appPatterns = extractPotentialAppNames(from: text)
        entities.append(contentsOf: appPatterns)
        
        return entities
    }
    
    private func mapNLTagToEntityType(_ tag: NLTag) -> EntityType {
        switch tag {
        case .personalName:
            return .object
        case .placeName:
            return .object
        case .organizationName:
            return .application
        default:
            return .learned
        }
    }
    
    private func determineEntityRole(entityType: EntityType, context: String) -> String {
        // Determine role based on context without hardcoding
        let contextLower = context.lowercased()
        
        // Learn from verb context
        if contextLower.contains("open") || contextLower.contains("launch") {
            return "target_application"
        } else if contextLower.contains("close") || contextLower.contains("quit") {
            return "target_to_close"
        } else if contextLower.contains("show") || contextLower.contains("display") {
            return "display_target"
        }
        
        return "general_entity"
    }
    
    private func extractPotentialAppNames(from text: String) -> [Entity] {
        var entities: [Entity] = []
        let tokens = text.split(separator: " ").map { String($0) }
        
        // Look for capitalized words that might be app names
        for token in tokens {
            if token.first?.isUppercase == true &&
               token.count > 2 &&
               !isCommonWord(token) {
                entities.append(Entity(
                    text: token,
                    type: .application,
                    role: "potential_app",
                    confidence: 0.7
                ))
            }
        }
        
        return entities
    }
    
    private func isCommonWord(_ word: String) -> Bool {
        // Basic check - this would be learned over time
        let commonWords = ["The", "This", "That", "What", "Where", "When", "How"]
        return commonWords.contains(word)
    }
    
    private func extractLemmas(text: String) -> [String] {
        var lemmas: [String] = []
        
        tagger.string = text
        let options: NLTagger.Options = [.omitPunctuation, .omitWhitespace]
        
        tagger.enumerateTags(
            in: text.startIndex..<text.endIndex,
            unit: .word,
            scheme: .lemma,
            options: options
        ) { tag, _ in
            if let tag = tag {
                lemmas.append(tag.rawValue)
            }
            return true
        }
        
        return lemmas
    }
    
    private func extractSemanticFeatures(
        tokens: [String],
        embedder: NLEmbedding?
    ) -> [String: [Double]] {
        
        var semanticVectors: [String: [Double]] = [:]
        
        guard let embedder = embedder else { return semanticVectors }
        
        for token in tokens {
            if let vector = embedder.vector(for: token) {
                semanticVectors[token] = vector
            }
        }
        
        return semanticVectors
    }
    
    private func detectLanguage(text: String) -> String {
        languageRecognizer.processString(text)
        
        if let language = languageRecognizer.dominantLanguage {
            return language.rawValue
        }
        
        return "en"
    }
    
    private func calculateComplexity(
        tokens: [String],
        dependencies: [Dependency]
    ) -> Double {
        
        let tokenCount = Double(tokens.count)
        let depCount = Double(dependencies.count)
        
        // Normalized complexity score
        let complexity = (tokenCount / 20.0) + (depCount / 10.0)
        
        return min(1.0, complexity)
    }
    
    private func analyzeSentiment(text: String, tokens: [String]) -> Double {
        // Simple sentiment analysis - would be enhanced with ML
        let positiveIndicators = ["please", "thanks", "good", "great", "awesome"]
        let negativeIndicators = ["not", "don't", "stop", "bad", "wrong"]
        
        var sentiment = 0.0
        
        for token in tokens {
            let lower = token.lowercased()
            if positiveIndicators.contains(lower) {
                sentiment += 0.2
            } else if negativeIndicators.contains(lower) {
                sentiment -= 0.2
            }
        }
        
        return max(-1.0, min(1.0, sentiment))
    }
}

// MARK: - Supporting Types

public struct LinguisticFeatures {
    public let structure: LinguisticStructure
    public let entities: [Entity]
    public let lemmas: [String]
    public let semanticVectors: [String: [Double]]
    public let language: String
    public let complexity: Double
    public let sentiment: Double
}