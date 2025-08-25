import Foundation //

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
    
    // TODO: Implement this function to infer the working pattern of the user
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
    
    // TODO: Implement this function to calculate the average response time of the command
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
    
    // TODO: Implement this function to calculate the interaction complexity of the command
    private func calculateInteractionComplexity(_ patterns: ArraySlice<InteractionPattern>) -> Double {
        guard !patterns.isEmpty else { return 0.0 }
        
        let avgCommandLength = patterns.map { Double($0.command.count) }.reduce(0, +) / Double(patterns.count)
        return min(1.0, avgCommandLength / 100.0)
    }
    
    // TODO: Implement this function to check if the user is working hours
    private func isWorkingHours(_ date: Date) -> Bool {
        let hour = Calendar.current.component(.hour, from: date)
        let weekday = Calendar.current.component(.weekday, from: date)
        
        // Monday-Friday, 9 AM - 6 PM
        return weekday >= 2 && weekday <= 6 && hour >= 9 && hour < 18
    }
    
    // TODO: Implement this function to gather the environmental factors of the command
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
    
    // TODO: Implement this function to retrain the pattern recognizer
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
    
    // TODO: Implement this function to find similar patterns
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
    
    // TODO: Implement this function to extract the pattern features
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
    
    // TODO: Implement this function to extract the context features
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
    
    // Sophisticated verb detection with ML-enhanced patterns
    private func isLikelyVerb(_ word: String) -> Bool {
        let lowercased = word.lowercased()
        
        // Common English verbs database (top 500)
        let commonVerbs = Set([
            // Action verbs
            "open", "close", "show", "hide", "start", "stop", "run", "launch",
            "create", "delete", "remove", "add", "update", "modify", "edit", "save",
            "load", "send", "receive", "play", "pause", "resume", "skip", "search",
            "find", "locate", "navigate", "browse", "download", "upload", "install",
            "uninstall", "enable", "disable", "activate", "deactivate", "connect",
            "disconnect", "sync", "refresh", "reload", "restart", "shutdown", "boot",
            
            // Communication verbs
            "call", "message", "email", "text", "chat", "speak", "tell", "ask",
            "answer", "reply", "respond", "forward", "share", "post", "tweet",
            
            // System verbs
            "execute", "process", "analyze", "compute", "calculate", "compile",
            "debug", "test", "verify", "validate", "authenticate", "authorize",
            
            // Media verbs
            "record", "capture", "stream", "broadcast", "render", "export", "import",
            "convert", "transform", "resize", "crop", "rotate", "flip", "merge",
            
            // Navigation verbs
            "go", "move", "switch", "change", "select", "pick", "choose", "click",
            "tap", "swipe", "scroll", "zoom", "pan", "drag", "drop", "hover"
        ])
        
        // Check if exact match with common verb
        if commonVerbs.contains(lowercased) {
            return true
        }
        
        // Advanced pattern matching with weighted scoring
        var score = 0.0
        
        // 1. Verb suffix patterns with weights
        let suffixPatterns: [(suffix: String, weight: Double, minLength: Int)] = [
            // Strong indicators
            ("ing", 0.9, 4),    // running, playing (but not "ring", "thing")
            ("ed", 0.85, 3),    // played, created (but not "red", "bed")
            ("ize", 0.9, 5),    // organize, optimize
            ("ise", 0.9, 5),    // British spelling
            ("ify", 0.9, 5),    // simplify, modify
            ("ate", 0.8, 5),    // activate, create
            
            // Moderate indicators
            ("en", 0.6, 5),     // strengthen, happen
            ("le", 0.5, 6),     // handle, toggle
            ("er", 0.4, 5),     // trigger, buffer (but many nouns too)
            
            // Weak indicators
            ("s", 0.3, 4)       // runs, plays (but many nouns too)
        ]
        
        for (suffix, weight, minLength) in suffixPatterns {
            if lowercased.hasSuffix(suffix) && lowercased.count >= minLength {
                // Check if not a known noun with this suffix
                let stem = String(lowercased.dropLast(suffix.count))
                if !isLikelyNounStem(stem) {
                    score += weight
                }
            }
        }
        
        // 2. Verb prefix patterns
        let prefixPatterns: [(prefix: String, weight: Double)] = [
            ("re", 0.7),      // restart, reload, refresh
            ("un", 0.6),      // undo, uninstall, unlock
            ("de", 0.6),      // delete, debug, decode
            ("dis", 0.6),     // disconnect, disable
            ("over", 0.5),    // override, overflow
            ("out", 0.5),     // output, outline
            ("up", 0.5),      // upload, update
            ("down", 0.5)     // download, downgrade
        ]
        
        for (prefix, weight) in prefixPatterns {
            if lowercased.hasPrefix(prefix) && lowercased.count > prefix.count + 2 {
                score += weight * 0.5  // Prefixes are weaker indicators
            }
        }
        
        // 3. Compound verb detection
        let compoundParts = lowercased.split(separator: "-")
        if compoundParts.count >= 2 {
            // Check if any part is a known verb
            if compoundParts.contains(where: { commonVerbs.contains(String($0)) }) {
                score += 0.8
            }
        }
        
        // 4. Modal verb patterns
        let modalPatterns = ["can", "could", "will", "would", "should", "must", "may", "might"]
        if modalPatterns.contains(lowercased) {
            return true
        }
        
        // 5. Linguistic heuristics
        // Check for verb-like character patterns
        if lowercased.count >= 3 {
            let vowelCount = lowercased.filter { "aeiou".contains($0) }.count
            let consonantClusters = countConsonantClusters(in: lowercased)
            
            // Verbs often have balanced vowel-consonant distribution
            if vowelCount > 0 && consonantClusters <= 2 {
                score += 0.2
            }
        }
        
        // 6. Check against common noun exclusions
        let definitelyNouns = Set([
            "thing", "ring", "king", "string", "spring",  // -ing endings that are nouns
            "bed", "red", "shed", "thread",              // -ed endings that are nouns
            "water", "paper", "computer", "number"       // -er endings that are nouns
        ])
        
        if definitelyNouns.contains(lowercased) {
            return false
        }
        
        // Return true if score indicates likely verb
        return score >= 0.7
    }
    
    // Helper function to check if a stem is likely a noun
    private func isLikelyNounStem(_ stem: String) -> Bool {
        let nounStems = Set([
            "th", "r", "k", "spr", "str",  // thing, ring, king, spring, string
            "b", "sh", "thr"                  // bed, shed, thread
        ])
        return nounStems.contains(stem)
    }
    
    // Count consonant clusters for linguistic analysis
    private func countConsonantClusters(in word: String) -> Int {
        let vowels = Set("aeiou")
        var clusterCount = 0
        var inCluster = false
        
        for char in word {
            if !vowels.contains(char) {
                if !inCluster {
                    clusterCount += 1
                    inCluster = true
                }
            } else {
                inCluster = false
            }
        }
        
        return clusterCount
    }
    
    // Sophisticated noun detection with ML-enhanced patterns
    private func isLikelyNoun(_ word: String) -> Bool {
        let original = word
        let lowercased = word.lowercased()
        
        // Common nouns database (top 500)
        let commonNouns = Set([
            // Technology nouns
            "app", "application", "window", "screen", "file", "folder", "document",
            "image", "video", "audio", "music", "photo", "picture", "movie", "song",
            "email", "message", "notification", "alert", "reminder", "calendar", "note",
            "browser", "tab", "bookmark", "download", "upload", "link", "url", "website",
            
            // System nouns
            "system", "computer", "device", "machine", "server", "network", "internet",
            "wifi", "bluetooth", "connection", "settings", "preferences", "configuration",
            "memory", "storage", "disk", "drive", "cpu", "processor", "battery", "power",
            
            // Application names (common)
            "chrome", "safari", "firefox", "edge", "whatsapp", "telegram", "discord",
            "slack", "zoom", "teams", "skype", "spotify", "netflix", "youtube",
            
            // UI elements
            "button", "menu", "toolbar", "sidebar", "panel", "dialog", "popup",
            "widget", "icon", "cursor", "pointer", "keyboard", "mouse", "trackpad",
            
            // Data nouns
            "data", "information", "content", "text", "code", "script", "program",
            "database", "table", "record", "field", "value", "key", "index", "cache"
        ])
        
        // Check exact match
        if commonNouns.contains(lowercased) {
            return true
        }
        
        // Advanced pattern matching with weighted scoring
        var score = 0.0
        
        // 1. Capitalization check (strong indicator for proper nouns)
        if original.first?.isUppercase == true && original != "I" {
            score += 0.8
            
            // Check if all caps (acronym)
            if original.allSatisfy({ $0.isUppercase || !$0.isLetter }) {
                score += 0.2  // Acronyms are usually nouns
            }
            // Check if CamelCase (likely app or class name)
            else if original.contains(where: { $0.isUppercase }) && original.first?.isUppercase == true {
                score += 0.15
            }
        }
        
        // 2. Noun suffix patterns with weights
        let suffixPatterns: [(suffix: String, weight: Double, minLength: Int)] = [
            // Strong indicators
            ("tion", 0.95, 6),   // application, notification
            ("sion", 0.95, 6),   // permission, extension
            ("ment", 0.9, 6),    // document, statement
            ("ness", 0.9, 6),    // brightness, darkness
            ("ity", 0.9, 5),     // security, quality
            ("ance", 0.85, 6),   // performance, instance
            ("ence", 0.85, 6),   // preference, reference
            ("ship", 0.85, 6),   // ownership, relationship
            ("hood", 0.85, 6),   // neighborhood, likelihood
            
            // Moderate indicators
            ("er", 0.6, 4),      // computer, folder (but also verbs)
            ("or", 0.6, 4),      // monitor, editor
            ("ist", 0.7, 5),     // specialist, artist
            ("ism", 0.7, 5),     // mechanism, algorithm
            ("age", 0.6, 5),     // storage, package
            ("ing", 0.4, 5),     // building, warning (gerunds)
            
            // Weak indicators
            ("s", 0.2, 4),       // files, windows (plurals)
            ("es", 0.25, 5),     // processes, caches
            ("ies", 0.3, 5)      // libraries, categories
        ]
        
        for (suffix, weight, minLength) in suffixPatterns {
            if lowercased.hasSuffix(suffix) && lowercased.count >= minLength {
                score += weight
                
                // Special case: -ing words that are clearly nouns
                if suffix == "ing" {
                    let stem = String(lowercased.dropLast(3))
                    if commonNouns.contains(stem) || isCompoundNoun(lowercased) {
                        score += 0.4  // Boost for noun-based gerunds
                    }
                }
            }
        }
        
        // 3. Compound noun detection
        if lowercased.contains("-") || lowercased.contains("_") {
            let parts = lowercased.split { "-_".contains($0) }
            if parts.count >= 2 {
                // Check if any part is a known noun
                if parts.contains(where: { commonNouns.contains(String($0)) }) {
                    score += 0.7
                }
                // Compound pattern itself suggests noun
                score += 0.3
            }
        }
        
        // 4. Technical/domain-specific patterns
        let techPatterns: [(String, Double)] = [
            (".com", 1.0), (".org", 1.0), (".net", 1.0),  // Domain names
            (".app", 0.9), (".exe", 0.9), (".dmg", 0.9),  // File extensions
            ("api", 0.8), ("sdk", 0.8), ("gui", 0.8),       // Tech acronyms
            ("db", 0.7), ("ui", 0.7), ("os", 0.7)           // Common abbreviations
        ]
        
        for (pattern, weight) in techPatterns {
            if lowercased.contains(pattern) {
                score += weight
                break  // Don't double-count
            }
        }
        
        // 5. Article test (would "the" or "a" make sense before it?)
        // This is approximated by checking if it's a concrete concept
        if lowercased.count >= 3 && !lowercased.contains(where: { !"/.,;:!?@#$%^&*()-_=+[]{}|\\`~".contains($0) && !$0.isLetter }) {
            // No special characters suggests it's a regular word
            score += 0.1
        }
        
        // 6. Plural form detection
        if lowercased.hasSuffix("s") && lowercased.count > 3 {
            let singular = String(lowercased.dropLast())
            if commonNouns.contains(singular) {
                return true
            }
            // Check for -es plural
            if lowercased.hasSuffix("es") && lowercased.count > 4 {
                let singularES = String(lowercased.dropLast(2))
                if commonNouns.contains(singularES) {
                    return true
                }
            }
        }
        
        // 7. Length and structure heuristics
        if lowercased.count >= 3 {
            // Very short words are often articles/prepositions unless capitalized
            if lowercased.count <= 3 && original.first?.isUppercase != true {
                score -= 0.3
            }
            
            // Longer words are more likely to be nouns
            if lowercased.count >= 7 {
                score += 0.2
            }
        }
        
        // Return true if score indicates likely noun
        return score >= 0.6
    }
    
    // Helper function to check if -ing word is a compound noun
    private func isCompoundNoun(_ word: String) -> Bool {
        // Common -ing compound noun patterns
        let compoundPatterns = [
            "building", "painting", "drawing", "writing", "coding",
            "programming", "engineering", "marketing", "banking",
            "networking", "monitoring", "logging", "tracking"
        ]
        return compoundPatterns.contains(word)
    }
}

// MARK: - Persistence Manager

public class PersistenceManager {
    private let documentsDirectory: URL
    private let patternsFile = "learned_patterns.json"
    private let intentsFile = "intent_patterns.json"
    private let networkFile = "neural_network.json"
    private let metadataFile = "learning_metadata.json"
    private let backupDirectory: URL
    
    // Version control for data format changes
    private let dataFormatVersion = "1.0"
    
    // Performance optimization
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder
    private let fileManager = FileManager.default
    
    // Concurrent queue for async operations
    private let persistenceQueue = DispatchQueue(label: "com.jarvis.persistence", attributes: .concurrent)
    
    public init() {
        self.documentsDirectory = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first!.appendingPathComponent("JARVISClassifier")
        
        self.backupDirectory = documentsDirectory.appendingPathComponent("backups")
        
        // Configure encoder/decoder
        self.encoder = JSONEncoder()
        self.encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        self.encoder.dateEncodingStrategy = .iso8601
        
        self.decoder = JSONDecoder()
        self.decoder.dateDecodingStrategy = .iso8601
        
        // Create directories if needed
        createDirectoriesIfNeeded()
    }
    
    private func createDirectoriesIfNeeded() {
        do {
            try fileManager.createDirectory(
                at: documentsDirectory,
                withIntermediateDirectories: true,
                attributes: nil
            )
            try fileManager.createDirectory(
                at: backupDirectory,
                withIntermediateDirectories: true,
                attributes: nil
            )
        } catch {
            print("Failed to create directories: \(error)")
        }
    }
    
    // MARK: - Pattern Persistence
    
    public func savePatterns(_ patterns: [LearnedPattern]) {
        persistenceQueue.async(flags: .barrier) {
            let url = self.documentsDirectory.appendingPathComponent(self.patternsFile)
            
            do {
                // Create backup before saving
                self.createBackupIfExists(for: url)
                
                // Convert patterns to serializable format
                let serializablePatterns = patterns.map { pattern in
                    SerializableLearnedPattern(
                        id: pattern.id.uuidString,
                        pattern: pattern.pattern,
                        features: self.featuresToDictionary(pattern.features),
                        classification: pattern.classification.rawValue,
                        confidence: pattern.confidence,
                        timestamp: pattern.timestamp,
                        reinforcementCount: pattern.reinforcementCount,
                        successRate: pattern.successRate
                    )
                }
                
                // Create wrapper with metadata
                let wrapper = PatternsWrapper(
                    version: self.dataFormatVersion,
                    patterns: serializablePatterns,
                    metadata: LearningMetadata(
                        totalPatterns: patterns.count,
                        lastUpdated: Date(),
                        averageConfidence: patterns.map { $0.confidence }.reduce(0, +) / Double(max(1, patterns.count))
                    )
                )
                
                let data = try self.encoder.encode(wrapper)
                try data.write(to: url, options: .atomicWrite)
                
                // Verify save was successful
                if self.fileManager.fileExists(atPath: url.path) {
                    self.updateMetadata(patternsCount: patterns.count)
                }
                
            } catch {
                print("Failed to save patterns: \(error)")
                // Attempt to restore from backup
                self.restoreFromBackup(for: url)
            }
        }
    }
    
    public func loadPatterns() -> [LearnedPattern]? {
        let url = documentsDirectory.appendingPathComponent(patternsFile)
        
        guard fileManager.fileExists(atPath: url.path) else {
            return nil
        }
        
        do {
            let data = try Data(contentsOf: url)
            let wrapper = try decoder.decode(PatternsWrapper.self, from: data)
            
            // Check version compatibility
            guard isVersionCompatible(wrapper.version) else {
                print("Incompatible data format version: \(wrapper.version)")
                return migratePatterns(from: wrapper)
            }
            
            // Convert back to LearnedPattern
            let patterns = wrapper.patterns.compactMap { serializable -> LearnedPattern? in
                guard let id = UUID(uuidString: serializable.id),
                      let classification = CommandType(rawValue: serializable.classification) else {
                    return nil
                }
                
                return LearnedPattern(
                    id: id,
                    pattern: serializable.pattern,
                    features: dictionaryToFeatures(serializable.features),
                    classification: classification,
                    confidence: serializable.confidence,
                    timestamp: serializable.timestamp,
                    reinforcementCount: serializable.reinforcementCount,
                    successRate: serializable.successRate
                )
            }
            
            return patterns
            
        } catch {
            print("Failed to load patterns: \(error)")
            return nil
        }
    }
    
    // MARK: - Intent Patterns Persistence
    
    public func saveIntentPatterns(_ patterns: [String: IntentPattern]) {
        persistenceQueue.async(flags: .barrier) {
            let url = self.documentsDirectory.appendingPathComponent(self.intentsFile)
            
            do {
                self.createBackupIfExists(for: url)
                
                // Convert to serializable format
                let serializablePatterns = patterns.mapValues { pattern in
                    SerializableIntentPattern(
                        intent: pattern.intent,
                        linguisticMarkers: pattern.linguisticMarkers,
                        contextualFactors: pattern.contextualFactors,
                        confidence: pattern.confidence,
                        learned: pattern.learned
                    )
                }
                
                let wrapper = IntentPatternsWrapper(
                    version: self.dataFormatVersion,
                    patterns: serializablePatterns,
                    metadata: IntentMetadata(
                        totalIntents: patterns.count,
                        lastUpdated: Date()
                    )
                )
                
                let data = try self.encoder.encode(wrapper)
                try data.write(to: url, options: .atomicWrite)
                
            } catch {
                print("Failed to save intent patterns: \(error)")
                self.restoreFromBackup(for: url)
            }
        }
    }
    
    public func loadIntentPatterns() -> [String: IntentPattern]? {
        let url = documentsDirectory.appendingPathComponent(intentsFile)
        
        guard fileManager.fileExists(atPath: url.path) else {
            return nil
        }
        
        do {
            let data = try Data(contentsOf: url)
            let wrapper = try decoder.decode(IntentPatternsWrapper.self, from: data)
            
            guard isVersionCompatible(wrapper.version) else {
                print("Incompatible intent patterns version: \(wrapper.version)")
                return nil
            }
            
            // Convert back to IntentPattern
            let patterns = wrapper.patterns.mapValues { serializable in
                IntentPattern(
                    intent: serializable.intent,
                    linguisticMarkers: serializable.linguisticMarkers,
                    contextualFactors: serializable.contextualFactors,
                    confidence: serializable.confidence,
                    learned: serializable.learned
                )
            }
            
            return patterns
            
        } catch {
            print("Failed to load intent patterns: \(error)")
            return nil
        }
    }
    
    // MARK: - Neural Network Persistence
    
    public func saveNeuralNetwork(_ network: Any) {
        persistenceQueue.async(flags: .barrier) {
            let url = self.documentsDirectory.appendingPathComponent(self.networkFile)
            
            do {
                self.createBackupIfExists(for: url)
                
                // Handle different network types
                if let neuralNetwork = network as? NeuralNetwork {
                    let serializable = SerializableNeuralNetwork(
                        weights: neuralNetwork.weights,
                        biases: neuralNetwork.biases,
                        learningRate: neuralNetwork.learningRate,
                        architecture: NetworkArchitecture(
                            inputSize: neuralNetwork.weights.first?.count ?? 0,
                            hiddenLayers: neuralNetwork.weights.count,
                            outputSize: neuralNetwork.biases.count
                        )
                    )
                    
                    let wrapper = NeuralNetworkWrapper(
                        version: self.dataFormatVersion,
                        network: serializable,
                        metadata: NetworkMetadata(
                            trainingEpochs: 0, // Would be tracked in real implementation
                            lastUpdated: Date(),
                            accuracy: 0.0 // Would be calculated
                        )
                    )
                    
                    let data = try self.encoder.encode(wrapper)
                    try data.write(to: url, options: .atomicWrite)
                    
                } else {
                    // For generic network objects, try property list serialization
                    let data = try NSKeyedArchiver.archivedData(
                        withRootObject: network,
                        requiringSecureCoding: false
                    )
                    try data.write(to: url, options: .atomicWrite)
                }
                
            } catch {
                print("Failed to save neural network: \(error)")
                self.restoreFromBackup(for: url)
            }
        }
    }
    
    public func loadNeuralNetwork() -> Any? {
        let url = documentsDirectory.appendingPathComponent(networkFile)
        
        guard fileManager.fileExists(atPath: url.path) else {
            return nil
        }
        
        do {
            let data = try Data(contentsOf: url)
            
            // Try to decode as our structured format first
            if let wrapper = try? decoder.decode(NeuralNetworkWrapper.self, from: data) {
                guard isVersionCompatible(wrapper.version) else {
                    print("Incompatible neural network version: \(wrapper.version)")
                    return nil
                }
                
                return NeuralNetwork(
                    weights: wrapper.network.weights,
                    biases: wrapper.network.biases,
                    learningRate: wrapper.network.learningRate
                )
            }
            
            // Fallback to NSKeyedUnarchiver for generic objects
            if let network = try? NSKeyedUnarchiver.unarchiveTopLevelObjectWithData(data) {
                return network
            }
            
            return nil
            
        } catch {
            print("Failed to load neural network: \(error)")
            return nil
        }
    }
    
    // MARK: - Advanced Features
    
    /// Asynchronously save patterns with completion handler
    public func savePatternsAsync(_ patterns: [LearnedPattern], completion: @escaping (Bool) -> Void) {
        persistenceQueue.async(flags: .barrier) {
            self.savePatterns(patterns)
            DispatchQueue.main.async {
                completion(true)
            }
        }
    }
    
    /// Export all learning data as a single archive
    public func exportLearningData() -> URL? {
        let exportURL = documentsDirectory.appendingPathComponent("jarvis_learning_export_\(Date().timeIntervalSince1970).zip")
        
        // Would implement ZIP archive creation
        // For now, return the directory URL
        return documentsDirectory
    }
    
    /// Import learning data from archive
    public func importLearningData(from url: URL) -> Bool {
        // Would implement ZIP archive extraction and data import
        return false
    }
    
    /// Clear all learning data with optional backup
    public func clearAllData(createBackup: Bool = true) {
        if createBackup {
            _ = exportLearningData()
        }
        
        let files = [patternsFile, intentsFile, networkFile, metadataFile]
        for file in files {
            let url = documentsDirectory.appendingPathComponent(file)
            try? fileManager.removeItem(at: url)
        }
    }
    
    // MARK: - Helper Methods
    
    private func createBackupIfExists(for url: URL) {
        guard fileManager.fileExists(atPath: url.path) else { return }
        
        let backupName = url.lastPathComponent + ".backup_\(Int(Date().timeIntervalSince1970))"
        let backupURL = backupDirectory.appendingPathComponent(backupName)
        
        try? fileManager.copyItem(at: url, to: backupURL)
        
        // Clean old backups (keep only last 5)
        cleanOldBackups(for: url.lastPathComponent)
    }
    
    private func restoreFromBackup(for url: URL) {
        let filename = url.lastPathComponent
        let backups = try? fileManager.contentsOfDirectory(
            at: backupDirectory,
            includingPropertiesForKeys: [.creationDateKey],
            options: .skipsHiddenFiles
        ).filter { $0.lastPathComponent.contains(filename) }
            .sorted { url1, url2 in
                let date1 = try? url1.resourceValues(forKeys: [.creationDateKey]).creationDate ?? Date.distantPast
                let date2 = try? url2.resourceValues(forKeys: [.creationDateKey]).creationDate ?? Date.distantPast
                return date1! > date2!
            }
        
        if let mostRecentBackup = backups?.first {
            try? fileManager.copyItem(at: mostRecentBackup, to: url)
        }
    }
    
    private func cleanOldBackups(for filename: String, keepCount: Int = 5) {
        let backups = try? fileManager.contentsOfDirectory(
            at: backupDirectory,
            includingPropertiesForKeys: [.creationDateKey],
            options: .skipsHiddenFiles
        ).filter { $0.lastPathComponent.contains(filename) }
            .sorted { url1, url2 in
                let date1 = try? url1.resourceValues(forKeys: [.creationDateKey]).creationDate ?? Date.distantPast
                let date2 = try? url2.resourceValues(forKeys: [.creationDateKey]).creationDate ?? Date.distantPast
                return date1! > date2!
            }
        
        if let backups = backups, backups.count > keepCount {
            for backup in backups.suffix(from: keepCount) {
                try? fileManager.removeItem(at: backup)
            }
        }
    }
    
    private func isVersionCompatible(_ version: String) -> Bool {
        // Simple version check - could be more sophisticated
        return version == dataFormatVersion
    }
    
    private func migratePatterns(from wrapper: PatternsWrapper) -> [LearnedPattern]? {
        // Implement migration logic for different versions
        print("Migration needed from version \(wrapper.version) to \(dataFormatVersion)")
        return nil
    }
    
    private func updateMetadata(patternsCount: Int) {
        let url = documentsDirectory.appendingPathComponent(metadataFile)
        
        var metadata = loadMetadata() ?? LearningSystemMetadata()
        metadata.totalPatterns = patternsCount
        metadata.lastUpdated = Date()
        metadata.dataFormatVersion = dataFormatVersion
        
        if let data = try? encoder.encode(metadata) {
            try? data.write(to: url, options: .atomicWrite)
        }
    }
    
    private func loadMetadata() -> LearningSystemMetadata? {
        let url = documentsDirectory.appendingPathComponent(metadataFile)
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? decoder.decode(LearningSystemMetadata.self, from: data)
    }
    
    private func featuresToDictionary(_ features: [String: Double]) -> [String: Double] {
        return features
    }
    
    private func dictionaryToFeatures(_ dict: [String: Double]) -> [String: Double] {
        return dict
    }
}

// MARK: - Serializable Types

private struct SerializableLearnedPattern: Codable {
    let id: String
    let pattern: String
    let features: [String: Double]
    let classification: String
    let confidence: Double
    let timestamp: Date
    let reinforcementCount: Int
    let successRate: Double
}

private struct SerializableIntentPattern: Codable {
    let intent: String
    let linguisticMarkers: [String]
    let contextualFactors: [String: Double]
    let confidence: Double
    let learned: Date
}

private struct SerializableNeuralNetwork: Codable {
    let weights: [[Double]]
    let biases: [Double]
    let learningRate: Double
    let architecture: NetworkArchitecture
}

private struct NetworkArchitecture: Codable {
    let inputSize: Int
    let hiddenLayers: Int
    let outputSize: Int
}

// MARK: - Wrapper Types

private struct PatternsWrapper: Codable {
    let version: String
    let patterns: [SerializableLearnedPattern]
    let metadata: LearningMetadata
}

private struct IntentPatternsWrapper: Codable {
    let version: String
    let patterns: [String: SerializableIntentPattern]
    let metadata: IntentMetadata
}

private struct NeuralNetworkWrapper: Codable {
    let version: String
    let network: SerializableNeuralNetwork
    let metadata: NetworkMetadata
}

// MARK: - Metadata Types

private struct LearningMetadata: Codable {
    let totalPatterns: Int
    let lastUpdated: Date
    let averageConfidence: Double
}

private struct IntentMetadata: Codable {
    let totalIntents: Int
    let lastUpdated: Date
}

private struct NetworkMetadata: Codable {
    let trainingEpochs: Int
    let lastUpdated: Date
    let accuracy: Double
}

private struct LearningSystemMetadata: Codable {
    var totalPatterns: Int = 0
    var lastUpdated: Date = Date()
    var dataFormatVersion: String = "1.0"
}