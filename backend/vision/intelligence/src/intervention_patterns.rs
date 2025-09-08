// Rust performance layer for Intervention Decision Engine
// High-performance pattern recognition and signal processing

use std::collections::HashMap;
use std::f64;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSignal {
    pub signal_type: String,
    pub strength: f64,
    pub confidence: f64,
    pub timestamp: u64,
    pub metadata: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatePattern {
    pub pattern_id: String,
    pub weights: HashMap<String, f64>,
    pub threshold: f64,
    pub confidence: f64,
    pub sample_count: usize,
}

#[derive(Debug, Clone)]
pub struct PatternMatcher {
    patterns: HashMap<String, StatePattern>,
    signal_buffer: Vec<UserSignal>,
    max_buffer_size: usize,
}

impl PatternMatcher {
    pub fn new(max_buffer_size: usize) -> Self {
        Self {
            patterns: HashMap::new(),
            signal_buffer: Vec::new(),
            max_buffer_size,
        }
    }

    /// Add a user signal to the processing buffer
    pub fn add_signal(&mut self, signal: UserSignal) {
        self.signal_buffer.push(signal);
        
        // Maintain buffer size limit
        if self.signal_buffer.len() > self.max_buffer_size {
            self.signal_buffer.remove(0);
        }
    }

    /// Fast pattern matching using vectorized operations
    pub fn match_patterns(&self, signals: &[UserSignal]) -> Vec<(String, f64)> {
        let mut matches = Vec::new();
        
        // Convert signals to feature vector
        let features = self.extract_features(signals);
        
        // Match against all patterns
        for (pattern_id, pattern) in &self.patterns {
            let score = self.calculate_pattern_score(&features, pattern);
            if score > pattern.threshold {
                matches.push((pattern_id.clone(), score));
            }
        }
        
        // Sort by score descending
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        matches
    }

    /// Extract numerical features from signals
    fn extract_features(&self, signals: &[UserSignal]) -> HashMap<String, f64> {
        let mut features = HashMap::new();
        
        if signals.is_empty() {
            return features;
        }
        
        // Signal type aggregations
        let mut type_strengths: HashMap<String, Vec<f64>> = HashMap::new();
        let mut type_confidences: HashMap<String, Vec<f64>> = HashMap::new();
        
        for signal in signals {
            type_strengths.entry(signal.signal_type.clone())
                .or_insert_with(Vec::new)
                .push(signal.strength);
                
            type_confidences.entry(signal.signal_type.clone())
                .or_insert_with(Vec::new)
                .push(signal.confidence);
        }
        
        // Calculate aggregated features
        for (signal_type, strengths) in type_strengths {
            let avg_strength = strengths.iter().sum::<f64>() / strengths.len() as f64;
            let max_strength = strengths.iter().fold(0.0, |a, &b| a.max(b));
            let strength_variance = self.calculate_variance(&strengths);
            
            features.insert(format!("{}_avg_strength", signal_type), avg_strength);
            features.insert(format!("{}_max_strength", signal_type), max_strength);
            features.insert(format!("{}_strength_variance", signal_type), strength_variance);
        }
        
        for (signal_type, confidences) in type_confidences {
            let avg_confidence = confidences.iter().sum::<f64>() / confidences.len() as f64;
            features.insert(format!("{}_avg_confidence", signal_type), avg_confidence);
        }
        
        // Temporal features
        if signals.len() > 1 {
            let time_span = signals.last().unwrap().timestamp - signals.first().unwrap().timestamp;
            features.insert("time_span".to_string(), time_span as f64);
            features.insert("signal_frequency".to_string(), signals.len() as f64 / (time_span as f64 + 1.0));
        }
        
        // Cross-signal features
        self.extract_correlation_features(signals, &mut features);
        
        features
    }

    /// Extract correlation features between different signal types
    fn extract_correlation_features(&self, signals: &[UserSignal], features: &mut HashMap<String, f64>) {
        let signal_types: Vec<String> = signals.iter()
            .map(|s| s.signal_type.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        
        // Calculate co-occurrence patterns
        for i in 0..signal_types.len() {
            for j in (i + 1)..signal_types.len() {
                let type_a = &signal_types[i];
                let type_b = &signal_types[j];
                
                let co_occurrence = self.calculate_co_occurrence(signals, type_a, type_b);
                features.insert(
                    format!("co_occurrence_{}_{}", type_a, type_b),
                    co_occurrence
                );
            }
        }
    }

    /// Calculate co-occurrence score between two signal types
    fn calculate_co_occurrence(&self, signals: &[UserSignal], type_a: &str, type_b: &str) -> f64 {
        let window_size = 5; // seconds
        let mut co_occurrences = 0;
        let mut total_a = 0;
        
        for signal_a in signals.iter().filter(|s| s.signal_type == type_a) {
            total_a += 1;
            
            // Check for type_b signals within window
            let has_b_nearby = signals.iter()
                .filter(|s| s.signal_type == type_b)
                .any(|s| (s.timestamp as i64 - signal_a.timestamp as i64).abs() <= window_size);
            
            if has_b_nearby {
                co_occurrences += 1;
            }
        }
        
        if total_a > 0 {
            co_occurrences as f64 / total_a as f64
        } else {
            0.0
        }
    }

    /// Calculate pattern matching score
    fn calculate_pattern_score(&self, features: &HashMap<String, f64>, pattern: &StatePattern) -> f64 {
        let mut score = 0.0;
        let mut matched_features = 0;
        
        for (feature_name, &feature_value) in features {
            if let Some(&weight) = pattern.weights.get(feature_name) {
                score += weight * feature_value;
                matched_features += 1;
            }
        }
        
        // Normalize by number of matched features and pattern confidence
        if matched_features > 0 {
            score = (score / matched_features as f64) * pattern.confidence;
        }
        
        score.max(0.0).min(1.0)
    }

    /// Calculate variance of a vector
    fn calculate_variance(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        
        variance
    }

    /// Learn new patterns from signal sequences
    pub fn learn_pattern(&mut self, pattern_id: String, positive_signals: &[Vec<UserSignal>], 
                        negative_signals: &[Vec<UserSignal>]) -> Result<(), String> {
        
        if positive_signals.is_empty() {
            return Err("Need positive examples to learn pattern".to_string());
        }

        // Extract features from positive examples
        let mut positive_features: Vec<HashMap<String, f64>> = Vec::new();
        for signals in positive_signals {
            positive_features.push(self.extract_features(signals));
        }

        // Extract features from negative examples
        let mut negative_features: Vec<HashMap<String, f64>> = Vec::new();
        for signals in negative_signals {
            negative_features.push(self.extract_features(signals));
        }

        // Learn weights using simple perceptron-like algorithm
        let weights = self.learn_weights(&positive_features, &negative_features)?;
        
        // Calculate threshold
        let threshold = self.calculate_threshold(&positive_features, &weights);
        
        // Create pattern
        let pattern = StatePattern {
            pattern_id: pattern_id.clone(),
            weights,
            threshold,
            confidence: self.calculate_pattern_confidence(&positive_features, &negative_features),
            sample_count: positive_signals.len() + negative_signals.len(),
        };
        
        self.patterns.insert(pattern_id, pattern);
        Ok(())
    }

    /// Learn feature weights from positive and negative examples
    fn learn_weights(&self, positive_features: &[HashMap<String, f64>], 
                    negative_features: &[HashMap<String, f64>]) -> Result<HashMap<String, f64>, String> {
        
        // Collect all feature names
        let mut all_features: std::collections::HashSet<String> = std::collections::HashSet::new();
        for features in positive_features.iter().chain(negative_features.iter()) {
            for feature_name in features.keys() {
                all_features.insert(feature_name.clone());
            }
        }
        
        let mut weights: HashMap<String, f64> = HashMap::new();
        
        // Simple weight calculation: average difference between positive and negative
        for feature_name in all_features {
            let positive_avg = self.calculate_feature_average(&feature_name, positive_features);
            let negative_avg = self.calculate_feature_average(&feature_name, negative_features);
            
            let weight = positive_avg - negative_avg;
            weights.insert(feature_name, weight);
        }
        
        // Normalize weights
        let max_weight = weights.values().map(|w| w.abs()).fold(0.0, f64::max);
        if max_weight > 0.0 {
            for weight in weights.values_mut() {
                *weight /= max_weight;
            }
        }
        
        Ok(weights)
    }

    /// Calculate average value for a feature across examples
    fn calculate_feature_average(&self, feature_name: &str, 
                               feature_sets: &[HashMap<String, f64>]) -> f64 {
        let values: Vec<f64> = feature_sets.iter()
            .filter_map(|features| features.get(feature_name))
            .copied()
            .collect();
        
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    }

    /// Calculate threshold for pattern detection
    fn calculate_threshold(&self, positive_features: &[HashMap<String, f64>], 
                          weights: &HashMap<String, f64>) -> f64 {
        let mut scores: Vec<f64> = Vec::new();
        
        for features in positive_features {
            let score = self.calculate_pattern_score(features, &StatePattern {
                pattern_id: "temp".to_string(),
                weights: weights.clone(),
                threshold: 0.0,
                confidence: 1.0,
                sample_count: 0,
            });
            scores.push(score);
        }
        
        if scores.is_empty() {
            0.5
        } else {
            // Use 90th percentile of positive scores as threshold
            scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let index = ((scores.len() as f64) * 0.1).floor() as usize;
            scores[index.min(scores.len() - 1)]
        }
    }

    /// Calculate pattern confidence
    fn calculate_pattern_confidence(&self, positive_features: &[HashMap<String, f64>],
                                  negative_features: &[HashMap<String, f64>]) -> f64 {
        let total_samples = positive_features.len() + negative_features.len();
        
        if total_samples < 10 {
            return 0.5; // Low confidence with few samples
        }
        
        // Confidence based on separation between positive and negative examples
        let positive_ratio = positive_features.len() as f64 / total_samples as f64;
        let balance_factor = 1.0 - (positive_ratio - 0.5).abs() * 2.0; // Penalize imbalanced data
        
        let sample_factor = (total_samples as f64 / 100.0).min(1.0); // More samples = higher confidence
        
        (balance_factor * sample_factor).max(0.1).min(0.95)
    }

    /// Get pattern statistics
    pub fn get_pattern_stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        
        stats.insert("total_patterns".to_string(), 
                    serde_json::Value::Number(serde_json::Number::from(self.patterns.len())));
        
        stats.insert("buffer_size".to_string(), 
                    serde_json::Value::Number(serde_json::Number::from(self.signal_buffer.len())));
        
        let avg_confidence = if !self.patterns.is_empty() {
            self.patterns.values().map(|p| p.confidence).sum::<f64>() / self.patterns.len() as f64
        } else {
            0.0
        };
        
        stats.insert("average_confidence".to_string(), 
                    serde_json::Value::Number(serde_json::Number::from_f64(avg_confidence).unwrap()));
        
        stats
    }

    /// Clear old signals from buffer
    pub fn cleanup_buffer(&mut self, max_age_seconds: u64) {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        self.signal_buffer.retain(|signal| {
            current_time - signal.timestamp <= max_age_seconds
        });
    }

    /// Export patterns for persistence
    pub fn export_patterns(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.patterns)
    }

    /// Import patterns from persistence
    pub fn import_patterns(&mut self, patterns_json: &str) -> Result<(), Box<dyn std::error::Error>> {
        let patterns: HashMap<String, StatePattern> = serde_json::from_str(patterns_json)?;
        self.patterns = patterns;
        Ok(())
    }
}

/// Fast signal processing utilities
pub mod signal_processing {
    use super::UserSignal;
    
    /// Apply exponential smoothing to signal strengths
    pub fn smooth_signals(signals: &mut [UserSignal], alpha: f64) {
        if signals.len() < 2 {
            return;
        }
        
        // Sort by timestamp first
        signals.sort_by_key(|s| s.timestamp);
        
        let mut smoothed_strength = signals[0].strength;
        
        for signal in signals.iter_mut().skip(1) {
            smoothed_strength = alpha * signal.strength + (1.0 - alpha) * smoothed_strength;
            signal.strength = smoothed_strength;
        }
    }
    
    /// Detect anomalous signals using simple statistical methods
    pub fn detect_anomalies(signals: &[UserSignal], threshold_sigma: f64) -> Vec<usize> {
        if signals.len() < 10 {
            return Vec::new(); // Need sufficient data
        }
        
        // Calculate mean and standard deviation of strengths
        let strengths: Vec<f64> = signals.iter().map(|s| s.strength).collect();
        let mean = strengths.iter().sum::<f64>() / strengths.len() as f64;
        
        let variance = strengths.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / strengths.len() as f64;
        let std_dev = variance.sqrt();
        
        let threshold = threshold_sigma * std_dev;
        
        // Find anomalies
        let mut anomalies = Vec::new();
        for (i, signal) in signals.iter().enumerate() {
            if (signal.strength - mean).abs() > threshold {
                anomalies.push(i);
            }
        }
        
        anomalies
    }
    
    /// Calculate signal entropy (measure of randomness)
    pub fn calculate_entropy(signals: &[UserSignal]) -> f64 {
        if signals.is_empty() {
            return 0.0;
        }
        
        // Discretize strength values into bins
        let bins = 10;
        let mut histogram = vec![0; bins];
        
        for signal in signals {
            let bin_index = ((signal.strength * bins as f64).floor() as usize).min(bins - 1);
            histogram[bin_index] += 1;
        }
        
        // Calculate entropy
        let total = signals.len() as f64;
        let mut entropy = 0.0;
        
        for &count in &histogram {
            if count > 0 {
                let p = count as f64 / total;
                entropy -= p * p.log2();
            }
        }
        
        entropy
    }
}

/// High-performance intervention scoring
pub struct InterventionScorer {
    feature_weights: HashMap<String, f64>,
    normalization_factors: HashMap<String, (f64, f64)>, // (min, max) for each feature
}

impl InterventionScorer {
    pub fn new() -> Self {
        Self {
            feature_weights: HashMap::new(),
            normalization_factors: HashMap::new(),
        }
    }
    
    /// Score an intervention decision based on multiple factors
    pub fn score_intervention(&self, features: &HashMap<String, f64>) -> f64 {
        let mut total_score = 0.0;
        let mut total_weight = 0.0;
        
        for (feature_name, &feature_value) in features {
            if let Some(&weight) = self.feature_weights.get(feature_name) {
                let normalized_value = self.normalize_feature(feature_name, feature_value);
                total_score += weight * normalized_value;
                total_weight += weight.abs();
            }
        }
        
        if total_weight > 0.0 {
            (total_score / total_weight).max(0.0).min(1.0)
        } else {
            0.5 // Default score
        }
    }
    
    /// Normalize feature value using stored min/max
    fn normalize_feature(&self, feature_name: &str, value: f64) -> f64 {
        if let Some(&(min_val, max_val)) = self.normalization_factors.get(feature_name) {
            if max_val > min_val {
                (value - min_val) / (max_val - min_val)
            } else {
                0.5
            }
        } else {
            value.max(0.0).min(1.0) // Assume already normalized
        }
    }
    
    /// Update feature weights and normalization from training data
    pub fn update_from_examples(&mut self, positive_examples: &[HashMap<String, f64>],
                               negative_examples: &[HashMap<String, f64>]) {
        // Update normalization factors
        let all_examples: Vec<&HashMap<String, f64>> = positive_examples.iter()
            .chain(negative_examples.iter())
            .collect();
        
        self.update_normalization_factors(&all_examples);
        
        // Update feature weights
        self.update_feature_weights(positive_examples, negative_examples);
    }
    
    fn update_normalization_factors(&mut self, examples: &[&HashMap<String, f64>]) {
        let mut feature_values: HashMap<String, Vec<f64>> = HashMap::new();
        
        for example in examples {
            for (feature_name, &value) in example.iter() {
                feature_values.entry(feature_name.clone())
                    .or_insert_with(Vec::new)
                    .push(value);
            }
        }
        
        for (feature_name, values) in feature_values {
            if !values.is_empty() {
                let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                self.normalization_factors.insert(feature_name, (min_val, max_val));
            }
        }
    }
    
    fn update_feature_weights(&mut self, positive_examples: &[HashMap<String, f64>],
                             negative_examples: &[HashMap<String, f64>]) {
        // Simple weight learning: difference in means
        let mut all_features: std::collections::HashSet<String> = std::collections::HashSet::new();
        
        for example in positive_examples.iter().chain(negative_examples.iter()) {
            for feature_name in example.keys() {
                all_features.insert(feature_name.clone());
            }
        }
        
        for feature_name in all_features {
            let positive_avg = self.calculate_average_feature_value(&feature_name, positive_examples);
            let negative_avg = self.calculate_average_feature_value(&feature_name, negative_examples);
            
            let weight = positive_avg - negative_avg;
            self.feature_weights.insert(feature_name, weight);
        }
    }
    
    fn calculate_average_feature_value(&self, feature_name: &str, 
                                     examples: &[HashMap<String, f64>]) -> f64 {
        let values: Vec<f64> = examples.iter()
            .filter_map(|example| example.get(feature_name))
            .copied()
            .collect();
        
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_pattern_matcher_creation() {
        let matcher = PatternMatcher::new(1000);
        assert_eq!(matcher.signal_buffer.len(), 0);
        assert_eq!(matcher.patterns.len(), 0);
    }

    #[test]
    fn test_signal_addition() {
        let mut matcher = PatternMatcher::new(5);
        
        for i in 0..10 {
            let signal = UserSignal {
                signal_type: format!("test_signal_{}", i),
                strength: 0.5,
                confidence: 0.8,
                timestamp: i as u64,
                metadata: HashMap::new(),
            };
            matcher.add_signal(signal);
        }
        
        // Buffer should be limited to max size
        assert_eq!(matcher.signal_buffer.len(), 5);
    }

    #[test]
    fn test_feature_extraction() {
        let matcher = PatternMatcher::new(100);
        
        let signals = vec![
            UserSignal {
                signal_type: "frustration".to_string(),
                strength: 0.8,
                confidence: 0.9,
                timestamp: 1000,
                metadata: HashMap::new(),
            },
            UserSignal {
                signal_type: "frustration".to_string(),
                strength: 0.6,
                confidence: 0.8,
                timestamp: 1005,
                metadata: HashMap::new(),
            },
        ];
        
        let features = matcher.extract_features(&signals);
        
        assert!(features.contains_key("frustration_avg_strength"));
        assert!(features.contains_key("frustration_avg_confidence"));
        assert!((features["frustration_avg_strength"] - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_intervention_scorer() {
        let mut scorer = InterventionScorer::new();
        
        let mut features = HashMap::new();
        features.insert("severity".to_string(), 0.8);
        features.insert("confidence".to_string(), 0.9);
        
        // Initially should return default score
        let score = scorer.score_intervention(&features);
        assert!(score >= 0.0 && score <= 1.0);
        
        // Add some weights
        scorer.feature_weights.insert("severity".to_string(), 0.7);
        scorer.feature_weights.insert("confidence".to_string(), 0.5);
        
        let score = scorer.score_intervention(&features);
        assert!(score > 0.0);
    }

    #[test]
    fn test_signal_smoothing() {
        let mut signals = vec![
            UserSignal {
                signal_type: "test".to_string(),
                strength: 1.0,
                confidence: 0.8,
                timestamp: 1000,
                metadata: HashMap::new(),
            },
            UserSignal {
                signal_type: "test".to_string(),
                strength: 0.0,
                confidence: 0.8,
                timestamp: 1001,
                metadata: HashMap::new(),
            },
            UserSignal {
                signal_type: "test".to_string(),
                strength: 1.0,
                confidence: 0.8,
                timestamp: 1002,
                metadata: HashMap::new(),
            },
        ];
        
        signal_processing::smooth_signals(&mut signals, 0.5);
        
        // After smoothing, the middle signal should be between 0 and 1
        assert!(signals[1].strength > 0.0 && signals[1].strength < 1.0);
    }
}