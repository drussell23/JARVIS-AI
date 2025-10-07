/**
 * Advanced CoreML Voice Engine - C++ Implementation
 * Ultra-fast Voice Activity Detection + Speaker Recognition
 * Zero hardcoding - fully adaptive and dynamic
 *
 * Features:
 * - Hardware-accelerated inference on Apple Neural Engine
 * - Real-time voice activity detection (<10ms latency)
 * - Speaker recognition (learns YOUR voice)
 * - Adaptive thresholds based on performance
 * - Dynamic model updates
 */

#ifndef COREML_VOICE_ENGINE_HPP
#define COREML_VOICE_ENGINE_HPP

#include <CoreML/CoreML.h>
#include <AudioToolbox/AudioToolbox.h>
#include <Accelerate/Accelerate.h>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <chrono>
#include <atomic>
#include <mutex>

namespace jarvis {
namespace voice {

// ============================================================================
// ADAPTIVE CONFIGURATION - Zero Hardcoding
// ============================================================================

struct AdaptiveConfig {
    // Voice Activity Detection thresholds
    float vad_threshold = 0.5f;           // Adapts based on success rate
    float vad_threshold_min = 0.2f;
    float vad_threshold_max = 0.9f;

    // Speaker recognition thresholds
    float speaker_threshold = 0.7f;       // Adapts to your voice
    float speaker_threshold_min = 0.4f;
    float speaker_threshold_max = 0.95f;

    // Audio processing parameters
    int sample_rate = 16000;              // Dynamic based on input
    int frame_size = 512;                 // Adapts to latency requirements
    int hop_length = 160;                 // Dynamic overlap

    // Performance tracking
    std::vector<float> vad_success_history;
    std::vector<float> speaker_success_history;
    float success_rate = 1.0f;

    // Adaptive learning
    bool enable_adaptive = true;
    float learning_rate = 0.01f;          // How fast to adapt
    int adaptation_window = 100;          // Samples to use for adaptation
};

// ============================================================================
// PERFORMANCE METRICS
// ============================================================================

struct PerformanceMetrics {
    std::atomic<uint64_t> total_inferences{0};
    std::atomic<uint64_t> successful_detections{0};
    std::atomic<uint64_t> false_positives{0};
    std::atomic<uint64_t> false_negatives{0};

    // Latency tracking (nanoseconds)
    std::vector<int64_t> inference_latencies;
    std::vector<int64_t> preprocessing_latencies;
    std::vector<int64_t> total_latencies;

    // Confidence scores
    std::vector<float> vad_confidences;
    std::vector<float> speaker_confidences;

    // Get average latency in milliseconds
    double get_avg_inference_latency_ms() const {
        if (inference_latencies.empty()) return 0.0;
        int64_t sum = 0;
        for (auto lat : inference_latencies) sum += lat;
        return (sum / inference_latencies.size()) / 1e6;  // ns to ms
    }

    // Get success rate
    float get_success_rate() const {
        if (total_inferences == 0) return 1.0f;
        return static_cast<float>(successful_detections) / total_inferences;
    }
};

// ============================================================================
// AUDIO FEATURES - Dynamic Feature Extraction
// ============================================================================

struct AudioFeatures {
    std::vector<float> mfcc;              // Mel-frequency cepstral coefficients
    std::vector<float> mel_spectrogram;   // Mel spectrogram
    std::vector<float> chroma;            // Chroma features
    float energy;                         // Signal energy
    float zcr;                           // Zero crossing rate
    float spectral_centroid;             // Spectral centroid
    float spectral_rolloff;              // Spectral rolloff

    // Feature dimensions (dynamic)
    int n_mfcc = 40;
    int n_mels = 128;
    int n_chroma = 12;

    AudioFeatures() : energy(0.0f), zcr(0.0f),
                     spectral_centroid(0.0f), spectral_rolloff(0.0f) {}
};

// ============================================================================
// VOICE ENGINE - Main Class
// ============================================================================

class CoreMLVoiceEngine {
public:
    /**
     * Constructor - Initializes CoreML models and adaptive systems
     * @param vad_model_path Path to Voice Activity Detection model
     * @param speaker_model_path Path to Speaker Recognition model
     * @param config Initial adaptive configuration
     */
    CoreMLVoiceEngine(
        const std::string& vad_model_path,
        const std::string& speaker_model_path,
        const AdaptiveConfig& config = AdaptiveConfig()
    );

    ~CoreMLVoiceEngine();

    /**
     * Detect voice activity in audio buffer
     * @param audio_buffer Raw audio samples (float32)
     * @param buffer_size Number of samples
     * @param confidence Output confidence score (0-1)
     * @return true if voice detected, false otherwise
     */
    bool detect_voice_activity(
        const float* audio_buffer,
        size_t buffer_size,
        float& confidence
    );

    /**
     * Recognize if speaker is the trained user
     * @param audio_buffer Raw audio samples (float32)
     * @param buffer_size Number of samples
     * @param confidence Output confidence score (0-1)
     * @return true if recognized speaker, false otherwise
     */
    bool recognize_speaker(
        const float* audio_buffer,
        size_t buffer_size,
        float& confidence
    );

    /**
     * Combined detection - VAD + Speaker Recognition
     * This is the main method for fast voice detection
     * @param audio_buffer Raw audio samples (float32)
     * @param buffer_size Number of samples
     * @param vad_confidence Output VAD confidence
     * @param speaker_confidence Output speaker confidence
     * @return true if YOUR voice detected, false otherwise
     */
    bool detect_user_voice(
        const float* audio_buffer,
        size_t buffer_size,
        float& vad_confidence,
        float& speaker_confidence
    );

    /**
     * Train speaker model with new sample
     * Adaptive learning - continuously improves recognition
     * @param audio_buffer Audio sample of user's voice
     * @param buffer_size Number of samples
     * @param label True if this is positive sample (user), false if negative
     */
    void train_speaker_model(
        const float* audio_buffer,
        size_t buffer_size,
        bool label
    );

    /**
     * Update adaptive thresholds based on performance
     * Called automatically after each detection
     * @param success Whether the detection was successful
     * @param vad_conf VAD confidence from last detection
     * @param speaker_conf Speaker confidence from last detection
     */
    void update_adaptive_thresholds(
        bool success,
        float vad_conf,
        float speaker_conf
    );

    /**
     * Get current configuration
     */
    const AdaptiveConfig& get_config() const { return config_; }

    /**
     * Get performance metrics
     */
    const PerformanceMetrics& get_metrics() const { return metrics_; }

    /**
     * Reset adaptive learning (start fresh)
     */
    void reset_adaptation();

    /**
     * Save adapted model to disk for persistence
     * @param path Path to save model
     * @return true if successful
     */
    bool save_model(const std::string& path);

    /**
     * Load adapted model from disk
     * @param path Path to load model from
     * @return true if successful
     */
    bool load_model(const std::string& path);

private:
    // CoreML models (using Objective-C++ bridge)
    void* vad_model_;                    // MLModel* for VAD
    void* speaker_model_;                // MLModel* for speaker recognition

    // Adaptive configuration
    AdaptiveConfig config_;
    std::mutex config_mutex_;            // Thread-safe config updates

    // Performance tracking
    PerformanceMetrics metrics_;
    std::mutex metrics_mutex_;           // Thread-safe metrics

    // Feature extraction
    std::unique_ptr<class FeatureExtractor> feature_extractor_;

    // Audio preprocessing
    std::vector<float> audio_buffer_;    // Reusable buffer
    std::vector<float> preprocessed_;    // Preprocessed audio

    // Speaker embeddings (for adaptive learning)
    std::vector<std::vector<float>> speaker_embeddings_;
    std::vector<std::vector<float>> non_speaker_embeddings_;
    std::mutex embeddings_mutex_;

    // Private methods

    /**
     * Preprocess audio (normalization, filtering, etc.)
     */
    void preprocess_audio(
        const float* input,
        size_t input_size,
        std::vector<float>& output
    );

    /**
     * Extract audio features for ML
     */
    AudioFeatures extract_features(
        const float* audio,
        size_t size
    );

    /**
     * Run CoreML inference with features
     */
    bool run_inference(
        void* model,
        const AudioFeatures& features,
        float& confidence
    );

    /**
     * Run CoreML inference with raw audio (for Silero VAD)
     */
    bool run_inference_raw(
        void* model,
        const float* audio_data,
        size_t audio_size,
        float& confidence
    );

    /**
     * Adapt thresholds using gradient descent
     */
    void gradient_descent_adaptation(
        bool success,
        float vad_conf,
        float speaker_conf
    );

    /**
     * Calculate similarity between embeddings
     */
    float calculate_similarity(
        const std::vector<float>& emb1,
        const std::vector<float>& emb2
    );

    /**
     * Update rolling statistics
     */
    void update_statistics(
        std::vector<float>& history,
        float value,
        size_t max_size
    );
};

// ============================================================================
// FEATURE EXTRACTOR - Advanced Audio Processing
// ============================================================================

class FeatureExtractor {
public:
    FeatureExtractor(int sample_rate, int n_fft = 2048, int hop_length = 512);
    ~FeatureExtractor();

    /**
     * Extract MFCC features using Accelerate framework
     */
    std::vector<float> extract_mfcc(
        const float* audio,
        size_t size,
        int n_mfcc = 40
    );

    /**
     * Extract Mel spectrogram
     */
    std::vector<float> extract_mel_spectrogram(
        const float* audio,
        size_t size,
        int n_mels = 128
    );

    /**
     * Calculate zero crossing rate
     */
    float calculate_zcr(const float* audio, size_t size);

    /**
     * Calculate energy
     */
    float calculate_energy(const float* audio, size_t size);

    /**
     * Calculate spectral features using vDSP
     */
    void calculate_spectral_features(
        const float* audio,
        size_t size,
        float& centroid,
        float& rolloff
    );

private:
    int sample_rate_;
    int n_fft_;
    int hop_length_;

    // FFT setup (using Accelerate framework)
    FFTSetup fft_setup_;
    DSPSplitComplex fft_buffer_;
    std::vector<float> window_;
    std::vector<float> mel_filterbank_;

    // Private helpers
    void initialize_mel_filterbank(int n_mels);
    void apply_window(const float* input, float* output, size_t size);
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Load CoreML model from file
 * @param path Path to .mlmodel or .mlmodelc file
 * @return MLModel* pointer (void* for C++ compatibility)
 */
void* load_coreml_model(const std::string& path);

/**
 * Check if CoreML model uses Neural Engine
 * @param model MLModel* pointer
 * @return true if using Neural Engine, false if CPU/GPU
 */
bool is_using_neural_engine(void* model);

/**
 * Get CoreML model compute units
 * @param model MLModel* pointer
 * @return "NeuralEngine", "CPU", or "GPU"
 */
std::string get_compute_unit(void* model);

} // namespace voice
} // namespace jarvis

#endif // COREML_VOICE_ENGINE_HPP
