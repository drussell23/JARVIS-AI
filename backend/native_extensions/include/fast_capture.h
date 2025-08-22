/**
 * Fast Screen Capture Engine for JARVIS Vision System
 * High-performance C++ extension for real-time screen capture on macOS
 * 
 * Design Principles:
 * - Zero hardcoding: Everything is discovered dynamically
 * - Thread-safe: Can be used from multiple Python threads
 * - Memory efficient: Uses zero-copy where possible
 * - Extensible: Easy to add new capture methods
 */

#ifndef JARVIS_FAST_CAPTURE_H
#define JARVIS_FAST_CAPTURE_H

#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <optional>
#include <unordered_map>
#include <functional>
#include <future>

#ifdef __APPLE__
#include <CoreGraphics/CoreGraphics.h>
#include <CoreFoundation/CoreFoundation.h>
#include <ImageIO/ImageIO.h>
#endif

namespace jarvis {
namespace vision {

// Forward declarations
struct WindowInfo;
struct CaptureConfig;
struct CaptureResult;
struct PerformanceMetrics;
class FastCaptureEngine;

/**
 * Dynamic window information - no hardcoded values
 */
struct WindowInfo {
    uint32_t window_id = 0;
    std::string app_name;
    std::string window_title;
    std::string bundle_identifier;  // com.apple.Safari, etc.
    int x = 0, y = 0, width = 0, height = 0;
    bool is_visible = false;
    bool is_minimized = false;
    bool is_fullscreen = false;
    int layer = 0;  // Window layer/z-order
    float alpha = 1.0f;  // Window transparency
    std::unordered_map<std::string, std::string> metadata;  // Extensible metadata
};

/**
 * Dynamic capture configuration
 */
struct CaptureConfig {
    // Capture options
    bool capture_cursor = false;
    bool capture_shadow = false;
    bool capture_only_visible = true;
    
    // Output format - dynamically determined if not specified
    std::string output_format = "auto";  // "auto", "jpeg", "png", "raw"
    int jpeg_quality = 85;  // 0-100
    
    // Performance options
    bool use_gpu_acceleration = true;
    bool parallel_capture = true;
    int max_threads = 0;  // 0 = auto-detect
    
    // Size constraints (0 = no constraint)
    int max_width = 0;
    int max_height = 0;
    bool maintain_aspect_ratio = true;
    
    // Filtering
    std::vector<std::string> include_apps;  // Empty = all apps
    std::vector<std::string> exclude_apps;
    std::function<bool(const WindowInfo&)> custom_filter;
    
    // Metadata options
    bool capture_metadata = true;
    bool include_color_profile = false;
};

/**
 * Capture result with comprehensive information
 */
struct CaptureResult {
    bool success = false;
    std::vector<uint8_t> image_data;
    std::string format;  // Actual format used
    int width = 0;
    int height = 0;
    int channels = 0;
    int bytes_per_pixel = 0;
    
    WindowInfo window_info;
    std::chrono::milliseconds capture_time{0};
    std::chrono::steady_clock::time_point timestamp;
    
    std::string error_message;
    std::string warning_message;
    
    // Performance data
    size_t memory_used = 0;
    bool gpu_accelerated = false;
    
    // Metadata
    std::unordered_map<std::string, std::string> metadata;
};

/**
 * Real-time performance metrics
 */
struct PerformanceMetrics {
    // Timing statistics
    double avg_capture_time_ms = 0.0;
    double min_capture_time_ms = 0.0;
    double max_capture_time_ms = 0.0;
    double p95_capture_time_ms = 0.0;  // 95th percentile
    double p99_capture_time_ms = 0.0;  // 99th percentile
    
    // Capture statistics
    uint64_t total_captures = 0;
    uint64_t successful_captures = 0;
    uint64_t failed_captures = 0;
    uint64_t bytes_processed = 0;
    
    // Resource usage
    size_t peak_memory_usage = 0;
    double avg_cpu_usage = 0.0;
    int gpu_captures = 0;
    
    // Timing
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point last_capture_time;
    
    // Per-window stats
    std::unordered_map<std::string, uint64_t> captures_per_app;
    std::unordered_map<std::string, double> avg_time_per_app;
};

/**
 * Capture event callback for real-time monitoring
 */
using CaptureCallback = std::function<void(const CaptureResult&)>;
using ErrorCallback = std::function<void(const std::string&)>;

/**
 * Main capture engine class - fully dynamic
 */
class FastCaptureEngine {
public:
    FastCaptureEngine();
    ~FastCaptureEngine();
    
    // Prevent copying (engine holds resources)
    FastCaptureEngine(const FastCaptureEngine&) = delete;
    FastCaptureEngine& operator=(const FastCaptureEngine&) = delete;
    
    // Allow moving
    FastCaptureEngine(FastCaptureEngine&&) noexcept;
    FastCaptureEngine& operator=(FastCaptureEngine&&) noexcept;
    
    // ===== Single Window Capture =====
    CaptureResult capture_window(uint32_t window_id, const CaptureConfig& config = {});
    CaptureResult capture_window_by_name(const std::string& app_name, 
                                       const std::string& window_title = "",
                                       const CaptureConfig& config = {});
    CaptureResult capture_frontmost_window(const CaptureConfig& config = {});
    
    // ===== Multi-Window Capture =====
    std::vector<CaptureResult> capture_all_windows(const CaptureConfig& config = {});
    std::vector<CaptureResult> capture_visible_windows(const CaptureConfig& config = {});
    std::vector<CaptureResult> capture_windows_by_app(const std::string& app_name,
                                                     const CaptureConfig& config = {});
    
    // ===== Screen Capture =====
    CaptureResult capture_main_screen(const CaptureConfig& config = {});
    CaptureResult capture_all_screens(const CaptureConfig& config = {});
    CaptureResult capture_screen_region(int x, int y, int width, int height,
                                      const CaptureConfig& config = {});
    
    // ===== Window Discovery (Dynamic) =====
    std::vector<WindowInfo> get_all_windows();
    std::vector<WindowInfo> get_visible_windows();
    std::vector<WindowInfo> get_windows_by_app(const std::string& app_name);
    std::optional<WindowInfo> find_window(const std::string& app_name, 
                                        const std::string& window_title = "");
    std::optional<WindowInfo> get_frontmost_window();
    
    // ===== Dynamic App Discovery =====
    std::vector<std::string> get_running_apps();
    std::unordered_map<std::string, std::vector<WindowInfo>> get_apps_with_windows();
    
    // ===== Performance & Monitoring =====
    PerformanceMetrics get_metrics() const;
    void reset_metrics();
    void enable_metrics(bool enable);
    
    // ===== Configuration =====
    void set_default_config(const CaptureConfig& config);
    CaptureConfig get_default_config() const;
    
    // ===== Callbacks for real-time monitoring =====
    void set_capture_callback(CaptureCallback callback);
    void set_error_callback(ErrorCallback callback);
    
    // ===== Utility Functions =====
    bool is_retina_display() const;
    float get_display_scale_factor() const;
    std::vector<std::pair<int, int>> get_screen_resolutions() const;
    
    // ===== Async Support =====
    std::future<CaptureResult> capture_window_async(uint32_t window_id, 
                                                   const CaptureConfig& config = {});
    std::future<std::vector<CaptureResult>> capture_all_windows_async(
                                                   const CaptureConfig& config = {});
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// ===== Utility Functions =====

// Dynamic image format detection
std::string detect_optimal_format(int width, int height, bool has_transparency);

// Image compression with dynamic quality selection
std::vector<uint8_t> compress_image(const uint8_t* raw_data, 
                                   int width, int height, int channels,
                                   const std::string& format,
                                   int quality = -1);  // -1 = auto

// Performance utilities
double estimate_capture_time(int width, int height, const CaptureConfig& config);
size_t estimate_memory_usage(int width, int height, const std::string& format);

// Window filtering utilities
std::function<bool(const WindowInfo&)> create_app_filter(const std::vector<std::string>& apps);
std::function<bool(const WindowInfo&)> create_size_filter(int min_width, int min_height);
std::function<bool(const WindowInfo&)> create_visibility_filter();

} // namespace vision
} // namespace jarvis

#endif // JARVIS_FAST_CAPTURE_H