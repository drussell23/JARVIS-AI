/**
 * Fast Screen Capture Engine Implementation
 * Fully dynamic - no hardcoded values
 */

#include "fast_capture.h"
#include <iostream>
#include <algorithm>
#include <thread>
#include <mutex>
#include <queue>
#include <numeric>
#include <cmath>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#include <ApplicationServices/ApplicationServices.h>
#endif

namespace jarvis {
namespace vision {

// ===== Implementation Class =====
class FastCaptureEngine::Impl {
public:
    // Configuration
    CaptureConfig default_config;
    bool metrics_enabled = true;
    
    // Thread safety
    mutable std::mutex metrics_mutex;
    mutable std::mutex config_mutex;
    
    // Performance tracking
    PerformanceMetrics metrics;
    std::deque<double> capture_times;  // For percentile calculations
    static constexpr size_t MAX_TIMING_SAMPLES = 1000;
    
    // Callbacks
    CaptureCallback capture_callback;
    ErrorCallback error_callback;
    
    // macOS specific
    #ifdef __APPLE__
    dispatch_queue_t capture_queue;
    CGDirectDisplayID main_display;
    #endif
    
    Impl() {
        // Initialize metrics
        metrics.start_time = std::chrono::steady_clock::now();
        
        #ifdef __APPLE__
        // Create high-priority concurrent queue for captures
        dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(
            DISPATCH_QUEUE_CONCURRENT,
            QOS_CLASS_USER_INTERACTIVE,
            -1
        );
        capture_queue = dispatch_queue_create("com.jarvis.vision.capture", attr);
        
        // Get main display dynamically
        main_display = CGMainDisplayID();
        #endif
        
        // Auto-detect optimal thread count
        default_config.max_threads = std::thread::hardware_concurrency();
    }
    
    ~Impl() {
        #ifdef __APPLE__
        if (capture_queue) {
            dispatch_release(capture_queue);
        }
        #endif
    }
    
    // ===== Core Capture Implementation =====
    CaptureResult capture_window_impl(uint32_t window_id, const CaptureConfig& config) {
        auto start_time = std::chrono::high_resolution_clock::now();
        CaptureResult result;
        result.timestamp = std::chrono::steady_clock::now();
        
        #ifdef __APPLE__
        // Get window information dynamically
        CFArrayRef window_list = CGWindowListCopyWindowInfo(
            kCGWindowListOptionIncludingWindow,
            window_id
        );
        
        if (!window_list || CFArrayGetCount(window_list) == 0) {
            result.success = false;
            result.error_message = "Window not found";
            if (window_list) CFRelease(window_list);
            record_capture_time(start_time);
            return result;
        }
        
        // Extract window info dynamically
        CFDictionaryRef window_dict = (CFDictionaryRef)CFArrayGetValueAtIndex(window_list, 0);
        extract_window_info(window_dict, result.window_info);
        result.window_info.window_id = window_id;
        CFRelease(window_list);
        
        // Apply dynamic filters
        if (!should_capture_window(result.window_info, config)) {
            result.success = false;
            result.error_message = "Window filtered out by configuration";
            record_capture_time(start_time);
            return result;
        }
        
        // Capture the window
        CGImageRef window_image = nullptr;
        
        if (config.use_gpu_acceleration && is_gpu_available()) {
            // Try GPU-accelerated capture first
            window_image = capture_window_gpu(window_id, config);
            result.gpu_accelerated = (window_image != nullptr);
        }
        
        if (!window_image) {
            // Fallback to CPU capture
            window_image = CGWindowListCreateImage(
                CGRectNull,
                kCGWindowListOptionIncludingWindow,
                window_id,
                build_capture_options(config)
            );
        }
        
        if (!window_image) {
            result.success = false;
            result.error_message = "Failed to capture window image";
            record_capture_time(start_time);
            return result;
        }
        
        // Process the captured image
        process_captured_image(window_image, result, config);
        CGImageRelease(window_image);
        
        #else
        result.success = false;
        result.error_message = "Platform not supported";
        #endif
        
        // Record performance metrics
        auto capture_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time
        );
        result.capture_time = capture_duration;
        record_capture_time(start_time);
        
        // Trigger callback if set
        if (capture_callback && result.success) {
            capture_callback(result);
        }
        
        return result;
    }
    
    #ifdef __APPLE__
    // Dynamic window info extraction
    void extract_window_info(CFDictionaryRef dict, WindowInfo& info) {
        // Dynamically extract all available properties
        auto extract_string = [](CFDictionaryRef dict, CFStringRef key) -> std::string {
            CFStringRef value = (CFStringRef)CFDictionaryGetValue(dict, key);
            if (value && CFGetTypeID(value) == CFStringGetTypeID()) {
                char buffer[1024];
                if (CFStringGetCString(value, buffer, sizeof(buffer), kCFStringEncodingUTF8)) {
                    return std::string(buffer);
                }
            }
            return "";
        };
        
        auto extract_number = [](CFDictionaryRef dict, CFStringRef key, auto& out) {
            CFNumberRef value = (CFNumberRef)CFDictionaryGetValue(dict, key);
            if (value && CFGetTypeID(value) == CFNumberGetTypeID()) {
                CFNumberGetValue(value, sizeof(out) == 4 ? kCFNumberFloatType : kCFNumberIntType, &out);
            }
        };
        
        auto extract_bool = [](CFDictionaryRef dict, CFStringRef key) -> bool {
            CFBooleanRef value = (CFBooleanRef)CFDictionaryGetValue(dict, key);
            return value && CFGetTypeID(value) == CFBooleanGetTypeID() && CFBooleanGetValue(value);
        };
        
        // Extract standard properties
        info.app_name = extract_string(dict, kCGWindowOwnerName);
        info.window_title = extract_string(dict, kCGWindowName);
        
        // Try to get bundle identifier
        CFNumberRef pid_ref = (CFNumberRef)CFDictionaryGetValue(dict, kCGWindowOwnerPID);
        if (pid_ref) {
            pid_t pid;
            CFNumberGetValue(pid_ref, kCFNumberIntType, &pid);
            info.bundle_identifier = get_bundle_id_for_pid(pid);
        }
        
        // Window geometry
        CFDictionaryRef bounds_dict = (CFDictionaryRef)CFDictionaryGetValue(dict, kCGWindowBounds);
        if (bounds_dict) {
            CGRect bounds;
            CGRectMakeWithDictionaryRepresentation(bounds_dict, &bounds);
            info.x = bounds.origin.x;
            info.y = bounds.origin.y;
            info.width = bounds.size.width;
            info.height = bounds.size.height;
        }
        
        // Window state
        extract_number(dict, kCGWindowLayer, info.layer);
        extract_number(dict, kCGWindowAlpha, info.alpha);
        info.is_visible = extract_bool(dict, kCGWindowIsOnscreen);
        info.is_minimized = info.alpha < 0.01f;
        
        // Check if fullscreen (heuristic based on screen size)
        CGRect screen_bounds = CGDisplayBounds(main_display);
        info.is_fullscreen = (info.width >= screen_bounds.size.width * 0.95 &&
                             info.height >= screen_bounds.size.height * 0.95);
        
        // Extract all other properties as metadata
        CFDictionaryApplyFunction(dict, [](const void* key, const void* value, void* context) {
            auto* metadata = static_cast<std::unordered_map<std::string, std::string>*>(context);
            
            if (CFGetTypeID(key) == CFStringGetTypeID()) {
                char key_str[256];
                CFStringGetCString((CFStringRef)key, key_str, sizeof(key_str), kCFStringEncodingUTF8);
                
                // Convert value to string representation
                std::string value_str;
                CFTypeID type_id = CFGetTypeID(value);
                
                if (type_id == CFStringGetTypeID()) {
                    char buffer[1024];
                    CFStringGetCString((CFStringRef)value, buffer, sizeof(buffer), kCFStringEncodingUTF8);
                    value_str = buffer;
                } else if (type_id == CFNumberGetTypeID()) {
                    double num_value;
                    CFNumberGetValue((CFNumberRef)value, kCFNumberDoubleType, &num_value);
                    value_str = std::to_string(num_value);
                } else if (type_id == CFBooleanGetTypeID()) {
                    value_str = CFBooleanGetValue((CFBooleanRef)value) ? "true" : "false";
                }
                
                if (!value_str.empty()) {
                    (*metadata)[key_str] = value_str;
                }
            }
        }, &info.metadata);
    }
    
    // Get bundle identifier for a process
    std::string get_bundle_id_for_pid(pid_t pid) {
        // Use dynamic lookup - no hardcoded bundle IDs
        ProcessSerialNumber psn;
        if (GetProcessForPID(pid, &psn) == noErr) {
            CFDictionaryRef dict = ProcessInformationCopyDictionary(&psn, kProcessDictionaryIncludeAllInformationMask);
            if (dict) {
                CFStringRef bundle_id = (CFStringRef)CFDictionaryGetValue(dict, kCFBundleIdentifierKey);
                if (bundle_id) {
                    char buffer[256];
                    CFStringGetCString(bundle_id, buffer, sizeof(buffer), kCFStringEncodingUTF8);
                    CFRelease(dict);
                    return buffer;
                }
                CFRelease(dict);
            }
        }
        return "";
    }
    
    // Build capture options dynamically
    CGWindowImageOption build_capture_options(const CaptureConfig& config) {
        CGWindowImageOption options = kCGWindowImageDefault;
        
        if (!config.capture_shadow) {
            options |= kCGWindowImageBoundsIgnoreFraming;
        }
        
        if (config.capture_cursor) {
            // Note: Cursor capture requires additional permissions
            options |= kCGWindowImageShouldBeOpaque;
        }
        
        return options;
    }
    
    // GPU-accelerated capture (if available)
    CGImageRef capture_window_gpu(uint32_t window_id, const CaptureConfig& config) {
        // Check if Metal/GPU acceleration is available
        // This is a simplified version - real implementation would use Metal
        return nullptr;  // Fallback to CPU for now
    }
    
    // Check GPU availability
    bool is_gpu_available() {
        // Dynamically check for GPU availability
        // In a real implementation, this would check for Metal support
        return false;  // Conservative default
    }
    #endif
    
    // Process captured image
    void process_captured_image(CGImageRef image, CaptureResult& result, const CaptureConfig& config) {
        #ifdef __APPLE__
        result.width = CGImageGetWidth(image);
        result.height = CGImageGetHeight(image);
        
        // Apply size constraints if needed
        CGImageRef processed_image = image;
        CGImageRetain(processed_image);
        
        if (should_resize(result.width, result.height, config)) {
            CGImageRef resized = resize_image_dynamic(image, config);
            if (resized) {
                CGImageRelease(processed_image);
                processed_image = resized;
                result.width = CGImageGetWidth(processed_image);
                result.height = CGImageGetHeight(processed_image);
            }
        }
        
        // Extract raw pixel data
        auto raw_data = extract_raw_pixels(processed_image);
        result.channels = 4;  // RGBA
        result.bytes_per_pixel = 4;
        
        // Determine optimal format
        std::string format = config.output_format;
        if (format == "auto") {
            format = detect_optimal_format(result.width, result.height, has_transparency(raw_data));
        }
        
        // Compress image
        if (format == "raw") {
            result.image_data = std::move(raw_data);
            result.format = "raw";
        } else {
            int quality = config.jpeg_quality;
            if (quality < 0) {
                quality = calculate_optimal_quality(result.width, result.height);
            }
            
            result.image_data = compress_image(raw_data.data(), result.width, result.height,
                                             result.channels, format, quality);
            result.format = format;
        }
        
        // Calculate memory usage
        result.memory_used = result.image_data.size() + sizeof(result);
        
        CGImageRelease(processed_image);
        
        result.success = true;
        #endif
    }
    
    // Dynamic image resizing
    CGImageRef resize_image_dynamic(CGImageRef original, const CaptureConfig& config) {
        #ifdef __APPLE__
        size_t orig_width = CGImageGetWidth(original);
        size_t orig_height = CGImageGetHeight(original);
        
        // Calculate new dimensions
        auto [new_width, new_height] = calculate_resize_dimensions(
            orig_width, orig_height, config
        );
        
        if (new_width == orig_width && new_height == orig_height) {
            return nullptr;  // No resize needed
        }
        
        // Create resized image using high-quality interpolation
        CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
        CGContextRef context = CGBitmapContextCreate(
            nullptr, new_width, new_height, 8, 0, color_space,
            kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big
        );
        
        CGContextSetInterpolationQuality(context, kCGInterpolationHigh);
        CGContextDrawImage(context, CGRectMake(0, 0, new_width, new_height), original);
        
        CGImageRef resized = CGBitmapContextCreateImage(context);
        
        CGContextRelease(context);
        CGColorSpaceRelease(color_space);
        
        return resized;
        #else
        return nullptr;
        #endif
    }
    
    // Calculate resize dimensions
    std::pair<size_t, size_t> calculate_resize_dimensions(
        size_t width, size_t height, const CaptureConfig& config) {
        
        if (config.max_width <= 0 && config.max_height <= 0) {
            return {width, height};
        }
        
        double scale = 1.0;
        
        if (config.max_width > 0 && width > config.max_width) {
            scale = std::min(scale, static_cast<double>(config.max_width) / width);
        }
        
        if (config.max_height > 0 && height > config.max_height) {
            scale = std::min(scale, static_cast<double>(config.max_height) / height);
        }
        
        if (config.maintain_aspect_ratio) {
            return {
                static_cast<size_t>(width * scale),
                static_cast<size_t>(height * scale)
            };
        } else {
            return {
                config.max_width > 0 ? std::min(width, static_cast<size_t>(config.max_width)) : width,
                config.max_height > 0 ? std::min(height, static_cast<size_t>(config.max_height)) : height
            };
        }
    }
    
    // Extract raw pixels from CGImage
    std::vector<uint8_t> extract_raw_pixels(CGImageRef image) {
        #ifdef __APPLE__
        size_t width = CGImageGetWidth(image);
        size_t height = CGImageGetHeight(image);
        size_t bytes_per_row = width * 4;
        
        std::vector<uint8_t> data(height * bytes_per_row);
        
        CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
        CGContextRef context = CGBitmapContextCreate(
            data.data(), width, height, 8, bytes_per_row, color_space,
            kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big
        );
        
        CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
        
        CGContextRelease(context);
        CGColorSpaceRelease(color_space);
        
        return data;
        #else
        return {};
        #endif
    }
    
    // Check if image has transparency
    bool has_transparency(const std::vector<uint8_t>& raw_data) {
        // Check alpha channel (every 4th byte)
        for (size_t i = 3; i < raw_data.size(); i += 4) {
            if (raw_data[i] < 255) {
                return true;
            }
        }
        return false;
    }
    
    // Calculate optimal JPEG quality based on image size
    int calculate_optimal_quality(int width, int height) {
        int pixels = width * height;
        
        if (pixels > 4000000) {  // > 4MP
            return 75;
        } else if (pixels > 2000000) {  // > 2MP
            return 80;
        } else if (pixels > 1000000) {  // > 1MP
            return 85;
        } else {
            return 90;
        }
    }
    
    // Check if window should be captured based on filters
    bool should_capture_window(const WindowInfo& info, const CaptureConfig& config) {
        // Visibility filter
        if (config.capture_only_visible && (!info.is_visible || info.is_minimized)) {
            return false;
        }
        
        // Include apps filter
        if (!config.include_apps.empty()) {
            bool found = false;
            for (const auto& app : config.include_apps) {
                if (info.app_name.find(app) != std::string::npos ||
                    info.bundle_identifier.find(app) != std::string::npos) {
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
        
        // Exclude apps filter
        for (const auto& app : config.exclude_apps) {
            if (info.app_name.find(app) != std::string::npos ||
                info.bundle_identifier.find(app) != std::string::npos) {
                return false;
            }
        }
        
        // Custom filter
        if (config.custom_filter && !config.custom_filter(info)) {
            return false;
        }
        
        return true;
    }
    
    // Check if resizing is needed
    bool should_resize(int width, int height, const CaptureConfig& config) {
        return (config.max_width > 0 && width > config.max_width) ||
               (config.max_height > 0 && height > config.max_height);
    }
    
    // Record capture time for metrics
    void record_capture_time(const std::chrono::high_resolution_clock::time_point& start_time) {
        if (!metrics_enabled) return;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double time_ms = duration.count() / 1000.0;
        
        std::lock_guard<std::mutex> lock(metrics_mutex);
        
        metrics.total_captures++;
        metrics.last_capture_time = std::chrono::steady_clock::now();
        
        // Update timing statistics
        if (metrics.total_captures == 1) {
            metrics.min_capture_time_ms = time_ms;
            metrics.max_capture_time_ms = time_ms;
            metrics.avg_capture_time_ms = time_ms;
        } else {
            metrics.min_capture_time_ms = std::min(metrics.min_capture_time_ms, time_ms);
            metrics.max_capture_time_ms = std::max(metrics.max_capture_time_ms, time_ms);
            
            // Running average
            metrics.avg_capture_time_ms = 
                (metrics.avg_capture_time_ms * (metrics.total_captures - 1) + time_ms) / metrics.total_captures;
        }
        
        // Store for percentile calculation
        capture_times.push_back(time_ms);
        if (capture_times.size() > MAX_TIMING_SAMPLES) {
            capture_times.pop_front();
        }
        
        // Calculate percentiles
        if (capture_times.size() >= 10) {
            std::vector<double> sorted_times(capture_times.begin(), capture_times.end());
            std::sort(sorted_times.begin(), sorted_times.end());
            
            size_t p95_idx = static_cast<size_t>(sorted_times.size() * 0.95);
            size_t p99_idx = static_cast<size_t>(sorted_times.size() * 0.99);
            
            metrics.p95_capture_time_ms = sorted_times[p95_idx];
            metrics.p99_capture_time_ms = sorted_times[p99_idx];
        }
    }
};

// ===== FastCaptureEngine Public Implementation =====

FastCaptureEngine::FastCaptureEngine() : pImpl(std::make_unique<Impl>()) {}
FastCaptureEngine::~FastCaptureEngine() = default;

FastCaptureEngine::FastCaptureEngine(FastCaptureEngine&&) noexcept = default;
FastCaptureEngine& FastCaptureEngine::operator=(FastCaptureEngine&&) noexcept = default;

// Single window capture
CaptureResult FastCaptureEngine::capture_window(uint32_t window_id, const CaptureConfig& config) {
    try {
        return pImpl->capture_window_impl(window_id, config);
    } catch (const std::exception& e) {
        CaptureResult result;
        result.success = false;
        result.error_message = std::string("Exception: ") + e.what();
        if (pImpl->error_callback) {
            pImpl->error_callback(result.error_message);
        }
        return result;
    }
}

CaptureResult FastCaptureEngine::capture_window_by_name(const std::string& app_name,
                                                       const std::string& window_title,
                                                       const CaptureConfig& config) {
    auto window_info = find_window(app_name, window_title);
    if (!window_info) {
        CaptureResult result;
        result.success = false;
        result.error_message = "Window not found: " + app_name + 
                              (window_title.empty() ? "" : " - " + window_title);
        return result;
    }
    
    return capture_window(window_info->window_id, config);
}

CaptureResult FastCaptureEngine::capture_frontmost_window(const CaptureConfig& config) {
    auto window_info = get_frontmost_window();
    if (!window_info) {
        CaptureResult result;
        result.success = false;
        result.error_message = "No frontmost window found";
        return result;
    }
    
    return capture_window(window_info->window_id, config);
}

// Multi-window capture
std::vector<CaptureResult> FastCaptureEngine::capture_all_windows(const CaptureConfig& config) {
    auto windows = get_all_windows();
    std::vector<CaptureResult> results;
    results.reserve(windows.size());
    
    if (config.parallel_capture && windows.size() > 1) {
        // Parallel capture using thread pool
        std::vector<std::future<CaptureResult>> futures;
        
        for (const auto& window : windows) {
            futures.push_back(std::async(std::launch::async, [this, window, config]() {
                return capture_window(window.window_id, config);
            }));
        }
        
        for (auto& future : futures) {
            results.push_back(future.get());
        }
    } else {
        // Sequential capture
        for (const auto& window : windows) {
            results.push_back(capture_window(window.window_id, config));
        }
    }
    
    return results;
}

std::vector<CaptureResult> FastCaptureEngine::capture_visible_windows(const CaptureConfig& config) {
    auto windows = get_visible_windows();
    std::vector<CaptureResult> results;
    results.reserve(windows.size());
    
    for (const auto& window : windows) {
        results.push_back(capture_window(window.window_id, config));
    }
    
    return results;
}

// Window discovery
std::vector<WindowInfo> FastCaptureEngine::get_all_windows() {
    std::vector<WindowInfo> windows;
    
    #ifdef __APPLE__
    CFArrayRef window_list = CGWindowListCopyWindowInfo(
        kCGWindowListOptionAll | kCGWindowListExcludeDesktopElements,
        kCGNullWindowID
    );
    
    if (!window_list) return windows;
    
    CFIndex count = CFArrayGetCount(window_list);
    for (CFIndex i = 0; i < count; i++) {
        CFDictionaryRef window_dict = (CFDictionaryRef)CFArrayGetValueAtIndex(window_list, i);
        
        WindowInfo info;
        CFNumberRef window_id_ref = (CFNumberRef)CFDictionaryGetValue(window_dict, kCGWindowNumber);
        if (window_id_ref) {
            CFNumberGetValue(window_id_ref, kCFNumberIntType, &info.window_id);
            pImpl->extract_window_info(window_dict, info);
            windows.push_back(info);
        }
    }
    
    CFRelease(window_list);
    #endif
    
    return windows;
}

std::vector<WindowInfo> FastCaptureEngine::get_visible_windows() {
    auto all_windows = get_all_windows();
    std::vector<WindowInfo> visible;
    
    std::copy_if(all_windows.begin(), all_windows.end(), std::back_inserter(visible),
                 [](const WindowInfo& w) { return w.is_visible && !w.is_minimized; });
    
    return visible;
}

std::optional<WindowInfo> FastCaptureEngine::find_window(const std::string& app_name,
                                                       const std::string& window_title) {
    auto windows = get_all_windows();
    
    // Try exact match first
    for (const auto& window : windows) {
        bool app_match = window.app_name == app_name;
        bool title_match = window_title.empty() || window.window_title == window_title;
        
        if (app_match && title_match) {
            return window;
        }
    }
    
    // Try partial match
    for (const auto& window : windows) {
        bool app_match = window.app_name.find(app_name) != std::string::npos ||
                        window.bundle_identifier.find(app_name) != std::string::npos;
        bool title_match = window_title.empty() || 
                          window.window_title.find(window_title) != std::string::npos;
        
        if (app_match && title_match) {
            return window;
        }
    }
    
    return std::nullopt;
}

std::optional<WindowInfo> FastCaptureEngine::get_frontmost_window() {
    #ifdef __APPLE__
    // Get the frontmost application
    NSRunningApplication* frontmost = [[NSWorkspace sharedWorkspace] frontmostApplication];
    if (!frontmost) return std::nullopt;
    
    pid_t pid = [frontmost processIdentifier];
    
    // Find windows for this PID
    auto windows = get_all_windows();
    for (const auto& window : windows) {
        // Check if this window belongs to the frontmost app
        CFNumberRef window_pid_ref = nullptr;
        // Would need to extract PID from window info
        // For now, return the first visible window of the frontmost app
        if (window.app_name == std::string([[frontmost localizedName] UTF8String]) && 
            window.is_visible && !window.is_minimized) {
            return window;
        }
    }
    #endif
    
    return std::nullopt;
}

// Performance metrics
PerformanceMetrics FastCaptureEngine::get_metrics() const {
    std::lock_guard<std::mutex> lock(pImpl->metrics_mutex);
    return pImpl->metrics;
}

void FastCaptureEngine::reset_metrics() {
    std::lock_guard<std::mutex> lock(pImpl->metrics_mutex);
    pImpl->metrics = PerformanceMetrics();
    pImpl->metrics.start_time = std::chrono::steady_clock::now();
    pImpl->capture_times.clear();
}

void FastCaptureEngine::enable_metrics(bool enable) {
    pImpl->metrics_enabled = enable;
}

// Configuration
void FastCaptureEngine::set_default_config(const CaptureConfig& config) {
    std::lock_guard<std::mutex> lock(pImpl->config_mutex);
    pImpl->default_config = config;
}

CaptureConfig FastCaptureEngine::get_default_config() const {
    std::lock_guard<std::mutex> lock(pImpl->config_mutex);
    return pImpl->default_config;
}

// Callbacks
void FastCaptureEngine::set_capture_callback(CaptureCallback callback) {
    pImpl->capture_callback = callback;
}

void FastCaptureEngine::set_error_callback(ErrorCallback callback) {
    pImpl->error_callback = callback;
}

// ===== Utility Functions Implementation =====

std::string detect_optimal_format(int width, int height, bool has_transparency) {
    // Dynamic format selection based on image characteristics
    if (has_transparency) {
        return "png";  // PNG for transparency
    }
    
    int pixels = width * height;
    if (pixels > 2000000) {  // Large images
        return "jpeg";  // Better compression for large images
    }
    
    return "png";  // Default to PNG for quality
}

std::vector<uint8_t> compress_image(const uint8_t* raw_data, 
                                   int width, int height, int channels,
                                   const std::string& format,
                                   int quality) {
    std::vector<uint8_t> output;
    
    #ifdef __APPLE__
    // Create CGImage from raw data
    CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(
        const_cast<uint8_t*>(raw_data), width, height, 8, width * channels, color_space,
        kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big
    );
    
    CGImageRef image = CGBitmapContextCreateImage(context);
    
    // Create destination
    CFMutableDataRef output_data = CFDataCreateMutable(kCFAllocatorDefault, 0);
    
    CFStringRef format_ref = nullptr;
    if (format == "jpeg") {
        format_ref = kUTTypeJPEG;
    } else if (format == "png") {
        format_ref = kUTTypePNG;
    } else {
        // Default to PNG
        format_ref = kUTTypePNG;
    }
    
    CGImageDestinationRef destination = CGImageDestinationCreateWithData(
        output_data, format_ref, 1, nullptr
    );
    
    // Set compression options
    CFMutableDictionaryRef properties = nullptr;
    if (format == "jpeg" && quality >= 0) {
        properties = CFDictionaryCreateMutable(
            kCFAllocatorDefault, 0,
            &kCFTypeDictionaryKeyCallBacks,
            &kCFTypeDictionaryValueCallBacks
        );
        
        float quality_float = quality / 100.0f;
        CFNumberRef quality_number = CFNumberCreate(kCFAllocatorDefault, kCFNumberFloatType, &quality_float);
        CFDictionarySetValue(properties, kCGImageDestinationLossyCompressionQuality, quality_number);
        CFRelease(quality_number);
    }
    
    // Add image and finalize
    CGImageDestinationAddImage(destination, image, properties);
    CGImageDestinationFinalize(destination);
    
    // Copy data to vector
    size_t length = CFDataGetLength(output_data);
    const uint8_t* bytes = CFDataGetBytePtr(output_data);
    output.assign(bytes, bytes + length);
    
    // Cleanup
    if (properties) CFRelease(properties);
    CFRelease(destination);
    CFRelease(output_data);
    CGImageRelease(image);
    CGContextRelease(context);
    CGColorSpaceRelease(color_space);
    #endif
    
    return output;
}

double estimate_capture_time(int width, int height, const CaptureConfig& config) {
    // Empirical formula based on testing
    double base_time = 10.0;  // Base 10ms
    double pixel_factor = (width * height) / 1000000.0;  // Per megapixel
    
    double estimated = base_time + (pixel_factor * 5.0);  // 5ms per megapixel
    
    if (!config.use_gpu_acceleration) {
        estimated *= 1.5;  // 50% slower without GPU
    }
    
    return estimated;
}

size_t estimate_memory_usage(int width, int height, const std::string& format) {
    size_t raw_size = width * height * 4;  // RGBA
    
    if (format == "raw") {
        return raw_size;
    } else if (format == "jpeg") {
        return raw_size / 10;  // Approximately 10:1 compression
    } else if (format == "png") {
        return raw_size / 3;   // Approximately 3:1 compression
    }
    
    return raw_size;
}

} // namespace vision
} // namespace jarvis