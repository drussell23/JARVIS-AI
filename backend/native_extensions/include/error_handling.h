/**
 * Error Handling and Memory Management for Fast Capture Engine
 * Provides RAII wrappers and exception-safe operations
 */

#ifndef JARVIS_ERROR_HANDLING_H
#define JARVIS_ERROR_HANDLING_H

#include <string>
#include <exception>
#include <memory>
#include <functional>
#include <chrono>
#include <sstream>

#ifdef __APPLE__
#include <CoreGraphics/CoreGraphics.h>
#include <CoreFoundation/CoreFoundation.h>
#endif

namespace jarvis {
namespace vision {

/**
 * Custom exception types for better error handling
 */
class CaptureException : public std::exception {
public:
    explicit CaptureException(const std::string& message) 
        : message_("Capture Error: " + message) {}
    
    const char* what() const noexcept override {
        return message_.c_str();
    }
    
private:
    std::string message_;
};

class WindowNotFoundException : public CaptureException {
public:
    explicit WindowNotFoundException(uint32_t window_id)
        : CaptureException("Window " + std::to_string(window_id) + " not found") {}
    
    explicit WindowNotFoundException(const std::string& app_name)
        : CaptureException("Window for app '" + app_name + "' not found") {}
};

class PermissionException : public CaptureException {
public:
    PermissionException()
        : CaptureException("Screen recording permission required. Please enable in System Preferences > Security & Privacy > Privacy > Screen Recording") {}
};

class GPUException : public CaptureException {
public:
    explicit GPUException(const std::string& detail)
        : CaptureException("GPU acceleration failed: " + detail) {}
};

/**
 * RAII wrapper for CGImageRef
 */
class CGImageHolder {
public:
    CGImageHolder() : image_(nullptr) {}
    explicit CGImageHolder(CGImageRef image) : image_(image) {}
    
    ~CGImageHolder() {
        if (image_) {
            CGImageRelease(image_);
        }
    }
    
    // Move constructor
    CGImageHolder(CGImageHolder&& other) noexcept : image_(other.image_) {
        other.image_ = nullptr;
    }
    
    // Move assignment
    CGImageHolder& operator=(CGImageHolder&& other) noexcept {
        if (this != &other) {
            if (image_) {
                CGImageRelease(image_);
            }
            image_ = other.image_;
            other.image_ = nullptr;
        }
        return *this;
    }
    
    // Delete copy operations
    CGImageHolder(const CGImageHolder&) = delete;
    CGImageHolder& operator=(const CGImageHolder&) = delete;
    
    CGImageRef get() const { return image_; }
    CGImageRef release() {
        CGImageRef temp = image_;
        image_ = nullptr;
        return temp;
    }
    
    void reset(CGImageRef image = nullptr) {
        if (image_) {
            CGImageRelease(image_);
        }
        image_ = image;
    }
    
    explicit operator bool() const { return image_ != nullptr; }
    
private:
    CGImageRef image_;
};

/**
 * RAII wrapper for CFTypeRef objects
 */
template<typename T>
class CFHolder {
public:
    CFHolder() : ref_(nullptr) {}
    explicit CFHolder(T ref) : ref_(ref) {}
    
    ~CFHolder() {
        if (ref_) {
            CFRelease(ref_);
        }
    }
    
    // Move constructor
    CFHolder(CFHolder&& other) noexcept : ref_(other.ref_) {
        other.ref_ = nullptr;
    }
    
    // Move assignment
    CFHolder& operator=(CFHolder&& other) noexcept {
        if (this != &other) {
            if (ref_) {
                CFRelease(ref_);
            }
            ref_ = other.ref_;
            other.ref_ = nullptr;
        }
        return *this;
    }
    
    // Delete copy operations
    CFHolder(const CFHolder&) = delete;
    CFHolder& operator=(const CFHolder&) = delete;
    
    T get() const { return ref_; }
    T release() {
        T temp = ref_;
        ref_ = nullptr;
        return temp;
    }
    
    void reset(T ref = nullptr) {
        if (ref_) {
            CFRelease(ref_);
        }
        ref_ = ref;
    }
    
    explicit operator bool() const { return ref_ != nullptr; }
    
private:
    T ref_;
};

/**
 * Scoped timer for performance measurement
 */
class ScopedTimer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::milliseconds;
    using Callback = std::function<void(Duration)>;
    
    explicit ScopedTimer(Callback callback) 
        : start_(Clock::now()), callback_(callback) {}
    
    ~ScopedTimer() {
        auto duration = std::chrono::duration_cast<Duration>(Clock::now() - start_);
        if (callback_) {
            callback_(duration);
        }
    }
    
private:
    TimePoint start_;
    Callback callback_;
};

/**
 * Memory pool for efficient allocation
 */
template<typename T>
class MemoryPool {
public:
    explicit MemoryPool(size_t block_size = 1024)
        : block_size_(block_size) {
        allocate_block();
    }
    
    ~MemoryPool() {
        for (auto& block : blocks_) {
            ::operator delete(block);
        }
    }
    
    T* allocate() {
        if (free_list_.empty()) {
            allocate_block();
        }
        
        T* ptr = free_list_.back();
        free_list_.pop_back();
        return ptr;
    }
    
    void deallocate(T* ptr) {
        if (ptr) {
            ptr->~T();
            free_list_.push_back(ptr);
        }
    }
    
private:
    void allocate_block() {
        T* block = static_cast<T*>(::operator new(block_size_ * sizeof(T)));
        blocks_.push_back(block);
        
        for (size_t i = 0; i < block_size_; ++i) {
            free_list_.push_back(&block[i]);
        }
    }
    
    size_t block_size_;
    std::vector<T*> blocks_;
    std::vector<T*> free_list_;
};

/**
 * Thread-safe error collector
 */
class ErrorCollector {
public:
    void add_error(const std::string& error) {
        std::lock_guard<std::mutex> lock(mutex_);
        errors_.push_back({
            std::chrono::steady_clock::now(),
            error
        });
        
        // Keep only last 100 errors
        if (errors_.size() > 100) {
            errors_.erase(errors_.begin());
        }
    }
    
    std::vector<std::string> get_recent_errors(size_t count = 10) const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::string> recent;
        
        size_t start = errors_.size() > count ? errors_.size() - count : 0;
        for (size_t i = start; i < errors_.size(); ++i) {
            recent.push_back(errors_[i].second);
        }
        
        return recent;
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        errors_.clear();
    }
    
private:
    mutable std::mutex mutex_;
    std::vector<std::pair<std::chrono::steady_clock::time_point, std::string>> errors_;
};

/**
 * Utility functions for safe operations
 */
inline bool check_screen_recording_permission() {
#ifdef __APPLE__
    // Check if we can create a screen image
    CGImageRef test_image = CGDisplayCreateImage(CGMainDisplayID());
    if (test_image) {
        CGImageRelease(test_image);
        return true;
    }
    return false;
#else
    return true;
#endif
}

template<typename Func>
auto with_error_handling(Func&& func, const std::string& operation) 
    -> decltype(func()) {
    try {
        return func();
    } catch (const std::exception& e) {
        throw CaptureException(operation + " failed: " + e.what());
    } catch (...) {
        throw CaptureException(operation + " failed: unknown error");
    }
}

/**
 * Safe string conversion from CFString
 */
inline std::string cf_string_to_std_string(CFStringRef cf_str) {
    if (!cf_str) return "";
    
    CFIndex length = CFStringGetLength(cf_str);
    CFIndex max_size = CFStringGetMaximumSizeForEncoding(length, kCFStringEncodingUTF8) + 1;
    
    std::vector<char> buffer(max_size);
    if (CFStringGetCString(cf_str, buffer.data(), max_size, kCFStringEncodingUTF8)) {
        return std::string(buffer.data());
    }
    
    return "";
}

} // namespace vision
} // namespace jarvis

#endif // JARVIS_ERROR_HANDLING_H