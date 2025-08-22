/**
 * Python bindings for Fast Capture Engine
 * Uses pybind11 for seamless Python integration
 */

#include <pybind11/pybind11.h> 
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/stl_bind.h>

#include "fast_capture.h"

namespace py = pybind11;
using namespace jarvis::vision;

// Convert raw image data to numpy array
py::array_t<uint8_t> image_data_to_numpy(const std::vector<uint8_t>& data, 
                                         int width, int height, int channels) {
    // Create numpy array with proper shape
    auto result = py::array_t<uint8_t>({height, width, channels});
    auto buf = result.request();
    uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);
    
    // Copy data
    std::memcpy(ptr, data.data(), data.size());
    
    return result;
}

// Convert CaptureResult to Python dict with numpy array
py::dict capture_result_to_dict(const CaptureResult& result) {
    py::dict d;
    
    d["success"] = result.success;
    d["width"] = result.width;
    d["height"] = result.height;
    d["channels"] = result.channels;
    d["format"] = result.format;
    d["capture_time_ms"] = result.capture_time.count();
    d["timestamp"] = result.timestamp;
    d["memory_used"] = result.memory_used;
    d["gpu_accelerated"] = result.gpu_accelerated;
    
    // Convert image data to numpy array if raw format
    if (result.format == "raw" && !result.image_data.empty()) {
        d["image"] = image_data_to_numpy(result.image_data, 
                                        result.width, result.height, result.channels);
    } else {
        // For compressed formats, return bytes
        d["image_data"] = py::bytes(reinterpret_cast<const char*>(result.image_data.data()), 
                                   result.image_data.size());
    }
    
    // Window info
    py::dict window_info;
    window_info["window_id"] = result.window_info.window_id;
    window_info["app_name"] = result.window_info.app_name;
    window_info["window_title"] = result.window_info.window_title;
    window_info["bundle_identifier"] = result.window_info.bundle_identifier;
    window_info["x"] = result.window_info.x;
    window_info["y"] = result.window_info.y;
    window_info["width"] = result.window_info.width;
    window_info["height"] = result.window_info.height;
    window_info["is_visible"] = result.window_info.is_visible;
    window_info["is_minimized"] = result.window_info.is_minimized;
    window_info["is_fullscreen"] = result.window_info.is_fullscreen;
    window_info["layer"] = result.window_info.layer;
    window_info["alpha"] = result.window_info.alpha;
    window_info["metadata"] = result.window_info.metadata;
    d["window_info"] = window_info;
    
    // Errors/warnings
    if (!result.error_message.empty()) {
        d["error"] = result.error_message;
    }
    if (!result.warning_message.empty()) {
        d["warning"] = result.warning_message;
    }
    
    // Metadata
    if (!result.metadata.empty()) {
        d["metadata"] = result.metadata;
    }
    
    return d;
}

// Module definition
PYBIND11_MODULE(fast_capture, m) {
    m.doc() = "Fast Screen Capture Engine for JARVIS Vision System";
    
    // ===== Enums and Constants =====
    m.attr("VERSION") = "1.0.0";
    
    // ===== WindowInfo =====
    py::class_<WindowInfo>(m, "WindowInfo")
        .def(py::init<>())
        .def_readwrite("window_id", &WindowInfo::window_id)
        .def_readwrite("app_name", &WindowInfo::app_name)
        .def_readwrite("window_title", &WindowInfo::window_title)
        .def_readwrite("bundle_identifier", &WindowInfo::bundle_identifier)
        .def_readwrite("x", &WindowInfo::x)
        .def_readwrite("y", &WindowInfo::y)
        .def_readwrite("width", &WindowInfo::width)
        .def_readwrite("height", &WindowInfo::height)
        .def_readwrite("is_visible", &WindowInfo::is_visible)
        .def_readwrite("is_minimized", &WindowInfo::is_minimized)
        .def_readwrite("is_fullscreen", &WindowInfo::is_fullscreen)
        .def_readwrite("layer", &WindowInfo::layer)
        .def_readwrite("alpha", &WindowInfo::alpha)
        .def_readwrite("metadata", &WindowInfo::metadata)
        .def("__repr__", [](const WindowInfo& w) {
            return "<WindowInfo: " + w.app_name + " - " + w.window_title + ">";
        });
    
    // ===== CaptureConfig =====
    py::class_<CaptureConfig>(m, "CaptureConfig")
        .def(py::init<>())
        .def_readwrite("capture_cursor", &CaptureConfig::capture_cursor)
        .def_readwrite("capture_shadow", &CaptureConfig::capture_shadow)
        .def_readwrite("capture_only_visible", &CaptureConfig::capture_only_visible)
        .def_readwrite("output_format", &CaptureConfig::output_format)
        .def_readwrite("jpeg_quality", &CaptureConfig::jpeg_quality)
        .def_readwrite("use_gpu_acceleration", &CaptureConfig::use_gpu_acceleration)
        .def_readwrite("parallel_capture", &CaptureConfig::parallel_capture)
        .def_readwrite("max_threads", &CaptureConfig::max_threads)
        .def_readwrite("max_width", &CaptureConfig::max_width)
        .def_readwrite("max_height", &CaptureConfig::max_height)
        .def_readwrite("maintain_aspect_ratio", &CaptureConfig::maintain_aspect_ratio)
        .def_readwrite("include_apps", &CaptureConfig::include_apps)
        .def_readwrite("exclude_apps", &CaptureConfig::exclude_apps)
        .def_readwrite("capture_metadata", &CaptureConfig::capture_metadata)
        .def_readwrite("include_color_profile", &CaptureConfig::include_color_profile)
        .def("set_custom_filter", [](CaptureConfig& config, py::function filter) {
            config.custom_filter = [filter](const WindowInfo& info) {
                return filter(info).cast<bool>();
            };
        });
    
    // ===== PerformanceMetrics =====
    py::class_<PerformanceMetrics>(m, "PerformanceMetrics")
        .def_readonly("avg_capture_time_ms", &PerformanceMetrics::avg_capture_time_ms)
        .def_readonly("min_capture_time_ms", &PerformanceMetrics::min_capture_time_ms)
        .def_readonly("max_capture_time_ms", &PerformanceMetrics::max_capture_time_ms)
        .def_readonly("p95_capture_time_ms", &PerformanceMetrics::p95_capture_time_ms)
        .def_readonly("p99_capture_time_ms", &PerformanceMetrics::p99_capture_time_ms)
        .def_readonly("total_captures", &PerformanceMetrics::total_captures)
        .def_readonly("successful_captures", &PerformanceMetrics::successful_captures)
        .def_readonly("failed_captures", &PerformanceMetrics::failed_captures)
        .def_readonly("bytes_processed", &PerformanceMetrics::bytes_processed)
        .def_readonly("peak_memory_usage", &PerformanceMetrics::peak_memory_usage)
        .def_readonly("avg_cpu_usage", &PerformanceMetrics::avg_cpu_usage)
        .def_readonly("gpu_captures", &PerformanceMetrics::gpu_captures)
        .def_readonly("start_time", &PerformanceMetrics::start_time)
        .def_readonly("last_capture_time", &PerformanceMetrics::last_capture_time)
        .def_readonly("captures_per_app", &PerformanceMetrics::captures_per_app)
        .def_readonly("avg_time_per_app", &PerformanceMetrics::avg_time_per_app)
        .def("__repr__", [](const PerformanceMetrics& m) {
            return "<PerformanceMetrics: avg=" + std::to_string(m.avg_capture_time_ms) + 
                   "ms, total=" + std::to_string(m.total_captures) + ">";
        });
    
    // ===== FastCaptureEngine =====
    py::class_<FastCaptureEngine>(m, "FastCaptureEngine")
        .def(py::init<>())
        
        // Single window capture
        .def("capture_window", [](FastCaptureEngine& self, uint32_t window_id, 
                                 const CaptureConfig& config) {
            return capture_result_to_dict(self.capture_window(window_id, config));
        }, py::arg("window_id"), py::arg("config") = CaptureConfig(),
           "Capture a single window by ID")
        
        .def("capture_window_by_name", [](FastCaptureEngine& self, 
                                         const std::string& app_name,
                                         const std::string& window_title,
                                         const CaptureConfig& config) {
            return capture_result_to_dict(
                self.capture_window_by_name(app_name, window_title, config)
            );
        }, py::arg("app_name"), py::arg("window_title") = "", 
           py::arg("config") = CaptureConfig(),
           "Capture a window by app name and optional window title")
        
        .def("capture_frontmost_window", [](FastCaptureEngine& self, 
                                           const CaptureConfig& config) {
            return capture_result_to_dict(self.capture_frontmost_window(config));
        }, py::arg("config") = CaptureConfig(),
           "Capture the frontmost window")
        
        // Multi-window capture
        .def("capture_all_windows", [](FastCaptureEngine& self, 
                                      const CaptureConfig& config) {
            auto results = self.capture_all_windows(config);
            py::list py_results;
            for (const auto& result : results) {
                py_results.append(capture_result_to_dict(result));
            }
            return py_results;
        }, py::arg("config") = CaptureConfig(),
           "Capture all windows")
        
        .def("capture_visible_windows", [](FastCaptureEngine& self, 
                                          const CaptureConfig& config) {
            auto results = self.capture_visible_windows(config);
            py::list py_results;
            for (const auto& result : results) {
                py_results.append(capture_result_to_dict(result));
            }
            return py_results;
        }, py::arg("config") = CaptureConfig(),
           "Capture only visible windows")
        
        .def("capture_windows_by_app", [](FastCaptureEngine& self, 
                                         const std::string& app_name,
                                         const CaptureConfig& config) {
            auto results = self.capture_windows_by_app(app_name, config);
            py::list py_results;
            for (const auto& result : results) {
                py_results.append(capture_result_to_dict(result));
            }
            return py_results;
        }, py::arg("app_name"), py::arg("config") = CaptureConfig(),
           "Capture all windows from a specific application")
        
        // Window discovery
        .def("get_all_windows", &FastCaptureEngine::get_all_windows,
             "Get information about all windows")
        .def("get_visible_windows", &FastCaptureEngine::get_visible_windows,
             "Get information about visible windows only")
        .def("get_windows_by_app", &FastCaptureEngine::get_windows_by_app,
             py::arg("app_name"),
             "Get windows for a specific application")
        .def("find_window", [](FastCaptureEngine& self, 
                              const std::string& app_name,
                              const std::string& window_title) {
            auto result = self.find_window(app_name, window_title);
            if (result) {
                return py::cast(*result);
            }
            return py::none();
        }, py::arg("app_name"), py::arg("window_title") = "",
           "Find a specific window")
        .def("get_frontmost_window", [](FastCaptureEngine& self) {
            auto result = self.get_frontmost_window();
            if (result) {
                return py::cast(*result);
            }
            return py::none();
        }, "Get the frontmost window")
        
        // Performance metrics
        .def("get_metrics", &FastCaptureEngine::get_metrics,
             "Get performance metrics")
        .def("reset_metrics", &FastCaptureEngine::reset_metrics,
             "Reset performance metrics")
        .def("enable_metrics", &FastCaptureEngine::enable_metrics,
             py::arg("enable"),
             "Enable or disable metrics collection")
        
        // Configuration
        .def("set_default_config", &FastCaptureEngine::set_default_config,
             py::arg("config"),
             "Set default capture configuration")
        .def("get_default_config", &FastCaptureEngine::get_default_config,
             "Get default capture configuration")
        
        // Callbacks
        .def("set_capture_callback", [](FastCaptureEngine& self, py::function callback) {
            self.set_capture_callback([callback](const CaptureResult& result) {
                py::gil_scoped_acquire acquire;
                callback(capture_result_to_dict(result));
            });
        }, py::arg("callback"),
           "Set callback for capture events")
        
        .def("set_error_callback", [](FastCaptureEngine& self, py::function callback) {
            self.set_error_callback([callback](const std::string& error) {
                py::gil_scoped_acquire acquire;
                callback(error);
            });
        }, py::arg("callback"),
           "Set callback for error events");
    
    // ===== Utility Functions =====
    m.def("detect_optimal_format", &detect_optimal_format,
          py::arg("width"), py::arg("height"), py::arg("has_transparency"),
          "Detect optimal image format based on characteristics");
    
    m.def("estimate_capture_time", &estimate_capture_time,
          py::arg("width"), py::arg("height"), py::arg("config"),
          "Estimate capture time for given dimensions");
    
    m.def("estimate_memory_usage", &estimate_memory_usage,
          py::arg("width"), py::arg("height"), py::arg("format"),
          "Estimate memory usage for given dimensions and format");
    
    // ===== Helper functions for Python =====
    m.def("create_size_filter", [](int min_width, int min_height) {
        return py::cpp_function([min_width, min_height](const WindowInfo& info) {
            return info.width >= min_width && info.height >= min_height;
        });
    }, py::arg("min_width"), py::arg("min_height"),
       "Create a filter for minimum window size");
    
    m.def("create_app_filter", [](const std::vector<std::string>& apps) {
        return py::cpp_function([apps](const WindowInfo& info) {
            for (const auto& app : apps) {
                if (info.app_name.find(app) != std::string::npos ||
                    info.bundle_identifier.find(app) != std::string::npos) {
                    return true;
                }
            }
            return false;
        });
    }, py::arg("apps"),
       "Create a filter for specific applications");
}