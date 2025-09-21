//! Metal shaders for vision processing on macOS
//! Optimized for Apple Silicon M1

#include <metal_stdlib>
using namespace metal;

// Shared structures
struct FrameParams {
    uint width;
    uint height;
    uint channels;
    float threshold;
};

// Frame difference detection kernel
kernel void frame_difference(
    texture2d<float, access::read> current [[texture(0)]],
    texture2d<float, access::read> previous [[texture(1)]],
    texture2d<float, access::write> difference [[texture(2)]],
    constant FrameParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) return;
    
    float4 currentPixel = current.read(gid);
    float4 previousPixel = previous.read(gid);
    
    // Calculate per-channel difference
    float4 diff = abs(currentPixel - previousPixel);
    
    // Compute magnitude
    float magnitude = length(diff.rgb);
    
    // Apply threshold
    float result = magnitude > params.threshold ? 1.0 : 0.0;
    
    difference.write(float4(result, result, result, 1.0), gid);
}

// Motion detection with optical flow
kernel void motion_detection(
    texture2d<float, access::read> current [[texture(0)]],
    texture2d<float, access::read> previous [[texture(1)]],
    device float* motion_vectors [[buffer(0)]],
    constant FrameParams& params [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) return;
    
    // Simple block matching for motion estimation
    const int search_radius = 8;
    float min_sad = INFINITY;
    int2 best_motion = int2(0, 0);
    
    float4 current_block = current.read(gid);
    
    for (int dy = -search_radius; dy <= search_radius; dy++) {
        for (int dx = -search_radius; dx <= search_radius; dx++) {
            uint2 prev_pos = uint2(gid.x + dx, gid.y + dy);
            
            if (prev_pos.x < params.width && prev_pos.y < params.height) {
                float4 prev_block = previous.read(prev_pos);
                float sad = length(current_block - prev_block);
                
                if (sad < min_sad) {
                    min_sad = sad;
                    best_motion = int2(dx, dy);
                }
            }
        }
    }
    
    // Store motion vector
    uint idx = gid.y * params.width + gid.x;
    motion_vectors[idx * 2] = float(best_motion.x);
    motion_vectors[idx * 2 + 1] = float(best_motion.y);
}

// Edge detection using Sobel operator
kernel void edge_detection(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> edges [[texture(1)]],
    constant FrameParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) return;
    
    // Sobel operators
    const float3x3 sobel_x = float3x3(
        -1.0, 0.0, 1.0,
        -2.0, 0.0, 2.0,
        -1.0, 0.0, 1.0
    );
    
    const float3x3 sobel_y = float3x3(
        -1.0, -2.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  2.0,  1.0
    );
    
    // Sample 3x3 neighborhood
    float3x3 neighborhood;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            uint2 pos = uint2(gid.x + i, gid.y + j);
            if (pos.x < params.width && pos.y < params.height) {
                float4 pixel = input.read(pos);
                neighborhood[j+1][i+1] = dot(pixel.rgb, float3(0.299, 0.587, 0.114));
            }
        }
    }
    
    // Apply Sobel operators
    float gx = 0.0, gy = 0.0;
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
            gx += neighborhood[j][i] * sobel_x[j][i];
            gy += neighborhood[j][i] * sobel_y[j][i];
        }
    }
    
    // Calculate edge magnitude
    float magnitude = sqrt(gx * gx + gy * gy);
    
    edges.write(float4(magnitude, magnitude, magnitude, 1.0), gid);
}

// Color analysis and histogram generation
kernel void color_analysis(
    texture2d<float, access::read> input [[texture(0)]],
    device atomic_uint* histogram_r [[buffer(0)]],
    device atomic_uint* histogram_g [[buffer(1)]],
    device atomic_uint* histogram_b [[buffer(2)]],
    constant FrameParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) return;
    
    float4 pixel = input.read(gid);
    
    // Convert to 0-255 range and update histograms
    uint r = uint(pixel.r * 255.0);
    uint g = uint(pixel.g * 255.0);
    uint b = uint(pixel.b * 255.0);
    
    atomic_fetch_add_explicit(&histogram_r[r], 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&histogram_g[g], 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&histogram_b[b], 1, memory_order_relaxed);
}

// Feature extraction kernel
kernel void feature_extraction(
    device const float* input [[buffer(0)]],
    device float* features [[buffer(1)]],
    constant FrameParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]]
) {
    // Shared memory for block processing
    threadgroup float shared_data[32][32];
    
    uint idx = gid.y * params.width + gid.x;
    
    if (gid.x < params.width && gid.y < params.height) {
        // Load data into shared memory
        shared_data[tid.y][tid.x] = input[idx * params.channels];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid.x == 0 && tid.y == 0) {
        // Calculate block features
        float sum = 0.0;
        float sum_sq = 0.0;
        
        for (uint j = 0; j < tg_size.y; j++) {
            for (uint i = 0; i < tg_size.x; i++) {
                float val = shared_data[j][i];
                sum += val;
                sum_sq += val * val;
            }
        }
        
        uint block_size = tg_size.x * tg_size.y;
        float mean = sum / float(block_size);
        float variance = (sum_sq / float(block_size)) - (mean * mean);
        float std_dev = sqrt(max(variance, 0.0));
        
        // Store features
        uint block_idx = (gid.y / tg_size.y) * (params.width / tg_size.x) + (gid.x / tg_size.x);
        features[block_idx * 3] = mean;
        features[block_idx * 3 + 1] = std_dev;
        features[block_idx * 3 + 2] = variance;
    }
}

// Fast image hash for duplicate detection
kernel void image_hash(
    texture2d<float, access::read> input [[texture(0)]],
    device uint64_t* hash_output [[buffer(0)]],
    constant FrameParams& params [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    threadgroup uint64_t local_hash[256];
    
    if (tid.x < 256) {
        local_hash[tid.x] = 0;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Sample image at regular intervals
    uint step_x = params.width / 16;
    uint step_y = params.height / 16;
    
    if (gid.x % step_x == 0 && gid.y % step_y == 0) {
        float4 pixel = input.read(gid);
        
        // Simple hash combining
        uint64_t pixel_hash = uint64_t(pixel.r * 255) << 16 |
                              uint64_t(pixel.g * 255) << 8 |
                              uint64_t(pixel.b * 255);
        
        // Mix with position
        pixel_hash ^= (uint64_t(gid.x) << 32) | uint64_t(gid.y);
        
        // Accumulate in shared memory
        atomic_fetch_xor_explicit((threadgroup atomic_uint64_t*)&local_hash[tid.x % 256],
                                  pixel_hash, memory_order_relaxed);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce to single hash
    if (tid.x == 0) {
        uint64_t final_hash = 0;
        for (int i = 0; i < 256; i++) {
            final_hash ^= local_hash[i];
            final_hash = final_hash * 0x517cc1b727220a95 + 0x85ebca6b;
        }
        hash_output[0] = final_hash;
    }
}