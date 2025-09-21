# Rust Self-Healing System Documentation

## Overview

The JARVIS Rust Self-Healing System automatically diagnoses and fixes issues preventing Rust components from loading, ensuring the system can recover from failures without manual intervention.

## Features

### 1. Automatic Diagnosis
- Detects various types of Rust component failures
- Identifies missing dependencies, build errors, and configuration issues
- Provides detailed diagnostics for troubleshooting

### 2. Self-Recovery Strategies
- **Automatic Build Recovery**: Rebuilds Rust components when missing
- **Dependency Resolution**: Installs missing crates automatically
- **Build Retry with Exponential Backoff**: Retries failed builds with increasing delays
- **Clean Build**: Removes artifacts and rebuilds from scratch when needed
- **Permission Fixes**: Corrects file permission issues
- **Memory Management**: Frees memory when builds fail due to resource constraints

### 3. Dynamic Component Switching
- Seamlessly switches between Rust and Python implementations
- Automatically upgrades to Rust when components become available
- Falls back to Python when Rust fails, ensuring continuous operation

## Architecture

### Components

1. **RustSelfHealer** (`vision/rust_self_healer.py`)
   - Core diagnosis and fixing logic
   - Manages fix strategies and retry logic
   - Tracks fix history and success rates

2. **DynamicComponentLoader** (`vision/dynamic_component_loader.py`)
   - Manages runtime switching between implementations
   - Periodically checks component availability
   - Integrates with self-healer for automatic recovery

3. **Unified Components** (`vision/unified_components.py`)
   - Provides consistent interfaces regardless of implementation
   - Handles state migration between implementations
   - Notifies callbacks on implementation changes

## API Endpoints

### `/self-healing/status`
Get the current status of the self-healing system.

```json
GET /self-healing/status

Response:
{
  "running": true,
  "total_fix_attempts": 5,
  "success_rate": 0.8,
  "last_successful_build": "2025-09-15T12:30:00",
  "retry_counts": {
    "not_built": 0,
    "build_failed": 1
  },
  "recent_fixes": [...]
}
```

### `/self-healing/diagnose`
Run a manual diagnosis of Rust component issues.

```json
POST /self-healing/diagnose

Response:
{
  "issue_type": "missing_dependencies",
  "details": {
    "missing_crates": ["pyo3", "numpy"]
  },
  "recommended_fix": "install_dependencies",
  "can_auto_fix": true
}
```

### `/self-healing/fix`
Attempt to fix Rust component issues.

```json
POST /self-healing/fix

Response:
{
  "success": true,
  "issue_type": "not_built",
  "strategy_used": "build",
  "error": null
}
```

### `/self-healing/force-check`
Force an immediate check of all components.

```json
POST /self-healing/force-check

Response:
{
  "success": true,
  "changes": {
    "bloom_filter": "upgraded_to_rust",
    "sliding_window": "upgraded_to_rust"
  },
  "message": "Found 2 component changes"
}
```

### `/self-healing/component-status`
Get the status of all dynamic components.

```json
GET /self-healing/component-status

Response:
{
  "loader_running": true,
  "check_interval_seconds": 60,
  "components": {
    "bloom_filter": {
      "active_implementation": "rust",
      "available_implementations": {
        "rust": {
          "available": true,
          "performance_score": 10.0,
          "error_count": 0
        },
        "python": {
          "available": true,
          "performance_score": 1.0,
          "error_count": 0
        }
      }
    }
  }
}
```

### `/self-healing/clean-build`
Clean and rebuild Rust components.

```json
POST /self-healing/clean-build

Response:
{
  "success": true,
  "message": "Clean build completed successfully"
}
```

## Configuration

### Environment Variables
- `RUST_CHECK_INTERVAL`: Seconds between automatic checks (default: 300)
- `RUST_MAX_RETRIES`: Maximum fix attempts before giving up (default: 3)
- `RUST_BUILD_TIMEOUT`: Build timeout in seconds (default: 600)

### Settings
The self-healer can be configured during initialization:

```python
healer = RustSelfHealer(
    check_interval=300,  # 5 minutes
    max_retries=3
)
```

## Issue Types and Fix Strategies

| Issue Type | Description | Fix Strategy |
|------------|-------------|--------------|
| `NOT_BUILT` | Rust components never built | Build components |
| `BUILD_FAILED` | Build process failed | Clean and rebuild |
| `MISSING_DEPENDENCIES` | Required crates missing | Install dependencies |
| `INCOMPATIBLE_VERSION` | Version mismatch | Rebuild components |
| `CORRUPTED_BINARY` | Binary file corrupted | Clean and rebuild |
| `MISSING_RUSTUP` | Rust not installed | Install Rust |
| `PERMISSION_ERROR` | File permission issues | Fix permissions |
| `OUT_OF_MEMORY` | Insufficient memory | Free memory and retry |

## Integration with Main Application

The self-healing system is automatically initialized during application startup:

```python
# In main.py lifespan handler
from vision.dynamic_component_loader import get_component_loader

loader = get_component_loader()
await loader.start()  # This also starts the self-healer
```

## Monitoring

The health check endpoint includes self-healing status:

```json
GET /health

Response:
{
  "status": "healthy",
  "self_healing": {
    "enabled": true,
    "fix_attempts": 10,
    "success_rate": 0.9,
    "last_successful_build": "2025-09-15T12:30:00"
  }
}
```

## Best Practices

1. **Let it run automatically**: The self-healer works best when allowed to run periodically
2. **Monitor success rates**: Low success rates may indicate persistent issues
3. **Check logs**: Detailed diagnostics are logged for troubleshooting
4. **Manual intervention**: Use the API endpoints for manual diagnosis when needed
5. **Resource management**: Ensure sufficient disk space and memory for builds

## Troubleshooting

### Common Issues

1. **Build timeouts**
   - Increase `RUST_BUILD_TIMEOUT` environment variable
   - Check system resources (CPU, memory, disk space)

2. **Repeated failures**
   - Check the build log at `vision/jarvis-rust-core/build.log`
   - Use `/self-healing/diagnose` to identify specific issues
   - Manual clean build may be needed: `/self-healing/clean-build`

3. **Permission errors**
   - Ensure the process has write permissions to the Rust directories
   - The self-healer will attempt to fix permissions automatically

4. **Missing dependencies**
   - The system maintains a list of known-good crate versions
   - Unknown crates will be installed with latest versions
   - Check `Cargo.toml` for version conflicts

## Performance Benefits

When Rust components are available:
- **Bloom Filter**: 10x faster duplicate detection
- **Sliding Window**: 5x faster frame analysis
- **Memory Pool**: 3x more efficient memory usage
- **Zero-Copy Operations**: Eliminates data copying overhead

## Future Enhancements

1. **Predictive Healing**: Detect issues before they cause failures
2. **Distributed Builds**: Use remote build servers for faster compilation
3. **Caching**: Cache successful builds for faster recovery
4. **Telemetry**: Send anonymous usage data to improve fix strategies