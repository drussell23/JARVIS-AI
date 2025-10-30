# WebSocket Self-Healing Validation System

## Overview

Comprehensive GitHub Actions workflow for validating WebSocket connection health, self-healing mechanisms, and real-time communication reliability.

## 🎯 Priority 5 Implementation

**Priority:** ⭐⭐⭐⭐ (High)

**Why Important:**
- Real-time communication must be reliable
- Prevents production outages
- Ensures user experience quality

**Impact Metrics:**
- 🎯 Prevents real-time communication failures
- ⏱️  Time Saved: 2-3 hrs/week
- 💰 ROI: Ensures system responsiveness

## Features

### ✨ Advanced Testing Capabilities

1. **Connection Lifecycle Testing** 🔌
   - Establishment validation
   - State management verification
   - Graceful shutdown testing
   - Connection pooling validation

2. **Self-Healing & Recovery** 🔄
   - Automatic reconnection testing
   - Circuit breaker validation
   - Recovery strategy verification
   - Resilience under failures

3. **Message Delivery & Reliability** 📨
   - Message delivery guarantees
   - Order preservation
   - Error handling validation
   - Timeout management

4. **Heartbeat & Health Monitoring** 💓
   - Ping/pong mechanism testing
   - Health score calculation
   - Latency monitoring
   - Connection quality metrics

5. **Concurrent Connections** 🔗
   - Multi-client testing
   - Load distribution
   - Resource management
   - Scalability validation

6. **Latency & Performance** ⚡
   - Response time measurement
   - Throughput testing
   - Performance benchmarking
   - SLA compliance

### 🚀 Dynamic & Configurable

**No Hardcoding:**
- Auto-detects service ports from config files
- Dynamically discovers WebSocket endpoints
- Loads test configuration from JSON
- Environment-specific settings

**Configuration-Driven:**
```json
{
  "test_duration": 300,
  "connection_count": 10,
  "chaos_mode": false,
  "stress_test": false,
  "thresholds": {
    "p95_latency_ms": 500,
    "success_rate": 0.95
  }
}
```

### 🔥 Advanced Features

**Chaos Engineering:**
- Random disconnections
- Network delays
- Message loss simulation
- Slow response testing

**Stress Testing:**
- High connection loads
- Message flooding
- Resource exhaustion scenarios
- Performance degradation detection

**Intelligent Monitoring:**
- Real-time health metrics
- Automatic issue creation on failure
- Trend analysis
- Predictive alerts

## Usage

### Automatic Triggers

**On Code Changes:**
```yaml
on:
  push:
    paths:
      - 'backend/api/**/*websocket*.py'
      - 'frontend/**/*websocket*'
```

**Scheduled Daily:**
```yaml
schedule:
  - cron: '0 2 * * *'  # 2 AM UTC daily
```

**On Pull Requests:**
```yaml
pull_request:
  types: [opened, synchronize, reopened]
```

### Manual Execution

**Basic Test:**
```bash
gh workflow run websocket-health-validation.yml
```

**Stress Test:**
```bash
gh workflow run websocket-health-validation.yml \
  -f test_duration=600 \
  -f connection_count=50 \
  -f stress_test=true
```

**Chaos Testing:**
```bash
gh workflow run websocket-health-validation.yml \
  -f chaos_mode=true \
  -f test_duration=300
```

## Test Suites

### 1. Connection Lifecycle (🔌)

**Tests:**
- WebSocket handshake
- Connection establishment
- Active connection validation
- Graceful disconnection
- Resource cleanup

**Success Criteria:**
- Connection established < 10s
- Clean handshake completion
- Proper state transitions
- No resource leaks

### 2. Self-Healing & Recovery (🔄)

**Tests:**
- Automatic reconnection
- Exponential backoff
- Circuit breaker activation
- Recovery from failures
- State preservation

**Success Criteria:**
- Reconnection < 10s
- Max 5 reconnection attempts
- State maintained after reconnection
- No data loss

### 3. Message Delivery (📨)

**Tests:**
- Text messages
- JSON payloads
- Binary data
- Large messages (>1MB)
- Message ordering

**Success Criteria:**
- 100% delivery rate
- Order preserved
- < 30s timeout
- Error handling works

### 4. Heartbeat Monitoring (💓)

**Tests:**
- Ping/pong cycles
- Latency measurement
- Health score calculation
- Connection quality

**Success Criteria:**
- Pong response < 5s
- Latency < 1000ms (p99)
- 95% success rate
- Health score > 80

### 5. Concurrent Connections (🔗)

**Tests:**
- Multiple clients
- Connection pooling
- Resource distribution
- Scalability limits

**Success Criteria:**
- Support 10+ connections
- Fair resource allocation
- No connection drops
- Stable performance

### 6. Latency & Performance (⚡)

**Tests:**
- Response time distribution
- Throughput measurement
- Load testing
- Performance regression

**Success Criteria:**
- P50 < 100ms
- P95 < 500ms
- P99 < 1000ms
- No degradation over time

## Configuration

### Environment Variables

```bash
# Test Configuration
TEST_DURATION=300          # Test duration in seconds
CONNECTION_COUNT=10        # Number of concurrent connections
CHAOS_MODE=false          # Enable chaos testing
STRESS_TEST=false         # Enable stress testing

# Service Ports (auto-detected if not set)
BACKEND_PORT=8000         # Backend API port
WS_ROUTER_PORT=8001       # WebSocket router port
FRONTEND_PORT=3000        # Frontend port

# Thresholds
LATENCY_THRESHOLD_MS=1000 # Max acceptable latency
SUCCESS_THRESHOLD=0.95    # Min success rate
```

### Configuration File

Edit `.github/workflows/config/websocket-test-config.json` for advanced settings:

```json
{
  "test_suites": {
    "heartbeat-monitoring": {
      "timeout": 120,
      "retry_attempts": 2,
      "ping_interval": 15,
      "pong_timeout": 5
    }
  },
  "thresholds": {
    "max_latency_p99_ms": 1000,
    "min_success_rate": 0.95
  }
}
```

## Monitoring & Alerts

### Automatic Issue Creation

When tests fail, an issue is automatically created:

```markdown
## WebSocket Health Validation Failure

**Branch:** main
**Commit:** abc123

### Action Required

WebSocket health tests have failed. This may indicate:
- Connection instability
- Self-healing mechanism issues
- Performance degradation
- Message delivery problems
```

### GitHub Step Summary

Comprehensive test report in GitHub Actions UI:

```
📊 WebSocket Self-Healing Validation Report
===========================================

✅ Connection Lifecycle: PASSED
✅ Self-Healing: PASSED
✅ Message Delivery: PASSED
✅ Heartbeat Monitoring: PASSED
⚠️  Concurrent Connections: DEGRADED
✅ Latency & Performance: PASSED

Key Metrics:
- Average Latency: 45ms
- P95 Latency: 250ms
- Success Rate: 98.5%
- Reconnections: 2
```

## Integration with Existing Tests

The workflow integrates with existing test suites:

```bash
# Run existing WebSocket tests
pytest tests/integration/test_*websocket*.py -v
```

Located at:
- `backend/tests/test_unified_websocket.py`
- `tests/integration/test_jarvis_websocket.py`
- `tests/integration/test_vision_websocket.py`

## Performance Metrics

### Tracked Metrics

**Connection Metrics:**
- Connection establishment time
- Active connection duration
- Reconnection frequency
- Connection success rate

**Message Metrics:**
- Messages sent/received
- Message delivery rate
- Message latency
- Error rate

**Health Metrics:**
- Health score (0-100)
- Latency (avg, p50, p95, p99)
- Heartbeat success rate
- Recovery time

### Example Output

```
📊 Connection Metrics:
  • conn_1:
    - Messages: 50 sent, 50 received
    - Reconnections: 0
    - Avg Latency: 45.23ms
    - Heartbeats: 10/10

📊 Latency Statistics:
  • Average: 45.23ms
  • P50: 42.15ms
  • P95: 89.34ms
  • P99: 125.67ms
```

## Troubleshooting

### Common Issues

**1. No WebSocket Server Found**
```
⚠️  No WebSocket server found, using mock testing
```
**Solution:** Tests will run in mock mode for CI environments without services.

**2. Connection Timeout**
```
❌ Connection timeout - server may be down
```
**Solution:** Check service health and port configuration.

**3. High Latency**
```
⚠️  Average latency (1250ms) exceeds threshold (1000ms)
```
**Solution:** Investigate network issues or server performance.

**4. Reconnection Failures**
```
❌ Reconnection attempts exceeded maximum (5)
```
**Solution:** Check self-healing configuration and server stability.

## Architecture

### Test Flow

```
┌─────────────────────────────────────────┐
│  Setup Environment                      │
│  - Detect Python version                │
│  - Discover service ports               │
│  - Generate test config                 │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Parallel Test Execution                │
│  ┌─────────────────────────────────┐   │
│  │ Connection Lifecycle            │   │
│  ├─────────────────────────────────┤   │
│  │ Self-Healing & Recovery         │   │
│  ├─────────────────────────────────┤   │
│  │ Message Delivery                │   │
│  ├─────────────────────────────────┤   │
│  │ Heartbeat Monitoring            │   │
│  ├─────────────────────────────────┤   │
│  │ Concurrent Connections          │   │
│  ├─────────────────────────────────┤   │
│  │ Latency & Performance           │   │
│  └─────────────────────────────────┘   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Generate Reports & Alerts              │
│  - Comprehensive summary                │
│  - Metrics dashboard                    │
│  - Issue creation (if needed)           │
└─────────────────────────────────────────┘
```

### Technology Stack

- **Python 3.10+** - Test runner
- **websockets** - WebSocket client library
- **asyncio** - Async operations
- **GitHub Actions** - CI/CD platform
- **pytest** - Test framework (optional)

## Best Practices

### 1. Run Tests Regularly

Schedule daily tests to catch degradation:
```yaml
schedule:
  - cron: '0 2 * * *'
```

### 2. Monitor Trends

Track metrics over time:
- Latency trends
- Success rate changes
- Resource usage patterns

### 3. Chaos Testing

Periodically run chaos tests:
```bash
gh workflow run websocket-health-validation.yml -f chaos_mode=true
```

### 4. Performance Baselines

Establish performance baselines:
- P95 latency targets
- Minimum success rates
- Maximum reconnection frequency

### 5. Alert Configuration

Configure appropriate alerts:
- Critical: Connection failures
- Warning: Performance degradation
- Info: Successful recovery

## Contributing

### Adding New Tests

1. Add test suite to matrix in workflow:
```yaml
matrix:
  test-suite:
    - name: new-test
      display: New Test
      icon: 🆕
```

2. Implement test in Python script:
```python
async def test_new_feature(self) -> TestReport:
    report = TestReport("New Feature")
    # Test implementation
    return report
```

3. Update configuration:
```json
{
  "test_suites": {
    "new-test": {
      "timeout": 60,
      "critical": true
    }
  }
}
```

## Resources

- [WebSocket RFC 6455](https://tools.ietf.org/html/rfc6455)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [websockets Library](https://websockets.readthedocs.io/)

## Support

For issues or questions:
1. Check workflow logs in GitHub Actions
2. Review test output in step summaries
3. Check automatically created issues
4. Review this documentation

## License

Part of JARVIS-AI-Agent project - see main LICENSE file.
