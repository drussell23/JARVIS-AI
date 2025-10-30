# WebSocket Health Validation - Quick Reference Card

## 🚀 Quick Start

### Run Basic Test
```bash
gh workflow run websocket-health-validation.yml
```

### Run Stress Test
```bash
gh workflow run websocket-health-validation.yml \
  -f connection_count=50 \
  -f stress_test=true
```

### Run Chaos Test
```bash
gh workflow run websocket-health-validation.yml \
  -f chaos_mode=true
```

---

## 📋 Test Suites

| Suite | Icon | What It Tests | Duration |
|-------|------|---------------|----------|
| Connection Lifecycle | 🔌 | Connection setup/teardown | ~30s |
| Self-Healing | 🔄 | Auto-reconnection | ~60s |
| Message Delivery | 📨 | Reliability & ordering | ~45s |
| Heartbeat | 💓 | Ping/pong & health | ~60s |
| Concurrent | 🔗 | Multi-client handling | ~90s |
| Performance | ⚡ | Latency & throughput | ~60s |

**Total Test Time:** ~5-6 minutes

---

## ⚙️ Configuration

### Quick Config Edit
```bash
# Edit test configuration
nano .github/workflows/config/websocket-test-config.json
```

### Key Settings
```json
{
  "test_duration": 300,        // Test duration (seconds)
  "connection_count": 10,      // Concurrent connections
  "success_threshold": 0.95,   // Min 95% success rate
  "latency_threshold_ms": 1000 // Max 1000ms latency
}
```

---

## 📊 Success Thresholds

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| Success Rate | > 95% | 90-95% | < 90% |
| P95 Latency | < 500ms | 500-1000ms | > 1000ms |
| P99 Latency | < 1000ms | 1000-2000ms | > 2000ms |
| Reconnections | 0-2 | 3-5 | > 5 |
| Error Rate | < 5% | 5-10% | > 10% |

---

## 🔍 Health Status

| Status | Emoji | Score | Meaning |
|--------|-------|-------|---------|
| Excellent | 🟢 | 95-100 | Perfect health |
| Good | 🟢 | 85-95 | Minor issues |
| Fair | 🟡 | 70-85 | Needs attention |
| Degraded | 🟠 | 50-70 | Performance issues |
| Critical | 🔴 | < 50 | Immediate action |

---

## 🔔 Automatic Triggers

**On Code Changes:**
- `backend/api/**/*websocket*.py`
- `frontend/**/*websocket*`
- `tests/**/*websocket*`

**On Schedule:**
- Daily at 2 AM UTC

**On Pull Requests:**
- When WebSocket code changes

---

## 🐛 Troubleshooting

### Common Issues

**No WebSocket server found**
```
⚠️  Tests run in mock mode (expected in CI)
```

**Connection timeout**
```bash
# Check service health
curl http://localhost:8000/health
curl http://localhost:8001/health
```

**High latency**
```bash
# Check system resources
top
netstat -an | grep ESTABLISHED
```

**Reconnection failures**
```bash
# Check logs
tail -f backend/logs/*.log | grep websocket
```

---

## 📈 Viewing Results

### GitHub Actions UI
1. Go to Actions tab
2. Select "WebSocket Self-Healing Validation"
3. Click on latest run
4. View step summaries

### Generated Reports
- Test summaries in job logs
- Metrics in artifacts
- Status badges in reports

### Artifacts
- `websocket_health_badge.svg` - Visual status
- `websocket_health_status.json` - Metrics data
- `websocket_health_status.md` - Human-readable

---

## 🎯 Test Scenarios

### Scenario 1: Normal Operation
```bash
gh workflow run websocket-health-validation.yml
```
**Expected:** All tests pass, < 100ms latency

### Scenario 2: Load Testing
```bash
gh workflow run websocket-health-validation.yml \
  -f connection_count=20 \
  -f test_duration=600
```
**Expected:** All tests pass, < 200ms latency

### Scenario 3: Stress Testing
```bash
gh workflow run websocket-health-validation.yml \
  -f connection_count=50 \
  -f stress_test=true
```
**Expected:** 90%+ success rate, < 500ms latency

### Scenario 4: Chaos Testing
```bash
gh workflow run websocket-health-validation.yml \
  -f chaos_mode=true \
  -f connection_count=10
```
**Expected:** Self-healing works, reconnections < 5

---

## 📝 Adding Custom Tests

### 1. Add to Matrix
```yaml
matrix:
  test-suite:
    - name: my-test
      display: My Test
      icon: 🆕
```

### 2. Implement Test
```python
async def test_my_feature(self) -> TestReport:
    report = TestReport("My Feature")
    # Your test logic
    report.add_result(True, "Test passed")
    return report
```

### 3. Update Config
```json
{
  "test_suites": {
    "my-test": {
      "timeout": 60,
      "critical": true
    }
  }
}
```

---

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TEST_DURATION` | 300 | Test duration (s) |
| `CONNECTION_COUNT` | 10 | Concurrent connections |
| `CHAOS_MODE` | false | Enable chaos testing |
| `STRESS_TEST` | false | Enable stress testing |
| `BACKEND_PORT` | 8000 | Backend API port |
| `WS_ROUTER_PORT` | 8001 | WebSocket router port |
| `LATENCY_THRESHOLD_MS` | 1000 | Max latency threshold |
| `SUCCESS_THRESHOLD` | 0.95 | Min success rate |

---

## 📚 Key Files

| File | Purpose |
|------|---------|
| `websocket-health-validation.yml` | Main workflow |
| `websocket_health_test.py` | Test implementation |
| `websocket-test-config.json` | Configuration |
| `websocket_status_badge.py` | Badge generator |
| `WEBSOCKET_HEALTH_VALIDATION.md` | Full docs |

---

## 🎓 Best Practices

✅ **DO:**
- Run tests before deployment
- Monitor daily test results
- Update thresholds based on trends
- Review failed tests immediately
- Keep configuration in sync

❌ **DON'T:**
- Ignore warning status
- Skip chaos testing
- Deploy without tests passing
- Disable critical tests
- Hardcode values

---

## 🆘 Getting Help

**Check Logs:**
```bash
# View workflow logs
gh run view --log

# View backend logs
tail -f backend/logs/*.log
```

**Check Issues:**
```bash
# List WebSocket issues
gh issue list --label websocket
```

**Documentation:**
- Full docs: `WEBSOCKET_HEALTH_VALIDATION.md`
- Implementation: `IMPLEMENTATION_SUMMARY.md`
- This card: `WEBSOCKET_QUICK_REFERENCE.md`

---

## 📊 Metrics Cheat Sheet

### Good Metrics (Target)
```
✅ Success Rate: 98%
✅ Avg Latency: 45ms
✅ P95 Latency: 120ms
✅ P99 Latency: 250ms
✅ Reconnections: 0
✅ Errors: 0
```

### Warning Metrics (Review)
```
⚠️  Success Rate: 92%
⚠️  Avg Latency: 180ms
⚠️  P95 Latency: 650ms
⚠️  P99 Latency: 1200ms
⚠️  Reconnections: 3
⚠️  Errors: 5
```

### Critical Metrics (Action Required)
```
❌ Success Rate: 75%
❌ Avg Latency: 800ms
❌ P95 Latency: 2000ms
❌ P99 Latency: 5000ms
❌ Reconnections: 10
❌ Errors: 20
```

---

## 🔄 Quick Commands Reference

```bash
# Run basic test
gh workflow run websocket-health-validation.yml

# Run with custom duration
gh workflow run websocket-health-validation.yml -f test_duration=600

# Run stress test
gh workflow run websocket-health-validation.yml -f stress_test=true

# Run chaos test
gh workflow run websocket-health-validation.yml -f chaos_mode=true

# View latest run
gh run list --workflow=websocket-health-validation.yml --limit 1

# View run details
gh run view

# View run logs
gh run view --log

# Download artifacts
gh run download

# List issues
gh issue list --label websocket

# Create test issue
gh issue create --title "WebSocket Test" --label websocket
```

---

## 📞 Support

**Priority Levels:**
- 🔴 **P0 (Critical):** All tests failing - immediate action
- 🟠 **P1 (High):** Multiple tests failing - same day
- 🟡 **P2 (Medium):** Single test failing - 1-2 days
- 🟢 **P3 (Low):** Warning status - review when possible

**Contact:**
- Create GitHub issue with `websocket` label
- Check workflow run logs
- Review documentation

---

**Last Updated:** 2025-10-30
**Version:** 1.0.0
**Status:** ✅ Production Ready
