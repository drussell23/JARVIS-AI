# Edge Cases & Testing

Comprehensive test scenarios, edge cases, and testing strategies for JARVIS AI Agent.

---

## Table of Contents

1. [Testing Framework](#testing-framework)
2. [Edge Cases](#edge-cases)
3. [Test Coverage](#test-coverage)
4. [Testing Strategies](#testing-strategies)
5. [CI/CD Testing](#cicd-testing)

---

## Testing Framework

### Test Suite Organization

```
tests/
├── unit/              # Unit tests (fast, isolated)
│   ├── intelligence/  # UAE, SAI, CAI tests
│   ├── voice/         # Voice system tests
│   ├── vision/        # Vision API tests
│   └── database/      # Database tests
├── integration/       # Integration tests
│   ├── hybrid/        # Hybrid orchestrator
│   ├── api/           # API endpoints
│   └── websocket/     # WebSocket tests
├── e2e/              # End-to-end tests
│   ├── voice_unlock/  # Voice unlock flow
│   └── vision_flow/   # Vision analysis flow
└── performance/       # Performance tests
    ├── benchmarks/    # Performance benchmarks
    └── load/          # Load testing
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific suite
pytest tests/unit/ -v
pytest tests/integration/ -v

# With coverage
pytest --cov=backend tests/

# Parallel execution
pytest -n auto tests/

# Specific test
pytest tests/unit/voice/test_speaker_recognition.py::test_voice_auth -v
```

---

## Edge Cases

### Voice System Edge Cases

#### 1. Background Noise

**Scenario:** High background noise interferes with STT

**Test:**
```python
def test_stt_with_background_noise():
    """Test STT accuracy with varying noise levels."""
    noise_levels = [-10, -20, -30, -40]  # dB

    for noise_db in noise_levels:
        audio = add_background_noise(clean_audio, noise_db)
        result = stt_engine.transcribe(audio)

        # Accuracy should degrade gracefully
        if noise_db > -30:
            assert result.confidence > 0.7
        else:
            assert result.confidence > 0.5
```

**Solution:**
- Audio preprocessing (bandpass filter)
- SNR estimation
- Confidence thresholds
- User feedback for low confidence

#### 2. Similar Speakers

**Scenario:** Multiple speakers with similar voices

**Test:**
```python
def test_speaker_discrimination():
    """Test discrimination between similar voices."""
    profiles = [
        load_profile("derek_1"),
        load_profile("derek_2"),  # Similar voice
        load_profile("john")       # Different voice
    ]

    sample = load_audio("derek_test.wav")

    # Should match derek_1, not derek_2
    result = speaker_verifier.verify(sample)
    assert result.profile_id == "derek_1"
    assert result.confidence > 0.75
```

**Solution:**
- 192-dimensional embeddings (high resolution)
- Quality bonus for clear samples
- Consistency checking
- Minimum sample count (59+)

#### 3. Wake Word False Positives

**Scenario:** "Hey JARVIS" detected in conversation

**Test:**
```python
def test_wake_word_false_positives():
    """Test wake word detection with similar phrases."""
    false_positives = [
        "Hey, Jarvis is cool",
        "Hey, are you serious?",
        "They are just fine"
    ]

    for phrase in false_positives:
        audio = text_to_speech(phrase)
        detected = wake_word_detector.detect(audio)
        assert not detected, f"False positive: {phrase}"
```

**Solution:**
- Sensitivity tuning (0.7 default)
- Energy-based fallback
- Confirmation timeout (1s)
- Double-check with STT

### Vision System Edge Cases

#### 1. Multi-Space Desktop

**Scenario:** User has 6 desktop spaces, target window on space 4

**Test:**
```python
def test_multi_space_window_detection():
    """Test window detection across desktop spaces."""
    # Create windows on different spaces
    create_window("Safari", space=1)
    create_window("Terminal", space=4)  # Target
    create_window("Finder", space=6)

    # Vision should find Terminal on space 4
    result = vision_api.analyze("Where is Terminal?")

    assert result.application == "Terminal"
    assert result.desktop_space == 4
    assert result.window_found is True
```

**Solution:**
- Yabai integration for space detection
- Multi-space iteration
- Coordinate translation
- Space activation before click

#### 2. Retina Display Scaling

**Scenario:** Coordinates doubled on Retina displays

**Test:**
```python
def test_retina_coordinate_translation():
    """Test coordinate translation on Retina displays."""
    # Logical coordinates (what user sees)
    logical_x, logical_y = 500, 500

    # Physical coordinates (what system needs)
    display = get_current_display()
    if display.is_retina:
        physical_x = logical_x * 2
        physical_y = logical_y * 2

    # Vision should handle automatically
    result = vision_api.click_at(logical_x, logical_y)
    assert result.success is True
    assert result.physical_coords == (physical_x, physical_y)
```

**Solution:**
- Automatic scale detection
- Coordinate translation layer
- Display-aware clicking
- Testing on Retina and non-Retina

#### 3. Vision API Rate Limiting

**Scenario:** Claude Vision API rate limit exceeded

**Test:**
```python
def test_vision_api_rate_limiting():
    """Test graceful handling of API rate limits."""
    # Send rapid requests
    for i in range(15):  # Limit is 10/min
        result = vision_api.analyze(f"Request {i}")

        if i < 10:
            assert result.success is True
        else:
            # Should fall back or queue
            assert result.success is False
            assert result.error == "RATE_LIMIT_EXCEEDED"
            assert result.retry_after > 0
```

**Solution:**
- Request queuing
- Exponential backoff
- Cache results (30s TTL)
- Fallback to local vision

### Database Edge Cases

#### 1. SQLite File Corruption

**Scenario:** Local database file corrupted

**Test:**
```python
def test_database_corruption_recovery():
    """Test recovery from database corruption."""
    # Simulate corruption
    corrupt_database("jarvis_local.db")

    # JARVIS should detect and recover
    db = JARVISLearningDatabase()

    assert db.is_healthy() is False
    recovery_result = db.attempt_recovery()

    assert recovery_result.success is True
    assert recovery_result.method == "cloud_restore"
    assert db.is_healthy() is True
```

**Solution:**
- Automatic corruption detection
- Cloud SQL restore
- Backup restoration
- Integrity checks on startup

#### 2. Cloud SQL Connection Lost

**Scenario:** Cloud SQL proxy crashes mid-operation

**Test:**
```python
def test_cloud_sql_connection_failure():
    """Test fallback when Cloud SQL connection fails."""
    # Start operation with cloud
    db = JARVISLearningDatabase(use_cloud=True)

    # Kill proxy mid-operation
    kill_cloud_sql_proxy()

    # Should fall back to SQLite
    result = db.store_pattern(pattern)

    assert result.success is True
    assert result.backend == "sqlite"
    assert result.will_sync_later is True
```

**Solution:**
- Connection health checks
- Automatic fallback to SQLite
- Queue for later sync
- Proxy restart attempts

### GCP & Hybrid Edge Cases

#### 1. VM Creation Failed (Quota)

**Scenario:** GCP quota exceeded, can't create VM

**Test:**
```python
def test_vm_creation_quota_exceeded():
    """Test handling of GCP quota exceeded."""
    # Set quota to 0
    mock_gcp_quota(cpus=0)

    # Try to create VM
    orchestrator = get_orchestrator()
    result = orchestrator.execute_command(
        command="analyze screen",
        command_type="vision_analyze"
    )

    # Should fall back to local processing
    assert result.success is True
    assert result.routed_to == "local"
    assert result.reason == "gcp_quota_exceeded"
```

**Solution:**
- Quota checking before creation
- Fallback to local processing
- User notification
- Quota increase guide

#### 2. Network Partition (Local ↔ Cloud)

**Scenario:** Network connection lost during cloud processing

**Test:**
```python
def test_network_partition_during_cloud_processing():
    """Test handling of network partition."""
    # Start cloud processing
    future = orchestrator.execute_command_async(
        command="complex analysis",
        command_type="ml_inference"
    )

    # Simulate network partition
    drop_all_packets()

    # Should timeout and retry locally
    result = await asyncio.wait_for(future, timeout=30)

    assert result.success is True
    assert result.routed_to == "local"
    assert result.reason == "network_timeout"
```

**Solution:**
- Connection timeouts (30s)
- Automatic retry logic
- Fallback to local
- Result caching

---

## Test Coverage

### Current Coverage

**Overall:** 78%
**Target:** 85%

**By Module:**
- Intelligence systems: 85% ✅
- Voice processing: 82% ✅
- Vision API: 75% 🟡
- Database: 90% ✅
- GCP integration: 68% 🔴
- API endpoints: 80% ✅

### Coverage Requirements

**New Code:**
- Unit tests: Required (>80%)
- Integration tests: Required for APIs
- E2E tests: Required for critical flows

**Critical Paths:**
- Voice unlock: 95% coverage
- Database operations: 90% coverage
- API endpoints: 85% coverage

---

## Testing Strategies

### 1. Property-Based Testing

Using Hypothesis for edge case discovery:

```python
from hypothesis import given, strategies as st

@given(
    audio_length=st.integers(min_value=100, max_value=10000),
    sample_rate=st.sampled_from([8000, 16000, 44100])
)
def test_stt_various_audio_configs(audio_length, sample_rate):
    """Test STT with various audio configurations."""
    audio = generate_audio(length=audio_length, rate=sample_rate)
    result = stt_engine.transcribe(audio)

    # Should handle gracefully
    assert result is not None
    assert 0 <= result.confidence <= 1.0
```

### 2. Chaos Engineering

Testing system resilience:

```python
def test_chaos_random_component_failures():
    """Test system resilience to random failures."""
    components = ["uae", "sai", "cai", "vision", "voice"]

    for i in range(10):
        # Randomly kill component
        component = random.choice(components)
        kill_component(component)

        # System should self-heal via SAI
        time.sleep(5)
        health = check_system_health()

        assert health[component] == "recovered"
```

### 3. Load Testing

Testing under high load:

```python
def test_concurrent_voice_commands():
    """Test handling of concurrent voice commands."""
    async def send_command():
        return await orchestrator.execute_command(
            command="what time is it?",
            command_type="voice_command"
        )

    # Send 50 concurrent commands
    tasks = [send_command() for _ in range(50)]
    results = await asyncio.gather(*tasks)

    # All should succeed
    successes = sum(1 for r in results if r.success)
    assert successes >= 48  # 96% success rate
```

---

## CI/CD Testing

### GitHub Actions Integration

All tests run automatically on:
- Every push
- Every pull request
- Nightly (full suite)

**Test Matrix:**
- Python: 3.10, 3.11
- OS: Ubuntu, macOS
- Scenarios: Unit, Integration, E2E

### Pre-Commit Hooks

Automatically run before commit:
```bash
# .pre-commit-config.yaml
- id: pytest-fast
  name: Fast unit tests
  entry: pytest tests/unit/ -x
  language: system
  pass_filenames: false
```

---

**Related Documentation:**
- [CI/CD Workflows](CI-CD-Workflows.md) - Automation details
- [Contributing Guidelines](Contributing-Guidelines.md) - How to add tests

---

**Last Updated:** 2025-10-30
