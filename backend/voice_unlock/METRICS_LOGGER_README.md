# 📊 Advanced Voice Unlock Metrics Logger

**Version**: 2.0.0-advanced-async
**Author**: JARVIS AI Development Team
**Last Updated**: November 13, 2025

---

## 🎯 Overview

The Advanced Voice Unlock Metrics Logger is a **fully async, zero-hardcoding, production-grade** biometric authentication monitoring system. It provides comprehensive, detailed insights into every voice unlock attempt, tracking performance, confidence trends, and processing stages in real-time.

### Why This Matters

Understanding your biometric authentication system is critical for:
- **Security Analysis**: Track confidence scores vs thresholds over time
- **Performance Optimization**: Identify bottlenecks in the unlock pipeline
- **Machine Learning**: Build datasets for model improvement
- **Debugging**: Trace exactly what happened during failed attempts
- **User Experience**: Monitor unlock times and success rates

---

## ✨ Key Features

### 🚀 Fully Asynchronous
- **Non-blocking I/O** using `aiofiles`
- Parallel file operations for maximum performance
- No impact on unlock latency

### 📈 Dynamic Stage Tracking
- **Zero hardcoding** - automatically detects algorithms, files, and modules
- **9 processing stages** tracked end-to-end:
  1. Audio Preparation
  2. Transcription (Hybrid STT)
  3. Intent Verification
  4. Speaker Identification
  5. Owner Verification
  6. Biometric Verification (Anti-Spoofing)
  7. Context Analysis (CAI)
  8. Scenario Analysis (SAI)
  9. Unlock Execution

### 🧠 Confidence Tracking Over Time
- **Historical trends** (last 10, last 30 attempts)
- **Trend direction** (improving, declining, stable)
- **Volatility analysis** (standard deviation)
- **Percentile ranking** (how does this attempt compare to history?)
- **Best/worst scores** ever recorded

### 🔬 Per-Stage Granularity
Each stage captures:
- **Exact timing** (milliseconds)
- **Success/failure** status
- **Algorithm used** (e.g., "SpeechBrain ECAPA-TDNN", "Whisper")
- **Module path** and function name (auto-detected via introspection)
- **Input/output sizes** (bytes)
- **Confidence scores** vs thresholds
- **Custom metadata**

### 📁 Three-Tier Storage
1. **Daily Logs**: Detailed per-attempt metrics (`unlock_metrics_YYYY-MM-DD.json`)
2. **Aggregated Stats**: Overall statistics (`unlock_stats.json`)
3. **Confidence Trends**: Historical data for trend analysis (`confidence_trends.json`)

---

## 📂 File Structure

```
~/.jarvis/logs/unlock_metrics/
├── unlock_metrics_2025-11-13.json    # Today's detailed logs
├── unlock_metrics_2025-11-12.json    # Yesterday's logs
├── unlock_stats.json                 # Aggregated statistics
└── confidence_trends.json            # Historical confidence data
```

### Daily Logs (`unlock_metrics_YYYY-MM-DD.json`)

One entry per unlock attempt with full details:

```json
{
  "timestamp": "2025-11-13T21:30:45.123",
  "date": "2025-11-13",
  "time": "21:30:45.123",
  "day_of_week": "Wednesday",
  "unix_timestamp": 1731542445.123,

  "success": true,
  "speaker_name": "Derek J. Russell",
  "transcribed_text": "unlock my screen",
  "error": null,

  "biometrics": {
    "speaker_confidence": 0.43,
    "stt_confidence": 0.85,
    "threshold": 0.35,
    "above_threshold": true,
    "confidence_margin": 0.08,
    "confidence_percentage": "43.0%",

    "confidence_vs_threshold": {
      "current_confidence": 0.43,
      "threshold": 0.35,
      "margin": 0.08,
      "margin_percentage": 22.86,
      "above_threshold": true
    },

    "confidence_trends": {
      "avg_last_10": 0.41,
      "avg_last_30": 0.39,
      "trend_direction": "improving",
      "volatility": 0.05,
      "best_ever": 0.48,
      "worst_ever": 0.31,
      "current_rank_percentile": 67.5
    }
  },

  "performance": {
    "total_latency_ms": 18543,
    "total_duration_ms": 18543,
    "transcription_time_ms": 8542,

    "stages_breakdown": {
      "audio_preparation": {
        "duration_ms": 25,
        "percentage": 0.13
      },
      "transcription": {
        "duration_ms": 8542,
        "percentage": 46.05
      },
      "biometric_verification": {
        "duration_ms": 5234,
        "percentage": 28.21
      }
    },

    "slowest_stage": "transcription",
    "fastest_stage": "audio_preparation"
  },

  "quality_indicators": {
    "audio_quality": "good",
    "voice_match_quality": "acceptable",
    "overall_confidence": 0.64
  },

  "processing_stages": [
    {
      "stage_name": "audio_preparation",
      "started_at": 1731542445.100,
      "ended_at": 1731542445.125,
      "duration_ms": 25.0,
      "success": true,
      "algorithm_used": "prepare_audio_for_stt",
      "module_path": "/Users/.../intelligent_voice_unlock_service.py",
      "function_name": "process_voice_unlock_command",
      "input_size_bytes": 337408,
      "output_size_bytes": 337408,
      "confidence_score": null,
      "threshold": null,
      "above_threshold": null,
      "error_message": null,
      "percentage_of_total": 0.13,
      "metadata": {
        "input_type": "bytes",
        "input_size_raw": 337408
      }
    },
    {
      "stage_name": "transcription",
      "started_at": 1731542445.125,
      "ended_at": 1731542453.667,
      "duration_ms": 8542.0,
      "success": true,
      "algorithm_used": "Whisper",
      "module_path": "/Users/.../intelligent_voice_unlock_service.py",
      "function_name": "process_voice_unlock_command",
      "input_size_bytes": null,
      "output_size_bytes": 16,
      "confidence_score": 0.85,
      "threshold": null,
      "above_threshold": null,
      "error_message": null,
      "percentage_of_total": 46.05,
      "metadata": {
        "transcribed_text": "unlock my screen",
        "speaker_identified": "Derek J. Russell",
        "sample_rate": 44100
      }
    },
    {
      "stage_name": "biometric_verification",
      "started_at": 1731542454.100,
      "ended_at": 1731542459.334,
      "duration_ms": 5234.0,
      "success": true,
      "algorithm_used": "SpeechBrain ECAPA-TDNN",
      "module_path": "/Users/.../intelligent_voice_unlock_service.py",
      "function_name": "process_voice_unlock_command",
      "input_size_bytes": null,
      "output_size_bytes": null,
      "confidence_score": 0.43,
      "threshold": 0.35,
      "above_threshold": true,
      "error_message": null,
      "percentage_of_total": 28.21,
      "metadata": {
        "speaker_name": "Derek J. Russell",
        "audio_size": 337408,
        "verification_method": "cosine_similarity",
        "embedding_dimension": 192
      }
    }
  ],

  "stage_summary": {
    "total_stages": 9,
    "successful_stages": 9,
    "failed_stages": 0,
    "stages_above_threshold": 1,
    "all_stages_passed": true
  },

  "system_info": {
    "platform": "Darwin",
    "platform_version": "Darwin Kernel Version 24.6.0",
    "python_version": "3.11.5",
    "stt_engine": "Whisper",
    "speaker_engine": "SpeechBrain"
  },

  "metadata": {
    "total_attempts_today": 5,
    "session_id": "20251113_213045",
    "logger_version": "2.0.0-advanced-async"
  }
}
```

### Aggregated Stats (`unlock_stats.json`)

```json
{
  "total_attempts": 147,
  "successful_attempts": 142,
  "failed_attempts": 5,
  "speakers": {
    "Derek J. Russell": {
      "total_attempts": 147,
      "successful_attempts": 142,
      "avg_confidence": 0.42,
      "best_confidence": 0.48,
      "worst_confidence": 0.31,
      "avg_duration_ms": 18234.5
    }
  },
  "last_updated": "2025-11-13T21:30:45.123"
}
```

### Confidence Trends (`confidence_trends.json`)

```json
{
  "Derek J. Russell": {
    "confidence_history": [0.38, 0.41, 0.43, 0.39, 0.42, ...],
    "success_history": [true, true, true, false, true, ...],
    "timestamps": [
      "2025-11-13T20:15:30.123",
      "2025-11-13T20:45:12.456",
      ...
    ]
  }
}
```

---

## 🚀 Usage

### Automatic Logging

The metrics logger is **automatically integrated** into the voice unlock service. Every unlock attempt is logged automatically with full details.

No manual intervention required!

### Accessing Logs

```bash
# View today's detailed logs
cat ~/.jarvis/logs/unlock_metrics/unlock_metrics_$(date +%Y-%m-%d).json | jq '.'

# View aggregated stats
cat ~/.jarvis/logs/unlock_metrics/unlock_stats.json | jq '.'

# View confidence trends
cat ~/.jarvis/logs/unlock_metrics/confidence_trends.json | jq '.'

# Count total attempts today
cat ~/.jarvis/logs/unlock_metrics/unlock_metrics_$(date +%Y-%m-%d).json | jq 'length'

# Get average confidence today
cat ~/.jarvis/logs/unlock_metrics/unlock_metrics_$(date +%Y-%m-%d).json | jq '[.[].biometrics.speaker_confidence] | add / length'

# Find failed attempts
cat ~/.jarvis/logs/unlock_metrics/unlock_metrics_$(date +%Y-%m-%d).json | jq '.[] | select(.success == false)'

# Show slowest stage breakdown
cat ~/.jarvis/logs/unlock_metrics/unlock_metrics_$(date +%Y-%m-%d).json | jq '.[].performance.stages_breakdown'
```

### Programmatic Access

```python
from voice_unlock.unlock_metrics_logger import get_metrics_logger

# Get the singleton logger instance
logger = get_metrics_logger()

# Get today's statistics (async)
stats = await logger.get_today_stats()
print(f"Success rate: {stats['success_rate']:.1f}%")
print(f"Average confidence: {stats['avg_confidence']:.2f}")
print(f"Average latency: {stats['avg_latency']:.0f}ms")
```

---

## 📊 Understanding the Metrics

### Biometric Confidence

- **speaker_confidence**: How confident the system is in speaker identification (0.0 - 1.0)
- **threshold**: Minimum confidence required to pass (default: 0.35)
- **confidence_margin**: How far above/below the threshold (positive = pass, negative = fail)
- **confidence_percentage**: Confidence as a percentage string

**What to look for:**
- Confidence consistently > 0.40 = excellent voice profile
- Confidence 0.35 - 0.40 = acceptable, may improve over time
- Confidence < 0.35 = failed authentication
- Increasing trend = system is learning your voice better

### Stage Performance

**Typical timing breakdown:**
- Audio Preparation: ~25ms (0.1% of total)
- Transcription: ~8-10 seconds (45-50% of total)
- Intent Verification: ~10ms (0.05% of total)
- Speaker Identification: ~100ms (0.5% of total)
- Owner Verification: ~5ms (0.02% of total)
- Biometric Verification: ~5-6 seconds (28-30% of total)
- Context Analysis: ~50ms (0.3% of total)
- Scenario Analysis: ~50ms (0.3% of total)
- Unlock Execution: ~3-4 seconds (18-20% of total)

**Total typical unlock time**: 18-22 seconds

**Optimization opportunities:**
- If transcription > 12s: Consider audio quality improvements
- If biometric_verification > 8s: May need more voice samples
- If unlock_execution > 5s: Check system performance

### Confidence Trends

- **trend_direction**:
  - "improving" = Recent attempts better than older ones (good!)
  - "declining" = Recent attempts worse (may need re-enrollment)
  - "stable" = Consistent performance
  - "new" = First attempt
  - "insufficient_data" = < 5 attempts

- **volatility**: Standard deviation of confidence scores
  - < 0.03 = Very stable
  - 0.03 - 0.06 = Normal variation
  - \> 0.06 = High variation (consider re-enrollment)

- **current_rank_percentile**: How this attempt compares to history
  - 100% = Best attempt ever
  - 50% = Median performance
  - 0% = Worst attempt ever

---

## 🔧 Configuration

### Changing the Log Directory

```python
from voice_unlock.unlock_metrics_logger import UnlockMetricsLogger

# Use a custom log directory
logger = UnlockMetricsLogger(log_dir="/path/to/custom/logs")
```

### Adjusting Confidence Threshold

The threshold is dynamically read from the speaker engine:

```python
# In your speaker verification service
self.threshold = 0.40  # Increase for higher security
```

The metrics logger will automatically use this value.

---

## 📈 Analysis Examples

### Daily Success Rate

```bash
#!/bin/bash
# daily_success_rate.sh

LOG_FILE=~/.jarvis/logs/unlock_metrics/unlock_metrics_$(date +%Y-%m-%d).json

if [ ! -f "$LOG_FILE" ]; then
  echo "No attempts today"
  exit 0
fi

TOTAL=$(jq 'length' "$LOG_FILE")
SUCCESS=$(jq '[.[] | select(.success == true)] | length' "$LOG_FILE")
RATE=$(echo "scale=2; $SUCCESS * 100 / $TOTAL" | bc)

echo "Total attempts: $TOTAL"
echo "Successful: $SUCCESS"
echo "Success rate: $RATE%"
```

### Confidence Trend Visualization

```python
#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Load confidence trends
trends_file = Path.home() / ".jarvis/logs/unlock_metrics/confidence_trends.json"
with open(trends_file) as f:
    trends = json.load(f)

# Plot for Derek J. Russell
speaker = "Derek J. Russell"
if speaker in trends:
    confidences = trends[speaker]['confidence_history']
    timestamps = trends[speaker]['timestamps']

    plt.figure(figsize=(12, 6))
    plt.plot(confidences, marker='o')
    plt.axhline(y=0.35, color='r', linestyle='--', label='Threshold')
    plt.title(f'Confidence Trend - {speaker}')
    plt.xlabel('Attempt Number')
    plt.ylabel('Confidence Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('confidence_trend.png')
    print(f"Saved to confidence_trend.png")
```

### Stage Performance Report

```python
#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Load today's logs
today = datetime.now().strftime("%Y-%m-%d")
log_file = Path.home() / f".jarvis/logs/unlock_metrics/unlock_metrics_{today}.json"

with open(log_file) as f:
    logs = json.load(f)

# Aggregate stage timings
stage_times = defaultdict(list)

for entry in logs:
    for stage in entry['processing_stages']:
        if stage['duration_ms']:
            stage_times[stage['stage_name']].append(stage['duration_ms'])

# Calculate averages
print("\n=== Stage Performance Report ===\n")
for stage, times in sorted(stage_times.items()):
    avg = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    print(f"{stage:30s}: avg={avg:7.1f}ms  min={min_time:7.1f}ms  max={max_time:7.1f}ms")
```

---

## 🐛 Troubleshooting

### No Logs Being Created

**Problem**: No files in `~/.jarvis/logs/unlock_metrics/`

**Solutions**:
1. Check backend is running: `ps aux | grep start_system.py`
2. Check permissions: `ls -la ~/.jarvis/logs/`
3. Look for errors: `tail -f ~/.jarvis/logs/jarvis_*.log | grep metrics`

### Incomplete Stage Data

**Problem**: Some stages missing from `processing_stages`

**Solutions**:
1. Stage failed before completion (check `error_message`)
2. Backend crashed mid-unlock (check system logs)
3. Import error in metrics logger (check Python logs)

### Confidence Trends Not Updating

**Problem**: `confidence_trends.json` not being updated

**Solutions**:
1. Check file permissions: `chmod 644 ~/.jarvis/logs/unlock_metrics/confidence_trends.json`
2. Check disk space: `df -h`
3. Look for async errors in logs

### Very High Latency

**Problem**: `total_duration_ms` > 30 seconds

**Solutions**:
1. Check `stages_breakdown` to find bottleneck
2. Most common: Slow transcription (try different STT engine)
3. Check CPU usage during unlock
4. Review `system_info.platform` for system details

---

## 🔬 Advanced Features

### Custom Stage Tracking

You can add custom stages to your own unlock logic:

```python
from voice_unlock.unlock_metrics_logger import get_metrics_logger

logger = get_metrics_logger()
stages = []

# Create a custom stage
stage_custom = logger.create_stage(
    "my_custom_check",
    custom_param="value"
)

try:
    # Your custom logic here
    result = do_something()

    stage_custom.complete(
        success=True,
        algorithm_used="My Algorithm",
        confidence_score=0.95,
        metadata={'result': result}
    )
except Exception as e:
    stage_custom.complete(
        success=False,
        error_message=str(e)
    )

stages.append(stage_custom)
```

### Querying Historical Data

```python
import json
from pathlib import Path
from datetime import datetime, timedelta

async def get_last_week_stats(speaker_name: str):
    """Get statistics for the last 7 days"""
    log_dir = Path.home() / ".jarvis/logs/unlock_metrics"

    all_attempts = []
    for i in range(7):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        log_file = log_dir / f"unlock_metrics_{date}.json"

        if log_file.exists():
            with open(log_file) as f:
                daily = json.load(f)
                all_attempts.extend([
                    e for e in daily
                    if e['speaker_name'] == speaker_name
                ])

    # Calculate stats
    total = len(all_attempts)
    successful = sum(1 for e in all_attempts if e['success'])
    confidences = [e['biometrics']['speaker_confidence'] for e in all_attempts]

    return {
        'total_attempts': total,
        'success_rate': (successful / total * 100) if total > 0 else 0,
        'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
        'best_confidence': max(confidences) if confidences else 0,
        'worst_confidence': min(confidences) if confidences else 0,
    }
```

---

## 📝 Best Practices

### 1. Regular Monitoring

Set up a daily cron job to check your unlock metrics:

```bash
# Add to crontab: crontab -e
0 9 * * * /path/to/daily_unlock_report.sh | mail -s "JARVIS Daily Unlock Report" you@example.com
```

### 2. Confidence Baseline

After enrollment, perform 10-20 test unlocks to establish your baseline:

```bash
# Check your baseline
cat ~/.jarvis/logs/unlock_metrics/confidence_trends.json | \
  jq '."Derek J. Russell".confidence_history[-20:] | add / length'
```

Your baseline should be >= 0.38 for reliable unlocking.

### 3. Periodic Re-enrollment

If you notice:
- `trend_direction`: "declining" for 3+ days
- `volatility` > 0.08
- Success rate < 90%

Consider re-enrolling your voice profile.

### 4. Storage Management

Logs can grow large. Archive old logs monthly:

```bash
# Archive logs older than 30 days
find ~/.jarvis/logs/unlock_metrics/ -name "unlock_metrics_*.json" -mtime +30 -exec gzip {} \;
```

---

## 🔐 Privacy & Security

### Data Sensitivity

The metrics logs contain:
- ✅ Confidence scores, timings, system info (safe to share)
- ⚠️ Speaker names, transcribed text (personally identifiable)
- ❌ **Never contains**: Passwords, audio data, voice embeddings

### Recommendations

1. **Restrict access**: `chmod 600 ~/.jarvis/logs/unlock_metrics/*`
2. **Encrypt backups**: Use `gpg` or FileVault
3. **Rotate logs**: Archive and delete logs > 90 days
4. **No cloud sync**: Keep logs local only

---

## 🤝 Contributing

Found a bug or have a feature request? Please open an issue!

### Development Setup

```bash
# Install dev dependencies
pip install aiofiles pytest pytest-asyncio

# Run tests
pytest backend/voice_unlock/tests/test_unlock_metrics_logger.py
```

---

## 📚 API Reference

### `UnlockMetricsLogger`

**Methods:**

- `create_stage(stage_name: str, **metadata) -> StageMetrics`
  - Creates and starts tracking a new processing stage
  - Returns a StageMetrics object

- `async log_unlock_attempt(...) -> None`
  - Logs a complete unlock attempt with all stages
  - Fully async, non-blocking

- `async get_today_stats() -> Dict[str, Any]`
  - Returns statistics for today's unlock attempts

### `StageMetrics`

**Attributes:**

- `stage_name: str` - Name of the stage
- `started_at: float` - Unix timestamp when stage started
- `ended_at: float` - Unix timestamp when stage ended
- `duration_ms: float` - Duration in milliseconds
- `success: bool` - Whether stage succeeded
- `algorithm_used: str` - Algorithm/method used
- `module_path: str` - File path (auto-detected)
- `function_name: str` - Function name (auto-detected)
- `confidence_score: float` - Confidence score if applicable
- `threshold: float` - Threshold if applicable
- `above_threshold: bool` - Whether score exceeded threshold
- `metadata: Dict` - Custom metadata

**Methods:**

- `complete(success: bool, **kwargs) -> None`
  - Marks stage as complete and calculates duration

---

## 📄 License

This metrics logger is part of the JARVIS AI Agent project.

---

## 🙏 Acknowledgments

- **SpeechBrain**: ECAPA-TDNN speaker recognition
- **OpenAI Whisper**: Speech-to-text transcription
- **aiofiles**: Async file I/O

---

**Questions?** Check the main [JARVIS Voice Unlock README](README.md) or open an issue!
