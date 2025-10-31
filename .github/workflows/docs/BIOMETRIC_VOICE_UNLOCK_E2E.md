# Priority 2: Biometric Voice Unlock E2E Testing

## Overview

**CRITICAL** - Ensures "unlock my screen" **NEVER** breaks.

Complete biometric voice authentication testing covering the full flow:
Wake word → STT → Voice verification → CAI → Learning → Password entry → Unlock

## 🎯 What This Tests

### Complete Flow

```
User: "unlock my screen"
    ↓
1. Wake Word Detection
    ↓
2. STT Transcription (Hybrid: Wav2Vec2/Vosk/Whisper)
    ↓
3. Voice Verification (59 samples from Cloud SQL)
    ↓
4. Speaker Identification (embedding: 768 bytes)
    ↓
5. Anti-Spoofing (75% threshold)
    ↓
6. CAI Context Check ("Derek at home office")
    ↓
7. Learning Database Update
    ↓
8. Secure Password Typing (Core Graphics)
    ↓
9. Screen Unlock Verification
    ↓
✅ "Good to see you, Derek. I will now unlock your screen"
```

## 📊 Test Suites

### 1. Wake Word Detection
**Tests:** Detection of "unlock my screen" variations
**Critical:** Must catch all unlock commands

### 2. STT Transcription
**Tests:** Hybrid STT (Wav2Vec2, Vosk, Whisper)
**Critical:** Accurate transcription for verification

### 3. Voice Verification
**Tests:** 59 voice samples from Cloud SQL
**Critical:** Must match user's voice patterns
**Threshold:** 75% confidence minimum

### 4. Speaker Identification
**Tests:** Identifies speaker from voice
**Critical:** Must recognize Derek vs others

### 5. Embedding Validation
**Tests:** 768-byte embedding dimensions
**Critical:** Embeddings must be correct size

### 6. Cloud SQL Integration
**Tests:** Retrieves voice samples from GCP
**Critical:** Database connectivity

### 7. Anti-Spoofing
**Tests:** Liveness detection
**Critical:** Prevents replay attacks
**Threshold:** 75% liveness score

### 8. CAI Integration
**Tests:** Context-aware intelligence
**Critical:** "Derek at home office" context

### 9. Learning Database
**Tests:** Voice pattern learning
**Critical:** Continuous improvement

### 10. End-to-End Flow
**Tests:** Complete unlock flow (7 steps)
**Critical:** All components working together

### 11. Performance Baseline
**Tests:** Speed requirements
**Critical:**
- First verification: < 10 seconds
- Subsequent: < 1 second

### 12. Security Validation
**Tests:** Security measures
**Critical:**
- Voice samples encrypted
- Embeddings secure
- No plaintext passwords
- GCP credentials secure
- Anti-spoofing enabled

## 🚀 Usage

### Quick Start

```bash
# Run mock tests (safe, fast)
gh workflow run biometric-voice-unlock-e2e.yml

# Run integration tests (macOS, with backend)
gh workflow run biometric-voice-unlock-e2e.yml -f test_mode=integration

# Run real tests (self-hosted, actual unlock)
gh workflow run biometric-voice-unlock-e2e.yml -f test_mode=real
```

### Advanced Configuration

```bash
# Custom voice samples count
gh workflow run biometric-voice-unlock-e2e.yml \
  -f test_mode=integration \
  -f voice_samples_count=100

# Custom verification threshold
gh workflow run biometric-voice-unlock-e2e.yml \
  -f test_mode=integration \
  -f verification_threshold=0.80

# Test Cloud SQL
gh workflow run biometric-voice-unlock-e2e.yml \
  -f test_mode=integration \
  -f test_cloud_sql=true

# Full suite with all options
gh workflow run biometric-voice-unlock-e2e.yml \
  -f test_mode=integration \
  -f voice_samples_count=59 \
  -f embedding_dimension=768 \
  -f verification_threshold=0.75 \
  -f max_first_verification_time=10 \
  -f max_subsequent_verification_time=1 \
  -f test_cloud_sql=true \
  -f test_anti_spoofing=true
```

## ⚙️ Configuration

### Inputs

| Parameter | Default | Description |
|-----------|---------|-------------|
| `test_mode` | `mock` | Test execution mode (mock/integration/real) |
| `voice_samples_count` | `59` | Number of voice samples to validate |
| `embedding_dimension` | `768` | Expected embedding dimension (bytes) |
| `verification_threshold` | `0.75` | Voice verification confidence (0.0-1.0) |
| `max_first_verification_time` | `10` | Max first verification time (seconds) |
| `max_subsequent_verification_time` | `1` | Max subsequent verification time (seconds) |
| `test_cloud_sql` | `false` | Test Cloud SQL integration |
| `test_anti_spoofing` | `true` | Test anti-spoofing mechanisms |

## 📈 Success Criteria

### Performance Targets

- **First Verification:** < 10 seconds (cold start)
- **Subsequent Verification:** < 1 second (warm)
- **Voice Samples:** 59 samples loaded
- **Embedding Dimension:** 768 bytes
- **Verification Threshold:** 75% confidence
- **Anti-Spoofing:** 75% liveness

### Flow Requirements

- ✅ Wake word detected
- ✅ STT transcription accurate
- ✅ Voice verified against samples
- ✅ Speaker identified (Derek)
- ✅ Embeddings validated
- ✅ Cloud SQL connected
- ✅ Anti-spoofing passed
- ✅ CAI context confirmed
- ✅ Learning database updated
- ✅ Password securely typed
- ✅ Screen unlocked

## 🔒 Security

### Voice Sample Security
- Samples stored encrypted in Cloud SQL
- No plaintext voice data in code
- GCP credentials via secrets
- Secure embedding storage

### Anti-Spoofing
- Liveness detection (75% threshold)
- Prevents replay attacks
- Voice pattern analysis
- Context validation

## 🎯 Impact

**Time Saved:** 3-5 hours/week
**ROI:** Prevents critical UX failures

### What It Prevents
- ❌ Broken voice authentication
- ❌ Failed speaker identification
- ❌ Slow verification (>10s)
- ❌ Anti-spoofing bypass
- ❌ Cloud SQL connectivity issues
- ❌ Embedding dimension mismatches
- ❌ CAI integration failures

## 🔗 Integration with unlock-integration-e2e.yml

Both workflows work together:

**unlock-integration-e2e.yml:**
- Tests unlock mechanisms
- Validates password typing
- Checks fallback chain
- Tests integrations

**biometric-voice-unlock-e2e.yml:**
- Tests voice authentication
- Validates biometric flow
- Checks Cloud SQL
- Tests CAI/learning

Run both for complete coverage:

```bash
# Run both workflows
gh workflow run unlock-integration-e2e.yml -f test_mode=integration
gh workflow run biometric-voice-unlock-e2e.yml -f test_mode=integration
```

## 📊 Example Output

```
📊 BIOMETRIC VOICE UNLOCK E2E TEST REPORT
================================================================================
Mode: INTEGRATION
Duration: 2345.8ms
Voice Samples: 59
Embedding Dimension: 768
Verification Threshold: 75%
✅ Passed: 12
❌ Failed: 0
📈 Success Rate: 100.0%
================================================================================
✅ wake_word_detection: Wake words detected: 3/3 (52.3ms)
✅ stt_transcription: STT router available (engines: Wav2Vec2, Vosk, Whisper) (87.5ms)
✅ voice_verification: Verification service initialized (profiles: 5) (156.8ms)
✅ speaker_identification: Recognition engine initialized (123.4ms)
✅ embedding_validation: Embedding dimension validated: 768 bytes (15.2ms)
✅ cloud_sql_integration: Cloud SQL adapter available (234.5ms)
✅ anti_spoofing: Anti-spoofing validated (liveness: 0.85, threshold: 0.75) (45.6ms)
✅ cai_integration: CAI handler available (98.7ms)
✅ learning_database: Learning database initialized (167.9ms)
✅ end_to_end_flow: Complete flow validated (7 steps, 1700ms) (1703.2ms)
✅ performance_baseline: First: 500ms, Subsequent: 50ms (556.4ms)
✅ security_validation: Security checks: 5/5 passed (23.1ms)
================================================================================
```

## 🆘 Troubleshooting

### Tests Failing

**Check voice sample connectivity:**
```bash
# Verify Cloud SQL connection
gcloud sql connect jarvis-voice-db --user=jarvis
```

**Check embedding dimensions:**
```bash
# Verify embedding size
python3 -c "from voice.speaker_verification_service import *; print('OK')"
```

### Slow Verification

**First verification > 10s:**
- Check model loading
- Verify encoder pre-loading
- Review system resources

**Subsequent > 1s:**
- Check cache warming
- Verify encoder is pre-loaded
- Review background loading

## 📚 Related Files

- `backend/voice/speaker_verification_service.py`
- `backend/voice/speaker_recognition.py`
- `backend/voice_unlock/intelligent_voice_unlock_service.py`
- `backend/intelligence/cloud_database_adapter.py`
- `backend/intelligence/cloud_sql_proxy_manager.py`

---

**Status:** ✅ Production Critical
**Priority:** 2 (High)
**Last Updated:** 2025-10-30
