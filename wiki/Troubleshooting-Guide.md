# Troubleshooting Guide

Common issues and solutions for JARVIS AI Agent.

---

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Voice System Issues](#voice-system-issues)
3. [Database Issues](#database-issues)
4. [GCP & Cloud Issues](#gcp--cloud-issues)
5. [Performance Issues](#performance-issues)
6. [General Issues](#general-issues)

---

## Installation Issues

### Python Dependencies Failed to Install

**Symptoms:** `pip install` errors, missing packages

**Solutions:**
```bash
# Update pip
pip install --upgrade pip setuptools wheel

# Install with specific Python version
python3.10 -m pip install -r backend/requirements.txt

# For M1/M2 Macs, use miniforge
conda install -c conda-forge package_name
```

### PyObjC Installation Fails (macOS M1/M2)

**Solution:**
```bash
# Use specific versions for miniforge compatibility
pip install pyobjc-core==10.1
pip install pyobjc-framework-Cocoa==12.0
pip install pyobjc-framework-Quartz==10.1
```

---

## Voice System Issues

### Wake Word Not Detected

**Symptoms:** "Hey JARVIS" not triggering

**Solutions:**
1. Check microphone permissions
2. Test microphone: `python -c "import sounddevice; print(sounddevice.query_devices())"`
3. Adjust sensitivity in `~/.jarvis/voice/config.json`
4. Try fallback energy-based detection

### Voice Authentication Failed

**Symptoms:** "Voice authentication failed. Access denied."

**Solutions:**
```bash
# Re-enroll your voice
python backend/voice_unlock/enroll_voice.py

# Check speaker profiles
python -c "
from intelligence.speaker_verification import get_speaker_verifier
verifier = get_speaker_verifier()
print(f'Profiles loaded: {len(verifier.profiles)}')
"

# Lower confidence threshold (testing only)
# Edit backend/voice_unlock/config.py
# CONFIDENCE_THRESHOLD = 0.70  # Lower from 0.75
```

### Speech-to-Text Not Working

**Solution:**
```bash
# Install SpeechBrain
pip install speechbrain==0.5.16 torchaudio==2.1.2

# Test STT
python -c "
from voice.stt_engine import get_stt_engine
engine = get_stt_engine()
print(f'STT Engine: {engine.engine_type}')
"
```

---

## Database Issues

### Cloud SQL Proxy Not Starting

**Symptoms:** Connection timeout, proxy errors

**Solutions:**
```bash
# Check if proxy binary exists
which cloud-sql-proxy

# Install if missing
gcloud components install cloud-sql-proxy

# Check GCP authentication
gcloud auth application-default login

# Restart proxy
python backend/intelligence/cloud_sql_proxy_manager.py restart --force

# Check logs
tail -f /tmp/cloud-sql-proxy.log
```

### Database Connection Timeout

**Solution:**
```bash
# Test direct connection
PGPASSWORD=JarvisSecure2025! psql -h 127.0.0.1 -U jarvis -d jarvis_learning

# Check proxy is running
lsof -i :5432

# Fall back to SQLite
# Edit .env: USE_CLOUD_SQL=false
```

### Database Sync Failing

**Solution:**
```bash
# Manual sync
cd backend
python -m database.sync_databases

# Check sync logs
tail -f logs/database_sync.log
```

---

## GCP & Cloud Issues

### VM Creation Failed

**Symptoms:** "GCP VM creation failed", quota errors

**Solutions:**
```bash
# Check GCP quotas
gcloud compute project-info describe --project=jarvis-473803

# Increase quota (in GCP Console)
# IAM & Admin → Quotas → Compute Engine API

# Check budget limits
# Edit backend/core/hybrid_config.yaml
# max_hourly_cost: 0.20  # Increase if needed
```

### VM Won't Terminate

**Solution:**
```bash
# List running VMs
gcloud compute instances list --project=jarvis-473803

# Force terminate
gcloud compute instances delete jarvis-spot-vm-XXXXX \
  --zone=us-central1-a --quiet

# Check orphaned VMs
python backend/core/gcp_vm_manager.py cleanup
```

### High Cloud Costs

**Solution:**
```bash
# Check cost tracking
curl http://localhost:8010/database/stats | jq '.cloud.cost_today_usd'

# Lower auto-scale threshold
# Edit backend/core/hybrid_config.yaml
# cloud_shift_threshold: 90  # Increase from 85

# Reduce VM lifetime
# Edit backend/core/hybrid_config.yaml
# auto_shutdown_idle_minutes: 10  # Reduce from 15
```

---

## Performance Issues

### High Memory Usage (Local)

**Symptoms:** >85% RAM, system slow

**Solutions:**
```bash
# Enable GCP auto-scaling
# Edit .env: GCP_VM_ENABLED=true

# Check memory status
python -c "
from core.platform_memory_monitor import PlatformMemoryMonitor
monitor = PlatformMemoryMonitor()
print(monitor.get_memory_status())
"

# Reduce local components
# Edit backend/core/hybrid_config.yaml
# local_threshold: 60  # Shift earlier
```

### Slow Response Times

**Symptoms:** 5-15 second delays

**Solutions:**
1. Enable cloud routing (if >85% RAM)
2. Check internet connection speed
3. Verify no CPU throttling: `sudo powermetrics --samplers smc -i1 -n1`
4. Clear cache: `rm -rf backend/cache/*`

### API Rate Limiting

**Solution:**
```bash
# Check rate limits
curl http://localhost:8010/health | jq '.rate_limits'

# Implement caching
# Results cached automatically in learning_database
```

---

## General Issues

### JARVIS Won't Start

**Symptoms:** `start_system.py` fails

**Solutions:**
```bash
# Check logs
tail -f jarvis_startup.log

# Verify Python version
python --version  # Should be 3.10 or 3.11

# Check port availability
lsof -i :8010

# Kill conflicting processes
pkill -f "uvicorn.*8010"

# Fresh restart
python start_system.py --restart
```

### WebSocket Connection Failed

**Solution:**
```bash
# Check backend is running
curl http://localhost:8010/health

# Test WebSocket
wscat -c ws://localhost:8010/ws

# Check firewall settings
sudo pfctl -sr | grep 8010
```

### Frontend Not Loading

**Solution:**
```bash
cd frontend

# Rebuild frontend
npm run build

# Check build errors
npm run build 2>&1 | tee build.log

# Serve static files
python -m http.server 3000
```

---

**For More Help:**
- Check [GitHub Issues](https://github.com/derekjrussell/JARVIS-AI-Agent/issues)
- Review logs in `backend/logs/`
- Enable debug mode: `DEBUG=true` in `.env`

---

**Last Updated:** 2025-10-30
