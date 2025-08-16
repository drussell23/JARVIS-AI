# 🚀 JARVIS Quick Start Guide - FIXED!

## The Problem
The `start_system.py` script is failing because of outdated dependencies in `backend/requirements.txt`. Here's how to bypass it and get JARVIS running.

## Quick Solution (3 Steps)

### Step 1: Install Core Dependencies
```bash
# Install just what you need (skip the problematic ones)
python3 -m pip install fastapi uvicorn pydantic python-multipart \
    websockets aiohttp requests python-dotenv psutil objgraph \
    pympler langchain langchain-community spacy transformers \
    torch numpy scipy
```

### Step 2: Start JARVIS
```bash
# Option A: Use the shell script
./run_jarvis.sh

# Option B: Start manually
cd backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 3: Access JARVIS
Open your browser to:
- API Docs: http://localhost:8000/docs
- Chat Demo: http://localhost:8000/demo/chat
- Voice Demo: http://localhost:8000/demo/voice

## Alternative: Skip All Dependencies
If you're still having issues, run in minimal mode:

```bash
# Set environment to disable features
export SKIP_NLP=1
export SKIP_VOICE=1
export SKIP_LANGCHAIN=1

# Run with minimal dependencies
cd backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## Common Issues & Fixes

### "sentencepiece failed to install"
```bash
# Install without version pin
python3 -m pip install sentencepiece
```

### "torch version not found"
```bash
# Install latest torch
python3 -m pip install torch
```

### "llama-cpp-python build failed"
```bash
# Skip it - not required for basic operation
# Or install pre-built wheel:
python3 -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### Port 8000 already in use
```bash
# Kill the process
lsof -ti:8000 | xargs kill -9

# Or use different port
python3 -m uvicorn main:app --port 8001
```

## Working Startup Command
Here's a one-liner that should work:

```bash
cd backend && python3 -m pip install fastapi uvicorn psutil && python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## Still Having Issues?
1. Check Python version: `python3 --version` (should be 3.8+)
2. Update pip: `python3 -m pip install --upgrade pip`
3. Use virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install fastapi uvicorn psutil
   cd backend && python -m uvicorn main:app
   ```

## Success Indicators
You'll know JARVIS is running when you see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [xxxxx] using WatchFiles
```

Then open http://localhost:8000/docs in your browser! 🎉