# start_system.py Improvements

## Overview
The `start_system.py` script has been enhanced to be smarter about dependency management, reducing startup time and avoiding unnecessary reinstallations.

## New Features

### 1. Smart Dependency Checking
- **Only installs missing packages** - The script now checks which packages are already installed and only installs what's missing
- **Shows progress** - Displays `[1/10] Installing: package-name` format for better visibility
- **Faster startup** - Skips installation of already-present dependencies

### 2. New Command Line Options

#### `--check-deps`
Check what dependencies are installed without starting the system:
```bash
python start_system.py --check-deps
```

This shows:
- Total installed packages
- Missing dependencies (if any)
- NLTK data status
- spaCy model status
- LangChain installation status
- llama-cpp-python status
- M1 Mac specific checks (llama.cpp and models)

#### `--skip-install`
Skip all dependency checks and installations:
```bash
python start_system.py --skip-install
```

#### `--async-mode`
Use asynchronous installation for better performance:
```bash
python start_system.py --async-mode
```

### 3. Improved Installation Process

**Before:**
- Reinstalled all dependencies every time
- Could take 5-10 minutes on each startup
- No visibility into what's being installed

**After:**
- Only installs missing dependencies
- Shows progress with `[1/5] Installing: package-name`
- Typically takes < 30 seconds if most deps are installed
- Checks NLTK data and spaCy models separately

### 4. Async Mode Improvements
- Installs dependencies in parallel (batches of 5)
- Significantly faster for multiple missing dependencies
- Better performance on multi-core systems

## Usage Examples

### Quick Start (Skip Installation)
```bash
# If you know everything is installed
python start_system.py --skip-install
```

### Check Dependencies Only
```bash
# See what's installed/missing without starting
python start_system.py --check-deps
```

### Normal Start with Smart Install
```bash
# Will only install what's missing
python start_system.py
```

### Fast Async Mode
```bash
# Use parallel installation
python start_system.py --async-mode
```

### Backend Only, No Browser
```bash
python start_system.py --backend-only --no-browser
```

## Benefits

1. **Faster Startup** - Typically 10x faster when dependencies are already installed
2. **Better Visibility** - See exactly what's being installed with progress indicators
3. **Smarter Logic** - Only downloads what's actually missing
4. **M1 Optimized** - Special checks for M1 Macs and llama.cpp
5. **Debugging Aid** - `--check-deps` helps diagnose installation issues

## Technical Details

The script now:
1. Runs `pip list --format=json` to get installed packages
2. Parses `requirements.txt` to extract package names
3. Compares to find missing dependencies
4. Only installs what's missing
5. Checks NLTK data and spaCy models separately
6. Provides detailed feedback throughout the process

This makes the development experience much smoother, especially when restarting the system frequently during development!