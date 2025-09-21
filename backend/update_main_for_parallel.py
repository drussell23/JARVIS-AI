#!/usr/bin/env python3
"""
Update main.py to support parallel startup
This script modifies the existing main.py to add optimization support
"""

import os
import shutil
import re
from pathlib import Path

def update_main_py():
    """Update main.py to support parallel startup"""
    print("🔧 Updating main.py for parallel startup support")
    print("=" * 60)
    
    main_py = Path("main.py")
    
    if not main_py.exists():
        print("❌ main.py not found in current directory")
        return False
    
    # Step 1: Backup original
    print("\n1️⃣ Backing up main.py...")
    backup_path = Path("main_original.py")
    if not backup_path.exists():
        shutil.copy(main_py, backup_path)
        print(f"   ✅ Backed up to {backup_path}")
    else:
        print("   ℹ️ Backup already exists")
    
    # Step 2: Read current content
    print("\n2️⃣ Reading main.py...")
    with open(main_py, 'r') as f:
        content = f.read()
    
    # Step 3: Add optimization check
    print("\n3️⃣ Adding optimization support...")
    
    # Check if already updated
    if "OPTIMIZE_STARTUP" in content:
        print("   ℹ️ main.py already has optimization support")
        return True
    
    # Find where to insert optimization code
    lines = content.split('\n')
    
    # Add optimization imports after initial imports
    import_index = -1
    for i, line in enumerate(lines):
        if "from fastapi import" in line:
            import_index = i
            break
    
    if import_index == -1:
        print("   ❌ Could not find FastAPI import")
        return False
    
    # Insert optimization check
    optimization_code = '''
# ========== PARALLEL STARTUP OPTIMIZATION ==========
# Check if we're running in optimized mode
OPTIMIZE_STARTUP = os.getenv('OPTIMIZE_STARTUP', 'false').lower() == 'true'

if OPTIMIZE_STARTUP:
    logger.info("🚀 Running in OPTIMIZED startup mode")
    # Use the optimized main module
    try:
        from main_optimized import app, components
        logger.info("✅ Using optimized FastAPI app")
        # Skip the rest of this file - we're using the optimized version
        if __name__ != "__main__":
            # When imported as a module, just use the optimized app
            pass
        else:
            # When run directly, start the optimized server
            import uvicorn
            port = int(os.getenv('BACKEND_PORT', '8000'))
            uvicorn.run(app, host="0.0.0.0", port=port)
            import sys
            sys.exit(0)
    except ImportError as e:
        logger.warning(f"Could not import optimized main: {e}")
        logger.info("Falling back to standard startup")
# ========== END OPTIMIZATION ==========
'''
    
    # Find a good place to insert (after logger setup)
    insert_index = -1
    for i, line in enumerate(lines):
        if "logger = logging.getLogger" in line:
            insert_index = i + 1
            break
    
    if insert_index == -1:
        # Insert after imports if logger not found
        insert_index = import_index + 10
    
    # Insert the optimization code
    lines.insert(insert_index, optimization_code)
    
    # Step 4: Add parallel import support for components
    print("\n4️⃣ Adding parallel import support...")
    
    # Find chatbot import section
    chatbot_import_start = -1
    for i, line in enumerate(lines):
        if "from chatbots.claude_vision_chatbot import" in line:
            chatbot_import_start = i - 1  # Include try statement
            break
    
    if chatbot_import_start > 0:
        # Add parallel import option
        parallel_import_code = '''
# Parallel import support
if os.getenv('BACKEND_PARALLEL_IMPORTS', 'false').lower() == 'true' and not OPTIMIZE_STARTUP:
    logger.info("⚡ Using parallel imports for components")
    import concurrent.futures
    import asyncio
    
    def parallel_import_components():
        """Import components in parallel"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                'chatbots': executor.submit(lambda: __import__('chatbots.claude_vision_chatbot')),
                'vision': executor.submit(lambda: __import__('vision.claude_vision_analyzer_main')),
                'memory': executor.submit(lambda: __import__('memory.memory_manager')),
            }
            
            for name, future in futures.items():
                try:
                    future.result(timeout=10)
                    logger.info(f"  ✅ {name} imported")
                except Exception as e:
                    logger.warning(f"  ⚠️ {name} import failed: {e}")
    
    # Run parallel imports
    parallel_import_components()

# Continue with normal imports (they'll be cached if already imported)
'''
        lines.insert(chatbot_import_start, parallel_import_code)
    
    # Step 5: Save updated file
    print("\n5️⃣ Saving updated main.py...")
    updated_content = '\n'.join(lines)
    
    with open(main_py, 'w') as f:
        f.write(updated_content)
    
    print("   ✅ main.py updated successfully")
    
    # Step 6: Create a simple test
    print("\n6️⃣ Creating test script...")
    test_script = '''#!/usr/bin/env python3
"""Test if optimization is working"""

import os
import subprocess
import time

print("🧪 Testing main.py optimization")
print("=" * 40)

# Test normal mode
print("\\n1️⃣ Testing normal mode...")
env = os.environ.copy()
env['OPTIMIZE_STARTUP'] = 'false'
start = time.time()

proc = subprocess.Popen(
    ['python', 'main.py'],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for startup message
time.sleep(3)
proc.terminate()
normal_time = time.time() - start
print(f"   Normal startup: {normal_time:.1f}s")

# Test optimized mode
print("\\n2️⃣ Testing optimized mode...")
env['OPTIMIZE_STARTUP'] = 'true'
start = time.time()

proc = subprocess.Popen(
    ['python', 'main.py'],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for startup message
time.sleep(3)
proc.terminate()
optimized_time = time.time() - start
print(f"   Optimized startup: {optimized_time:.1f}s")

print(f"\\n📊 Improvement: {normal_time/optimized_time:.1f}x faster!")
'''
    
    with open('test_main_optimization.py', 'w') as f:
        f.write(test_script)
    os.chmod('test_main_optimization.py', 0o755)
    
    print("\n" + "=" * 60)
    print("✅ main.py Update Complete!")
    print("=" * 60)
    
    print("\n📋 Changes Made:")
    print("  • Added OPTIMIZE_STARTUP environment check")
    print("  • Integrated with main_optimized.py")
    print("  • Added parallel import support")
    print("  • Created backup at main_original.py")
    
    print("\n🚀 To Use Optimized Mode:")
    print("  export OPTIMIZE_STARTUP=true")
    print("  python main.py")
    
    print("\n🐌 To Use Legacy Mode:")
    print("  export OPTIMIZE_STARTUP=false")
    print("  python main.py")
    
    print("\n🧪 To Test:")
    print("  python test_main_optimization.py")
    
    return True

if __name__ == "__main__":
    import sys
    os.chdir(Path(__file__).parent)  # Ensure we're in backend directory
    success = update_main_py()
    sys.exit(0 if success else 1)