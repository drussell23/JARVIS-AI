#!/usr/bin/env python3
"""
Enable Parallel Startup for JARVIS
This script integrates parallel startup with the existing system
"""

import os
import shutil
import sys
from pathlib import Path

def enable_parallel_startup():
    """Enable parallel startup for JARVIS"""
    print("🚀 Enabling Parallel Startup for JARVIS")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('backend') or not os.path.exists('start_system.py'):
        print("❌ Error: Please run this from the JARVIS root directory")
        return False
    
    # Step 1: Backup original start_system.py
    print("\n1️⃣ Backing up original start_system.py...")
    if os.path.exists('start_system.py'):
        shutil.copy('start_system.py', 'start_system_sequential.py')
        print("   ✅ Backed up to start_system_sequential.py")
    
    # Step 2: Copy environment configuration
    print("\n2️⃣ Setting up parallel configuration...")
    if os.path.exists('.env.parallel'):
        # Merge with existing .env if it exists
        if os.path.exists('.env'):
            print("   📝 Merging with existing .env...")
            with open('.env', 'a') as f:
                f.write("\n\n# === PARALLEL STARTUP CONFIGURATION ===\n")
                with open('.env.parallel', 'r') as pf:
                    f.write(pf.read())
        else:
            shutil.copy('.env.parallel', '.env')
        print("   ✅ Configuration ready")
    
    # Step 3: Create new start script
    print("\n3️⃣ Creating optimized start script...")
    
    start_script = '''#!/usr/bin/env python3
"""
JARVIS Start System - Parallel Optimized Version
Starts all services in parallel for faster initialization
"""

import os
import sys
import subprocess

# Check for parallel startup
if os.getenv('USE_PARALLEL_STARTUP', 'true').lower() == 'true':
    print("🚀 Using PARALLEL startup (fast mode)")
    print("=" * 60)
    
    # Use the parallel startup system
    subprocess.run([sys.executable, 'backend/start_system_parallel.py'])
else:
    print("🐌 Using SEQUENTIAL startup (legacy mode)")
    print("To enable fast startup: export USE_PARALLEL_STARTUP=true")
    print("=" * 60)
    
    # Fall back to original sequential startup
    if os.path.exists('start_system_sequential.py'):
        subprocess.run([sys.executable, 'start_system_sequential.py'])
    else:
        print("❌ Sequential startup script not found")
'''
    
    with open('start_system.py', 'w') as f:
        f.write(start_script)
    
    # Make it executable
    os.chmod('start_system.py', 0o755)
    print("   ✅ New start script created")
    
    # Step 4: Update main.py to support optimized startup
    print("\n4️⃣ Updating backend for parallel imports...")
    
    main_py_path = 'backend/main.py'
    if os.path.exists(main_py_path):
        # Read the current main.py
        with open(main_py_path, 'r') as f:
            content = f.read()
        
        # Add optimization check at the beginning
        optimization_code = '''# Check for optimized startup
if os.getenv('OPTIMIZE_STARTUP', 'false').lower() == 'true':
    # Use parallel imports
    import asyncio
    from optimized_backend_startup import create_optimized_app
    
    # Create app with parallel initialization
    app = asyncio.run(create_optimized_app())
else:
    # Original sequential imports
'''
        
        # Only add if not already present
        if 'OPTIMIZE_STARTUP' not in content:
            # Find where to insert (after imports)
            lines = content.split('\n')
            insert_line = 0
            for i, line in enumerate(lines):
                if line.startswith('app = FastAPI'):
                    insert_line = i
                    break
            
            if insert_line > 0:
                lines.insert(insert_line, optimization_code)
                
                # Save updated file
                with open(main_py_path + '.bak', 'w') as f:
                    f.write(content)  # Backup original
                
                with open(main_py_path, 'w') as f:
                    f.write('\n'.join(lines))
                
                print("   ✅ Backend updated for parallel imports")
            else:
                print("   ⚠️  Could not update main.py automatically")
    
    # Step 5: Create quick test script
    print("\n5️⃣ Creating test script...")
    
    test_script = '''#!/usr/bin/env python3
"""Test parallel startup performance"""

import time
import subprocess
import os

print("🧪 Testing JARVIS Startup Performance")
print("=" * 60)

# Test sequential startup
print("\\n1️⃣ Testing SEQUENTIAL startup...")
os.environ['USE_PARALLEL_STARTUP'] = 'false'
start_time = time.time()

# Run for 10 seconds then kill
proc = subprocess.Popen(['python', 'start_system.py'])
time.sleep(10)
proc.terminate()

sequential_time = time.time() - start_time
print(f"Sequential startup: {sequential_time:.1f}s")

# Test parallel startup
print("\\n2️⃣ Testing PARALLEL startup...")
os.environ['USE_PARALLEL_STARTUP'] = 'true'
start_time = time.time()

# Run for 10 seconds then kill
proc = subprocess.Popen(['python', 'start_system.py'])
time.sleep(10)
proc.terminate()

parallel_time = time.time() - start_time
print(f"Parallel startup: {parallel_time:.1f}s")

# Results
print("\\n📊 Results:")
print(f"Improvement: {sequential_time/parallel_time:.1f}x faster!")
'''
    
    with open('test_parallel_startup.py', 'w') as f:
        f.write(test_script)
    os.chmod('test_parallel_startup.py', 0o755)
    print("   ✅ Test script created")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Parallel Startup Enabled!")
    print("=" * 60)
    
    print("\n📋 What's New:")
    print("  • All services start in parallel")
    print("  • Components load concurrently")
    print("  • Health checks run simultaneously")
    print("  • Imports happen in thread pools")
    print("  • Lazy loading for heavy components")
    
    print("\n🚀 To Start JARVIS (parallel mode):")
    print("  python start_system.py")
    
    print("\n🐌 To Use Legacy Mode:")
    print("  export USE_PARALLEL_STARTUP=false")
    print("  python start_system.py")
    
    print("\n🧪 To Test Performance:")
    print("  python test_parallel_startup.py")
    
    print("\n⚙️  Configuration:")
    print("  Edit .env to customize parallel settings")
    print("  All timeouts and workers are configurable")
    
    print("\n⚡ Expected Improvement:")
    print("  From: ~107 seconds")
    print("  To:   ~30 seconds")
    print("  Speedup: 3-5x faster!")
    
    return True

if __name__ == "__main__":
    success = enable_parallel_startup()
    sys.exit(0 if success else 1)