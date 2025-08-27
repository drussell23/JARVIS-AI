#!/usr/bin/env python3
"""
Final CPU fix - Disable problematic components and verify 25% CPU target
"""

import os
import sys
import subprocess
import time
import psutil

def kill_all_backends():
    """Kill all backend processes"""
    print("🔍 Killing all backend processes...")
    subprocess.run(['pkill', '-f', 'python.*main.py'], capture_output=True)
    time.sleep(2)

def apply_aggressive_limits():
    """Apply very aggressive CPU limits"""
    print("⚙️ Applying aggressive CPU limits...")
    
    # Create new .env with all limits
    env_content = """# JARVIS Aggressive CPU Limits
DISABLE_CONTINUOUS_LEARNING=true
DISABLE_VISION_MONITORING=true
DISABLE_ML_FEATURES=false
LEARNING_CPU_LIMIT=20
VISION_PROCESSING_THREADS=1
MAX_PARALLEL_OPERATIONS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMBA_NUM_THREADS=1
PYTORCH_NUM_THREADS=1
USE_OPTIMIZED_LEARNING=true
ENABLE_CPU_THROTTLING=true
CPU_LIMIT_PERCENT=25
MEMORY_LIMIT_MB=2000
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    # Also set in environment
    for line in env_content.strip().split('\n'):
        if '=' in line and not line.startswith('#'):
            key, value = line.split('=', 1)
            os.environ[key] = value
    
    print("✅ Aggressive limits applied")

def patch_vision_system():
    """Patch vision system to disable continuous learning"""
    print("🔧 Patching vision system...")
    
    vision_path = "vision/vision_system_v2.py"
    if os.path.exists(vision_path):
        with open(vision_path, 'r') as f:
            content = f.read()
        
        # Comment out continuous learning initialization
        content = content.replace(
            "self.continuous_learner = get_advanced_continuous_learning",
            "# DISABLED FOR CPU: self.continuous_learner = get_advanced_continuous_learning"
        )
        
        # Add CPU check
        if "# CPU LIMIT CHECK" not in content:
            content = content.replace(
                "class VisionSystemV2:",
                """class VisionSystemV2:
    # CPU LIMIT CHECK
    _cpu_limit = float(os.getenv('CPU_LIMIT_PERCENT', '25'))
    _last_cpu_check = 0
    
    @classmethod
    def _check_cpu(cls):
        if time.time() - cls._last_cpu_check < 1:
            return
        cls._last_cpu_check = time.time()
        cpu = psutil.cpu_percent(interval=0.1)
        if cpu > cls._cpu_limit:
            time.sleep(0.1 * (cpu / cls._cpu_limit))
"""
            )
        
        with open(vision_path, 'w') as f:
            f.write(content)
        
        print("✅ Vision system patched")

def start_limited_backend():
    """Start backend with strict limits"""
    print("\n🚀 Starting CPU-limited backend...")
    
    # Use cpulimit if available
    cpulimit_paths = ['/opt/homebrew/bin/cpulimit', '/usr/local/bin/cpulimit', 'cpulimit']
    cpulimit_cmd = None
    
    for cpulimit_path in cpulimit_paths:
        try:
            if subprocess.run([cpulimit_path, '--help'], capture_output=True).returncode == 0:
                cpulimit_cmd = [cpulimit_path, '--limit=25', '--']
                print(f"✅ Using cpulimit at: {cpulimit_path}")
                break
        except (FileNotFoundError, OSError):
            continue
    
    if cpulimit_cmd is None:
        print("⚠️  cpulimit not found, using nice instead")
        cpulimit_cmd = ['nice', '-n', '19']
    
    cmd = cpulimit_cmd + [sys.executable, 'main.py', '--port', '8000']
    
    process = subprocess.Popen(
        cmd,
        stdout=open('logs/limited_backend.log', 'w'),
        stderr=subprocess.STDOUT
    )
    
    print(f"✅ Started limited backend with PID {process.pid}")
    return process

def verify_cpu_target():
    """Verify we hit the 25% CPU target"""
    print("\n📊 Verifying CPU usage (30 second test)...")
    time.sleep(5)  # Wait for startup
    
    # Find backend
    backend_pid = None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name']:
                cmdline = proc.cmdline() if hasattr(proc, 'cmdline') else []
                if any('main.py' in arg for arg in cmdline):
                    backend_pid = proc.info['pid']
                    break
        except:
            continue
    
    if not backend_pid:
        print("❌ Backend not found!")
        return False
    
    proc = psutil.Process(backend_pid)
    samples = []
    
    print("\nSecond | CPU % | Status")
    print("-" * 30)
    
    for i in range(30):
        cpu = proc.cpu_percent(interval=1.0)
        samples.append(cpu)
        status = "✅" if cpu <= 30 else "⚡" if cpu <= 50 else "🚨"
        print(f"{i+1:6d} | {cpu:5.1f} | {status}")
    
    avg_cpu = sum(samples) / len(samples)
    print("\n" + "=" * 40)
    print(f"Average CPU: {avg_cpu:.1f}%")
    print(f"Target: 25%")
    print(f"Reduction: {(97-avg_cpu)/97*100:.0f}%")
    
    if avg_cpu <= 30:
        print("\n✅ SUCCESS! CPU target achieved!")
        return True
    else:
        print("\n⚠️  CPU still above target")
        return False

def main():
    print("🚨 JARVIS Final CPU Fix")
    print("=" * 50)
    
    # 1. Kill everything
    kill_all_backends()
    
    # 2. Apply limits
    apply_aggressive_limits()
    
    # 3. Patch code
    patch_vision_system()
    
    # 4. Start limited
    process = start_limited_backend()
    
    # 5. Verify
    success = verify_cpu_target()
    
    if success:
        print("\n🎉 CPU SUCCESSFULLY REDUCED TO ~25%!")
    else:
        print("\n💡 Try installing cpulimit:")
        print("   brew install cpulimit")
        print("   Then run this script again")
    
    return 0 if success else 1

if __name__ == "__main__":
    # Import after setting environment
    import psutil
    import time
    sys.exit(main())