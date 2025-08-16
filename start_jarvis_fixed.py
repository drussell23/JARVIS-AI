#!/usr/bin/env python3
"""
Fixed JARVIS startup script that handles all dependencies and import issues
"""

import os
import sys
import subprocess
import time
import webbrowser
import signal

class JARVISStarter:
    def __init__(self):
        self.process = None
        
    def check_dependencies(self):
        """Check and install missing dependencies"""
        print("🔍 Checking dependencies...")
        
        required_packages = [
            "psutil", "fastapi", "uvicorn", "aiohttp", 
            "pydantic", "python-multipart", "websockets",
            "langchain", "langchain-community", "spacy",
            "transformers", "torch", "icalendar", "apscheduler",
            "objgraph", "pympler", "requests", "python-dotenv"
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing.append(package)
        
        if missing:
            print(f"📦 Installing missing packages: {', '.join(missing)}")
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        else:
            print("✅ All dependencies installed")
    
    def start_backend(self):
        """Start the JARVIS backend"""
        print("\n🚀 Starting JARVIS backend...")
        
        # Set environment variables
        os.environ["USE_QUANTIZED_MODELS"] = "true"
        os.environ["PREFER_LANGCHAIN"] = "0"  # Start without LangChain
        os.environ["PYTHONPATH"] = os.path.abspath("backend")
        
        # Change to backend directory
        os.chdir("backend")
        
        # Start uvicorn
        cmd = [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
        
        print(f"Running: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Monitor startup
        print("⏳ Waiting for server to start...")
        start_time = time.time()
        server_started = False
        
        while time.time() - start_time < 30:  # 30 second timeout
            # Check if process is still running
            if self.process.poll() is not None:
                # Process died, show error
                stdout, stderr = self.process.communicate()
                print("\n❌ Server failed to start!")
                print("Error output:")
                print(stderr.decode())
                return False
            
            # Try to connect
            try:
                import requests
                response = requests.get("http://localhost:8000/docs", timeout=1)
                if response.status_code == 200:
                    server_started = True
                    break
            except:
                pass
            
            time.sleep(1)
            print(".", end="", flush=True)
        
        print()
        
        if server_started:
            print("✅ JARVIS backend started successfully!")
            return True
        else:
            print("❌ Server startup timeout")
            return False
    
    def open_browser(self):
        """Open JARVIS in browser"""
        print("\n🌐 Opening JARVIS in browser...")
        webbrowser.open("http://localhost:8000/docs")
        
        print("\n" + "="*50)
        print("✅ JARVIS is running!")
        print("="*50)
        print("\n📋 Access points:")
        print("   - API Docs: http://localhost:8000/docs")
        print("   - Chat Demo: http://localhost:8000/demo/chat")
        print("   - Voice Demo: http://localhost:8000/demo/voice")
        print("   - Memory Dashboard: http://localhost:8000/demo/memory")
        print("\n💡 Tips:")
        print("   - Run 'python jarvis_quick_fix.py' to set up optimized models")
        print("   - Run 'python optimize_memory_advanced.py' to free up memory")
        print("\n⚠️  Press Ctrl+C to stop JARVIS")
    
    def run(self):
        """Main run method"""
        print("""
╔══════════════════════════════════════════════════╗
║             🤖 JARVIS AI System                  ║
║         Fixed Startup Script v1.0                ║
╚══════════════════════════════════════════════════╝
        """)
        
        # Check dependencies
        self.check_dependencies()
        
        # Start backend
        if self.start_backend():
            # Open browser
            self.open_browser()
            
            # Keep running
            try:
                self.process.wait()
            except KeyboardInterrupt:
                self.stop()
        else:
            print("\n❌ Failed to start JARVIS")
            print("💡 Try running: python backend/main.py")
    
    def stop(self):
        """Stop JARVIS gracefully"""
        print("\n\n👋 Stopping JARVIS...")
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        print("✅ JARVIS stopped successfully")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\nReceived interrupt signal...")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    starter = JARVISStarter()
    starter.run()