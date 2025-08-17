#!/usr/bin/env python3
"""
Unified startup script for AI-Powered Chatbot System
Prioritizes Claude API for cloud-based AI, with fallback options
"""

import os
import sys
import subprocess
import time
import signal
import platform
import json
from pathlib import Path
import argparse
import webbrowser
from typing import List, Dict, Optional
import asyncio
import psutil

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    YELLOW = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class UnifiedSystemManager:
    """Unified system manager that prioritizes Claude but supports all modes"""
    
    def __init__(self):
        self.processes = []
        self.backend_dir = Path("backend")
        self.frontend_dir = Path("frontend")
        self.ports = {
            "main_api": 8000,
            "training_api": 8001,
            "frontend": 3000,
            "llama_cpp": 8080
        }
        self.demos = {
            "chat": "http://localhost:8000/docs",
            "voice": "http://localhost:8000/voice_demo.html",
            "automation": "http://localhost:8000/automation_demo.html",
            "rag": "http://localhost:8000/rag_demo.html",
            "training": "http://localhost:8001/llm_demo.html"
        }
        self.is_m1_mac = platform.system() == "Darwin" and platform.machine() == "arm64"
        self.memory_warned = False
        self.claude_configured = False
        self.use_local_models = False
        
    def print_header(self):
        """Print system header"""
        print(f"\n{Colors.HEADER}{'='*60}")
        print(f"{Colors.BOLD}🤖 JARVIS System - Claude AI Powered 🚀{Colors.ENDC}")
        print(f"{Colors.CYAN}☁️  Cloud-based AI for superior performance{Colors.ENDC}")
        if self.is_m1_mac:
            print(f"{Colors.GREEN}✨ Perfect for M1 Mac - No local memory usage!{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")
        
    def check_claude_config(self):
        """Check if Claude API is configured (required)"""
        # Load .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
            
        api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not api_key:
            print(f"{Colors.FAIL}❌ ANTHROPIC_API_KEY not found!{Colors.ENDC}")
            print(f"\n{Colors.YELLOW}To set up Claude (required):{Colors.ENDC}")
            print("1. Get an API key from: https://console.anthropic.com/")
            print("2. Create a .env file with:")
            print("   ANTHROPIC_API_KEY=your-api-key-here")
            print(f"\n{Colors.CYAN}Claude API provides:{Colors.ENDC}")
            print("  • Superior language understanding")
            print("  • Accurate calculations and reasoning")
            print("  • No local memory usage")
            print("  • 200k token context window")
            return False
            
        self.claude_configured = True
        return True
        
    def check_python_version(self):
        """Check Python version"""
        print(f"{Colors.BLUE}Checking Python version...{Colors.ENDC}")
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print(f"{Colors.FAIL}❌ Python 3.8+ required. Current: {version.major}.{version.minor}{Colors.ENDC}")
            sys.exit(1)
        print(f"{Colors.GREEN}✓ Python {version.major}.{version.minor} detected{Colors.ENDC}")
        
    def check_claude_setup(self):
        """Check Claude API setup and configuration"""
        if not self.claude_configured:
            return False
            
        print(f"\n{Colors.BLUE}Checking Claude API setup...{Colors.ENDC}")
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        print(f"{Colors.GREEN}✓ Claude API key found{Colors.ENDC}")
        print(f"{Colors.GREEN}✓ Claude mode enabled{Colors.ENDC}")
        
        # Check if anthropic package is installed
        try:
            import anthropic
            print(f"{Colors.GREEN}✓ Anthropic package installed{Colors.ENDC}")
            return True
        except ImportError:
            print(f"{Colors.WARNING}⚠️  Anthropic package not installed{Colors.ENDC}")
            print(f"   Run: pip install anthropic")
            return False
            
    def check_essential_dependencies(self):
        """Check only essential dependencies"""
        print(f"\n{Colors.BLUE}Checking essential dependencies...{Colors.ENDC}")
        
        essential_packages = {
            "fastapi": "FastAPI web framework",
            "uvicorn": "ASGI server",
            "pydantic": "Data validation",
            "psutil": "System monitoring",
            "anthropic": "Claude API client",
            "python-dotenv": "Environment variables"
        }
            
        missing = []
        for package, description in essential_packages.items():
            try:
                if package == "python-dotenv":
                    __import__("dotenv")
                else:
                    __import__(package.replace("-", "_"))
                print(f"{Colors.GREEN}✓ {description} ({package}){Colors.ENDC}")
            except ImportError:
                missing.append(package)
                print(f"{Colors.WARNING}⚠️  {description} ({package}) - missing{Colors.ENDC}")
                
        if missing:
            print(f"\n{Colors.YELLOW}Install missing packages:{Colors.ENDC}")
            print(f"pip install {' '.join(missing)}")
            return False
            
        return True
        
        
    def check_memory_status(self):
        """Check system memory (informational only)"""
        print(f"\n{Colors.BLUE}System memory status:{Colors.ENDC}")
        
        mem = psutil.virtual_memory()
        memory_percent = mem.percent
        available_gb = mem.available / (1024**3)
        total_gb = mem.total / (1024**3)
        
        print(f"  Total: {total_gb:.1f} GB")
        print(f"  Used: {mem.used / (1024**3):.1f} GB ({memory_percent:.1f}%)")
        print(f"  Available: {available_gb:.1f} GB")
        
        print(f"\n{Colors.GREEN}✓ With Claude AI, memory usage is not a concern!{Colors.ENDC}")
        print(f"  All AI processing happens in the cloud")
        return True
            
    def create_directories(self):
        """Create necessary directories"""
        print(f"\n{Colors.BLUE}Creating directories...{Colors.ENDC}")
        
        directories = [
            self.backend_dir / "logs",
            self.backend_dir / "static",
            self.backend_dir / "static" / "demos"  # For demo HTML files
        ]
            
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        print(f"{Colors.GREEN}✓ Directories created{Colors.ENDC}")
        
    def check_port_available(self, port: int) -> bool:
        """Check if a port is available"""
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result != 0
        
    def start_backend_services(self):
        """Start backend services"""
        print(f"\n{Colors.BLUE}Starting backend services...{Colors.ENDC}")
        
        # Check main API port
        if not self.check_port_available(self.ports["main_api"]):
            print(f"{Colors.WARNING}⚠️  Port {self.ports['main_api']} is already in use{Colors.ENDC}")
            print(f"{Colors.CYAN}Attempting to kill existing process...{Colors.ENDC}")
            try:
                if platform.system() == "Darwin":
                    subprocess.run(f"lsof -ti:{self.ports['main_api']} | xargs kill -9", 
                                 shell=True, capture_output=True)
                else:
                    subprocess.run(f"fuser -k {self.ports['main_api']}/tcp", 
                                 shell=True, capture_output=True)
                time.sleep(1)
            except:
                pass
                
        # Set environment variables
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["USE_CLAUDE"] = "1"
        
        print(f"{Colors.CYAN}Starting Claude-powered API on port {self.ports['main_api']}...{Colors.ENDC}")
            
        # Start main API
        main_api_process = subprocess.Popen(
            [sys.executable, "run_server.py"],
            cwd=self.backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env
        )
        self.processes.append(main_api_process)
        
        # Wait for service to start
        print(f"{Colors.CYAN}Waiting for API to initialize...{Colors.ENDC}")
        time.sleep(5)
        
        # Check if service is running
        if main_api_process.poll() is not None:
            output = main_api_process.stdout.read().decode('utf-8') if main_api_process.stdout else ""
            print(f"{Colors.FAIL}❌ Failed to start API{Colors.ENDC}")
            if output:
                print(f"{Colors.FAIL}Error output:{Colors.ENDC}")
                print(output[:500])
            self.cleanup()
            sys.exit(1)
            
        print(f"{Colors.GREEN}✓ Backend API started{Colors.ENDC}")
            
    def start_training_api(self):
        """Start training API for local models"""
        print(f"{Colors.CYAN}Starting training API on port {self.ports['training_api']}...{Colors.ENDC}")
        try:
            if (self.backend_dir / "training_interface.py").exists():
                training_api_process = subprocess.Popen(
                    [sys.executable, "training_interface.py"],
                    cwd=self.backend_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy()
                )
                self.processes.append(training_api_process)
                
                time.sleep(2)
                
                if training_api_process.poll() is not None:
                    print(f"{Colors.WARNING}⚠️  Training API skipped (not available){Colors.ENDC}")
                    self.processes.remove(training_api_process)
            else:
                print(f"{Colors.WARNING}⚠️  Training API not found{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.WARNING}⚠️  Failed to start training API: {e}{Colors.ENDC}")
            
    def start_frontend(self):
        """Start frontend service if needed"""
        if not self.frontend_dir.exists():
            return
            
        print(f"\n{Colors.BLUE}Starting frontend...{Colors.ENDC}")
        
        # Check if port is available
        if not self.check_port_available(self.ports['frontend']):
            print(f"{Colors.WARNING}⚠️  Port {self.ports['frontend']} is already in use{Colors.ENDC}")
            return
            
        # Check if it's a React app with package.json
        if (self.frontend_dir / "package.json").exists():
            # Check if node_modules exists
            if not (self.frontend_dir / "node_modules").exists():
                print(f"{Colors.WARNING}⚠️  Frontend dependencies not installed. Run: cd frontend && npm install{Colors.ENDC}")
                return
                
            print(f"{Colors.CYAN}Starting JARVIS React Interface...{Colors.ENDC}")
            
            # Set environment to use port 3000
            env = os.environ.copy()
            env["PORT"] = str(self.ports['frontend'])
            env["BROWSER"] = "none"  # Don't auto-open browser from React
            
            # Start React development server
            frontend_process = subprocess.Popen(
                ["npm", "start"],
                cwd=self.frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env
            )
            self.processes.append(frontend_process)
            print(f"{Colors.GREEN}✓ JARVIS Interface starting on port {self.ports['frontend']}...{Colors.ENDC}")
            print(f"{Colors.CYAN}   Note: React may take 10-15 seconds to compile{Colors.ENDC}")
        elif (self.frontend_dir / "index.html").exists():
            # Fallback to simple HTTP server
            frontend_process = subprocess.Popen(
                [sys.executable, "-m", "http.server", str(self.ports['frontend'])],
                cwd=self.frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.processes.append(frontend_process)
            print(f"{Colors.GREEN}✓ Frontend started on port {self.ports['frontend']}{Colors.ENDC}")
            
    def print_access_info(self):
        """Print access information"""
        print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.BOLD}🎉 System is ready!{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")
        
        print(f"{Colors.CYAN}Main Services:{Colors.ENDC}")
        print(f"  🔌 API Documentation: http://localhost:{self.ports['main_api']}/docs")
        print(f"  💬 Basic Chat:        http://localhost:{self.ports['main_api']}/")
        
        if self.frontend_dir.exists() and (self.frontend_dir / "package.json").exists():
            print(f"  🎯 JARVIS Interface:  http://localhost:{self.ports['frontend']}/ {Colors.GREEN}← Iron Man UI{Colors.ENDC}")
        elif self.frontend_dir.exists():
            print(f"  📱 Frontend:          http://localhost:{self.ports['frontend']}")
            
        print(f"\n{Colors.GREEN}✨ Powered by Claude AI:{Colors.ENDC}")
        print(f"  • Superior language understanding")
        print(f"  • Accurate calculations and reasoning") 
        print(f"  • No local memory usage")
        print(f"  • 200k token context window")
        print(f"  • Fast cloud-based responses")
                
        print(f"\n{Colors.WARNING}Press Ctrl+C to stop all services{Colors.ENDC}")
        
    def cleanup(self):
        """Cleanup processes on exit"""
        print(f"\n{Colors.BLUE}Shutting down services...{Colors.ENDC}")
        for proc in self.processes:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            except Exception as e:
                print(f"{Colors.WARNING}Warning during cleanup: {e}{Colors.ENDC}")
        print(f"{Colors.GREEN}✓ All services stopped{Colors.ENDC}")
        
    def handle_signal(self, signum, frame):
        """Handle interrupt signal"""
        self.cleanup()
        sys.exit(0)
        
    def run(self, skip_install: bool = False, open_browser: bool = True):
        """Run the system"""
        # Setup signal handler
        signal.signal(signal.SIGINT, self.handle_signal)
        
        # Print header
        self.print_header()
        
        # Check Python version
        self.check_python_version()
        
        # Check Claude configuration (required)
        if not self.check_claude_config():
            print(f"\n{Colors.FAIL}❌ Claude API configuration required. Please set up your API key.{Colors.ENDC}")
            sys.exit(1)
            
        # Check dependencies
        if not skip_install:
            # Check Claude setup
            if not self.check_claude_setup():
                print(f"\n{Colors.FAIL}❌ Claude setup incomplete. Please install required packages.{Colors.ENDC}")
                sys.exit(1)
                    
            # Check essential dependencies
            if not self.check_essential_dependencies():
                print(f"\n{Colors.FAIL}❌ Missing essential dependencies. Please install required packages.{Colors.ENDC}")
                sys.exit(1)
        else:
            print(f"{Colors.WARNING}⚠️  Skipping dependency installation{Colors.ENDC}")
            
        # Check memory status (informational only for Claude)
        self.check_memory_status()
            
        # Create directories
        self.create_directories()
        
        # Start services
        self.start_backend_services()
        self.start_frontend()
        
        # Print access info
        self.print_access_info()
        
        # Open browser if requested
        if open_browser:
            time.sleep(2)
            # Prefer JARVIS interface if available
            if self.frontend_dir.exists() and (self.frontend_dir / "package.json").exists():
                print(f"{Colors.CYAN}Opening JARVIS Interface in browser...{Colors.ENDC}")
                time.sleep(8)  # Give React more time to compile
                webbrowser.open(f"http://localhost:{self.ports['frontend']}/")
            else:
                webbrowser.open(f"http://localhost:{self.ports['main_api']}/docs")
            
        # Keep running
        try:
            while True:
                time.sleep(1)
                # Check if processes are still running
                for i, proc in enumerate(self.processes):
                    if proc.poll() is not None:
                        service_names = ["Main API", "Training API", "Frontend"]
                        service_name = service_names[i] if i < len(service_names) else f"Service {i}"
                        
                        output = proc.stdout.read().decode('utf-8') if proc.stdout else ""
                        print(f"{Colors.FAIL}❌ {service_name} stopped unexpectedly{Colors.ENDC}")
                        if output:
                            print(f"{Colors.FAIL}Error:{Colors.ENDC}")
                            print(output[:500])
                            
        except KeyboardInterrupt:
            pass


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="JARVIS System Launcher - Claude AI Powered")
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip dependency installation"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check setup and exit"
    )
    
    args = parser.parse_args()
    
    # Create and run system manager
    manager = UnifiedSystemManager()
    
    if args.check_only:
        manager.print_header()
        manager.check_python_version()
        
        if not manager.check_claude_config():
            print(f"\n{Colors.FAIL}❌ Claude API not configured.{Colors.ENDC}")
            sys.exit(1)
            
        all_good = manager.check_claude_setup() and manager.check_essential_dependencies()
        
        if all_good:
            print(f"\n{Colors.GREEN}✅ System ready! Claude API configured.{Colors.ENDC}")
        else:
            print(f"\n{Colors.FAIL}❌ Some checks failed. Please fix issues above.{Colors.ENDC}")
            
        sys.exit(0 if all_good else 1)
        
    try:
        manager.run(
            skip_install=args.skip_install,
            open_browser=not args.no_browser
        )
    except Exception as e:
        print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
        manager.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()