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
        print(f"{Colors.BOLD}🤖 AI-Powered Chatbot System Launcher 🚀{Colors.ENDC}")
        
        # Check Claude configuration
        self.check_claude_config()
        
        if self.claude_configured:
            print(f"{Colors.CYAN}☁️  Claude API Mode - Cloud-powered AI{Colors.ENDC}")
            if self.is_m1_mac:
                print(f"{Colors.GREEN}✨ Perfect for M1 Mac - No local memory usage!{Colors.ENDC}")
        else:
            print(f"{Colors.YELLOW}💻 Local Model Mode{Colors.ENDC}")
            
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")
        
    def check_claude_config(self):
        """Check if Claude API is configured"""
        # Load .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
            
        api_key = os.getenv("ANTHROPIC_API_KEY")
        use_claude = os.getenv("USE_CLAUDE", "1") == "1"
        
        self.claude_configured = bool(api_key and use_claude)
        return self.claude_configured
        
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
            "psutil": "System monitoring"
        }
        
        # Add Claude-specific if configured
        if self.claude_configured:
            essential_packages["anthropic"] = "Claude API client"
            essential_packages["python-dotenv"] = "Environment variables"
            
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
        
    def check_optional_dependencies(self):
        """Check optional dependencies for local models"""
        if self.claude_configured and not self.use_local_models:
            return True
            
        print(f"\n{Colors.BLUE}Checking optional dependencies for local models...{Colors.ENDC}")
        
        optional_packages = {
            "transformers": "Hugging Face models",
            "torch": "PyTorch deep learning",
            "nltk": "Natural language toolkit",
            "spacy": "Advanced NLP"
        }
        
        available = []
        missing = []
        
        for package, description in optional_packages.items():
            try:
                __import__(package)
                available.append(package)
                print(f"{Colors.GREEN}✓ {description} ({package}){Colors.ENDC}")
            except ImportError:
                missing.append(package)
                print(f"{Colors.YELLOW}○ {description} ({package}) - not installed{Colors.ENDC}")
                
        if missing and not self.claude_configured:
            print(f"\n{Colors.WARNING}Some features may be limited without:{Colors.ENDC}")
            for pkg in missing[:3]:
                print(f"  - {pkg}")
                
        return True
        
    def check_memory_status(self):
        """Check system memory and provide recommendations"""
        print(f"\n{Colors.BLUE}System memory status:{Colors.ENDC}")
        
        mem = psutil.virtual_memory()
        memory_percent = mem.percent
        available_gb = mem.available / (1024**3)
        total_gb = mem.total / (1024**3)
        
        print(f"  Total: {total_gb:.1f} GB")
        print(f"  Used: {mem.used / (1024**3):.1f} GB ({memory_percent:.1f}%)")
        print(f"  Available: {available_gb:.1f} GB")
        
        if self.claude_configured:
            print(f"\n{Colors.GREEN}✓ With Claude, memory usage is not a concern!{Colors.ENDC}")
            print(f"  All AI processing happens in the cloud")
            return True
            
        # Memory recommendations for local models
        if memory_percent < 50:
            print(f"{Colors.GREEN}✓ Memory OK for all features{Colors.ENDC}")
            return True
        elif memory_percent < 65:
            print(f"{Colors.WARNING}⚠️  Memory OK for basic features{Colors.ENDC}")
            self.memory_warned = True
            return True
        elif memory_percent < 80:
            print(f"{Colors.WARNING}⚠️  Memory high - limited features available{Colors.ENDC}")
            self.memory_warned = True
            
            if not self.claude_configured:
                print(f"\n{Colors.CYAN}💡 Tip: Use Claude API for better performance!{Colors.ENDC}")
                print(f"   1. Get API key: https://console.anthropic.com/")
                print(f"   2. Set ANTHROPIC_API_KEY in .env file")
            return True
        else:
            print(f"{Colors.FAIL}❌ Memory critical ({memory_percent:.1f}%){Colors.ENDC}")
            self.memory_warned = True
            
            if not self.claude_configured:
                print(f"\n{Colors.YELLOW}⚠️  Consider using Claude API instead of local models{Colors.ENDC}")
            return False
            
    def create_directories(self):
        """Create necessary directories"""
        print(f"\n{Colors.BLUE}Creating directories...{Colors.ENDC}")
        
        directories = [
            self.backend_dir / "logs",
            self.backend_dir / "static"
        ]
        
        # Only create model/data directories if using local models
        if not self.claude_configured or self.use_local_models:
            directories.extend([
                self.backend_dir / "data",
                self.backend_dir / "models",
                self.backend_dir / "checkpoints",
                self.backend_dir / "domain_knowledge",
                self.backend_dir / "faiss_index",
                self.backend_dir / "chroma_db",
                self.backend_dir / "knowledge_base"
            ])
            
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
        
        if self.claude_configured:
            env["USE_CLAUDE"] = "1"
            print(f"{Colors.CYAN}Starting Claude-powered API on port {self.ports['main_api']}...{Colors.ENDC}")
        else:
            env["USE_CLAUDE"] = "0"
            print(f"{Colors.CYAN}Starting API with local models on port {self.ports['main_api']}...{Colors.ENDC}")
            
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
        
        # Start training API only if using local models
        if not self.claude_configured or self.use_local_models:
            self.start_training_api()
            
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
            
        if not self.claude_configured or self.use_local_models:
            print(f"  🔧 Training API:      http://localhost:{self.ports['training_api']}")
            
        if self.claude_configured:
            print(f"\n{Colors.GREEN}✨ Using Claude API:{Colors.ENDC}")
            print(f"  • No local memory usage")
            print(f"  • Superior language understanding")
            print(f"  • Fast responses")
            print(f"  • 200k token context window")
        else:
            print(f"\n{Colors.YELLOW}💻 Using Local Models:{Colors.ENDC}")
            print(f"  • Running on your machine")
            print(f"  • No API costs")
            print(f"  • Privacy-focused")
            
            if not self.memory_warned:
                print(f"\n{Colors.CYAN}💡 Tip: For better performance on M1 Macs,{Colors.ENDC}")
                print(f"   consider using Claude API!")
                
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
        
    def run(self, skip_install: bool = False, open_browser: bool = True, force_local: bool = False):
        """Run the system"""
        # Setup signal handler
        signal.signal(signal.SIGINT, self.handle_signal)
        
        # Force local models if requested
        if force_local:
            self.use_local_models = True
            self.claude_configured = False
            
        # Print header
        self.print_header()
        
        # Check Python version
        self.check_python_version()
        
        # Check dependencies
        if not skip_install:
            # Check Claude setup if configured
            if self.claude_configured and not force_local:
                if not self.check_claude_setup():
                    print(f"\n{Colors.WARNING}Claude setup incomplete. Falling back to local models.{Colors.ENDC}")
                    self.claude_configured = False
                    
            # Check essential dependencies
            if not self.check_essential_dependencies():
                print(f"\n{Colors.FAIL}❌ Missing essential dependencies. Please install required packages.{Colors.ENDC}")
                sys.exit(1)
                
            # Check optional dependencies if using local models
            if not self.claude_configured or self.use_local_models:
                self.check_optional_dependencies()
        else:
            print(f"{Colors.WARNING}⚠️  Skipping dependency installation{Colors.ENDC}")
            
        # Check memory status
        memory_ok = self.check_memory_status()
        if not memory_ok and not self.claude_configured:
            print(f"\n{Colors.FAIL}❌ Insufficient memory for local models{Colors.ENDC}")
            sys.exit(1)
            
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
    parser = argparse.ArgumentParser(description="AI-Powered Chatbot System Launcher")
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
        "--force-local",
        action="store_true",
        help="Force local model mode even if Claude is configured"
    )
    parser.add_argument(
        "--claude-only",
        action="store_true",
        help="Run in Claude-only mode (fail if not configured)"
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
        
        all_good = True
        
        if manager.claude_configured:
            if manager.check_claude_setup() and manager.check_essential_dependencies():
                print(f"\n{Colors.GREEN}✅ Claude mode ready!{Colors.ENDC}")
            else:
                all_good = False
        else:
            if manager.check_essential_dependencies() and manager.check_optional_dependencies():
                print(f"\n{Colors.GREEN}✅ Local mode ready!{Colors.ENDC}")
            else:
                all_good = False
                
        if not all_good:
            print(f"\n{Colors.FAIL}❌ Some checks failed. Please fix issues above.{Colors.ENDC}")
            
        sys.exit(0 if all_good else 1)
        
    if args.claude_only:
        if not manager.claude_configured:
            print(f"{Colors.FAIL}❌ Claude not configured. Please set ANTHROPIC_API_KEY.{Colors.ENDC}")
            sys.exit(1)
        args.force_local = False
        
    try:
        manager.run(
            skip_install=args.skip_install,
            open_browser=not args.no_browser,
            force_local=args.force_local
        )
    except Exception as e:
        print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
        manager.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()