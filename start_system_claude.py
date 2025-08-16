#!/usr/bin/env python3
"""
Simplified startup script for Claude-powered JARVIS
Optimized for cloud-based AI with minimal local resource usage
"""

import os
import sys
import subprocess
import time
import signal
import platform
from pathlib import Path
import argparse
import webbrowser
from typing import List
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


class ClaudeSystemManager:
    """Simplified system manager for Claude-powered JARVIS"""
    
    def __init__(self):
        self.processes = []
        self.backend_dir = Path("backend")
        self.ports = {
            "main_api": 8000,
            "frontend": 3000
        }
        self.is_m1_mac = platform.system() == "Darwin" and platform.machine() == "arm64"
        
    def print_header(self):
        """Print system header"""
        print(f"\n{Colors.HEADER}{'='*60}")
        print(f"{Colors.BOLD}🤖 JARVIS System Launcher (Claude-Powered) 🚀{Colors.ENDC}")
        if self.is_m1_mac:
            print(f"{Colors.CYAN}Perfect for M1 Mac - No local memory usage!{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")
        
    def check_python_version(self):
        """Check Python version"""
        print(f"{Colors.BLUE}Checking Python version...{Colors.ENDC}")
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print(f"{Colors.FAIL}❌ Python 3.8+ required. Current: {version.major}.{version.minor}{Colors.ENDC}")
            sys.exit(1)
        print(f"{Colors.GREEN}✓ Python {version.major}.{version.minor} detected{Colors.ENDC}")
        
    def check_claude_setup(self):
        """Check if Claude API is configured"""
        print(f"\n{Colors.BLUE}Checking Claude API setup...{Colors.ENDC}")
        
        # Load .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            print(f"{Colors.WARNING}⚠️  python-dotenv not installed{Colors.ENDC}")
            
        api_key = os.getenv("ANTHROPIC_API_KEY")
        use_claude = os.getenv("USE_CLAUDE", "1") == "1"
        
        if not api_key:
            print(f"{Colors.FAIL}❌ ANTHROPIC_API_KEY not found!{Colors.ENDC}")
            print(f"\n{Colors.YELLOW}To set up Claude:{Colors.ENDC}")
            print("1. Get an API key from: https://console.anthropic.com/")
            print("2. Create a .env file with:")
            print("   ANTHROPIC_API_KEY=your-api-key-here")
            print("   USE_CLAUDE=1")
            return False
            
        print(f"{Colors.GREEN}✓ Claude API key found{Colors.ENDC}")
        
        if use_claude:
            print(f"{Colors.GREEN}✓ Claude mode enabled{Colors.ENDC}")
        else:
            print(f"{Colors.WARNING}⚠️  Claude mode disabled (set USE_CLAUDE=1){Colors.ENDC}")
            
        # Check if anthropic package is installed
        try:
            import anthropic
            print(f"{Colors.GREEN}✓ Anthropic package installed{Colors.ENDC}")
        except ImportError:
            print(f"{Colors.WARNING}⚠️  Anthropic package not installed{Colors.ENDC}")
            print(f"   Run: pip install anthropic")
            return False
            
        return True
        
    def check_minimal_dependencies(self):
        """Check only essential dependencies for Claude mode"""
        print(f"\n{Colors.BLUE}Checking essential dependencies...{Colors.ENDC}")
        
        essential_packages = {
            "fastapi": "FastAPI web framework",
            "uvicorn": "ASGI server",
            "pydantic": "Data validation",
            "anthropic": "Claude API client",
            "python-dotenv": "Environment variables"
        }
        
        missing = []
        for package, description in essential_packages.items():
            try:
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
        """Check system memory (informational only for Claude)"""
        print(f"\n{Colors.BLUE}System memory status:{Colors.ENDC}")
        
        mem = psutil.virtual_memory()
        print(f"  Total: {mem.total / (1024**3):.1f} GB")
        print(f"  Used: {mem.used / (1024**3):.1f} GB ({mem.percent:.1f}%)")
        print(f"  Available: {mem.available / (1024**3):.1f} GB")
        
        print(f"\n{Colors.GREEN}✓ With Claude, memory usage is not a concern!{Colors.ENDC}")
        print(f"  All processing happens in the cloud")
        
    def create_directories(self):
        """Create necessary directories"""
        print(f"\n{Colors.BLUE}Creating directories...{Colors.ENDC}")
        
        directories = [
            self.backend_dir / "logs",
            self.backend_dir / "static"
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
        """Start backend API service"""
        print(f"\n{Colors.BLUE}Starting backend services...{Colors.ENDC}")
        
        # Check port
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
        env["USE_CLAUDE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        
        # Start main API
        print(f"{Colors.CYAN}Starting Claude-powered API on port {self.ports['main_api']}...{Colors.ENDC}")
        
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
        
    def print_access_info(self):
        """Print access information"""
        print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.BOLD}🎉 JARVIS is ready with Claude AI!{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")
        
        print(f"{Colors.CYAN}Access your services:{Colors.ENDC}")
        print(f"  🔌 API Documentation: http://localhost:{self.ports['main_api']}/docs")
        print(f"  💬 Chat Interface:    http://localhost:{self.ports['main_api']}/")
        
        print(f"\n{Colors.GREEN}✨ Benefits of Claude:{Colors.ENDC}")
        print(f"  • No local memory usage")
        print(f"  • Superior language understanding")
        print(f"  • Fast responses")
        print(f"  • 200k token context window")
        
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
        
    def run(self, skip_checks: bool = False):
        """Run the system"""
        # Setup signal handler
        signal.signal(signal.SIGINT, self.handle_signal)
        
        # Print header
        self.print_header()
        
        # Run checks
        if not skip_checks:
            self.check_python_version()
            
            if not self.check_claude_setup():
                print(f"\n{Colors.FAIL}❌ Claude setup incomplete. Please configure API key.{Colors.ENDC}")
                sys.exit(1)
                
            if not self.check_minimal_dependencies():
                print(f"\n{Colors.FAIL}❌ Missing dependencies. Please install required packages.{Colors.ENDC}")
                sys.exit(1)
        
        # Show memory status (informational)
        self.check_memory_status()
        
        # Create directories
        self.create_directories()
        
        # Start services
        self.start_backend_services()
        
        # Print access info
        self.print_access_info()
        
        # Keep running
        try:
            while True:
                time.sleep(1)
                # Check if process is still running
                for proc in self.processes:
                    if proc.poll() is not None:
                        output = proc.stdout.read().decode('utf-8') if proc.stdout else ""
                        print(f"{Colors.FAIL}❌ Service stopped unexpectedly{Colors.ENDC}")
                        if output:
                            print(f"{Colors.FAIL}Error:{Colors.ENDC}")
                            print(output[:500])
                        self.cleanup()
                        sys.exit(1)
        except KeyboardInterrupt:
            pass


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="JARVIS System Launcher (Claude-Powered)")
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip dependency and setup checks"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check setup and exit"
    )
    
    args = parser.parse_args()
    
    # Create and run system manager
    manager = ClaudeSystemManager()
    
    if args.check_only:
        manager.print_header()
        manager.check_python_version()
        if manager.check_claude_setup() and manager.check_minimal_dependencies():
            print(f"\n{Colors.GREEN}✅ All checks passed! Ready to run.{Colors.ENDC}")
        else:
            print(f"\n{Colors.FAIL}❌ Some checks failed. Please fix issues above.{Colors.ENDC}")
        sys.exit(0)
    
    try:
        manager.run(skip_checks=args.skip_checks)
    except Exception as e:
        print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
        manager.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()