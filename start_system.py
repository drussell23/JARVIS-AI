#!/usr/bin/env python3
"""
Comprehensive startup script for AI-Powered Chatbot System
Handles dependency installation and service launching
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

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class SystemManager:
    """Manages the AI Chatbot system startup"""
    
    def __init__(self):
        self.processes = []
        self.backend_dir = Path("backend")
        self.frontend_dir = Path("frontend")
        self.ports = {
            "main_api": 8000,
            "training_api": 8001,
            "frontend": 3000
        }
        self.demos = {
            "chat": "http://localhost:8000/docs",
            "voice": "http://localhost:8000/voice_demo.html",
            "automation": "http://localhost:8000/automation_demo.html",
            "rag": "http://localhost:8000/rag_demo.html",
            "training": "http://localhost:8001/llm_demo.html"
        }
        
    def print_header(self):
        """Print system header"""
        print(f"\n{Colors.HEADER}{'='*60}")
        print(f"{Colors.BOLD}🤖 AI-Powered Chatbot System Launcher 🚀{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")
        
    def check_python_version(self):
        """Check Python version"""
        print(f"{Colors.BLUE}Checking Python version...{Colors.ENDC}")
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print(f"{Colors.FAIL}❌ Python 3.8+ required. Current: {version.major}.{version.minor}{Colors.ENDC}")
            sys.exit(1)
        print(f"{Colors.GREEN}✓ Python {version.major}.{version.minor} detected{Colors.ENDC}")
        
    def check_node_installed(self) -> bool:
        """Check if Node.js is installed"""
        try:
            result = subprocess.run(["node", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"{Colors.GREEN}✓ Node.js {result.stdout.strip()} detected{Colors.ENDC}")
                return True
        except FileNotFoundError:
            pass
        return False
        
    def install_backend_dependencies(self):
        """Install backend Python dependencies"""
        print(f"\n{Colors.BLUE}Installing backend dependencies...{Colors.ENDC}")
        
        requirements_file = self.backend_dir / "requirements.txt"
        if not requirements_file.exists():
            print(f"{Colors.WARNING}⚠️  requirements.txt not found in backend directory{Colors.ENDC}")
            return
            
        try:
            # Upgrade pip first
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            
            # Install requirements
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            
            # Download NLTK data
            print(f"{Colors.BLUE}Downloading NLTK data...{Colors.ENDC}")
            try:
                import nltk
                nltk.download('punkt', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                print(f"{Colors.GREEN}✓ NLTK data downloaded{Colors.ENDC}")
            except:
                print(f"{Colors.WARNING}⚠️  Could not download NLTK data{Colors.ENDC}")
                
            # Download spaCy model
            print(f"{Colors.BLUE}Downloading spaCy model...{Colors.ENDC}")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "spacy", "download", "en_core_web_sm"
                ])
                print(f"{Colors.GREEN}✓ spaCy model downloaded{Colors.ENDC}")
            except:
                print(f"{Colors.WARNING}⚠️  Could not download spaCy model{Colors.ENDC}")
                
            print(f"{Colors.GREEN}✓ Backend dependencies installed{Colors.ENDC}")
            
        except subprocess.CalledProcessError as e:
            print(f"{Colors.FAIL}❌ Failed to install backend dependencies: {e}{Colors.ENDC}")
            sys.exit(1)
            
    def setup_frontend(self):
        """Setup frontend (React) if it exists"""
        if not self.frontend_dir.exists():
            print(f"{Colors.WARNING}⚠️  Frontend directory not found. Creating basic frontend...{Colors.ENDC}")
            self.create_basic_frontend()
            return
            
        print(f"\n{Colors.BLUE}Setting up frontend...{Colors.ENDC}")
        
        if not self.check_node_installed():
            print(f"{Colors.WARNING}⚠️  Node.js not installed. Skipping frontend setup.{Colors.ENDC}")
            print(f"    Install Node.js from https://nodejs.org/ to use the React frontend")
            return
            
        # Check if package.json exists
        package_json = self.frontend_dir / "package.json"
        if not package_json.exists():
            self.create_basic_frontend()
            return
            
        # Install npm dependencies
        try:
            print(f"{Colors.BLUE}Installing npm dependencies...{Colors.ENDC}")
            subprocess.check_call(["npm", "install"], cwd=self.frontend_dir)
            print(f"{Colors.GREEN}✓ Frontend dependencies installed{Colors.ENDC}")
        except subprocess.CalledProcessError as e:
            print(f"{Colors.WARNING}⚠️  Failed to install frontend dependencies: {e}{Colors.ENDC}")
            
    def create_basic_frontend(self):
        """Create a basic frontend structure"""
        print(f"{Colors.BLUE}Creating basic frontend...{Colors.ENDC}")
        
        # Create frontend directory
        self.frontend_dir.mkdir(exist_ok=True)
        
        # Create package.json
        package_json = {
            "name": "ai-chatbot-frontend",
            "version": "1.0.0",
            "description": "AI-Powered Chatbot Frontend",
            "scripts": {
                "start": "python -m http.server 3000",
                "dev": "python -m http.server 3000"
            }
        }
        
        with open(self.frontend_dir / "package.json", "w") as f:
            json.dump(package_json, f, indent=2)
            
        # Create basic index.html
        index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-Powered Chatbot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #1a1a2e;
            color: #eee;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
        }
        .container {
            text-align: center;
            padding: 2rem;
        }
        h1 {
            color: #00d4ff;
        }
        .demo-links {
            margin-top: 2rem;
        }
        .demo-links a {
            display: inline-block;
            margin: 0.5rem;
            padding: 1rem 2rem;
            background: linear-gradient(45deg, #00a8cc, #00d4ff);
            color: #000;
            text-decoration: none;
            border-radius: 25px;
            font-weight: bold;
            transition: transform 0.3s;
        }
        .demo-links a:hover {
            transform: scale(1.05);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AI-Powered Chatbot System</h1>
        <p>Welcome to your AI Assistant powered by advanced NLP and custom LLMs</p>
        
        <div class="demo-links">
            <a href="http://localhost:8000/docs" target="_blank">📚 API Documentation</a>
            <a href="http://localhost:8000/voice_demo.html" target="_blank">🎤 Voice Assistant</a>
            <a href="http://localhost:8000/automation_demo.html" target="_blank">⚡ Automation Demo</a>
            <a href="http://localhost:8000/rag_demo.html" target="_blank">🧠 RAG System</a>
            <a href="http://localhost:8001/llm_demo.html" target="_blank">🔧 LLM Training</a>
        </div>
    </div>
</body>
</html>"""
        
        with open(self.frontend_dir / "index.html", "w") as f:
            f.write(index_html)
            
        print(f"{Colors.GREEN}✓ Basic frontend created{Colors.ENDC}")
        
    def create_directories(self):
        """Create necessary directories"""
        print(f"\n{Colors.BLUE}Creating directories...{Colors.ENDC}")
        
        directories = [
            self.backend_dir / "data",
            self.backend_dir / "models",
            self.backend_dir / "checkpoints",
            self.backend_dir / "logs",
            self.backend_dir / "domain_knowledge",
            self.backend_dir / "faiss_index",
            self.backend_dir / "chroma_db",
            self.backend_dir / "knowledge_base"
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
        
        # Check ports
        for service, port in self.ports.items():
            if not self.check_port_available(port):
                print(f"{Colors.WARNING}⚠️  Port {port} ({service}) is already in use{Colors.ENDC}")
                response = input("Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    sys.exit(1)
                    
        # Start main API
        print(f"{Colors.CYAN}Starting main API on port {self.ports['main_api']}...{Colors.ENDC}")
        main_api_process = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=self.backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.processes.append(main_api_process)
        
        # Start training API
        print(f"{Colors.CYAN}Starting training API on port {self.ports['training_api']}...{Colors.ENDC}")
        training_api_process = subprocess.Popen(
            [sys.executable, "training_interface.py"],
            cwd=self.backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.processes.append(training_api_process)
        
        # Wait for services to start
        time.sleep(3)
        
        # Check if services are running
        for proc in self.processes:
            if proc.poll() is not None:
                print(f"{Colors.FAIL}❌ Failed to start backend service{Colors.ENDC}")
                self.cleanup()
                sys.exit(1)
                
        print(f"{Colors.GREEN}✓ Backend services started{Colors.ENDC}")
        
    def start_frontend(self):
        """Start frontend service"""
        if not self.frontend_dir.exists():
            return
            
        print(f"\n{Colors.BLUE}Starting frontend...{Colors.ENDC}")
        
        # Check if it's a React app or basic HTML
        if (self.frontend_dir / "package.json").exists():
            # Try to start with npm
            if self.check_node_installed():
                try:
                    frontend_process = subprocess.Popen(
                        ["npm", "start"],
                        cwd=self.frontend_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    self.processes.append(frontend_process)
                    print(f"{Colors.GREEN}✓ Frontend started on port {self.ports['frontend']}{Colors.ENDC}")
                except:
                    print(f"{Colors.WARNING}⚠️  Failed to start frontend with npm{Colors.ENDC}")
            else:
                # Start simple HTTP server
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
        print(f"{Colors.BOLD}🎉 System is ready! Access your services at:{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")
        
        print(f"{Colors.CYAN}Main Services:{Colors.ENDC}")
        print(f"  📱 Frontend:        http://localhost:{self.ports['frontend']}")
        print(f"  🔌 Main API:        http://localhost:{self.ports['main_api']}")
        print(f"  🔧 Training API:    http://localhost:{self.ports['training_api']}")
        
        print(f"\n{Colors.CYAN}Demo Interfaces:{Colors.ENDC}")
        for name, url in self.demos.items():
            print(f"  📄 {name.title():<15} {url}")
            
        print(f"\n{Colors.WARNING}Press Ctrl+C to stop all services{Colors.ENDC}\n")
        
    def cleanup(self):
        """Cleanup processes on exit"""
        print(f"\n{Colors.BLUE}Shutting down services...{Colors.ENDC}")
        for proc in self.processes:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
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
        
        # Install dependencies
        if not skip_install:
            self.install_backend_dependencies()
            self.setup_frontend()
        else:
            print(f"{Colors.WARNING}⚠️  Skipping dependency installation{Colors.ENDC}")
            
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
            webbrowser.open(f"http://localhost:{self.ports['frontend']}")
            
        # Keep running
        try:
            while True:
                time.sleep(1)
                # Check if processes are still running
                for proc in self.processes:
                    if proc.poll() is not None:
                        print(f"{Colors.WARNING}⚠️  A service has stopped unexpectedly{Colors.ENDC}")
                        
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
        "--backend-only",
        action="store_true",
        help="Start only backend services"
    )
    
    args = parser.parse_args()
    
    # Create and run system manager
    manager = SystemManager()
    
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