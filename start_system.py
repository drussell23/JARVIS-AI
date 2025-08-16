#!/usr/bin/env python3
"""
Comprehensive startup script for AI-Powered Chatbot System
Handles dependency installation and service launching
Async version for better performance
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
    YELLOW = '\033[93m'  # Same as WARNING
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
            "frontend": 3000,
            "llama_cpp": 8080  # llama.cpp server port
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
        
    def print_header(self):
        """Print system header"""
        print(f"\n{Colors.HEADER}{'='*60}")
        if self.is_m1_mac:
            print(f"{Colors.BOLD}🤖 AI-Powered Chatbot System Launcher 🚀 (M1 Optimized){Colors.ENDC}")
        else:
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
        
    def check_llama_cpp_installed(self) -> bool:
        """Check if llama.cpp is installed"""
        try:
            result = subprocess.run(["which", "llama-server"], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
            
    def check_llama_model_exists(self) -> bool:
        """Check if a llama.cpp model exists"""
        model_path = Path.home() / "Documents" / "ai-models" / "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
        return model_path.exists()
        
    def check_installed_packages(self):
        """Check which packages are already installed"""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                import json
                installed = json.loads(result.stdout)
                return {pkg['name'].lower(): pkg['version'] for pkg in installed}
        except:
            pass
        return {}
    
    def check_missing_dependencies(self):
        """Check for missing dependencies from requirements.txt"""
        requirements_file = self.backend_dir / "requirements.txt"
        if not requirements_file.exists():
            return []
        
        installed = self.check_installed_packages()
        missing = []
        
        with open(requirements_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Parse requirement
                    if '>=' in line or '==' in line or '<=' in line:
                        pkg_name = line.split('>=')[0].split('==')[0].split('<=')[0].strip()
                    elif '@' in line:  # URL-based dependencies
                        pkg_name = line.split('@')[0].strip()
                    else:
                        pkg_name = line.strip()
                    
                    # Check if installed
                    if pkg_name.lower() not in installed:
                        missing.append(line)
        
        return missing
    
    def install_backend_dependencies(self):
        """Install backend Python dependencies"""
        print(f"\n{Colors.BLUE}Checking backend dependencies...{Colors.ENDC}")
        
        requirements_file = self.backend_dir / "requirements.txt"
        if not requirements_file.exists():
            print(f"{Colors.WARNING}⚠️  requirements.txt not found in backend directory{Colors.ENDC}")
            return
        
        # Check for missing dependencies
        missing = self.check_missing_dependencies()
        
        if not missing:
            print(f"{Colors.GREEN}✓ All backend dependencies already installed{Colors.ENDC}")
        else:
            print(f"{Colors.CYAN}Found {len(missing)} missing dependencies{Colors.ENDC}")
            print(f"{Colors.BLUE}Installing missing dependencies...{Colors.ENDC}")
            
            try:
                # Install only missing dependencies
                total = len(missing)
                for i, dep in enumerate(missing, 1):
                    print(f"  [{i}/{total}] Installing: {dep}")
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", dep
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                
                print(f"{Colors.GREEN}✓ Missing dependencies installed{Colors.ENDC}")
                
            except subprocess.CalledProcessError as e:
                print(f"{Colors.FAIL}❌ Failed to install dependencies: {e}{Colors.ENDC}")
                sys.exit(1)
        
        # Check NLTK data
        print(f"{Colors.BLUE}Checking NLTK data...{Colors.ENDC}")
        try:
            import nltk
            nltk_data_path = Path.home() / "nltk_data"
            required_data = ['punkt', 'averaged_perceptron_tagger', 'stopwords', 'wordnet']
            missing_data = []
            
            for data in required_data:
                if not (nltk_data_path / "tokenizers" / data).exists() and \
                   not (nltk_data_path / "corpora" / data).exists() and \
                   not (nltk_data_path / "taggers" / data).exists():
                    missing_data.append(data)
            
            if missing_data:
                print(f"{Colors.CYAN}Downloading missing NLTK data...{Colors.ENDC}")
                for data in missing_data:
                    nltk.download(data, quiet=True)
                print(f"{Colors.GREEN}✓ NLTK data ready{Colors.ENDC}")
            else:
                print(f"{Colors.GREEN}✓ NLTK data already downloaded{Colors.ENDC}")
        except ImportError:
            print(f"{Colors.WARNING}⚠️  NLTK not installed{Colors.ENDC}")
        
        # Check spaCy model
        print(f"{Colors.BLUE}Checking spaCy model...{Colors.ENDC}")
        try:
            import spacy
            try:
                spacy.load("en_core_web_sm")
                print(f"{Colors.GREEN}✓ spaCy model already installed{Colors.ENDC}")
            except OSError:
                print(f"{Colors.CYAN}Downloading spaCy model...{Colors.ENDC}")
                subprocess.check_call([
                    sys.executable, "-m", "spacy", "download", "en_core_web_sm"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                print(f"{Colors.GREEN}✓ spaCy model downloaded{Colors.ENDC}")
        except ImportError:
            print(f"{Colors.WARNING}⚠️  spaCy not installed{Colors.ENDC}")
            
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
    
    def check_memory_status(self):
        """Check system memory and provide recommendations"""
        print(f"\n{Colors.BLUE}Checking system memory...{Colors.ENDC}")
        
        mem = psutil.virtual_memory()
        memory_percent = mem.percent
        available_gb = mem.available / (1024**3)
        used_gb = mem.used / (1024**3)
        total_gb = mem.total / (1024**3)
        
        print(f"  Total: {total_gb:.1f} GB")
        print(f"  Used: {used_gb:.1f} GB ({memory_percent:.1f}%)")
        print(f"  Available: {available_gb:.1f} GB")
        
        # Check memory thresholds
        if memory_percent < 50:
            print(f"{Colors.GREEN}✓ Memory OK for all features including LangChain{Colors.ENDC}")
            return True
        elif memory_percent < 65:
            print(f"{Colors.WARNING}⚠️  Memory OK for Intelligent mode (LangChain disabled){Colors.ENDC}")
            self.memory_warned = True
            return True
        elif memory_percent < 80:
            print(f"{Colors.WARNING}⚠️  Memory high - limited features available{Colors.ENDC}")
            self.memory_warned = True
            self.suggest_memory_optimization()
            return True
        else:
            print(f"{Colors.FAIL}❌ Memory critical ({memory_percent:.1f}%) - only basic features available{Colors.ENDC}")
            self.memory_warned = True
            self.suggest_memory_optimization()
            return False
    
    def suggest_memory_optimization(self):
        """Suggest ways to free memory"""
        print(f"\n{Colors.CYAN}Memory Optimization Suggestions:{Colors.ENDC}")
        
        # Get top memory users
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
            try:
                pinfo = proc.info
                if pinfo['memory_percent'] > 2:  # Show processes using > 2%
                    processes.append(pinfo)
            except:
                pass
        
        # Sort by memory usage
        processes.sort(key=lambda x: x['memory_percent'], reverse=True)
        
        if processes:
            print(f"\nTop memory-consuming processes:")
            for i, proc in enumerate(processes[:5]):
                print(f"  {i+1}. {proc['name'][:30]:30} {proc['memory_percent']:.1f}%")
        
        print(f"\n{Colors.CYAN}To free memory:{Colors.ENDC}")
        print(f"  1. Close unnecessary browser tabs")
        print(f"  2. Quit unused applications")
        print(f"  3. Close IDE/editor instances you're not using")
        print(f"  4. Run memory optimization: curl -X POST http://localhost:8000/chat/optimize-memory")
        print(f"\nOr continue with limited features.")
    
    def optimize_memory_if_needed(self):
        """Try to optimize memory if it's too high"""
        mem = psutil.virtual_memory()
        if mem.percent > 80:
            print(f"\n{Colors.WARNING}Memory usage is critical ({mem.percent:.1f}%){Colors.ENDC}")
            response = input(f"\nAttempt automatic memory optimization? (y/n): ")
            
            if response.lower() == 'y':
                print(f"{Colors.CYAN}Attempting memory optimization...{Colors.ENDC}")
                
                # Run basic optimizations
                try:
                    # Python garbage collection
                    import gc
                    gc.collect()
                    
                    # Kill some helper processes if on macOS
                    if platform.system() == "Darwin":
                        # Kill Cursor helpers if present
                        subprocess.run(["pkill", "-f", "Cursor Helper"], capture_output=True)
                        # Kill Chrome helpers
                        subprocess.run(["pkill", "-f", "Chrome Helper"], capture_output=True)
                        time.sleep(2)
                    
                    # Check memory again
                    new_mem = psutil.virtual_memory()
                    freed_mb = (mem.used - new_mem.used) / (1024 * 1024)
                    
                    if freed_mb > 0:
                        print(f"{Colors.GREEN}✓ Freed {freed_mb:.0f} MB of memory{Colors.ENDC}")
                        print(f"Memory now at {new_mem.percent:.1f}%")
                    else:
                        print(f"{Colors.WARNING}Could not free significant memory{Colors.ENDC}")
                        
                except Exception as e:
                    print(f"{Colors.WARNING}Optimization error: {e}{Colors.ENDC}")
        
    def start_llama_cpp_server(self):
        """Start llama.cpp server for M1 Macs"""
        if not self.is_m1_mac:
            return
            
        print(f"\n{Colors.BLUE}Checking M1 optimization setup...{Colors.ENDC}")
        
        # Check if llama.cpp is installed
        if not self.check_llama_cpp_installed():
            print(f"{Colors.WARNING}⚠️  llama.cpp not found. Run ./backend/setup_llama_m1.sh to install{Colors.ENDC}")
            return
            
        # Check if model exists
        if not self.check_llama_model_exists():
            print(f"{Colors.WARNING}⚠️  No llama.cpp model found. Run ./backend/setup_llama_m1.sh to download{Colors.ENDC}")
            return
            
        # Check if port is available
        if not self.check_port_available(self.ports['llama_cpp']):
            print(f"{Colors.WARNING}⚠️  Port {self.ports['llama_cpp']} is already in use (llama.cpp might already be running){Colors.ENDC}")
            # Check if it's actually llama.cpp
            try:
                import requests
                response = requests.get(f"http://localhost:{self.ports['llama_cpp']}/health", timeout=1)
                if response.status_code == 200:
                    print(f"{Colors.GREEN}✓ llama.cpp server already running{Colors.ENDC}")
                    return
            except:
                pass
                
        # Start llama.cpp server
        print(f"{Colors.CYAN}Starting llama.cpp server on port {self.ports['llama_cpp']}...{Colors.ENDC}")
        
        model_path = Path.home() / "Documents" / "ai-models" / "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
        
        llama_process = subprocess.Popen(
            [
                "llama-server",
                "-m", str(model_path),
                "-c", "2048",
                "--host", "0.0.0.0",
                "--port", str(self.ports['llama_cpp']),
                "-ngl", "1",
                "--n-gpu-layers", "1"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        self.processes.append(llama_process)
        
        # Wait for llama.cpp to start
        print(f"{Colors.CYAN}Waiting for llama.cpp to initialize...{Colors.ENDC}")
        time.sleep(5)
        
        # Check if it started successfully
        try:
            import requests
            response = requests.get(f"http://localhost:{self.ports['llama_cpp']}/health", timeout=2)
            if response.status_code == 200:
                print(f"{Colors.GREEN}✓ llama.cpp server started successfully{Colors.ENDC}")
            else:
                print(f"{Colors.WARNING}⚠️  llama.cpp server may not be ready yet{Colors.ENDC}")
        except:
            print(f"{Colors.WARNING}⚠️  Could not verify llama.cpp server status{Colors.ENDC}")
    
    def start_backend_services(self):
        """Start backend services"""
        print(f"\n{Colors.BLUE}Starting backend services...{Colors.ENDC}")
        
        # Check ports (only backend ports)
        backend_ports = {"main_api": self.ports["main_api"], "training_api": self.ports["training_api"]}
        for service, port in backend_ports.items():
            if not self.check_port_available(port):
                print(f"{Colors.WARNING}⚠️  Port {port} ({service}) is already in use{Colors.ENDC}")
                print(f"{Colors.CYAN}Attempting to kill existing process...{Colors.ENDC}")
                try:
                    subprocess.run(f"lsof -ti:{port} | xargs kill -9", shell=True, capture_output=True)
                    time.sleep(1)
                except:
                    pass
        
        # Set M1 optimization environment variables
        env = os.environ.copy()
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["PYTHONUNBUFFERED"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        
        # Detect if running on M1 Mac
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            print(f"{Colors.CYAN}🍎 M1 Mac detected - using optimized settings{Colors.ENDC}")
                    
        # Start main API
        print(f"{Colors.CYAN}Starting main API on port {self.ports['main_api']}...{Colors.ENDC}")
        
        # Always use system Python for now (has all dependencies)
        python_cmd = sys.executable
        print(f"{Colors.CYAN}Using Python: {python_cmd}{Colors.ENDC}")
            
        main_api_process = subprocess.Popen(
            [python_cmd, "run_server.py"],
            cwd=self.backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            env=env
        )
        self.processes.append(main_api_process)
        
        # Start training API (optional - may fail if dependencies missing)
        print(f"{Colors.CYAN}Starting training API on port {self.ports['training_api']}...{Colors.ENDC}")
        try:
            # Check if training_interface.py exists
            if (self.backend_dir / "training_interface.py").exists():
                training_api_process = subprocess.Popen(
                    [python_cmd, "training_interface.py"],
                    cwd=self.backend_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env
                )
                self.processes.append(training_api_process)
                
                # Give it a moment to start
                time.sleep(2)
                
                # Check if it's still running
                if training_api_process.poll() is not None:
                    # Read output to see what went wrong
                    output = training_api_process.stdout.read().decode('utf-8') if training_api_process.stdout else ""
                    if "ModuleNotFoundError" in output:
                        print(f"{Colors.WARNING}⚠️  Training API skipped (missing dependencies){Colors.ENDC}")
                        self.processes.remove(training_api_process)
                    else:
                        print(f"{Colors.WARNING}⚠️  Training API failed to start{Colors.ENDC}")
                        if output:
                            print(f"Error: {output[:200]}...")
                        self.processes.remove(training_api_process)
            else:
                print(f"{Colors.WARNING}⚠️  Training API not found{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.WARNING}⚠️  Failed to start training API: {e}{Colors.ENDC}")
        
        # Wait for services to start
        print(f"{Colors.CYAN}Waiting for services to initialize (this may take 30-60 seconds on M1)...{Colors.ENDC}")
        time.sleep(10)
        
        # Check if services are running
        for i, proc in enumerate(self.processes):
            if proc.poll() is not None:
                # Read output to show error
                output = proc.stdout.read().decode('utf-8') if proc.stdout else ""
                service_name = "Main API" if i == 0 else "Training API"
                print(f"{Colors.FAIL}❌ Failed to start {service_name}{Colors.ENDC}")
                if output:
                    print(f"{Colors.FAIL}Error output:{Colors.ENDC}")
                    print(output[:1000])
                self.cleanup()
                sys.exit(1)
                
        print(f"{Colors.GREEN}✓ Backend services started{Colors.ENDC}")
        
    def start_frontend(self):
        """Start frontend service"""
        if not self.frontend_dir.exists():
            return
            
        print(f"\n{Colors.BLUE}Starting frontend...{Colors.ENDC}")
        
        # Check if port is available
        if not self.check_port_available(self.ports['frontend']):
            print(f"{Colors.WARNING}⚠️  Port {self.ports['frontend']} is already in use{Colors.ENDC}")
            print(f"{Colors.CYAN}Attempting to kill existing process...{Colors.ENDC}")
            try:
                subprocess.run(f"lsof -ti:{self.ports['frontend']} | xargs kill -9", shell=True, capture_output=True)
                time.sleep(1)
            except:
                pass
        
        # Check if it's a React app or basic HTML
        if (self.frontend_dir / "package.json").exists():
            # Try to start with npm
            if self.check_node_installed():
                try:
                    frontend_process = subprocess.Popen(
                        ["npm", "start"],
                        cwd=self.frontend_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        env={**os.environ, "PORT": str(self.ports['frontend'])}
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
        
        if self.is_m1_mac:
            print(f"  🤖 llama.cpp:       http://localhost:{self.ports['llama_cpp']} (M1 optimized)")
        
        print(f"\n{Colors.CYAN}Demo Interfaces:{Colors.ENDC}")
        for name, url in self.demos.items():
            print(f"  📄 {name.title():<15} {url}")
            
        if self.is_m1_mac:
            print(f"\n{Colors.GREEN}🍎 M1 Optimization Active:{Colors.ENDC}")
            print(f"  Using llama.cpp for native M1 performance")
            print(f"  No PyTorch bus errors!")
            
        print(f"\n{Colors.WARNING}Press Ctrl+C to stop all services{Colors.ENDC}")
        
        # Show memory warning if applicable
        if self.memory_warned:
            print(f"\n{Colors.YELLOW}⚠️  Note: High memory usage detected. Some features may be limited.{Colors.ENDC}")
            print(f"{Colors.YELLOW}   Run 'curl -X POST http://localhost:8000/chat/optimize-memory' to free memory{Colors.ENDC}")
        
        print()
        
    def cleanup(self):
        """Cleanup processes on exit"""
        print(f"\n{Colors.BLUE}Shutting down services...{Colors.ENDC}")
        for proc in self.processes:
            try:
                # Handle both subprocess.Popen and asyncio.subprocess.Process
                if hasattr(proc, 'poll'):
                    # Regular subprocess.Popen
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                elif hasattr(proc, 'returncode'):
                    # asyncio.subprocess.Process
                    if proc.returncode is None:
                        proc.terminate()
                        # Can't wait synchronously on async process
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
        
        # Install dependencies
        if not skip_install:
            self.install_backend_dependencies()
            self.setup_frontend()
        else:
            print(f"{Colors.WARNING}⚠️  Skipping dependency installation{Colors.ENDC}")
            
        # Create directories
        self.create_directories()
        
        # Check memory status
        self.check_memory_status()
        
        # Offer memory optimization if needed
        self.optimize_memory_if_needed()
        
        # Start llama.cpp for M1 Macs
        if self.is_m1_mac:
            self.start_llama_cpp_server()
        
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
                for i, proc in enumerate(self.processes):
                    if proc.poll() is not None:
                        # Read the output to see what went wrong
                        output = proc.stdout.read().decode('utf-8') if proc.stdout else ""
                        
                        # Determine service name
                        if self.is_m1_mac and len(self.processes) > 3:
                            # With llama.cpp: 0=llama, 1=main, 2=training, 3=frontend
                            service_names = ["llama.cpp", "Main API", "Training API", "Frontend"]
                        else:
                            # Without llama.cpp: 0=main, 1=training, 2=frontend
                            service_names = ["Main API", "Training API", "Frontend"]
                        
                        service_name = service_names[i] if i < len(service_names) else f"Service {i}"
                        
                        print(f"{Colors.FAIL}❌ {service_name} has stopped unexpectedly{Colors.ENDC}")
                        if output:
                            print(f"{Colors.FAIL}Error output:{Colors.ENDC}")
                            print(output[:500])  # Show first 500 chars of error
                        
        except KeyboardInterrupt:
            pass

    # Async methods for better performance
    async def install_backend_dependencies_async(self):
        """Install backend Python dependencies asynchronously"""
        print(f"\n{Colors.BLUE}Checking backend dependencies (async)...{Colors.ENDC}")
        
        requirements_file = self.backend_dir / "requirements.txt"
        if not requirements_file.exists():
            print(f"{Colors.WARNING}⚠️  requirements.txt not found in backend directory{Colors.ENDC}")
            return
        
        # Check for missing dependencies
        missing = self.check_missing_dependencies()
        
        if not missing:
            print(f"{Colors.GREEN}✓ All backend dependencies already installed{Colors.ENDC}")
        else:
            print(f"{Colors.CYAN}Found {len(missing)} missing dependencies{Colors.ENDC}")
            print(f"{Colors.BLUE}Installing missing dependencies asynchronously...{Colors.ENDC}")
            
            try:
                # Install missing dependencies in parallel (batches of 5)
                batch_size = 5
                for i in range(0, len(missing), batch_size):
                    batch = missing[i:i + batch_size]
                    tasks = []
                    
                    for dep in batch:
                        print(f"  Installing: {dep}")
                        proc = await asyncio.create_subprocess_exec(
                            sys.executable, "-m", "pip", "install", dep,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        tasks.append(proc.communicate())
                    
                    # Wait for batch to complete
                    await asyncio.gather(*tasks)
                
                print(f"{Colors.GREEN}✓ Missing dependencies installed{Colors.ENDC}")
                
            except Exception as e:
                print(f"{Colors.FAIL}❌ Failed to install dependencies: {e}{Colors.ENDC}")
                sys.exit(1)
    
    async def start_backend_services_async(self):
        """Start backend services asynchronously"""
        print(f"\n{Colors.CYAN}Starting backend services...{Colors.ENDC}")
        
        # Start main API
        main_api_cmd = [
            sys.executable,
            "-m", "uvicorn",
            "main:app",
            "--host", "0.0.0.0",
            "--port", str(self.ports['main_api'])
        ]
        
        proc = await asyncio.create_subprocess_exec(
            *main_api_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.backend_dir)
        )
        self.processes.append(proc)
        print(f"{Colors.GREEN}✓ Main API started on port {self.ports['main_api']}{Colors.ENDC}")
    
    async def monitor_services_async(self):
        """Monitor services asynchronously"""
        while True:
            await asyncio.sleep(1)
            # Check if processes are still running
            for i, proc in enumerate(self.processes):
                if isinstance(proc, asyncio.subprocess.Process) and proc.returncode is not None:
                    service_names = ["Main API", "Training API", "Frontend"]
                    service_name = service_names[i] if i < len(service_names) else f"Service {i}"
                    print(f"{Colors.FAIL}❌ {service_name} has stopped unexpectedly{Colors.ENDC}")
                    self.cleanup()
                    sys.exit(1)
    
    async def run_async(self, skip_install: bool = False, open_browser: bool = True):
        """Run the system asynchronously"""
        try:
            # Show header
            self.print_header()
            
            # Check Python version
            self.check_python_version()
            
            # Install dependencies unless skipped
            if not skip_install:
                await self.install_backend_dependencies_async()
                # Keep frontend setup sync for now
                self.setup_frontend()
            else:
                print(f"\n{Colors.WARNING}Skipping dependency installation (--skip-install flag){Colors.ENDC}")
            
            # Create necessary directories
            self.create_directories()
            
            # Start services
            await self.start_backend_services_async()
            # Keep frontend sync for now
            self.start_frontend()
            
            # Show status
            self.print_access_info()
            
            # Open browser if requested
            if open_browser:
                await asyncio.sleep(3)
                webbrowser.open(f"http://localhost:{self.ports['frontend']}")
            
            # Monitor services
            print(f"\n{Colors.GREEN}System is running! Press Ctrl+C to stop all services.{Colors.ENDC}")
            
            await self.monitor_services_async()
                        
        except KeyboardInterrupt:
            print(f"\n{Colors.WARNING}Shutting down...{Colors.ENDC}")
            self.cleanup()
        except Exception as e:
            print(f"\n{Colors.FAIL}Error: {e}{Colors.ENDC}")
            self.cleanup()
            sys.exit(1)


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
    parser.add_argument(
        "--async-mode",
        action="store_true",
        help="Use async mode for better performance"
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check dependencies and exit"
    )
    parser.add_argument(
        "--optimize-memory",
        action="store_true",
        help="Run memory optimization before starting"
    )
    parser.add_argument(
        "--memory-status",
        action="store_true",
        help="Check memory status and exit"
    )
    
    args = parser.parse_args()
    
    # Create and run system manager
    manager = SystemManager()
    
    # Check dependencies only
    if args.check_deps:
        manager.print_header()
        print(f"{Colors.BLUE}Checking dependencies...{Colors.ENDC}\n")
        
        # Check installed packages
        installed = manager.check_installed_packages()
        print(f"{Colors.GREEN}✓ Found {len(installed)} installed packages{Colors.ENDC}")
        
        # Check missing dependencies
        missing = manager.check_missing_dependencies()
        if missing:
            print(f"{Colors.WARNING}⚠️  Found {len(missing)} missing dependencies:{Colors.ENDC}")
            for dep in missing[:10]:  # Show first 10
                print(f"    - {dep}")
            if len(missing) > 10:
                print(f"    ... and {len(missing) - 10} more")
        else:
            print(f"{Colors.GREEN}✓ All required dependencies are installed{Colors.ENDC}")
        
        # Check NLTK data
        try:
            import nltk
            nltk_data_path = Path.home() / "nltk_data"
            if nltk_data_path.exists():
                print(f"{Colors.GREEN}✓ NLTK data found at {nltk_data_path}{Colors.ENDC}")
            else:
                print(f"{Colors.WARNING}⚠️  NLTK data not found{Colors.ENDC}")
        except ImportError:
            print(f"{Colors.WARNING}⚠️  NLTK not installed{Colors.ENDC}")
        
        # Check spaCy model
        try:
            import spacy
            try:
                spacy.load("en_core_web_sm")
                print(f"{Colors.GREEN}✓ spaCy model 'en_core_web_sm' is installed{Colors.ENDC}")
            except OSError:
                print(f"{Colors.WARNING}⚠️  spaCy model 'en_core_web_sm' not installed{Colors.ENDC}")
        except ImportError:
            print(f"{Colors.WARNING}⚠️  spaCy not installed{Colors.ENDC}")
        
        # Check LangChain
        if 'langchain' in installed:
            print(f"{Colors.GREEN}✓ LangChain {installed['langchain']} is installed{Colors.ENDC}")
        else:
            print(f"{Colors.WARNING}⚠️  LangChain not installed{Colors.ENDC}")
        
        # Check llama-cpp-python
        if 'llama-cpp-python' in installed or 'llama_cpp_python' in installed:
            print(f"{Colors.GREEN}✓ llama-cpp-python is installed{Colors.ENDC}")
        else:
            print(f"{Colors.WARNING}⚠️  llama-cpp-python not installed{Colors.ENDC}")
        
        # Check if M1 Mac
        if manager.is_m1_mac:
            print(f"\n{Colors.CYAN}🍎 M1 Mac detected{Colors.ENDC}")
            if manager.check_llama_cpp_installed():
                print(f"{Colors.GREEN}✓ llama.cpp is installed{Colors.ENDC}")
            else:
                print(f"{Colors.WARNING}⚠️  llama.cpp not installed{Colors.ENDC}")
            
            if manager.check_llama_model_exists():
                print(f"{Colors.GREEN}✓ Mistral model found{Colors.ENDC}")
            else:
                print(f"{Colors.WARNING}⚠️  Mistral model not found{Colors.ENDC}")
        
        sys.exit(0)
    
    # Check memory status only
    if args.memory_status:
        manager.print_header()
        manager.check_memory_status()
        
        # Show optimization suggestions if memory is high
        mem = psutil.virtual_memory()
        if mem.percent > 50:
            print(f"\n{Colors.CYAN}For full LangChain features, memory should be < 50%{Colors.ENDC}")
            manager.suggest_memory_optimization()
        
        sys.exit(0)
    
    # Run memory optimization if requested
    if args.optimize_memory:
        manager.print_header()
        print(f"{Colors.BLUE}Running memory optimization...{Colors.ENDC}")
        manager.optimize_memory_if_needed()
        print()
        # Continue with startup after optimization
    
    try:
        if args.async_mode:
            # Run in async mode
            asyncio.run(manager.run_async(
                skip_install=args.skip_install,
                open_browser=not args.no_browser
            ))
        else:
            # Run in sync mode (default)
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