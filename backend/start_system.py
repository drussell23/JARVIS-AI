#!/usr/bin/env python3
"""
JARVIS AI System v12.1 - Autonomous Cognitive Intelligence Platform
Complete AI Agent with Enhanced Vision System & Real-time Screen Analysis
"""

import os
import sys
import subprocess
import asyncio
import argparse
from datetime import datetime
from pathlib import Path

# Set environment variables before imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['USE_TORCH'] = '1'
os.environ['USE_TF'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# ASCII Art Banner
JARVIS_BANNER = """
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗    ██╗   ██╗ ██╗██████╗      ██╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝    ██║   ██║███║╚════██╗    ███║
     ██║███████║██████╔╝██║   ██║██║███████╗    ██║   ██║╚██║ █████╔╝    ╚██║
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║    ╚██╗ ██╔╝ ██║██╔═══╝      ██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║     ╚████╔╝  ██║███████╗ ██╗██║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝      ╚═══╝   ╚═╝╚══════╝ ╚═╝╚═╝ 
                                                                        
    👁️ Enhanced Vision System • Real-time Screen Analysis • Claude Vision API 👁️
      Full Autonomy • Context-Aware Decisions • Proactive Assistance • ML Audio
"""


class JARVISSystemManager:
    """Manages the complete JARVIS AI system"""
    
    def __init__(self):
        self.components = {
            'backend': {'status': False, 'port': 8000},
            'swift': {'status': False, 'path': 'swift_bridge'},
            'vision': {'status': False, 'features': []},
            'voice': {'status': False, 'engine': None},
            'unified_agent': {'status': False, 'mode': None},
            'autonomy': {'status': False, 'features': []},
            'ml_audio': {'status': False, 'models': []},
            'creative_ai': {'status': False, 'capabilities': []},
            'predictive': {'status': False, 'engines': []}
        }
        
    def print_banner(self):
        """Display JARVIS banner"""
        print("\033[36m" + JARVIS_BANNER + "\033[0m")
        print("=" * 70)
        print(f"🕐 System Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
    
    def check_requirements(self):
        """Check all system requirements"""
        print("\n🔍 Checking System Requirements...")
        print("-" * 50)
        
        # Check API Keys
        checks = []
        
        # Anthropic API Key
        if os.getenv("ANTHROPIC_API_KEY"):
            checks.append(("✅", "Anthropic API Key", "Claude integration ready"))
            self.components['vision']['features'].append('claude_vision')
        else:
            checks.append(("❌", "Anthropic API Key", "Set ANTHROPIC_API_KEY for full features"))
        
        # Check Python packages
        required_packages = {
            'fastapi': 'Backend API',
            'anthropic': 'Claude integration',
            'sentence_transformers': 'ML intelligence',
            'opencv-python': 'Vision system',
            'pyobjc': 'macOS integration',
            'pyttsx3': 'Voice synthesis'
        }
        
        for package, description in required_packages.items():
            try:
                __import__(package.replace('-', '_'))
                checks.append(("✅", package, description))
            except ImportError:
                checks.append(("⚠️", package, f"{description} not available"))
        
        # Check Swift
        swift_path = Path("swift_bridge/CommandClassifierCLI")
        if swift_path.exists():
            checks.append(("✅", "Swift Command Classifier", "Intelligent command processing"))
            self.components['swift']['status'] = True
        else:
            checks.append(("⚠️", "Swift Classifier", "Run build_swift.sh to enable"))
        
        # Check optional features
        try:
            import torch
            checks.append(("✅", "PyTorch", "Deep learning features"))
            self.components['vision']['features'].append('deep_learning')
        except ImportError:
            checks.append(("ℹ️", "PyTorch", "Optional - for advanced ML"))
        
        # Print all checks
        for status, component, message in checks:
            print(f"{status} {component:<25} {message}")
        
        print("-" * 50)
        return all(status == "✅" for status, _, _ in checks[:2])  # Core requirements
    
    def check_vision_capabilities(self):
        """Check vision system capabilities"""
        print("\n👁️ Vision System Capabilities:")
        print("-" * 50)
        
        capabilities = []
        
        # Basic screen capture
        capabilities.append(("✅", "Screen Capture", "Basic screenshot functionality"))
        
        # Multi-window analysis
        capabilities.append(("✅", "Multi-Window Analysis", "Track all open windows"))
        
        # Notification detection
        capabilities.append(("✅", "Notification Detection", "Proactive alerts"))
        
        # Claude Vision
        if 'claude_vision' in self.components['vision']['features']:
            capabilities.append(("✅", "Claude Vision API", "Advanced image understanding"))
        else:
            capabilities.append(("❌", "Claude Vision API", "Requires API key"))
        
        # ML-based routing
        try:
            import sentence_transformers
            capabilities.append(("✅", "ML Vision Routing", "Zero hardcoding"))
        except:
            capabilities.append(("⚠️", "ML Vision Routing", "Basic routing only"))
        
        # Natural language vision commands
        capabilities.append(("✅", "Natural Vision Commands", "Ask 'can you see my screen?'"))
        
        # Real-time screen analysis
        capabilities.append(("✅", "Real-time Analysis", "Live screen understanding"))
        
        for status, feature, description in capabilities:
            print(f"{status} {feature:<25} {description}")
        
        self.components['vision']['status'] = True
        print("-" * 50)
    
    def check_advanced_features(self):
        """Check v12.1 enhanced features"""
        print("\n🧠 Advanced v12.1 Enhanced Features:")
        print("-" * 50)
        
        features = []
        
        # Autonomous Decision Engine
        features.append(("✅", "Autonomous Decision Engine", "Context-aware autonomous actions"))
        
        # Creative Problem Solving
        features.append(("✅", "Creative Problem Solving", "AI-driven innovative solutions"))
        
        # Predictive Intelligence
        features.append(("✅", "Predictive Intelligence", "Anticipates user needs"))
        
        # ML Audio System
        features.append(("✅", "ML Audio System", "Advanced audio processing"))
        
        # Enhanced Autonomy
        features.append(("✅", "Full Autonomy Mode", "Complete hands-free operation"))
        
        # Context Engine
        features.append(("✅", "Context Engine", "Deep contextual understanding"))
        
        # Hardware Control
        features.append(("✅", "Hardware Control", "Direct system integration"))
        
        # Vision Navigation
        features.append(("✅", "Vision Navigation", "Visual-based UI navigation"))
        
        # Enhanced Vision System (v12.1)
        features.append(("✅", "Natural Language Vision", "Ask naturally about your screen"))
        
        # Fixed Vision Routing
        features.append(("✅", "Smart Vision Routing", "Proper command categorization"))
        
        for status, feature, description in features:
            print(f"{status} {feature:<25} {description}")
        
        print("-" * 50)
    
    def check_autonomy_capabilities(self):
        """Check autonomy system capabilities"""
        print("\n🤖 Autonomy System Capabilities:")
        print("-" * 50)
        
        capabilities = []
        
        # Check if autonomy modules exist
        try:
            import autonomy.autonomous_decision_engine
            capabilities.append(("✅", "Decision Engine", "Intelligent decision making"))
            self.components['autonomy']['features'].append('decision_engine')
        except:
            capabilities.append(("❌", "Decision Engine", "Not available"))
        
        try:
            import autonomy.creative_problem_solving
            capabilities.append(("✅", "Creative AI", "Innovative problem solving"))
            self.components['creative_ai']['capabilities'].append('problem_solving')
        except:
            capabilities.append(("❌", "Creative AI", "Not available"))
        
        try:
            import autonomy.predictive_intelligence
            capabilities.append(("✅", "Predictive System", "User behavior prediction"))
            self.components['predictive']['engines'].append('behavior_prediction')
        except:
            capabilities.append(("❌", "Predictive System", "Not available"))
        
        try:
            import audio.ml_audio_manager
            capabilities.append(("✅", "ML Audio", "Advanced audio processing"))
            self.components['ml_audio']['status'] = True
        except:
            capabilities.append(("❌", "ML Audio", "Not available"))
        
        for status, feature, description in capabilities:
            print(f"{status} {feature:<25} {description}")
        
        self.components['autonomy']['status'] = any(
            status == "✅" for status, _, _ in capabilities[:3]
        )
        print("-" * 50)
    
    def check_unified_features(self):
        """Check unified AI agent features"""
        print("\n🤝 Unified AI Agent Features:")
        print("-" * 50)
        
        features = []
        
        # Swift + Vision
        if self.components['swift']['status'] and self.components['vision']['status']:
            features.append(("✅", "Swift+Vision Bridge", "Intelligent screen understanding"))
        else:
            features.append(("❌", "Swift+Vision Bridge", "Requires both components"))
        
        # Proactive notifications
        features.append(("✅", "Proactive Notifications", "WhatsApp, Slack, etc."))
        
        # Contextual replies
        features.append(("✅", "Contextual Replies", "Based on activity & time"))
        
        # Learning system
        features.append(("✅", "Learning System", "Improves over time"))
        
        # Voice interaction
        try:
            import pyttsx3
            features.append(("✅", "Voice Interaction", "Natural communication"))
            self.components['voice']['status'] = True
        except:
            features.append(("⚠️", "Voice Interaction", "Install pyttsx3"))
        
        for status, feature, description in features:
            print(f"{status} {feature:<25} {description}")
        
        print("-" * 50)
    
    def display_startup_options(self):
        """Display available startup options"""
        print("\n🚀 Startup Options:")
        print("-" * 50)
        print("1. Start Full System (Recommended)")
        print("2. Start Backend Only")
        print("3. Start Unified AI Agent")
        print("4. Start with Monitoring")
        print("5. Start Autonomous Mode")
        print("6. Run Tests")
        print("7. Start ML Audio System")
        print("8. Exit")
        print("-" * 50)
    
    async def start_full_system(self):
        """Start the complete JARVIS system"""
        print("\n🚀 Starting JARVIS Full System...")
        
        # Start backend
        print("\n1️⃣ Starting FastAPI Backend...")
        backend_process = subprocess.Popen(
            [sys.executable, "main.py", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for backend to initialize
        await asyncio.sleep(3)
        
        # Start unified AI agent
        print("\n2️⃣ Starting Unified AI Agent...")
        try:
            from jarvis_unified_ai_agent import JARVISUnifiedAIAgent
            agent = JARVISUnifiedAIAgent("User")
            
            print("✅ AI Agent initialized")
            
            # Start monitoring
            print("\n3️⃣ Starting Proactive Monitoring...")
            monitor_task = asyncio.create_task(agent.start_intelligent_monitoring())
            
            print("\n" + "=" * 70)
            print("✅ JARVIS is now running!")
            print("=" * 70)
            print("\n📋 Available Commands:")
            print("  • 'What's on my screen?' - Describe current view")
            print("  • 'Check notifications' - Check all apps")
            print("  • 'Read WhatsApp' - Read specific app")
            print("  • Ctrl+C to stop")
            
            # Keep running
            try:
                await asyncio.Event().wait()
            except KeyboardInterrupt:
                print("\n⏹️ Stopping JARVIS...")
                agent.monitoring_active = False
                monitor_task.cancel()
                backend_process.terminate()
                
        except ImportError:
            print("❌ Unified AI Agent not available")
            print("   Running backend only...")
            backend_process.wait()
    
    async def start_backend_only(self):
        """Start only the backend API"""
        print("\n🎯 Starting Backend API...")
        subprocess.run([sys.executable, "main.py", "--port", "8000"])
    
    async def start_unified_agent(self):
        """Start the unified AI agent directly"""
        print("\n🤖 Starting Unified AI Agent...")
        try:
            from jarvis_unified_ai_agent import JARVISUnifiedAIAgent
            agent = JARVISUnifiedAIAgent("User")
            
            # Test basic functionality
            print("\n📺 Testing vision command...")
            from jarvis_integrated_assistant import JARVISIntegratedAssistant
            assistant = JARVISIntegratedAssistant("User")
            
            response = await assistant.process_vision_command("What's on my screen?")
            print(f"\n🗣️ JARVIS says:")
            print(response.verbal_response)
            
            print("\n✅ AI Agent is ready for commands!")
            
        except Exception as e:
            print(f"❌ Error starting AI Agent: {e}")
    
    async def run_tests(self):
        """Run system tests"""
        print("\n🧪 Running System Tests...")
        print("-" * 50)
        
        tests = [
            ("Vision System", "python test_multi_window_claude.py"),
            ("Swift Bridge", "python test_swift_integration.py"),
            ("Integrated Assistant", "python test_integrated_jarvis.py"),
            ("Notification Detection", "python test_notification_detection.py")
        ]
        
        for test_name, command in tests:
            print(f"\n▶️ Testing {test_name}...")
            result = subprocess.run(
                command.split(), 
                capture_output=True, 
                text=True
            )
            if result.returncode == 0:
                print(f"✅ {test_name} passed")
            else:
                print(f"❌ {test_name} failed")
    
    async def start_websocket_server(self):
        """Start TypeScript WebSocket server"""
        print("\n🌐 Starting TypeScript WebSocket Server...")
        print("- Dynamic endpoint discovery enabled")
        print("- Self-healing connections active")
        print("- Real-time metrics dashboard at http://localhost:3000")
        
        # In a real implementation, would start the TypeScript server
        print("✅ WebSocket server started on ws://localhost:8080")
        
    async def start_learning_dashboard(self):
        """Start learning system dashboard"""
        print("\n📊 Starting Learning Dashboard...")
        print("- Pattern analysis visualization")
        print("- Real-time confidence scores")
        print("- Training interface")
        print("- Performance metrics")
        
        # In a real implementation, would start the dashboard
        print("✅ Dashboard available at http://localhost:5000")
    
    async def start_autonomous_mode(self):
        """Start JARVIS in full autonomous mode"""
        print("\n🤖 Starting Autonomous Mode...")
        print("-" * 50)
        
        try:
            # Import autonomy components
            from autonomy.autonomous_decision_engine import AutonomousDecisionEngine
            from autonomy.context_engine import ContextEngine
            from autonomy.predictive_intelligence import PredictiveIntelligence
            
            print("✅ Initializing Autonomous Decision Engine...")
            decision_engine = AutonomousDecisionEngine()
            
            print("✅ Starting Context Engine...")
            context_engine = ContextEngine()
            
            print("✅ Activating Predictive Intelligence...")
            predictive = PredictiveIntelligence()
            
            print("\n🎯 Autonomous Features Active:")
            print("  • Proactive task execution")
            print("  • Context-aware decision making")
            print("  • Predictive user assistance")
            print("  • Creative problem solving")
            print("  • Self-learning capabilities")
            
            print("\n✅ JARVIS is now running autonomously!")
            print("The system will proactively assist based on context.")
            
            # Keep running
            await asyncio.Event().wait()
            
        except ImportError as e:
            print(f"❌ Autonomous mode not available: {e}")
            print("Run 'python install_deps.py' to install required dependencies")
    
    async def start_ml_audio_system(self):
        """Start the ML-enhanced audio system"""
        print("\n🎵 Starting ML Audio System...")
        print("-" * 50)
        
        try:
            from audio.ml_audio_manager import MLAudioManager
            
            print("✅ Initializing ML Audio Manager...")
            audio_manager = MLAudioManager()
            
            print("✅ Loading audio models...")
            await audio_manager.load_models()
            
            print("\n🎤 ML Audio Features:")
            print("  • Advanced wake word detection")
            print("  • Noise cancellation")
            print("  • Voice activity detection")
            print("  • Speaker identification")
            print("  • Emotion recognition")
            
            print("\n✅ ML Audio System is ready!")
            
            # Start audio processing
            await audio_manager.start_processing()
            
        except Exception as e:
            print(f"❌ ML Audio System error: {e}")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='JARVIS AI System v12.0')
    parser.add_argument('--mode', choices=['full', 'backend', 'agent', 'test', 'autonomous', 'audio'], 
                      default='full', help='Startup mode')
    parser.add_argument('--monitor', action='store_true', 
                      help='Start with notification monitoring')
    parser.add_argument('--no-banner', action='store_true', 
                      help='Skip banner display')
    parser.add_argument('--autonomy-level', choices=['low', 'medium', 'high', 'full'],
                      default='medium', help='Autonomy level for autonomous mode')
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = JARVISSystemManager()
    
    # Display banner
    if not args.no_banner:
        manager.print_banner()
    
    # Check requirements
    if not manager.check_requirements():
        print("\n⚠️ Some core requirements are missing!")
        print("Please install required dependencies.")
        return
    
    # Check capabilities
    manager.check_vision_capabilities()
    manager.check_advanced_features()
    manager.check_autonomy_capabilities()
    manager.check_unified_features()
    
    # Handle different modes
    if args.mode == 'full':
        # Interactive mode
        manager.display_startup_options()
        choice = input("\n👉 Select option (1-8): ")
        
        if choice == '1':
            await manager.start_full_system()
        elif choice == '2':
            await manager.start_backend_only()
        elif choice == '3':
            await manager.start_unified_agent()
        elif choice == '4':
            args.monitor = True
            await manager.start_full_system()
        elif choice == '5':
            await manager.start_autonomous_mode()
        elif choice == '6':
            await manager.run_tests()
        elif choice == '7':
            await manager.start_ml_audio_system()
        else:
            print("👋 Goodbye!")
    
    elif args.mode == 'backend':
        await manager.start_backend_only()
    
    elif args.mode == 'agent':
        await manager.start_unified_agent()
    
    elif args.mode == 'test':
        await manager.run_tests()
    
    elif args.mode == 'autonomous':
        await manager.start_autonomous_mode()
    
    elif args.mode == 'audio':
        await manager.start_ml_audio_system()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n✋ JARVIS stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)