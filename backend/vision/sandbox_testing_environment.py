#!/usr/bin/env python3
"""
Sandbox Testing Environment for Vision System v2.0
Provides isolated execution environment for testing generated capabilities
"""

import asyncio
import sys
import os
import tempfile
import shutil
import resource
import signal
import time
import traceback
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import subprocess
import json
import psutil
import docker
from contextlib import contextmanager
import uuid

logger = logging.getLogger(__name__)


@dataclass
class SandboxConfig:
    """Configuration for sandbox environment"""
    # Resource limits
    max_cpu_percent: float = 50.0
    max_memory_mb: int = 512
    max_disk_mb: int = 100
    max_execution_time: float = 30.0
    max_processes: int = 5
    
    # Network configuration
    allow_network: bool = False
    allowed_hosts: List[str] = field(default_factory=list)
    
    # File system configuration
    temp_dir_prefix: str = "sandbox_"
    readonly_paths: List[str] = field(default_factory=lambda: ["/usr", "/lib", "/bin"])
    writable_paths: List[str] = field(default_factory=lambda: ["/tmp"])
    
    # Container configuration
    use_docker: bool = True
    docker_image: str = "python:3.9-slim"
    docker_memory_limit: str = "512m"
    docker_cpu_quota: int = 50000  # 50% of one CPU


@dataclass
class TestCase:
    """Test case for capability validation"""
    test_id: str
    name: str
    input_data: Dict[str, Any]
    expected_output: Optional[Dict[str, Any]] = None
    expected_behavior: Optional[str] = None
    timeout: float = 10.0
    validation_func: Optional[Callable] = None


@dataclass
class TestResult:
    """Result of a test execution"""
    test_id: str
    success: bool
    duration: float
    output: Optional[Any] = None
    error: Optional[str] = None
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)


class ProcessSandbox:
    """Process-based sandbox for lightweight isolation"""
    
    def __init__(self, config: SandboxConfig):
        self.config = config
        self.temp_dir = None
        self.process = None
        
    @contextmanager
    def sandbox_context(self):
        """Create sandbox context"""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix=self.config.temp_dir_prefix)
        
        try:
            # Set up resource limits
            self._set_resource_limits()
            
            yield self
            
        finally:
            # Cleanup
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                
    def _set_resource_limits(self):
        """Set resource limits for the process"""
        if sys.platform != "win32":
            # Memory limit
            memory_bytes = self.config.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            
            # CPU time limit
            cpu_seconds = int(self.config.max_execution_time)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            
            # Process limit
            resource.setrlimit(resource.RLIMIT_NPROC, 
                             (self.config.max_processes, self.config.max_processes))
    
    async def execute_code(self, code: str, test_input: Dict[str, Any]) -> TestResult:
        """Execute code in sandboxed process"""
        start_time = time.time()
        
        # Create execution script
        script_path = os.path.join(self.temp_dir, "sandbox_script.py")
        
        # Wrapper code for execution
        wrapper_code = f'''
import sys
import json
import asyncio

# Sandboxed code
{code}

# Test execution
async def run_test():
    try:
        # Parse input
        test_input = {json.dumps(test_input)}
        
        # Execute the handler (assuming it's defined in the code)
        # This is simplified - in practice, dynamically find the handler
        result = await handle_capability(test_input['command'], test_input.get('context'))
        
        # Output result
        print(json.dumps({{'success': True, 'result': result}}))
        
    except Exception as e:
        import traceback
        print(json.dumps({{
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }}))

# Run the test
asyncio.run(run_test())
'''
        
        with open(script_path, 'w') as f:
            f.write(wrapper_code)
        
        # Execute in subprocess
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.temp_dir,
                env={**os.environ, 'PYTHONPATH': self.temp_dir}
            )
            
            # Wait with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.max_execution_time
                )
            except asyncio.TimeoutError:
                process.kill()
                return TestResult(
                    test_id="process_sandbox",
                    success=False,
                    duration=time.time() - start_time,
                    error="Execution timeout",
                    logs=[f"Process killed after {self.config.max_execution_time}s"]
                )
            
            duration = time.time() - start_time
            
            # Parse output
            try:
                output = json.loads(stdout.decode())
                return TestResult(
                    test_id="process_sandbox",
                    success=output.get('success', False),
                    duration=duration,
                    output=output.get('result'),
                    error=output.get('error'),
                    logs=stderr.decode().split('\n') if stderr else []
                )
            except json.JSONDecodeError:
                return TestResult(
                    test_id="process_sandbox",
                    success=False,
                    duration=duration,
                    error="Invalid output format",
                    logs=[stdout.decode(), stderr.decode()]
                )
                
        except Exception as e:
            return TestResult(
                test_id="process_sandbox",
                success=False,
                duration=time.time() - start_time,
                error=str(e),
                logs=[traceback.format_exc()]
            )


class DockerSandbox:
    """Docker-based sandbox for stronger isolation"""
    
    def __init__(self, config: SandboxConfig):
        self.config = config
        self.client = None
        self.container = None
        
    def _init_docker(self):
        """Initialize Docker client"""
        try:
            self.client = docker.from_env()
            # Test connection
            self.client.ping()
            return True
        except Exception as e:
            logger.error(f"Docker initialization failed: {e}")
            return False
    
    @contextmanager
    def sandbox_context(self):
        """Create Docker sandbox context"""
        if not self._init_docker():
            raise RuntimeError("Docker not available")
            
        try:
            # Create container
            self.container = self.client.containers.create(
                self.config.docker_image,
                command="sleep infinity",
                detach=True,
                mem_limit=self.config.docker_memory_limit,
                cpu_quota=self.config.docker_cpu_quota,
                cpu_period=100000,
                network_mode="none" if not self.config.allow_network else "bridge",
                security_opt=["no-new-privileges"],
                read_only=True,
                tmpfs={'/tmp': 'size=100M,mode=1777'}
            )
            
            # Start container
            self.container.start()
            
            yield self
            
        finally:
            # Cleanup
            if self.container:
                try:
                    self.container.stop(timeout=5)
                    self.container.remove()
                except:
                    pass
                    
    async def execute_code(self, code: str, test_input: Dict[str, Any]) -> TestResult:
        """Execute code in Docker container"""
        start_time = time.time()
        
        # Create execution script
        script_content = f'''
import sys
import json
import asyncio

# Sandboxed code
{code}

# Test execution
async def run_test():
    try:
        # Parse input
        test_input = {json.dumps(test_input)}
        
        # Execute the handler
        result = await handle_capability(test_input['command'], test_input.get('context'))
        
        # Output result
        print(json.dumps({{'success': True, 'result': result}}))
        
    except Exception as e:
        import traceback
        print(json.dumps({{
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }}))

# Run the test
asyncio.run(run_test())
'''
        
        # Copy script to container
        script_path = "/tmp/sandbox_script.py"
        self.container.put_archive(
            "/tmp",
            self._create_tar_archive("sandbox_script.py", script_content)
        )
        
        # Execute script
        try:
            exec_result = self.container.exec_run(
                f"python {script_path}",
                demux=True,
                stream=False
            )
            
            duration = time.time() - start_time
            stdout, stderr = exec_result.output
            
            # Parse output
            if stdout:
                try:
                    output = json.loads(stdout.decode())
                    return TestResult(
                        test_id="docker_sandbox",
                        success=output.get('success', False),
                        duration=duration,
                        output=output.get('result'),
                        error=output.get('error'),
                        logs=[stderr.decode()] if stderr else []
                    )
                except json.JSONDecodeError:
                    return TestResult(
                        test_id="docker_sandbox",
                        success=False,
                        duration=duration,
                        error="Invalid output format",
                        logs=[stdout.decode(), stderr.decode() if stderr else ""]
                    )
            else:
                return TestResult(
                    test_id="docker_sandbox",
                    success=False,
                    duration=duration,
                    error="No output",
                    logs=[stderr.decode() if stderr else ""]
                )
                
        except Exception as e:
            return TestResult(
                test_id="docker_sandbox",
                success=False,
                duration=time.time() - start_time,
                error=str(e),
                logs=[traceback.format_exc()]
            )
    
    def _create_tar_archive(self, filename: str, content: str) -> bytes:
        """Create TAR archive with a single file"""
        import tarfile
        import io
        
        tar_stream = io.BytesIO()
        tar = tarfile.open(fileobj=tar_stream, mode='w')
        
        # Create file
        file_data = content.encode('utf-8')
        tarinfo = tarfile.TarInfo(name=filename)
        tarinfo.size = len(file_data)
        tarinfo.mode = 0o755
        
        tar.addfile(tarinfo, io.BytesIO(file_data))
        tar.close()
        
        return tar_stream.getvalue()


class SandboxTestRunner:
    """Manages sandbox testing for capabilities"""
    
    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self.test_results: Dict[str, List[TestResult]] = {}
        
    async def test_capability(
        self,
        capability_code: str,
        test_cases: List[TestCase],
        use_docker: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Test a capability with multiple test cases"""
        use_docker = use_docker if use_docker is not None else self.config.use_docker
        
        # Choose sandbox type
        if use_docker and self._docker_available():
            sandbox_class = DockerSandbox
        else:
            sandbox_class = ProcessSandbox
            if use_docker:
                logger.warning("Docker not available, falling back to process sandbox")
        
        results = []
        
        # Run each test case
        for test_case in test_cases:
            logger.info(f"Running test case: {test_case.name}")
            
            # Create sandbox
            sandbox = sandbox_class(self.config)
            
            with sandbox.sandbox_context():
                # Execute test
                result = await sandbox.execute_code(
                    capability_code,
                    test_case.input_data
                )
                
                # Update result with test info
                result.test_id = test_case.test_id
                
                # Custom validation if provided
                if test_case.validation_func and result.output:
                    try:
                        validation_passed = test_case.validation_func(result.output)
                        result.success = result.success and validation_passed
                    except Exception as e:
                        result.success = False
                        result.error = f"Validation error: {str(e)}"
                
                results.append(result)
                
                # Log result
                status = "✅" if result.success else "❌"
                logger.info(f"{status} Test {test_case.name}: {result.duration:.2f}s")
                
        # Aggregate results
        test_summary = {
            'total_tests': len(test_cases),
            'passed': sum(1 for r in results if r.success),
            'failed': sum(1 for r in results if not r.success),
            'average_duration': sum(r.duration for r in results) / len(results) if results else 0,
            'results': results,
            'sandbox_type': 'docker' if isinstance(sandbox, DockerSandbox) else 'process'
        }
        
        return test_summary
    
    def _docker_available(self) -> bool:
        """Check if Docker is available"""
        try:
            client = docker.from_env()
            client.ping()
            return True
        except:
            return False
    
    def create_standard_test_cases(self, capability_name: str) -> List[TestCase]:
        """Create standard test cases for a capability"""
        test_cases = []
        
        # Basic functionality test
        test_cases.append(TestCase(
            test_id=f"{capability_name}_basic",
            name="Basic Functionality",
            input_data={
                'command': f"Test {capability_name}",
                'context': {'test_mode': True}
            },
            timeout=5.0
        ))
        
        # Empty input test
        test_cases.append(TestCase(
            test_id=f"{capability_name}_empty",
            name="Empty Input",
            input_data={
                'command': "",
                'context': {}
            },
            timeout=5.0
        ))
        
        # Large input test
        test_cases.append(TestCase(
            test_id=f"{capability_name}_large",
            name="Large Input",
            input_data={
                'command': "x" * 1000,
                'context': {'data': ['item'] * 100}
            },
            timeout=10.0
        ))
        
        # Timeout test
        test_cases.append(TestCase(
            test_id=f"{capability_name}_timeout",
            name="Timeout Test",
            input_data={
                'command': "simulate timeout",
                'context': {'delay': 30}
            },
            timeout=5.0
        ))
        
        return test_cases
    
    async def benchmark_capability(
        self,
        capability_code: str,
        benchmark_input: Dict[str, Any],
        iterations: int = 10
    ) -> Dict[str, Any]:
        """Benchmark capability performance"""
        sandbox = ProcessSandbox(self.config)
        
        timings = []
        memory_usage = []
        
        with sandbox.sandbox_context():
            for i in range(iterations):
                # Monitor resources before execution
                process = psutil.Process()
                mem_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Execute
                start = time.time()
                result = await sandbox.execute_code(capability_code, benchmark_input)
                duration = time.time() - start
                
                # Monitor resources after execution
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                
                if result.success:
                    timings.append(duration)
                    memory_usage.append(mem_after - mem_before)
                    
        if timings:
            return {
                'iterations': len(timings),
                'avg_duration': sum(timings) / len(timings),
                'min_duration': min(timings),
                'max_duration': max(timings),
                'avg_memory_mb': sum(memory_usage) / len(memory_usage),
                'success_rate': len(timings) / iterations
            }
        else:
            return {
                'iterations': 0,
                'success_rate': 0.0,
                'error': 'All benchmark iterations failed'
            }


class MockScreenCapture:
    """Mock screen capture for testing vision capabilities"""
    
    @staticmethod
    def generate_test_screen_data(scenario: str = "default") -> Dict[str, Any]:
        """Generate mock screen data for testing"""
        scenarios = {
            "default": {
                "width": 1920,
                "height": 1080,
                "elements": [
                    {"type": "button", "text": "Submit", "x": 100, "y": 200, "width": 80, "height": 30},
                    {"type": "input", "text": "", "x": 100, "y": 150, "width": 200, "height": 30},
                    {"type": "text", "text": "Welcome", "x": 100, "y": 100, "width": 100, "height": 20}
                ]
            },
            "error_dialog": {
                "width": 1920,
                "height": 1080,
                "elements": [
                    {"type": "dialog", "title": "Error", "x": 760, "y": 440, "width": 400, "height": 200},
                    {"type": "text", "text": "An error occurred", "x": 800, "y": 500, "width": 320, "height": 20},
                    {"type": "button", "text": "OK", "x": 900, "y": 580, "width": 60, "height": 30}
                ]
            }
        }
        
        return scenarios.get(scenario, scenarios["default"])


# Singleton instance
_sandbox_runner: Optional[SandboxTestRunner] = None


def get_sandbox_test_runner(config: Optional[SandboxConfig] = None) -> SandboxTestRunner:
    """Get singleton instance of sandbox test runner"""
    global _sandbox_runner
    if _sandbox_runner is None:
        _sandbox_runner = SandboxTestRunner(config)
    return _sandbox_runner