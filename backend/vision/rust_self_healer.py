"""
Rust Component Self-Healer for JARVIS.
Automatically diagnoses and fixes issues preventing Rust components from loading.
"""

import os
import sys
import asyncio
import subprocess
import logging
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import psutil

logger = logging.getLogger(__name__)

class RustIssueType(Enum):
    """Types of issues that can prevent Rust from working."""
    NOT_BUILT = "not_built"
    BUILD_FAILED = "build_failed"
    MISSING_DEPENDENCIES = "missing_dependencies"
    INCOMPATIBLE_VERSION = "incompatible_version"
    CORRUPTED_BINARY = "corrupted_binary"
    MISSING_RUSTUP = "missing_rustup"
    WRONG_TARGET = "wrong_target"
    PERMISSION_ERROR = "permission_error"
    OUT_OF_MEMORY = "out_of_memory"
    UNKNOWN = "unknown"

class FixStrategy(Enum):
    """Strategies for fixing Rust issues."""
    BUILD = "build"
    REBUILD = "rebuild"
    INSTALL_DEPS = "install_dependencies"
    INSTALL_RUST = "install_rust"
    CLEAN_BUILD = "clean_and_build"
    UPDATE_RUST = "update_rust"
    FIX_PERMISSIONS = "fix_permissions"
    FREE_MEMORY = "free_memory"
    RETRY_LATER = "retry_later"

class RustSelfHealer:
    """
    Automatically diagnoses and fixes Rust component issues.
    Monitors component health and attempts to restore functionality.
    """
    
    def __init__(self, check_interval: int = 300, max_retries: int = 3):
        """
        Initialize the self-healer.
        
        Args:
            check_interval: Seconds between health checks (default: 5 minutes)
            max_retries: Maximum fix attempts before giving up
        """
        self.check_interval = check_interval
        self.max_retries = max_retries
        self.vision_dir = Path(__file__).parent
        self.rust_core_dir = self.vision_dir / "jarvis-rust-core"
        self.backend_dir = self.vision_dir.parent
        
        self._running = False
        self._check_task: Optional[asyncio.Task] = None
        self._fix_history: List[Dict[str, Any]] = []
        self._retry_counts: Dict[RustIssueType, int] = {}
        self._last_successful_build: Optional[datetime] = None
        
    async def start(self):
        """Start the self-healing system."""
        if self._running:
            return
            
        self._running = True
        logger.info("Rust self-healer started")
        
        # Do initial diagnosis
        await self.diagnose_and_fix()
        
        # Start periodic checks
        self._check_task = asyncio.create_task(self._periodic_check())
        
    async def stop(self):
        """Stop the self-healing system."""
        self._running = False
        
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Rust self-healer stopped")
        
    async def _periodic_check(self):
        """Periodically check and fix Rust components."""
        while self._running:
            try:
                await asyncio.sleep(self.check_interval)
                
                if self._running:
                    # Check if Rust is working
                    if not await self._is_rust_working():
                        logger.info("Rust components not working, attempting to fix...")
                        await self.diagnose_and_fix()
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic check: {e}")
                
    async def _is_rust_working(self) -> bool:
        """Check if Rust components are currently working."""
        try:
            import jarvis_rust_core
            # Try to access a component
            if hasattr(jarvis_rust_core, 'RustAdvancedMemoryPool'):
                return True
        except ImportError:
            pass
        return False
        
    async def diagnose_and_fix(self) -> bool:
        """
        Diagnose issues and attempt to fix them.
        Returns True if fixed successfully.
        """
        logger.info("Diagnosing Rust component issues...")
        
        # Diagnose the issue
        issue_type, details = await self._diagnose_issue()
        
        logger.info(f"Diagnosed issue: {issue_type.value} - {details}")
        
        # Check retry limit
        retry_count = self._retry_counts.get(issue_type, 0)
        if retry_count >= self.max_retries:
            logger.warning(f"Max retries ({self.max_retries}) reached for {issue_type.value}")
            return False
            
        # Determine fix strategy
        strategy = self._determine_fix_strategy(issue_type, details)
        
        logger.info(f"Applying fix strategy: {strategy.value}")
        
        # Apply the fix
        success = await self._apply_fix(strategy, issue_type, details)
        
        # Update retry count
        if success:
            self._retry_counts[issue_type] = 0
            self._last_successful_build = datetime.now()
            logger.info("✅ Fix applied successfully!")
        else:
            self._retry_counts[issue_type] = retry_count + 1
            logger.warning(f"Fix failed, retry count: {self._retry_counts[issue_type]}")
            
        # Record in history
        self._fix_history.append({
            'timestamp': datetime.now(),
            'issue': issue_type.value,
            'strategy': strategy.value,
            'success': success,
            'details': details
        })
        
        # Cleanup old history (keep last 100 entries)
        if len(self._fix_history) > 100:
            self._fix_history = self._fix_history[-100:]
            
        return success
        
    async def _diagnose_issue(self) -> Tuple[RustIssueType, Dict[str, Any]]:
        """Diagnose what's preventing Rust from working."""
        details = {}
        
        # Check if Rust is installed
        if not await self._is_rust_installed():
            return RustIssueType.MISSING_RUSTUP, {'error': 'Rust not installed'}
            
        # Check if target directory exists
        target_dir = self.rust_core_dir / "target"
        if not target_dir.exists():
            return RustIssueType.NOT_BUILT, {'error': 'Never built'}
            
        # Check for library file
        lib_path = self._get_library_path()
        if not lib_path or not lib_path.exists():
            # Check if there's a build log
            build_log = self.rust_core_dir / "build.log"
            if build_log.exists():
                # Analyze build log
                log_content = build_log.read_text()
                if "error[E0463]" in log_content or "can't find crate" in log_content:
                    missing_crates = self._extract_missing_crates(log_content)
                    return RustIssueType.MISSING_DEPENDENCIES, {'missing_crates': missing_crates}
                elif "error: could not compile" in log_content:
                    return RustIssueType.BUILD_FAILED, {'log': log_content[-1000:]}  # Last 1000 chars
                    
            return RustIssueType.NOT_BUILT, {'error': 'Library file missing'}
            
        # Check if we can import it
        try:
            # Add to Python path temporarily
            sys.path.insert(0, str(self.rust_core_dir / "target" / "release"))
            import jarvis_rust_core
            
            # Check if it has expected components
            expected_components = ['RustAdvancedMemoryPool', 'RustImageProcessor']
            missing = [c for c in expected_components if not hasattr(jarvis_rust_core, c)]
            
            if missing:
                return RustIssueType.INCOMPATIBLE_VERSION, {'missing_components': missing}
                
            # If we get here, it should be working
            return RustIssueType.UNKNOWN, {'error': 'Components exist but not loading properly'}
            
        except ImportError as e:
            error_msg = str(e)
            
            # Check for common import errors
            if "symbol not found" in error_msg:
                return RustIssueType.INCOMPATIBLE_VERSION, {'error': error_msg}
            elif "Permission denied" in error_msg:
                return RustIssueType.PERMISSION_ERROR, {'file': lib_path}
            elif "image not found" in error_msg or "Library not loaded" in error_msg:
                return RustIssueType.CORRUPTED_BINARY, {'error': error_msg}
            else:
                return RustIssueType.UNKNOWN, {'error': error_msg}
                
        except Exception as e:
            return RustIssueType.UNKNOWN, {'error': str(e)}
            
        finally:
            # Remove from path
            if str(self.rust_core_dir / "target" / "release") in sys.path:
                sys.path.remove(str(self.rust_core_dir / "target" / "release"))
                
    def _determine_fix_strategy(self, issue: RustIssueType, details: Dict[str, Any]) -> FixStrategy:
        """Determine the best fix strategy for an issue."""
        strategy_map = {
            RustIssueType.NOT_BUILT: FixStrategy.BUILD,
            RustIssueType.BUILD_FAILED: FixStrategy.CLEAN_BUILD,
            RustIssueType.MISSING_DEPENDENCIES: FixStrategy.INSTALL_DEPS,
            RustIssueType.INCOMPATIBLE_VERSION: FixStrategy.REBUILD,
            RustIssueType.CORRUPTED_BINARY: FixStrategy.CLEAN_BUILD,
            RustIssueType.MISSING_RUSTUP: FixStrategy.INSTALL_RUST,
            RustIssueType.WRONG_TARGET: FixStrategy.REBUILD,
            RustIssueType.PERMISSION_ERROR: FixStrategy.FIX_PERMISSIONS,
            RustIssueType.OUT_OF_MEMORY: FixStrategy.FREE_MEMORY,
            RustIssueType.UNKNOWN: FixStrategy.CLEAN_BUILD
        }
        
        # Check if we have enough memory for building
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        if available_memory < 2.0:  # Need at least 2GB for Rust builds
            return FixStrategy.FREE_MEMORY
            
        return strategy_map.get(issue, FixStrategy.RETRY_LATER)
        
    async def _apply_fix(self, strategy: FixStrategy, issue: RustIssueType, details: Dict[str, Any]) -> bool:
        """Apply a fix strategy."""
        try:
            if strategy == FixStrategy.BUILD:
                return await self._build_rust_components()
                
            elif strategy == FixStrategy.REBUILD:
                await self._clean_build_artifacts()
                return await self._build_rust_components()
                
            elif strategy == FixStrategy.CLEAN_BUILD:
                await self._clean_build_artifacts()
                await self._reset_cargo_cache()
                return await self._build_rust_components()
                
            elif strategy == FixStrategy.INSTALL_DEPS:
                missing_crates = details.get('missing_crates', [])
                await self._install_missing_crates(missing_crates)
                return await self._build_rust_components()
                
            elif strategy == FixStrategy.INSTALL_RUST:
                if await self._install_rust():
                    return await self._build_rust_components()
                return False
                
            elif strategy == FixStrategy.UPDATE_RUST:
                if await self._update_rust():
                    return await self._build_rust_components()
                return False
                
            elif strategy == FixStrategy.FIX_PERMISSIONS:
                lib_path = details.get('file')
                if lib_path and await self._fix_permissions(lib_path):
                    return True
                return False
                
            elif strategy == FixStrategy.FREE_MEMORY:
                await self._free_memory()
                # Wait a bit for memory to be freed
                await asyncio.sleep(5)
                return await self._build_rust_components()
                
            elif strategy == FixStrategy.RETRY_LATER:
                # Just wait and let the periodic check try again
                return False
                
            else:
                logger.warning(f"Unknown fix strategy: {strategy}")
                return False
                
        except Exception as e:
            logger.error(f"Error applying fix {strategy.value}: {e}")
            return False
            
    async def _is_rust_installed(self) -> bool:
        """Check if Rust is installed."""
        try:
            result = await self._run_command(["rustc", "--version"])
            return result.returncode == 0
        except FileNotFoundError:
            return False
            
    async def _install_rust(self) -> bool:
        """Install Rust using rustup."""
        logger.info("Installing Rust...")
        
        # Download and run rustup
        if sys.platform == "win32":
            # Windows
            installer_url = "https://win.rustup.rs"
            installer_path = "rustup-init.exe"
        else:
            # Unix-like (macOS, Linux)
            installer_url = "https://sh.rustup.rs"
            installer_path = "rustup-init.sh"
            
        try:
            # Download installer
            import urllib.request
            urllib.request.urlretrieve(installer_url, installer_path)
            
            if sys.platform != "win32":
                # Make executable on Unix
                os.chmod(installer_path, 0o755)
                
            # Run installer
            cmd = [installer_path, "-y", "--default-toolchain", "stable"]
            if sys.platform != "win32":
                cmd = ["sh", installer_path, "-y", "--default-toolchain", "stable"]
                
            result = await self._run_command(cmd)
            
            # Cleanup
            os.remove(installer_path)
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Failed to install Rust: {e}")
            return False
            
    async def _update_rust(self) -> bool:
        """Update Rust to latest stable."""
        logger.info("Updating Rust...")
        result = await self._run_command(["rustup", "update", "stable"])
        return result.returncode == 0
        
    async def _build_rust_components(self, retry_count: int = 0, max_retries: int = 3) -> bool:
        """Build Rust components with exponential backoff retry."""
        logger.info("Building Rust components...")
        
        # Use the build script if available
        build_script = self.vision_dir / "build_rust_components.py"
        
        for attempt in range(max_retries):
            if attempt > 0:
                # Exponential backoff: 2^attempt seconds (2s, 4s, 8s)
                wait_time = 2 ** attempt
                logger.info(f"Retry {attempt}/{max_retries} after {wait_time}s...")
                await asyncio.sleep(wait_time)
            
            try:
                if build_script.exists():
                    result = await self._run_command(
                        [sys.executable, str(build_script)],
                        cwd=str(self.vision_dir),
                        capture_output=True
                    )
                else:
                    # Direct cargo build
                    result = await self._run_command(
                        ["cargo", "build", "--release", "--features", "python-bindings"],
                        cwd=str(self.rust_core_dir),
                        capture_output=True
                    )
                
                # Save build log
                build_log = self.rust_core_dir / "build.log"
                build_log.write_text(result.stdout + "\n" + result.stderr)
                
                if result.returncode == 0:
                    # Run maturin if needed
                    if (self.rust_core_dir / "pyproject.toml").exists():
                        maturin_result = await self._run_command(
                            ["maturin", "develop", "--release"],
                            cwd=str(self.rust_core_dir)
                        )
                        if maturin_result.returncode == 0:
                            logger.info("✅ Build successful!")
                            return True
                        else:
                            logger.warning(f"Maturin failed: {maturin_result.stderr}")
                    else:
                        logger.info("✅ Build successful!")
                        return True
                else:
                    # Analyze failure
                    if "error[E0463]" in result.stderr or "can't find crate" in result.stderr:
                        # Missing dependencies - try to install them
                        missing_crates = self._extract_missing_crates(result.stderr)
                        if missing_crates and attempt < max_retries - 1:
                            logger.info(f"Missing crates detected: {missing_crates}")
                            await self._install_missing_crates(missing_crates)
                            continue  # Retry with newly installed deps
                    elif "could not find `Cargo.toml`" in result.stderr:
                        logger.error("Cargo.toml not found!")
                        return False
                    elif "no space left on device" in result.stderr:
                        # Try to free space
                        await self._free_memory()
                        continue
                        
            except Exception as e:
                logger.error(f"Build attempt {attempt + 1} failed: {e}")
        
        logger.error(f"Build failed after {max_retries} attempts")
        return False
            
    async def _clean_build_artifacts(self):
        """Clean build artifacts."""
        logger.info("Cleaning build artifacts...")
        
        target_dir = self.rust_core_dir / "target"
        if target_dir.exists():
            shutil.rmtree(target_dir)
            
        # Remove any .so/.dylib/.dll files
        for pattern in ["*.so", "*.dylib", "*.dll", "*.pyd"]:
            for file in self.rust_core_dir.glob(pattern):
                file.unlink()
                
    async def _reset_cargo_cache(self):
        """Reset Cargo cache for this project."""
        logger.info("Resetting Cargo cache...")
        
        # Remove Cargo.lock
        cargo_lock = self.rust_core_dir / "Cargo.lock"
        if cargo_lock.exists():
            cargo_lock.unlink()
            
    async def _install_missing_crates(self, crates: List[str]):
        """Install missing crates with automatic version resolution."""
        if not crates:
            return
            
        logger.info(f"Installing missing crates: {crates}")
        
        # Add to Cargo.toml if needed
        cargo_toml = self.rust_core_dir / "Cargo.toml"
        if cargo_toml.exists():
            try:
                import toml
            except ImportError:
                # Install toml if not available
                await self._run_command([sys.executable, "-m", "pip", "install", "toml"])
                import toml
            
            # Load existing Cargo.toml
            cargo_data = toml.load(cargo_toml)
            
            # Common crate versions that work well together
            crate_versions = {
                'pyo3': '0.20',
                'numpy': '0.20',
                'ndarray': '0.15',
                'rayon': '1.8',
                'serde': '1.0',
                'serde_json': '1.0',
                'tokio': '1.35',
                'async-trait': '0.1',
                'anyhow': '1.0',
                'thiserror': '1.0',
                'log': '0.4',
                'env_logger': '0.10',
                'crossbeam': '0.8',
                'parking_lot': '0.12',
                'once_cell': '1.19',
                'metal': '0.27',
                'objc': '0.2',
                'cocoa': '0.25',
                'core-foundation': '0.9',
                'fnv': '1.0',
                'ordered-float': '4.2',
                'nalgebra-sparse': '0.9',
                'maturin': '1.4'
            }
            
            # Ensure dependencies section exists
            if 'dependencies' not in cargo_data:
                cargo_data['dependencies'] = {}
            
            modified = False
            for crate in crates:
                if crate not in cargo_data['dependencies']:
                    # Add with known good version or latest
                    version = crate_versions.get(crate, '*')
                    cargo_data['dependencies'][crate] = version
                    logger.info(f"Adding {crate} = \"{version}\" to Cargo.toml")
                    modified = True
            
            if modified:
                # Write back to Cargo.toml
                with open(cargo_toml, 'w') as f:
                    toml.dump(cargo_data, f)
                logger.info("Updated Cargo.toml with missing dependencies")
                    
    def _extract_missing_crates(self, build_log: str) -> List[str]:
        """Extract missing crate names from build log."""
        import re
        
        # Pattern to match missing crates
        pattern = r"can't find crate for `(\w+)`"
        matches = re.findall(pattern, build_log)
        
        return list(set(matches))
        
    async def _fix_permissions(self, file_path: Path) -> bool:
        """Fix file permissions."""
        logger.info(f"Fixing permissions for {file_path}")
        
        try:
            # Make readable and executable
            os.chmod(file_path, 0o755)
            return True
        except Exception as e:
            logger.error(f"Failed to fix permissions: {e}")
            return False
            
    async def _free_memory(self):
        """Try to free up memory for building."""
        logger.info("Attempting to free memory...")
        
        # Clear Python caches
        import gc
        gc.collect()
        
        # Clear system caches (macOS specific)
        if sys.platform == "darwin":
            await self._run_command(["sudo", "purge"], check=False)
            
    def _get_library_path(self) -> Optional[Path]:
        """Get the expected library path."""
        if sys.platform == "darwin":
            lib_name = "libjarvis_rust_core.dylib"
        elif sys.platform == "win32":
            lib_name = "jarvis_rust_core.dll"
        else:
            lib_name = "libjarvis_rust_core.so"
            
        lib_path = self.rust_core_dir / "target" / "release" / lib_name
        
        if lib_path.exists():
            return lib_path
            
        # Check for maturin output
        for pattern in ["*.so", "*.dylib", "*.pyd"]:
            for file in self.rust_core_dir.glob(pattern):
                if "jarvis_rust_core" in file.name:
                    return file
                    
        return None
        
    async def _run_command(self, cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
        """Run a command asynchronously."""
        if 'capture_output' not in kwargs:
            kwargs['capture_output'] = True
        if 'text' not in kwargs:
            kwargs['text'] = True
            
        # Set a longer timeout for build commands
        timeout = 300  # 5 minutes default
        if 'cargo' in cmd[0] or 'maturin' in cmd[0] or 'build' in str(cmd):
            timeout = 600  # 10 minutes for build commands
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE if kwargs.get('capture_output') else None,
            stderr=asyncio.subprocess.PIPE if kwargs.get('capture_output') else None,
            cwd=kwargs.get('cwd')
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise Exception(f"Command timed out after {timeout} seconds: {' '.join(cmd)}")
        
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode,
            stdout=stdout.decode() if stdout else "",
            stderr=stderr.decode() if stderr else ""
        )
        
    def get_health_report(self) -> Dict[str, Any]:
        """Get a health report of the self-healing system."""
        # Recent fixes
        recent_fixes = self._fix_history[-10:] if self._fix_history else []
        
        # Success rate
        if self._fix_history:
            success_count = sum(1 for f in self._fix_history if f['success'])
            success_rate = success_count / len(self._fix_history)
        else:
            success_rate = 0.0
            
        return {
            'running': self._running,
            'last_successful_build': self._last_successful_build.isoformat() if self._last_successful_build else None,
            'retry_counts': dict(self._retry_counts),
            'recent_fixes': recent_fixes,
            'total_fix_attempts': len(self._fix_history),
            'success_rate': success_rate,
            'is_rust_working': asyncio.create_task(self._is_rust_working())
        }


# Global instance
_self_healer: Optional[RustSelfHealer] = None

def get_self_healer() -> RustSelfHealer:
    """Get the global self-healer instance."""
    global _self_healer
    if _self_healer is None:
        _self_healer = RustSelfHealer()
    return _self_healer

async def start_self_healing():
    """Start the self-healing system."""
    healer = get_self_healer()
    await healer.start()
    return healer