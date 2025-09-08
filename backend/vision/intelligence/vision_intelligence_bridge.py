"""
Vision Intelligence Bridge - Integrates Python, Rust, and Swift components
Provides seamless integration between all vision intelligence components
"""

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import base64
from datetime import datetime

# Dynamic imports with fallbacks
try:
    from .visual_state_management_system import (
        VisualStateManagementSystem,
        ApplicationStateTracker,
        StateObservation
    )
except ImportError:
    VisualStateManagementSystem = None
    ApplicationStateTracker = None
    StateObservation = None

try:
    from .vsms_core import get_vsms
    VSMS_AVAILABLE = True
except ImportError:
    VSMS_AVAILABLE = False
    get_vsms = None

try:
    # Import Rust components (will be available after building)
    import vision_intelligence as rust_vi
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    rust_vi = None

logger = logging.getLogger(__name__)


class SwiftBridge:
    """Bridge to Swift Vision Intelligence components"""
    
    def __init__(self):
        self.swift_executable = Path(__file__).parent / "VisionIntelligence"
        self.swift_process = None
        self._ensure_swift_built()
    
    def _ensure_swift_built(self):
        """Ensure Swift component is built"""
        swift_source = Path(__file__).parent / "VisionIntelligence.swift"
        if swift_source.exists() and not self.swift_executable.exists():
            logger.info("Building Swift Vision Intelligence component...")
            try:
                subprocess.run([
                    "swiftc",
                    "-O",
                    "-framework", "Vision",
                    "-framework", "CoreML",
                    "-framework", "AppKit",
                    "-framework", "Accelerate",
                    str(swift_source),
                    "-o", str(self.swift_executable)
                ], check=True)
                logger.info("Swift component built successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to build Swift component: {e}")
    
    async def analyze_screenshot(self, image_data: bytes, app_id: str) -> Dict[str, Any]:
        """Analyze screenshot using Swift Vision framework"""
        if not self.swift_executable.exists():
            return {"error": "Swift component not available"}
        
        try:
            # Save image temporarily
            temp_path = Path(f"/tmp/jarvis_vision_{datetime.now().timestamp()}.png")
            with open(temp_path, 'wb') as f:
                f.write(image_data)
            
            # Call Swift analyzer
            result = subprocess.run([
                str(self.swift_executable),
                "analyze",
                str(temp_path),
                app_id
            ], capture_output=True, text=True)
            
            # Clean up
            temp_path.unlink()
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {"error": f"Swift analysis failed: {result.stderr}"}
                
        except Exception as e:
            logger.error(f"Swift bridge error: {e}")
            return {"error": str(e)}


class VisionIntelligenceBridge:
    """Main bridge coordinating Python, Rust, and Swift components"""
    
    def __init__(self):
        self.vsms = VisualStateManagementSystem() if VisualStateManagementSystem else None
        self.vsms_core = get_vsms() if VSMS_AVAILABLE else None
        self.swift_bridge = SwiftBridge()
        self.rust_pattern_matcher = None
        self.rust_feature_extractor = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize Rust components if available
        if RUST_AVAILABLE:
            try:
                rust_vi.initialize_vision_intelligence()
                self.rust_pattern_matcher = rust_vi.PatternMatcher()
                self.rust_feature_extractor = rust_vi.FeatureExtractor()
                logger.info("Rust vision intelligence components initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Rust components: {e}")
    
    async def analyze_visual_state(self, 
                                  screenshot: Union[bytes, np.ndarray], 
                                  app_id: str,
                                  metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze visual state using all available components
        
        Args:
            screenshot: Screenshot data as bytes or numpy array
            app_id: Application identifier
            metadata: Additional metadata
            
        Returns:
            Comprehensive analysis results
        """
        results = {
            'app_id': app_id,
            'timestamp': datetime.now().isoformat(),
            'components_used': []
        }
        
        # Convert screenshot to bytes if needed
        if isinstance(screenshot, np.ndarray):
            from PIL import Image
            import io
            img = Image.fromarray(screenshot)
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            screenshot_bytes = img_bytes.getvalue()
        else:
            screenshot_bytes = screenshot
        
        # Parallel analysis using all components
        tasks = []
        
        # Python VSMS analysis
        if self.vsms:
            tasks.append(self._analyze_with_python(screenshot, app_id, metadata))
        
        # VSMS Core analysis (full implementation)
        if self.vsms_core:
            tasks.append(self._analyze_with_vsms_core(screenshot, app_id))
        
        # Swift Vision framework analysis
        tasks.append(self._analyze_with_swift(screenshot_bytes, app_id))
        
        # Rust pattern matching
        if RUST_AVAILABLE:
            tasks.append(self._analyze_with_rust(screenshot_bytes, app_id))
        
        # Gather results
        analysis_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge results
        for i, result in enumerate(analysis_results):
            if isinstance(result, Exception):
                logger.error(f"Analysis component {i} failed: {result}")
                continue
            
            if isinstance(result, dict):
                component_name = result.get('component', f'component_{i}')
                results[component_name] = result
                results['components_used'].append(component_name)
        
        # Combine insights for final state determination
        final_state = self._determine_final_state(results)
        results['final_state'] = final_state
        
        return results
    
    async def _analyze_with_vsms_core(self, 
                                     screenshot: np.ndarray,
                                     app_id: str) -> Dict[str, Any]:
        """Analyze using VSMS Core implementation"""
        try:
            # Process through VSMS
            result = await self.vsms_core.process_visual_observation(screenshot, app_id)
            
            # Extract relevant information
            vsms_result = {
                'component': 'vsms_core',
                'app_id': result.get('app_id'),
                'detected_state': result.get('detected_state'),
                'confidence': result.get('confidence', 0.0),
                'app_identity': result.get('app_identity'),
                'content': result.get('content'),
                'predictions': result.get('predictions', []),
                'recommendations': result.get('recommendations', {})
            }
            
            # Add insights if available
            if result.get('recommendations', {}).get('warnings'):
                vsms_result['warnings'] = result['recommendations']['warnings']
            
            if result.get('recommendations', {}).get('workflow_hint'):
                vsms_result['workflow_detected'] = result['recommendations']['workflow_hint']
            
            return vsms_result
            
        except Exception as e:
            logger.error(f"VSMS Core analysis failed: {e}")
            return {'component': 'vsms_core', 'error': str(e)}
    
    async def _analyze_with_python(self, 
                                  screenshot: np.ndarray, 
                                  app_id: str,
                                  metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze using Python VSMS"""
        try:
            visual_data = {
                'screenshot': screenshot,
                'features': self._extract_python_features(screenshot),
                'metadata': metadata or {},
                'timestamp': datetime.now().isoformat()
            }
            
            result = await self.vsms.process_visual_input(app_id, visual_data)
            result['component'] = 'python_vsms'
            return result
            
        except Exception as e:
            logger.error(f"Python VSMS analysis failed: {e}")
            return {'component': 'python_vsms', 'error': str(e)}
    
    async def _analyze_with_swift(self, 
                                 screenshot_bytes: bytes, 
                                 app_id: str) -> Dict[str, Any]:
        """Analyze using Swift Vision framework"""
        try:
            result = await self.swift_bridge.analyze_screenshot(screenshot_bytes, app_id)
            result['component'] = 'swift_vision'
            return result
            
        except Exception as e:
            logger.error(f"Swift analysis failed: {e}")
            return {'component': 'swift_vision', 'error': str(e)}
    
    async def _analyze_with_rust(self, 
                                screenshot_bytes: bytes, 
                                app_id: str) -> Dict[str, Any]:
        """Analyze using Rust pattern matching"""
        try:
            # Extract features using Rust
            features = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._extract_rust_features,
                screenshot_bytes
            )
            
            # Create visual pattern
            pattern = rust_vi.VisualPattern(f"{app_id}_{datetime.now().timestamp()}")
            pattern.update_from_observation(features)
            
            # Match against known patterns
            match_result = self.rust_pattern_matcher.match_pattern(pattern)
            
            result = {
                'component': 'rust_pattern_matcher',
                'features': features,
                'match': match_result if match_result else None,
                'pattern_count': self.rust_pattern_matcher.pattern_count()
            }
            
            # Add pattern if learning
            if not match_result:
                self.rust_pattern_matcher.add_pattern(pattern)
                result['new_pattern_added'] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Rust analysis failed: {e}")
            return {'component': 'rust_pattern_matcher', 'error': str(e)}
    
    def _extract_python_features(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Extract features using Python/NumPy"""
        features = {}
        
        try:
            # Basic feature extraction
            features['shape'] = screenshot.shape
            features['mean_color'] = screenshot.mean(axis=(0, 1)).tolist() if screenshot.ndim == 3 else float(screenshot.mean())
            features['std_color'] = screenshot.std(axis=(0, 1)).tolist() if screenshot.ndim == 3 else float(screenshot.std())
            
            # Edge detection (simplified)
            if screenshot.ndim == 3:
                gray = np.dot(screenshot[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                gray = screenshot
            
            # Simple edge detection using gradient
            dx = np.diff(gray, axis=1)
            dy = np.diff(gray, axis=0)
            features['edge_density'] = float((np.abs(dx).mean() + np.abs(dy).mean()) / 2)
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
        
        return features
    
    def _extract_rust_features(self, screenshot_bytes: bytes) -> Dict[str, Any]:
        """Extract features using Rust"""
        if not self.rust_feature_extractor:
            return {}
        
        # Convert bytes to format Rust expects
        # This would use the actual Rust feature extractor
        features = {
            'color_histogram': [0.1] * 256,  # Placeholder
            'edge_features': [0.5] * 64,     # Placeholder
            'texture_descriptors': [0.3] * 32  # Placeholder
        }
        
        return features
    
    def _determine_final_state(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from all components to determine final state"""
        final_state = {
            'state_id': None,
            'confidence': 0.0,
            'consensus': False
        }
        
        # Extract state predictions from each component
        predictions = []
        
        # VSMS Core prediction (highest priority)
        if 'vsms_core' in results and 'detected_state' in results['vsms_core']:
            predictions.append({
                'source': 'vsms_core',
                'state_id': results['vsms_core']['detected_state'],
                'confidence': results['vsms_core'].get('confidence', 0.0),
                'has_recommendations': bool(results['vsms_core'].get('recommendations'))
            })
        
        # Python VSMS prediction
        if 'python_vsms' in results and 'state_analysis' in results['python_vsms']:
            state_analysis = results['python_vsms']['state_analysis']
            if state_analysis.get('state_id'):
                predictions.append({
                    'source': 'python_vsms',
                    'state_id': state_analysis['state_id'],
                    'confidence': state_analysis.get('confidence', 0.0)
                })
        
        # Swift prediction
        if 'swift_vision' in results and 'state_id' in results['swift_vision']:
            predictions.append({
                'source': 'swift_vision',
                'state_id': results['swift_vision']['state_id'],
                'confidence': results['swift_vision'].get('confidence', 0.0)
            })
        
        # Rust prediction
        if 'rust_pattern_matcher' in results and results['rust_pattern_matcher'].get('match'):
            match_data = results['rust_pattern_matcher']['match']
            predictions.append({
                'source': 'rust_pattern_matcher',
                'state_id': match_data[0] if isinstance(match_data, tuple) else str(match_data),
                'confidence': match_data[1] if isinstance(match_data, tuple) else 0.5
            })
        
        # Determine consensus
        if not predictions:
            return final_state
        
        # Sort by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Check for agreement
        state_votes = {}
        for pred in predictions:
            state_id = pred['state_id']
            if state_id not in state_votes:
                state_votes[state_id] = {
                    'votes': 0,
                    'total_confidence': 0.0,
                    'sources': []
                }
            state_votes[state_id]['votes'] += 1
            state_votes[state_id]['total_confidence'] += pred['confidence']
            state_votes[state_id]['sources'].append(pred['source'])
        
        # Find state with most votes/highest confidence
        best_state = max(state_votes.items(), 
                        key=lambda x: (x[1]['votes'], x[1]['total_confidence']))
        
        final_state['state_id'] = best_state[0]
        final_state['confidence'] = best_state[1]['total_confidence'] / best_state[1]['votes']
        final_state['consensus'] = best_state[1]['votes'] >= 2
        final_state['sources'] = best_state[1]['sources']
        
        return final_state
    
    async def train_on_labeled_state(self, 
                                   screenshot: Union[bytes, np.ndarray],
                                   app_id: str,
                                   state_id: str,
                                   state_type: Optional[str] = None) -> Dict[str, Any]:
        """Train all components on a labeled state"""
        results = {
            'trained_components': [],
            'app_id': app_id,
            'state_id': state_id
        }
        
        # Train each component
        # This would implement supervised learning for each component
        
        return results
    
    def get_system_insights(self) -> Dict[str, Any]:
        """Get insights from all components"""
        insights = {
            'components': {
                'python_vsms': self.vsms.get_system_insights() if self.vsms else None,
                'vsms_core': self.vsms_core.get_insights() if self.vsms_core else None,
                'rust_available': RUST_AVAILABLE,
                'swift_available': self.swift_bridge.swift_executable.exists()
            }
        }
        
        if RUST_AVAILABLE and self.rust_pattern_matcher:
            insights['components']['rust_patterns'] = {
                'pattern_count': self.rust_pattern_matcher.pattern_count()
            }
        
        # Add VSMS Core specific insights
        if self.vsms_core:
            vsms_insights = self.vsms_core.get_insights()
            insights['vsms_summary'] = {
                'tracked_applications': vsms_insights.get('tracked_applications', 0),
                'total_states': vsms_insights.get('total_states', 0),
                'personalization_score': vsms_insights.get('personalization_score', 0.0),
                'stuck_states': vsms_insights.get('stuck_states', []),
                'preferred_states': vsms_insights.get('preferred_states', [])
            }
        
        return insights
    
    def save_learned_states(self):
        """Save learned states from all components"""
        if self.vsms:
            self.vsms.save_all_states()
        
        # Save VSMS Core state
        if self.vsms_core:
            self.vsms_core.state_history.save_to_disk()
            self.vsms_core._save_state_definitions()
            if hasattr(self.vsms_core, 'state_intelligence'):
                self.vsms_core.state_intelligence._save_intelligence_data()
        
        # Save Rust patterns
        if RUST_AVAILABLE and self.rust_pattern_matcher:
            # This would serialize Rust patterns
            pass


# Convenience functions
_bridge_instance = None

def get_vision_intelligence_bridge() -> VisionIntelligenceBridge:
    """Get or create the global bridge instance"""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = VisionIntelligenceBridge()
    return _bridge_instance

async def analyze_screenshot(screenshot: Union[bytes, np.ndarray], 
                           app_id: str,
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function to analyze screenshot"""
    bridge = get_vision_intelligence_bridge()
    return await bridge.analyze_visual_state(screenshot, app_id, metadata)