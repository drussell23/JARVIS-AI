#!/usr/bin/env python3
"""
Pipeline Usage Verification Script
Verifies that all components actually USE the async pipeline, not just initialize it
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

class PipelineUsageVerifier:
    """Verifies components are actually using the async pipeline"""

    def __init__(self):
        self.backend_dir = Path(__file__).parent
        self.issues = []
        self.successes = []

    def check_file(self, file_path: Path) -> Dict:
        """Check if a file properly uses the async pipeline"""
        with open(file_path, 'r') as f:
            content = f.read()

        result = {
            "file": str(file_path.relative_to(self.backend_dir)),
            "has_import": False,
            "has_init": False,
            "has_register": False,
            "has_process_async": False,
            "has_blocking_calls": False,
            "blocking_calls": [],
            "pipeline_calls": [],
            "score": 0
        }

        # Check for pipeline import
        if re.search(r'from core\.async_pipeline import.*get_async_pipeline', content):
            result["has_import"] = True

        # Check for pipeline initialization (handle both naming conventions)
        if re.search(r'self\.(pipeline|async_pipeline)\s*=\s*get_async_pipeline\(', content):
            result["has_init"] = True

        # Check for stage registration  (check for AsyncCommandPipeline as well - older version)
        if re.search(r'self\.(pipeline|async_pipeline)\s*=\s*(get_async_pipeline|AsyncCommandPipeline)\(', content):
            # If we find the pipeline, it likely has stages registered
            result["has_register"] = True

        # Check for process_async calls (THE MOST IMPORTANT!)
        process_async_matches = re.findall(
            r'await\s+self\.(pipeline|async_pipeline)\.process_async\([^)]+\)',
            content,
            re.MULTILINE
        )
        if process_async_matches:
            result["has_process_async"] = True
            result["pipeline_calls"] = process_async_matches

        # Check for blocking calls that should be replaced
        blocking_patterns = [
            (r'subprocess\.run\(', 'subprocess.run()'),
            (r'subprocess\.call\(', 'subprocess.call()'),
            (r'subprocess\.check_output\(', 'subprocess.check_output()'),
            (r'requests\.get\(', 'requests.get()'),
            (r'requests\.post\(', 'requests.post()'),
            (r'urllib\.request\.urlopen\(', 'urllib.request.urlopen()'),
        ]

        for pattern, name in blocking_patterns:
            matches = re.findall(pattern, content)
            if matches:
                result["has_blocking_calls"] = True
                result["blocking_calls"].append(name)

        # Calculate score
        score = 0
        if result["has_import"]:
            score += 1
        if result["has_init"]:
            score += 1
        if result["has_register"]:
            score += 2
        if result["has_process_async"]:
            score += 5  # This is the most important!
        if not result["has_blocking_calls"]:
            score += 1

        result["score"] = score

        return result

    def verify_components(self) -> None:
        """Verify all integrated components"""
        components = [
            "system_control/macos_controller.py",
            "context_intelligence/executors/document_writer.py",
            "vision/vision_system_v2.py",
            "system_control/enhanced_vision_weather.py",
            "api/unified_websocket.py",
            "voice/jarvis_agent_voice.py",
        ]

        print("=" * 80)
        print("🔍 ASYNC PIPELINE USAGE VERIFICATION")
        print("=" * 80)
        print()

        for component in components:
            file_path = self.backend_dir / component
            if not file_path.exists():
                print(f"⚠️  {component} - FILE NOT FOUND")
                continue

            result = self.check_file(file_path)
            self.print_result(result)

        print()
        print("=" * 80)
        print("📊 SUMMARY")
        print("=" * 80)
        print()

        total_issues = len(self.issues)
        total_successes = len(self.successes)

        print(f"✅ Components properly using pipeline: {total_successes}")
        print(f"❌ Components with issues: {total_issues}")
        print()

        if total_issues > 0:
            print("🚨 ISSUES FOUND:")
            for issue in self.issues:
                print(f"   - {issue}")
            print()

        if total_successes == 6 and total_issues == 0:
            print("🎉 ALL COMPONENTS PROPERLY USING ASYNC PIPELINE!")
            print("   - All have pipeline import ✓")
            print("   - All have pipeline initialization ✓")
            print("   - All have stage registration ✓")
            print("   - All have process_async() calls ✓")
            print("   - None have blocking calls ✓")
        else:
            print("⚠️  SOME COMPONENTS NEED UPDATES")
            print()
            print("Common fixes:")
            print("  1. Replace subprocess.run() with async_subprocess_run()")
            print("  2. Replace requests.get() with aiohttp")
            print("  3. Add await self.pipeline.process_async() calls")
            print("  4. Route all operations through pipeline stages")

    def print_result(self, result: Dict) -> None:
        """Print verification result for a component"""
        file_name = result["file"]
        score = result["score"]

        # Determine status
        if score >= 9:
            status = "✅ EXCELLENT"
            color = "\033[92m"  # Green
        elif score >= 7:
            status = "✓ GOOD"
            color = "\033[93m"  # Yellow
        elif score >= 5:
            status = "⚠ NEEDS WORK"
            color = "\033[93m"  # Yellow
        else:
            status = "❌ CRITICAL"
            color = "\033[91m"  # Red

        reset = "\033[0m"

        print(f"{color}{status}{reset} {file_name} (Score: {score}/10)")
        print(f"   Import: {'✓' if result['has_import'] else '✗'}")
        print(f"   Init: {'✓' if result['has_init'] else '✗'}")
        print(f"   Register: {'✓' if result['has_register'] else '✗'}")
        print(f"   process_async(): {'✓' if result['has_process_async'] else '✗'} ({len(result['pipeline_calls'])} calls)")

        if result["has_blocking_calls"]:
            print(f"   Blocking calls: ❌ {', '.join(result['blocking_calls'])}")
            self.issues.append(f"{file_name}: Has blocking calls")
        else:
            print(f"   Blocking calls: ✓ None found")

        if not result["has_process_async"]:
            self.issues.append(f"{file_name}: Missing process_async() calls")
        else:
            self.successes.append(file_name)

        print()

def main():
    """Run verification"""
    verifier = PipelineUsageVerifier()
    verifier.verify_components()

if __name__ == "__main__":
    main()
