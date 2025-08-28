#!/usr/bin/env python3
"""
Safe Capability Synthesis for Vision System v2.0
Ensures generated capabilities are safe, secure, and follow best practices
"""

import ast
import re
import inspect
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import subprocess
import tempfile
import asyncio
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class SafetyViolation:
    """Represents a safety violation found in code"""
    violation_type: str
    severity: str  # low, medium, high, critical
    line_number: int
    code_snippet: str
    description: str
    fix_suggestion: Optional[str] = None

@dataclass
class SecurityCheck:
    """Security check result"""
    check_name: str
    passed: bool
    violations: List[SafetyViolation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SynthesisConstraints:
    """Constraints for safe capability synthesis"""
    # Resource limits
    max_execution_time: float = 5.0  # seconds
    max_memory_usage: int = 100 * 1024 * 1024  # 100MB
    max_file_operations: int = 10
    
    # Code restrictions
    forbidden_imports: List[str] = field(default_factory=lambda: [
        'os', 'subprocess', 'eval', 'exec', '__import__',
        'compile', 'open', 'file', 'input', 'raw_input'
    ])
    
    allowed_builtins: Set[str] = field(default_factory=lambda: {
        'len', 'range', 'enumerate', 'zip', 'map', 'filter',
        'sum', 'min', 'max', 'abs', 'round', 'sorted',
        'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple'
    })
    
    # Network restrictions
    allow_network: bool = False
    allowed_domains: List[str] = field(default_factory=list)
    
    # File system restrictions
    allowed_paths: List[str] = field(default_factory=lambda: ['/tmp', '/var/tmp'])
    read_only_paths: List[str] = field(default_factory=list)

class CodeAnalyzer:
    """Analyzes code for safety and security issues"""
    
    def __init__(self):
        self.ast_checks = [
            self._check_forbidden_imports,
            self._check_dangerous_functions,
            self._check_eval_exec,
            self._check_file_operations,
            self._check_network_operations,
            self._check_subprocess_usage,
            self._check_global_usage,
            self._check_infinite_loops
        ]
        
    def analyze_code(self, code: str) -> List[SecurityCheck]:
        """Perform comprehensive code analysis"""
        checks = []
        
        try:
            # Parse code into AST
            tree = ast.parse(code)
            
            # Run AST-based checks
            for check_func in self.ast_checks:
                check = check_func(tree, code)
                checks.append(check)
                
            # Run regex-based checks
            checks.extend(self._run_regex_checks(code))
            
            # Check code complexity
            checks.append(self._check_complexity(tree, code))
            
        except SyntaxError as e:
            checks.append(SecurityCheck(
                check_name="syntax",
                passed=False,
                violations=[SafetyViolation(
                    violation_type="syntax_error",
                    severity="critical",
                    line_number=e.lineno or 0,
                    code_snippet=code.split('\n')[e.lineno - 1] if e.lineno else "",
                    description=f"Syntax error: {str(e)}"
                )]
            ))
            
        return checks
    
    def _check_forbidden_imports(self, tree: ast.AST, code: str) -> SecurityCheck:
        """Check for forbidden imports"""
        violations = []
        
        class ImportChecker(ast.NodeVisitor):
            def visit_Import(self, node):
                for alias in node.names:
                    if any(forbidden in alias.name for forbidden in [
                        'os', 'subprocess', 'sys', 'importlib', '__builtin__'
                    ]):
                        violations.append(SafetyViolation(
                            violation_type="forbidden_import",
                            severity="high",
                            line_number=node.lineno,
                            code_snippet=ast.unparse(node),
                            description=f"Forbidden import: {alias.name}",
                            fix_suggestion="Remove or replace with safe alternative"
                        ))
                        
            def visit_ImportFrom(self, node):
                if node.module and any(forbidden in node.module for forbidden in [
                    'os', 'subprocess', 'sys', 'importlib'
                ]):
                    violations.append(SafetyViolation(
                        violation_type="forbidden_import",
                        severity="high",
                        line_number=node.lineno,
                        code_snippet=ast.unparse(node),
                        description=f"Forbidden import from: {node.module}",
                        fix_suggestion="Remove or replace with safe alternative"
                    ))
                    
        ImportChecker().visit(tree)
        
        return SecurityCheck(
            check_name="forbidden_imports",
            passed=len(violations) == 0,
            violations=violations
        )
    
    def _check_dangerous_functions(self, tree: ast.AST, code: str) -> SecurityCheck:
        """Check for dangerous function calls"""
        violations = []
        dangerous_funcs = {'eval', 'exec', 'compile', '__import__', 'globals', 'locals'}
        
        class FunctionChecker(ast.NodeVisitor):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id in dangerous_funcs:
                    violations.append(SafetyViolation(
                        violation_type="dangerous_function",
                        severity="critical",
                        line_number=node.lineno,
                        code_snippet=ast.unparse(node),
                        description=f"Dangerous function call: {node.func.id}",
                        fix_suggestion="Remove or reimplement without dynamic code execution"
                    ))
                self.generic_visit(node)
                
        FunctionChecker().visit(tree)
        
        return SecurityCheck(
            check_name="dangerous_functions",
            passed=len(violations) == 0,
            violations=violations
        )
    
    def _check_eval_exec(self, tree: ast.AST, code: str) -> SecurityCheck:
        """Specific check for eval/exec usage"""
        violations = []
        
        # Check for eval/exec in strings
        if 'eval(' in code or 'exec(' in code:
            for i, line in enumerate(code.split('\n'), 1):
                if 'eval(' in line or 'exec(' in line:
                    violations.append(SafetyViolation(
                        violation_type="dynamic_execution",
                        severity="critical",
                        line_number=i,
                        code_snippet=line.strip(),
                        description="Potential eval/exec usage detected",
                        fix_suggestion="Use safe alternatives like ast.literal_eval for data parsing"
                    ))
                    
        return SecurityCheck(
            check_name="eval_exec",
            passed=len(violations) == 0,
            violations=violations
        )
    
    def _check_file_operations(self, tree: ast.AST, code: str) -> SecurityCheck:
        """Check for unsafe file operations"""
        violations = []
        
        class FileOpChecker(ast.NodeVisitor):
            def visit_Call(self, node):
                # Check for open() calls
                if isinstance(node.func, ast.Name) and node.func.id == 'open':
                    violations.append(SafetyViolation(
                        violation_type="file_operation",
                        severity="medium",
                        line_number=node.lineno,
                        code_snippet=ast.unparse(node),
                        description="Direct file operation detected",
                        fix_suggestion="Use sandboxed file operations or virtual file system"
                    ))
                    
                # Check for path operations
                if isinstance(node.func, ast.Attribute):
                    if (node.func.attr in ['remove', 'unlink', 'rmdir', 'mkdir'] and
                        hasattr(node.func.value, 'id') and node.func.value.id == 'os'):
                        violations.append(SafetyViolation(
                            violation_type="file_operation",
                            severity="high",
                            line_number=node.lineno,
                            code_snippet=ast.unparse(node),
                            description=f"File system modification: {node.func.attr}",
                            fix_suggestion="Operations should be limited to sandboxed directories"
                        ))
                        
                self.generic_visit(node)
                
        FileOpChecker().visit(tree)
        
        return SecurityCheck(
            check_name="file_operations",
            passed=len(violations) == 0,
            violations=violations
        )
    
    def _check_network_operations(self, tree: ast.AST, code: str) -> SecurityCheck:
        """Check for network operations"""
        violations = []
        network_modules = {'socket', 'urllib', 'requests', 'http', 'ftplib', 'telnetlib'}
        
        class NetworkChecker(ast.NodeVisitor):
            def visit_Import(self, node):
                for alias in node.names:
                    if any(net_mod in alias.name for net_mod in network_modules):
                        violations.append(SafetyViolation(
                            violation_type="network_operation",
                            severity="medium",
                            line_number=node.lineno,
                            code_snippet=ast.unparse(node),
                            description=f"Network module import: {alias.name}",
                            fix_suggestion="Network operations should be explicitly allowed and sandboxed"
                        ))
                        
        NetworkChecker().visit(tree)
        
        return SecurityCheck(
            check_name="network_operations",
            passed=len(violations) == 0,
            violations=violations
        )
    
    def _check_subprocess_usage(self, tree: ast.AST, code: str) -> SecurityCheck:
        """Check for subprocess usage"""
        violations = []
        
        if 'subprocess' in code or 'Popen' in code:
            for i, line in enumerate(code.split('\n'), 1):
                if 'subprocess' in line or 'Popen' in line:
                    violations.append(SafetyViolation(
                        violation_type="subprocess_usage",
                        severity="critical",
                        line_number=i,
                        code_snippet=line.strip(),
                        description="Subprocess usage detected",
                        fix_suggestion="Direct system command execution is not allowed"
                    ))
                    
        return SecurityCheck(
            check_name="subprocess_usage",
            passed=len(violations) == 0,
            violations=violations
        )
    
    def _check_global_usage(self, tree: ast.AST, code: str) -> SecurityCheck:
        """Check for global variable usage"""
        violations = []
        
        class GlobalChecker(ast.NodeVisitor):
            def visit_Global(self, node):
                violations.append(SafetyViolation(
                    violation_type="global_usage",
                    severity="low",
                    line_number=node.lineno,
                    code_snippet=f"global {', '.join(node.names)}",
                    description="Global variable usage",
                    fix_suggestion="Use class attributes or return values instead"
                ))
                
        GlobalChecker().visit(tree)
        
        return SecurityCheck(
            check_name="global_usage",
            passed=len(violations) == 0,
            violations=violations
        )
    
    def _check_infinite_loops(self, tree: ast.AST, code: str) -> SecurityCheck:
        """Check for potential infinite loops"""
        violations = []
        
        class LoopChecker(ast.NodeVisitor):
            def visit_While(self, node):
                # Check for while True without break
                if isinstance(node.test, ast.Constant) and node.test.value is True:
                    has_break = self._has_break(node)
                    if not has_break:
                        violations.append(SafetyViolation(
                            violation_type="infinite_loop",
                            severity="medium",
                            line_number=node.lineno,
                            code_snippet="while True: ...",
                            description="Potential infinite loop detected",
                            fix_suggestion="Add break condition or use bounded iteration"
                        ))
                self.generic_visit(node)
                
            def _has_break(self, node):
                for child in ast.walk(node):
                    if isinstance(child, ast.Break):
                        return True
                return False
                
        LoopChecker().visit(tree)
        
        return SecurityCheck(
            check_name="infinite_loops",
            passed=len(violations) == 0,
            violations=violations
        )
    
    def _run_regex_checks(self, code: str) -> List[SecurityCheck]:
        """Run regex-based security checks"""
        checks = []
        
        # Check for hardcoded credentials
        credential_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', "hardcoded_password"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "hardcoded_api_key"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "hardcoded_secret")
        ]
        
        violations = []
        for pattern, violation_type in credential_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                violations.append(SafetyViolation(
                    violation_type=violation_type,
                    severity="high",
                    line_number=line_num,
                    code_snippet=match.group(),
                    description="Hardcoded credentials detected",
                    fix_suggestion="Use environment variables or secure configuration"
                ))
                
        checks.append(SecurityCheck(
            check_name="hardcoded_credentials",
            passed=len(violations) == 0,
            violations=violations
        ))
        
        return checks
    
    def _check_complexity(self, tree: ast.AST, code: str) -> SecurityCheck:
        """Check code complexity"""
        violations = []
        
        # Count nodes
        node_count = len(list(ast.walk(tree)))
        if node_count > 500:
            violations.append(SafetyViolation(
                violation_type="high_complexity",
                severity="medium",
                line_number=0,
                code_snippet="",
                description=f"Code complexity too high: {node_count} AST nodes",
                fix_suggestion="Simplify code or break into smaller functions"
            ))
            
        # Check nesting depth
        max_depth = self._get_max_nesting_depth(tree)
        if max_depth > 5:
            violations.append(SafetyViolation(
                violation_type="deep_nesting",
                severity="low",
                line_number=0,
                code_snippet="",
                description=f"Deep nesting detected: {max_depth} levels",
                fix_suggestion="Refactor to reduce nesting depth"
            ))
            
        return SecurityCheck(
            check_name="complexity",
            passed=len(violations) == 0,
            violations=violations
        )
    
    def _get_max_nesting_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth"""
        class DepthCalculator(ast.NodeVisitor):
            def __init__(self):
                self.max_depth = 0
                self.current_depth = 0
                
            def visit(self, node):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    self.current_depth += 1
                    self.max_depth = max(self.max_depth, self.current_depth)
                    self.generic_visit(node)
                    self.current_depth -= 1
                else:
                    self.generic_visit(node)
                    
        calculator = DepthCalculator()
        calculator.visit(tree)
        return calculator.max_depth

class SafetyValidator:
    """Validates generated capabilities for safety"""
    
    def __init__(self, constraints: Optional[SynthesisConstraints] = None):
        self.constraints = constraints or SynthesisConstraints()
        self.code_analyzer = CodeAnalyzer()
        
    async def validate_capability(self, code: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a capability for safety"""
        validation_result = {
            'safe': True,
            'security_checks': [],
            'violations': [],
            'risk_score': 0.0,
            'recommendations': []
        }
        
        # Run security analysis
        security_checks = self.code_analyzer.analyze_code(code)
        validation_result['security_checks'] = security_checks
        
        # Collect all violations
        all_violations = []
        for check in security_checks:
            if not check.passed:
                all_violations.extend(check.violations)
                
        validation_result['violations'] = all_violations
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(all_violations)
        validation_result['risk_score'] = risk_score
        
        # Determine if safe
        validation_result['safe'] = risk_score < 0.7 and not any(
            v.severity == 'critical' for v in all_violations
        )
        
        # Generate recommendations
        validation_result['recommendations'] = self._generate_recommendations(all_violations)
        
        return validation_result
    
    def _calculate_risk_score(self, violations: List[SafetyViolation]) -> float:
        """Calculate overall risk score (0-1)"""
        if not violations:
            return 0.0
            
        severity_scores = {
            'low': 0.1,
            'medium': 0.3,
            'high': 0.6,
            'critical': 1.0
        }
        
        total_score = sum(severity_scores.get(v.severity, 0.5) for v in violations)
        
        # Normalize to 0-1 range
        return min(1.0, total_score / max(len(violations), 1))
    
    def _generate_recommendations(self, violations: List[SafetyViolation]) -> List[str]:
        """Generate safety recommendations"""
        recommendations = []
        
        violation_types = set(v.violation_type for v in violations)
        
        if 'forbidden_import' in violation_types:
            recommendations.append(
                "Replace system-level imports with safe alternatives or sandbox operations"
            )
            
        if 'dangerous_function' in violation_types:
            recommendations.append(
                "Avoid dynamic code execution; use static analysis or predefined operations"
            )
            
        if 'file_operation' in violation_types:
            recommendations.append(
                "Limit file operations to sandboxed directories with proper access controls"
            )
            
        if 'hardcoded_credentials' in violation_types:
            recommendations.append(
                "Use secure configuration management for sensitive data"
            )
            
        if not violations:
            recommendations.append("Code appears safe for execution")
            
        return recommendations

class CapabilitySynthesizer:
    """Safe synthesis of new capabilities with validation"""
    
    def __init__(self):
        self.safety_validator = SafetyValidator()
        self.synthesis_templates = self._load_synthesis_templates()
        
    def _load_synthesis_templates(self) -> Dict[str, str]:
        """Load safe code templates"""
        return {
            'safe_wrapper': '''
async def {function_name}_safe(self, *args, **kwargs):
    """Safe wrapper for {function_name}"""
    # Input validation
    {input_validation}
    
    # Resource limits
    start_time = time.time()
    timeout = {timeout}
    
    try:
        # Execute with timeout
        result = await asyncio.wait_for(
            self.{function_name}(*args, **kwargs),
            timeout=timeout
        )
        
        # Output validation
        {output_validation}
        
        return result
        
    except asyncio.TimeoutError:
        logger.error(f"{function_name} timed out after {timeout}s")
        return {{'error': 'Operation timed out'}}
    except Exception as e:
        logger.error(f"Error in {function_name}: {{e}}")
        return {{'error': str(e)}}
''',
            'input_validation': '''
    # Validate input parameters
    if not isinstance(command, str):
        raise TypeError("Command must be a string")
    if len(command) > 1000:
        raise ValueError("Command too long")
    if context and not isinstance(context, dict):
        raise TypeError("Context must be a dictionary")
''',
            'output_validation': '''
    # Validate output
    if not isinstance(result, dict):
        raise TypeError("Result must be a dictionary")
    if 'success' not in result:
        result['success'] = False
'''
        }
    
    async def synthesize_safe_capability(
        self,
        original_code: str,
        capability_name: str,
        constraints: Optional[SynthesisConstraints] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Synthesize a safe version of a capability"""
        constraints = constraints or SynthesisConstraints()
        
        # Validate original code
        validation_result = await self.safety_validator.validate_capability(
            original_code,
            {'capability_name': capability_name}
        )
        
        if validation_result['safe']:
            # Code is already safe, add minimal wrapper
            safe_code = self._add_safety_wrapper(original_code, capability_name, constraints)
        else:
            # Code needs modification
            safe_code = await self._make_code_safe(
                original_code,
                validation_result['violations'],
                constraints
            )
            
        # Re-validate
        final_validation = await self.safety_validator.validate_capability(
            safe_code,
            {'capability_name': capability_name}
        )
        
        return safe_code, final_validation
    
    def _add_safety_wrapper(
        self,
        code: str,
        capability_name: str,
        constraints: SynthesisConstraints
    ) -> str:
        """Add safety wrapper to code"""
        # Extract function name from code
        match = re.search(r'async def (\w+)', code)
        if not match:
            match = re.search(r'def (\w+)', code)
            
        if not match:
            raise ValueError("Could not find function definition in code")
            
        function_name = match.group(1)
        
        # Generate wrapper
        wrapper = self.synthesis_templates['safe_wrapper'].format(
            function_name=function_name,
            input_validation=self.synthesis_templates['input_validation'],
            output_validation=self.synthesis_templates['output_validation'],
            timeout=constraints.max_execution_time
        )
        
        # Combine original code with wrapper
        safe_code = f"{code}\n\n{wrapper}"
        
        # Add necessary imports
        imports = "import time\nimport asyncio\nimport logging\n\nlogger = logging.getLogger(__name__)\n\n"
        
        return imports + safe_code
    
    async def _make_code_safe(
        self,
        code: str,
        violations: List[SafetyViolation],
        constraints: SynthesisConstraints
    ) -> str:
        """Modify code to fix safety violations"""
        safe_code = code
        
        # Group violations by type
        violations_by_type = defaultdict(list)
        for violation in violations:
            violations_by_type[violation.violation_type].append(violation)
            
        # Apply fixes
        if 'forbidden_import' in violations_by_type:
            safe_code = self._fix_forbidden_imports(safe_code, violations_by_type['forbidden_import'])
            
        if 'dangerous_function' in violations_by_type:
            safe_code = self._fix_dangerous_functions(safe_code, violations_by_type['dangerous_function'])
            
        if 'file_operation' in violations_by_type:
            safe_code = self._sandbox_file_operations(safe_code, violations_by_type['file_operation'])
            
        if 'hardcoded_credentials' in violations_by_type:
            safe_code = self._fix_hardcoded_credentials(safe_code, violations_by_type['hardcoded_credentials'])
            
        # Add safety wrapper
        safe_code = self._add_safety_wrapper(safe_code, "modified_capability", constraints)
        
        return safe_code
    
    def _fix_forbidden_imports(self, code: str, violations: List[SafetyViolation]) -> str:
        """Remove or replace forbidden imports"""
        lines = code.split('\n')
        
        for violation in violations:
            if violation.line_number > 0 and violation.line_number <= len(lines):
                # Comment out the forbidden import
                lines[violation.line_number - 1] = f"# REMOVED: {lines[violation.line_number - 1]}"
                
        return '\n'.join(lines)
    
    def _fix_dangerous_functions(self, code: str, violations: List[SafetyViolation]) -> str:
        """Replace dangerous functions with safe alternatives"""
        replacements = {
            'eval': 'ast.literal_eval',
            'exec': '# exec disabled',
            '__import__': '# dynamic import disabled'
        }
        
        safe_code = code
        for old, new in replacements.items():
            safe_code = safe_code.replace(f'{old}(', f'{new}(')
            
        return safe_code
    
    def _sandbox_file_operations(self, code: str, violations: List[SafetyViolation]) -> str:
        """Sandbox file operations to safe directories"""
        # This is a simplified version - in production, use proper sandboxing
        safe_code = code.replace('open(', 'self._safe_open(')
        
        # Add safe_open method
        safe_open_impl = '''
def _safe_open(self, filepath, mode='r'):
    """Safe file open with path validation"""
    import os
    
    # Ensure path is within allowed directories
    abs_path = os.path.abspath(filepath)
    allowed = False
    
    for allowed_path in ['/tmp', '/var/tmp']:
        if abs_path.startswith(allowed_path):
            allowed = True
            break
            
    if not allowed:
        raise PermissionError(f"Access denied to path: {filepath}")
        
    # Limit write operations
    if 'w' in mode or 'a' in mode:
        logger.warning(f"Write operation to {filepath}")
        
    return open(filepath, mode)
'''
        
        return safe_open_impl + "\n\n" + safe_code
    
    def _fix_hardcoded_credentials(self, code: str, violations: List[SafetyViolation]) -> str:
        """Replace hardcoded credentials with config lookups"""
        safe_code = code
        
        # Simple replacement - in production, use proper secret management
        patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', 'password = os.environ.get("PASSWORD", "")'),
            (r'api_key\s*=\s*["\'][^"\']+["\']', 'api_key = os.environ.get("API_KEY", "")'),
            (r'secret\s*=\s*["\'][^"\']+["\']', 'secret = os.environ.get("SECRET", "")')
        ]
        
        for pattern, replacement in patterns:
            safe_code = re.sub(pattern, replacement, safe_code, flags=re.IGNORECASE)
            
        return safe_code

# Singleton instances
_safety_validator: Optional[SafetyValidator] = None
_capability_synthesizer: Optional[CapabilitySynthesizer] = None

def get_safety_validator() -> SafetyValidator:
    """Get singleton instance of safety validator"""
    global _safety_validator
    if _safety_validator is None:
        _safety_validator = SafetyValidator()
    return _safety_validator

def get_capability_synthesizer() -> CapabilitySynthesizer:
    """Get singleton instance of capability synthesizer"""
    global _capability_synthesizer
    if _capability_synthesizer is None:
        _capability_synthesizer = CapabilitySynthesizer()
    return _capability_synthesizer