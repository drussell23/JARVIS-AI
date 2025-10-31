#!/usr/bin/env python3
"""
Advanced CI/CD Auto-PR Manager
Dynamically creates PRs for failed CI/CD workflows with intelligent analysis
"""

import asyncio
import aiohttp
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin
import yaml


@dataclass
class WorkflowRun:
    """Workflow run data structure"""
    id: str
    name: str
    conclusion: str
    html_url: str
    head_branch: str
    head_sha: str
    run_number: int
    run_attempt: int
    created_at: str
    updated_at: str
    actor: Dict[str, Any]


@dataclass
class FailedJob:
    """Failed job data structure"""
    id: str
    name: str
    conclusion: str
    html_url: str
    started_at: str
    completed_at: str
    steps: List[Dict[str, Any]]
    runner_name: Optional[str] = None
    duration_seconds: Optional[int] = None


@dataclass
class FailureAnalysis:
    """Failure analysis results"""
    error_patterns: List[Dict[str, Any]]
    suggested_fixes: List[str]
    severity: str
    category: str
    similar_failures: List[str]


class ConfigManager:
    """Manages configuration loading and validation"""

    def __init__(self, config_path: str = ".github/workflows/config/ci-auto-pr-config.yml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                print(f"⚠️ Config file not found: {self.config_path}, using defaults")
                return self._get_default_config()
        except Exception as e:
            print(f"❌ Error loading config: {e}, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            'monitoring': {
                'auto_discover_workflows': True,
                'exclude_patterns': ['.*dependabot.*'],
                'triggers': {'on_failure': True}
            },
            'pr_settings': {
                'branch_prefix': 'fix/ci',
                'labels': ['ci-failure', 'automated-pr'],
                'auto_assign': {'enabled': True, 'assign_to': 'actor'}
            },
            'analysis': {
                'fetch_logs': True,
                'max_log_lines': 100,
                'parse_errors': True
            },
            'safety': {
                'max_prs_per_day': 20,
                'skip_if_similar_pr_exists': True
            }
        }

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get nested config value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value if value is not None else default


class GitHubAPIClient:
    """Async GitHub API client with rate limiting and retry logic"""

    def __init__(self, token: str, repo: str, session: Optional[aiohttp.ClientSession] = None):
        self.token = token
        self.repo = repo
        self.base_url = "https://api.github.com"
        self.session = session
        self._own_session = session is None

    async def __aenter__(self):
        if self._own_session:
            self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._own_session and self.session:
            await self.session.close()

    @property
    def headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'JARVIS-CI-Auto-PR-Manager'
        }

    async def _request(
        self,
        method: str,
        endpoint: str,
        max_retries: int = 3,
        **kwargs
    ) -> Tuple[int, Any]:
        """Make async HTTP request with retry logic"""
        url = urljoin(self.base_url, endpoint.lstrip('/'))

        for attempt in range(max_retries):
            try:
                async with self.session.request(
                    method,
                    url,
                    headers=self.headers,
                    **kwargs
                ) as response:
                    data = await response.json() if response.content_type == 'application/json' else await response.text()

                    if response.status == 429:  # Rate limited
                        retry_after = int(response.headers.get('Retry-After', 60))
                        print(f"⏱️ Rate limited, waiting {retry_after}s...")
                        await asyncio.sleep(retry_after)
                        continue

                    return response.status, data

            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise

        raise Exception(f"Failed after {max_retries} attempts")

    async def get_workflow_run(self, run_id: str) -> Dict[str, Any]:
        """Get workflow run details"""
        status, data = await self._request('GET', f'/repos/{self.repo}/actions/runs/{run_id}')
        if status != 200:
            raise Exception(f"Failed to fetch workflow run: {status}")
        return data

    async def get_workflow_jobs(self, run_id: str) -> List[Dict[str, Any]]:
        """Get workflow jobs with pagination"""
        all_jobs = []
        page = 1
        per_page = 100

        while True:
            status, data = await self._request(
                'GET',
                f'/repos/{self.repo}/actions/runs/{run_id}/jobs',
                params={'page': page, 'per_page': per_page}
            )

            if status != 200:
                raise Exception(f"Failed to fetch jobs: {status}")

            jobs = data.get('jobs', [])
            all_jobs.extend(jobs)

            if len(jobs) < per_page:
                break

            page += 1

        return all_jobs

    async def get_job_logs(self, job_id: str) -> str:
        """Get job logs"""
        status, data = await self._request('GET', f'/repos/{self.repo}/actions/jobs/{job_id}/logs')
        return data if isinstance(data, str) else ""

    async def list_open_prs(self, base: str) -> List[Dict[str, Any]]:
        """List open PRs for a branch"""
        status, data = await self._request(
            'GET',
            f'/repos/{self.repo}/pulls',
            params={'state': 'open', 'base': base}
        )
        return data if status == 200 else []

    async def search_issues(self, query: str) -> List[Dict[str, Any]]:
        """Search issues/PRs"""
        status, data = await self._request('GET', '/search/issues', params={'q': query})
        return data.get('items', []) if status == 200 else []


class FailureAnalyzer:
    """Analyzes workflow failures and provides insights"""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.error_patterns = self._load_error_patterns()

    def _load_error_patterns(self) -> List[Dict[str, Any]]:
        """Load error patterns from config"""
        return self.config.get('analysis.error_patterns', [
            {'pattern': r'ERROR|Error|error', 'severity': 'high', 'emoji': '❌'},
            {'pattern': r'FAIL|Failed|failed', 'severity': 'high', 'emoji': '💥'},
            {'pattern': r'WARN|Warning|warning', 'severity': 'medium', 'emoji': '⚠️'},
        ])

    async def analyze_failure(
        self,
        job: FailedJob,
        logs: Optional[str] = None
    ) -> FailureAnalysis:
        """Analyze a failed job and provide insights"""
        error_patterns = []
        suggested_fixes = []
        severity = 'medium'
        category = 'unknown'

        # Analyze logs if available
        if logs:
            error_patterns = self._extract_error_patterns(logs)
            suggested_fixes = self._generate_suggestions(logs, job)
            severity = self._determine_severity(error_patterns)
            category = self._categorize_failure(logs, job)

        # Analyze based on job name and steps
        if not logs:
            category = self._categorize_from_job_name(job.name)
            suggested_fixes = self._generate_basic_suggestions(job)

        return FailureAnalysis(
            error_patterns=error_patterns,
            suggested_fixes=suggested_fixes,
            severity=severity,
            category=category,
            similar_failures=[]
        )

    def _extract_error_patterns(self, logs: str) -> List[Dict[str, Any]]:
        """Extract error patterns from logs"""
        patterns = []
        max_lines = self.config.get('analysis.max_log_lines', 100)
        lines = logs.split('\n')[-max_lines:]

        for pattern_config in self.error_patterns:
            pattern = pattern_config['pattern']
            matches = []

            for i, line in enumerate(lines):
                if re.search(pattern, line, re.IGNORECASE):
                    matches.append({
                        'line_number': len(lines) - max_lines + i,
                        'content': line.strip(),
                        'severity': pattern_config.get('severity', 'medium'),
                        'emoji': pattern_config.get('emoji', '🔴')
                    })

            if matches:
                patterns.append({
                    'pattern': pattern,
                    'matches': matches[:10],  # Limit to first 10 matches
                    'count': len(matches)
                })

        return patterns

    def _generate_suggestions(self, logs: str, job: FailedJob) -> List[str]:
        """Generate fix suggestions based on logs"""
        suggestions = []

        # Common patterns and their fixes
        if re.search(r'timeout|timed out', logs, re.IGNORECASE):
            suggestions.append("Consider increasing timeout values or optimizing slow operations")

        if re.search(r'permission denied|forbidden', logs, re.IGNORECASE):
            suggestions.append("Check file permissions and workflow permissions in GitHub Actions")

        if re.search(r'module not found|cannot import', logs, re.IGNORECASE):
            suggestions.append("Verify all dependencies are installed in requirements.txt or package.json")

        if re.search(r'connection refused|connection timeout', logs, re.IGNORECASE):
            suggestions.append("Check service availability and network connectivity")

        if re.search(r'assertion.*failed|test.*failed', logs, re.IGNORECASE):
            suggestions.append("Review test cases and ensure code changes haven't broken existing functionality")

        if re.search(r'out of memory|memory error', logs, re.IGNORECASE):
            suggestions.append("Increase runner memory or optimize memory usage in the application")

        if re.search(r'command not found', logs, re.IGNORECASE):
            suggestions.append("Ensure all required tools are installed in the workflow environment")

        return suggestions or ["Review the logs above for specific error messages"]

    def _generate_basic_suggestions(self, job: FailedJob) -> List[str]:
        """Generate basic suggestions without logs"""
        suggestions = []

        job_name_lower = job.name.lower()

        if 'test' in job_name_lower:
            suggestions.append("Review failing tests and update code or test expectations")
        elif 'lint' in job_name_lower or 'quality' in job_name_lower:
            suggestions.append("Run linters locally and fix code quality issues")
        elif 'build' in job_name_lower:
            suggestions.append("Check build configuration and dependencies")
        elif 'deploy' in job_name_lower:
            suggestions.append("Verify deployment configuration and credentials")

        return suggestions or ["Check the workflow logs for more details"]

    def _determine_severity(self, error_patterns: List[Dict[str, Any]]) -> str:
        """Determine overall severity based on error patterns"""
        severities = [p.get('matches', [{}])[0].get('severity', 'low') for p in error_patterns if p.get('matches')]

        if 'high' in severities:
            return 'high'
        elif 'medium' in severities:
            return 'medium'
        return 'low'

    def _categorize_failure(self, logs: str, job: FailedJob) -> str:
        """Categorize the type of failure"""
        logs_lower = logs.lower()

        if re.search(r'test.*fail|assertion.*error', logs_lower):
            return 'test_failure'
        elif re.search(r'syntax.*error|parse.*error', logs_lower):
            return 'syntax_error'
        elif re.search(r'lint|flake8|pylint|eslint', logs_lower):
            return 'linting_error'
        elif re.search(r'import.*error|module.*not.*found', logs_lower):
            return 'dependency_error'
        elif re.search(r'timeout|timed out', logs_lower):
            return 'timeout'
        elif re.search(r'permission|forbidden|unauthorized', logs_lower):
            return 'permission_error'
        elif re.search(r'network|connection', logs_lower):
            return 'network_error'

        return self._categorize_from_job_name(job.name)

    def _categorize_from_job_name(self, job_name: str) -> str:
        """Categorize based on job name"""
        job_name_lower = job_name.lower()

        if 'test' in job_name_lower:
            return 'test_failure'
        elif 'lint' in job_name_lower or 'quality' in job_name_lower:
            return 'linting_error'
        elif 'build' in job_name_lower:
            return 'build_error'
        elif 'deploy' in job_name_lower:
            return 'deployment_error'
        elif 'security' in job_name_lower:
            return 'security_error'

        return 'unknown'


class PRManager:
    """Manages PR creation and updates"""

    def __init__(self, config: ConfigManager, github_client: GitHubAPIClient):
        self.config = config
        self.github = github_client

    def generate_branch_name(self, workflow_name: str, run_number: int) -> str:
        """Generate dynamic branch name"""
        prefix = self.config.get('pr_settings.branch_prefix', 'fix/ci')
        sanitized = re.sub(r'[^a-z0-9-]', '-', workflow_name.lower())
        sanitized = re.sub(r'-+', '-', sanitized).strip('-')
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

        return f"{prefix}/{sanitized}-run{run_number}-{timestamp}"

    def generate_pr_title(self, workflow_name: str, run_number: int, branch: str, commit_sha: str) -> str:
        """Generate PR title from template"""
        template = self.config.get('pr_settings.title_template', '🚨 Fix CI/CD: {workflow_name} (Run #{run_number})')

        return template.format(
            workflow_name=workflow_name,
            run_number=run_number,
            branch=branch,
            commit_sha=commit_sha[:7]
        )

    def generate_pr_body(
        self,
        workflow_run: WorkflowRun,
        failed_jobs: List[FailedJob],
        analyses: List[FailureAnalysis]
    ) -> str:
        """Generate comprehensive PR body"""
        body = f"""## 🚨 CI/CD Failure Alert

The **{workflow_run.name}** workflow failed on branch `{workflow_run.head_branch}`.

### Summary
- **Failed Jobs**: {len(failed_jobs)}
- **Commit**: `{workflow_run.head_sha[:7]}`
- **Run**: [#{workflow_run.run_number}]({workflow_run.html_url})
- **Triggered by**: @{workflow_run.actor.get('login', 'unknown')}
- **Timestamp**: {workflow_run.created_at}

### Failed Jobs

"""

        for i, (job, analysis) in enumerate(zip(failed_jobs, analyses), 1):
            severity_emoji = '🔴' if analysis.severity == 'high' else '🟡' if analysis.severity == 'medium' else '🟢'
            category_emoji = self._get_category_emoji(analysis.category)

            body += f"""#### {i}. {severity_emoji} {job.name}

- **Category**: {category_emoji} {analysis.category.replace('_', ' ').title()}
- **Severity**: {analysis.severity.upper()}
- **Duration**: {job.duration_seconds}s
- **Job URL**: [View Details]({job.html_url})

"""

            if job.steps:
                body += "**Failed Steps:**\n"
                for step in job.steps:
                    if step.get('conclusion') == 'failure':
                        body += f"- Step {step.get('number', '?')}: {step.get('name', 'Unknown')}\n"
                body += "\n"

            if analysis.suggested_fixes:
                body += "**Suggested Fixes:**\n"
                for fix in analysis.suggested_fixes[:3]:
                    body += f"- {fix}\n"
                body += "\n"

        body += """### Next Steps

1. 📋 Review the failure analysis above
2. 🔍 Check the [workflow logs]({workflow_url}) for detailed information
3. 🔧 Implement suggested fixes
4. ✅ Push changes to this branch
5. 🚀 Verify all checks pass before merging

### Auto-Fix Attempts

This workflow will automatically attempt common fixes based on the failure type.

---

🤖 *Auto-generated by JARVIS CI/CD Auto-PR Manager* | [Configuration](.github/workflows/config/ci-auto-pr-config.yml)
""".format(workflow_url=workflow_run.html_url)

        return body

    def _get_category_emoji(self, category: str) -> str:
        """Get emoji for failure category"""
        emojis = {
            'test_failure': '🧪',
            'linting_error': '📝',
            'syntax_error': '📝',
            'build_error': '🏗️',
            'dependency_error': '📦',
            'timeout': '⏱️',
            'permission_error': '🔒',
            'network_error': '🌐',
            'security_error': '🔐',
            'deployment_error': '🚀',
            'unknown': '❓'
        }
        return emojis.get(category, '❓')

    def generate_failure_report(
        self,
        workflow_run: WorkflowRun,
        failed_jobs: List[FailedJob],
        analyses: List[FailureAnalysis]
    ) -> str:
        """Generate detailed markdown failure report"""
        report = f"""# CI/CD Failure Analysis Report

## Executive Summary

- **Workflow**: {workflow_run.name}
- **Run Number**: #{workflow_run.run_number}
- **Branch**: `{workflow_run.head_branch}`
- **Commit**: `{workflow_run.head_sha}`
- **Status**: ❌ FAILED
- **Timestamp**: {workflow_run.created_at}
- **Triggered By**: @{workflow_run.actor.get('login', 'unknown')}
- **Workflow URL**: [View Run]({workflow_run.html_url})

## Failure Overview

Total Failed Jobs: **{len(failed_jobs)}**

| # | Job Name | Category | Severity | Duration |
|---|----------|----------|----------|----------|
"""

        for i, (job, analysis) in enumerate(zip(failed_jobs, analyses), 1):
            report += f"| {i} | {job.name} | {analysis.category} | {analysis.severity} | {job.duration_seconds}s |\n"

        report += "\n## Detailed Analysis\n\n"

        for i, (job, analysis) in enumerate(zip(failed_jobs, analyses), 1):
            report += f"""### {i}. {job.name}

**Status**: ❌ {job.conclusion}
**Category**: {analysis.category.replace('_', ' ').title()}
**Severity**: {analysis.severity.upper()}
**Started**: {job.started_at}
**Completed**: {job.completed_at}
**Duration**: {job.duration_seconds} seconds
**Job URL**: [View Logs]({job.html_url})

#### Failed Steps

"""
            if job.steps:
                for step in job.steps:
                    if step.get('conclusion') == 'failure':
                        report += f"- **Step {step.get('number')}**: {step.get('name')}\n"
            else:
                report += "*No step information available*\n"

            report += "\n#### Error Analysis\n\n"

            if analysis.error_patterns:
                report += "**Detected Error Patterns:**\n\n"
                for pattern in analysis.error_patterns[:5]:
                    report += f"- Pattern: `{pattern.get('pattern')}`\n"
                    report += f"  - Occurrences: {pattern.get('count')}\n"
                    if pattern.get('matches'):
                        report += "  - Sample matches:\n"
                        for match in pattern['matches'][:3]:
                            report += f"    - Line {match.get('line_number')}: `{match.get('content')[:100]}`\n"
                    report += "\n"
            else:
                report += "*No specific error patterns detected*\n\n"

            report += "#### Suggested Fixes\n\n"

            if analysis.suggested_fixes:
                for j, fix in enumerate(analysis.suggested_fixes, 1):
                    report += f"{j}. {fix}\n"
            else:
                report += "*No automated suggestions available*\n"

            report += "\n---\n\n"

        report += """## Action Items

- [ ] Review detailed logs for each failed job
- [ ] Implement suggested fixes
- [ ] Add or update tests to prevent regression
- [ ] Verify fixes locally before pushing
- [ ] Update CI/CD configuration if needed

## Additional Resources

- [Workflow File](.github/workflows/)
- [CI/CD Documentation](../../docs/ci-cd/)
- [Troubleshooting Guide](../../docs/troubleshooting/)

---

📊 *Report generated on {timestamp}*
🤖 *JARVIS CI/CD Auto-PR Manager*
""".format(timestamp=datetime.now().isoformat())

        return report

    async def check_similar_prs(self, base_branch: str, workflow_name: str) -> Optional[str]:
        """Check if similar PR already exists"""
        if not self.config.get('safety.skip_if_similar_pr_exists', True):
            return None

        open_prs = await self.github.list_open_prs(base_branch)

        for pr in open_prs:
            # Check if PR title contains workflow name and has ci-failure label
            if workflow_name.lower() in pr.get('title', '').lower():
                labels = [label['name'] for label in pr.get('labels', [])]
                if 'ci-failure' in labels:
                    return pr.get('html_url')

        return None


class CIAutoPRManager:
    """Main orchestrator for CI/CD auto-PR creation"""

    def __init__(self):
        self.config = ConfigManager()
        self.token = os.environ.get('GITHUB_TOKEN')
        self.repo = os.environ.get('GITHUB_REPOSITORY')

        if not self.token or not self.repo:
            raise ValueError("GITHUB_TOKEN and GITHUB_REPOSITORY must be set")

    async def process_workflow_failure(self, run_id: str) -> Dict[str, Any]:
        """Process a failed workflow and create PR"""
        result = {
            'success': False,
            'pr_url': None,
            'error': None,
            'skipped': False,
            'skip_reason': None
        }

        async with GitHubAPIClient(self.token, self.repo) as github:
            try:
                # Fetch workflow run details
                print(f"📥 Fetching workflow run details for ID: {run_id}")
                workflow_data = await github.get_workflow_run(run_id)
                workflow_run = WorkflowRun(**{
                    'id': str(workflow_data['id']),
                    'name': workflow_data['name'],
                    'conclusion': workflow_data['conclusion'],
                    'html_url': workflow_data['html_url'],
                    'head_branch': workflow_data['head_branch'],
                    'head_sha': workflow_data['head_sha'],
                    'run_number': workflow_data['run_number'],
                    'run_attempt': workflow_data.get('run_attempt', 1),
                    'created_at': workflow_data['created_at'],
                    'updated_at': workflow_data['updated_at'],
                    'actor': workflow_data.get('actor', {})
                })

                print(f"✅ Workflow: {workflow_run.name}")
                print(f"✅ Conclusion: {workflow_run.conclusion}")

                # Fetch jobs
                print(f"📥 Fetching workflow jobs...")
                all_jobs = await github.get_workflow_jobs(run_id)
                failed_job_data = [job for job in all_jobs if job.get('conclusion') == 'failure']

                print(f"✅ Found {len(failed_job_data)} failed jobs")

                # Parse failed jobs
                failed_jobs = []
                for job_data in failed_job_data:
                    started = datetime.fromisoformat(job_data['started_at'].replace('Z', '+00:00'))
                    completed = datetime.fromisoformat(job_data['completed_at'].replace('Z', '+00:00'))
                    duration = int((completed - started).total_seconds())

                    failed_jobs.append(FailedJob(
                        id=str(job_data['id']),
                        name=job_data['name'],
                        conclusion=job_data['conclusion'],
                        html_url=job_data['html_url'],
                        started_at=job_data['started_at'],
                        completed_at=job_data['completed_at'],
                        steps=job_data.get('steps', []),
                        runner_name=job_data.get('runner_name'),
                        duration_seconds=duration
                    ))

                # Check for similar PRs
                pr_manager = PRManager(self.config, github)
                similar_pr = await pr_manager.check_similar_prs(workflow_run.head_branch, workflow_run.name)

                if similar_pr:
                    result['skipped'] = True
                    result['skip_reason'] = f"Similar PR already exists: {similar_pr}"
                    print(f"⏭️ Skipping: {result['skip_reason']}")
                    return result

                # Analyze failures
                print(f"🔍 Analyzing failures...")
                analyzer = FailureAnalyzer(self.config)
                analyses = []

                # Fetch logs if enabled
                fetch_logs = self.config.get('analysis.fetch_logs', True)

                for job in failed_jobs:
                    logs = None
                    if fetch_logs:
                        try:
                            logs = await github.get_job_logs(job.id)
                        except Exception as e:
                            print(f"⚠️ Could not fetch logs for job {job.name}: {e}")

                    analysis = await analyzer.analyze_failure(job, logs)
                    analyses.append(analysis)
                    print(f"  ✓ Analyzed: {job.name} (Category: {analysis.category}, Severity: {analysis.severity})")

                # Generate PR content
                print(f"📝 Generating PR content...")
                branch_name = pr_manager.generate_branch_name(workflow_run.name, workflow_run.run_number)
                pr_title = pr_manager.generate_pr_title(
                    workflow_run.name,
                    workflow_run.run_number,
                    workflow_run.head_branch,
                    workflow_run.head_sha
                )
                pr_body = pr_manager.generate_pr_body(workflow_run, failed_jobs, analyses)
                failure_report = pr_manager.generate_failure_report(workflow_run, failed_jobs, analyses)

                # Output results
                output_data = {
                    'workflow_name': workflow_run.name,
                    'workflow_url': workflow_run.html_url,
                    'branch': workflow_run.head_branch,
                    'commit_sha': workflow_run.head_sha[:7],
                    'run_number': workflow_run.run_number,
                    'failed_count': len(failed_jobs),
                    'fix_branch': branch_name,
                    'pr_title': pr_title,
                    'analyses': [asdict(a) for a in analyses],
                    'timestamp': datetime.now().isoformat()
                }

                # Save outputs
                self._save_outputs(output_data, pr_body, failure_report)

                result['success'] = True
                result['data'] = output_data
                print(f"✅ Successfully processed workflow failure")

                return result

            except Exception as e:
                result['error'] = str(e)
                print(f"❌ Error processing workflow: {e}")
                import traceback
                traceback.print_exc()
                return result

    def _save_outputs(self, data: Dict[str, Any], pr_body: str, report: str):
        """Save outputs for GitHub Actions"""
        # Save to GITHUB_OUTPUT
        with open(os.environ.get('GITHUB_OUTPUT', '/dev/null'), 'a') as f:
            for key, value in data.items():
                if isinstance(value, (str, int, float, bool)):
                    f.write(f"{key}={value}\n")

        # Save detailed data
        Path('workflow_failure_details.json').write_text(json.dumps(data, indent=2))
        Path('pr_body.md').write_text(pr_body)
        Path('CI_FAILURE_REPORT.md').write_text(report)

        print(f"💾 Outputs saved")


async def main():
    """Main entry point"""
    run_id = os.environ.get('WORKFLOW_RUN_ID')

    if not run_id:
        print("❌ WORKFLOW_RUN_ID environment variable is required")
        sys.exit(1)

    manager = CIAutoPRManager()
    result = await manager.process_workflow_failure(run_id)

    if result.get('skipped'):
        print(f"⏭️ {result.get('skip_reason')}")
        sys.exit(0)
    elif not result.get('success'):
        print(f"❌ Failed: {result.get('error')}")
        sys.exit(1)
    else:
        print(f"✅ Success!")
        sys.exit(0)


if __name__ == '__main__':
    asyncio.run(main())
