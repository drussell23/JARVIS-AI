# 🤖 GitHub Actions CI/CD Guide

Complete guide to the automated CI/CD pipelines for the JARVIS AI Agent project.

## 📋 Table of Contents
- [Overview](#overview)
- [Workflows](#workflows)
- [Setup & Configuration](#setup--configuration)
- [Secrets Management](#secrets-management)
- [Deployment Process](#deployment-process)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This project uses a comprehensive GitHub Actions setup with:
- ✅ Automated code quality checks
- 🔒 Security scanning (CodeQL, Trivy, Gitleaks)
- 🧪 Automated testing
- 📦 Dependency management (Dependabot)
- 🚀 Automated deployments
- 🤖 PR automation and labeling
- 📊 Performance monitoring

## 🔄 Workflows

### 1. **Super-Linter** (`super-linter.yml`)
**Triggers:** Push to any branch, PRs to main/develop

**Purpose:** Comprehensive code linting across multiple languages

**What it checks:**
- Python: Black, Flake8, Pylint, MyPy, isort, Bandit
- JavaScript/TypeScript: ESLint
- Shell scripts: shellcheck, shfmt
- YAML, JSON, XML validation
- Dockerfile linting
- Markdown formatting
- SQL validation
- Secret detection

**Passing criteria:** No critical linting errors

---

### 2. **CodeQL Security Analysis** (`codeql-analysis.yml`)
**Triggers:** Push to main/develop, PRs, daily at 2 AM UTC, manual

**Purpose:** Deep security analysis of codebase

**Features:**
- Multi-language scanning (Python, JavaScript/TypeScript)
- Extended security queries
- Security-and-quality ruleset
- Automated SARIF upload to GitHub Security tab
- Daily scheduled scans

**Key checks:**
- SQL injection vulnerabilities
- Cross-site scripting (XSS)
- Command injection
- Path traversal
- Authentication issues
- Cryptographic weaknesses

---

### 3. **Database Validation** (`database-validation.yml`)
**Triggers:** Push/PR affecting backend, daily at 3 AM UTC, manual

**Purpose:** Validate database configuration and connections

**Checks:**
- .env.example completeness
- Database connection code patterns
- Hardcoded credential detection
- SQL injection vulnerability scanning
- Cloud SQL Proxy configuration
- Database migration setup
- Connection pooling configuration

---

### 4. **Environment Variable Validation** (`env-validation.yml`)
**Triggers:** Push/PR affecting code, .env.example, or workflow file

**Purpose:** Comprehensive environment variable validation

**Features:**
- Tracks all env vars used in codebase
- Compares usage vs. documentation
- Reports documentation coverage (requires >80%)
- Detects hardcoded sensitive data
- Validates critical env vars are documented

**Coverage targets:**
- Minimum 80% documentation coverage
- All critical vars must be documented

---

### 5. **Comprehensive CI/CD Pipeline** (`ci-cd-pipeline.yml`)
**Triggers:** Push to any branch, PRs, manual

**Purpose:** Full CI/CD pipeline with multiple stages

**Stages:**

#### Phase 1: Code Quality
- Black formatting check
- Flake8 linting
- Pylint advanced linting
- MyPy type checking
- isort import sorting
- Bandit security scanning
- Safety dependency vulnerability check

#### Phase 2: Build & Test
- Matrix testing (Python 3.10, 3.11 on Ubuntu & macOS)
- Unit tests with coverage
- Integration tests
- Parallel test execution
- Coverage reporting to Codecov

#### Phase 3: Architecture Validation
- Hybrid architecture config validation
- Component dependency analysis
- Intelligence system validation (UAE, SAI, CAI)

#### Phase 4: Performance Testing
- Performance benchmarks
- Load testing (on main/develop only)

#### Phase 5: Security Scanning
- Trivy filesystem scanning
- Gitleaks secret scanning

#### Phase 6: Pipeline Reporting
- Comprehensive summary generation
- Automatic PR comments with results

---

### 6. **PR Automation** (`pr-automation.yml`)
**Triggers:** PR events, PR reviews, comments

**Purpose:** Automate PR workflows and provide feedback

**Features:**

#### Auto-Labeling
- File-based labeling (backend, frontend, tests, etc.)
- Size labels (XS, S, M, L, XL)
- Intelligent context-aware labeling
- Technology-specific labels (database, API, security)

#### PR Size Analysis
- Automatic size calculation
- Warning for large PRs
- Review checklist generation

#### PR Title Validation
- Conventional Commits format enforcement
- Types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert

#### PR Description Checks
- Minimum length validation
- Section completeness check
- Improvement suggestions

#### Conflict Detection
- Automatic merge conflict detection
- Resolution instructions

#### Review Management
- Automatic reviewer assignment
- Review reminders

---

### 7. **Deployment Pipeline** (`deployment.yml`)
**Triggers:** Push to main, version tags (v*.*.*), manual dispatch

**Purpose:** Automated deployments with environment protection

**Environments:**
- **Staging:** Automatic on push to main
- **Production:** Tag-based or manual approval required

**Deployment Flow:**

1. **Pre-Deployment Checks**
   - Environment determination
   - Version extraction
   - Security validation

2. **Run Tests**
   - Critical test suite
   - Skippable via manual trigger

3. **Build Artifacts**
   - Backend deployment package
   - Frontend build (if applicable)
   - Version tagging

4. **Deploy to Staging**
   - Automatic deployment
   - Smoke tests
   - Health checks

5. **Deploy to Production**
   - Manual approval required (environment protection)
   - Database backup creation
   - Zero-downtime deployment
   - Comprehensive health checks
   - Team notifications

6. **Create Release**
   - Automatic GitHub release
   - Generated release notes
   - Changelog from commits

7. **Post-Deployment Monitoring**
   - 5-minute monitoring period
   - Error rate tracking
   - Performance verification

---

### 8. **Dependabot** (`dependabot.yml`)
**Purpose:** Automated dependency updates

**Update Schedule:**
- **Python (Backend):** Weekly (Mondays, 9 AM)
- **NPM (Frontend):** Weekly (Mondays, 9 AM)
- **GitHub Actions:** Weekly (Tuesdays, 9 AM)
- **Docker:** Weekly (Wednesdays, 9 AM)
- **Terraform:** Weekly (Thursdays, 9 AM)

**Features:**
- Grouped updates for related packages
- Automatic PR creation
- Reviewer assignment
- Semantic versioning strategy
- Major version update protection for critical deps

---

## ⚙️ Setup & Configuration

### Required GitHub Secrets

Add these secrets in: **Settings → Secrets and variables → Actions**

#### Production Deployment
```
GCP_SERVICE_ACCOUNT_KEY          # GCP service account JSON for staging
GCP_PRODUCTION_SERVICE_ACCOUNT_KEY  # GCP service account JSON for production
```

#### Optional (for enhanced features)
```
CODECOV_TOKEN                    # Codecov integration
SLACK_WEBHOOK_URL                # Slack notifications
DISCORD_WEBHOOK_URL              # Discord notifications
```

### Environment Configuration

#### Staging Environment
1. Go to **Settings → Environments → New environment**
2. Name: `staging`
3. No protection rules needed (auto-deploy)

#### Production Environment
1. Go to **Settings → Environments → New environment**
2. Name: `production`
3. **Enable protection rules:**
   - ✅ Required reviewers (add team members)
   - ✅ Wait timer: 5 minutes (optional)
   - ✅ Restrict to protected branches: `main`

### Branch Protection Rules

Recommended settings for `main` branch:

1. Go to **Settings → Branches → Add rule**
2. Branch name pattern: `main`
3. **Enable:**
   - ✅ Require pull request reviews (1+ reviewer)
   - ✅ Dismiss stale reviews
   - ✅ Require status checks to pass:
     - `Code Quality Analysis`
     - `Build & Test`
     - `Validate Database Configuration`
     - `Validate Environment Variables`
     - `CodeQL`
   - ✅ Require branches to be up to date
   - ✅ Require conversation resolution
   - ✅ Require signed commits (recommended)
   - ✅ Include administrators

---

## 🔐 Secrets Management

### Never Commit
- API keys
- Passwords
- Private keys
- Service account credentials
- Database credentials

### Always Use
- GitHub Secrets for CI/CD
- Environment variables in code
- `.env.example` for documentation (with placeholder values)

### Best Practices
1. Use `os.getenv()` with defaults in Python
2. Use `process.env` in JavaScript
3. Document all env vars in `.env.example`
4. Rotate secrets regularly
5. Use least-privilege service accounts

---

## 🚀 Deployment Process

### Staging Deployment (Automatic)
```bash
# Push to main
git checkout main
git merge feature/my-feature
git push origin main

# Deployment triggers automatically
# Monitor at: https://github.com/USER/REPO/actions
```

### Production Deployment (Manual Approval)

#### Option 1: Tag-based (Recommended)
```bash
# Create and push a version tag
git tag -a v1.2.3 -m "Release version 1.2.3"
git push origin v1.2.3

# Workflow requires manual approval in GitHub
# Go to Actions → Deployment Pipeline → Review deployments
```

#### Option 2: Manual Trigger
1. Go to **Actions → Deployment Pipeline**
2. Click **Run workflow**
3. Select **production** environment
4. Choose deployment options
5. Click **Run workflow**
6. Approve deployment when prompted

---

## 🐛 Troubleshooting

### Workflow Failed - Code Quality
**Issue:** Linting errors

**Solution:**
```bash
# Run locally
pip install black flake8 pylint mypy isort
black backend/
isort backend/
flake8 backend/
```

### Workflow Failed - Tests
**Issue:** Test failures

**Solution:**
```bash
cd backend
pytest tests/ -v
# Fix failing tests
```

### Workflow Failed - Database Validation
**Issue:** Hardcoded credentials detected

**Solution:**
```python
# BAD
password = "my-secret-password"

# GOOD
password = os.getenv("JARVIS_DB_PASSWORD")
```

### Workflow Failed - Env Validation
**Issue:** Coverage below 80%

**Solution:**
1. Add missing vars to `.env.example`
2. Document all env vars used in code

### Deployment Failed - Pre-checks
**Issue:** Security validation failed

**Solution:**
- Check commit messages for sensitive keywords
- Ensure `.env.example` exists
- Review code for exposed secrets

### Deployment Failed - Health Checks
**Issue:** Service not responding after deployment

**Solution:**
1. Check GCP logs
2. Verify service account permissions
3. Ensure environment variables are set
4. Check database connectivity

---

## 📊 Monitoring & Observability

### GitHub Actions Dashboard
- View workflow runs: `/actions`
- Check security alerts: `/security`
- Review Dependabot PRs: `/pulls`

### Key Metrics to Monitor
- ✅ Test pass rate
- 🐛 Linting error trends
- 🔒 Security vulnerabilities
- 📦 Dependency freshness
- 🚀 Deployment frequency
- ⏱️ Deployment duration

---

## 🎯 Best Practices

### For Developers
1. **Run tests locally** before pushing
2. **Use conventional commits** for PR titles
3. **Write meaningful descriptions** in PRs
4. **Keep PRs small** (< 500 lines preferred)
5. **Update documentation** with code changes
6. **Review security warnings** seriously

### For Maintainers
1. **Review Dependabot PRs** weekly
2. **Monitor security alerts** daily
3. **Approve production deploys** carefully
4. **Keep secrets rotated** regularly
5. **Update workflows** as needed

---

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)
- [CodeQL Documentation](https://codeql.github.com/docs/)
- [Dependabot Documentation](https://docs.github.com/en/code-security/dependabot)

---

## 🆘 Support

**Questions or Issues?**
- Open an issue with label `ci/cd`
- Tag maintainers for urgent matters
- Check workflow logs for detailed error messages

**Contributing?**
- All workflow changes should be tested
- Update this guide when adding new workflows
- Follow security best practices

---

**Last Updated:** 2025-10-30
**Maintained by:** JARVIS Team
