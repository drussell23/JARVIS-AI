# JARVIS CI/CD Pipelines

This directory contains GitHub Actions workflows for continuous integration and deployment of the JARVIS AI Assistant.

## 📋 Workflows Overview

### 1. 🧪 **Test Pipeline** (`test.yml`)
**Trigger:** On every push and pull request

**Purpose:** Ensure code quality and validate hybrid architecture integration

**Jobs:**
- **Unit & Integration Tests**
  - Runs pytest on `tests/unit/` and `tests/integration/`
  - Generates code coverage reports
  - Tests multiple Python versions (3.10, 3.11)

- **Hybrid Architecture Validation**
  - Validates `hybrid_config.yaml` structure
  - Verifies UAE/SAI/CAI configuration
  - Checks backend capabilities and routing rules
  - Confirms intelligence system integration

- **Code Quality Check**
  - Runs flake8 linting
  - Catches critical syntax errors

**Coverage:**
- Uploaded to Codecov automatically
- View at: `https://codecov.io/gh/[your-org]/JARVIS-AI-Agent`

---

### 2. 🚀 **Deployment Pipeline** (`deploy-to-gcp.yml`)
**Trigger:**
- Push to `main` or `multi-monitor-support` branches
- Changes to `backend/**` files
- Manual workflow dispatch

**Purpose:** Deploy JARVIS backend to GCP VM with zero-downtime and automatic rollback

**Jobs:**

#### **Pre-Deployment Checks**
- Validates hybrid configuration
- Checks critical files exist
- Ensures intelligence systems are enabled

#### **Deploy**
1. **Backup Current Deployment**
   - Creates timestamped backup
   - Keeps last 5 backups
   - Enables quick rollback

2. **Deploy New Version**
   - Stops running backend
   - Pulls latest code from branch
   - Updates dependencies
   - Starts new backend

3. **Health Checks (with retries)**
   - Tests `/health` endpoint (5 retries)
   - Validates Hybrid Orchestrator initialization
   - Confirms UAE/SAI/CAI availability

4. **Automatic Rollback**
   - Triggers if health checks fail
   - Reverts to previous commit
   - Restarts with old version
   - Logs failure details

#### **Post-Deployment Tests**
- Backend health validation
- Hybrid architecture API tests
- Integration test suite

**Features:**
- ✅ Zero-downtime deployment
- ✅ Automatic rollback on failure
- ✅ Health check retries
- ✅ Deployment backups
- ✅ Detailed summaries in GitHub Actions

---

### 3. 🔄 **Database Sync Pipeline** (`sync-databases.yml`)
**Trigger:**
- Scheduled: Every 6 hours (`0 */6 * * *`)
- Manual workflow dispatch

**Purpose:** Synchronize learning databases and aggregate patterns across deployments

**Jobs:**

#### **Sync**
1. **Export Learning Data**
   - Gathers metrics from learning database
   - Identifies patterns and insights
   - Prepares data for aggregation

2. **Sync to GCP**
   - Runs database optimization
   - Cleans up old patterns (30+ days)
   - Updates aggregated statistics

3. **Backup**
   - Creates timestamped database backups
   - Keeps last 7 days of backups
   - Ensures data safety

#### **Health Check**
- Validates database integrity
- Checks cache hit rates
- Monitors pattern count
- Reports warnings for issues

**Manual Trigger:**
```bash
# From GitHub Actions UI
# Select "Sync Learning Databases"
# Click "Run workflow"
# Option: force_full_sync (true/false)
```

---

## 🔧 Required GitHub Secrets

Configure these in your repository settings (`Settings > Secrets and variables > Actions`):

```bash
GCP_SA_KEY           # GCP service account JSON key
GCP_PROJECT_ID       # jarvis-473803
GCP_VM_NAME          # jarvis-backend-vm
GCP_ZONE             # us-central1-a
```

### How to Get GCP Service Account Key:
```bash
# Create service account
gcloud iam service-accounts create jarvis-deployer \
  --display-name="JARVIS GitHub Actions Deployer"

# Grant permissions
gcloud projects add-iam-policy-binding jarvis-473803 \
  --member="serviceAccount:jarvis-deployer@jarvis-473803.iam.gserviceaccount.com" \
  --role="roles/compute.instanceAdmin.v1"

# Create key
gcloud iam service-accounts keys create jarvis-sa-key.json \
  --iam-account=jarvis-deployer@jarvis-473803.iam.gserviceaccount.com

# Copy contents to GitHub secret GCP_SA_KEY
cat jarvis-sa-key.json
```

---

## 📊 Workflow Status Badges

Add these to your main README.md:

```markdown
![Tests](https://github.com/[your-username]/JARVIS-AI-Agent/workflows/Test%20JARVIS/badge.svg)
![Deployment](https://github.com/[your-username]/JARVIS-AI-Agent/workflows/Deploy%20JARVIS%20to%20GCP/badge.svg)
![Database Sync](https://github.com/[your-username]/JARVIS-AI-Agent/workflows/Sync%20Learning%20Databases/badge.svg)
```

---

## 🎯 Best Practices

### When to Commit
1. **Feature branches** → Tests run automatically
2. **multi-monitor-support branch** → Tests + Deployment
3. **main branch** → Tests + Deployment to production

### Deployment Strategy
```
feature-branch → multi-monitor-support → main
     ↓                    ↓                ↓
   tests              tests+deploy      tests+deploy
                      (staging)         (production)
```

### Manual Deployment
When you need to deploy without code changes:

```bash
# Go to GitHub Actions
# Select "Deploy JARVIS to GCP"
# Click "Run workflow"
# Select branch
# (Optional) Skip pre-deployment tests
```

---

## 🐛 Troubleshooting

### Deployment Failed
1. Check GitHub Actions logs
2. Deployment automatically rolls back
3. Previous version still running
4. Fix issue and push again

### Tests Failing
1. Review test output in Actions tab
2. Run tests locally: `cd backend && pytest tests/`
3. Fix issues and commit

### Database Sync Issues
1. Check sync workflow logs
2. SSH to GCP VM: `gcloud compute ssh jarvis-backend-vm --zone=us-central1-a`
3. Check database: `cd ~/backend/backend && venv/bin/python -c "from intelligence.learning_database import *"`

---

## 📈 Monitoring

### View Deployment History
```bash
# GitHub Actions tab shows:
- All workflow runs
- Success/failure status
- Deployment duration
- Commit deployed
```

### Check Current Deployment
```bash
# Backend status
curl http://34.10.137.70:8010/health

# View deployed commit
gcloud compute ssh jarvis-backend-vm --zone=us-central1-a \
  --command="cd ~/backend && git rev-parse HEAD"
```

---

## 🚀 Future Enhancements

### Planned Features:
- [ ] Multi-environment deployments (dev/staging/prod)
- [ ] Canary deployments (gradual rollout)
- [ ] Performance regression testing
- [ ] Automated database migrations
- [ ] Slack/Discord notifications
- [ ] Load testing before deployment

---

## 📝 Workflow Files

| File | Purpose | Trigger |
|------|---------|---------|
| `test.yml` | Run tests and validate config | Every push/PR |
| `deploy-to-gcp.yml` | Deploy to GCP with rollback | Push to main/multi-monitor-support |
| `sync-databases.yml` | Sync learning databases | Every 6 hours or manual |

---

## ✅ Current Status

**Hybrid Architecture:** ✅ Fully Integrated
- UAE (Unified Awareness Engine)
- SAI (Self-Aware Intelligence)
- CAI (Context Awareness Intelligence)
- Learning Database

**CI/CD:** ✅ Production Ready
- Automated testing
- Zero-downtime deployment
- Automatic rollback
- Database synchronization

**Deployment Target:** GCP VM `34.10.137.70:8010`

---

---

## 🆕 **NEW WORKFLOWS ADDED (2025-10-30)**

### 4. 🔍 **Super-Linter** (`super-linter.yml`)
**Trigger:** Push to any branch, PRs to main/develop

**Purpose:** Comprehensive multi-language code quality enforcement

**Features:**
- Python: Black, Flake8, Pylint, MyPy, isort, Bandit
- JavaScript/TypeScript linting
- Shell script validation
- YAML/JSON/XML validation
- Markdown formatting
- Secret detection

---

### 5. 🔒 **CodeQL Security Analysis** (`codeql-analysis.yml`)
**Trigger:** Push/PR, Daily at 2 AM UTC, Manual

**Purpose:** Advanced security scanning and vulnerability detection

**Features:**
- Multi-language analysis (Python, JS/TS)
- Extended security queries
- SARIF upload to GitHub Security
- Daily automated scans
- Detects SQL injection, XSS, auth issues

---

### 6. 🗄️ **Database Validation** (`database-validation.yml`)
**Trigger:** Push/PR affecting backend, Daily at 3 AM UTC

**Purpose:** Comprehensive database configuration validation

**Checks:**
- .env.example completeness
- Hardcoded credential detection
- SQL injection vulnerability scanning
- Cloud SQL Proxy configuration
- Connection pooling validation
- Migration framework detection

---

### 7. 📋 **Environment Variable Validation** (`env-validation.yml`)
**Trigger:** Push/PR affecting code files, .env.example

**Purpose:** Ensure all env vars are documented and secure

**Features:**
- Tracks env var usage across codebase
- Reports documentation coverage (requires >80%)
- Detects hardcoded sensitive data
- Validates critical vars documented
- Comprehensive reporting

---

### 8. 🚀 **Comprehensive CI/CD Pipeline** (`ci-cd-pipeline.yml`)
**Trigger:** Push to any branch, PRs, Manual

**Purpose:** Full-stack CI/CD with 6 phases

**Phases:**
1. **Code Quality:** Black, Flake8, Pylint, MyPy, Bandit, Safety
2. **Build & Test:** Matrix testing (Python 3.10/3.11, Ubuntu/macOS)
3. **Architecture Validation:** Hybrid config, dependencies
4. **Performance Testing:** Benchmarks and load testing
5. **Security Scanning:** Trivy, Gitleaks
6. **Pipeline Reporting:** Summary and PR comments

---

### 9. 🤖 **PR Automation** (`pr-automation.yml`)
**Trigger:** PR events, reviews, comments

**Purpose:** Intelligent PR automation and validation

**Features:**
- **Auto-Labeling:** File-based + intelligent context labels
- **Size Analysis:** Automatic PR size calculation with warnings
- **Title Validation:** Conventional Commits enforcement
- **Description Checks:** Quality and completeness validation
- **Conflict Detection:** Automatic merge conflict alerts
- **Review Management:** Auto-assignment and reminders

---

### 10. 🚢 **Deployment Pipeline** (`deployment.yml`)
**Trigger:** Push to main, Version tags, Manual dispatch

**Purpose:** Production-ready deployment with environment protection

**Environments:**
- **Staging:** Auto-deploy on main push
- **Production:** Manual approval required

**Flow:**
1. Pre-deployment checks & version tagging
2. Run critical test suite
3. Build backend & frontend artifacts
4. Deploy to staging with smoke tests
5. Deploy to production (manual approval)
6. Create GitHub release
7. Post-deployment monitoring (5 min)

**Features:**
- Zero-downtime deployments
- Database backups before production
- Health checks with automatic rollback
- Release note generation
- Team notifications

---

### 11. 📦 **Dependabot** (`dependabot.yml`)
**Purpose:** Automated dependency updates

**Schedule:**
- Python: Weekly (Mondays)
- NPM: Weekly (Mondays)
- GitHub Actions: Weekly (Tuesdays)
- Docker: Weekly (Wednesdays)
- Terraform: Weekly (Thursdays)

**Features:**
- Grouped updates for related packages
- Automatic PR creation with reviewers
- Major version protection for critical deps
- Semantic versioning strategy

---

## 📚 **New Documentation**

### Configuration Files
- `.github/linters/.python-black` - Black formatter config
- `.github/linters/.isort.cfg` - Import sorting config
- `.github/labeler.yml` - Auto-labeling rules (40+ labels)
- `.github/PULL_REQUEST_TEMPLATE.md` - Comprehensive PR template

### Documentation
- `.github/GITHUB_ACTIONS_GUIDE.md` - **Complete 200+ line guide** covering:
  - All workflows in detail
  - Setup instructions
  - Secrets management
  - Deployment processes
  - Troubleshooting
  - Best practices

---

## 🎯 **Complete Workflow Matrix**

| # | Workflow | File | Status | Purpose |
|---|----------|------|--------|---------|
| 1 | Test Pipeline | `test.yml` | ✅ Existing | Unit/integration tests |
| 2 | GCP Deployment | `deploy-to-gcp.yml` | ✅ Existing | GCP VM deployment |
| 3 | Database Sync | `sync-databases.yml` | ✅ Existing | Learning DB sync |
| 4 | Config Validation | `validate-config.yml` | ✅ Existing | Config checks |
| 5 | Super-Linter | `super-linter.yml` | 🆕 NEW | Multi-language linting |
| 6 | CodeQL Security | `codeql-analysis.yml` | 🆕 NEW | Security analysis |
| 7 | Database Validation | `database-validation.yml` | 🆕 NEW | DB config validation |
| 8 | Env Validation | `env-validation.yml` | 🆕 NEW | Env var validation |
| 9 | CI/CD Pipeline | `ci-cd-pipeline.yml` | 🆕 NEW | Comprehensive pipeline |
| 10 | PR Automation | `pr-automation.yml` | 🆕 NEW | PR automation |
| 11 | Deployment | `deployment.yml` | 🆕 NEW | Production deployment |
| 12 | Dependabot | `dependabot.yml` | 🆕 NEW | Dependency updates |

---

## 🛡️ **Security Features**

### Automated Security Scanning
- ✅ CodeQL daily scans
- ✅ Trivy filesystem scanning
- ✅ Gitleaks secret detection
- ✅ Bandit Python security analysis
- ✅ Safety dependency vulnerability checks
- ✅ Hardcoded credential detection

### Security Best Practices
- ✅ No secrets in code (enforced)
- ✅ Environment variable validation
- ✅ Regular dependency updates
- ✅ Security alerts in GitHub Security tab
- ✅ SARIF report uploads

---

## 📊 **Enhanced Monitoring**

### Automated Reporting
- GitHub Actions summaries for all workflows
- PR comments with detailed feedback
- Deployment status notifications
- Security alert integration
- Coverage reports to Codecov

### Key Metrics Tracked
- Code quality scores
- Test coverage percentage
- Deployment frequency
- Security vulnerabilities
- Dependency freshness
- PR size distribution

---

## 🚀 **Getting Started with New Workflows**

### 1. Add Required Secrets
```bash
# Required for production deployment
GCP_PRODUCTION_SERVICE_ACCOUNT_KEY
CODECOV_TOKEN (optional but recommended)
```

### 2. Configure Environments
- Create `staging` environment (no protection)
- Create `production` environment (manual approval required)

### 3. Set Up Branch Protection
- Require status checks on `main` branch
- Require 1+ reviews
- Enable automated checks

### 4. Enable Dependabot
- Dependabot automatically enabled
- Review PRs weekly
- Configure auto-merge for patch updates (optional)

---

## 🎓 **Learning Resources**

### Complete Documentation
- **[GITHUB_ACTIONS_GUIDE.md](.github/GITHUB_ACTIONS_GUIDE.md)** - Comprehensive guide
- **[PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md)** - PR guidelines
- Individual workflow files have detailed inline comments

### Quick Links
- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [CodeQL Docs](https://codeql.github.com/docs/)
- [Dependabot Docs](https://docs.github.com/en/code-security/dependabot)

---

## 📈 **System Status**

**CI/CD Maturity:** 🔥 **Advanced**
- ✅ Automated testing (multi-platform)
- ✅ Security scanning (daily)
- ✅ Code quality enforcement
- ✅ Automated deployments
- ✅ Environment protection
- ✅ Dependency management
- ✅ PR automation
- ✅ Comprehensive monitoring

**Deployment Capability:** 🚀 **Production-Ready**
- Zero-downtime deployments
- Automatic rollback
- Multi-environment support
- Health check monitoring
- Release automation

---

Last Updated: 2025-10-30
