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

Last Updated: 2025-10-24
