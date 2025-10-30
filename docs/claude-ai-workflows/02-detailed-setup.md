# 📖 Detailed Setup Guide - Claude AI Workflows

**Complete installation and configuration walkthrough**

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Pre-Installation Checklist](#pre-installation-checklist)
3. [Installation Methods](#installation-methods)
4. [Configuration](#configuration)
5. [Advanced Setup](#advanced-setup)
6. [Verification & Testing](#verification--testing)
7. [Post-Installation](#post-installation)

---

## System Requirements

### GitHub Repository Requirements
- ✅ GitHub repository (public or private)
- ✅ Admin access to repository
- ✅ GitHub Actions enabled
- ✅ Secrets management access

### Anthropic Requirements
- ✅ Anthropic account (free tier works)
- ✅ API access enabled
- ✅ Valid payment method (after free tier)

### Local Development Requirements
- ✅ Git installed
- ✅ GitHub CLI (optional but recommended)
- ✅ Terminal/Command line access

### Optional
- ⭐ Branch protection rules
- ⭐ Environment secrets (for staging/production)
- ⭐ Slack/Discord webhooks
- ⭐ CodeCov account

---

## Pre-Installation Checklist

### Step 1: Verify Repository Access

```bash
# Clone repository (if not already)
git clone https://github.com/YOUR-USERNAME/JARVIS-AI-Agent.git
cd JARVIS-AI-Agent

# Verify you have push access
git push --dry-run

# Check GitHub CLI authentication
gh auth status
```

### Step 2: Check Existing Workflows

```bash
# List existing workflows
ls .github/workflows/

# Check for conflicts
grep -r "ANTHROPIC_API_KEY" .github/workflows/
```

### Step 3: Understand Your Current Setup

```bash
# Check repository settings
gh repo view --json name,isPrivate,defaultBranch,hasIssuesEnabled

# List existing secrets
gh secret list

# Check Actions status
gh api repos/:owner/:repo/actions/permissions
```

---

## Installation Methods

### Method 1: Already Installed (Your Case)

The workflows are already in your repository! You just need to configure the API key.

**Skip to:** [Configuration](#configuration)

### Method 2: Fresh Installation

If setting up from scratch:

#### 2.1 Download Workflow Files

```bash
# Create workflows directory
mkdir -p .github/workflows

# Download workflow files
curl -o .github/workflows/claude-pr-analyzer.yml \
  https://raw.githubusercontent.com/drussell23/JARVIS-AI-Agent/main/.github/workflows/claude-pr-analyzer.yml

curl -o .github/workflows/claude-auto-fix.yml \
  https://raw.githubusercontent.com/drussell23/JARVIS-AI-Agent/main/.github/workflows/claude-auto-fix.yml

curl -o .github/workflows/claude-test-generator.yml \
  https://raw.githubusercontent.com/drussell23/JARVIS-AI-Agent/main/.github/workflows/claude-test-generator.yml

curl -o .github/workflows/claude-security-analyzer.yml \
  https://raw.githubusercontent.com/drussell23/JARVIS-AI-Agent/main/.github/workflows/claude-security-analyzer.yml

curl -o .github/workflows/claude-docs-generator.yml \
  https://raw.githubusercontent.com/drussell23/JARVIS-AI-Agent/main/.github/workflows/claude-docs-generator.yml
```

#### 2.2 Commit Workflows

```bash
git add .github/workflows/claude-*.yml
git commit -m "feat: Add Claude AI-powered workflows"
git push origin main
```

### Method 3: Fork & Customize

```bash
# Fork the repository
gh repo fork drussell23/JARVIS-AI-Agent

# Clone your fork
git clone https://github.com/YOUR-USERNAME/JARVIS-AI-Agent.git

# Create your own branch
git checkout -b feature/add-claude-ai

# Customize workflows as needed
# ... edit files ...

# Commit and push
git add .
git commit -m "feat: Customize Claude AI workflows"
git push origin feature/add-claude-ai
```

---

## Configuration

### 1. Anthropic API Key Setup

#### 1.1 Create Anthropic Account

```
URL: https://console.anthropic.com/signup
Steps:
1. Enter email
2. Create password
3. Verify email
4. Complete profile
```

#### 1.2 Understand API Tiers

**Free Tier ($0/month)**
- $5 free credit
- Rate limits: 50 requests/minute
- Perfect for testing

**Pay-as-you-go**
- $0.003 per 1K input tokens
- $0.015 per 1K output tokens
- Typical PR: $0.20-0.50

#### 1.3 Generate API Key

```
Console: https://console.anthropic.com/settings/keys

Steps:
1. Click "Create Key"
2. Name: "github-actions-jarvis"
3. Copy key immediately
4. Store in password manager
```

**API Key Format:**
```
sk-ant-api03-[BASE64_STRING]-[BASE64_STRING]-[SHORT_STRING]

Example Format:
sk-ant-api03-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-XXXXXXXX
```

### 2. GitHub Secrets Configuration

#### 2.1 Add Primary Secret

**Via Web Interface:**
```
1. Go to: https://github.com/YOUR-USER/JARVIS-AI-Agent/settings/secrets/actions
2. Click: "New repository secret"
3. Name: ANTHROPIC_API_KEY
4. Value: [Paste your API key]
5. Click: "Add secret"
```

**Via GitHub CLI:**
```bash
# Set secret
gh secret set ANTHROPIC_API_KEY

# You'll be prompted to paste the key
# Paste and press Ctrl+D (Unix) or Ctrl+Z (Windows)

# Or one-liner
echo "your-api-key" | gh secret set ANTHROPIC_API_KEY
```

**Via API:**
```bash
# Install libsodium for encryption
# macOS: brew install libsodium
# Ubuntu: sudo apt-get install libsodium-dev

# Encrypt and set secret (advanced)
gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  /repos/OWNER/REPO/actions/secrets/ANTHROPIC_API_KEY \
  -f encrypted_value='YOUR_ENCRYPTED_VALUE' \
  -f key_id='YOUR_KEY_ID'
```

#### 2.2 Verify Secret

```bash
# List secrets
gh secret list

# Expected output:
# ANTHROPIC_API_KEY    2025-10-30T06:34:32Z

# Check secret is accessible
gh secret list --json name,updatedAt | jq '.[] | select(.name=="ANTHROPIC_API_KEY")'
```

#### 2.3 Optional Secrets

```bash
# For enhanced features
gh secret set CODECOV_TOKEN           # Coverage reporting
gh secret set SLACK_WEBHOOK_URL       # Slack notifications
gh secret set DISCORD_WEBHOOK_URL     # Discord notifications
```

### 3. Workflow Permissions

#### 3.1 Verify Workflow Permissions

```
Settings → Actions → General → Workflow permissions

Required:
☑ Read and write permissions
☑ Allow GitHub Actions to create and approve pull requests
```

**Via CLI:**
```bash
gh api \
  --method GET \
  /repos/OWNER/REPO/actions/permissions
```

#### 3.2 Set Permissions (if needed)

```bash
gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  /repos/OWNER/REPO/actions/permissions \
  -f default_workflow_permissions='write' \
  -F can_approve_pull_request_reviews=true
```

---

## Advanced Setup

### 1. Environment Configuration

#### 1.1 Create Environments

```
Settings → Environments → New environment

Create two environments:
- staging (no protection)
- production (require reviews)
```

**Via CLI:**
```bash
# Create staging environment
gh api \
  --method PUT \
  /repos/OWNER/REPO/environments/staging \
  -f wait_timer=0

# Create production with protection
gh api \
  --method PUT \
  /repos/OWNER/REPO/environments/production \
  -f wait_timer=300 \
  -f reviewers[][type]=User \
  -f reviewers[][id]=YOUR_USER_ID
```

#### 1.2 Environment-Specific Secrets

```bash
# Add secret to specific environment
gh secret set ANTHROPIC_API_KEY \
  --env production \
  --body "your-production-key"

gh secret set ANTHROPIC_API_KEY \
  --env staging \
  --body "your-staging-key"
```

### 2. Branch Protection Rules

#### 2.1 Protect Main Branch

```
Settings → Branches → Add rule

Branch name pattern: main

☑ Require pull request reviews before merging
  ☑ Require approvals: 1
  ☑ Dismiss stale reviews
☑ Require status checks to pass
  ☑ Require branches to be up to date
  Required checks:
    - Claude AI PR Analyzer
    - Claude AI Security Analyzer
    - Code Quality Analysis
☑ Require conversation resolution
☑ Include administrators
```

**Via API:**
```bash
gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  /repos/OWNER/REPO/branches/main/protection \
  -f required_status_checks[strict]=true \
  -f required_status_checks[contexts][]='Claude AI PR Analyzer' \
  -f required_status_checks[contexts][]='Claude AI Security Analyzer' \
  -f required_pull_request_reviews[required_approving_review_count]=1 \
  -f enforce_admins=true
```

### 3. Webhook Configuration

#### 3.1 Set Up Slack Notifications

```bash
# Add Slack webhook
gh secret set SLACK_WEBHOOK_URL --body "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

**Create workflow for Slack:**
```yaml
# .github/workflows/slack-notify.yml
name: Slack Notifications

on:
  workflow_run:
    workflows: ["Claude AI PR Analyzer"]
    types: [completed]

jobs:
  notify:
    runs-on: ubuntu-latest
    steps:
      - name: Send Slack notification
        uses: slackapi/slack-github-action@v1
        with:
          webhook: ${{ secrets.SLACK_WEBHOOK_URL }}
          payload: |
            {
              "text": "Claude AI Analysis Complete",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "*PR:* ${{ github.event.workflow_run.head_repository.full_name }}#${{ github.event.workflow_run.pull_requests[0].number }}\n*Status:* ${{ github.event.workflow_run.conclusion }}"
                  }
                }
              ]
            }
```

---

## Verification & Testing

### 1. Verify Installation

```bash
# Check workflows exist
ls -la .github/workflows/claude-*.yml

# Verify secret
gh secret list | grep ANTHROPIC_API_KEY

# Check workflow files are valid YAML
yamllint .github/workflows/claude-*.yml

# Or use Python
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/claude-pr-analyzer.yml'))"
```

### 2. Test Run

#### 2.1 Manual Trigger Test

```bash
# Trigger workflow manually
gh workflow run claude-security-analyzer.yml

# Watch it run
gh run watch
```

#### 2.2 Create Test PR

```bash
# Create test branch
git checkout -b test/claude-setup-verification

# Add test file
cat > setup_test.py << 'EOF'
# This file tests the Claude AI setup
def setup_verification():
    """Verify Claude AI workflows are working."""
    return "Setup successful!"
EOF

# Commit and push
git add setup_test.py
git commit -m "test: Verify Claude AI workflows setup"
git push -u origin test/claude-setup-verification

# Create PR
gh pr create \
  --title "test: Claude AI Setup Verification" \
  --body "This PR verifies that Claude AI workflows are properly configured and running." \
  --base main \
  --head test/claude-setup-verification
```

#### 2.3 Monitor Test PR

```bash
# Get PR number
PR_NUMBER=$(gh pr list --head test/claude-setup-verification --json number --jq '.[0].number')

# Watch checks
watch -n 5 "gh pr checks $PR_NUMBER"

# View AI comments
gh pr view $PR_NUMBER --comments

# Check workflow runs
gh run list --workflow="claude-pr-analyzer.yml" --limit 5
```

### 3. Validate Results

Expected results within 5 minutes:

✅ **Workflows Running:**
- Claude AI PR Analyzer
- Claude AI Auto-Fix
- Claude AI Test Generator
- Claude AI Security Analyzer
- Claude AI Documentation Generator

✅ **PR Comments:**
- Detailed code review from Claude
- Security analysis
- PR size analysis

✅ **Labels Applied:**
- Intelligent labels based on content

✅ **New Commits (possibly):**
- Auto-fixes
- Generated tests
- Documentation

---

## Post-Installation

### 1. Clean Up Test PR

```bash
# Close test PR
gh pr close $PR_NUMBER --comment "Setup verification complete! ✅"

# Delete test branch
git branch -D test/claude-setup-verification
git push origin --delete test/claude-setup-verification

# Delete test file
git checkout main
git branch -D test/claude-setup-verification
```

### 2. Configure Team Access

```bash
# Add team reviewers
gh api \
  --method PUT \
  /repos/OWNER/REPO/teams/TEAM_NAME/permissions \
  -f permission=push

# Configure CODEOWNERS
cat > .github/CODEOWNERS << 'EOF'
# Auto-assign reviewers
*.yml @YOUR_TEAM
*.py @YOUR_TEAM
EOF

git add .github/CODEOWNERS
git commit -m "chore: Add CODEOWNERS"
git push
```

### 3. Set Up Monitoring

```bash
# Create dashboard bookmark
echo "Workflows Dashboard: https://github.com/$GITHUB_REPOSITORY/actions"
echo "Secrets Manager: https://github.com/$GITHUB_REPOSITORY/settings/secrets/actions"
echo "Anthropic Usage: https://console.anthropic.com/settings/usage"
```

### 4. Document for Team

Create a team guide in your repository:

```markdown
# Claude AI Workflows - Team Guide

## Quick Commands
- `@claude` - Full PR analysis
- `@claude fix` - Auto-fix code
- `@claude generate tests` - Create tests

## When to Use
- Every PR gets automatic analysis
- Mention @claude for re-analysis
- Review AI suggestions carefully

## Cost Awareness
- Average PR: $0.20-0.50
- Monthly budget: ~$20
- Check usage: [Anthropic Console](https://console.anthropic.com/settings/usage)

## Support
- Issues: Create GitHub issue with `ai-workflows` label
- Questions: #dev-ai-tools Slack channel
```

---

## Troubleshooting Setup

### Issue: API Key Not Working

**Symptoms:**
- Workflows fail with "API key invalid"
- 401 Unauthorized errors

**Solutions:**
```bash
# Regenerate API key
# 1. Go to: https://console.anthropic.com/settings/keys
# 2. Delete old key
# 3. Create new key
# 4. Update GitHub secret

gh secret set ANTHROPIC_API_KEY --body "new-key-here"

# Trigger test run
gh workflow run claude-pr-analyzer.yml
```

### Issue: Workflows Not Triggering

**Symptoms:**
- No workflows appear in PR
- Actions tab is empty

**Solutions:**
```bash
# Check workflow permissions
gh api /repos/OWNER/REPO/actions/permissions

# Enable workflows
gh api \
  --method PUT \
  /repos/OWNER/REPO/actions/permissions \
  -f enabled=true

# Check if workflows are disabled
cat .github/workflows/claude-pr-analyzer.yml | grep -A 5 "^on:"
```

### Issue: Insufficient Permissions

**Symptoms:**
- "Permission denied" errors
- Cannot create commits

**Solutions:**
```
Settings → Actions → General → Workflow permissions
☑ Read and write permissions
☑ Allow GitHub Actions to create and approve pull requests
```

---

## Advanced Configuration

### Custom Model Selection

Edit workflows to use different Claude models:

```yaml
# In .github/workflows/claude-pr-analyzer.yml
# Change this line:
model="claude-sonnet-4-20250514"

# To:
model="claude-opus-4-20250514"  # More powerful, more expensive
# or
model="claude-haiku-4-20250514"  # Faster, cheaper
```

### Adjust Token Limits

```yaml
# Increase for larger PRs
max_tokens=8000

# To:
max_tokens=16000  # For very large PRs
```

### Custom System Prompts

```yaml
# Customize AI behavior in workflows
system="""You are an expert code reviewer...
Additional instructions:
- Focus on security
- Prioritize performance
- Enforce specific coding standards
"""
```

---

## Next Steps

1. ✅ Read [First PR Walkthrough](./03-first-pr-walkthrough.md)
2. ✅ Study [Architecture Overview](./04-architecture-overview.md)
3. ✅ Review [Best Practices](./19-best-practices.md)
4. ✅ Set up [Monitoring](./16-monitoring-observability.md)

---

**Setup Complete!** Your Claude AI-powered CI/CD is now fully configured and operational. 🎉

[← Back to Index](./README.md) | [Next: First PR Walkthrough →](./03-first-pr-walkthrough.md)
