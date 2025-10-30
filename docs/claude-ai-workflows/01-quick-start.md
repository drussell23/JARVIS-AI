# 🚀 Quick Start Guide - Claude AI Workflows

**Get your AI-powered CI/CD up and running in 5 minutes**

---

## ⏱️ Time to Complete: 5 minutes

## 📋 Prerequisites

- GitHub repository with admin access
- Anthropic account (free to create)
- Basic understanding of GitHub Actions

---

## Step 1: Get Your Anthropic API Key (2 minutes)

### 1.1 Create Anthropic Account
1. Go to https://console.anthropic.com/
2. Sign up with email or Google
3. Verify your email
4. Complete registration

### 1.2 Generate API Key
1. Navigate to https://console.anthropic.com/settings/keys
2. Click **"Create Key"**
3. Name it: `github-actions-jarvis`
4. Copy the key (starts with `sk-ant-api03-...`)

⚠️  **IMPORTANT:** Save this key securely. You won't be able to see it again!

```
Example API Key Format:
sk-ant-api03-[RANDOM_STRING]-[RANDOM_STRING]-[SHORT_STRING]
```

---

## Step 2: Add API Key to GitHub (1 minute)

### Option A: Via Web Interface (Recommended)

1. Go to your repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"**
4. Enter details:
   - **Name:** `ANTHROPIC_API_KEY`
   - **Secret:** Paste your API key
5. Click **"Add secret"**

### Option B: Via GitHub CLI

```bash
gh secret set ANTHROPIC_API_KEY --body 'your-api-key-here'
```

### Option C: Via Terminal (macOS/Linux)

```bash
# Navigate to your repo
cd /path/to/JARVIS-AI-Agent

# Set secret
echo "your-api-key-here" | gh secret set ANTHROPIC_API_KEY
```

### Verify Secret Was Added

```bash
gh secret list | grep ANTHROPIC_API_KEY
```

Expected output:
```
ANTHROPIC_API_KEY    2025-10-30T06:34:32Z
```

---

## Step 3: Verify Workflows Are Active (30 seconds)

### 3.1 Check Workflows Directory

```bash
ls .github/workflows/claude-*.yml
```

Expected output:
```
.github/workflows/claude-auto-fix.yml
.github/workflows/claude-docs-generator.yml
.github/workflows/claude-pr-analyzer.yml
.github/workflows/claude-security-analyzer.yml
.github/workflows/claude-test-generator.yml
```

### 3.2 Verify on GitHub

1. Go to your repository
2. Click **Actions** tab
3. You should see workflows listed:
   - Claude AI PR Analyzer
   - Claude AI Auto-Fix
   - Claude AI Test Generator
   - Claude AI Security Analyzer
   - Claude AI Documentation Generator

---

## Step 4: Test the System (1 minute)

### Create a Test PR

```bash
# Create test branch
git checkout -b test/claude-ai-demo

# Create simple test file
cat > demo_test.py << 'EOF'
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total

def filter_positives(items):
    result = []
    for item in items:
        if item > 0:
            result.append(item)
    return result
EOF

# Commit and push
git add demo_test.py
git commit -m "test: Demo Claude AI workflows"
git push -u origin test/claude-ai-demo

# Create PR
gh pr create --title "test: Claude AI Demo" \
  --body "Testing the Claude AI workflows" \
  --base main
```

### What to Expect

Within 2-5 minutes, Claude AI will:

1. ✅ **Analyze** your code
2. ✅ **Post** detailed review comment
3. ✅ **Generate** tests
4. ✅ **Add** documentation
5. ✅ **Apply** intelligent labels
6. ✅ **Commit** improvements

---

## Step 5: Watch the Magic Happen (2-5 minutes)

### Monitor Progress

1. Go to your PR page
2. Watch the **Checks** tab
3. See Claude AI workflows running
4. Review AI-generated comments

### Expected Results

#### 📝 PR Comment from Claude AI
```markdown
## 🤖 Claude AI Code Review

### Overall Assessment: 7/10

Your code is functional but could benefit from:
- Type hints for better code clarity
- Docstrings for documentation
- List comprehensions for better performance
- Input validation

### Suggested Improvements:
[Detailed suggestions with code examples]
```

#### 🏷️ Auto-Applied Labels
- `needs-work` or `ready-to-merge`
- `documentation`
- `needs-tests`
- `performance`

#### 🔧 New Commits from `claude-ai[bot]`
- Tests generated in `tests/` directory
- Docstrings added
- Code improvements applied

---

## ✅ Success Checklist

- [x] Anthropic API key created
- [x] API key added to GitHub Secrets
- [x] Workflows visible in Actions tab
- [x] Test PR created
- [x] Claude AI posted review
- [x] AI-generated commits visible

---

## 🎉 You're Done!

Your Claude AI-powered CI/CD is now active! Every PR will automatically:
- Get comprehensive code reviews
- Receive auto-fixes
- Have tests generated
- Get security scans
- Be intelligently labeled

---

## 🎯 Next Steps

### Immediate
1. Review Claude's analysis of your test PR
2. Merge or close the test PR
3. Try mentioning `@claude` in PR comments
4. Explore other workflows

### This Week
1. Read [Detailed Setup Guide](./02-detailed-setup.md)
2. Review [Workflow Reference](./05-workflow-reference.md)
3. Study [Best Practices](./19-best-practices.md)

### This Month
1. Optimize costs with [Cost Optimization](./13-cost-optimization.md)
2. Customize with [Customization Guide](./12-customization-guide.md)
3. Set up monitoring with [Monitoring Guide](./16-monitoring-observability.md)

---

## 💬 Interactive Commands

Try these in your PR comments:

```bash
@claude                  # Full PR analysis
@claude fix             # Auto-fix code issues
@claude generate tests  # Create tests
@claude generate docs   # Update documentation
@claude security scan   # Run security analysis
```

---

## 🐛 Troubleshooting

### "Workflow not running"
**Cause:** API key not set or incorrect
**Fix:** Verify secret name is exactly `ANTHROPIC_API_KEY`

```bash
gh secret list | grep ANTHROPIC_API_KEY
```

### "API key invalid"
**Cause:** Incorrect or expired API key
**Fix:** Generate new key and update secret

```bash
# Delete old secret
gh secret delete ANTHROPIC_API_KEY

# Add new one
gh secret set ANTHROPIC_API_KEY --body 'new-key-here'
```

### "Workflows skipped"
**Cause:** Normal behavior - some workflows are conditional
**Check:** Look for AI workflows specifically

```bash
gh pr view <PR-NUMBER> --json statusCheckRollup \
  --jq '.statusCheckRollup[] | select(.name | contains("AI"))'
```

### "No AI comments"
**Cause:** Workflow still running or failed
**Fix:** Check Actions tab for errors

```bash
gh run list --workflow="claude-pr-analyzer.yml" --limit 1
```

---

## 📊 Cost Estimate

Your first month estimate:
- **Test PRs:** ~$1
- **Real PRs (5-10):** ~$10-20
- **Daily security scans:** ~$5
- **Total:** ~$15-25/month

**ROI:** Saves 10-20 hours = $500-2000 value

---

## 🎓 Quick Reference

### Essential Commands

```bash
# Check secret exists
gh secret list

# View workflow runs
gh run list

# View specific workflow
gh run view <run-id> --log

# Check PR status
gh pr view <number> --json statusCheckRollup

# Trigger workflow manually
gh workflow run claude-pr-analyzer.yml
```

### Important Links

- **Anthropic Console:** https://console.anthropic.com/
- **GitHub Actions:** https://github.com/YOUR-REPO/actions
- **Documentation:** [Main Index](./README.md)
- **Support:** [Troubleshooting](./17-troubleshooting.md)

---

## 🌟 What's Next?

You now have an AI-powered development assistant that:
- Reviews every PR like a senior developer
- Catches bugs before they reach production
- Generates comprehensive tests
- Keeps documentation up to date
- Scans for security vulnerabilities daily

**Welcome to the future of CI/CD!** 🚀

---

## 📚 Related Guides

- [Detailed Setup Guide](./02-detailed-setup.md) - Full installation walkthrough
- [First PR Walkthrough](./03-first-pr-walkthrough.md) - Detailed PR creation guide
- [Architecture Overview](./04-architecture-overview.md) - How it all works
- [Best Practices](./19-best-practices.md) - Tips for optimal usage

---

**Need Help?** Check the [FAQ](./18-faq.md) or [Troubleshooting Guide](./17-troubleshooting.md)

**Ready for More?** Continue to [Detailed Setup Guide](./02-detailed-setup.md) →
