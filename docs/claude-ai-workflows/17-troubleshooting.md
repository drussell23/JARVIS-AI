# 🔧 Troubleshooting Guide - Claude AI Workflows

Complete troubleshooting reference for common issues and solutions.

---

## Quick Diagnostics

```bash
# Run full diagnostic
./scripts/diagnose-claude-workflows.sh

# Or manual checks:
gh secret list | grep ANTHROPIC
gh workflow list
gh run list --limit 5
```

---

## Common Issues

### 1. "API Key Invalid" Error

**Symptoms:**
- Workflow fails with 401 error
- "Invalid API key" in logs

**Causes:**
- Wrong API key
- Expired key
- Secret name mismatch

**Solutions:**
```bash
# Verify secret exists
gh secret list | grep ANTHROPIC_API_KEY

# Regenerate key at: https://console.anthropic.com/settings/keys

# Update secret
gh secret set ANTHROPIC_API_KEY --body 'new-key-here'

# Test
gh workflow run claude-pr-analyzer.yml
```

---

### 2. Workflows Not Running

**Symptoms:**
- No workflows appear on PR
- "Skipped" status

**Causes:**
- Workflows disabled
- Incorrect triggers
- Permission issues

**Solutions:**
```bash
# Check if Actions enabled
gh api /repos/OWNER/REPO/actions/permissions

# Enable
gh api --method PUT /repos/OWNER/REPO/actions/permissions -f enabled=true

# Check workflow file syntax
yamllint .github/workflows/claude-*.yml
```

---

### 3. "Permission Denied" Errors

**Symptoms:**
- Cannot create commits
- Cannot post comments

**Fix:**
```
Settings → Actions → General → Workflow permissions
☑ Read and write permissions
☑ Allow GitHub Actions to create and approve pull requests
```

---

### 4. High Costs

**Symptoms:**
- Unexpected Anthropic bills
- High token usage

**Solutions:**
```bash
# Check usage
open https://console.anthropic.com/settings/usage

# Reduce costs:
# 1. Limit file analysis
# 2. Use smaller model (haiku)
# 3. Reduce max_tokens
# 4. Skip docs generation on small PRs
```

---

### 5. Slow Performance

**Causes:**
- Large PRs (100+ files)
- Complex analysis

**Solutions:**
```yaml
# In workflows, add:
if: |
  github.event.pull_request.changed_files < 50

# Or increase timeout:
timeout-minutes: 30  # from 15
```

---

## Error Messages Reference

### "Rate limit exceeded"
- **Cause:** Too many API calls
- **Fix:** Wait 1 minute, retry

### "Timeout"
- **Cause:** Large PR or slow API
- **Fix:** Increase `timeout-minutes` in workflow

### "No module named 'anthropic'"
- **Cause:** Missing dependency
- **Fix:** Workflow should auto-install, check logs

---

## Support Channels

1. Check [FAQ](./18-faq.md)
2. Search [GitHub Issues](https://github.com/drussell23/JARVIS-AI-Agent/issues)
3. Create new issue with logs
4. Contact on Discussions

---

[← Back to Index](./README.md)
