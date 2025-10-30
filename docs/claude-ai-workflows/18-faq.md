# ❓ Frequently Asked Questions - Claude AI Workflows

---

## General Questions

### What is Claude AI?
Claude is Anthropic's AI assistant. We use Claude Sonnet 4 for code analysis, review, and generation.

### How much does it cost?
- **Average:** $10-30/month for active repo
- **Per PR:** ~$0.20-0.50
- **Daily scans:** ~$0.10
- **Free tier:** $5 credit to start

### Is my code sent to Anthropic?
Yes, changed files in PRs are sent to Claude for analysis. Anthropic doesn't train on your data. See [Security & Privacy](./21-security-privacy.md).

### Can I use a different AI model?
Yes! Edit workflows to use:
- `claude-opus-4` (more powerful, $$$)
- `claude-haiku-4` (faster, cheaper)
- Custom models (advanced)

---

## Setup Questions

### Do I need to install anything locally?
No! Everything runs in GitHub Actions. Just add the API key.

### Can I use this on private repositories?
Yes! Works on both public and private repos.

### What if I don't have admin access?
You need admin access to add secrets. Ask your repo admin.

### How do I get an API key?
1. Sign up at https://console.anthropic.com/
2. Go to Settings → API Keys
3. Create new key
4. Add to GitHub Secrets

---

## Usage Questions

### How do I trigger Claude manually?
Mention `@claude` in PR comments:
```
@claude                  # Full analysis
@claude fix             # Auto-fix
@claude generate tests  # Create tests
```

### Why are some workflows skipped?
Smart skipping saves time and money:
- Dependabot workflows skip on human PRs
- Rollback only runs if merge fails
- Conditional logic based on PR content

### Can Claude automatically merge PRs?
Not by default. You can enable auto-merge for safe updates (patch versions only).

### Will Claude commit to my PR?
Yes! Claude can:
- Fix code issues
- Generate tests
- Add documentation
All committed as `claude-ai[bot]`

---

## Performance Questions

### How long does analysis take?
- Small PRs (< 10 files): 1-2 minutes
- Medium PRs (10-50 files): 2-5 minutes
- Large PRs (50+ files): 5-10 minutes

### Can I make it faster?
Yes:
```yaml
# Use faster model
model: "claude-haiku-4"

# Reduce tokens
max_tokens: 4000

# Skip files
if: github.event.pull_request.changed_files < 20
```

### Why is it slow?
- Large PRs (100+ files)
- API rate limits
- Complex analysis
- Concurrent workflows

---

## Cost Questions

### How can I reduce costs?
1. Use haiku model (cheaper)
2. Skip docs on small PRs
3. Reduce max_tokens
4. Limit file analysis
5. See [Cost Optimization](./13-cost-optimization.md)

### What if I exceed my budget?
Set up Anthropic spending limits:
```
Console → Settings → Billing → Set spend limit
```

### Is there a free tier?
Yes! $5 free credit. Good for ~20-30 PRs.

---

## Security Questions

### Is it safe to use?
Yes! See our [Security & Privacy](./21-security-privacy.md) guide.

### Can Claude see my secrets?
No! We filter out:
- API keys
- Passwords
- Credentials
- Environment variables

### What data does Anthropic keep?
Anthropic doesn't train on your data or store it long-term. See their [privacy policy](https://www.anthropic.com/legal/privacy).

---

## Troubleshooting

### Workflows not running?
```bash
# Check:
gh secret list | grep ANTHROPIC
gh workflow list
gh api /repos/OWNER/REPO/actions/permissions
```

### API key invalid?
Regenerate and update:
```bash
gh secret set ANTHROPIC_API_KEY
```

### Need more help?
See [Troubleshooting Guide](./17-troubleshooting.md)

---

[← Back to Index](./README.md)
