#!/bin/bash

###############################################################################
# JARVIS GitHub Wiki Deployment Script
#
# This script automates the deployment of Wiki documentation to GitHub's
# Wiki repository (separate from the main codebase).
#
# Usage: ./deploy-wiki.sh
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
WIKI_REPO_URL="https://github.com/drussell23/JARVIS-AI.wiki.git"
WIKI_DIR="JARVIS-AI.wiki"
SOURCE_WIKI_DIR="wiki"

###############################################################################
# Helper Functions
###############################################################################

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

###############################################################################
# Pre-flight Checks
###############################################################################

print_header "JARVIS Wiki Deployment"

# Check if source wiki directory exists
if [ ! -d "$SOURCE_WIKI_DIR" ]; then
    print_error "Source wiki directory '$SOURCE_WIKI_DIR' not found!"
    exit 1
fi

# Check if git is installed
if ! command -v git &> /dev/null; then
    print_error "Git is not installed. Please install Git first."
    exit 1
fi

# Count wiki files
WIKI_FILE_COUNT=$(find "$SOURCE_WIKI_DIR" -name "*.md" | wc -l | xargs)
print_info "Found $WIKI_FILE_COUNT Wiki pages to deploy"

###############################################################################
# Step 1: Clone or Update Wiki Repository
###############################################################################

print_header "Step 1: Preparing Wiki Repository"

if [ -d "$WIKI_DIR" ]; then
    print_warning "Wiki directory already exists. Pulling latest changes..."
    cd "$WIKI_DIR"
    git pull origin master || {
        print_error "Failed to pull latest changes. Repository may be in a bad state."
        cd ..
        read -p "Delete and re-clone? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cd ..
            rm -rf "$WIKI_DIR"
            git clone "$WIKI_REPO_URL"
            cd "$WIKI_DIR"
        else
            exit 1
        fi
    }
    cd ..
else
    print_info "Cloning Wiki repository..."
    git clone "$WIKI_REPO_URL" || {
        print_error "Failed to clone Wiki repository. Check your permissions and URL."
        exit 1
    }
fi

print_success "Wiki repository ready"

###############################################################################
# Step 2: Copy Wiki Files
###############################################################################

print_header "Step 2: Copying Wiki Files"

# Remove old files (except .git)
print_info "Cleaning old Wiki files..."
find "$WIKI_DIR" -type f -name "*.md" -delete

# Copy new files
print_info "Copying new Wiki files..."
cp "$SOURCE_WIKI_DIR"/*.md "$WIKI_DIR/"

# Verify files were copied
COPIED_COUNT=$(find "$WIKI_DIR" -name "*.md" | wc -l | xargs)
if [ "$COPIED_COUNT" -eq "0" ]; then
    print_error "No files were copied! Aborting."
    exit 1
fi

print_success "Copied $COPIED_COUNT Wiki pages"

###############################################################################
# Step 3: Generate Sidebar
###############################################################################

print_header "Step 3: Generating Wiki Sidebar"

# GitHub Wiki uses _Sidebar.md for navigation
cat > "$WIKI_DIR/_Sidebar.md" << 'EOF'
# 📚 JARVIS Wiki

## 🏠 Getting Started
- [Home](Home)
- [Quick Reference](Quick-Reference)
- [Setup & Installation](Setup-&-Installation)

## 📖 Documentation
- [Architecture & Design](Architecture-&-Design)
- [API Documentation](API-Documentation)
- [Troubleshooting Guide](Troubleshooting-Guide)

## 🚀 Development
- [CI/CD Workflows](CI-CD-Workflows)
- [Contributing Guidelines](Contributing-Guidelines)
- [Edge Cases & Testing](Edge-Cases-&-Testing)

## 🗺️ Future
- [MAS Roadmap](MAS-Roadmap)

---

**Version:** v17.4.0
**Last Updated:** $(date +"%Y-%m-%d")
EOF

print_success "Sidebar generated"

###############################################################################
# Step 4: Generate Footer
###############################################################################

print_header "Step 4: Generating Wiki Footer"

# GitHub Wiki uses _Footer.md for page footers
cat > "$WIKI_DIR/_Footer.md" << 'EOF'
---

📚 **JARVIS AI Agent Wiki** | [Main Repository](https://github.com/drussell23/JARVIS-AI) | [Issues](https://github.com/drussell23/JARVIS-AI/issues) | [Discussions](https://github.com/drussell23/JARVIS-AI/discussions)

Last updated: $(date +"%Y-%m-%d %H:%M:%S")
EOF

print_success "Footer generated"

###############################################################################
# Step 5: Commit and Push Changes
###############################################################################

print_header "Step 5: Deploying to GitHub"

cd "$WIKI_DIR"

# Configure git (if needed)
git config user.name "$(git config user.name || echo 'JARVIS Wiki Bot')"
git config user.email "$(git config user.email || echo 'noreply@jarvis.ai')"

# Check for changes
if git diff --quiet && git diff --cached --quiet; then
    print_warning "No changes detected. Wiki is already up to date!"
    cd ..
    exit 0
fi

# Show what changed
print_info "Changes detected:"
git status --short

# Add all files
git add .

# Commit with detailed message
COMMIT_MSG="📚 Update Wiki Documentation

Deployed $COPIED_COUNT Wiki pages:
$(ls *.md | grep -v "_" | sed 's/\.md$//' | sed 's/^/- /' | head -20)

Timestamp: $(date +"%Y-%m-%d %H:%M:%S")
Deployment: Automated via deploy-wiki.sh"

git commit -m "$COMMIT_MSG"

# Push to GitHub
print_info "Pushing to GitHub..."
git push origin master || {
    print_error "Failed to push to GitHub. Check your credentials and permissions."
    cd ..
    exit 1
}

cd ..

print_success "Wiki deployed successfully!"

###############################################################################
# Summary
###############################################################################

print_header "Deployment Complete"

echo ""
print_success "Your Wiki is now live at:"
echo -e "${GREEN}https://github.com/drussell23/JARVIS-AI/wiki${NC}"
echo ""

print_info "Deployed Pages:"
ls "$WIKI_DIR"/*.md | grep -v "_" | xargs -n 1 basename | sed 's/\.md$//' | sed 's/^/  ✓ /'

echo ""
print_info "Summary:"
echo "  📄 Total Pages: $COPIED_COUNT"
echo "  🔗 Sidebar: Generated"
echo "  📋 Footer: Generated"
echo "  🚀 Status: Live on GitHub"

echo ""
print_success "Done! Visit your Wiki to see the changes."

exit 0
