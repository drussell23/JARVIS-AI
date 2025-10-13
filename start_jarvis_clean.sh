#!/bin/bash

# JARVIS Clean Start Script
# Ensures all processes are cleaned up before starting

echo "================================================"
echo "🚀 JARVIS Clean Start"
echo "================================================"

# Step 1: Perform cleanup
echo ""
echo "1️⃣ Cleaning up any stuck processes..."
cd backend && python test_cleanup.py --auto

# Step 2: Wait a moment for cleanup to complete
echo ""
echo "2️⃣ Waiting for cleanup to complete..."
sleep 2

# Step 3: Start JARVIS normally
echo ""
echo "3️⃣ Starting JARVIS..."
cd .. && ./jarvis.sh

echo ""
echo "================================================"
echo "✅ JARVIS startup sequence complete!"
echo "================================================"