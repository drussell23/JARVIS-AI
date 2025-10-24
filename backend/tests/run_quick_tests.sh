#!/bin/bash
# Quick test runner without coverage for fast feedback

cd "$(dirname "$0")/.."

echo "🧪 Running Quick Tests (no coverage)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Run tests without coverage, with minimal plugins
pytest tests/test_hypothesis_examples.py \
  -v \
  --tb=short \
  --disable-warnings \
  -p no:cov \
  --timeout=10 \
  "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ All tests passed!"
else
    echo ""
    echo "❌ Some tests failed (exit code: $EXIT_CODE)"
fi

exit $EXIT_CODE
