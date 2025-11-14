#!/bin/bash
# Quick script to export voice unlock metrics to Excel

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/export_metrics_to_excel.py"

# Make Python script executable
chmod +x "$PYTHON_SCRIPT"

echo "📊 JARVIS Voice Unlock Metrics → Excel Exporter"
echo "================================================"
echo ""

# Parse arguments
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage:"
    echo "  ./export_to_excel.sh              # Export today's metrics"
    echo "  ./export_to_excel.sh --all        # Export all available metrics"
    echo "  ./export_to_excel.sh --date 2025-11-13  # Export specific date"
    echo "  ./export_to_excel.sh --open       # Export and open in Excel"
    echo ""
    exit 0
fi

# Run the Python script with all arguments
python3 "$PYTHON_SCRIPT" "$@"

# If --open flag was passed, the script already opened it
# Otherwise, show the output location
if [[ ! " $@ " =~ " --open " ]]; then
    echo ""
    echo "📁 Excel files saved to: ~/.jarvis/logs/unlock_metrics/excel_exports/"
    echo ""
    echo "To open the latest file:"
    echo "  open ~/.jarvis/logs/unlock_metrics/excel_exports/unlock_metrics_*.xlsx"
fi
