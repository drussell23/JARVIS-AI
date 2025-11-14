# 📊 Excel & Database Export Guide

Complete guide for exporting voice unlock metrics to **Excel** and **SQLite/CloudSQL Database**.

---

## 🚀 Quick Start

### Export to Excel (Easiest!)

```bash
# Export today's metrics to Excel
cd ~/Documents/repos/JARVIS-AI-Agent/backend/voice_unlock
./export_to_excel.sh

# Export and automatically open in Excel
./export_to_excel.sh --open

# Export all historical data
./export_to_excel.sh --all

# Export specific date
./export_to_excel.sh --date 2025-11-13
```

Excel files will be saved to:
```
~/.jarvis/logs/unlock_metrics/excel_exports/
```

### View Database (SQLite)

```bash
# Install SQLite browser (if you want GUI)
brew install --cask db-browser-for-sqlite

# Open database in GUI
open -a "DB Browser for SQLite" ~/.jarvis/logs/unlock_metrics/unlock_metrics.db

# Or query from command line
sqlite3 ~/.jarvis/logs/unlock_metrics/unlock_metrics.db "SELECT * FROM unlock_attempts ORDER BY timestamp DESC LIMIT 10"
```

---

## 📁 What Gets Exported

### Excel Workbook Structure

When you export, you get a **multi-sheet Excel workbook** with:

#### **Sheet 1: Unlock Attempts**
Complete list of every unlock attempt with:
- Timestamp, Date, Time, Day of Week
- Success/Failure status ✅❌
- Speaker name
- Transcribed text
- Confidence scores (speaker, STT)
- Threshold comparisons
- Duration (ms and seconds)
- Quality indicators
- Trend analysis
- All stage pass/fail status

#### **Sheet 2: Stage Performance**
Detailed breakdown of each processing stage:
- Audio Preparation timing
- Transcription timing & algorithm
- Intent Verification
- Speaker Identification
- Owner Verification
- Biometric Verification (with confidence vs threshold)
- Context Analysis (CAI)
- Scenario Analysis (SAI)
- Unlock Execution

#### **Sheet 3: Biometrics Detail**
Deep dive into biometric metrics:
- Speaker confidence over time
- STT confidence trends
- Confidence margins
- Historical comparisons (last 10, last 30)
- Trend direction (improving/declining/stable)
- Volatility analysis
- Best/worst scores
- Percentile rankings

#### **Sheet 4: Overall Statistics**
Aggregated stats across all attempts:
- Total attempts
- Success rate
- Per-speaker statistics
- Average confidence
- Average duration

#### **Sheet 5: Confidence Trends**
Historical confidence tracking:
- Attempt-by-attempt confidence scores
- Success/failure patterns
- Timestamp tracking
- Visual trend data

---

## 📊 Excel Features

### Professional Formatting
- **Blue headers** with white text
- **Auto-sized columns** for readability
- **Frozen header row** for scrolling
- **Borders** on all cells
- **Emoji indicators** (✅❌) for quick visual scanning

### Excel Analysis Tips

```excel
# In Excel, you can:
1. Sort by confidence score (see your best/worst attempts)
2. Filter by success/failure
3. Create pivot tables for trend analysis
4. Chart confidence over time
5. Calculate averages by day of week
6. Identify which stages take longest
```

---

## 🗄️ Database Structure

### SQLite Tables

#### **unlock_attempts** (Main table)
```sql
-- All unlock attempt data
id, timestamp, date, time, success, speaker_name,
transcribed_text, speaker_confidence, stt_confidence,
threshold, total_duration_ms, trend_direction, etc.
```

#### **processing_stages** (Detailed stages)
```sql
-- Each stage from each attempt
id, attempt_id, stage_name, duration_ms, success,
algorithm_used, confidence_score, threshold, etc.
```

#### **stage_breakdown** (Quick performance queries)
```sql
-- Stage timing summary
id, attempt_id, stage_name, duration_ms, percentage
```

### Useful SQL Queries

```bash
# Connect to database
sqlite3 ~/.jarvis/logs/unlock_metrics/unlock_metrics.db
```

```sql
-- Get today's success rate
SELECT
    COUNT(*) as total,
    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
    ROUND(100.0 * SUM(success) / COUNT(*), 2) as success_rate
FROM unlock_attempts
WHERE date = date('now');

-- Average confidence by day of week
SELECT
    day_of_week,
    AVG(speaker_confidence) as avg_confidence,
    COUNT(*) as attempts
FROM unlock_attempts
GROUP BY day_of_week
ORDER BY
    CASE day_of_week
        WHEN 'Monday' THEN 1
        WHEN 'Tuesday' THEN 2
        WHEN 'Wednesday' THEN 3
        WHEN 'Thursday' THEN 4
        WHEN 'Friday' THEN 5
        WHEN 'Saturday' THEN 6
        WHEN 'Sunday' THEN 7
    END;

-- Find slowest unlocks
SELECT
    timestamp,
    speaker_name,
    total_duration_ms / 1000.0 as duration_sec,
    slowest_stage
FROM unlock_attempts
ORDER BY total_duration_ms DESC
LIMIT 10;

-- Stage performance analysis
SELECT
    stage_name,
    AVG(duration_ms) as avg_duration_ms,
    MIN(duration_ms) as min_duration_ms,
    MAX(duration_ms) as max_duration_ms,
    COUNT(*) as times_executed
FROM processing_stages
GROUP BY stage_name
ORDER BY avg_duration_ms DESC;

-- Confidence trends (last 30 attempts)
SELECT
    date,
    time,
    speaker_confidence,
    success,
    trend_direction
FROM unlock_attempts
ORDER BY timestamp DESC
LIMIT 30;

-- Failed attempts analysis
SELECT
    timestamp,
    transcribed_text,
    speaker_confidence,
    threshold,
    error
FROM unlock_attempts
WHERE success = 0
ORDER BY timestamp DESC;
```

---

## 🔄 Automatic Export Setup

### Daily Auto-Export to Excel

Create a cron job to export daily:

```bash
# Edit crontab
crontab -e

# Add this line (exports at 11:59 PM daily)
59 23 * * * cd ~/Documents/repos/JARVIS-AI-Agent/backend/voice_unlock && ./export_to_excel.sh --date $(date +\%Y-\%m-\%d) >> ~/jarvis_export.log 2>&1
```

### Weekly Comprehensive Export

```bash
# Export all data every Sunday at midnight
0 0 * * 0 cd ~/Documents/repos/JARVIS-AI-Agent/backend/voice_unlock && ./export_to_excel.sh --all >> ~/jarvis_export.log 2>&1
```

---

## 💾 Data Storage Locations

### JSON Files (Raw Data)
```
~/.jarvis/logs/unlock_metrics/
├── unlock_metrics_2025-11-13.json  (Daily logs)
├── unlock_stats.json                (Aggregated stats)
└── confidence_trends.json           (Historical trends)
```

### Excel Files (Exported)
```
~/.jarvis/logs/unlock_metrics/excel_exports/
├── unlock_metrics_2025-11-13.xlsx  (Single day)
└── unlock_metrics_ALL_20251113_210530.xlsx  (All data)
```

### SQLite Database
```
~/.jarvis/logs/unlock_metrics/
└── unlock_metrics.db  (Local database)
```

### CloudSQL (Synced automatically)
```
Database: jarvis_learning
Tables:
- voice_unlock_attempts
- voice_unlock_stages
```

---

## 🔧 Installation

### Install Required Packages

```bash
# Python packages for Excel export
pip install pandas openpyxl

# These are already included in JARVIS requirements.txt
# But install manually if needed:
pip install aiofiles asyncio
```

### Optional: SQLite Browser

```bash
# GUI tool for browsing SQLite databases
brew install --cask db-browser-for-sqlite
```

---

## 📝 Usage Examples

### Example 1: Daily Review

```bash
# Every morning, export yesterday's data
./export_to_excel.sh --date $(date -v-1d +%Y-%m-%d) --open

# Opens in Excel automatically for review
```

### Example 2: Weekly Analysis

```bash
# Export all data every week
./export_to_excel.sh --all

# Then open and create pivot tables in Excel:
# - Success rate by day of week
# - Average confidence trends
# - Stage performance comparison
```

### Example 3: Troubleshooting Failed Attempts

```bash
# Query database for failures
sqlite3 ~/.jarvis/logs/unlock_metrics/unlock_metrics.db <<EOF
SELECT
    timestamp,
    transcribed_text,
    speaker_confidence,
    threshold,
    confidence_margin,
    slowest_stage,
    error
FROM unlock_attempts
WHERE success = 0
ORDER BY timestamp DESC
LIMIT 10;
EOF
```

### Example 4: Python Analysis

```python
#!/usr/bin/env python3
import pandas as pd

# Load Excel file
excel_file = "~/.jarvis/logs/unlock_metrics/excel_exports/unlock_metrics_2025-11-13.xlsx"
df = pd.read_excel(excel_file, sheet_name='Unlock Attempts')

# Calculate statistics
print(f"Total attempts: {len(df)}")
print(f"Success rate: {df['Success'].str.contains('✅').sum() / len(df) * 100:.1f}%")
print(f"Average confidence: {df['Confidence Score'].mean():.3f}")
print(f"Average duration: {df['Total Duration (sec)'].mean():.1f}s")

# Plot confidence over time
import matplotlib.pyplot as plt
df['Confidence Score'].plot(kind='line', title='Confidence Over Time')
plt.axhline(y=0.35, color='r', linestyle='--', label='Threshold')
plt.ylabel('Confidence Score')
plt.xlabel('Attempt Number')
plt.legend()
plt.savefig('confidence_trend.png')
```

---

## 🎯 Common Tasks

### How to check today's success rate?

```bash
# Option 1: Excel
./export_to_excel.sh --open
# Look at "Overall Statistics" sheet

# Option 2: Database
sqlite3 ~/.jarvis/logs/unlock_metrics/unlock_metrics.db \
  "SELECT ROUND(100.0 * SUM(success) / COUNT(*), 1) || '%' as success_rate
   FROM unlock_attempts WHERE date = date('now')"
```

### How to see confidence trends?

```bash
# Option 1: Excel
./export_to_excel.sh --open
# Go to "Confidence Trends" sheet → Create a line chart

# Option 2: View JSON
cat ~/.jarvis/logs/unlock_metrics/confidence_trends.json | jq '."Derek J. Russell".confidence_history[-20:]'
```

### How to find what's slowing down unlocks?

```bash
# Export to Excel and sort "Stage Performance" sheet by "Duration (ms)"
./export_to_excel.sh --open

# Or query database
sqlite3 ~/.jarvis/logs/unlock_metrics/unlock_metrics.db \
  "SELECT stage_name, AVG(duration_ms) as avg_ms
   FROM processing_stages
   GROUP BY stage_name
   ORDER BY avg_ms DESC"
```

---

## 📊 Excel Tips & Tricks

### Create a Dashboard

1. Open your Excel file
2. Insert → PivotTable
3. Recommended dashboards:
   - Success rate by day of week
   - Average confidence by hour
   - Stage performance comparison
   - Trend line (confidence over time)

### Conditional Formatting

1. Select "Confidence Score" column
2. Home → Conditional Formatting → Color Scales
3. Green for high, red for low

### Charts

1. Select time series data
2. Insert → Line Chart
3. Add threshold line (0.35) as reference

---

## 🔍 Troubleshooting

### Excel file won't open

```bash
# Check if file exists
ls -lh ~/.jarvis/logs/unlock_metrics/excel_exports/

# Try exporting again
./export_to_excel.sh
```

### Database is empty

```bash
# Check if database exists
ls -lh ~/.jarvis/logs/unlock_metrics/unlock_metrics.db

# Database is populated automatically on first unlock attempt
# Try unlocking your screen once to generate data
```

### Import errors

```bash
# Install missing packages
pip install pandas openpyxl aiofiles

# Verify installation
python3 -c "import pandas, openpyxl; print('✅ Packages installed')"
```

---

## 📚 Additional Resources

- [Main Metrics Logger README](METRICS_LOGGER_README.md)
- [Voice Unlock README](README.md)
- [Python Pandas Documentation](https://pandas.pydata.org/docs/)
- [SQLite Documentation](https://www.sqlite.org/docs.html)

---

**Questions?** The Excel export script is self-contained and safe to run anytime. Your data is always backed up in JSON, SQLite, and CloudSQL!
