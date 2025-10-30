#!/usr/bin/env python3
"""
Generate WebSocket health status badge

Creates a dynamic badge showing WebSocket health status based on test results.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path


class HealthStatus:
    """WebSocket health status levels"""
    EXCELLENT = {"color": "brightgreen", "emoji": "🟢", "label": "Excellent"}
    GOOD = {"color": "green", "emoji": "🟢", "label": "Good"}
    FAIR = {"color": "yellow", "emoji": "🟡", "label": "Fair"}
    DEGRADED = {"color": "orange", "emoji": "🟠", "label": "Degraded"}
    CRITICAL = {"color": "red", "emoji": "🔴", "label": "Critical"}
    UNKNOWN = {"color": "lightgrey", "emoji": "⚪", "label": "Unknown"}


def calculate_health_status(metrics: dict) -> dict:
    """
    Calculate overall health status from metrics

    Args:
        metrics: Dictionary containing test metrics

    Returns:
        Health status dict with color, emoji, and label
    """
    success_rate = metrics.get("success_rate", 0.0)
    avg_latency = metrics.get("avg_latency_ms", 0.0)
    reconnections = metrics.get("reconnections", 0)
    errors = metrics.get("errors", 0)

    # Calculate health score
    score = 100.0

    # Deduct for failures
    score -= (1.0 - success_rate) * 50

    # Deduct for high latency
    if avg_latency > 1000:
        score -= 20
    elif avg_latency > 500:
        score -= 10
    elif avg_latency > 200:
        score -= 5

    # Deduct for reconnections
    score -= min(reconnections * 5, 20)

    # Deduct for errors
    score -= min(errors * 2, 10)

    # Determine status
    if score >= 95:
        return HealthStatus.EXCELLENT
    elif score >= 85:
        return HealthStatus.GOOD
    elif score >= 70:
        return HealthStatus.FAIR
    elif score >= 50:
        return HealthStatus.DEGRADED
    else:
        return HealthStatus.CRITICAL


def generate_badge_svg(status: dict, metrics: dict) -> str:
    """
    Generate SVG badge

    Args:
        status: Health status dict
        metrics: Test metrics

    Returns:
        SVG content as string
    """
    label = "WebSocket Health"
    message = status["label"]
    color = status["color"]
    emoji = status["emoji"]

    # Calculate dimensions
    label_width = len(label) * 7 + 10
    message_width = len(message) * 7 + 10
    total_width = label_width + message_width

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20">
  <linearGradient id="b" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <mask id="a">
    <rect width="{total_width}" height="20" rx="3" fill="#fff"/>
  </mask>
  <g mask="url(#a)">
    <path fill="#555" d="M0 0h{label_width}v20H0z"/>
    <path fill="#{color}" d="M{label_width} 0h{message_width}v20H{label_width}z"/>
    <path fill="url(#b)" d="M0 0h{total_width}v20H0z"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="{label_width/2}" y="15" fill="#010101" fill-opacity=".3">{label}</text>
    <text x="{label_width/2}" y="14">{label}</text>
    <text x="{label_width + message_width/2}" y="15" fill="#010101" fill-opacity=".3">{emoji} {message}</text>
    <text x="{label_width + message_width/2}" y="14">{emoji} {message}</text>
  </g>
</svg>"""

    return svg


def generate_markdown_badge(status: dict, metrics: dict) -> str:
    """
    Generate markdown badge

    Args:
        status: Health status dict
        metrics: Test metrics

    Returns:
        Markdown badge string
    """
    emoji = status["emoji"]
    label = status["label"]
    success_rate = metrics.get("success_rate", 0.0) * 100
    avg_latency = metrics.get("avg_latency_ms", 0.0)

    return f"""## {emoji} WebSocket Health: {label}

**Status:** {label}
**Success Rate:** {success_rate:.1f}%
**Avg Latency:** {avg_latency:.1f}ms
**Last Updated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}

---
"""


def generate_json_status(status: dict, metrics: dict) -> str:
    """
    Generate JSON status report

    Args:
        status: Health status dict
        metrics: Test metrics

    Returns:
        JSON string
    """
    data = {
        "status": status["label"],
        "emoji": status["emoji"],
        "color": status["color"],
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }
    return json.dumps(data, indent=2)


def main():
    """Main function"""
    # Load metrics from environment or file
    metrics_file = os.getenv("METRICS_FILE", "websocket_metrics.json")

    if Path(metrics_file).exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
    else:
        # Use sample metrics for demonstration
        metrics = {
            "success_rate": float(os.getenv("SUCCESS_RATE", "0.95")),
            "avg_latency_ms": float(os.getenv("AVG_LATENCY_MS", "45.0")),
            "reconnections": int(os.getenv("RECONNECTIONS", "0")),
            "errors": int(os.getenv("ERRORS", "0")),
            "tests_passed": int(os.getenv("TESTS_PASSED", "6")),
            "tests_failed": int(os.getenv("TESTS_FAILED", "0")),
        }

    # Calculate status
    status = calculate_health_status(metrics)

    # Generate outputs
    output_dir = Path(os.getenv("OUTPUT_DIR", "."))
    output_dir.mkdir(exist_ok=True)

    # SVG Badge
    svg_content = generate_badge_svg(status, metrics)
    svg_path = output_dir / "websocket_health_badge.svg"
    with open(svg_path, "w") as f:
        f.write(svg_content)
    print(f"✅ Generated SVG badge: {svg_path}")

    # Markdown Badge
    md_content = generate_markdown_badge(status, metrics)
    md_path = output_dir / "websocket_health_status.md"
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"✅ Generated Markdown badge: {md_path}")

    # JSON Status
    json_content = generate_json_status(status, metrics)
    json_path = output_dir / "websocket_health_status.json"
    with open(json_path, "w") as f:
        f.write(json_content)
    print(f"✅ Generated JSON status: {json_path}")

    # Output to GitHub Actions
    if os.getenv("GITHUB_OUTPUT"):
        with open(os.getenv("GITHUB_OUTPUT"), "a") as f:
            f.write(f"health_status={status['label']}\n")
            f.write(f"health_emoji={status['emoji']}\n")
            f.write(f"health_color={status['color']}\n")

    # Print summary
    print(f"\n{status['emoji']} WebSocket Health: {status['label']}")
    print(f"Success Rate: {metrics['success_rate']*100:.1f}%")
    print(f"Avg Latency: {metrics['avg_latency_ms']:.1f}ms")

    # Exit with appropriate code
    if status in [HealthStatus.CRITICAL, HealthStatus.DEGRADED]:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
