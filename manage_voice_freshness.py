#!/usr/bin/env python3
"""
Voice Sample Freshness Manager for JARVIS
Automatically manages sample aging, archival, and refresh recommendations
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from intelligence.learning_database import get_learning_database


async def main():
    print("\n" + "="*80)
    print("🎤 JARVIS VOICE SAMPLE FRESHNESS MANAGER")
    print("="*80)

    # Initialize database
    print("\n📊 Connecting to database...")
    db = await get_learning_database()

    speaker_name = "Derek J. Russell"

    # Get freshness report
    print(f"\n📋 Generating freshness report for {speaker_name}...")
    report = await db.get_sample_freshness_report(speaker_name)

    if 'error' in report:
        print(f"❌ Error: {report['error']}")
        return

    # Display age distribution
    print(f"\n📅 SAMPLE AGE DISTRIBUTION")
    print("-" * 80)

    total_samples = 0
    for age_bracket, data in report['age_distribution'].items():
        count = data['count']
        avg_conf = data['avg_confidence']
        avg_qual = data['avg_quality']
        total_samples += count

        # Visual bar
        bar_length = int(count / 2)  # Scale for display
        bar = "█" * bar_length

        print(f"{age_bracket:15} | {bar:20} {count:3} samples")
        print(f"{'':15} | Confidence: {avg_conf:.1%}, Quality: {avg_qual:.1%}")
        print()

    print(f"{'TOTAL':15} | {total_samples} samples")

    # Display recommendations
    if report['recommendations']:
        print(f"\n💡 RECOMMENDATIONS")
        print("-" * 80)
        for rec in report['recommendations']:
            priority_icon = "🔴" if rec['priority'] == 'HIGH' else "🟡"
            print(f"{priority_icon} [{rec['priority']}] {rec['action']}")
            print(f"   Reason: {rec['reason']}")
            print()
    else:
        print(f"\n✅ No immediate recommendations - samples are fresh!")

    # Manage freshness
    print(f"\n🔧 MANAGING SAMPLE FRESHNESS")
    print("-" * 80)

    response = input("\nRun automatic freshness management? (y/n): ")
    if response.lower() == 'y':
        print("\n🔄 Running freshness management...")
        stats = await db.manage_sample_freshness(
            speaker_name=speaker_name,
            max_age_days=30,
            target_sample_count=100
        )

        if 'error' in stats:
            print(f"❌ Error: {stats['error']}")
        else:
            print(f"\n📊 FRESHNESS MANAGEMENT RESULTS")
            print("-" * 80)
            print(f"Total samples:     {stats['total_samples']}")
            print(f"Fresh samples:     {stats['fresh_samples']}")
            print(f"Stale samples:     {stats['stale_samples']}")
            print(f"Samples archived:  {stats['samples_archived']}")
            print(f"Samples retained:  {stats['samples_retained']}")
            print(f"Freshness score:   {stats['freshness_score']:.1%}")

            if stats['actions']:
                print(f"\n🎯 Actions taken:")
                for action in stats['actions']:
                    print(f"   • {action}")

            print(f"\n✅ Freshness management complete!")

    # Offer to run enrollment
    if report['recommendations']:
        high_priority = any(r['priority'] == 'HIGH' for r in report['recommendations'])
        if high_priority:
            print(f"\n🎙️  HIGH PRIORITY: Your voice profile needs refresh!")
            response = input("\nWould you like to record fresh samples now? (y/n): ")
            if response.lower() == 'y':
                print(f"\n🚀 Launching enrollment script...")
                print(f"\n   Run: python backend/voice/enroll_voice.py --refresh --samples 10")
                print(f"   Or:  python backend/voice/enroll_voice.py --samples 30")

    # Cleanup
    await db.close()

    print("\n" + "="*80)
    print("✅ FRESHNESS MANAGEMENT COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
