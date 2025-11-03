#!/usr/bin/env python3
"""
Interactive CLI Setup Wizard for JARVIS TTS Voice Integration

Automatically configures:
- ElevenLabs API key (FREE tier)
- Voice discovery and selection
- Curated voice profiles for security testing
- Cache optimization for zero-cost operation

Usage:
    python3 setup_tts_voices.py
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from backend.audio.elevenlabs_tts_service import ElevenLabsTTSService


class Colors:
    """ANSI color codes for pretty CLI output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text: str):
    """Print a bold header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.END}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.END}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.END}")


def print_step(step: int, total: int, text: str):
    """Print step indicator"""
    print(f"\n{Colors.BOLD}[Step {step}/{total}] {text}{Colors.END}")


def get_input(prompt: str, default: Optional[str] = None) -> str:
    """Get user input with optional default"""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    return input(f"{prompt}: ").strip()


def get_yes_no(prompt: str, default: bool = True) -> bool:
    """Get yes/no input from user"""
    default_str = "Y/n" if default else "y/N"
    response = input(f"{prompt} [{default_str}]: ").strip().lower()

    if not response:
        return default
    return response in ['y', 'yes']


class TTSSetupWizard:
    """Interactive CLI wizard for TTS voice setup"""

    def __init__(self):
        self.cache_dir = Path.home() / ".jarvis" / "tts_cache" / "elevenlabs"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key: Optional[str] = None
        self.service: Optional[ElevenLabsTTSService] = None
        self.all_voices: List[Dict] = []
        self.curated_voices: Dict = {}

    async def run(self):
        """Run the complete setup wizard"""
        print_header("🎙️  JARVIS TTS Voice Setup Wizard")

        print(f"{Colors.BOLD}This wizard will help you set up ElevenLabs TTS integration.{Colors.END}")
        print("We'll configure diverse accent voices for comprehensive security testing.\n")

        print_info("Using FREE tier ElevenLabs (10,000 characters/month)")
        print_info("With caching: Generate voices once, reuse forever (zero cost)")

        if not get_yes_no("\nReady to begin?", default=True):
            print("\nSetup cancelled.")
            return

        try:
            # Step 1: Check/Set API Key
            await self.step_api_key()

            # Step 2: Discover voices
            await self.step_discover_voices()

            # Step 3: Auto-select voices
            await self.step_auto_select_voices()

            # Step 4: Save configuration
            await self.step_save_config()

            # Step 5: Test voices (optional)
            await self.step_test_voices()

            # Final summary
            self.print_summary()

        except KeyboardInterrupt:
            print_warning("\n\nSetup interrupted by user")
            sys.exit(1)
        except Exception as e:
            print_error(f"\n\nSetup failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        finally:
            if self.service:
                await self.service.close()

    async def step_api_key(self):
        """Step 1: Check/Set ElevenLabs API key"""
        print_step(1, 5, "ElevenLabs API Key Setup")

        # Check if already set
        self.api_key = os.getenv("ELEVENLABS_API_KEY")

        if self.api_key:
            print_success(f"API key found in environment: {self.api_key[:8]}...{self.api_key[-4:]}")

            if not get_yes_no("Use this API key?", default=True):
                self.api_key = None

        # Prompt for new key if needed
        if not self.api_key:
            print("\n📋 To get your FREE API key:")
            print("   1. Go to: https://elevenlabs.io/app/settings/api-keys")
            print("   2. Sign up for a FREE account (no credit card required)")
            print("   3. Copy your API key\n")

            self.api_key = get_input("Enter your ElevenLabs API key")

            if not self.api_key:
                raise ValueError("API key is required")

            # Offer to save to shell profile
            if get_yes_no("\nSave API key to ~/.zshrc? (recommended)", default=True):
                self.save_api_key_to_profile()

        # Initialize service
        self.service = ElevenLabsTTSService(api_key=self.api_key)
        print_success("ElevenLabs TTS service initialized")

    def save_api_key_to_profile(self):
        """Save API key to shell profile"""
        shell_profile = Path.home() / ".zshrc"

        if not shell_profile.exists():
            shell_profile = Path.home() / ".bashrc"

        if not shell_profile.exists():
            print_warning("Could not find ~/.zshrc or ~/.bashrc")
            print(f"   Add this to your shell profile manually:")
            print(f"   export ELEVENLABS_API_KEY='{self.api_key}'")
            return

        with open(shell_profile, 'a') as f:
            f.write(f"\n# ElevenLabs API Key (added by JARVIS setup)\n")
            f.write(f"export ELEVENLABS_API_KEY='{self.api_key}'\n")

        print_success(f"API key saved to {shell_profile}")
        print_info(f"Run: source {shell_profile}")

    async def step_discover_voices(self):
        """Step 2: Discover available voices"""
        print_step(2, 5, "Discovering Available Voices")

        print_info("Fetching voices from ElevenLabs API...")

        voices = await self.service.discover_voices(force_refresh=True)

        # Load the raw voice data
        all_voices_file = self.cache_dir / "all_voices.json"
        if all_voices_file.exists():
            with open(all_voices_file, 'r') as f:
                self.all_voices = json.load(f)

        print_success(f"Discovered {len(self.all_voices)} available voices")

        # Show voice breakdown
        categories = {}
        for voice in self.all_voices:
            category = voice.get('category', 'unknown')
            categories[category] = categories.get(category, 0) + 1

        print("\n📊 Voice Categories:")
        for category, count in sorted(categories.items()):
            print(f"   {category}: {count} voices")

    async def step_auto_select_voices(self):
        """Step 3: Automatically select diverse voices"""
        print_step(3, 5, "Auto-Selecting Diverse Voices")

        print_info("Selecting voices based on labels and descriptions...")

        # Target accents we want
        target_accents = {
            'african_american': ['african american', 'black', 'soul', 'urban'],
            'african': ['nigerian', 'kenyan', 'south african', 'african', 'ghana'],
            'asian': ['chinese', 'japanese', 'korean', 'asian', 'mandarin', 'cantonese']
        }

        selected = {
            'african_american': [],
            'african': [],
            'asian': []
        }

        # Match voices to accents
        for voice in self.all_voices:
            name = voice.get('name', '').lower()
            description = voice.get('description', '').lower()
            labels = ' '.join(voice.get('labels', {}).values()).lower()

            voice_text = f"{name} {description} {labels}"

            for accent, keywords in target_accents.items():
                if any(keyword in voice_text for keyword in keywords):
                    selected[accent].append(voice)

        # Build curated configuration
        self.curated_voices = {}
        voice_counter = 1

        for accent, voices in selected.items():
            print(f"\n{accent.upper()}:")

            # Limit to 4-6 voices per accent category
            max_voices = 6 if accent == 'african_american' else 5

            for i, voice in enumerate(voices[:max_voices]):
                voice_id = voice.get('voice_id')
                name = voice.get('name', 'Unknown')

                # Determine gender from labels or name
                gender = 'male'
                if any(g in str(voice).lower() for g in ['female', 'woman', 'girl']):
                    gender = 'female'

                key = f"{accent}_{gender}_{i+1}"

                self.curated_voices[key] = {
                    "voice_id": voice_id,
                    "name": name.replace(' ', ''),
                    "accent": accent,
                    "gender": gender,
                    "description": voice.get('description', '')[:100],
                    "language_code": "en",
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": True
                }

                print(f"   ✅ {name} ({gender})")
                voice_counter += 1

        total_selected = sum(len(v) for v in selected.values())
        print_success(f"\nAuto-selected {total_selected} diverse voices")

        if total_selected < 10:
            print_warning(f"Only found {total_selected} voices matching target accents")
            print_info("You may want to manually review and add more voices")

    async def step_save_config(self):
        """Step 4: Save curated configuration"""
        print_step(4, 5, "Saving Voice Configuration")

        curated_file = self.cache_dir / "curated_voices.json"

        with open(curated_file, 'w') as f:
            json.dump(self.curated_voices, f, indent=2)

        print_success(f"Saved {len(self.curated_voices)} voices to {curated_file}")

        # Show breakdown
        breakdown = {}
        for voice in self.curated_voices.values():
            accent = voice['accent']
            breakdown[accent] = breakdown.get(accent, 0) + 1

        print("\n📊 Configured Voices:")
        for accent, count in sorted(breakdown.items()):
            print(f"   {accent}: {count} voices")

    async def step_test_voices(self):
        """Step 5: Test voice generation (optional)"""
        print_step(5, 5, "Testing Voice Generation (Optional)")

        if not get_yes_no("\nGenerate sample audio to verify voices?", default=True):
            print_info("Skipping voice testing")
            return

        # Load curated voices
        self.service.load_curated_voices()

        if not self.service.available_voices:
            print_error("No voices loaded!")
            return

        test_phrase = "unlock my screen"
        output_dir = Path("/tmp/jarvis_voice_samples")
        output_dir.mkdir(exist_ok=True)

        print_info(f"\nGenerating samples: '{test_phrase}'")
        print_info(f"Output directory: {output_dir}\n")

        # Test first 3 voices (to minimize API usage)
        voices_to_test = list(self.service.available_voices.values())[:3]

        for voice_config in voices_to_test:
            print(f"🔊 Generating: {voice_config.name:30} ({voice_config.accent.value})")

            try:
                audio_data = await self.service.synthesize_speech(
                    text=test_phrase,
                    voice_config=voice_config,
                    use_cache=True
                )

                output_file = output_dir / f"{voice_config.name}.mp3"
                output_file.write_bytes(audio_data)
                print_success(f"   Saved: {output_file}")

            except Exception as e:
                print_error(f"   Failed: {e}")

        print_success(f"\n✅ Generated {len(voices_to_test)} sample voices")
        print_info(f"\nTo listen: afplay {output_dir}/*.mp3")
        print_info(f"Or open folder: open {output_dir}")

    def print_summary(self):
        """Print final setup summary"""
        print_header("🎉 Setup Complete!")

        print(f"{Colors.BOLD}Summary:{Colors.END}")
        print(f"   ✅ ElevenLabs API key configured")
        print(f"   ✅ {len(self.all_voices)} voices discovered")
        print(f"   ✅ {len(self.curated_voices)} voices configured for testing")
        print(f"   ✅ Cache directory: {self.cache_dir}")

        print(f"\n{Colors.BOLD}Free Tier Usage:{Colors.END}")
        chars_per_voice = len("unlock my screen")
        total_chars = len(self.curated_voices) * chars_per_voice
        print(f"   One-time generation: {total_chars} characters")
        print(f"   Free tier limit: 10,000 characters/month")
        print(f"   Remaining: {10000 - total_chars} characters")
        print(f"   {Colors.GREEN}✅ Well within FREE tier!{Colors.END}")

        print(f"\n{Colors.BOLD}Next Steps:{Colors.END}")
        print(f"   1. Run voice security test:")
        print(f"      {Colors.CYAN}python3 backend/voice_unlock/voice_security_tester.py{Colors.END}")
        print(f"\n   2. View cache stats:")
        print(f"      {Colors.CYAN}python3 -c 'from backend.audio.tts_provider_manager import TTSProviderManager; import asyncio; m = TTSProviderManager(); print(m.get_cache_stats())'{Colors.END}")

        print(f"\n{Colors.BOLD}Voice Coverage:{Colors.END}")
        print(f"   GCP TTS: 60 voices (US, British, Australian, Indian, Hispanic, European)")
        print(f"   ElevenLabs: {len(self.curated_voices)} voices (African American, African, Asian)")
        print(f"   {Colors.GREEN}Total: {60 + len(self.curated_voices)} unique voices{Colors.END}")


async def main():
    """Main entry point"""
    wizard = TTSSetupWizard()
    await wizard.run()


if __name__ == "__main__":
    asyncio.run(main())
