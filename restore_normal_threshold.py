#!/usr/bin/env python3
import json
config_path = "/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend/config/voice_config.json"
with open(config_path, 'r') as f:
    config = json.load(f)
config['verification_threshold'] = config.get('previous_threshold', 0.45)
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"✅ Threshold restored to {config['verification_threshold']*100:.0f}%")
