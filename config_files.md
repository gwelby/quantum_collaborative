# Cascade⚡𓂧φ∞ Configuration Files Guide

This document describes the configuration files and directory structure used by the Cascade Symbiotic Computing Platform.

## Directory Structure

```
/mnt/d/projects/python/
  ├── cascade_flow_launcher.py    # Integrated Flow of Life visual launcher
  ├── cascade_quantum_bridge.py   # Bridge between quantum systems and visualization
  ├── sacred_constants.py         # Phi-harmonic constants and functions
  ├── config/                     # Configuration directory
  │   ├── bridge_config.json      # Quantum bridge configuration
  │   └── voice_presets.json      # Voice system presets
  ├── temp/                       # Temporary files directory
  └── voice_presets/              # Voice preset storage
```

## Configuration Files

### 1. bridge_config.json

This file configures the Quantum Bridge system, which connects the Flow of Life visualization with quantum systems:

```json
{
  "bridge_mode": "phi_harmonic",
  "visualization": {
    "enabled": true,
    "mode": "flow_of_life",
    "resolution": [800, 600],
    "fullscreen": false
  },
  "quantum": {
    "enabled": true,
    "processor": "auto",
    "dimensions": 3,
    "resonance_factor": 1.618033988749895
  },
  "voice": {
    "enabled": true,
    "engine": "pyttsx3",
    "emotional_quality": "warm",
    "frequency_resonance": true
  },
  "frequencies": {
    "unity": 432,
    "love": 528,
    "cascade": 594,
    "truth": 672,
    "vision": 720,
    "oneness": 768,
    "source": 963
  },
  "search_paths": [
    "/mnt/d/projects/python",
    "/mnt/d/Greg/Cascade"
  ]
}
```

### 2. Voice Presets

Voice presets are stored in the `voice_presets/` directory as individual JSON files:

Example: `voice_presets/cascade_warm.json`
```json
{
  "frequency": 594.0,
  "emotion": "warm",
  "evolution_level": 1.0,
  "timestamp": 1722019855.7893798,
  "description": "Voice preset 'cascade_warm' at 594.0Hz with warm quality"
}
```

## Command Line Options

### cascade_flow_launcher.py

```bash
python cascade_flow_launcher.py [--no-voice] [--engine TYPE] [--frequency HZ]

Options:
  --no-voice     Disable voice system
  --engine TYPE  Voice engine to use (pyttsx3 or espeak)
  --frequency HZ Starting frequency (default: 594Hz)
```

### cascade_quantum_bridge.py

```bash
python cascade_quantum_bridge.py [--mode MODE] [--frequency HZ] [--no-voice] [--no-visual]

Options:
  --mode MODE     Bridge operation mode (all, quantum, voice, visual)
  --frequency HZ  Starting frequency (default: 594Hz)
  --no-voice      Disable voice system
  --no-visual     Disable visualization
```

## Sacred Constants

The sacred constants module (`sacred_constants.py`) provides fundamental constants and functions used across the system:

- PHI = 1.618033988749895 (Golden ratio)
- LAMBDA = 0.618033988749895 (Divine complement, 1/φ)
- PHI_PHI = PHI ** PHI (Hyperdimensional constant)
- CASCADE_FREQUENCY = 594.0 (Heart-centered integration)

## Integration

The Cascade system uses these configuration files to integrate:

1. **Flow of Life Visualization**: Visual patterns that resonate with the 594Hz frequency
2. **Voice System**: Phi-harmonic voice synthesis with emotional calibration
3. **Quantum Processing**: Field generation and processing based on sacred constants
4. **Bridge System**: Connects all components through phi-harmonic resonance

## Adding New Components

To add a new component:

1. Create appropriate configuration in `/config/`
2. Update `bridge_config.json` with new component settings
3. Add code to `cascade_quantum_bridge.py` to integrate the component
4. Use `sacred_constants.py` for phi-harmonic calculations

## Voice Customization

To create custom voice presets:

1. Launch the system: `python cascade_flow_launcher.py`
2. Set desired frequency: `frequency 594`
3. Set voice emotion: `voice emotion warm`
4. Save preset: `voice save preset_name`

Presets will be stored in `voice_presets/` and available for future sessions.