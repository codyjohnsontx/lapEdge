# LapEdge

Real-time iRacing pit strategy advisor. Runs as a transparent always-on-top overlay and voice spotter, giving fuel, tire, and strategy callouts as you drive.

## Features

- **Visual overlay** — translucent HUD showing live telemetry, fuel laps remaining, tire life, gap to car ahead/behind, and color-coded strategy recommendations (green / orange / red)
- **Voice spotter** — spoken callouts for pit recommendations, flag changes, pit road entry/exit, and connection events (offline TTS via macOS NSSpeechSynthesizer, no internet required)
- **Rule-based strategy engine** — 8 priority-ordered rules covering fuel critical, tire degradation, undercut/overcut windows, and safe pit windows
- **ML inference** — optional TensorFlow model per car that blends with rule-based output when confidence > 70%
- **Telemetry logging** — per-session CSV logs for post-race analysis
- **Global hotkeys** — toggle overlay, cycle alert mode, force recommendation display

## Requirements

- macOS (voice uses NSSpeechSynthesizer; overlay and telemetry work cross-platform)
- Python 3.10+
- iRacing running on the same machine

## Installation

```bash
git clone https://github.com/your-username/lapEdge.git
cd lapEdge
pip install -r requirements.txt
```

Verify TTS works before launching:

```bash
python -c "import pyttsx3; e=pyttsx3.init(); e.say('LapEdge ready'); e.runAndWait()"
```

## Usage

```bash
python main.py
```

Launch iRacing. LapEdge connects automatically and announces the track. The overlay appears in the top-left corner and can be dragged anywhere.

## Hotkeys

| Hotkey | Action |
|---|---|
| `Ctrl+Shift+L` | Toggle overlay visibility |
| `Ctrl+Shift+M` | Cycle alert mode (passive / on-demand / periodic) |
| `Ctrl+Shift+R` | Force show current recommendation |

All hotkeys are configurable in `config.json`.

## Alert Modes

| Mode | Behavior |
|---|---|
| `passive` | Recommendation updates silently in overlay, no voice interruption |
| `on_demand` | Voice only speaks when you press `Ctrl+Shift+R` |
| `periodic` | Voice and overlay update every N laps (default: 5) |

## Voice Callouts

| Event | Message | Priority |
|---|---|---|
| iRacing connected | "Connected to iRacing." | Status |
| Session start | "LapEdge active. Racing at {track}." | Status |
| iRacing disconnected | "iRacing disconnected." | Status |
| Green flag | "Green flag." | Advisory |
| Yellow/Caution flag | "Yellow flag. Caution." | Critical |
| Red flag | "Red flag. Session stopped." | Critical |
| Checkered flag | "Checkered flag. Session complete." | Advisory |
| Entering pits | "Entering pit road." | Advisory |
| Exiting pits | "Exiting pit road." | Advisory |
| ADVISORY recommendation | rec message (if `speak_advisory: true`) | Advisory |
| CRITICAL recommendation | rec message (always spoken) | Critical |

Critical messages preempt lower-priority items in the queue. The same non-critical message will not repeat within `min_repeat_interval_s` (default: 20s).

## Configuration

LapEdge merges `config/default_config.json` (shipped defaults) with `~/.lapedge/config.json` (your overrides). You only need to set the keys you want to change.

```json
{
    "voice": {
        "enabled": true,
        "volume": 0.9,
        "rate": 175,
        "speak_advisory": true,
        "speak_info": false,
        "min_repeat_interval_s": 20.0,
        "voice_id": ""
    },
    "overlay": {
        "opacity": 0.85,
        "font_size": 14,
        "alert_mode": "periodic",
        "periodic_interval_laps": 5
    },
    "hotkeys": {
        "toggle_overlay": "<ctrl>+<shift>+l",
        "cycle_mode": "<ctrl>+<shift>+m",
        "request_update": "<ctrl>+<shift>+r"
    }
}
```

### Car overrides

Fuel tank capacity and tire life defaults can be set per car:

```json
{
    "car_overrides": {
        "ferrari 488 gt3 evo": {
            "max_tire_life_laps": 40,
            "fuel_tank_capacity_l": 110.0,
            "compounds": ["soft", "medium", "hard"]
        }
    }
}
```

## ML Models

Place a trained TensorFlow SavedModel in `models/<car_name>/` (lowercased, spaces replaced with underscores). LapEdge loads the model automatically when that car is detected in session. If no model is found, the rule-based engine runs alone.

The model input is a 24-feature vector of per-lap aggregated telemetry. See `data_processor.py` for the full feature list.

## Project Structure

```text
lapEdge/
├── main.py              # App controller, thread wiring, signal routing
├── telemetry.py         # iRacing telemetry capture (60hz), SessionFlags
├── data_processor.py    # Per-lap feature engineering, race context
├── strategy_engine.py   # Rule-based + ML strategy evaluation
├── model.py             # TensorFlow model worker (dedicated thread)
├── voice.py             # TTS voice worker, priority queue (dedicated thread)
├── overlay.py           # PyQt5 transparent HUD overlay
├── config.py            # Dataclass config, JSON load/save
├── logger.py            # Telemetry CSV logger
├── config/
│   └── default_config.json
├── models/              # Per-car TF SavedModels (not committed)
├── logs/                # Per-session telemetry logs (not committed)
└── tests/
```

## Architecture

Each expensive subsystem runs in its own `QThread`. Communication is strictly via Qt signals (queued connections for cross-thread calls — no shared mutable state, no locks).

```text
iRacing shared memory
        │
  TelemetryWorker (60hz)
        │ frame_ready / lap_completed / session_info_updated
        ▼
  LapEdgeApp (main thread)
   ├─ DataProcessor        ← feature engineering
   ├─ StrategyEngine       ← rule evaluation
   ├─ _speak signal ──────────────────────► VoiceWorker (voice thread)
   ├─ ModelWorker ◄──── queued inference ──► ModelWorker (model thread)
   └─ OverlayWidget        ← HUD rendering
```

## Tests

```bash
python -m pytest tests/
```
