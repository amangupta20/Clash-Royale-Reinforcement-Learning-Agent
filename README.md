# Clash Royale Reinforcement Learning Agent

An experimental reinforcement learning project to train an autonomous agent to play Clash Royale via computer vision and automated input (no private game APIs). The agent observes the game through screen capture of a BlueStacks emulator window, converts pixels into a structured state, and executes actions using ADB shell commands.

## Current Status (Tooling & Prototype Phase)

We are in an early tooling phase validating capture, preprocessing, and input pipelines before implementing the full Gymnasium environment and PPO training loop.

### Key Decisions (So Far)

- **Screen Capture (Windows):** Using `windows-capture` (event-driven) for the BlueStacks window. `DXcam` retained as fallback.
- **Standard Frame Resolution:** All frames cropped then downscaled to **480x854** for deterministic, lower‑compute downstream processing.
- **ADB Control:** Pure Python `adb-shell` library (no external adb binary) for launching the game and issuing shell/input commands.
- **Template Matching:** OpenCV template matching prototype for card detection proof-of-concept.
- **Environment Config:** `.env` driven configuration for crop offsets, window name, paths, and device connection.

### Helper Scripts Overview (`HelperScripts/`)

| Script                                        | Purpose                                                                          |
| --------------------------------------------- | -------------------------------------------------------------------------------- |
| `windows-capture-testing.py`                  | One-shot frame grab for a specified BlueStacks window.                           |
| `windows-capture-with required resolution.py` | Capture → crop → resize → save normalized 480p frame.                            |
| `downscale.py`                                | Standalone image downscaling utility (defaults 480x854).                         |
| `Card_Template_matching_Example.py`           | OpenCV template match demo for card detection latency check.                     |
| `py_adb.py`                                   | Interactive adb-shell client + auto launch of Clash Royale intent.               |
| `frame-extract.py`                            | Extract frames from recorded gameplay videos at intervals for dataset creation.  |
| `.env.example`                                | Template environment variables (window name, crop, device IP/port, asset paths). |
| `requirements.txt`                            | Minimal dependency list for current tooling layer.                               |

### Environment Variables (Excerpt)

Create a `.env` based on `.env.example`:

```
WINDOW_NAME="BlueStacks App Player 1"
CROP_LEFT=657
CROP_RIGHT=657
TARGET_WIDTH=480
TARGET_HEIGHT=854
ADB_DEVICE_IP=127.0.0.1
ADB_DEVICE_PORT=5555
CARD_IMAGE_PATH=path/to/card.png
GAME_STATE_IMAGE_PATH=path/to/state.png
```

## Planned Architecture (Roadmap Alignment)

| Layer                | Description                                                                         |
| -------------------- | ----------------------------------------------------------------------------------- |
| Capture & Preprocess | Event-driven window capture → crop → standard 480p tensor.                          |
| CV Extraction        | YOLO (troops), OCR (tower health), pixel counting (elixir), template match (cards). |
| State Assembly       | Handcrafted feature vector (global + unit + tower + hand features).                 |
| Action Space         | Hierarchical MultiDiscrete: [card_slot, x_tile, y_tile] with action masking.        |
| RL Core              | PPO (Stable-Baselines3) with composite reward (terminal + shaping).                 |
| Network              | Multi-modal (CNN for spatial grid + MLP for scalars, dual actor/critic heads).      |
| Scaling (Future)     | Parallel rollouts & self-play (Ray RLlib) after single-instance stability.          |

## Next Immediate Tasks

1. Integrate capture + downscale + template match into a single prototype pipeline script.
2. Add timing benchmarks (capture latency, preprocessing ms, template match ms).
3. Introduce structured logging & error handling wrappers.
4. Draft minimal `ClashRoyaleEnv` scaffold (reset/step placeholders) once pipeline stable.

## Installing & Running (Tooling Phase)

```
python -m venv .venv
./.venv/Scripts/activate  # Windows PowerShell
pip install -r HelperScripts/requirements.txt
cp HelperScripts/.env.example .env
# Edit .env with correct window name & paths
python HelperScripts/windows-capture-testing.py
python HelperScripts/windows-capture-with\ required\ resolution.py
python HelperScripts/py_adb.py
```

## Safety & Anti-Detection Measures (Planned)

- Coordinate jitter & variable tap delays.
- Avoid deterministic frame pacing for action issuance.
- Optional throttling & humanized randomization for non-critical actions.

## ConPort Knowledge Base Integration

Decisions, system patterns, technical specs, and progress items are synchronized via ConPort MCP (see context_portal/). This README reflects the currently logged early-phase decisions.

## License

(Add a license file if intending to release publicly.)

---

_This README will expand as the project transitions from tooling prototypes to the formal RL environment and training stack._
