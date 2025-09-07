# Research logging and reproducibility

This folder provides a lightweight, code-first workflow to save the exact context of your experiments so conference/paper writing is painless later.

What you get:

- Per-experiment run directory with config, metrics, notes, and artifacts
- An append-only `experiments.jsonl` index for quick tabulation
- Simple CLI (`log_experiment.py`) with zero external dependencies
- Templates for configs and notes

## Folder layout

- `experiments/` — one subfolder per run: `YYYYMMDD_HHMMSS_slug/`
  - `config.json` — all knobs for that run (inputs, model, CV pipeline settings, etc.)
  - `metrics.json` — measured outputs (fps, accuracy, latency, etc.)
  - `notes.md` — brief rationale, observations, and decisions
  - `env.json` — OS/Python/git commit snapshot
  - `artifacts/` — plots, screenshots, CSVs, confusion matrices, etc.
- `experiments.jsonl` — newline-delimited summary index for fast analysis
- `templates/` — starter files for configs and notes
- `log_experiment.py` — CLI to create runs and append to index

## Quick start

1. Copy and edit the sample config

- From `templates/experiment.config.json` to your working copy.

2. Run your evaluation to compute metrics

- Produce a small metrics JSON like: `{ "fps": 36.4, "card_match_acc": 0.94, "cap_latency_ms": 7.8 }`.

3. Log the experiment

- Use the CLI to persist the run folder and index entry (see “How to use”).

## How to use

- Minimal example (metrics JSON string):

```powershell
python research/log_experiment.py --name "res-480x854-gray2" --config research/templates/experiment.config.json --metrics '{"fps":36.4,"card_match_acc":0.94}' --tags resolution,cv,grayscale
```

- With metrics from a file and artifacts:

```powershell
python research/log_experiment.py --name "ablation-gray-vs-rgb" --config path/to/config.json --metrics-file path/to/metrics.json --artifact path/to/plot.png --artifact path/to/sample_overlay.jpg --tags ablation,cv
```

The command creates `research/experiments/YYYYMMDD_HHMMSS_res-480x854-gray2/` with all files plus appends a line to `research/experiments.jsonl`.

## What to capture (paper-ready)

- Configuration:
  - Capture & crop: library, window, crop offsets; standardized resolution (e.g., 480x854)
  - CV pipeline: hand card detection (OpenCV template match), elixir estimator, OCR options, grayscale/downscale specifics
  - Evaluation dataset: date, split, any filters; fixed seeds
- Metrics:
  - Throughput: fps; capture latency (ms); preprocessing latency (ms); detection latency (ms)
  - Accuracy: card detection precision/recall or hit@1; OCR/Troop detection metrics as applicable
  - Stability: dropped frames, ADB input success rate
- Artifacts:
  - Representative frames, overlays, PR curves/plots, error cases
- Environment:
  - OS, Python, key package versions; git commit hash
- Decision hook:
  - When a result changes a design choice, log a short decision in ConPort (summary + rationale)

## Protocol examples you can reuse

1. Finalizing resolution (e.g., 480x854)

- For each candidate resolution, log a run with:
  - Constant crop settings and identical evaluation subset
  - Metrics: fps, total latency, memory usage if available, card_match_acc
  - Artifact: grid of side-by-side overlays (qualitative)
- Decision rule example: choose smallest resolution with ≥X% of baseline accuracy and ≥Y fps.

2. Grayscale/2 decision

- Compare RGB vs grayscale vs grayscale/2 (downscale first, grayscale after, or vice versa — be explicit in config)
- Log metrics: card_match_acc, detection latency, overall fps; add 5–10 diverse frames for qualitative checks
- Document noise/lighting sensitivity in notes; choose grayscale mode with minimal accuracy drop and best latency

### Hand cards evaluation helper

Use the helper to compare full vs deck area and grayscale/div2 variants and optionally log a run:

```powershell
python research/evaluate_hand_cards.py --config research/configs/hand_cards_eval.json --out research/outputs/hand_cards_eval --log-run --log-name "hand-cards-ablation" --log-tags "cv,hand-cards,ablation"
```

Artifacts and metrics will be written under `research/outputs/hand_cards_eval/`. When `--log-run` is provided, a summarized run is also stored via `research/log_experiment.py` under `research/experiments/`.

## Tips

- Keep configs small and explicit. Do not rely on defaults for critical choices.
- Prefer small, repeatable evaluation subsets for ablations; only scale up for the final confirmation.
- Append links or file paths to any external spreadsheets or papers you reference.

## Exporting for the paper

- The `experiments.jsonl` file can be converted into tables
- Artifacts inside each run can be copied directly into figures
- Decisions are kept in ConPort; export them when writing (we can automate export later)

---

This workflow is intentionally simple. If you want heavier tooling later (e.g., MLflow), this structure can be migrated with minimal effort.
