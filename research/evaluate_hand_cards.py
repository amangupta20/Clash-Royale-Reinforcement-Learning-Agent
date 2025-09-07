#!/usr/bin/env python3
"""
Evaluate OpenCV template matching for hand card detection across preprocessing variants.

Variants compared:
  1) full_rgb:      Use full-area frames in RGB at original scale
  2) deck_rgb:      Use deck-area frames in RGB at original scale
  3) deck_rgb_div2: Use deck-area frames downscaled by 2 (RGB)
  4) deck_gray_div2:Use deck-area frames downscaled by 2, then grayscale

Outputs:
  - metrics JSON with aggregate timings and detection counts per variant
  - CSV of per-frame/per-template scores
  - annotated images for visual inspection under out_dir

This script imports OpenCV (cv2) only when running evaluations so that `-h` works even
if OpenCV isn't installed yet.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time
from typing import Dict, List, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_config(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_image_files(dir_path: pathlib.Path, patterns=(".png", ".jpg", ".jpeg")) -> List[pathlib.Path]:
    return [p for p in sorted(dir_path.iterdir()) if p.suffix.lower() in patterns]


def ensure_dir(p: pathlib.Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def preprocess_image(img, variant: str):
    """Return processed image according to variant.
    Order for gray_div2 follows config guidance: downscale then grayscale.
    """
    import cv2  # local import

    if variant == "full_rgb" or variant == "deck_rgb":
        return img
    elif variant == "deck_rgb_div2":
        return cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    elif variant == "deck_gray_div2":
        small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def preprocess_template(tpl, variant: str):
    """Preprocess template to match the image processing pipeline."""
    import cv2  # local import

    if variant in ("full_rgb", "deck_rgb"):
        return tpl
    elif variant == "deck_rgb_div2":
        return cv2.resize(tpl, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    elif variant == "deck_gray_div2":
        small = cv2.resize(tpl, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def match_templates(img, templates: List[Tuple[str, any]], method, threshold: float):
    """Run matchTemplate for each template; return list of detections and scores.

    Returns: list of dicts {name, max_val, max_loc, w, h, passed}
    """
    import cv2  # local import

    results = []
    for name, tpl in templates:
        th, tw = tpl.shape[:2]
        res = cv2.matchTemplate(img, tpl, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        results.append({
            "name": name,
            "max_val": float(max_val),
            "max_loc": tuple(int(x) for x in max_loc),
            "w": int(tw),
            "h": int(th),
            "passed": bool(max_val >= threshold),
        })
    return results


def draw_detections(img, detections: List[dict], out_path: pathlib.Path, draw_threshold: float):
    import cv2  # local import

    vis = img.copy()
    # If grayscale, convert to BGR for drawing
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    for det in detections:
        if det["max_val"] >= draw_threshold:
            x, y = det["max_loc"]
            w, h = det["w"], det["h"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis, f"{det['name']}:{det['max_val']:.2f}", (x, max(0, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), vis)


def _percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[int(k)])
    d0 = s[f] * (c - k)
    d1 = s[c] * (k - f)
    return float(d0 + d1)


def run_eval(cfg: dict, out_dir: pathlib.Path, log_run: bool, log_name: str | None, log_tags: str | None,
             repeats: int = 3, warmup: int = 1, present_k: int | None = None) -> dict:
    import cv2  # local import

    ensure_dir(out_dir)
    # Inputs
    cards_dir = ROOT / "assets" / "cards"
    full_frames = [ROOT / "assets" / "deck" / "full_area.png", ROOT / "assets" / "deck" / "full_area_v2.png"]
    deck_frames = [ROOT / "assets" / "deck" / "deck_area.png", ROOT / "assets" / "deck" / "deck_area_v2.png"]

    # Load templates
    templates_raw: List[Tuple[str, any]] = []
    for p in list_image_files(cards_dir):
        tpl = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if tpl is None:
            print(f"[warn] Failed to read template: {p}")
            continue
        templates_raw.append((p.stem, tpl))
    if not templates_raw:
        raise SystemExit("No templates found in assets/cards/")

    # Variants
    variants = [
        ("full_rgb", full_frames),
        ("deck_rgb", deck_frames),
        ("deck_rgb_div2", deck_frames),
        ("deck_gray_div2", deck_frames),
    ]

    method = getattr(cv2, cfg.get("method", "TM_CCOEFF_NORMED"))
    threshold = float(cfg.get("threshold", 0.85))
    k_present = int(present_k if present_k is not None else cfg.get("present_k", 4))

    # Output CSV for detailed scores
    csv_path = out_dir / "scores.csv"
    csv_f = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(csv_f)
    writer.writerow(["variant", "frame", "template", "max_val", "passed"])

    metrics: Dict[str, Dict[str, float]] = {}

    for variant, frames in variants:
        agg_prep, agg_match, agg_total = [], [], []
        agg_pass_all = 0
        agg_scores_all: List[float] = []
        # For top-k metrics per frame
        agg_topk_avg_vals: List[float] = []  # per frame average of top-k max_vals
        agg_topk_detection_rates: List[float] = []  # per frame (#above_thresh in top-k)/k

        # Preprocess templates once per variant
        templates_proc = [(name, preprocess_template(tpl, variant)) for name, tpl in templates_raw]

        for r in range(repeats):
            run_prep, run_match, run_total = [], [], []
            run_pass_all = 0
            run_scores_all: List[float] = []
            run_topk_avg_vals: List[float] = []
            run_topk_rates: List[float] = []

            for frame_path in frames:
                img_color = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if img_color is None:
                    print(f"[warn] Failed to read frame: {frame_path}")
                    continue

                t0 = time.perf_counter()
                img_proc = preprocess_image(img_color, variant)
                t1 = time.perf_counter()

                match_img = img_proc

                t2 = time.perf_counter()
                detections = match_templates(match_img, templates_proc, method, threshold)
                t3 = time.perf_counter()

                # Only save annotations on the last repeat to keep outputs tidy
                if r == repeats - 1:
                    out_img = out_dir / f"annot_{variant}_{frame_path.stem}.png"
                    draw_detections(match_img, detections, out_img, threshold)

                # Write CSV rows (record all repeats)
                for det in detections:
                    writer.writerow([variant, frame_path.name, det["name"], f"{det['max_val']:.6f}", int(det["passed"])])
                    run_pass_all += int(det["passed"])
                    run_scores_all.append(det["max_val"])

                # Compute top-k per-frame metrics
                # Sort all scores desc, take top-k
                sorted_scores = sorted((d["max_val"] for d in detections), reverse=True)
                topk = sorted_scores[:max(1, k_present)]
                topk_avg = sum(topk) / float(len(topk))
                topk_above = sum(1 for s in topk if s >= threshold)
                topk_rate = topk_above / float(len(topk))

                run_topk_avg_vals.append(topk_avg)
                run_topk_rates.append(topk_rate)

                run_prep.append((t1 - t0) * 1000.0)
                run_match.append((t3 - t2) * 1000.0)
                run_total.append((t3 - t0) * 1000.0)

            # Aggregate this repeat, skip warmup repeats
            if r >= warmup:
                agg_prep.extend(run_prep)
                agg_match.extend(run_match)
                agg_total.extend(run_total)
                agg_pass_all += run_pass_all
                agg_scores_all.extend(run_scores_all)
                # one entry per frame for top-k metrics
                agg_topk_avg_vals.extend(run_topk_avg_vals)
                agg_topk_detection_rates.extend(run_topk_rates)

        # Aggregate per-variant metrics after repeats
        if agg_total:
            # Stats over all-template scores (legacy)
            all_scores_mean = float(statistics.mean(agg_scores_all)) if agg_scores_all else 0.0
            all_scores_median = _percentile(agg_scores_all, 0.5)
            all_scores_p90 = _percentile(agg_scores_all, 0.90)
            all_scores_p95 = _percentile(agg_scores_all, 0.95)
            all_scores_max = max(agg_scores_all) if agg_scores_all else 0.0
            all_scores_min = min(agg_scores_all) if agg_scores_all else 0.0

            # Top-k per-frame metrics
            topk_avg_mean = float(statistics.mean(agg_topk_avg_vals)) if agg_topk_avg_vals else 0.0
            topk_avg_median = _percentile(agg_topk_avg_vals, 0.5)
            topk_avg_p90 = _percentile(agg_topk_avg_vals, 0.90)
            topk_avg_p95 = _percentile(agg_topk_avg_vals, 0.95)
            topk_rate_mean = float(statistics.mean(agg_topk_detection_rates)) if agg_topk_detection_rates else 0.0
            topk_rate_median = _percentile(agg_topk_detection_rates, 0.5)

            metrics[variant] = {
                "avg_prep_ms": float(statistics.mean(agg_prep)),
                "avg_match_ms": float(statistics.mean(agg_match)),
                "avg_total_ms": float(statistics.mean(agg_total)),
                # Legacy all-templates stats (kept for reference)
                "all_templates": {
                    "detections_passed": int(agg_pass_all),
                    "avg_max_val": all_scores_mean,
                    "median_max_val": all_scores_median,
                    "p90_max_val": all_scores_p90,
                    "p95_max_val": all_scores_p95,
                    "max_max_val": all_scores_max,
                    "min_max_val": all_scores_min,
                    "detection_rate": float(agg_pass_all) / float(len(frames) * len(templates_proc) * max(1, repeats - warmup)),
                },
                # Top-k present-cards oriented metrics
                "present_k": k_present,
                "topk_avg_max_val_mean": topk_avg_mean,
                "topk_avg_max_val_median": topk_avg_median,
                "topk_avg_max_val_p90": topk_avg_p90,
                "topk_avg_max_val_p95": topk_avg_p95,
                "topk_detection_rate_mean": topk_rate_mean,
                "topk_detection_rate_median": topk_rate_median,
                # Context
                "frames": len(frames),
                "templates": len(templates_proc),
                "threshold": threshold,
                "repeats": repeats,
                "warmup_discarded": warmup,
            }

    csv_f.close()

    # Save metrics
    metrics_obj = {"variants": metrics, "num_variants": len(metrics)}
    (out_dir / "metrics.json").write_text(json.dumps(metrics_obj, indent=2), encoding="utf-8")

    # Save the effective config used
    (out_dir / "config.used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # Optional: log experiment using existing logger
    if log_run:
        logger = ROOT / "research" / "log_experiment.py"
        cmd = [
            sys.executable,
            str(logger),
            "--name", (log_name or "cv-ablation-hand-cards"),
            "--config", str(out_dir / "config.used.json"),
            "--metrics-file", str(out_dir / "metrics.json"),
        ]
        if log_tags:
            cmd += ["--tags", log_tags]
        # Attach a couple of artifacts
        # Pick at most 4 images to attach
        attached = 0
        for p in sorted(out_dir.glob("annot_*.png")):
            cmd += ["--artifact", str(p)]
            attached += 1
            if attached >= 4:
                break
        print("[info] Logging experiment via:", " ".join(cmd))
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print(f"[warn] log_experiment failed: {e}")

    return metrics_obj


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate hand card detection (OpenCV template matching)")
    ap.add_argument("--config", default=str((ROOT / "research" / "configs" / "hand_cards_eval.json").resolve()),
                    help="Path to evaluation config JSON")
    ap.add_argument("--out", default=str((ROOT / "research" / "outputs" / "hand_cards_eval").resolve()),
                    help="Output directory for artifacts and metrics")
    ap.add_argument("--log-run", action="store_true", help="Also log the run via research/log_experiment.py")
    ap.add_argument("--log-name", default=None, help="Name/slug for the experiment log entry")
    ap.add_argument("--log-tags", default="cv,ablation,hand-cards", help="Comma-separated tags for experiment log")
    ap.add_argument("--repeats", type=int, default=3, help="Number of repeated runs for timing stats")
    ap.add_argument("--warmup", type=int, default=1, help="Number of initial repeats to discard as warmup")
    ap.add_argument("--present-k", type=int, default=None, help="Expected number of present cards (top-k). Defaults to config.present_k or 4.")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg_path = pathlib.Path(args.config)
    out_dir = pathlib.Path(args.out)
    cfg = load_config(cfg_path)
    res = run_eval(cfg, out_dir, bool(args.log_run), args.log_name, args.log_tags,
                   repeats=args.repeats, warmup=args.warmup, present_k=args.present_k)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
