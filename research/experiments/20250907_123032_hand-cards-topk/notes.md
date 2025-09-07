# Notes

Experiment: hand-cards-topk (repeats=4, warmup=1, present_k=4)

Purpose

- Evaluate hand card detection (OpenCV template matching) across preprocessing variants with top‑k interpretation (k=4) to reflect that only 4 of 8 templates are expected per frame.

Key Findings

- Top‑k correctness: For all variants, the top 4 matches per frame exceed threshold (topk_detection_rate_mean ≈ 1.0), confirming that the expected deck cards are reliably identified among 8 templates.
- Cropping to deck area: Confidence (top‑k avg) remains high while performance jumps massively versus full-area.
  - full_rgb → deck_rgb: ~457.64 ms → ~34.47 ms avg_total_ms; top‑k avg ≈ 0.98 in both.
- Downscale and grayscale:
  - deck_rgb → deck_rgb_div2: ~34.47 ms → ~17.23 ms; top‑k avg ≈ 0.98 → ≈ 0.968 (small drop).
  - deck_rgb → deck_gray_div2: ~34.47 ms → ~1.12 ms (≈30× faster) with top‑k avg ≈ 0.978 → ≈ 0.968 (negligible impact in this dataset).

Decision Implications

- Adopt deck-area cropping for hand card detection; negligible confidence change, large speedup.
- Adopt grayscale/2 (downscale_then_grayscale) for production; speed gains significantly outweigh the small reduction in average confidence.

Numbers (threshold=0.85, frames=2, templates=8)

- full_rgb: avg_total_ms ~457.64; topk_avg_max_val_mean ~0.975; topk_detection_rate_mean ~1.0
- deck_rgb: avg_total_ms ~34.47; topk_avg_max_val_mean ~0.978; topk_detection_rate_mean ~1.0
- deck_rgb_div2: avg_total_ms ~17.23; topk_avg_max_val_mean ~0.968; topk_detection_rate_mean ~1.0
- deck_gray_div2: avg_total_ms ~1.12; topk_avg_max_val_mean ~0.968; topk_detection_rate_mean ~1.0

Method Notes

- Repeats=4 with warmup=1 to remove initial outlier. Stats aggregate over 3 kept repeats.
- Confidence is summarized per frame by averaging the top‑k template scores, then aggregated (mean/median/p90/p95).
- assets folder shows the assets everything was run on for this experiment
