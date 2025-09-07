# Notes

Experiment: hand-cards-ablation-warm (repeats=4, warmup=1)

Summary

- Cropping from full area to just the deck area did not reduce detection confidence (median max_val ~0.73 across both) but massively improved performance (avg_total_ms ~452 → ~42 ms).
- Moving from deck RGB to grayscale/2 (downscale by 2 then grayscale) further improved performance substantially (~42 ms → ~1.07 ms) with only a slight decrease in average confidence (avg_max_val ~0.717 → ~0.713; median ~0.734 → ~0.731). The performance gains significantly outweigh the small confidence drop.

Details (threshold=0.85, assets/cards current deck, frames=2)

- full_rgb: avg_total_ms ~452.22; median_conf ~0.735; detection_rate 0.50
- deck_rgb: avg_total_ms ~41.95; median_conf ~0.734; detection_rate 0.50
- deck_rgb_div2: avg_total_ms ~10.93; median_conf ~0.710; detection_rate 0.50
- deck_gray_div2: avg_total_ms ~1.07; median_conf ~0.731; detection_rate 0.50

Decision Implications

- Adopt deck-area cropping for hand card detection to minimize compute without hurting confidence.
- Prefer grayscale/2 for production (downscale_then_grayscale) given 40x+ speedup vs deck RGB with negligible confidence impact on this dataset.

Method Notes

- Confidence statistics reported: avg/median/p90/p95/min/max of matchTemplate max_val per template and frame.
- Timing stability improved by discarding 1 warmup repeat and averaging over 3 kept repeats (total repeats=4).
