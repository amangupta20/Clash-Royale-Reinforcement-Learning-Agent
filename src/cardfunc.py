"""
Card identification helpers operating on in-memory frames.

- Accept frames as np.ndarray (BGR) directly, no disk I/O.
- Perform cropping/resizing inside the function to meet time-critical needs.
- Provide a stub identify_hand_cards() that you can extend with template matching.
"""

from __future__ import annotations

import os
import pathlib
from typing import Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np


def preprocess_frame(
	frame_bgr: np.ndarray,
	target_size: Tuple[int, int] = None,
	crop_left: Optional[int] = None,
	crop_right: Optional[int] = None,
) -> np.ndarray:
	"""Crop (if configured) and resize a frame in-memory.

	- frame_bgr: ndarray HxWx3 (BGR)
	- target_size: (width, height) or None to keep original size
	- crop_left/right: columns to trim from the left/right before resize
	Returns: processed BGR ndarray
	"""
	img = frame_bgr
	h, w = img.shape[:2]
	if crop_left is not None or crop_right is not None:
		l = int(crop_left or 0)
		r = int(crop_right or 0)
		if l + r < w:
			img = img[:, l : w - r]
			h, w = img.shape[:2]
	if target_size:
		tw, th = target_size
		if (w, h) != (tw, th):
			img = cv.resize(img, (tw, th), interpolation=cv.INTER_AREA)
	return img


def identify_hand_cards(
	frame_bgr: np.ndarray,
	*,
	crop_left: Optional[int] = None,
	crop_right: Optional[int] = None,
	target_width: int = 480,
	target_height: int = 854,
	deck_bottom_ratio: float = 0.22,
	threshold: float = 0.85,
	return_annotated: bool = False,
) -> Tuple[List[Dict], Optional[np.ndarray]]:
	"""Identify cards in hand using grayscale/2 template matching for all assets.

	- frame_bgr: input frame (BGR ndarray)
	- crop_left/right: optional cropping before resize
	- target_width/height: size normalization
	- deck_bottom_ratio: portion of frame height at bottom considered as deck ROI
	- threshold: TM_CCOEFF_NORMED threshold for a valid detection
	- return_annotated: if True, returns an annotated copy for debugging
	Returns: (cards, annotated_frame)
	  cards: list of dicts {slot: int, name: str, score: float, bbox: (x,y,w,h)}
	"""
	# Pull env defaults if present
	env_l = os.getenv("CROP_LEFT")
	env_r = os.getenv("CROP_RIGHT")
	crop_left = int(env_l) if crop_left is None and env_l else crop_left
	crop_right = int(env_r) if crop_right is None and env_r else crop_right

	img = preprocess_frame(
		frame_bgr,
		target_size=(target_width, target_height),
		crop_left=crop_left,
		crop_right=crop_right,
	)

	# 1) Deck ROI (bottom band of the frame)
	H, W = img.shape[:2]
	roi_h = max(40, int(H * float(deck_bottom_ratio)))
	y0 = H - roi_h
	deck_roi = img[y0:H, :]

	# 2) Grayscale/2 pipeline: downscale ROI by 2 then grayscale
	deck_small = cv.resize(deck_roi, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
	match_img = cv.cvtColor(deck_small, cv.COLOR_BGR2GRAY)

	# 3) Load templates once: grayscale/2 for all assets in assets/cards
	templates = _get_templates_gray_div2()

	method = cv.TM_CCOEFF_NORMED

	# 4) Match all templates and keep best per slot bin (4 bins across width)
	bins = 4
	bin_w = match_img.shape[1] / float(bins)
	best_per_bin: List[Optional[Dict]] = [None] * bins

	for name, tpl in templates:
		th, tw = tpl.shape[:2]
		if th > match_img.shape[0] or tw > match_img.shape[1]:
			continue
		res = cv.matchTemplate(match_img, tpl, method)
		min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
		if float(max_val) < float(threshold):
			continue
		cx = max_loc[0] + tw / 2.0
		bin_idx = int(cx // bin_w)
		bin_idx = max(0, min(bins - 1, bin_idx))
		det = {
			"name": name,
			"max_val": float(max_val),
			"max_loc": (int(max_loc[0]), int(max_loc[1])),
			"w": int(tw),
			"h": int(th),
		}
		cur = best_per_bin[bin_idx]
		if cur is None or det["max_val"] > cur["max_val"]:
			best_per_bin[bin_idx] = det

	cards: List[Dict] = []
	annotated: Optional[np.ndarray] = img.copy() if return_annotated else None

	# 5) Scale detections back to processed-frame coordinates and build outputs
	for i, det in enumerate(best_per_bin, start=1):
		if not det:
			continue
		x_small, y_small = det["max_loc"]
		w_small, h_small = det["w"], det["h"]
		# Scale from small ROI to full processed frame
		x = int(x_small * 2)
		y = int(y_small * 2) + y0
		w_box = int(w_small * 2)
		h_box = int(h_small * 2)
		cards.append({
			"slot": i,
			"name": det["name"],
			"score": det["max_val"],
			"bbox": (x, y, w_box, h_box),
		})
		if annotated is not None:
			cv.rectangle(annotated, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
			cv.putText(annotated, f"{det['name']}:{det['max_val']:.2f}", (x, max(0, y - 6)),
					   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

	# Sort cards by slot for stable output
	cards.sort(key=lambda d: d["slot"]) 

	return cards, annotated


# -----------------------
# Templates cache helpers
# -----------------------
_TEMPLATES_CACHE: Optional[List[Tuple[str, np.ndarray]]]= None


def _get_templates_gray_div2() -> List[Tuple[str, np.ndarray]]:
	global _TEMPLATES_CACHE
	if _TEMPLATES_CACHE is not None:
		return _TEMPLATES_CACHE

	# Project root is parent of src/
	root = pathlib.Path(__file__).resolve().parents[1]
	cards_dir = root / "assets" / "cards"
	templates: List[Tuple[str, np.ndarray]] = []
	for p in sorted(cards_dir.iterdir()):
		if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
			continue
		tpl_color = cv.imread(str(p), cv.IMREAD_COLOR)
		if tpl_color is None:
			continue
		small = cv.resize(tpl_color, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
		gray = cv.cvtColor(small, cv.COLOR_BGR2GRAY)
		templates.append((p.stem, gray))

	_TEMPLATES_CACHE = templates
	return templates


