"""
Card identification helpers operating on in-memory frames.

- Accept frames as np.ndarray (BGR) directly, no disk I/O.
- Perform cropping/resizing inside the function to meet time-critical needs.
- Provide a stub identify_hand_cards() that you can extend with template matching.
"""

from __future__ import annotations

import os
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
	return_annotated: bool = False,
) -> Tuple[List[Dict], Optional[np.ndarray]]:
	"""Identify cards in hand from the provided frame.

	- frame_bgr: input frame (BGR ndarray)
	- crop_left/right: optional cropping before resize
	- target_width/height: size normalization
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

	# Stub: grayscale + simple edge map to illustrate the flow; replace with template matching
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	# Example: dummy result (no cards) for now
	cards: List[Dict] = []

	annotated: Optional[np.ndarray] = None
	if return_annotated:
		annotated = img.copy()
		# Example overlay (replace with real detections)
		cv.putText(
			annotated,
			"identify_hand_cards: stub",
			(12, 28),
			cv.FONT_HERSHEY_SIMPLEX,
			0.7,
			(0, 255, 255),
			2,
			cv.LINE_AA,
		)

	return cards, annotated

