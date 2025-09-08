"""
Always-on Windows screen capture with on-demand snapshots.

Design goals
- Keep the capture session running to avoid start/stop overhead.
- Provide get_snapshot() that returns a NumPy ndarray (BGR) without touching disk.
- Do minimal work in the callback unless a snapshot is requested.

Notes
- This relies on the 'windows_capture' package used in HelperScripts.
- The Frame->ndarray conversion attempts several known attributes/methods.
- As a last resort, it can fall back to a temporary disk write (off by default).
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional, Tuple

import numpy as np

try:
	# Same library used in HelperScripts/windows-capture-with required resolution.py
	from windows_capture import WindowsCapture, Frame, InternalCaptureControl
except Exception as _e:  # pragma: no cover
	WindowsCapture = None  # type: ignore
	Frame = object  # type: ignore
	InternalCaptureControl = object  # type: ignore


class _SnapshotState:
	"""Holds synchronization primitives and the last requested frame."""

	def __init__(self) -> None:
		self.request_event = threading.Event()
		self.result_event = threading.Event()
		self.lock = threading.Lock()
		self.latest_frame: Optional[np.ndarray] = None
		self.latest_ts: float = 0.0
		self.error: Optional[str] = None

	def publish(self, frame_bgr: np.ndarray) -> None:
		with self.lock:
			# Copy to decouple from capture buffer
			self.latest_frame = frame_bgr.copy()
			self.latest_ts = time.time()


def _frame_to_ndarray_bgr(frame, allow_disk_fallback: bool = False) -> Optional[np.ndarray]:
	"""Best-effort conversion of windows_capture.Frame to np.ndarray (BGR).

	Tries common attributes/methods seen in WindowsCapture variants. If none work,
	optionally falls back to a temporary file write+read (disabled by default).
	"""

	# 1) Direct ndarray access
	for attr in ("as_numpy", "to_ndarray", "ndarray", "image", "bgr"):
		try:
			if hasattr(frame, attr):
				obj = getattr(frame, attr)
				arr = obj() if callable(obj) else obj
				if isinstance(arr, np.ndarray):
					# Try to ensure 3-channel BGR
					if arr.ndim == 3 and arr.shape[2] in (3, 4):
						if arr.shape[2] == 4:
							# BGRA -> BGR (drop alpha)
							return arr[:, :, :3].copy()
						return arr.copy()
		except Exception:
			pass

	# 2) Raw buffer heuristics (if available)
	for buf_attr, shape_attr in (("raw", "shape"), ("buffer", "shape")):
		try:
			if hasattr(frame, buf_attr) and hasattr(frame, shape_attr):
				buf = getattr(frame, buf_attr)
				shp = getattr(frame, shape_attr)
				if buf is not None and shp is not None:
					arr = np.frombuffer(buf, dtype=np.uint8)
					arr = arr.reshape(shp)
					if arr.ndim == 3 and arr.shape[2] in (3, 4):
						return arr[:, :, :3].copy()
		except Exception:
			pass

	# 3) Optional disk fallback for compatibility (discouraged for perf)
	if allow_disk_fallback and hasattr(frame, "save_as_image"):
		try:
			import cv2 as cv  # local import
			import tempfile

			with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
				tmp_path = tmp.name
			try:
				frame.save_as_image(tmp_path)  # type: ignore[attr-defined]
				arr = cv.imread(tmp_path, cv.IMREAD_COLOR)
				return arr
			finally:
				try:
					os.remove(tmp_path)
				except OSError:
					pass
		except Exception:
			pass

	return None


class AlwaysOnCapture:
	"""Always-on capture with on-demand snapshots.

	Usage:
		cap = AlwaysOnCapture(window_name="BlueStacks App Player 1")
		cap.start()
		frame = cap.get_snapshot(timeout=0.03)  # np.ndarray | None
		cap.stop()
	"""

	def __init__(self, window_name: Optional[str] = None) -> None:
		if WindowsCapture is None:
			raise RuntimeError("windows_capture is not available in this environment")

		env_window = os.getenv("WINDOW_NAME")
		self.window_name = window_name or env_window
		if not self.window_name:
			raise ValueError("WINDOW_NAME is not set. Pass window_name or set env WINDOW_NAME.")

		self._state = _SnapshotState()
		self._capture = WindowsCapture(
			cursor_capture=None,
			draw_border=None,
			monitor_index=None,
			window_name=self.window_name,
		)
		self._started = False
		self._closed = False

		# Register events using the library's event API directly
		def on_frame_arrived(frame, capture_control):
			# Fast path: only act when a snapshot is requested
			if self._state.request_event.is_set() and not self._state.result_event.is_set():
				arr = _frame_to_ndarray_bgr(frame, allow_disk_fallback=False)
				if arr is not None:
					self._state.publish(arr)
					self._state.result_event.set()

		def on_closed():
			self._closed = True

		# Attach handlers
		try:
			self._capture.event(on_frame_arrived)
			self._capture.event(on_closed)
		except Exception:
			# If the library's event registration differs, we silently continue; get_snapshot will just never resolve
			pass

	def start(self) -> None:
		if self._started:
			return
		self._capture.start()
		self._started = True

	def stop(self) -> None:
		# Depending on windows_capture API, this may be a no-op if the source window closes.
		try:
			if hasattr(self._capture, "stop"):
				self._capture.stop()  # type: ignore[call-arg]
		finally:
			self._started = False

	def get_snapshot(self, timeout: float = 0.03) -> Optional[np.ndarray]:
		"""Request the next arriving frame and return it as ndarray (BGR).

		- timeout: seconds to wait for a fresh frame.
		Returns None on timeout or if conversion failed.
		"""
		if not self._started:
			raise RuntimeError("Capture is not started. Call start() first.")

		# Setup a fresh request
		self._state.result_event.clear()
		self._state.request_event.set()

		got = self._state.result_event.wait(timeout)
		# Clear the request flag regardless of outcome
		self._state.request_event.clear()

		if not got:
			return None
		with self._state.lock:
			if self._state.latest_frame is None:
				return None
			return self._state.latest_frame.copy()


# No decorator helpers needed; handlers are registered directly in __init__

