"""
ONNX-based OCR helper (reference implementation).
This file is a moved copy of the original helper to centralize ONNX tools.
"""

import onnxruntime as ort
import numpy as np
import cv2 as cv

ONNX_AVAILABLE = True

class ONNXDigitOCR:
    """Fast ONNX-based OCR optimized for digit recognition."""

    def __init__(self, rec_model_path: str = None, det_model_path: str = None):
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime not installed. Install with: pip install onnxruntime")
        self.rec_model_path = rec_model_path
        self.det_model_path = det_model_path
        self._init_session()

    def _init_session(self):
        # Example session creation for recognition model
        if self.rec_model_path:
            self.rec_sess = ort.InferenceSession(self.rec_model_path, providers=['CPUExecutionProvider'])
        else:
            self.rec_sess = None

    def recognize_digits(self, image: np.ndarray) -> str:
        # Placeholder: preprocessing + run session + postprocess
        return ""

def export_paddleocr_to_onnx(output_dir: str = "./onnx_models"):
    print("To export PaddleOCR models to ONNX, run paddle2onnx as described in setup_paddleocr2_onnx.ps1")
