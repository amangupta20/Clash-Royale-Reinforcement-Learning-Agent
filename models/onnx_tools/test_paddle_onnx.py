"""
Test PaddleOCR with ONNX models
Uses PaddleOCR's built-in use_onnx=True parameter for faster inference
"""

import cv2 as cv
import numpy as np
import time
from pathlib import Path
from paddleocr import PaddleOCR

class PaddleOCRWithONNX:
    """PaddleOCR using ONNX backend for faster inference"""
    
    def __init__(self, use_onnx: bool = True):
        """Initialize PaddleOCR with ONNX support"""
        print("=" * 60)
        print(f"Initializing PaddleOCR (use_onnx={use_onnx})")
        print("=" * 60)
        
        # Path to converted ONNX model (PaddleOCR 2.x format)
        onnx_model_dir = "inference/rec_onnx"
        
        if use_onnx and not Path(onnx_model_dir).exists():
            print(f"Warning: ONNX model directory not found: {onnx_model_dir}")
            print("Run setup_paddleocr2_onnx.ps1 first to download and convert models")
            print("Falling back to standard PaddleOCR")
            use_onnx = False
        
        start = time.perf_counter()
        
        # Initialize PaddleOCR 2.x with ONNX backend
        # Based on working example from katacr project
        init_params = {
            'lang': 'en',  # English character set (includes digits)
            'use_angle_cls': False,  # No rotation detection
            'show_log': False,
        }
        
        if use_onnx:
            # ONNX models - all three required for use_onnx=True in PaddleOCR 2.x
            onnx_det_path = "inference/det_onnx/model.onnx"
            onnx_rec_path = "inference/rec_onnx/model.onnx"
            onnx_cls_path = "inference/cls_onnx/model.onnx"
            
            init_params.update({
                'use_onnx': True,
                'use_gpu': False,  # ONNX Runtime on CPU
                'det_model_dir': onnx_det_path,
                'rec_model_dir': onnx_rec_path,
                'cls_model_dir': onnx_cls_path,
            })
            print(f"ONNX Detection:     {onnx_det_path}")
            print(f"ONNX Recognition:   {onnx_rec_path}")
            print(f"ONNX Classification: {onnx_cls_path}")
        else:
            # Standard Paddle models - use GPU
            init_params['use_gpu'] = True
            init_params['det'] = False  # Only recognition needed
        
        self.ocr = PaddleOCR(**init_params)
        
        elapsed = (time.perf_counter() - start) * 1000
        print(f"Initialization time: {elapsed:.2f}ms")
        print(f"Using ONNX: {use_onnx}")
        print()
    
    def recognize(self, image: np.ndarray) -> tuple[str, float, float]:
        """
        Run OCR on image
        Returns: (text, confidence, elapsed_ms)
        """
        start = time.perf_counter()
        
        # PaddleOCR expects images in BGR format
        if len(image.shape) == 2:
            image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        
        # Run OCR (PaddleOCR 2.x uses ocr() method)
        result = self.ocr.ocr(image, det=False, rec=True, cls=False)
        
        elapsed = (time.perf_counter() - start) * 1000
        
        # Parse result
        if result and result[0]:
            # PaddleOCR returns: [[bbox, (text, confidence)], ...]
            # Since we disabled detection, we get direct recognition results
            if isinstance(result[0], dict):
                text = result[0].get('rec_texts', [''])[0]
                confidence = result[0].get('rec_scores', [0.0])[0]
            elif isinstance(result[0], list) and len(result[0]) > 0:
                if isinstance(result[0][0], tuple):
                    text, confidence = result[0][0]
                else:
                    text = str(result[0][0])
                    confidence = 0.0
            else:
                text = ''
                confidence = 0.0
        else:
            text = ''
            confidence = 0.0
        
        return text, confidence, elapsed


def benchmark_paddle_onnx():
    """Benchmark PaddleOCR with and without ONNX"""
    
    print("=" * 60)
    print("PaddleOCR ONNX Benchmark")
    print("=" * 60)
    print()
    
    # Create test images with digits
    test_cases = []
    for digit_text in ["10", "3456", "999", "1234", "5", "87", "6543"]:
        # Create image: white background, black text
        img = np.ones((28, len(digit_text) * 20 + 10, 3), dtype=np.uint8) * 255
        cv.putText(img, digit_text, (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        test_cases.append((img, digit_text))
    
    print(f"Created {len(test_cases)} test images")
    print()
    
    # Test with standard PaddleOCR
    print("-" * 60)
    print("Testing Standard PaddleOCR (without ONNX)")
    print("-" * 60)
    
    ocr_standard = PaddleOCRWithONNX(use_onnx=False)
    
    times_standard = []
    successes_standard = 0
    
    for image, expected in test_cases:
        text, confidence, elapsed = ocr_standard.recognize(image)
        times_standard.append(elapsed)
        
        success = expected in text or text in expected
        if success:
            successes_standard += 1
        
        status = "✓" if success else "✗"
        print(f"{status} Expected '{expected}', got '{text}' (conf: {confidence:.3f}, time: {elapsed:.1f}ms)")
    
    print()
    print(f"Success rate: {successes_standard}/{len(test_cases)} ({successes_standard/len(test_cases)*100:.1f}%)")
    print(f"Average time: {np.mean(times_standard):.2f}ms")
    print()
    
    # Test with ONNX PaddleOCR
    print("-" * 60)
    print("Testing PaddleOCR with ONNX Backend")
    print("-" * 60)
    
    try:
        ocr_onnx = PaddleOCRWithONNX(use_onnx=True)
        
        times_onnx = []
        successes_onnx = 0
        
        for image, expected in test_cases:
            text, confidence, elapsed = ocr_onnx.recognize(image)
            times_onnx.append(elapsed)
            
            success = expected in text or text in expected
            if success:
                successes_onnx += 1
            
            status = "✓" if success else "✗"
            print(f"{status} Expected '{expected}', got '{text}' (conf: {confidence:.3f}, time: {elapsed:.1f}ms)")
        
        print()
        print(f"Success rate: {successes_onnx}/{len(test_cases)} ({successes_onnx/len(test_cases)*100:.1f}%)")
        print(f"Average time: {np.mean(times_onnx):.2f}ms")
        print()
        
        # Comparison
        print("=" * 60)
        print("Performance Comparison")
        print("=" * 60)
        print(f"Standard PaddleOCR: {np.mean(times_standard):.2f}ms (success: {successes_standard}/{len(test_cases)})")
        print(f"ONNX PaddleOCR:     {np.mean(times_onnx):.2f}ms (success: {successes_onnx}/{len(test_cases)})")
        
        if np.mean(times_standard) > 0:
            speedup = np.mean(times_standard) / np.mean(times_onnx)
            print(f"Speedup:            {speedup:.2f}x faster with ONNX")
        
    except Exception as e:
        print(f"Error testing ONNX backend: {e}")
        print("ONNX test skipped")


if __name__ == "__main__":
    benchmark_paddle_onnx()
