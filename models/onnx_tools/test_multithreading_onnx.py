"""
Test ONNX OCR with Multi-threading
Checks if we can parallelize tower health OCR using ThreadPoolExecutor
"""

import cv2 as cv
import numpy as np
import time
from pathlib import Path
from paddleocr import PaddleOCR
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class ThreadSafePaddleOCRWithONNX:
    """Thread-safe PaddleOCR with ONNX backend"""
    
    def __init__(self, use_onnx: bool = True):
        """Initialize PaddleOCR with ONNX support"""
        print("=" * 60)
        print(f"Initializing Thread-Safe PaddleOCR (use_onnx={use_onnx})")
        print("=" * 60)
        
        init_params = {
            'lang': 'en',
            'use_angle_cls': False,
            'show_log': False,
        }
        
        if use_onnx:
            init_params.update({
                'use_onnx': True,
                'use_gpu': False,
                'det_model_dir': "inference/det_onnx/model.onnx",
                'rec_model_dir': "inference/rec_onnx/model.onnx",
                'cls_model_dir': "inference/cls_onnx/model.onnx",
            })
        else:
            init_params['use_gpu'] = True
            init_params['det'] = False
        
        start = time.perf_counter()
        self.ocr = PaddleOCR(**init_params)
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"Initialization: {elapsed:.2f}ms")
        print(f"Using ONNX: {use_onnx}")
        
        # Thread lock for safety (may not be needed with ONNX)
        self.lock = threading.Lock()
        print()
    
    def recognize(self, image: np.ndarray, use_lock: bool = True) -> tuple[str, float, float]:
        """
        Run OCR on image with optional thread locking
        Returns: (text, confidence, elapsed_ms)
        """
        if len(image.shape) == 2:
            image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        
        start = time.perf_counter()
        
        if use_lock:
            with self.lock:
                result = self.ocr.ocr(image, det=False, rec=True, cls=False)
        else:
            result = self.ocr.ocr(image, det=False, rec=True, cls=False)
        
        elapsed = (time.perf_counter() - start) * 1000
        
        # Parse result
        if result and result[0]:
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


def test_sequential_vs_parallel():
    """Compare sequential vs parallel OCR processing"""
    
    print("=" * 60)
    print("Multi-threading Test: Sequential vs Parallel OCR")
    print("=" * 60)
    print()
    
    # Create test images (simulating 6 tower health ROIs + 1 elixir)
    test_cases = []
    for i, digit_text in enumerate(["10", "3456", "999", "1234", "5", "87", "6543"]):
        img = np.ones((28, len(digit_text) * 20 + 10, 3), dtype=np.uint8) * 255
        cv.putText(img, digit_text, (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        test_cases.append((f"ROI_{i}", img, digit_text))
    
    print(f"Created {len(test_cases)} test images (simulating elixir + tower health)")
    print()
    
    # Test with ONNX
    print("-" * 60)
    print("Testing ONNX Backend")
    print("-" * 60)
    
    ocr_onnx = ThreadSafePaddleOCRWithONNX(use_onnx=True)
    
    # Sequential processing
    print("\n1. Sequential Processing (with lock):")
    start = time.perf_counter()
    sequential_results = []
    for name, image, expected in test_cases:
        text, conf, elapsed = ocr_onnx.recognize(image, use_lock=True)
        sequential_results.append((name, text, expected, elapsed))
        print(f"  {name}: '{text}' (expected: '{expected}') - {elapsed:.1f}ms")
    sequential_time = (time.perf_counter() - start) * 1000
    print(f"  Total: {sequential_time:.2f}ms")
    
    # Parallel processing with lock
    print("\n2. Parallel Processing (with lock, max_workers=7):")
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=7) as executor:
        futures = {executor.submit(ocr_onnx.recognize, img, True): (name, expected) 
                   for name, img, expected in test_cases}
        parallel_locked_results = []
        for future in as_completed(futures):
            name, expected = futures[future]
            text, conf, elapsed = future.result()
            parallel_locked_results.append((name, text, expected, elapsed))
            print(f"  {name}: '{text}' (expected: '{expected}') - {elapsed:.1f}ms")
    parallel_locked_time = (time.perf_counter() - start) * 1000
    print(f"  Total: {parallel_locked_time:.2f}ms")
    
    # Parallel processing WITHOUT lock (risky but may work with ONNX)
    print("\n3. Parallel Processing (NO lock, max_workers=7):")
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=7) as executor:
        futures = {executor.submit(ocr_onnx.recognize, img, False): (name, expected) 
                   for name, img, expected in test_cases}
        parallel_unlocked_results = []
        for future in as_completed(futures):
            name, expected = futures[future]
            try:
                text, conf, elapsed = future.result()
                parallel_unlocked_results.append((name, text, expected, elapsed))
                print(f"  {name}: '{text}' (expected: '{expected}') - {elapsed:.1f}ms")
            except Exception as e:
                print(f"  {name}: ERROR - {e}")
    parallel_unlocked_time = (time.perf_counter() - start) * 1000
    print(f"  Total: {parallel_unlocked_time:.2f}ms")
    
    # Test with different worker counts
    print("\n4. Testing Different Worker Counts (with lock):")
    for workers in [1, 2, 3, 4, 7]:
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(ocr_onnx.recognize, img, True): name 
                       for name, img, _ in test_cases}
            for future in as_completed(futures):
                future.result()
        elapsed = (time.perf_counter() - start) * 1000
        speedup = sequential_time / elapsed
        print(f"  Workers={workers}: {elapsed:.2f}ms (speedup: {speedup:.2f}x)")
    
    # Summary
    print("\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)
    print(f"Sequential (baseline):        {sequential_time:.2f}ms")
    print(f"Parallel with lock (7):       {parallel_locked_time:.2f}ms")
    print(f"Parallel without lock (7):    {parallel_unlocked_time:.2f}ms")
    print()
    
    if parallel_locked_time < sequential_time:
        speedup = sequential_time / parallel_locked_time
        print(f"✓ Parallel speedup: {speedup:.2f}x faster")
        print(f"✓ Time saved: {sequential_time - parallel_locked_time:.2f}ms per frame")
    else:
        slowdown = parallel_locked_time / sequential_time
        print(f"✗ Parallel is slower: {slowdown:.2f}x")
        print("  Likely due to GIL or thread contention")
    
    if parallel_unlocked_time < parallel_locked_time:
        print(f"\n✓ Removing lock helps: {parallel_locked_time - parallel_unlocked_time:.2f}ms saved")
        print("  ONNX Runtime appears thread-safe!")
    else:
        print(f"\n✗ Lock is necessary for thread safety")
    
    print("\n" + "=" * 60)
    print("Recommendation")
    print("=" * 60)
    
    best_time = min(sequential_time, parallel_locked_time, parallel_unlocked_time)
    if best_time == sequential_time:
        print("✓ Use SEQUENTIAL processing")
        print("  Multi-threading doesn't help due to Python GIL")
    elif best_time == parallel_locked_time:
        print("✓ Use PARALLEL with LOCK")
        print(f"  Speedup: {sequential_time / parallel_locked_time:.2f}x")
    else:
        print("✓ Use PARALLEL without LOCK")
        print(f"  Speedup: {sequential_time / parallel_unlocked_time:.2f}x")
        print("  ONNX Runtime is thread-safe!")


if __name__ == "__main__":
    test_sequential_vs_parallel()
