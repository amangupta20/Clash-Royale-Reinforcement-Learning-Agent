import time
import numpy as np
import cv2 as cv
from typing import List, Dict, Tuple, Optional
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from paddleocr import PaddleOCR  # GPU-enabled PaddleOCR expected to be installed
#import paddle  # noqa: F401  (kept in case future device checks desired)
#paddle.set_device('gpu:0')


class MinimalPerception:
    """
    Minimal perception implementation for Phase 0.
    
    This class provides fast perception using OCR:
    - OCR for elixir extraction
    - OCR for tower health extraction
    
    Performance Targets:
    - >95% elixir OCR accuracy
    - >95% tower health OCR accuracy
    """
    
    # Elixir number ROI (from ocr.txt)
    ELIXIR_NUMBER_ROI = (816, 1030, 41, 28)  # elixer numer 816 1030, 816 1058, 857 1030,857 1058
    
    # Tower health bar ROIs (from ocr.txt)
    # Friendly towers
    FRIENDLY_TOWER_ROIS = [
        (778, 670, 48, 20),   # own princess tower left 778 670,778 690, 826 670,826 690
        (1097, 670, 45, 20),  # own princess tower right 1097 670, 1097 690,1142,670, 1142 690
        (941, 817, 57, 20),   # own king 941 817, 941 837,997 817, 998 837
    ]
    
    # Enemy towers
    ENEMY_TOWER_ROIS = [
        (778, 146, 48, 20),   # enemy princess tower left 778 146,778 166, 826 146, 826 166
        (1097, 146, 45, 20),  # enemy princess tower right 1097 146,1097 166, 1142 146, 1142 166
        (941, 17, 57, 20),    # enemy king 941 17, 941 37,997 17, 998 37
    ]
    
    def __init__(self, save_debug_images: bool = False):
        """Initialize the minimal perception components."""

        # Debug image saving
        self.save_debug_images = save_debug_images
        if self.save_debug_images:
            self.debug_dir = f"debug_ocr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(self.debug_dir, exist_ok=True)

        # Initialize PaddleOCR with ONNX Runtime
        self.ocr_reader = self._initialize_ocr_reader()

        # Shared executor for parallel OCR processing
        # 7 workers optimal for processing 6 towers + 1 elixir in parallel
        # ONNX Runtime is thread-safe, no locking needed
        self._ocr_executor = ThreadPoolExecutor(max_workers=7, thread_name_prefix="onnx_ocr")
        
        # Game time tracking
        self._match_start_time = None
        self._is_in_match = False

    def _initialize_ocr_reader(self):
        """Create a PaddleOCR 2.7.3 instance with ONNX Runtime for optimal performance.

        Uses ONNX models converted from PaddleOCR models for:
        - 1.45x faster inference (43ms vs 62ms per call)
        - Thread-safe execution (enables parallel processing)
        - 13x faster initialization (331ms vs 4458ms)
        
        Requires:
        - PaddleOCR 2.7.3 (not 3.x)
        - ONNX models in ./inference/{det,rec,cls}_onnx/model.onnx
        - Run setup_paddleocr2_onnx.ps1 to download and convert models
        """

        # ONNX configuration for optimal performance
        onnx_kwargs = dict(
            lang='en',
            use_angle_cls=False,  # PaddleOCR 2.x API
            show_log=False,
            use_onnx=True,  # Enable ONNX Runtime
            use_gpu=False,  # ONNX uses CPU (still faster than GPU Paddle)
            det_model_dir="inference/det_onnx/model.onnx",
            rec_model_dir="inference/rec_onnx/model.onnx",
            cls_model_dir="inference/cls_onnx/model.onnx",
        )

        try:
            print("Initializing PaddleOCR 2.7.3 with ONNX Runtime...")
            print("  Detection:     inference/det_onnx/model.onnx")
            print("  Recognition:   inference/rec_onnx/model.onnx")
            print("  Classification: inference/cls_onnx/model.onnx")
            
            reader = PaddleOCR(**onnx_kwargs)
            self.ocr_device = 'onnx_cpu'
            
            return reader
            
        except Exception as onnx_err:
            print(f"✗ ONNX initialization failed: {onnx_err}")
            print("\nTo fix:")
            print("  1. Ensure PaddleOCR 2.7.3 is installed (not 3.x)")
            print("  2. Run: .\\setup_paddleocr2_onnx.ps1")
            print("  3. Verify models exist in inference/{det,rec,cls}_onnx/")
            
            raise RuntimeError(
                "Failed to initialize PaddleOCR with ONNX Runtime.\n"
                "Run setup_paddleocr2_onnx.ps1 to download and convert ONNX models."
            ) from onnx_err

    def _run_ocr_with_timeout(self, image: np.ndarray, timeout_seconds: float = 5.0):
        """Invoke PaddleOCR with ONNX Runtime (thread-safe, no lock needed)."""

        ocr_image = self._ensure_three_channel(image)
        # PaddleOCR 2.x API: ocr(image, det=False, rec=True, cls=False)
        future = self._ocr_executor.submit(
            self.ocr_reader.ocr, 
            ocr_image,
            det=False,  # Recognition only (we have pre-cropped ROIs)
            rec=True,
            cls=False
        )
        try:
            return future.result(timeout_seconds)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise TimeoutError("OCR operation timed out") from exc

    @staticmethod
    def _ensure_three_channel(image: np.ndarray) -> np.ndarray:
        """Ensure the image passed to PaddleOCR has 3 color channels and is memory-contiguous."""

        if image is None:
            raise ValueError("OCR image is None")

        # Make sure data is uint8
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        if image.ndim == 2:
            image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        
        # CRITICAL: Ensure the array is C-contiguous for maximum performance
        # Non-contiguous arrays (e.g., from slicing) cause massive slowdowns in PaddleOCR
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        
        return image
        
    def detect_elixir(self, frame: np.ndarray, use_ocr: bool = True) -> int:
        """
        Detect current elixir count using OCR.
        
        Args:
            frame: Full screen frame in BGR format
            use_ocr: Parameter kept for compatibility but always uses OCR
            
        Returns:
            Integer elixir count (0-10)
        """
        # Always use OCR now
        return self._detect_elixir_ocr(frame)
    
    def _detect_elixir_ocr(self, frame: np.ndarray) -> int:
        """
        Detect current elixir count using OCR (PaddleOCR).
        
        Args:
            frame: Full screen frame in BGR format
            
        Returns:
            Integer elixir count (0-10)
        """
        overall_start = time.perf_counter()
        
        # Extract elixir number ROI (raw, no preprocessing)
        x, y, width, height = self.ELIXIR_NUMBER_ROI
        elixir_roi = frame[y:y+height, x:x+width]
        
        # Save debug images
        if self.save_debug_images:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            cv.imwrite(os.path.join(self.debug_dir, f"elixir_original_{timestamp}.png"), elixir_roi)
        
        # Use PaddleOCR to extract text with timeout
        try:
            ocr_start = time.perf_counter()

            # PaddleOCR expects BGR color images - use raw ROI directly
            results = self._run_ocr_with_timeout(elixir_roi, timeout_seconds=1)

            ocr_time = (time.perf_counter() - ocr_start) * 1000
            
            # Parse PaddleOCR 2.x result format: [[[bbox, (text, confidence)], ...], ...]
            if results and results[0]:
                # results[0] contains all text detections for this image
                detections = results[0]
                
                if isinstance(detections, list) and len(detections) > 0:
                    # Get first detection (should be only one for our ROI)
                    first_detection = detections[0]
                    
                    # Extract text and confidence from detection
                    if isinstance(first_detection, tuple) and len(first_detection) == 2:
                        # Format: (text, confidence)
                        text, confidence = first_detection
                    elif isinstance(first_detection, list) and len(first_detection) >= 2:
                        # Format: [bbox, (text, confidence)]
                        text_info = first_detection[1]
                        if isinstance(text_info, tuple) and len(text_info) == 2:
                            text, confidence = text_info
                        else:
                            print(f"Elixir: Unexpected text format: {text_info}")
                            return 0
                    else:
                        print(f"Elixir: Unexpected detection format: {first_detection}")
                        return 0
                    
                    print(f"Elixir OCR: text='{text}', conf={confidence:.3f} ({ocr_time:.1f}ms)")
                    
                    # CRITICAL: Reject if text contains any letters (OCR confusion like 'ZSOE', 'ZSO8')
                    has_letters = any(c.isalpha() for c in str(text))
                    
                    if has_letters:
                        print(f"  Rejected: Contains letters: '{text}'")
                    else:
                        # Extract only digits
                        clean_text = ''.join(c for c in str(text) if c.isdigit())
                        if clean_text:
                            elixir = int(clean_text)
                            
                            # Validate range [0-10]
                            if elixir < 0 or elixir > 10:
                                print(f"  Rejected: {elixir} out of range [0-10]")
                            else:
                                processing_time = (time.perf_counter() - overall_start) * 1000
                                print(f"ELIXIR OCR SUCCESS: {elixir}/10 ({processing_time:.1f}ms total)")
                                return elixir
                        else:
                            print(f"  No digits found in '{text}'")
            
            print(f"No valid OCR results for elixir")
                   
        except TimeoutError:
            print("Elixir OCR timed out after 5 seconds")
        except Exception as e:
            print(f"Elixir OCR error: {e}")
            import traceback
            traceback.print_exc()
        
        # If OCR fails, return 0
        processing_time = (time.perf_counter() - overall_start) * 1000
        print(f"ELIXIR OCR FAILED: 0/10 ({processing_time:.2f}ms)")
        
        return 0
    
    def detect_tower_health(self, frame: np.ndarray) -> Dict[str, List[int]]:
        """
        Detect tower health values using OCR.
        
        Args:
            frame: Full screen frame in BGR format
            
        Returns:
            Dictionary with tower health values:
            {
                'friendly': [king_health, princess1_health, princess2_health],
                'enemy': [king_health, princess1_health, princess2_health]
            }
        """
        overall_start = time.perf_counter()
        
        # Initialize results with correct sizes based on actual ROIs
        friendly_health = [0] * len(self.FRIENDLY_TOWER_ROIS)
        enemy_health = [0] * len(self.ENEMY_TOWER_ROIS)
        
        # Define tower names for debug saving
        friendly_names = ["friendly_princess_left", "friendly_princess_right", "friendly_king"]
        enemy_names = ["enemy_princess_left", "enemy_princess_right", "enemy_king_top"]
        
        # Extract all tower ROIs at once for batch processing
        all_tower_rois = []
        all_tower_names = []
        tower_indices = []  # Track which result goes where: (is_friendly, index)
        
        for i, roi in enumerate(self.FRIENDLY_TOWER_ROIS):
            x, y, width, height = roi
            tower_roi = frame[y:y+height, x:x+width]
            # Ensure memory is contiguous for faster GPU transfer
            tower_roi = np.ascontiguousarray(self._ensure_three_channel(tower_roi))
            all_tower_rois.append(tower_roi)
            all_tower_names.append(friendly_names[i])
            tower_indices.append(('friendly', i))
        
        for i, roi in enumerate(self.ENEMY_TOWER_ROIS):
            x, y, width, height = roi
            tower_roi = frame[y:y+height, x:x+width]
            # Ensure memory is contiguous for faster GPU transfer
            tower_roi = np.ascontiguousarray(self._ensure_three_channel(tower_roi))
            all_tower_rois.append(tower_roi)
            all_tower_names.append(enemy_names[i])
            tower_indices.append(('enemy', i))
        
        # Save debug images if requested
        if self.save_debug_images:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            for name, roi_img in zip(all_tower_names, all_tower_rois):
                cv.imwrite(os.path.join(self.debug_dir, f"{name}_original_{timestamp}.png"), roi_img)
        
        # Process all towers in PARALLEL (ONNX Runtime is thread-safe)
        # Submit all OCR tasks to executor and collect results
        from concurrent.futures import as_completed
        
        futures = {}
        for tower_roi, tower_name, (tower_type, tower_idx) in zip(all_tower_rois, all_tower_names, tower_indices):
            future = self._ocr_executor.submit(self._extract_tower_health_from_roi, tower_roi, tower_name)
            futures[future] = (tower_type, tower_idx)
        
        # Collect results as they complete
        for future in as_completed(futures):
            tower_type, tower_idx = futures[future]
            try:
                health = future.result()
                if tower_type == 'friendly':
                    friendly_health[tower_idx] = health
                else:
                    enemy_health[tower_idx] = health
            except Exception as e:
                print(f"Error extracting tower health: {e}")
                # Leave health at 0 if extraction fails
        
        processing_time = (time.perf_counter() - overall_start) * 1000
        print(f"TOWER HEALTH OCR (PARALLEL): {processing_time:.2f}ms (avg {processing_time/len(all_tower_rois):.1f}ms per tower)")
        print(f"  Expected speedup: 3.84x vs sequential (~350ms → ~91ms)")
        
        return {
            'friendly': friendly_health,
            'enemy': enemy_health
        }
    
    def _extract_tower_health_from_roi(self, tower_roi: np.ndarray, tower_name: str = "tower") -> int:
        """Extract health from preprocessed tower ROI using ONNX Runtime."""
        try:
            ocr_start = time.perf_counter()
            results = self._run_ocr_with_timeout(tower_roi, timeout_seconds=5)
            ocr_time = (time.perf_counter() - ocr_start) * 1000
            
            # Parse PaddleOCR 2.x result format: [[[bbox, (text, confidence)], ...], ...]
            if results and results[0]:
                # results[0] contains all text detections for this image
                detections = results[0]
                
                if isinstance(detections, list) and len(detections) > 0:
                    # Get first detection (should be only one for our ROI)
                    first_detection = detections[0]
                    
                    if isinstance(first_detection, tuple) and len(first_detection) == 2:
                        # Format: (text, confidence)
                        text, confidence = first_detection
                    elif isinstance(first_detection, list) and len(first_detection) >= 2:
                        # Format: [bbox, (text, confidence)]
                        text_info = first_detection[1]
                        if isinstance(text_info, tuple) and len(text_info) == 2:
                            text, confidence = text_info
                        else:
                            print(f"{tower_name}: Unexpected text format: {text_info}")
                            return 0
                    else:
                        print(f"{tower_name}: Unexpected detection format: {first_detection}")
                        return 0
                    
                    # Extract digits only
                    clean_text = ''.join(c for c in str(text) if c.isdigit())
                    
                    if clean_text and len(clean_text) >= 3:
                        health = int(clean_text)
                        print(f"{tower_name} OCR: {health} ({ocr_time:.1f}ms, text='{text}', conf={confidence:.3f})")
                        return health
                    else:
                        print(f"{tower_name}: Too few digits: '{clean_text}' ({len(clean_text)} digits, need 3+)")
                        
        except Exception as e:
            print(f"{tower_name} OCR error: {e}")
        
        return 0
    
    def _extract_tower_health(self, frame: np.ndarray, roi: Tuple[int, int, int, int], tower_name: str = "tower") -> int:
        """
        Extract health value from a single tower health bar ROI using PaddleOCR.
        
        Args:
            frame: Full screen frame in BGR format
            roi: (x, y, width, height) of the health bar region
            tower_name: Name of the tower for debug image saving
            
        Returns:
            Integer health value (3-4 digit number)
        """
        x, y, width, height = roi
        
        # Expand ROI by 2 pixels in each direction (with bounds checking)
        padding = 2
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(frame.shape[1], x + width + padding)
        y_end = min(frame.shape[0], y + height + padding)
        
        tower_roi = frame[y_start:y_end, x_start:x_end]
        
        # Save debug images
        if self.save_debug_images:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            cv.imwrite(os.path.join(self.debug_dir, f"{tower_name}_original_{timestamp}.png"), tower_roi)
        
        # Use PaddleOCR to extract text with timeout
        try:
            ocr_start = time.perf_counter()

            # PaddleOCR expects BGR color images - use the expanded ROI directly
            results = self._run_ocr_with_timeout(tower_roi, timeout_seconds=5)

            ocr_time = (time.perf_counter() - ocr_start) * 1000
            print(f"PaddleOCR completed for {tower_name} in {ocr_time:.2f}ms")
            
            # Parse new PaddleOCR result structure (dict with 'rec_texts' and 'rec_scores')
            if results and len(results) > 0:
                result_dict = results[0]
                
                # Check if result is a dictionary with the new structure
                if isinstance(result_dict, dict) and 'rec_texts' in result_dict and 'rec_scores' in result_dict:
                    texts = result_dict['rec_texts']
                    scores = result_dict['rec_scores']
                    
                    print(f"{tower_name} OCR found {len(texts)} text(s): texts={texts}, scores={scores}")
                    
                    if texts and len(texts) > 0:
                        text = texts[0]
                        confidence = scores[0] if scores else 0.0
                        
                        print(f"  Processing: text='{text}', confidence={confidence:.4f}")
                        
                        # CRITICAL: Reject if text contains any letters (OCR confusion like 'ZSOE', 'ZSO8')
                        has_letters = any(c.isalpha() for c in str(text))
                        
                        if has_letters:
                            print(f"  Rejected: Contains letters (not pure digits)")
                        else:
                            # Clean text to extract only numbers
                            clean_text = ''.join(c for c in str(text) if c.isdigit())
                            if clean_text and len(clean_text) >= 3:  # Expect at least 3 digits for tower health
                                health = int(clean_text)
                                print(f"{tower_name} OCR SUCCESS: {health} (digits: '{clean_text}', text='{text}', conf={confidence:.4f})")
                                return health
                            else:
                                print(f"  Not enough digits: '{clean_text}' has only {len(clean_text)} digit(s), need 3+")
            
            print(f"No valid OCR results for {tower_name}")
                   
        except TimeoutError:
            print(f"Tower health OCR for {tower_name} timed out after 5 seconds")
        except Exception as e:
            print(f"Tower health OCR error for {tower_name}: {e}")
            import traceback
            traceback.print_exc()
        # If OCR fails, return 0 (tower destroyed or not visible)
        return 0
