import cv2 as cv
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class UltraFastDeckMatcher:
    def __init__(self, deck=None):
        if deck is None:
            raise ValueError("Deck is required")
        self.deck = deck
        self.templates = self.load_templates()
        
        # Use fewer workers to reduce context switching overhead
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Aggressive OpenCV optimizations
        cv.setUseOptimized(True)
        cv.setNumThreads(1)
        
        # Pre-compute crop region for ultra-fast cropping
        self.CROP_LEFT = 783
        self.CROP_RIGHT = 720
        self.CROP_TOP = 845
        self.CROP_BOT = 46
        
        # Pre-allocate arrays to avoid memory allocation overhead
        self._temp_gray = None
        self._temp_resized = None

    def load_templates(self):
        templates = []
        for card in self.deck:
            template = cv.imread(f"assets/cards/{card}.png", cv.IMREAD_REDUCED_GRAYSCALE_2)
            if template is not None:
                templates.append((card, template))
        return templates

    def _ultra_fast_crop_and_process(self, frame):
        """Ultra-optimized single-pass crop, grayscale, and resize"""
        height, width = frame.shape[:2]
        
        # Direct crop without validation (for speed)
        cropped = frame[self.CROP_TOP:height-self.CROP_BOT, 
                       self.CROP_LEFT:width-self.CROP_RIGHT]
        
        # Single conversion: BGR → Gray
        gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
        
        # Resize with fastest interpolation
        return cv.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2), 
                        interpolation=cv.INTER_NEAREST)  # NEAREST is faster than AREA

    def _batch_match_templates(self, game_image, template_batch):
        """Match multiple templates in one function call"""
        results = []
        threshold = 0.8
        
        for card_name, template in template_batch:
            result = cv.matchTemplate(game_image, template, cv.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv.minMaxLoc(result)
            
            if max_val >= threshold:
                # Inline slot detection for maximum speed
                x = max_loc[0]
                if x < 40:
                    slot = 1
                elif x < 80:
                    slot = 2
                elif x < 120:
                    slot = 3
                elif x < 160:
                    slot = 4
                else:
                    slot = None
                    
                if slot is not None:
                    results.append((slot, card_name, max_val))
        
        return results

    def detect_slots_ultra_fast(self, frame):
        start = time.perf_counter()
        
        # Ultra-fast preprocessing in one shot
        game_image = self._ultra_fast_crop_and_process(frame)
        
        # Split templates into batches for reduced threading overhead
        batch_size = 2  # Process 2 templates per thread
        template_batches = [
            self.templates[i:i + batch_size] 
            for i in range(0, len(self.templates), batch_size)
        ]
        
        # Submit batched jobs
        futures = [
            self.executor.submit(self._batch_match_templates, game_image, batch)
            for batch in template_batches
        ]
        
        # Collect all candidates
        all_candidates = []
        for future in futures:
            all_candidates.extend(future.result())
        
        # Assign best match per slot
        detected_slots = {1: None, 2: None, 3: None, 4: None}
        all_candidates.sort(key=lambda x: x[2], reverse=True)
        
        for slot, card_name, confidence in all_candidates:
            if detected_slots[slot] is None:
                detected_slots[slot] = card_name
        
        end = time.perf_counter()
        print(f"ULTRA FAST slot detection took {(end-start)*1000:.1f}ms")
        return detected_slots

# Keep the old crop function for compatibility
def crop(frame):
    CROP_LEFT = 783
    CROP_RIGHT = 720
    CROP_TOP = 845
    CROP_BOT = 46
    height, width = frame.shape[:2]
    
    if CROP_TOP >= height - CROP_BOT or CROP_LEFT >= width - CROP_RIGHT:
        return None
    
    return frame[CROP_TOP:height-CROP_BOT, CROP_LEFT:width-CROP_RIGHT]
