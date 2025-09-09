import cv2 as cv
import cardslot
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class DeckMatcher:
    def __init__(self, deck=None):
        if deck is None:
            raise ValueError("Deck is required")
        self.deck = deck
        self.templates = self.load_templates()
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # OpenCV optimizations
        cv.setUseOptimized(True)
        cv.setNumThreads(1)
        
        # Smart result caching only (no frame caching - proven ineffective)
        self.last_result = None
        self.last_detection_time = 0
        self.cache_duration = 2.0  # Cache results for 2 seconds
        self.result_history = []
        self.stability_threshold = 3  # Need 3 consistent results
        self.cache_hits = 0
        self.total_calls = 0

    def load_templates(self):
        templates = []
        for card in self.deck:
            template = cv.imread(f"assets/cards/{card}.png", cv.IMREAD_REDUCED_GRAYSCALE_2)
            if template is not None:
                templates.append((card, template))
        return templates
    
    def _fast_which_slot(self, x):
        if x < 40:
            return 1
        elif x < 80:
            return 2
        elif x < 120:
            return 3
        elif x < 160:
            return 4
        return None

    def _match_single_template(self, game_image, card_template_pair):
        card_name, template = card_template_pair
        result = cv.matchTemplate(game_image, template, cv.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv.minMaxLoc(result)
        return card_name, max_val, max_loc[0]

    def _detect_slots_internal(self, frame):
        """Internal detection without caching"""
        overall_start = time.perf_counter()
        
        # Preprocessing
        preprocess_start = time.perf_counter()
        cropped = crop(frame)
        if cropped is None:
            return {1: None, 2: None, 3: None, 4: None}
            
        gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
        game_image = cv.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv.INTER_AREA)
        preprocess_end = time.perf_counter()
        
        # Template matching
        detected_slots = {1: None, 2: None, 3: None, 4: None}
        threshold = 0.8
        
        matching_start = time.perf_counter()
        futures = [
            self.executor.submit(self._match_single_template, game_image, card_template)
            for card_template in self.templates
        ]
        
        collection_start = time.perf_counter()
        candidates = []
        for future in futures:
            card_name, max_val, x_coord = future.result()
            if max_val >= threshold:
                slot = self._fast_which_slot(x_coord)
                if slot is not None:
                    candidates.append((slot, card_name, max_val))
        collection_end = time.perf_counter()
        
        # Slot assignment
        candidates.sort(key=lambda x: x[2], reverse=True)
        for slot, card_name, confidence in candidates:
            if detected_slots[slot] is None:
                detected_slots[slot] = card_name
        
        overall_end = time.perf_counter()
        
        # Timing breakdown
        preprocess_time = (preprocess_end - preprocess_start) * 1000
        matching_time = (collection_start - matching_start) * 1000
        collection_time = (collection_end - collection_start) * 1000
        total_time = (overall_end - overall_start) * 1000
        
        print(f"TIMING BREAKDOWN:")
        print(f"  Preprocessing: {preprocess_time:.1f}ms")
        print(f"  Template matching: {matching_time:.1f}ms")
        print(f"  Result collection: {collection_time:.1f}ms")
        print(f"  TOTAL: {total_time:.1f}ms")
        
        return detected_slots

    def detect_slots(self, frame):
        """Main detection method with smart caching"""
        self.total_calls += 1
        current_time = time.time()
        
        # Check cache first
        if (self.last_result is not None and 
            current_time - self.last_detection_time < self.cache_duration):
            self.cache_hits += 1
            print(f"SMART CACHE HIT: {(current_time - self.last_detection_time):.1f}s ago (hit rate: {(self.cache_hits/self.total_calls)*100:.1f}%)")
            return self.last_result
        
        # Perform detection
        result = self._detect_slots_internal(frame)
        
        # Check result stability for caching
        self.result_history.append(result)
        if len(self.result_history) > self.stability_threshold:
            self.result_history.pop(0)
        
        # Cache stable results
        if (len(self.result_history) >= self.stability_threshold and 
            all(r == result for r in self.result_history)):
            self.last_result = result
            self.last_detection_time = current_time
            print(f"CACHING STABLE RESULT: {result}")
        
        return result

def crop(frame):
    CROP_LEFT = 783
    CROP_RIGHT = 720
    CROP_TOP = 845
    CROP_BOT = 46
    height, width = frame.shape[:2]
    
    if CROP_TOP >= height - CROP_BOT or CROP_LEFT >= width - CROP_RIGHT:
        return None
    
    return frame[CROP_TOP:height-CROP_BOT, CROP_LEFT:width-CROP_RIGHT]
