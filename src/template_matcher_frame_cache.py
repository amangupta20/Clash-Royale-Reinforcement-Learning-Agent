import cv2 as cv
import cardslot
import time
import numpy as np
import hashlib
from concurrent.futures import ThreadPoolExecutor

class FrameCachedDeckMatcher:
    def __init__(self, deck=None):
        if deck is None:
            raise ValueError("Deck is required")
        self.deck = deck
        self.templates = self.load_templates()
        self.executor = ThreadPoolExecutor(max_workers=8)
        cv.setUseOptimized(True)
        cv.setNumThreads(1)
        
        # ONLY frame preprocessing cache
        self._frame_cache = {}
        self._cache_max_size = 100

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

    def _get_frame_hash(self, frame):
        height, width = frame.shape[:2]
        CROP_LEFT, CROP_RIGHT, CROP_TOP, CROP_BOT = 783, 720, 845, 46
        deck_region = frame[CROP_TOP:height-CROP_BOT:8, CROP_LEFT:width-CROP_RIGHT:8]
        return hashlib.md5(deck_region.tobytes()).hexdigest()[:12]
    
    def _preprocess_frame_cached(self, frame):
        frame_hash = self._get_frame_hash(frame)
        
        if frame_hash in self._frame_cache:
            return self._frame_cache[frame_hash], True  # Return cache hit status
        
        cropped = crop(frame)
        if cropped is None:
            return None, False
            
        gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
        game_image = cv.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv.INTER_AREA)
        
        if len(self._frame_cache) >= self._cache_max_size:
            keys_to_remove = list(self._frame_cache.keys())[:20]
            for key in keys_to_remove:
                del self._frame_cache[key]
        
        self._frame_cache[frame_hash] = game_image
        return game_image, False

    def _match_single_template(self, game_image, card_template_pair):
        card_name, template = card_template_pair
        result = cv.matchTemplate(game_image, template, cv.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv.minMaxLoc(result)
        return card_name, max_val, max_loc[0]

    def detect_slots(self, frame):
        overall_start = time.perf_counter()
        
        preprocess_start = time.perf_counter()
        game_image, cache_hit = self._preprocess_frame_cached(frame)
        if game_image is None:
            return {1: None, 2: None, 3: None, 4: None}
        preprocess_end = time.perf_counter()
        
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
        
        candidates.sort(key=lambda x: x[2], reverse=True)
        for slot, card_name, confidence in candidates:
            if detected_slots[slot] is None:
                detected_slots[slot] = card_name
        
        overall_end = time.perf_counter()
        
        preprocess_time = (preprocess_end - preprocess_start) * 1000
        matching_time = (collection_start - matching_start) * 1000
        collection_time = (collection_end - collection_start) * 1000
        total_time = (overall_end - overall_start) * 1000
        
        cache_status = "FRAME CACHE HIT" if cache_hit else "NO CACHE"
        print(f"FRAME CACHING TEST - {cache_status}:")
        print(f"  Preprocessing: {preprocess_time:.1f}ms")
        print(f"  Template matching: {matching_time:.1f}ms")
        print(f"  Result collection: {collection_time:.1f}ms")
        print(f"  TOTAL: {total_time:.1f}ms")
        
        return detected_slots

def crop(frame):
    CROP_LEFT = 783
    CROP_RIGHT = 720
    CROP_TOP = 845
    CROP_BOT = 46
    height, width = frame.shape[:2]
    
    if CROP_TOP >= height - CROP_BOT or CROP_LEFT >= width - CROP_RIGHT:
        return None
    
    return frame[CROP_TOP:height-CROP_BOT, CROP_LEFT:width-CROP_RIGHT]
