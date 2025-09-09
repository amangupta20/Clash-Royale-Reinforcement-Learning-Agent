import cv2 as cv
import time
import numpy as np

class DeckMatcher:
    def __init__(self, deck=None):
        if deck is None:
            raise ValueError("Deck is required")
        self.deck = deck
        self.templates = self.load_templates()
        
        # OpenCV optimizations
        cv.setUseOptimized(True)
        cv.setNumThreads(4)  # Use more threads since we're not using ThreadPoolExecutor
        
        # Pre-compute slot boundaries for vectorized operations
        self.slot_boundaries = np.array([40, 80, 120, 160])

    def load_templates(self):
        templates = []
        for card in self.deck:
            template = cv.imread(f"assets/cards/{card}.png", cv.IMREAD_REDUCED_GRAYSCALE_2)
            if template is not None:
                templates.append((card, template))
        return templates

    def _fast_which_slot(self, x):
        """Vectorized slot detection"""
        return int(np.searchsorted(self.slot_boundaries, x) + 1) if x < 160 else None

    def detect_slots(self, frame):
        start = time.time()
        
        # Optimized preprocessing pipeline
        cropped = self._fast_crop(frame)
        if cropped is None:
            return {1: None, 2: None, 3: None, 4: None}
        
        # Single operation: crop → grayscale → resize
        gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
        game_image = cv.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
        
        # Pre-allocated results
        detected_slots = {1: None, 2: None, 3: None, 4: None}
        threshold = 0.8
        best_scores = {1: 0, 2: 0, 3: 0, 4: 0}  # Track best score per slot
        
        # Sequential processing (faster than threading for small workloads)
        for card_name, template in self.templates:
            result = cv.matchTemplate(game_image, template, cv.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv.minMaxLoc(result)
            
            if max_val >= threshold:
                slot = self._fast_which_slot(max_loc[0])
                if slot is not None and max_val > best_scores[slot]:
                    detected_slots[slot] = card_name
                    best_scores[slot] = max_val
        
        end = time.time()
        print(f"Slot detection took {(end-start)*1000:.1f}ms")
        return detected_slots

    def _fast_crop(self, frame):
        """Optimized cropping with bounds checking"""
        h, w = frame.shape[:2]
        
        # Hard-coded optimized crop (remove variables)
        if h < 891 or w < 1503:  # 845+46 and 783+720
            return None
            
        return frame[845:h-46, 783:w-720]

def crop(frame):
    """Fallback crop function for compatibility"""
    CROP_LEFT = 783
    CROP_RIGHT = 720
    CROP_TOP = 845
    CROP_BOT = 46
    height, width = frame.shape[:2]
    
    if CROP_TOP >= height - CROP_BOT or CROP_LEFT >= width - CROP_RIGHT:
        return None
    
    return frame[CROP_TOP:height-CROP_BOT, CROP_LEFT:width-CROP_RIGHT]
