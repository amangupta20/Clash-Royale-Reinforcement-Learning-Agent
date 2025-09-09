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
        self.executor = ThreadPoolExecutor(max_workers=8)  # Match deck size
        # Optimize OpenCV for speed
        cv.setUseOptimized(True)
        cv.setNumThreads(1)  # Avoid oversubscription with ThreadPoolExecutor
        
        # Pre-allocate slot mapping ranges for faster slot detection
        self.slot_ranges = [(0, 40), (40, 80), (80, 120), (120, 160)]

    def load_templates(self):
        templates = []
        for card in self.deck:
            template = cv.imread(f"assets/cards/{card}.png", cv.IMREAD_REDUCED_GRAYSCALE_2)
            if template is not None:
                templates.append((card, template))
        return templates
    def _fast_which_slot(self, x):
        """Optimized slot detection without function call overhead"""
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
        """Single template matching for parallel execution"""
        card_name, template = card_template_pair
        result = cv.matchTemplate(game_image, template, cv.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv.minMaxLoc(result)
        return card_name, max_val, max_loc[0]  # Only return x coordinate

    def detect_slots(self, frame):
        start = time.time()
        
        # Single-pass preprocessing: crop → grayscale → resize
        cropped = crop(frame)
        if cropped is None:
            return dict((i, None) for i in range(1, 5))
            
        # Convert to grayscale and resize in one go
        gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
        game_image = cv.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv.INTER_AREA)
            
        # Dictionary to hold detected cards in slots
        detected_slots = {1: None, 2: None, 3: None, 4: None}
        threshold = 0.8
        
        # Submit all template matching jobs at once
        futures = [
            self.executor.submit(self._match_single_template, game_image, card_template)
            for card_template in self.templates
        ]
        
        # Process results as they complete and assign best matches
        candidates = []
        for future in futures:
            card_name, max_val, x_coord = future.result()
            if max_val >= threshold:
                slot = self._fast_which_slot(x_coord)
                if slot is not None:
                    candidates.append((slot, card_name, max_val))
        
        # Assign highest confidence match per slot
        candidates.sort(key=lambda x: x[2], reverse=True)
        for slot, card_name, confidence in candidates:
            if detected_slots[slot] is None:
                detected_slots[slot] = card_name
                # Early exit if all slots filled
                if all(v is not None for v in detected_slots.values()):
                    break
                
        end = time.time()
        print(f"Slot detection took {end-start:.4f} seconds")
        return detected_slots

def crop(frame):
    CROP_LEFT = 783
    CROP_RIGHT = 720
    CROP_TOP=845
    CROP_BOT=46
    height, width = frame.shape[:2]
        # Validate crop coordinates
    if CROP_TOP >= height - CROP_BOT:
        print(f"Error: Invalid vertical crop range. CROP_TOP({CROP_TOP}) >= height-CROP_BOT({height-CROP_BOT})")
        return None
    if CROP_LEFT >= width - CROP_RIGHT:
        print(f"Error: Invalid horizontal crop range. CROP_LEFT({CROP_LEFT}) >= width-CROP_RIGHT({width-CROP_RIGHT})")
        return None
    
    cropped_image = frame[CROP_TOP:height-CROP_BOT, CROP_LEFT:width-CROP_RIGHT]
    

    return cropped_image
