import cv2 as cv
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class ROIBasedDeckMatcher:
    def __init__(self, deck=None):
        if deck is None:
            raise ValueError("Deck is required")
        self.deck = deck
        self.templates = self.load_templates()
        
        # Use 4 workers - one per slot
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # OpenCV optimizations
        cv.setUseOptimized(True)
        cv.setNumThreads(1)
        
        # Pre-compute crop region
        self.CROP_LEFT = 783
        self.CROP_RIGHT = 720
        self.CROP_TOP = 845
        self.CROP_BOT = 46
        
        # Calculate ROI sizes based on actual template dimensions
        if self.templates:
            # Get max template dimensions for ROI sizing
            max_template_h = max(tpl.shape[0] for _, tpl in self.templates)
            max_template_w = max(tpl.shape[1] for _, tpl in self.templates)
            
            # Add padding around templates for better matching
            roi_width = max_template_w + 10
            roi_height = max_template_h + 10
        else:
            roi_width, roi_height = 45, 60
        
        # Define slot ROIs with better spacing (based on original slot detection ranges)
        # Original ranges: 0-40, 40-80, 80-120, 120-160 (after downscaling by 2)
        self.slot_rois = {
            1: (0, 0, min(roi_width, 40), roi_height),      # x, y, width, height
            2: (35, 0, min(roi_width, 45), roi_height),     # Slight overlap for safety
            3: (75, 0, min(roi_width, 45), roi_height), 
            4: (115, 0, min(roi_width, 45), roi_height)
        }
        
        print(f"ROI setup: template_max={max_template_w}x{max_template_h}, roi_size={roi_width}x{roi_height}")
        print(f"Slot ROIs: {self.slot_rois}")

    def load_templates(self):
        templates = []
        for card in self.deck:
            template = cv.imread(f"assets/cards/{card}.png", cv.IMREAD_REDUCED_GRAYSCALE_2)
            if template is not None:
                templates.append((card, template))
        return templates

    def _crop_and_process(self, frame):
        """Fast crop, grayscale, and resize"""
        height, width = frame.shape[:2]
        cropped = frame[self.CROP_TOP:height-self.CROP_BOT, 
                       self.CROP_LEFT:width-self.CROP_RIGHT]
        gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
        return cv.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2), 
                        interpolation=cv.INTER_AREA)

    def _match_slot_roi(self, game_image, slot_num):
        """Match all templates within a specific slot ROI"""
        x, y, w, h = self.slot_rois[slot_num]
        
        # Ensure ROI doesn't exceed image bounds
        img_h, img_w = game_image.shape[:2]
        x = min(x, img_w - 1)
        y = min(y, img_h - 1)
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        # Extract ROI for this slot
        roi = game_image[y:y+h, x:x+w]
        
        best_match = None
        best_confidence = 0.0
        threshold = 0.7  # Lowered threshold for testing
        matches_found = 0
        
        # Match all templates against this ROI
        for card_name, template in self.templates:
            # Skip if template is larger than ROI
            if template.shape[0] > roi.shape[0] or template.shape[1] > roi.shape[1]:
                continue
                
            result = cv.matchTemplate(roi, template, cv.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv.minMaxLoc(result)
            
            if max_val >= threshold:
                matches_found += 1
                if max_val > best_confidence:
                    best_match = card_name
                    best_confidence = max_val
        
        # Debug info
        if matches_found > 0:
            print(f"  Slot {slot_num}: found {matches_found} matches, best={best_match} ({best_confidence:.3f})")
        
        return slot_num, best_match

    def detect_slots_roi_based(self, frame):
        start = time.perf_counter()
        
        # Preprocess frame once
        game_image = self._crop_and_process(frame)
        print(f"Processed image size: {game_image.shape}")
        
        # Submit one job per slot
        futures = [
            self.executor.submit(self._match_slot_roi, game_image, slot_num)
            for slot_num in range(1, 5)
        ]
        
        # Collect results
        detected_slots = {1: None, 2: None, 3: None, 4: None}
        for future in futures:
            slot_num, card_name = future.result()
            detected_slots[slot_num] = card_name
        
        end = time.perf_counter()
        print(f"ROI-based slot detection took {(end-start)*1000:.1f}ms")
        return detected_slots

# Keep compatibility
def crop(frame):
    CROP_LEFT = 783
    CROP_RIGHT = 720
    CROP_TOP = 845
    CROP_BOT = 46
    height, width = frame.shape[:2]
    
    if CROP_TOP >= height - CROP_BOT or CROP_LEFT >= width - CROP_RIGHT:
        return None
    
    return frame[CROP_TOP:height-CROP_BOT, CROP_LEFT:width-CROP_RIGHT]
