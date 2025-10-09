"""
Template Matching for Hand Cards (T005)

This module implements the TemplateCardMatcher class for Phase 0 of the Clash Royale RL Agent.
It uses OpenCV template matching to detect 8 deck cards in hand slots with high performance.

Performance Target: <4ms for all 4 card matches, >80% accuracy
"""

import time
import numpy as np
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Optional


class TemplateCardMatcher:
    """
    Template matching for hand card detection in Phase 0.
    
    This class uses OpenCV template matching with normalized cross-correlation
    to detect cards in the hand region. It's optimized for sub-4ms performance
    on all 4 card slots using parallel processing.
    
    Performance Target: <4ms for all matches, >80% accuracy
    """
    
    # Hand ROI coordinates based on main.py values for 1920x1080 resolution
    # CROP_LEFT = 788, CROP_RIGHT = 675, CROP_TOP = 893, CROP_BOT = 50
    CROP_LEFT = 788
    CROP_RIGHT = 675
    CROP_TOP = 893
    CROP_BOT = 50
    
    # Full hand ROI for extraction
    HAND_ROI = (CROP_LEFT, CROP_TOP, CROP_RIGHT, CROP_BOT)
    
    # Card slot x-coordinate ranges (scaled to hand ROI)
    SLOT_RANGES = [
        (0, 55),    # Slot 1
        (55, 110),  # Slot 2
        (110, 165), # Slot 3
        (165, 220)  # Slot 4
    ]
    
    def __init__(self, card_names: List[str]):
        """
        Initialize the template card matcher.
        
        Args:
            card_names: List of 8 card names in the deck
        """
        if not card_names or len(card_names) != 8:
            raise ValueError("Exactly 8 card names required for deck")
            
        self.card_names = card_names
        self.templates = self._load_templates()
        
        # Thread pool for parallel template matching
        self.executor = ThreadPoolExecutor(max_workers=len(self.card_names))
        
        # Optimize OpenCV for speed
        cv.setUseOptimized(True)
        cv.setNumThreads(1)
        
        # Confidence threshold for detection
        self.threshold = 0.7
        
    def _load_templates(self) -> List[Tuple[str, np.ndarray]]:
        """
        Load card templates from assets/cards/ directory.
        
        Returns:
            List of (card_name, template) tuples
        """
        templates = []
        for card_name in self.card_names:
            template_path = f"assets/cards/{card_name}.png"
            template = cv.imread(template_path, cv.IMREAD_REDUCED_GRAYSCALE_2)
            if template is not None:
                templates.append((card_name, template))
            else:
                print(f"Warning: Could not load template for {card_name}")
        return templates
    
    def _which_slot(self, x: int) -> Optional[int]:
        """
        Determine which card slot a detected card belongs to based on x-coordinate.
        
        Args:
            x: X-coordinate of the detected card
            
        Returns:
            Slot number (1-4) or None if outside all slots
        """
        for i, (x_min, x_max) in enumerate(self.SLOT_RANGES):
            if x_min <= x < x_max:
                return i + 1
        return None
    
    def _match_single_template(self, game_image: np.ndarray, card_template: Tuple[str, np.ndarray]) -> Tuple[str, float, int]:
        """
        Perform template matching for a single card.
        
        Args:
            game_image: Preprocessed hand region image
            card_template: (card_name, template) tuple
            
        Returns:
            (card_name, max_val, x_coord) tuple
        """
        card_name, template = card_template
        
        # Use normalized cross-correlation for template matching
        result = cv.matchTemplate(game_image, template, cv.TM_CCOEFF_NORMED)
        
        # Find the location with the highest correlation
        _, max_val, _, max_loc = cv.minMaxLoc(result)
        
        return card_name, max_val, max_loc[0]
    
    def detect_hand_cards(self, frame: np.ndarray) -> Dict[int, Dict[str, any]]:
        """
        Detect cards in the hand ROI using template matching.
        
        Args:
            frame: Full screen frame in BGR format
            
        Returns:
            Dictionary mapping slot numbers to detection results:
            {
                slot_number: {
                    'card_id': str or None,
                    'confidence': float,
                    'position': (x, y) or None
                },
                ...
            }
        """
        overall_start = time.perf_counter()
        
        # Extract hand ROI
        x, y, width, height = self.HAND_ROI
        hand_roi = frame[y:y+height, x:x+width]
        
        # Initialize results dictionary
        results = {}
        for i in range(1, 5):
            results[i] = {'card_id': None, 'confidence': 0.0, 'position': None}
        
        # Preprocess hand ROI
        preprocess_start = time.perf_counter()
        gray = cv.cvtColor(hand_roi, cv.COLOR_BGR2GRAY)
        game_image = cv.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv.INTER_AREA)
        preprocess_time = (time.perf_counter() - preprocess_start) * 1000
        
        # Parallel template matching
        matching_start = time.perf_counter()
        futures = [
            self.executor.submit(self._match_single_template, game_image, template)
            for template in self.templates
        ]
        
        # Collect candidates above threshold
        candidates = []
        for future in futures:
            card_name, max_val, x_coord = future.result()
            if max_val >= self.threshold:
                slot = self._which_slot(x_coord)
                if slot is not None:
                    candidates.append((slot, card_name, max_val, x_coord))
        
        matching_time = (time.perf_counter() - matching_start) * 1000
        
        # Sort candidates by confidence and assign to slots
        assignment_start = time.perf_counter()
        candidates.sort(key=lambda x: x[2], reverse=True)  # Sort by confidence
        
        for slot, card_name, confidence, x_coord in candidates:
            if results[slot]['card_id'] is None:  # Slot not yet assigned
                # Scale coordinates back to original frame
                original_x = x * 2 + x  # Scale back from half resolution
                original_y = y + (height // 4)  # Approximate y position in hand ROI
                results[slot] = {
                    'card_id': card_name,
                    'confidence': confidence,
                    'position': (original_x, original_y)
                }
        
        assignment_time = (time.perf_counter() - assignment_start) * 1000
        total_time = (time.perf_counter() - overall_start) * 1000
        
        # Print timing metrics
        print(f"TEMPLATE MATCHING TIMING:")
        print(f"  Preprocessing: {preprocess_time:.2f}ms")
        print(f"  Template matching: {matching_time:.2f}ms")
        print(f"  Slot assignment: {assignment_time:.2f}ms")
        print(f"  TOTAL: {total_time:.2f}ms")
        
        return results
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for the template matcher.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            'threshold': self.threshold,
            'num_templates': len(self.templates),
            'target_time_ms': 4.0,
            'target_accuracy': 0.8
        }