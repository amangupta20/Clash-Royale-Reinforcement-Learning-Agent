import cv2 as cv
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor


class DeckMatcher:
    """Detects cards in a player's hand using template matching.

    This class is optimized for speed by using a thread pool to perform
    template matching in parallel.

    Attributes:
        deck: A list of card names in the player's deck.
        templates: A list of tuples, where each tuple contains a card name and
            its corresponding template image.
        executor: A ThreadPoolExecutor for parallel template matching.
        slot_ranges: A list of tuples defining the x-coordinate ranges for each
            card slot.
    """
    def __init__(self, deck: list[str] | None = None):
        if deck is None:
            raise ValueError("A deck of cards is required to initialize the DeckMatcher.")
        self.deck = deck
        self.templates = self.load_templates()

        # A thread pool is used to run the template matching for each card in parallel.
        # The number of workers is set to the number of cards in the deck for optimal performance.
        self.executor = ThreadPoolExecutor(max_workers=len(self.deck))

        # Optimize OpenCV for speed by enabling optimizations and setting the number of threads to 1.
        # This is done to avoid oversubscription, as we are already using a thread pool.
        cv.setUseOptimized(True)
        cv.setNumThreads(1)
        
        # Pre-allocate slot mapping ranges for faster slot detection.
        # These ranges correspond to the x-coordinates of the four card slots.
        self.slot_ranges = [(0, 55), (55, 110), (110, 165), (165, 220)]

    def load_templates(self) -> list[tuple[str, np.ndarray]]:
        """Loads the card templates from the assets folder.

        The templates are loaded as grayscale images and resized to half their
        original size for faster template matching.

        Returns:
            A list of tuples, where each tuple contains a card name and its
            corresponding template image.
        """
        templates = []
        for card in self.deck:
            template = cv.imread(f"assets/cards/{card}.png", cv.IMREAD_REDUCED_GRAYSCALE_2)
            if template is not None:
                templates.append((card, template))
        return templates
    def _fast_which_slot(self, x: int) -> int | None:
        """Determines which card slot a detected card belongs to.

        This is an optimized version that avoids the overhead of a function
        call.

        Args:
            x: The x-coordinate of the detected card.

        Returns:
            The slot number (1-4), or None if the coordinate is out of range.
        """
        if x < 55:
            return 1
        elif x < 110:
            return 2
        elif x < 165:
            return 3
        elif x < 220:
            return 4
        return None

    def _match_single_template(
        self, game_image: np.ndarray, card_template_pair: tuple[str, np.ndarray]
    ) -> tuple[str, float, int]:
        """Performs template matching for a single card.

        This function is designed to be executed in parallel by the thread pool.

        Args:
            game_image: The grayscale game image to search in.
            card_template_pair: A tuple containing the card name and its
                template image.

        Returns:
            A tuple containing the card name, the maximum correlation value, and
            the x-coordinate of the match.
        """
        card_name, template = card_template_pair
        # Use normalized cross-correlation for template matching
        result = cv.matchTemplate(game_image, template, cv.TM_CCOEFF_NORMED)
        # Find the location with the highest correlation
        _, max_val, _, max_loc = cv.minMaxLoc(result)
        # Return the card name, the maximum correlation value, and the x-coordinate of the match
        return card_name, max_val, max_loc[0]

    def detect_slots(self, hand_roi: np.ndarray) -> dict[int, str | None]:
        """Detects cards in the hand ROI.

        Args:
            hand_roi: The pre-extracted hand region in BGR format.

        Returns:
            A dictionary mapping slot numbers to detected card names.
        """
        overall_start = time.perf_counter()
        
        # Time preprocessing
        preprocess_start = time.perf_counter()
        
        # If the hand ROI is not valid, return an empty dictionary
        if hand_roi is None:
            return dict((i, None) for i in range(1, 5))
            
        # Preprocess the hand ROI for template matching.
        # Convert the image to grayscale and resize it to half its original size.
        gray = cv.cvtColor(hand_roi, cv.COLOR_BGR2GRAY)
        game_image = cv.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv.INTER_AREA)
        preprocess_end = time.perf_counter()
        preprocess_time = (preprocess_end - preprocess_start) * 1000
            
        # This dictionary will store the detected card for each slot.
        detected_slots = {1: None, 2: None, 3: None, 4: None}
        # The threshold for a successful match.
        threshold = 0.9

        # Submit the template matching tasks to the thread pool.
        matching_start = time.perf_counter()
        futures = [
            self.executor.submit(self._match_single_template, game_image, card_template)
            for card_template in self.templates
        ]
        
        # Collect the results from the thread pool.
        collection_start = time.perf_counter()
        candidates = []
        for future in futures:
            card_name, max_val, x_coord = future.result()
            # If the correlation is above the threshold, consider it a potential match.
            if max_val >= threshold:
                slot = self._fast_which_slot(x_coord)
                if slot is not None:
                    candidates.append((slot, card_name, max_val))
        collection_end = time.perf_counter()
        
        # Assign the best match to each slot.
        # The candidates are sorted by confidence (max_val) in descending order.
        assignment_start = time.perf_counter()
        candidates.sort(key=lambda x: x[2], reverse=True)
        for slot, card_name, confidence in candidates:
            # If the slot has not been assigned a card yet, assign the current card to it.
            if detected_slots[slot] is None:
                detected_slots[slot] = card_name
        assignment_end = time.perf_counter()
        
        overall_end = time.perf_counter()
        
        # Calculate timing breakdown
        matching_time = (collection_start - matching_start) * 1000
        collection_time = (collection_end - collection_start) * 1000
        assignment_time = (assignment_end - assignment_start) * 1000
        total_time = (overall_end - overall_start) * 1000
        
        print(f"TIMING BREAKDOWN:")
        print(f"  Preprocessing: {preprocess_time:.1f}ms")
        print(f"  Template matching: {matching_time:.1f}ms")
        print(f"  Result collection: {collection_time:.1f}ms")
        print(f"  Slot assignment: {assignment_time:.1f}ms")
        print(f"  TOTAL: {total_time:.1f}ms")
        
        return detected_slots

# crop() function removed - now handled by ROI extraction in main.py
