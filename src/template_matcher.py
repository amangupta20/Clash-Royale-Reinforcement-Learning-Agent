import cv2 as cv
import cardslot
class DeckMatcher:
    def __init__(self, deck=None):
        if deck is None:
            raise ValueError("Deck is required")
        self.deck = deck
        self.templates = self.load_templates()

    def load_templates(self):
        templates = []
        for card in self.deck:
            template = cv.imread(f"assets/cards/{card}.png", cv.IMREAD_REDUCED_GRAYSCALE_2)
            if template is not None:
                templates.append((card, template))
        return templates
    def detect_slots(self, frame):
        game_image = frame
         # Dictionary to hold detected cards in slots
        detected_slots = dict((i, None) for i in range(1, 5))
        for card in self.templates:
            result = cv.matchTemplate(game_image, card[1], cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            threshold = 0.6
            if max_val >= threshold:
                slot = cardslot.which_slot(max_loc)
                if detected_slots[slot] is None:
                    detected_slots[slot] = card[0]


        return detected_slots