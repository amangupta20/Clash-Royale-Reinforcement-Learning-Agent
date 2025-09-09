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

        game_image = self.crop(frame)
         # Dictionary to hold detected cards in slots
        detected_slots = dict((i, None) for i in range(1, 5))
        for card in self.templates:
            result = cv.matchTemplate(game_image, card[1], cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            threshold = 0.8
            if max_val >= threshold:
                print(f"Detected {card[0]} with confidence {max_val} at location {max_loc}")
                 # Determine which slot the card belongs to 
                slot = cardslot.which_slot(max_loc)
                if slot is not None and detected_slots[slot] is None:
                    detected_slots[slot] = card[0]
        def crop(frame):
            CROP_LEFT = 657
            CROP_RIGHT = 657
            TARGET_WIDTH = 360
            TARGET_HEIGHT = 110
            CROP_TOP=500
            CROP_BOT=100
            height, width = frame.shape[:2]
            cropped_image = frame[CROP_TOP:height-CROP_BOT, CROP_LEFT:width-CROP_RIGHT]
            cv.imwrite("cropped.png", cropped_image)
            return cropped_image
        
            return detected_slots