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
            template = cv.imread(f"assets/cards/{card}.png", cv.IMREAD_GRAYSCALE)
            if template is not None:
                templates.append((card, template))
        return templates
    def detect_slots(self, frame):
        crop_frame = crop(frame)
        game_image = cv.cvtColor(crop_frame, cv.COLOR_BGR2GRAY)
        if game_image is None:
            print("Error: Failed to crop frame, returning empty slot detection")
            return dict((i, None) for i in range(1, 5))
            
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
    CROP_LEFT = 783
    CROP_RIGHT = 720
    CROP_TOP=845
    CROP_BOT=46
    height, width = frame.shape[:2]
    
    print(f"Frame dimensions: {width}x{height}")
    print(f"Crop coordinates: top={CROP_TOP}, bottom={height-CROP_BOT}, left={CROP_LEFT}, right={width-CROP_RIGHT}")
    
    # Validate crop coordinates
    if CROP_TOP >= height - CROP_BOT:
        print(f"Error: Invalid vertical crop range. CROP_TOP({CROP_TOP}) >= height-CROP_BOT({height-CROP_BOT})")
        return None
    if CROP_LEFT >= width - CROP_RIGHT:
        print(f"Error: Invalid horizontal crop range. CROP_LEFT({CROP_LEFT}) >= width-CROP_RIGHT({width-CROP_RIGHT})")
        return None
    
    cropped_image = frame[CROP_TOP:height-CROP_BOT, CROP_LEFT:width-CROP_RIGHT]
    
    cv.imwrite("cropped.png", cropped_image)
    print("saved cropped.png")
    return cropped_image

    return detected_slots