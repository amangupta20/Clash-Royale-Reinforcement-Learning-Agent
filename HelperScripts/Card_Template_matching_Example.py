import cv2 as cv
import numpy as np
import time
import os
from dotenv import load_dotenv
import cardslot
load_dotenv()

# Get environment variables
card_image_path = os.getenv("CARD_IMAGE_PATH")
game_state_image_path = os.getenv("GAME_STATE_IMAGE_PATH")

# Validate environment variables
if not card_image_path or not game_state_image_path:
    print("Error: Please set CARD_IMAGE_PATH and GAME_STATE_IMAGE_PATH in your .env file.")
    exit()

# Check if files exist
if not os.path.exists(card_image_path):
    print(f"Error: Card image not found at '{card_image_path}'")
    exit()
if not os.path.exists(game_state_image_path):
    print(f"Error: Game state image not found at '{game_state_image_path}'")
    exit()


# Load images using paths from .env
game_image = cv.imread(game_state_image_path, cv.IMREAD_REDUCED_GRAYSCALE_2)
knight = cv.imread(card_image_path, cv.IMREAD_REDUCED_GRAYSCALE_2)

# Check if images loaded correctly
if game_image is None:
    print(f"Error: Could not load game state image from '{game_state_image_path}'")
    exit()
if knight is None:
    print(f"Error: Could not load card image from '{card_image_path}'")
    exit()

start = time.time()
result= cv.matchTemplate(game_image, knight, cv.TM_CCOEFF_NORMED)
end=time.time()
duration_ms = (end - start) * 1000
print(f"Template matching took: {duration_ms:.2f} ms")
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

threshold = 0.6
if max_val >= threshold:
    print(f"Best match: {max_val} at location {max_loc}")
    print(f"Card slot: {cardslot.which_slot(max_loc)}")
    cv.rectangle(game_image, max_loc, (max_loc[0] + knight.shape[1], max_loc[1] + knight.shape[0]), (0, 255, 0), 2)
    cv.imshow("Matched Result", game_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("No good match found.")