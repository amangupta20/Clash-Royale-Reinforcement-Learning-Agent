import cv2 as cv
import numpy as np
import time

#game_image=cv.imread("test/images/2025-08-25-mob_11.jpg",cv.IMREAD_UNCHANGED)
game_image=cv.imread("test/images/deck.png",cv.IMREAD_REDUCED_COLOR_2)
knight=cv.imread("test/images/image.png", cv.IMREAD_REDUCED_COLOR_2)
start=time.time()
result= cv.matchTemplate(game_image, knight, cv.TM_CCOEFF_NORMED)
end=time.time()
duration_ms = (end - start) * 1000
print(f"Template matching took: {duration_ms:.2f} ms")
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

threshold = 0.4
if max_val >= threshold:
    print(f"Best match: {max_val} at location {max_loc}")
    
    cv.rectangle(game_image, max_loc, (max_loc[0] + knight.shape[1], max_loc[1] + knight.shape[0]), (0, 255, 0), 2)
    cv.imshow("Matched Result", game_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("No good match found.")