import cv2
import numpy as np
import time
import threading

from capture import FrameShare

def template_matching_thread(frame_share: FrameShare, stop_event: threading.Event):
    template = cv2.imread('archers.png', 0)
    if template is None:
        print("Error: Could not load template image 'archers.png'")
        stop_event.set()
        return
        
    w, h = template.shape[::-1]

    while not stop_event.is_set():
        frame_copy = None
        with frame_share.lock:
            if frame_share.latest_frame is not None:
                # Create a copy to avoid modifying the original frame while it's being updated
                frame_copy = frame_share.latest_frame.copy()
        
        if frame_copy is not None:
            gray_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            print(f"Max match value: {max_val}, Location: {max_loc}")

            # Draw a rectangle around the best match
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(frame_copy, top_left, bottom_right, (0, 255, 0), 2)

            cv2.imshow('Template Matching', frame_copy)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

        time.sleep(5)

    cv2.destroyAllWindows()