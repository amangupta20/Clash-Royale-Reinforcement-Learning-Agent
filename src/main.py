import json

from template_matcher import DeckMatcher
from capture import DoubleBuffer, capture_thread, create_frame_roi

import cv2 as cv
import numpy as np

import threading
import time

def extract_hand_roi(frame: np.ndarray) -> np.ndarray:
    """
    Extract hand region using ROI instead of crop function.
    Converts your 4-side crop values to ROI coordinates.
    """
    # Your current crop values
    CROP_LEFT = 788
    CROP_RIGHT = 675
    CROP_TOP = 893
    CROP_BOT = 50
    
    height, width = frame.shape[:2]
    
    # Validate coordinates (same as your current validation)
    if CROP_TOP >= height - CROP_BOT or CROP_LEFT >= width - CROP_RIGHT:
        return None
    
    # Convert to ROI parameters
    x = CROP_LEFT
    y = CROP_TOP
    roi_width = width - CROP_LEFT - CROP_RIGHT   # 1920 - 788 - 675 = 457
    roi_height = height - CROP_TOP - CROP_BOT    # 1080 - 893 - 50 = 137
    
    # Extract ROI (zero-copy view)
    return create_frame_roi(frame, x, y, roi_width, roi_height)

def __main__():

    # loading card deck from the json file
    deck_path = "deck.json"
    deck=[]
    try:
        with open(deck_path, "r") as f:
            deck = json.load(f)
            if not len(deck) == 8:
                raise ValueError("Deck must contain exactly 8 cards.")
            print(f"the cards in the current deck are {deck}")



    except FileNotFoundError:
        print(f"File not found: {deck_path}")
        print("Exiting ...")
        return
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {deck_path} ")
        print("Exiting ...")
        return
    except ValueError as e:
        print(f"Error: {e}")
        print("Exiting ...")
        return
    stop_event = threading.Event()
    buffer = DoubleBuffer()
    count = 0
    tim = 0
    # Start the capture thread
    capture = threading.Thread(target=capture_thread, args=(buffer, stop_event))
    capture.start()

    # Wait for the first frame to be captured
    print("Waiting for the first frame...")
    while not stop_event.is_set():
        if buffer.read() is not None:
            print("First frame received!")
            break
        time.sleep(0.5)

    # Initialize the DeckMatcher with the loaded deck
    deck_matcher = DeckMatcher(deck)
    try:
        while not stop_event.is_set():
            # Use the optimized read_copy method for safe processing
            frame = buffer.read_copy()
            if frame is not None:
                # Extract hand ROI using zero-copy operation (replaces crop)
                hand_roi = extract_hand_roi(frame)
                
                if hand_roi is not None:
                    
                    
                    # Get frame info for debugging
                    if count == 0:
                        frame_info = buffer.get_frame_info()
                        print(f"Frame info: {frame_info}")
                        print(f"Hand ROI shape: {hand_roi.shape}")
                        print(f"ROI shares memory: {np.shares_memory(frame, hand_roi)}")
                
                    count += 1
                    start = time.time()
                    
                    # Pass ROI directly to matcher (Option 2 approach)
                    detected_slots = deck_matcher.detect_slots(hand_roi)
                    
                    end = time.time()
                    tim += end - start
                    print(f"Detected slots: {detected_slots}")
                else:
                    print("Invalid crop coordinates for current frame")
                    
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("Stopping threads...")
        print(f"average processing time: {tim/count:.4f} seconds")
        stop_event.set()
    
    capture.join()
    print("Threads stopped.")


__main__()
