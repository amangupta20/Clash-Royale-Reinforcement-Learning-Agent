import json

from template_matcher import DeckMatcher
from capture import DoubleBuffer, capture_thread, create_frame_roi

import cv2 as cv
import numpy as np

import threading
import time

def extract_hand_roi(frame: np.ndarray) -> np.ndarray:
    """
    Extracts the region of interest (ROI) containing the player's hand from a game frame.

    The coordinates are hardcoded for a specific screen resolution and game layout.
    This function uses a zero-copy slicing operation for performance.

    Args:
        frame: The input game frame.

    Returns:
        The hand ROI as a NumPy array, or None if the coordinates are invalid.
    """
    # The crop values define the boundaries of the hand region.
    CROP_LEFT = 788
    CROP_RIGHT = 675
    CROP_TOP = 893
    CROP_BOT = 50
    
    height, width = frame.shape[:2]
    
    # Validate the crop coordinates to ensure they are within the frame boundaries.
    if CROP_TOP >= height - CROP_BOT or CROP_LEFT >= width - CROP_RIGHT:
        return None
    
    # Convert the crop values to ROI parameters (x, y, width, height).
    x = CROP_LEFT
    y = CROP_TOP
    roi_width = width - CROP_LEFT - CROP_RIGHT
    roi_height = height - CROP_TOP - CROP_BOT
    
    # Extract the ROI using a zero-copy view for efficiency.
    return create_frame_roi(frame, x, y, roi_width, roi_height)

def __main__():
    """
    The main entry point of the application.

    This function initializes the screen capture, loads the card deck,
    and enters a loop to detect cards in the player's hand.
    """

    # Load the deck of cards from the deck.json file.
    # The deck must contain exactly 8 cards.
    deck_path = "deck.json"
    deck=[]
    try:
        with open(deck_path, "r") as f:
            deck = json.load(f)
            if not len(deck) == 8:
                raise ValueError("Deck must contain exactly 8 cards.")
            print(f"The cards in the current deck are: {deck}")



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

    # Create a threading event to signal the capture thread to stop.
    stop_event = threading.Event()
    # Create a double buffer to store the captured frames.
    buffer = DoubleBuffer()
    count = 0
    tim = 0

    # Start the screen capture in a separate thread.
    capture = threading.Thread(target=capture_thread, args=(buffer, stop_event))
    capture.start()

    # Wait for the first frame to be captured before starting the processing loop.
    print("Waiting for the first frame...")
    while not stop_event.is_set():
        if buffer.read() is not None:
            print("First frame received!")
            break
        time.sleep(0.5)

    # Initialize the DeckMatcher with the loaded deck.
    deck_matcher = DeckMatcher(deck)

    try:
        # The main processing loop.
        while not stop_event.is_set():
            # Read a frame from the buffer.
            frame = buffer.read_copy()
            if frame is not None:
                # Extract the hand ROI from the frame.
                hand_roi = extract_hand_roi(frame)
                
                if hand_roi is not None:
                    # Print some debug information for the first frame.
                    if count == 0:
                        frame_info = buffer.get_frame_info()
                        print(f"Frame info: {frame_info}")
                        print(f"Hand ROI shape: {hand_roi.shape}")
                        print(f"ROI shares memory: {np.shares_memory(frame, hand_roi)}")
                
                    count += 1
                    start = time.time()
                    
                    # Detect the cards in the hand ROI.
                    detected_slots = deck_matcher.detect_slots(hand_roi)
                    
                    end = time.time()
                    tim += end - start
                    print(f"Detected slots: {detected_slots}")
                else:
                    print("Invalid crop coordinates for current frame")
                    
            # Wait for a short interval before processing the next frame.
            time.sleep(0.3)
    except KeyboardInterrupt:
        # Handle KeyboardInterrupt (Ctrl+C) to gracefully stop the application.
        print("Stopping threads...")
        print(f"Average processing time: {tim/count:.4f} seconds")
        stop_event.set()
    
    # Wait for the capture thread to finish.
    capture.join()
    print("Threads stopped.")


__main__()
