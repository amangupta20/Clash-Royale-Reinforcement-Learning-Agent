import json
# Test the fast sequential version
from template_matcher_fast import DeckMatcher
import cv2 as cv
import threading
import time
from capture import DoubleBuffer, capture_thread

def __main__():
    # loading card deck from the json file
    deck_path = "deck.json"
    deck=[]
    try:
        with open(deck_path, "r") as f:
            deck = json.load(f)
            if not len(deck) == 8:
                raise ValueError("Deck must contain exactly 8 cards.")
                return 
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
    print("Testing FAST sequential template matching...")
    
    try:
        iteration = 0
        while not stop_event.is_set() and iteration < 20:  # Test 20 iterations
            frame = buffer.read()
            if frame is not None:
                detected_slots = deck_matcher.detect_slots(frame)
                print(f"Detected slots: {detected_slots}")
                iteration += 1
            time.sleep(2)  # Test every 2 seconds
    except KeyboardInterrupt:
        print("Stopping threads...")
        stop_event.set()
    
    capture.join()
    print("Threads stopped.")

__main__()
