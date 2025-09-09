import json
from template_matcher_ultra_fast import UltraFastDeckMatcher
import cv2 as cv
import threading
import time
from capture import DoubleBuffer, capture_thread

def __main__():
    # Loading card deck from the json file
    deck_path = "deck.json"
    deck = []
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

    # Initialize the ULTRA FAST DeckMatcher
    print("Testing ULTRA FAST template matching...")
    deck_matcher = UltraFastDeckMatcher(deck)
    
    processing_times = []
    try:
        while not stop_event.is_set():
            frame = buffer.read()
            if frame is not None:
                start_time = time.perf_counter()
                detected_slots = deck_matcher.detect_slots_ultra_fast(frame)
                end_time = time.perf_counter()
                
                processing_time = (end_time - start_time) * 1000  # Convert to ms
                processing_times.append(processing_time)
                
                print(f"Detected slots: {detected_slots}")
            time.sleep(5)
    except KeyboardInterrupt:
        print("Stopping threads...")
        stop_event.set()
    
    capture.join()
    
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)
        print(f"Performance summary:")
        print(f"  Average: {avg_time:.1f}ms")
        print(f"  Min: {min_time:.1f}ms") 
        print(f"  Max: {max_time:.1f}ms")
    
    print("Threads stopped.")

__main__()
