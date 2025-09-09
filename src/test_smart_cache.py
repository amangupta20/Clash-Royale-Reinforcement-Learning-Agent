import json
from template_matcher import DeckMatcher  # Use original without frame cache
import cv2 as cv
import threading
import time
from capture import DoubleBuffer, capture_thread

class SmartResultCache:
    def __init__(self, deck_matcher):
        self.deck_matcher = deck_matcher
        self.last_result = None
        self.last_detection_time = 0
        self.cache_duration = 2.0
        self.result_history = []
        self.stability_threshold = 3
        self.cache_hits = 0
        self.total_calls = 0
    
    def detect_with_smart_caching(self, frame):
        self.total_calls += 1
        current_time = time.time()
        
        # Check if we have a recent cached result
        if (self.last_result is not None and 
            current_time - self.last_detection_time < self.cache_duration):
            self.cache_hits += 1
            print(f"SMART CACHE HIT: Using result from {current_time - self.last_detection_time:.1f}s ago")
            return self.last_result, True
        
        # Perform detection
        result = self.deck_matcher.detect_slots(frame)
        
        # Check if result is stable
        self.result_history.append(result)
        if len(self.result_history) > self.stability_threshold:
            self.result_history.pop(0)
        
        # If we have consistent results, cache them
        if (len(self.result_history) >= self.stability_threshold and 
            all(r == result for r in self.result_history)):
            self.last_result = result
            self.last_detection_time = current_time
            print(f"CACHING STABLE RESULT: {result}")
        
        return result, False

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

    # SMART RESULT CACHE ONLY TEST
    print("Testing SMART RESULT CACHE only...")
    deck_matcher = DeckMatcher(deck)  # Original without frame cache
    smart_cache = SmartResultCache(deck_matcher)
    
    processing_times = []
    try:
        while not stop_event.is_set():
            frame = buffer.read()
            if frame is not None:
                count += 1
                start = time.perf_counter()
                detected_slots, was_cached = smart_cache.detect_with_smart_caching(frame)
                end = time.perf_counter()
                
                processing_time = (end - start) * 1000  # ms
                tim += end - start
                processing_times.append(processing_time)
                
                print(f"Detected slots: {detected_slots}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping threads...")
        stop_event.set()
    
    capture.join()
    
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)
        cache_hit_rate = (smart_cache.cache_hits / smart_cache.total_calls) * 100
        print(f"SMART CACHE Performance:")
        print(f"  Average: {avg_time:.1f}ms")
        print(f"  Min: {min_time:.1f}ms")
        print(f"  Max: {max_time:.1f}ms")
        print(f"  Cache hit rate: {cache_hit_rate:.1f}%")
        print(f"  Total samples: {len(processing_times)}")
    
    print("Threads stopped.")

__main__()
