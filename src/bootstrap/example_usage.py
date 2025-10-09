"""
Example Usage of Phase 0 Bootstrap Components

This script demonstrates how to use the Phase 0 bootstrap components:
- BootstrapCapture for screen capture
- TemplateCardMatcher for hand card detection
- MinimalPerception for elixir and tower health detection
"""

import json
import time
import threading
import numpy as np

from capture import DoubleBuffer, capture_thread
from template_matcher import TemplateCardMatcher
from minimal_perception import MinimalPerception
from state_builder import MinimalStateBuilder, StateVector
from concurrent.futures import ThreadPoolExecutor


def main():
    """Main function demonstrating Phase 0 component usage."""
    
    # Load deck configuration
    try:
        with open("deck.json", "r") as f:
            deck = json.load(f)
            if len(deck) != 8:
                raise ValueError("Deck must contain exactly 8 cards")
            print(f"Loaded deck: {deck}")
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"Error loading deck: {e}")
        return
    
    # Initialize Phase 0 components
    buffer = DoubleBuffer()
    stop_event = threading.Event()
    card_matcher = TemplateCardMatcher(deck)
    perception = MinimalPerception(save_debug_images=False)
    state_builder = MinimalStateBuilder(deck)
    
    # Start screen capture in a separate thread
    capture = threading.Thread(target=capture_thread, args=(buffer, stop_event))
    capture.start()
    
    # Wait for the first frame to be captured
    print("Waiting for the first frame...")
    while not stop_event.is_set():
        if buffer.read() is not None:
            print("First frame received!")
            # Start match timer when first frame is received
            perception.start_match_timer()
            break
        time.sleep(0.5)
    
    print("Phase 0 components initialized. Press Ctrl+C to stop.")
    
    try:
        frame_count = 0
        total_time = 0
        
        while not stop_event.is_set():
            # Grab a frame
            frame = buffer.read_copy()
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Process frame with all Phase 0 components in parallel
            start_time = time.perf_counter()
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit detection tasks
                cards_future = executor.submit(card_matcher.detect_hand_cards, frame)
                perception_future = executor.submit(perception.detect_all_perceptions, frame)
                
                # Collect results
                detected_cards = cards_future.result()
                elixir_count, tower_health, game_time = perception_future.result()
                
                # Build state vector
                state_vector = state_builder.build_state(
                    frame=frame,
                    detected_cards=detected_cards,
                    elixir_count=elixir_count,
                    tower_health=tower_health,
                    match_time=game_time
                )
           
            # Calculate processing time
            processing_time = (time.perf_counter() - start_time) * 1000
            frame_count += 1
            total_time += processing_time
            
            # Print results every 10 frames
            if frame_count % 10 == 0:
                print(f"\n--- Frame {frame_count} ---")
                print(f"Processing time: {processing_time:.2f}ms (avg: {total_time/frame_count:.2f}ms)")
                print(f"Elixir: {elixir_count}/10")
                print(f"Game time: {game_time:.1f}s")
                print(f"Hand cards: {detected_cards}")
                print(f"Friendly towers: {tower_health['friendly']}")
                print(f"Enemy towers: {tower_health['enemy']}")
                
                # Print state vector summary
                global_features = state_vector.get_global_features()
                hand_features = state_vector.get_hand_features()

                print(f"State vector shape: {state_vector.vector.shape}")
                print(f"Global features: elixir={global_features[0]:.2f}, match_time={global_features[2]:.2f}")
                print(f"Game time features: double_elixir={game_time_features[0]:.2f}, until_overtime={game_time_features[1]:.2f}")
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.03)
            
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()
    
    finally:
        # Wait for capture thread to finish
        capture.join()
        print(f"Processed {frame_count} frames with average time: {total_time/frame_count:.2f}ms")


if __name__ == "__main__":
    main()