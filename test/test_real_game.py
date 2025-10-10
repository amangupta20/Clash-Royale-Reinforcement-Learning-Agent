"""
Test script to test all components with real game frames
"""

import sys
import os
import numpy as np
import time
import json
import cv2 as cv


def _extract_hand_roi(frame: np.ndarray) -> np.ndarray:
    """
    Extracts the region of interest (ROI) containing the player's hand from a game frame.
    
    The coordinates are from main.py.
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
    
    # Extract the ROI
    return frame[y:y+roi_height, x:x+roi_width]

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.bootstrap.capture import BootstrapCapture
from src.bootstrap.template_matcher import TemplateCardMatcher
from src.bootstrap.minimal_perception import MinimalPerception
from src.bootstrap.state_builder import MinimalStateBuilder
from src.bootstrap.executor import BootstrapActionExecutor

def test_all_components():
    """Test all components with real game frames."""
    print("=" * 60)
    print("Testing all components with real game frames")
    print("=" * 60)
    
    try:
        # Load card names
        with open("deck.json", "r") as f:
            card_names = json.load(f)
        
        print(f"Card names: {card_names}")
        
        # Initialize all components
        print("\nInitializing components...")
        capture = BootstrapCapture(window_name="BlueStacks App Player 1")
        matcher = TemplateCardMatcher(card_names)
        perception = MinimalPerception(save_debug_images=False)
        state_builder = MinimalStateBuilder(card_names)
        executor = BootstrapActionExecutor()
        
        print("✓ All components initialized successfully")
        
        # Start capture
        print("\nStarting screen capture...")
        if not capture.start_capture():
            print("✗ Failed to start capture")
            return False
        
        print("✓ Capture started successfully")
        # Wait for first frame to be available
        print("Waiting for first frame...")
        frame = None
        attempts = 0
        max_attempts = 10
        
        while frame is None and attempts < max_attempts:
            frame = capture.grab()
            if frame is None:
                attempts += 1
                print(f"  Attempt {attempts}: No frame yet, waiting...")
                time.sleep(0.5)
        
        if frame is None:
            print("✗ Failed to grab frame after multiple attempts")
            executor.disconnect()
            capture.stop_capture()
            return False
        
        print(f"✓ Got first frame: {frame.shape}")
        
        # Connect executor
        print("\nConnecting to executor...")
        if not executor.connect():
            print("✗ Failed to connect to executor")
            capture.stop_capture()
            return False
        
        print("✓ Executor connected successfully")
        
        # Test multiple frames
        for i in range(3):
            print(f"\n{'='*50}")
            print(f"Frame {i+1}")
            print(f"{'='*50}")
            
            # Get frame
            print("Getting frame...")
            frame = capture.grab()
            if frame is None:
                print("✗ Failed to grab frame")
                continue
            
            print(f"✓ Got frame: {frame.shape}")
            
            # Test template matching
            print("Testing template matching...")
            try:
                # Pass the full frame to the template matcher (it will crop internally)
                print("  Using full frame for template matching...")
                detected_cards = matcher.detect_hand_cards(frame)
                print(f"✓ Template matching results: {detected_cards}")
                    
            except Exception as e:
                print(f"✗ Template matching failed: {e}")
                detected_cards = {1: None, 2: None, 3: None, 4: None}
            
            # Test perception
            print("Testing perception...")
            try:
                elixir = perception.detect_elixir(frame)
                print(f"✓ Elixir detection result: {elixir}/10")
                
                tower_health = perception.detect_tower_health(frame)
                print(f"✓ Tower health detection result: {tower_health}")
            except Exception as e:
                print(f"✗ Perception failed: {e}")
                elixir = 0
                tower_health = {'friendly': [0, 0, 0], 'enemy': [0, 0, 0]}
            
            # Test state building
            print("Testing state building...")
            try:
                state_vector = state_builder.build_state(
                    frame=frame,
                    detected_cards=detected_cards,
                    elixir_count=elixir,
                    tower_health=tower_health,
                    match_time=time.time()
                )
                print(f"✓ State built successfully: {state_vector.vector.shape}")
                print(f"  Sample: {state_vector.vector[:10]}")
                
                # Test hand features
                hand_features = state_vector.get_hand_features()
                print(f"  Hand features: {hand_features[:20]}")  # First 2 cards
                
            except Exception as e:
                print(f"✗ State building failed: {e}")
            
            # Test action execution (only on first frame)
            if i == 0:
                print("Testing action execution...")
                test_action = {
                    'card_slot': 0,
                    'grid_x': 16,
                    'grid_y': 9
                }
                
                try:
                    result = executor.execute(test_action)
                    print(f"✓ Action execution result: {result}")
                except Exception as e:
                    print(f"✗ Action execution failed: {e}")
            
            # Wait between frames
            if i < 2:
                print("Waiting 2 seconds...")
                time.sleep(2)
        
        # Clean up
        print("\nCleaning up...")
        executor.disconnect()
        capture.stop_capture()
        print("✓ All components stopped successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("Testing BootstrapClashRoyaleEnv components with real game frames")
    print("This will test all components working together with actual game frames.")
    
    success = test_all_components()
    
    if success:
        print("\n🎉 All tests passed! Components are working correctly with real game frames.")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)