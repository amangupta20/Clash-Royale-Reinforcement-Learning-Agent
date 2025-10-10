"""
Test script for MinimalStateBuilder (T008)

This script tests the StateVector data entity and MinimalStateBuilder class
to ensure they produce valid outputs for Phase 0 of the Clash Royale RL Agent.
"""

import json
import numpy as np
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from bootstrap.state_builder import StateVector, MinimalStateBuilder
from bootstrap.template_matcher import TemplateCardMatcher
from bootstrap.minimal_perception import MinimalPerception


def test_state_vector():
    """Test StateVector data entity validation."""
    print("Testing StateVector validation...")
    
    # Test valid StateVector
    valid_vector = np.random.uniform(-1, 1, size=(53,)).astype(np.float32)
    valid_metadata = {'test': True}
    
    try:
        state_vector = StateVector(
            vector=valid_vector,
            timestamp=time.time(),
            metadata=valid_metadata
        )
        print("✓ Valid StateVector created successfully")
    except Exception as e:
        print(f"✗ Failed to create valid StateVector: {e}")
        return False
    
    # Test invalid shape
    try:
        invalid_vector = np.random.uniform(-1, 1, size=(49,)).astype(np.float32)
        StateVector(
            vector=invalid_vector,
            timestamp=time.time(),
            metadata=valid_metadata
        )
        print("✗ StateVector with invalid shape was accepted")
        return False
    except ValueError:
        print("✓ StateVector correctly rejected invalid shape")
    
    # Test NaN values
    try:
        nan_vector = np.full((53,), np.nan, dtype=np.float32)
        StateVector(
            vector=nan_vector,
            timestamp=time.time(),
            metadata=valid_metadata
        )
        print("✗ StateVector with NaN values was accepted")
        return False
    except ValueError:
        print("✓ StateVector correctly rejected NaN values")
    
    # Test out-of-range values (only NaN/Inf should be rejected now)
    try:
        oor_vector = np.full((53,), np.nan, dtype=np.float32)  # NaN values
        StateVector(
            vector=oor_vector,
            timestamp=time.time(),
            metadata=valid_metadata
        )
        print("✗ StateVector with NaN values was accepted")
        return False
    except ValueError:
        print("✓ StateVector correctly rejected NaN values")
    
    # Test feature extraction methods
    global_features = state_vector.get_global_features()
    hand_features = state_vector.get_hand_features()
    
    if global_features.shape != (13,):
        print(f"✗ Global features has wrong shape: {global_features.shape}")
        return False
    
    if hand_features.shape != (40,):
        print(f"✗ Hand features has wrong shape: {hand_features.shape}")
        return False
    
    print("✓ All StateVector tests passed")
    return True


def test_minimal_state_builder():
    """Test MinimalStateBuilder class."""
    print("\nTesting MinimalStateBuilder...")
    
    # Load deck configuration
    try:
        with open("deck.json", "r") as f:
            deck = json.load(f)
            if len(deck) != 8:
                raise ValueError("Deck must contain exactly 8 cards")
            print(f"Loaded deck: {deck}")
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"Error loading deck: {e}")
        return False
    
    # Initialize MinimalStateBuilder
    try:
        state_builder = MinimalStateBuilder(deck)
        print("✓ MinimalStateBuilder initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize MinimalStateBuilder: {e}")
        return False
    
    # Create mock inputs
    frame = np.random.randint(0, 255, size=(1080, 1920, 3), dtype=np.uint8)
    
    detected_cards = {
        1: {'card_id': 'archers', 'confidence': 0.85, 'position': (100, 100)},
        2: {'card_id': 'giant', 'confidence': 0.90, 'position': (200, 100)},
        3: {'card_id': 'knight', 'confidence': 0.80, 'position': (300, 100)},
        4: {'card_id': None, 'confidence': 0.0, 'position': None}  # Empty slot
    }
    
    elixir_count = 5
    
    tower_health = {
        'friendly': [1400, 1400, 3600],  # princess left, princess right, king
        'enemy': [700, 700, 3600]        # princess left, princess right, king
    }
    
    match_time = 120.0  # 2 minutes
    
    # Test state building
    try:
        start_time = time.perf_counter()
        state_vector = state_builder.build_state(
            frame=frame,
            detected_cards=detected_cards,
            elixir_count=elixir_count,
            tower_health=tower_health,
            match_time=match_time
        )
        processing_time = (time.perf_counter() - start_time) * 1000
        
        print(f"✓ State built successfully in {processing_time:.2f}ms")
        
        # Validate output
        if state_vector.vector.shape != (53,):
            print(f"✗ State vector has wrong shape: {state_vector.vector.shape}")
            return False
        
        if not np.all(np.isfinite(state_vector.vector)):
            print("✗ State vector contains non-finite values")
            return False
        
        # Check some expected values
        global_features = state_vector.get_global_features()
        
        # Player elixir should be normalized (5/10 = 0.5)
        if abs(global_features[0] - 0.5) > 0.01:
            print(f"✗ Player elixir not normalized correctly: {global_features[0]}")
            return False
        
        # Match time should be normalized (120/300 = 0.4)
        if abs(global_features[2] - 0.4) > 0.01:
            print(f"✗ Match time not normalized correctly: {global_features[2]}")
            return False
        
        # Tower health should be as-is (not normalized)
        # Order: friendly princess left (3), friendly princess right (4), friendly king (5)
        # Friendly princess tower health should be 1400
        if abs(global_features[3] - 1400.0) > 0.01:
            print(f"✗ Princess tower health not correct: {global_features[3]}")
            return False
        
        # Check mid-game phase indicator (1.0 for mid-game)
        if abs(global_features[10] - 1.0) > 0.01:
            print(f"✗ Mid-game phase indicator not correct: {global_features[10]}")
            return False
        
        # Check hand features
        hand_features = state_vector.get_hand_features()
        
        # First card (archers) should have card ID (0/7 = 0.0)
        if abs(hand_features[0] - 0.0) > 0.01:  # First card in deck
            print(f"✗ First card ID incorrect: {hand_features[0]}")
            return False
        
        # Check archers attributes: [is_air, attack_air, is_wincondition, is_tank, is_swarm, is_spell, is_aoe, is_building]
        # Archers should be: [1, 1, 0, 0, 0, 0, 0, 0]
        expected_archers_attrs = [1, 1, 0, 0, 0, 0, 0, 0]
        actual_archers_attrs = hand_features[1:9]
        if not np.allclose(actual_archers_attrs, expected_archers_attrs):
            print(f"✗ Archers attributes incorrect: {actual_archers_attrs}, expected: {expected_archers_attrs}")
            return False
        
        # Check elixir cost (should be 3 for archers)
        if abs(hand_features[9] - 3.0) > 0.01:
            print(f"✗ Archers elixir cost incorrect: {hand_features[9]}")
            return False
        
        # Second card (giant) should have card ID (1/7 = ~0.143)
        if abs(hand_features[10] - (1.0/7.0)) > 0.01:
            print(f"✗ Second card ID incorrect: {hand_features[10]}")
            return False
        
        # Check giant attributes: [0, 0, 0, 1, 0, 0, 0, 0]
        expected_giant_attrs = [0, 0, 0, 1, 0, 0, 0, 0]
        actual_giant_attrs = hand_features[11:19]
        if not np.allclose(actual_giant_attrs, expected_giant_attrs):
            print(f"✗ Giant attributes incorrect: {actual_giant_attrs}, expected: {expected_giant_attrs}")
            return False
        
        # Check elixir cost (should be 5 for giant)
        if abs(hand_features[19] - 5.0) > 0.01:
            print(f"✗ Giant elixir cost incorrect: {hand_features[19]}")
            return False
        
        print("✓ All StateBuilder tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Failed to build state: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance():
    """Test performance of StateBuilder."""
    print("\nTesting StateBuilder performance...")
    
    # Load deck configuration
    try:
        with open("deck.json", "r") as f:
            deck = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"Error loading deck: {e}")
        return False
    
    # Initialize components
    state_builder = MinimalStateBuilder(deck)
    
    # Create mock inputs
    frame = np.random.randint(0, 255, size=(1080, 1920, 3), dtype=np.uint8)
    
    detected_cards = {
        1: {'card_id': 'archers', 'confidence': 0.85, 'position': (100, 100)},
        2: {'card_id': 'giant', 'confidence': 0.90, 'position': (200, 100)},
        3: {'card_id': 'knight', 'confidence': 0.80, 'position': (300, 100)},
        4: {'card_id': 'mini_pekka', 'confidence': 0.75, 'position': (400, 100)}
    }
    
    elixir_count = 5
    tower_health = {
        'friendly': [1400, 1400, 3600],
        'enemy': [700, 700, 3600]
    }
    match_time = 120.0
    
    # Run multiple iterations
    num_iterations = 100
    processing_times = []
    
    for i in range(num_iterations):
        start_time = time.perf_counter()
        state_vector = state_builder.build_state(
            frame=frame,
            detected_cards=detected_cards,
            elixir_count=elixir_count,
            tower_health=tower_health,
            match_time=match_time
        )
        processing_time = (time.perf_counter() - start_time) * 1000
        processing_times.append(processing_time)
    
    avg_time = np.mean(processing_times)
    p95_time = np.percentile(processing_times, 95)
    
    print(f"✓ Average processing time: {avg_time:.2f}ms")
    print(f"✓ P95 processing time: {p95_time:.2f}ms")
    
    # Performance target: <10ms processing time
    if avg_time > 10:
        print(f"✗ Average processing time exceeds target: {avg_time:.2f}ms > 10ms")
        return False
    
    print("✓ Performance test passed")
    return True


def test_edge_cases():
    """Test edge cases for StateBuilder."""
    print("\nTesting edge cases...")
    
    # Load deck configuration
    try:
        with open("deck.json", "r") as f:
            deck = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"Error loading deck: {e}")
        return False
    
    # Initialize components
    state_builder = MinimalStateBuilder(deck)
    
    # Create mock inputs
    frame = np.random.randint(0, 255, size=(1080, 1920, 3), dtype=np.uint8)
    
    # Test with no cards detected
    empty_detected_cards = {
        1: {'card_id': None, 'confidence': 0.0, 'position': None},
        2: {'card_id': None, 'confidence': 0.0, 'position': None},
        3: {'card_id': None, 'confidence': 0.0, 'position': None},
        4: {'card_id': None, 'confidence': 0.0, 'position': None}
    }
    
    try:
        state_vector = state_builder.build_state(
            frame=frame,
            detected_cards=empty_detected_cards,
            elixir_count=0,
            tower_health={'friendly': [0, 0, 0], 'enemy': [0, 0, 0]},
            match_time=0.0
        )
        print("✓ StateBuilder handled empty card detection")
    except Exception as e:
        print(f"✗ Failed with empty card detection: {e}")
        return False
    
    # Test with maximum values
    max_detected_cards = {
        1: {'card_id': 'valkyrie', 'confidence': 1.0, 'position': (100, 100)},
        2: {'card_id': 'musketeer', 'confidence': 1.0, 'position': (200, 100)},
        3: {'card_id': 'minions', 'confidence': 1.0, 'position': (300, 100)},
        4: {'card_id': 'goblin_hut', 'confidence': 1.0, 'position': (400, 100)}
    }
    
    try:
        state_vector = state_builder.build_state(
            frame=frame,
            detected_cards=max_detected_cards,
            elixir_count=10,
            tower_health={'friendly': [1400, 1400, 3600], 'enemy': [1400, 1400, 3600]},
            match_time=300.0
        )
        print("✓ StateBuilder handled maximum values")
    except Exception as e:
        print(f"✗ Failed with maximum values: {e}")
        return False
    
    print("✓ Edge case tests passed")
    return True


if __name__ == "__main__":
    print("Running StateBuilder tests...")
    
    results = []
    results.append(test_state_vector())
    results.append(test_minimal_state_builder())
    results.append(test_performance())
    results.append(test_edge_cases())
    
    if all(results):
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
        exit(1)