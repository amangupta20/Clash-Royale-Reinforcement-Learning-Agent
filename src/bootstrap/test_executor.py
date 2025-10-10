"""
Test script for the BootstrapActionExecutor.

This script validates the action execution functionality including:
- ADB connection to BlueStacks emulator
- Coordinate mapping from grid to screen pixels
- Card slot to screen coordinate mapping
- Action execution with humanization
- Error handling scenarios
"""

import time
import json
from typing import Dict, Any, List
from executor import BootstrapActionExecutor


def test_connection(executor: BootstrapActionExecutor) -> bool:
    """
    Test ADB connection to BlueStacks emulator.
    
    Args:
        executor: BootstrapActionExecutor instance
        
    Returns:
        bool: True if connection test passed, False otherwise
    """
    print("Testing ADB connection...")
    
    try:
        # Test connection
        if executor.connect():
            print("✓ Successfully connected to BlueStacks emulator")
            
            # Verify connection status
            if executor.is_connected():
                print("✓ Connection status verified")
                return True
            else:
                print("✗ Connection status mismatch")
                return False
        else:
            print("✗ Failed to connect to BlueStacks emulator")
            return False
            
    except Exception as e:
        print(f"✗ Connection test failed with error: {e}")
        return False


def test_coordinate_mapping(executor: BootstrapActionExecutor) -> bool:
    """
    Test coordinate mapping from grid to screen pixels.
    
    Args:
        executor: BootstrapActionExecutor instance
        
    Returns:
        bool: True if coordinate mapping test passed, False otherwise
    """
    print("\nTesting coordinate mapping...")
    
    try:
        # Test grid to screen coordinate mapping
        test_cases = [
            (0, 0),      # Top-left corner
            (31, 17),    # Bottom-right corner
            (16, 8),     # Center of grid
            (10, 5),     # Random position
        ]
        
        for grid_x, grid_y in test_cases:
            screen_coords = executor._grid_to_screen_coords(grid_x, grid_y)
            print(f"✓ Grid ({grid_x}, {grid_y}) -> Screen {screen_coords}")
            
            # Verify coordinates are within screen bounds
            screen_config = executor.screen_config
            arena_bounds = screen_config['arena_bounds']
            
            if not (arena_bounds['left'] <= screen_coords[0] <= arena_bounds['right']):
                print(f"✗ X coordinate {screen_coords[0]} out of bounds")
                return False
                
            if not (arena_bounds['top'] <= screen_coords[1] <= arena_bounds['bottom']):
                print(f"✗ Y coordinate {screen_coords[1]} out of bounds")
                return False
        
        print("✓ All coordinate mappings are within valid bounds")
        return True
        
    except Exception as e:
        print(f"✗ Coordinate mapping test failed with error: {e}")
        return False


def test_card_slot_mapping(executor: BootstrapActionExecutor) -> bool:
    """
    Test card slot to screen coordinate mapping.
    
    Args:
        executor: BootstrapActionExecutor instance
        
    Returns:
        bool: True if card slot mapping test passed, False otherwise
    """
    print("\nTesting card slot mapping...")
    
    try:
        # Test all 4 card slots
        card_positions = executor.screen_config['card_positions']
        
        for slot in range(1, 5):
            coords = card_positions[slot]
            print(f"✓ Card slot {slot} -> Screen {coords}")
            
            # Verify coordinates are reasonable
            if coords[0] < 0 or coords[0] > executor.screen_config['width']:
                print(f"✗ Card slot {slot} X coordinate {coords[0]} out of screen bounds")
                return False
                
            if coords[1] < 0 or coords[1] > executor.screen_config['height']:
                print(f"✗ Card slot {slot} Y coordinate {coords[1]} out of screen bounds")
                return False
        
        print("✓ All card slot mappings are within valid bounds")
        return True
        
    except Exception as e:
        print(f"✗ Card slot mapping test failed with error: {e}")
        return False


def test_action_validation(executor: BootstrapActionExecutor) -> bool:
    """
    Test action parameter validation.
    
    Args:
        executor: BootstrapActionExecutor instance
        
    Returns:
        bool: True if validation test passed, False otherwise
    """
    print("\nTesting action validation...")
    
    try:
        # Test valid actions
        valid_actions = [
            {'card_slot': 0, 'grid_x': 0, 'grid_y': 0},
            {'card_slot': 3, 'grid_x': 31, 'grid_y': 17},
            {'card_slot': 2, 'grid_x': 16, 'grid_y': 8},
        ]
        
        for action in valid_actions:
            if executor._validate_action(action):
                print(f"✓ Valid action accepted: {action}")
            else:
                print(f"✗ Valid action rejected: {action}")
                return False
        
        # Test invalid actions
        invalid_actions = [
            {'card_slot': -1, 'grid_x': 0, 'grid_y': 0},  # Invalid card_slot
            {'card_slot': 4, 'grid_x': 0, 'grid_y': 0},   # Invalid card_slot
            {'card_slot': 0, 'grid_x': -1, 'grid_y': 0},  # Invalid grid_x
            {'card_slot': 0, 'grid_x': 32, 'grid_y': 0},  # Invalid grid_x
            {'card_slot': 0, 'grid_x': 0, 'grid_y': -1},  # Invalid grid_y
            {'card_slot': 0, 'grid_x': 0, 'grid_y': 18},  # Invalid grid_y
            {'card_slot': 0, 'grid_x': 0},                # Missing grid_y
            {},                                            # Empty action
        ]
        
        for action in invalid_actions:
            if not executor._validate_action(action):
                print(f"✓ Invalid action rejected: {action}")
            else:
                print(f"✗ Invalid action accepted: {action}")
                return False
        
        print("✓ All validation tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Action validation test failed with error: {e}")
        return False


def test_humanization_features(executor: BootstrapActionExecutor) -> bool:
    """
    Test humanization features (jitter and delays).
    
    Args:
        executor: BootstrapActionExecutor instance
        
    Returns:
        bool: True if humanization test passed, False otherwise
    """
    print("\nTesting humanization features...")
    
    try:
        # Test coordinate jitter
        base_coords = (500, 800)
        jittered_coords = []
        
        for i in range(10):
            jittered = executor._add_jitter(base_coords)
            jittered_coords.append(jittered)
            print(f"✓ Jitter test {i+1}: {base_coords} -> {jittered}")
        
        # Verify jitter is within expected range
        jitter_min, jitter_max = executor.jitter_range
        for jittered in jittered_coords:
            x_diff = abs(jittered[0] - base_coords[0])
            y_diff = abs(jittered[1] - base_coords[1])
            
            # Allow jitter in range [0, jitter_max] since random.randint can produce 0
            if x_diff < 0 or x_diff > jitter_max:
                print(f"✗ X jitter {x_diff} out of range [0, {jitter_max}]")
                return False
                
            if y_diff < 0 or y_diff > jitter_max:
                print(f"✗ Y jitter {y_diff} out of range [0, {jitter_max}]")
                return False
        
        print("✓ Coordinate jitter is within expected range")
        
        # Test delay range
        delay_min, delay_max = executor.delay_range
        print(f"✓ Delay range configured: [{delay_min}ms, {delay_max}ms]")
        
        return True
        
    except Exception as e:
        print(f"✗ Humanization test failed with error: {e}")
        return False


def test_action_execution(executor: BootstrapActionExecutor, dry_run: bool = True) -> bool:
    """
    Test action execution (with optional dry run).
    
    Args:
        executor: BootstrapActionExecutor instance
        dry_run: If True, only test action preparation without actual execution
        
    Returns:
        bool: True if action execution test passed, False otherwise
    """
    print(f"\nTesting action execution (dry_run={dry_run})...")
    
    try:
        # Test action preparation
        test_actions = [
            {'card_slot': 0, 'grid_x': 5, 'grid_y': 8},
            {'card_slot': 2, 'grid_x': 16, 'grid_y': 8},
            {'card_slot': 3, 'grid_x': 25, 'grid_y': 12},
        ]
        
        for i, action in enumerate(test_actions):
            print(f"\nAction {i+1}: {action}")
            
            # Validate action
            if not executor._validate_action(action):
                print(f"✗ Action validation failed: {action}")
                return False
            
            # Prepare coordinates
            card_slot = action['card_slot']
            display_card_slot = card_slot + 1
            deploy_coords = executor._grid_to_screen_coords(action['grid_x'], action['grid_y'])
            card_coords = executor.screen_config['card_positions'][display_card_slot]
            
            print(f"  Card slot {display_card_slot} -> {card_coords}")
            print(f"  Grid ({action['grid_x']}, {action['grid_y']}) -> {deploy_coords}")
            
            # Add jitter
            jittered_card_coords = executor._add_jitter(card_coords)
            jittered_deploy_coords = executor._add_jitter(deploy_coords)
            
            print(f"  Jittered card coords: {jittered_card_coords}")
            print(f"  Jittered deploy coords: {jittered_deploy_coords}")
            
            # Execute action if not dry run
            if not dry_run and executor.is_connected():
                result = executor.execute_with_result(action)
                if result.success:
                    print(f"  ✓ Action executed successfully in {result.execution_time_ms:.2f}ms")
                else:
                    print(f"  ✗ Action execution failed: {result.error_message}")
                    return False
            elif not dry_run:
                print("  ⚠ Skipping execution (not connected)")
        
        print("✓ Action execution tests completed")
        return True
        
    except Exception as e:
        print(f"✗ Action execution test failed with error: {e}")
        return False


def test_error_handling(executor: BootstrapActionExecutor) -> bool:
    """
    Test error handling scenarios.
    
    Args:
        executor: BootstrapActionExecutor instance
        
    Returns:
        bool: True if error handling test passed, False otherwise
    """
    print("\nTesting error handling...")
    
    try:
        # Test execution with invalid action
        invalid_action = {'card_slot': -1, 'grid_x': 0, 'grid_y': 0}
        result = executor.execute_with_result(invalid_action)
        
        if not result.success:
            print(f"✓ Invalid action properly handled: {result.error_message}")
        else:
            print("✗ Invalid action was not properly handled")
            return False
        
        # Test execution when not connected
        if executor.is_connected():
            executor.disconnect()
        
        valid_action = {'card_slot': 0, 'grid_x': 0, 'grid_y': 0}
        result = executor.execute_with_result(valid_action)
        
        if not result.success:
            print(f"✓ Disconnected state properly handled: {result.error_message}")
        else:
            print("✗ Disconnected state was not properly handled")
            return False
        
        print("✓ Error handling tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed with error: {e}")
        return False


def main():
    """Main test function."""
    print("=== BootstrapActionExecutor Test Suite ===\n")
    
    # Test both resolutions
    resolutions = ["1920x1080", "1280x720"]
    
    for resolution in resolutions:
        print(f"\n{'='*50}")
        print(f"Testing with resolution: {resolution}")
        print(f"{'='*50}")
        
        # Create executor
        executor = BootstrapActionExecutor(
            resolution=resolution,
            jitter_range=(5, 10),
            delay_range=(50, 200)
        )
        
        # Run tests
        test_results = []
        
        # Test 1: Connection (requires BlueStacks to be running)
        test_results.append(test_connection(executor))
        
        # Test 2: Coordinate mapping
        test_results.append(test_coordinate_mapping(executor))
        
        # Test 3: Card slot mapping
        test_results.append(test_card_slot_mapping(executor))
        
        # Test 4: Action validation
        test_results.append(test_action_validation(executor))
        
        # Test 5: Humanization features
        test_results.append(test_humanization_features(executor))
        
        # Test 6: Action execution (dry run)
        test_results.append(test_action_execution(executor, dry_run=True))
        
        # Test 7: Error handling
        test_results.append(test_error_handling(executor))
        
        # Clean up
        if executor.is_connected():
            executor.disconnect()
        
        # Print results
        passed = sum(test_results)
        total = len(test_results)
        
        print(f"\n{'='*50}")
        print(f"Test Results for {resolution}: {passed}/{total} passed")
        print(f"{'='*50}")
        
        if passed == total:
            print("✓ All tests passed!")
        else:
            print(f"✗ {total - passed} test(s) failed")
    
    print("\n=== Test Suite Complete ===")


if __name__ == "__main__":
    main()