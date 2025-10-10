"""
Live test script for the BootstrapActionExecutor.

This script tests the actual action execution in the game (not dry run).
Make sure BlueStacks is running with Clash Royale open before running this script.
"""

import time
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'action'))

# Direct imports to avoid circular import issues
from executor import BootstrapActionExecutor


def test_live_execution():
    """Test live action execution in the game."""
    print("=== Live Action Execution Test ===")
    print("Make sure BlueStacks is running with Clash Royale open!")
    print("This test will execute actual actions in the game.\n")
    
    # Create executor
    executor = BootstrapActionExecutor(
        resolution="1920x1080",
        jitter_range=(5, 10),
        delay_range=(50, 200)
    )
    
    try:
        # Connect to the device
        print("Connecting to BlueStacks...")
        if not executor.connect():
            print("✗ Failed to connect to BlueStacks")
            return False
        
        print("✓ Successfully connected to BlueStacks")
        
        # Test actions (these will actually execute in the game)
        test_actions = [
            {'card_slot': 0, 'grid_x': 16, 'grid_y': 8},   # Center of arena
            {'card_slot': 1, 'grid_x': 8, 'grid_y': 5},    # Left side
            {'card_slot': 2, 'grid_x': 24, 'grid_y': 12},  # Right side
        ]
        
        print(f"\nExecuting {len(test_actions)} test actions...")
        print("Watch the game screen to see the actions being executed!\n")
        
        for i, action in enumerate(test_actions):
            print(f"--- Action {i+1}/{len(test_actions)} ---")
            print(f"Action: {action}")
            
            # Execute the action
            result = executor.execute_with_result(action)
            
            if result.success:
                print(f"✓ Action executed successfully in {result.execution_time_ms:.2f}ms")
            else:
                print(f"✗ Action execution failed: {result.error_message}")
                return False
            
            # Wait between actions
            if i < len(test_actions) - 1:
                print("Waiting 3 seconds before next action...")
                time.sleep(3.0)
        
        print("\n✓ All live action tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Live test failed with error: {e}")
        return False
        
    finally:
        # Clean up
        if executor.is_connected():
            executor.disconnect()
            print("✓ Disconnected from BlueStacks")


def main():
    """Main function."""
    print("BootstrapActionExecutor Live Test")
    print("="*50)
    
    # Ask for confirmation
    response = input("\nAre you ready to test live actions in Clash Royale? (y/n): ")
    if response.lower() != 'y':
        print("Test cancelled.")
        return
    
    # Run the live test
    success = test_live_execution()
    
    if success:
        print("\n🎉 Live test completed successfully!")
        print("The BootstrapActionExecutor is working correctly and can execute actions in the game.")
    else:
        print("\n❌ Live test failed.")
        print("Please check the error messages above and ensure:")
        print("1. BlueStacks is running")
        print("2. Clash Royale is open and in a battle")
        print("3. ADB is enabled in BlueStacks settings")
        print("4. You have enough elixir to play cards")


if __name__ == "__main__":
    main()