"""
Example usage of the BootstrapActionExecutor for Phase 0.

This script demonstrates how to integrate the BootstrapActionExecutor with
existing components like BootstrapCapture, TemplateCardMatcher, and MinimalPerception
to create a complete action execution pipeline.

Usage Examples:
1. Basic action execution
2. Integration with perception pipeline
3. Batch action execution
4. Error handling and recovery
"""

import time
import json
from typing import Dict, Any, List, Optional

from executor import BootstrapActionExecutor
from capture import BootstrapCapture
from template_matcher import TemplateCardMatcher
from minimal_perception import MinimalPerception
from state_builder import MinimalStateBuilder


class BootstrapActionPipeline:
    """
    Complete action pipeline for Phase 0 of the Clash Royale RL Agent.
    
    This class integrates the BootstrapActionExecutor with perception components
    to provide a complete action execution pipeline.
    """
    
    def __init__(self, 
                 resolution: str = "1920x1080",
                 window_name: str = "BlueStacks App Player 1"):
        """
        Initialize the action pipeline.
        
        Args:
            resolution: Screen resolution ("1920x1080" or "1280x720")
            window_name: Name of the BlueStacks window
        """
        self.resolution = resolution
        
        # Initialize components
        self.executor = BootstrapActionExecutor(resolution=resolution)
        self.capture = BootstrapCapture(window_name=window_name)
        self.card_matcher = TemplateCardMatcher()
        self.perception = MinimalPerception()
        
        # Load deck configuration
        with open('deck.json', 'r') as f:
            self.card_names = json.load(f)
        
        self.state_builder = MinimalStateBuilder(self.card_names)
        
        # Pipeline state
        self.is_running = False
        self.last_frame = None
        self.last_state = None
        
        print(f"BootstrapActionPipeline initialized with resolution: {resolution}")
    
    def start(self) -> bool:
        """
        Start the action pipeline.
        
        Returns:
            bool: True if pipeline started successfully, False otherwise
        """
        try:
            # Connect to ADB
            if not self.executor.connect():
                print("Failed to connect to ADB device")
                return False
            
            # Start screen capture
            if not self.capture.start_capture():
                print("Failed to start screen capture")
                return False
            
            self.is_running = True
            print("Action pipeline started successfully")
            return True
            
        except Exception as e:
            print(f"Failed to start action pipeline: {e}")
            return False
    
    def stop(self) -> None:
        """Stop the action pipeline."""
        try:
            self.is_running = False
            
            # Stop screen capture
            if self.capture:
                self.capture.stop_capture()
            
            # Disconnect from ADB
            if self.executor:
                self.executor.disconnect()
            
            print("Action pipeline stopped")
            
        except Exception as e:
            print(f"Error stopping action pipeline: {e}")
    
    def get_current_state(self) -> Optional[Dict[str, Any]]:
        """
        Get the current game state.
        
        Returns:
            Dict containing current game state or None if failed
        """
        try:
            # Capture frame
            frame = self.capture.get_frame()
            if frame is None:
                print("Failed to capture frame")
                return None
            
            self.last_frame = frame
            
            # Detect cards
            detected_cards = self.card_matcher.detect_cards(frame)
            
            # Detect elixir and towers
            elixir_count = self.perception.detect_elixir(frame)
            tower_health = self.perception.detect_towers(frame)
            
            # Build state vector
            state_vector = self.state_builder.build_state(
                frame=frame,
                detected_cards=detected_cards,
                elixir_count=elixir_count,
                tower_health=tower_health
            )
            
            self.last_state = {
                'state_vector': state_vector,
                'detected_cards': detected_cards,
                'elixir_count': elixir_count,
                'tower_health': tower_health,
                'timestamp': time.time()
            }
            
            return self.last_state
            
        except Exception as e:
            print(f"Error getting current state: {e}")
            return None
    
    def execute_action(self, action: Dict[str, Any]) -> bool:
        """
        Execute an action in the game.
        
        Args:
            action: Dictionary containing action parameters:
                - card_slot: Integer (0-7) for card selection
                - grid_x: Integer (0-31) for x position on the grid
                - grid_y: Integer (0-17) for y position on the grid
                
        Returns:
            bool: True if action was executed successfully, False otherwise
        """
        try:
            if not self.is_running:
                print("Pipeline is not running")
                return False
            
            # Get current state before action
            state_before = self.get_current_state()
            if state_before is None:
                print("Failed to get state before action")
                return False
            
            # Validate action based on current state
            if not self._validate_action_with_state(action, state_before):
                print("Action validation failed")
                return False
            
            # Execute the action
            result = self.executor.execute_with_result(action)
            
            if result.success:
                print(f"Action executed successfully in {result.execution_time_ms:.2f}ms")
                
                # Wait for action to complete
                time.sleep(0.5)
                
                # Get state after action
                state_after = self.get_current_state()
                
                return True
            else:
                print(f"Action execution failed: {result.error_message}")
                return False
                
        except Exception as e:
            print(f"Error executing action: {e}")
            return False
    
    def _validate_action_with_state(self, 
                                   action: Dict[str, Any], 
                                   state: Dict[str, Any]) -> bool:
        """
        Validate action based on current game state.
        
        Args:
            action: Action to validate
            state: Current game state
            
        Returns:
            bool: True if action is valid, False otherwise
        """
        # Basic action validation
        if not self.executor._validate_action(action):
            return False
        
        # Check if card is available in hand
        card_slot = action['card_slot']
        detected_cards = state['detected_cards']
        
        # Convert 0-based card_slot to 1-based display slot
        display_slot = card_slot + 1
        
        if display_slot not in detected_cards:
            print(f"Card slot {display_slot} not detected in hand")
            return False
        
        # Check if player has enough elixir
        elixir_count = state['elixir_count']
        card_name = detected_cards[display_slot]['card_id']
        
        if card_name:
            card_cost = self.state_builder.CARD_COSTS.get(card_name, 0)
            if elixir_count < card_cost:
                print(f"Not enough elixir: need {card_cost}, have {elixir_count}")
                return False
        
        return True
    
    def get_action_space_info(self) -> Dict[str, Any]:
        """
        Get information about the action space.
        
        Returns:
            Dict containing action space information
        """
        return {
            'card_slots': list(range(8)),  # 0-7 for 8-card deck
            'grid_width': 32,
            'grid_height': 18,
            'grid_coordinates': [(x, y) for x in range(32) for y in range(18)],
            'resolution': self.resolution,
            'screen_config': self.executor.get_screen_config()
        }


def example_basic_usage():
    """Example of basic action execution."""
    print("=== Basic Usage Example ===")
    
    # Create pipeline
    pipeline = BootstrapActionPipeline(resolution="1920x1080")
    
    try:
        # Start pipeline
        if not pipeline.start():
            print("Failed to start pipeline")
            return
        
        # Get current state
        state = pipeline.get_current_state()
        if state:
            print(f"Current elixir: {state['elixir_count']}")
            print(f"Detected cards: {list(state['detected_cards'].keys())}")
        
        # Execute a simple action
        action = {
            'card_slot': 0,  # First card in deck
            'grid_x': 16,    # Center of arena (x)
            'grid_y': 8,     # Center of arena (y)
        }
        
        print(f"Executing action: {action}")
        success = pipeline.execute_action(action)
        
        if success:
            print("Action executed successfully!")
        else:
            print("Action execution failed")
            
    finally:
        # Clean up
        pipeline.stop()


def example_batch_execution():
    """Example of batch action execution."""
    print("\n=== Batch Execution Example ===")
    
    # Create pipeline
    pipeline = BootstrapActionPipeline(resolution="1920x1080")
    
    try:
        # Start pipeline
        if not pipeline.start():
            print("Failed to start pipeline")
            return
        
        # Define a sequence of actions
        actions = [
            {'card_slot': 0, 'grid_x': 10, 'grid_y': 5},
            {'card_slot': 1, 'grid_x': 20, 'grid_y': 10},
            {'card_slot': 2, 'grid_x': 16, 'grid_y': 12},
        ]
        
        # Execute actions with delays
        for i, action in enumerate(actions):
            print(f"Executing action {i+1}/{len(actions)}: {action}")
            
            success = pipeline.execute_action(action)
            
            if success:
                print(f"✓ Action {i+1} completed")
            else:
                print(f"✗ Action {i+1} failed")
            
            # Wait between actions
            time.sleep(2.0)
            
    finally:
        # Clean up
        pipeline.stop()


def example_error_handling():
    """Example of error handling and recovery."""
    print("\n=== Error Handling Example ===")
    
    # Create pipeline
    pipeline = BootstrapActionPipeline(resolution="1920x1080")
    
    try:
        # Start pipeline
        if not pipeline.start():
            print("Failed to start pipeline")
            return
        
        # Test invalid actions
        invalid_actions = [
            {'card_slot': -1, 'grid_x': 0, 'grid_y': 0},  # Invalid card slot
            {'card_slot': 0, 'grid_x': 32, 'grid_y': 0},  # Invalid grid_x
            {'card_slot': 0, 'grid_x': 0, 'grid_y': 18},  # Invalid grid_y
        ]
        
        for action in invalid_actions:
            print(f"Testing invalid action: {action}")
            success = pipeline.execute_action(action)
            
            if not success:
                print(f"✓ Invalid action properly rejected: {action}")
            else:
                print(f"✗ Invalid action was accepted: {action}")
        
        # Test action with insufficient elixir
        # (This would require a specific game state to test properly)
        print("Note: Testing with insufficient elixir requires specific game state")
        
    finally:
        # Clean up
        pipeline.stop()


def main():
    """Main function with examples."""
    print("BootstrapActionExecutor Usage Examples\n")
    
    # Run examples
    example_basic_usage()
    example_batch_execution()
    example_error_handling()
    
    print("\n=== Examples Complete ===")


if __name__ == "__main__":
    main()