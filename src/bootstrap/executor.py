"""
Bootstrap Action Executor for Phase 0 of the Clash Royale RL Agent.

This module implements the BootstrapActionExecutor class that handles card deployment
actions via ADB with basic humanization features for the BlueStacks emulator.

Action Execution Flow:
1. Connect to BlueStacks emulator via adb-shell library
2. Map action space (card_slot, grid_x, grid_y) to screen coordinates
3. Select card from hand with coordinate jitter
4. Wait random delay (50-200ms)
5. Deploy to target position with jitter
6. Verify action completion

Performance Target: <100ms per action execution
"""

import os
import time
import random
import logging
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from adb_shell.adb_device import AdbDeviceTcp

import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the action module directly
import importlib.util
spec = importlib.util.spec_from_file_location("action_executor", os.path.join(parent_dir, "action", "executor.py"))
action_executor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(action_executor)

# Get the classes from the module
BaseActionExecutor = action_executor.BaseActionExecutor
ActionResult = action_executor.ActionResult

# Load environment variables from .env file
load_dotenv()

# Device IP and port from environment variables, with defaults
DEVICE_IP = os.environ.get("ADB_DEVICE_IP", "127.0.0.1")
DEVICE_PORT = os.environ.get("ADB_DEVICE_PORT", "5555")
DEVICE_ADDRESS = f"{DEVICE_IP}:{DEVICE_PORT}"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BootstrapActionExecutor(BaseActionExecutor):
    """An action executor for the bootstrap phase of the Clash Royale AI.

    This class implements the `BaseActionExecutor` interface to provide
    ADB-based action execution with basic humanization features for the
    BlueStacks emulator. It supports different screen resolutions and includes
    features like coordinate jitter and variable timing to simulate human-like
    interactions.

    Attributes:
        resolution: The screen resolution, e.g., "1920x1080".
        screen_config: A dictionary containing the screen configuration for the
            specified resolution.
        jitter_range: A tuple defining the range of random jitter to apply to
            coordinates.
        delay_range: A tuple defining the range of random delays between
            actions.
        device: The ADB device connection.
        is_connected_flag: A boolean indicating whether the device is connected.
        grid_width: The width of the action grid.
        grid_height: The height of the action grid.
    """

    # Screen resolution configuration for BlueStacks
    # Supports both 1920×1080 and 1280×720 resolutions
    SCREEN_RESOLUTIONS = {
        "1920x1080": {
            "width": 1920,
            "height": 1080,
            "card_positions": {
                1: (320, 1785),  # Card 1 (far left)
                2: (520, 1785),  # Card 2 (left-middle)
                3: (750, 1785),  # Card 3 (right-middle)
                4: (950, 1785),  # Card 4 (far right)
            },
            "arena_bounds": {
                "left": 0,
                "right": 1080,
                "top": 505,
                "bottom": 1440
            }
        },
        "1280x720": {
            "width": 1280,
            "height": 720,
            "card_positions": {
                1: (143, 656),   # Card 1 (far left)
                2: (273, 656),   # Card 2 (left-middle)
                3: (400, 656),   # Card 3 (right-middle)
                4: (530, 656),   # Card 4 (far right)
            },
            "arena_bounds": {
                "left": 40,
                "right": 1240,
                "top": 240,
                "bottom": 680
            }
        }
    }
    
    def __init__(self, 
                 resolution: str = "1920x1080",
                 jitter_range: Tuple[int, int] = (5, 10),
                 delay_range: Tuple[int, int] = (50, 200)):
        """Initializes the BootstrapActionExecutor.

        Args:
            resolution: The screen resolution, e.g., "1920x1080".
            jitter_range: A tuple defining the range of random jitter.
            delay_range: A tuple defining the range of random delays.

        Raises:
            ValueError: If the specified resolution is not supported.
        """
        if resolution not in self.SCREEN_RESOLUTIONS:
            raise ValueError(f"Unsupported resolution: {resolution}. "
                           f"Supported resolutions: {list(self.SCREEN_RESOLUTIONS.keys())}")
        
        self.resolution = resolution
        self.screen_config = self.SCREEN_RESOLUTIONS[resolution]
        self.jitter_range = jitter_range
        self.delay_range = delay_range
        
        # ADB device connection
        self.device: Optional[AdbDeviceTcp] = None
        self.is_connected_flag = False
        
        # Grid configuration (32×18)
        self.grid_width = 32
        self.grid_height = 18
        
        logger.info(f"BootstrapActionExecutor initialized with resolution: {resolution}")
    
    def connect(self) -> bool:
        """Connects to the BlueStacks emulator via the adb-shell library.

        Returns:
            True if the connection was successful, False otherwise.
        """
        try:
            logger.info(f"Attempting to connect to device at {DEVICE_IP}:{DEVICE_PORT} using adb-shell...")
            
            # Create ADB device with timeout
            self.device = AdbDeviceTcp(
                DEVICE_IP, 
                int(DEVICE_PORT), 
                default_transport_timeout_s=9.0
            )
            
            # Connect to the device
            self.device.connect(rsa_keys=None, auth_timeout_s=5)
            logger.info(f"Successfully connected to device at {DEVICE_ADDRESS}")
            
            # Test the connection with a simple command
            try:
                result = self.device.shell("echo 'connection test'")
                logger.info(f"Connection test successful: {result.strip()}")
                self.is_connected_flag = True
                return True
            except Exception as test_error:
                logger.error(f"Device found but not responding: {test_error}")
                return False
                
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            logger.error("Please ensure:")
            logger.error("1. BlueStacks or emulator is running")
            logger.error("2. ADB is enabled in BlueStacks settings")
            logger.error("3. Device is reachable at specified IP and port")
            self.is_connected_flag = False
            return False
    
    def disconnect(self) -> None:
        """Disconnects from the BlueStacks emulator."""
        if self.device:
            try:
                self.device.close()
                logger.info("Disconnected from device")
            except Exception as e:
                logger.warning(f"Error during disconnection: {e}")
            finally:
                self.device = None
                self.is_connected_flag = False
    
    def is_connected(self) -> bool:
        """Checks if connected to the BlueStacks emulator.

        Returns:
            True if connected, False otherwise.
        """
        return self.is_connected_flag and self.device is not None
    
    def execute(self, action: Dict[str, Any]) -> bool:
        """Executes a card deployment action in the game environment.

        Args:
            action: A dictionary containing the action parameters, including
                'card_slot', 'grid_x', and 'grid_y'.

        Returns:
            True if the action was executed successfully, False otherwise.
        """
        start_time = time.perf_counter()
        
        try:
            # Validate action parameters
            if not self._validate_action(action):
                logger.error(f"Invalid action parameters: {action}")
                return False
            
            card_slot = action['card_slot']
            grid_x = action['grid_x']
            grid_y = action['grid_y']
            
            # Handle "no action" (card_slot = 4)
            if card_slot == 4:
                execution_time = (time.perf_counter() - start_time) * 1000
                logger.info(f"No action executed in {execution_time:.2f}ms")
                return True
            
            # Convert card slot (0-3) to display card slot (1-4)
            display_card_slot = card_slot + 1
            
            # Map grid coordinates to screen coordinates
            deploy_coords = self._grid_to_screen_coords(grid_x, grid_y)
            
            # Get card position coordinates
            card_coords = self.screen_config['card_positions'][display_card_slot]
            
            # Add humanization jitter to coordinates
            jittered_card_coords = self._add_jitter(card_coords)
            jittered_deploy_coords = self._add_jitter(deploy_coords)
            
            # Execute the action sequence
            if self._execute_action_sequence(jittered_card_coords, jittered_deploy_coords):
                execution_time = (time.perf_counter() - start_time) * 1000
                logger.info(f"Action executed successfully in {execution_time:.2f}ms")
                return True
            else:
                logger.error("Action execution failed")
                return False
                
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Action execution failed with error: {e} (after {execution_time:.2f}ms)")
            return False
    
    def execute_with_result(self, action: Dict[str, Any]) -> ActionResult:
        """Executes a card deployment action and returns a detailed result.

        Args:
            action: A dictionary containing the action parameters.

        Returns:
            An `ActionResult` object with the detailed result of the action
            execution.
        """
        start_time = time.perf_counter()
        
        try:
            success = self.execute(action)
            execution_time = (time.perf_counter() - start_time) * 1000
            
            metadata = {
                'action': action,
                'resolution': self.resolution,
                'jitter_range': self.jitter_range,
                'delay_range': self.delay_range
            }
            
            return ActionResult(
                success=success,
                execution_time_ms=execution_time,
                metadata=metadata
            )
            
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            return ActionResult(
                success=False,
                execution_time_ms=execution_time,
                error_message=str(e),
                metadata={'action': action}
            )
    
    def _validate_action(self, action: Dict[str, Any]) -> bool:
        """Validates the action parameters.

        Args:
            action: The action dictionary to validate.

        Returns:
            True if the action is valid, False otherwise.
        """
        required_keys = ['card_slot', 'grid_x', 'grid_y']
        
        # Check required keys
        if not all(key in action for key in required_keys):
            logger.error(f"Missing required keys in action: {action}")
            return False
        
        # Validate card_slot (0-3 for 4 visible cards in hand, 4 for no action)
        card_slot = action['card_slot']
        if not isinstance(card_slot, int) or not (0 <= card_slot <= 4):
            logger.error(f"Invalid card_slot: {card_slot}. Must be integer 0-4")
            return False
        
        # Handle "no action" (card_slot = 4)
        if card_slot == 4:
            logger.info("No action selected, skipping card deployment")
            return True
        
        # Validate grid_x (0-31)
        grid_x = action['grid_x']
        if not isinstance(grid_x, int) or not (0 <= grid_x < self.grid_width):
            logger.error(f"Invalid grid_x: {grid_x}. Must be integer 0-{self.grid_width-1}")
            return False
        
        # Validate grid_y (0-17)
        grid_y = action['grid_y']
        if not isinstance(grid_y, int) or not (0 <= grid_y < self.grid_height):
            logger.error(f"Invalid grid_y: {grid_y}. Must be integer 0-{self.grid_height-1}")
            return False
        
        return True
    
    def _grid_to_screen_coords(self, grid_x: int, grid_y: int) -> Tuple[int, int]:
        """Converts grid coordinates to screen coordinates.

        This method maps the 32x18 grid to actual screen pixels within the
        arena bounds, covering the playable area of the game.

        Args:
            grid_x: The x-coordinate on the grid (0-31).
            grid_y: The y-coordinate on the grid (0-17).

        Returns:
            A tuple of (x, y) screen coordinates in pixels.
        """
        arena_bounds = self.screen_config['arena_bounds']
        
        # Calculate arena width and height
        arena_width = arena_bounds['right'] - arena_bounds['left']
        arena_height = arena_bounds['bottom'] - arena_bounds['top']
        
        # Calculate cell dimensions
        cell_width = arena_width / self.grid_width
        cell_height = arena_height / self.grid_height
        
        # Convert grid coordinates to screen coordinates
        # Center of the grid cell
        screen_x = int(arena_bounds['left'] + grid_x * cell_width + cell_width / 2)
        screen_y = int(arena_bounds['top'] + grid_y * cell_height + cell_height / 2)
        
        return (screen_x, screen_y)
    
    def _add_jitter(self, coords: Tuple[int, int]) -> Tuple[int, int]:
        """Adds random jitter to coordinates for humanization.

        This is a basic anti-detection measure; more sophisticated
        humanization will be implemented in Phase 4.

        Args:
            coords: The original (x, y) coordinates.

        Returns:
            The jittered (x, y) coordinates.
        """
        x, y = coords
        jitter_min, jitter_max = self.jitter_range
        
        # Random jitter in the specified range
        jitter_x = random.randint(-jitter_max, jitter_max)
        jitter_y = random.randint(-jitter_max, jitter_max)
        
        return (x + jitter_x, y + jitter_y)
    
    def _execute_action_sequence(self,
                                card_coords: Tuple[int, int],
                                deploy_coords: Tuple[int, int]) -> bool:
        """Executes the complete action sequence: select card, wait, deploy.

        Args:
            card_coords: The coordinates of the card to select.
            deploy_coords: The coordinates to deploy the card to.

        Returns:
            True if the action sequence was executed successfully, False
            otherwise.
        """
        try:
            # Step 1: Select the card
            logger.info(f"Tapping card at {card_coords}")
            if not self._execute_tap(card_coords):
                logger.error(f"Failed to tap card at coordinates: {card_coords}")
                return False
            
            # Step 2: Wait random delay (50-200ms)
            delay_ms = random.randint(*self.delay_range)
            time.sleep(delay_ms / 1000.0)
            
            logger.info(f"Deploying at {deploy_coords}")
            if not self._execute_tap(deploy_coords):
                logger.error(f"Failed to deploy card at coordinates: {deploy_coords}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error during action sequence execution: {e}")
            return False
    
    def _execute_tap(self, coords: Tuple[int, int]) -> bool:
        """Executes a tap command at the given coordinates.

        Args:
            coords: The (x, y) coordinates to tap.

        Returns:
            True if the tap was successful, False otherwise.
        """
        if not self.is_connected():
            logger.error("Not connected to device")
            return False
        
        try:
            x, y = coords
            command = f"input tap {x} {y}"
            
            # Execute the tap command via ADB shell
            result = self.device.shell(command)
            
            # Check for errors in the result
            if result and "Error" in result:
                logger.error(f"ADB tap command failed: {result}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing tap command: {e}")
            return False
    
    def get_screen_config(self) -> Dict[str, Any]:
        """Gets the current screen configuration.

        Returns:
            A dictionary containing the screen configuration details.
        """
        return {
            'resolution': self.resolution,
            'screen_config': self.screen_config,
            'grid_width': self.grid_width,
            'grid_height': self.grid_height,
            'jitter_range': self.jitter_range,
            'delay_range': self.delay_range
        }