"""
Bootstrap Gymnasium Environment for Phase 0 (T010)

This module implements the BootstrapClashRoyaleEnv class that integrates all Phase 0 components
into a unified RL environment using the Gymnasium API.

Phase 0 environment uses simplified state and reward; upgraded in Phase 1

Performance Targets:
- Step execution: <500ms P95 (Phase 0 requirement)
- Frame capture: <50ms average
- State building: <10ms
"""

import time
import json
import logging
from typing import Dict, Any, Tuple, Optional, List
from enum import Enum
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2 as cv

# Import Phase 0 components
from .capture import BootstrapCapture
from .template_matcher import TemplateCardMatcher
from .minimal_perception import MinimalPerception
from .state_builder import MinimalStateBuilder, StateVector
from .executor import BootstrapActionExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GamePhase(Enum):
    """Game phase enumeration for state tracking."""
    MENU = "menu"
    LOADING = "loading"
    PLAYING = "playing"
    ENDED = "ended"


@dataclass
class EnvironmentConfig:
    """Configuration for BootstrapClashRoyaleEnv."""
    # Screen capture settings
    window_name: str = "BlueStacks App Player 1"
    roi: Optional[Tuple[int, int, int, int]] = None
    
    # Action executor settings
    resolution: str = "1920x1080"
    jitter_range: Tuple[int, int] = (5, 10)
    delay_range: Tuple[int, int] = (50, 200)
    
    # Performance settings
    max_step_time_ms: float = 500.0
    action_delay_ms: float = 1000.0
    
    # Game settings
    card_names: List[str] = None
    
    def __post_init__(self):
        """Initialize default card names if not provided."""
        if self.card_names is None:
            # Load default deck from deck.json
            try:
                with open("deck.json", "r") as f:
                    self.card_names = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load deck.json: {e}")
                self.card_names = [
                    "archers", "giant", "knight", "mini_pekka",
                    "goblin_hut", "minions", "musketeer", "valkyrie"
                ]


class BootstrapClashRoyaleEnv(gym.Env):
    """
    Bootstrap Clash Royale Environment for Phase 0.
    
    This class implements a Gymnasium environment that integrates all Phase 0 components:
    - BootstrapCapture for screen capture
    - TemplateCardMatcher for card detection
    - MinimalPerception for elixir and tower health OCR
    - MinimalStateBuilder for state vector construction
    - BootstrapActionExecutor for ADB-based action execution
    
    Phase 0 environment uses simplified state and reward; upgraded in Phase 1
    
    Action Space: MultiDiscrete([4, 32, 18])
    - card_slot: 0-3 (4 visible cards)
    - grid_x: 0-31 (horizontal grid)
    - grid_y: 0-17 (vertical grid)
    
    Observation Space: Box(low=-1, high=1, shape=(53,), dtype=np.float32)
    - 53-dimensional state vector from Phase 0
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30
    }
    
    def __init__(self, config: Optional[EnvironmentConfig] = None, render_mode: Optional[str] = None):
        """
        Initialize the BootstrapClashRoyaleEnv.
        
        Args:
            config: Environment configuration
            render_mode: Render mode ('human', 'rgb_array', or None)
            
        Raises:
            ValueError: If render_mode is invalid or card_names is incorrect
        """
        super().__init__()
        
        # Validate render mode
        if render_mode is not None and render_mode not in self.metadata['render_modes']:
            raise ValueError(f"Invalid render mode: {render_mode}")
        self.render_mode = render_mode
        
        # Set configuration
        self.config = config or EnvironmentConfig()
        
        # Validate card names
        if not self.config.card_names or len(self.config.card_names) != 8:
            raise ValueError("Exactly 8 card names required for deck")
        
        # Define action and observation spaces
        self.action_space = spaces.MultiDiscrete([4, 32, 18])
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(53,), 
            dtype=np.float32
        )
        
        # Initialize Phase 0 components
        self.capture = BootstrapCapture(
            window_name=self.config.window_name,
            roi=self.config.roi,
            show_fps=False
        )
        
        self.card_matcher = TemplateCardMatcher(self.config.card_names)
        
        self.perception = MinimalPerception(save_debug_images=False)
        
        self.state_builder = MinimalStateBuilder(self.config.card_names)
        
        self.executor = BootstrapActionExecutor(
            resolution=self.config.resolution,
            jitter_range=self.config.jitter_range,
            delay_range=self.config.delay_range
        )
        
        # Game state tracking
        self._game_phase = GamePhase.MENU
        self._match_start_time = None
        self._last_tower_health = None
        self._step_count = 0
        self._episode_count = 0
        
        # Performance tracking
        self._step_times = []
        
        # Card elixir costs for action validation
        self._card_costs = {
            'archers': 3,
            'giant': 5,
            'knight': 3,
            'mini_pekka': 4,
            'goblin_hut': 5,
            'minions': 3,
            'musketeer': 4,
            'valkyrie': 4
        }
        
        logger.info(f"BootstrapClashRoyaleEnv initialized with {len(self.config.card_names)} cards")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to start a new episode.
        
        Environment Flow:
        1. Wait for menu → Click battle → Wait for match start
        2. Return initial state
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        start_time = time.perf_counter()
        
        try:
            # Start capture if not already running
            if not self.capture.is_capturing:
                self.capture.start_capture()
            
            # Connect to executor if not connected
            if not self.executor.is_connected():
                if not self.executor.connect():
                    raise RuntimeError("Failed to connect to action executor")
            
            # Reset game state
            self._game_phase = GamePhase.MENU
            self._match_start_time = None
            self._last_tower_health = None
            self._step_count = 0
            self._episode_count += 1
            
            # Navigate to battle and wait for match start
            self._navigate_to_battle()
            
            # Get initial state
            obs = self._get_current_state()
            
            reset_time = (time.perf_counter() - start_time) * 1000
            logger.info(f"Environment reset completed in {reset_time:.2f}ms")
            
            info = {
                'episode_count': self._episode_count,
                'game_phase': self._game_phase.value,
                'reset_time_ms': reset_time,
                'step_count': self._step_count
            }
            
            return obs, info
            
        except Exception as e:
            logger.error(f"Reset failed: {e}")
            # Return zero observation on error
            obs = np.zeros(53, dtype=np.float32)
            info = {
                'episode_count': self._episode_count,
                'game_phase': self._game_phase.value,
                'error': str(e),
                'step_count': self._step_count
            }
            return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Environment Flow:
        1. Execute action → Wait → Capture frame → Build state → Compute reward → Check done
        
        Args:
            action: Action to take as numpy array [card_slot, grid_x, grid_y]
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self._game_phase == GamePhase.ENDED:
            raise RuntimeError("Environment is done. Call reset() before step().")
        
        step_start_time = time.perf_counter()
        self._step_count += 1
        
        try:
            # Extract action components
            card_slot = int(action[0])
            grid_x = int(action[1])
            grid_y = int(action[2])
            
            # Validate action
            if not self._validate_action(card_slot, grid_x, grid_y):
                logger.warning(f"Invalid action: [{card_slot}, {grid_x}, {grid_y}]")
                reward = -0.1  # Small penalty for invalid action
                terminated = False
                truncated = False
                obs = self._get_current_state()
                
                info = {
                    'step_count': self._step_count,
                    'game_phase': self._game_phase.value,
                    'action_valid': False,
                    'action': {'card_slot': card_slot, 'grid_x': grid_x, 'grid_y': grid_y}
                }
                
                return obs, reward, terminated, truncated, info
            
            # Store previous tower health for reward calculation
            prev_tower_health = self._get_tower_health()
            
            # Execute action
            action_success = self._execute_action(card_slot, grid_x, grid_y)
            
            # Wait for action to complete
            time.sleep(self.config.action_delay_ms / 1000.0)
            
            # Get current state
            obs = self._get_current_state()
            
            # Calculate reward
            reward, terminated, truncated = self._calculate_reward(prev_tower_health, action_success)
            
            # Update game phase if needed
            if terminated:
                self._game_phase = GamePhase.ENDED
            
            # Track performance
            step_time = (time.perf_counter() - step_start_time) * 1000
            self._step_times.append(step_time)
            
            info = {
                'step_count': self._step_count,
                'game_phase': self._game_phase.value,
                'action_valid': True,
                'action_success': action_success,
                'action': {'card_slot': card_slot, 'grid_x': grid_x, 'grid_y': grid_y},
                'step_time_ms': step_time,
                'avg_step_time_ms': np.mean(self._step_times[-100:]) if self._step_times else 0.0
            }
            
            # Log performance warning if step took too long
            if step_time > self.config.max_step_time_ms:
                logger.warning(f"Slow step: {step_time:.2f}ms > {self.config.max_step_time_ms}ms")
            
            return obs, reward, terminated, truncated, info
            
        except Exception as e:
            logger.error(f"Step failed: {e}")
            
            # Return error state
            obs = np.zeros(53, dtype=np.float32)
            reward = -1.0  # Large penalty for errors
            terminated = True
            truncated = False
            
            step_time = (time.perf_counter() - step_start_time) * 1000
            
            info = {
                'step_count': self._step_count,
                'game_phase': self._game_phase.value,
                'error': str(e),
                'step_time_ms': step_time
            }
            
            return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return
        
        try:
            # Get current frame
            frame = self.capture.grab()
            if frame is None:
                return
            
            if self.render_mode == 'human':
                # Display frame with OpenCV
                cv.imshow('Clash Royale Environment', frame)
                cv.waitKey(1)
            elif self.render_mode == 'rgb_array':
                # Return frame as RGB array
                return cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                
        except Exception as e:
            logger.error(f"Render failed: {e}")
    
    def close(self):
        """Close the environment and clean up resources."""
        try:
            # Stop capture
            if hasattr(self, 'capture') and self.capture.is_capturing:
                self.capture.stop_capture()
            
            # Disconnect executor
            if hasattr(self, 'executor') and self.executor.is_connected():
                self.executor.disconnect()
            
            # Close any OpenCV windows
            if self.render_mode == 'human':
                cv.destroyAllWindows()
            
            logger.info("Environment closed successfully")
            
        except Exception as e:
            logger.error(f"Error during close: {e}")
    
    def _navigate_to_battle(self):
        """Navigate from menu to battle and wait for match start."""
        logger.info("Navigating to battle...")
        
        # This is a simplified implementation
        # In a full implementation, we would:
        # 1. Detect if we're in the main menu
        # 2. Click the battle button
        # 3. Wait for match to start
        # 4. Detect when we're in the playing phase
        
        # For Phase 0, we'll assume we're already in a match or can start one
        self._game_phase = GamePhase.PLAYING
        self._match_start_time = time.time()
        
        logger.info("Navigation completed - ready to play")
    
    def _get_current_state(self) -> np.ndarray:
        """
        Get the current state from all perception components.
        
        Returns:
            53-dimensional state vector
        """
        try:
            # Capture frame
            frame = self.capture.grab()
            if frame is None:
                logger.warning("Failed to capture frame")
                return np.zeros(53, dtype=np.float32)
            
            # Detect cards in hand
            detected_cards = self.card_matcher.detect_hand_cards(frame)
            
            # Detect elixir
            elixir_count = self.perception.detect_elixir(frame)
            
            # Detect tower health
            tower_health = self.perception.detect_tower_health(frame)
            
            # Calculate match time
            match_time = 0.0
            if self._match_start_time is not None:
                match_time = time.time() - self._match_start_time
            
            # Build state vector
            state_vector = self.state_builder.build_state(
                frame=frame,
                detected_cards=detected_cards,
                elixir_count=elixir_count,
                tower_health=tower_health,
                match_time=match_time
            )
            
            # Update last tower health for reward calculation
            self._last_tower_health = tower_health
            
            # Ensure we have exactly 53 dimensions
            state = state_vector.vector
            if state.shape[0] != 53:
                logger.warning(f"State vector has {state.shape[0]} dimensions, expected 53")
                # Pad or truncate to 53 dimensions
                if state.shape[0] < 53:
                    state = np.pad(state, (0, 53 - state.shape[0]), 'constant')
                else:
                    state = state[:53]
            
            return state
            
        except Exception as e:
            logger.error(f"Failed to get current state: {e}")
            return np.zeros(53, dtype=np.float32)
    
    def _validate_action(self, card_slot: int, grid_x: int, grid_y: int) -> bool:
        """
        Validate the action parameters.
        
        Args:
            card_slot: Card slot (0-3)
            grid_x: Grid X coordinate (0-31)
            grid_y: Grid Y coordinate (0-17)
            
        Returns:
            True if action is valid, False otherwise
        """
        # Validate card_slot (0-3)
        if not (0 <= card_slot <= 3):
            return False
        
        # Validate grid coordinates
        if not (0 <= grid_x < 32) or not (0 <= grid_y < 18):
            return False
        
        # Check if we can afford the card (simplified check)
        # In a full implementation, we would check the current elixir
        # against the card cost from the detected cards
        
        return True
    
    def _execute_action(self, card_slot: int, grid_x: int, grid_y: int) -> bool:
        """
        Execute the action via the action executor.
        
        Args:
            card_slot: Card slot (0-3)
            grid_x: Grid X coordinate (0-31)
            grid_y: Grid Y coordinate (0-17)
            
        Returns:
            True if action was executed successfully, False otherwise
        """
        try:
            action = {
                'card_slot': card_slot,
                'grid_x': grid_x,
                'grid_y': grid_y
            }
            
            return self.executor.execute(action)
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return False
    
    def _get_tower_health(self) -> Dict[str, List[int]]:
        """
        Get current tower health for reward calculation.
        
        Returns:
            Dictionary with tower health values
        """
        if self._last_tower_health is None:
            return {
                'friendly': [0, 0, 0],
                'enemy': [0, 0, 0]
            }
        
        return self._last_tower_health
    
    def _calculate_reward(self, 
                         prev_tower_health: Dict[str, List[int]], 
                         action_success: bool) -> Tuple[float, bool, bool]:
        """
        Calculate reward based on tower damage and game outcome.
        
        Basic reward function:
        - Terminal: +1.0 (win), -1.0 (loss)
        - Shaping: +0.1 per enemy tower damage, -0.1 per friendly tower damage
        - Small penalty for failed actions
        
        Args:
            prev_tower_health: Previous tower health values
            action_success: Whether the action was executed successfully
            
        Returns:
            Tuple of (reward, terminated, truncated)
        """
        reward = 0.0
        terminated = False
        truncated = False
        
        # Get current tower health
        current_tower_health = self._get_tower_health()
        
        # Calculate tower damage
        if prev_tower_health and current_tower_health:
            # Enemy tower damage (positive reward)
            enemy_damage = 0
            for i in range(3):
                prev_health = prev_tower_health['enemy'][i] if i < len(prev_tower_health['enemy']) else 0
                curr_health = current_tower_health['enemy'][i] if i < len(current_tower_health['enemy']) else 0
                damage = prev_health - curr_health
                enemy_damage += max(0, damage)
            
            reward += enemy_damage * 0.1
            
            # Friendly tower damage (negative reward)
            friendly_damage = 0
            for i in range(3):
                prev_health = prev_tower_health['friendly'][i] if i < len(prev_tower_health['friendly']) else 0
                curr_health = current_tower_health['friendly'][i] if i < len(current_tower_health['friendly']) else 0
                damage = prev_health - curr_health
                friendly_damage += max(0, damage)
            
            reward -= friendly_damage * 0.1
        
        # Check for game end (simplified)
        # In a full implementation, we would detect victory/defeat screens
        if current_tower_health:
            # Check if any king tower is destroyed
            friendly_king_health = current_tower_health['friendly'][2] if len(current_tower_health['friendly']) > 2 else 0
            enemy_king_health = current_tower_health['enemy'][2] if len(current_tower_health['enemy']) > 2 else 0
            
            if friendly_king_health <= 0:
                # Loss
                reward -= 1.0
                terminated = True
                logger.info("Game ended - Defeat")
            elif enemy_king_health <= 0:
                # Win
                reward += 1.0
                terminated = True
                logger.info("Game ended - Victory")
        
        # Small penalty for failed actions
        if not action_success:
            reward -= 0.05
        
        # Check for timeout (truncated)
        if self._match_start_time and (time.time() - self._match_start_time) > 300:  # 5 minutes
            truncated = True
            logger.info("Game ended - Timeout")
        
        return reward, terminated, truncated
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the environment.
        
        Returns:
            Dictionary with performance metrics
        """
        metrics = {
            'episode_count': self._episode_count,
            'step_count': self._step_count,
            'game_phase': self._game_phase.value,
            'avg_step_time_ms': np.mean(self._step_times) if self._step_times else 0.0,
            'p95_step_time_ms': np.percentile(self._step_times, 95) if self._step_times else 0.0,
            'max_step_time_ms': max(self._step_times) if self._step_times else 0.0,
        }
        
        # Add component metrics
        if hasattr(self.card_matcher, 'get_performance_metrics'):
            metrics['card_matcher'] = self.card_matcher.get_performance_metrics()
        
        return metrics


# Phase 0 uses simplified BootstrapClashRoyaleEnv; full environment with advanced features added in Phase 1 (T023)