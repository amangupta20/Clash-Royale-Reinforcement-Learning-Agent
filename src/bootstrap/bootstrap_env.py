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
    action_delay_ms: float = 2500.0  # 1 second delay between actions
    
    # Game settings
    card_names: List[str] = None
    
    # Manual outcome input settings for Phase 0
    manual_outcome_input: bool = True
    outcome_check_delay_seconds: int = 120
    
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
        # 5 options for card_slot (0-3 for cards + 4 for no action)
        self.action_space = spaces.MultiDiscrete([5, 32, 18])
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
        self._zero_health_frames = 0
        
        # Manual win/loss tracking for Phase 0
        self._manual_outcome_input = self.config.manual_outcome_input
        self._outcome_check_delay = self.config.outcome_check_delay_seconds
        self._pending_outcome = None
        
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
        
        # Store last detected cards for action validation
        self._last_detected_cards = {}
        
        # Rate limiting for state building
        self._state_build_interval = 2.5  # 2.5 seconds between state builds
        self._cached_state = None
        
        # Action timing
        self._last_action_time = 0
        
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
        #time.sleep(1.0)  # Ensure minimum delay between actions
        if self._game_phase == GamePhase.ENDED:
            # Instead of raising an error, reset the environment automatically
            logger.info("Environment was in ENDED state, auto-resetting...")
            obs, info = self.reset(seed=None, options=None)
            return obs, 0.0, False, False, info
        
        step_start_time = time.perf_counter()
        self._step_count += 1
        
        # Log received action
        logger.info(f"Step {self._step_count}: Received action: {action}")
        
        try:
            # Extract action components
            if isinstance(action, np.ndarray):
                if action.ndim == 1:
                    card_slot = int(action[0])
                    grid_x = int(action[1])
                    grid_y = int(action[2])
                else:
                    # Handle case where action is 2D with batch dimension
                    card_slot = int(action[0, 0])
                    grid_x = int(action[0, 1])
                    grid_y = int(action[0, 2])
            else:
                # Handle case where action might be a list or other type
                card_slot = int(action[0])
                grid_x = int(action[1])
                grid_y = int(action[2])
            
            # Store current state for elixir penalty check
            self._current_state = self._get_current_state()
            
            # Check for elixir penalty (elixir = 10)
            elixir_penalty = 0
            if self._current_state is not None and len(self._current_state) > 0 and self._current_state[0] >= 9:
                elixir_penalty = -2
            
            # Handle "no action" (card_slot = 4)
            if card_slot == 4:
              #  time.sleep(0.5)
                obs = self._get_current_state()
                reward = elixir_penalty  # Only elixir penalty
                terminated = False
                truncated = False
                
                info = {
                    'step_count': self._step_count,
                    'game_phase': self._game_phase.value,
                    'action_valid': True,
                    'action': {'card_slot': card_slot, 'grid_x': grid_x, 'grid_y': grid_y},
                    'action_type': 'no_action',
                    'elixir_penalty': elixir_penalty
                }
                
                logger.info(f"No action taken (step {self._step_count})")
                return obs, reward, terminated, truncated, info
            
            # Validate action
            if not self._validate_action(card_slot, grid_x, grid_y):
                logger.warning(f"Invalid action: [{card_slot}, {grid_x}, {grid_y}]")
                reward = -0.1 + elixir_penalty  # Invalid action penalty + elixir penalty
                terminated = False
                truncated = False
                obs = self._get_current_state()
                
                info = {
                    'step_count': self._step_count,
                    'game_phase': self._game_phase.value,
                    'action_valid': False,
                    'action': {'card_slot': card_slot, 'grid_x': grid_x, 'grid_y': grid_y},
                    'elixir_penalty': elixir_penalty
                }
                
                return obs, reward, terminated, truncated, info
            
            # Store previous tower health for reward calculation
            prev_tower_health = self._get_tower_health()
            
            # Execute action
            action_success = self._execute_action(card_slot, grid_x, grid_y)
            if card_slot != 4: # Only wait if we actually played a card
                time.sleep(1)

            
            # Get current state
            obs = self._get_current_state()
            
            logger.info(f"Action executed: card_slot={card_slot}, grid_x={grid_x}, grid_y={grid_y}, success={action_success}")
            
            # Calculate reward
            reward, terminated, truncated = self._calculate_reward(prev_tower_health, action_success)
            
            # Add elixir penalty
            elixir_penalty = 0
            if self._current_state is not None and len(self._current_state) > 0 and self._current_state[0] >= 10:
                elixir_penalty = -0.01
            reward += elixir_penalty
            
            # Debug: Log reward and termination
            logger.info(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
            
            # Update game phase if needed
            if terminated:
                self._game_phase = GamePhase.ENDED
                logger.info(f"Game phase updated to ENDED at step {self._step_count}")
            
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
        
        # This is a simplified implementation for Phase 0
        # In Phase 0, we assume the user will manually navigate to battles
        # or the agent will train in whatever state it finds
        
        # Set to playing phase to allow actions
        self._game_phase = GamePhase.PLAYING
        self._match_start_time = time.time()
        
        # Reset termination flags
        self._pending_outcome = None
        
        # Ensure action cooldown is reset
        self._last_action_time = 0.0
        
        logger.info("Navigation completed - ready to play")
        logger.info("Note: Make sure Clash Royale is in a playable state")
        
        # Get current state to log elixir and cards
        try:
            current_state = self._get_current_state()
            logger.info(f"Elixir: {current_state[0]}")
            logger.info(f"Detected cards: {np.sum(current_state[13:17])}")
        except Exception as e:
            logger.warning(f"Could not get current state for logging: {e}")
    
    def _get_current_state(self) -> np.ndarray:
        """
        Get the current state from all perception components.
        Uses rate limiting to avoid continuous OCR and template matching.
        
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
            logger.info(f"Detected cards: {detected_cards}")
            # Store detected cards for action validation
            self._last_detected_cards = detected_cards
            
            # Debug: Log detected cards
            if self._step_count % 100 == 0:  # Log every 100 steps to avoid spam
                logger.info(f"Detected cards in slots: {list(detected_cards.keys())}")
            
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
            
            # Update rate limiting info
            
            # Ensure we have exactly 53 dimensions
            state = state_vector.vector
            if state.shape[0] != 53:
                logger.warning(f"State vector has {state.shape[0]} dimensions, expected 53")
                # Pad or truncate to 53 dimensions
                if state.shape[0] < 53:
                    state = np.pad(state, (0, 53 - state.shape[0]), 'constant')
                else:
                    state = state[:53]
            
            # Cache the state
            self._cached_state = state
            
            return state
            
        except Exception as e:
            logger.error(f"Failed to get current state: {e}")
            return np.zeros(53, dtype=np.float32)
    
    def _validate_action(self, card_slot: int, grid_x: int, grid_y: int) -> bool:
        """
        Validate the action parameters.
        
        Args:
            card_slot: Card slot (0-3 for cards, 4 for no action)
            grid_x: Grid X coordinate (0-31)
            grid_y: Grid Y coordinate (0-17)
            
        Returns:
            True if action is valid, False otherwise
        """
        # Check for "no action" option
        if card_slot == 4:
            # No action is always valid, grid coordinates are ignored
            return True
        
        # Validate card_slot (0-3)
        if not (0 <= card_slot <= 3):
            return False
        
        # Check if there's a card detected in this slot
        # Only validate if not "no action"
        '''
        if card_slot != 4:
            #time.sleep(1)
            # Template matcher uses 1-based indexing (1-4), so adjust for 0-based (0-3)
            template_slot = card_slot + 1
            
            # Fallback: If no cards are detected at all, allow any action
            # This prevents the agent from getting stuck
            if not self._last_detected_cards:
                logger.warning(f"No cards detected at all, allowing action in slot {card_slot}")
                return True

            if template_slot not in self._last_detected_cards:
                logger.warning(f"No card detected in slot {card_slot} (template slot {template_slot}). Available slots: {list(self._last_detected_cards.keys())}")
                # Allow action anyway to prevent getting stuck
                return True
            
            # Check if the detected card is valid
            detected_card = self._last_detected_cards[template_slot]
            if not detected_card:
                logger.warning(f"No card detected in slot {card_slot} (template slot {template_slot})")
                return True  # Allow action to prevent getting stuck
            
            # Template matcher returns 'card_id', not 'card_name'
            if 'card_id' not in detected_card:
                logger.warning(f"Invalid card detection in slot {card_slot} (template slot {template_slot}): {detected_card}")
                return True  # Allow action to prevent getting stuck
        '''
        # Validate grid coordinates
        if not (0 <= grid_x < 32) or not (0 <= grid_y < 18):
            return False
        
        # Check if the card is actually detected
        template_slot = card_slot + 1
        detected_card = self._last_detected_cards.get(template_slot)
        if not detected_card or 'card_id' not in detected_card:
            logger.warning(f"Card slot {card_slot} is not detected in hand")
            return False
        if self._last_detected_cards[template_slot]['card_id'] is None:
            logger.warning(f"No card detected in slot {card_slot} (template slot {template_slot}). Available slots: {list(self._last_detected_cards.keys())}")
            return False
        
        # Check if we can afford the card
        # Get current elixir from state
        current_elixir = 0
        if hasattr(self, '_current_state') and self._current_state is not None and len(self._current_state) > 0:
            current_elixir = self._current_state[0]  # Elixir is at index 0
        
        # Get card cost from detected card
        card_id = detected_card['card_id']
        if card_id in self._card_costs:
            if current_elixir < self._card_costs[card_id]:
                logger.warning(f"Not enough elixir for {card_id} (need {self._card_costs[card_id]}, have {current_elixir})")
                return False
        
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
            # Handle "no action" case - don't execute anything
            if card_slot == 4:
                return True
            
            # Convert 0-based slot to 1-based for executor (template matcher uses 1-based)
            action = {
                'card_slot': card_slot,  # Convert to 1-based (1-4)
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
                try:
                    prev_health = int(prev_tower_health['enemy'][i]) if i < len(prev_tower_health['enemy']) else 0
                    curr_health = int(current_tower_health['enemy'][i]) if i < len(current_tower_health['enemy']) else 0
                    damage = prev_health - curr_health
                    enemy_damage += max(0, damage)
                except (TypeError, ValueError, IndexError):
                    # Skip if we can't parse the health value
                    continue
            
            reward += enemy_damage * 0.1
            
            # Friendly tower damage (negative reward)
            friendly_damage = 0
            for i in range(3):
                try:
                    prev_health = int(prev_tower_health['friendly'][i]) if i < len(prev_tower_health['friendly']) else 0
                    curr_health = int(current_tower_health['friendly'][i]) if i < len(current_tower_health['friendly']) else 0
                    damage = prev_health - curr_health
                    friendly_damage += max(0, damage)
                except (TypeError, ValueError, IndexError):
                    # Skip if we can't parse the health value
                    continue
            
            reward -= friendly_damage * 0.1
        
        # Check for game end (simplified)
        # In a full implementation, we would detect victory/defeat screens
        
            # Only terminate if a king tower is actually destroyed (health <= 0)
           
        
        # Manual outcome input for Phase 0 (when automatic detection is not available)
        if self._manual_outcome_input and not terminated:
            # Check if we should prompt for manual outcome
            if self._match_start_time:
                match_duration = time.time() - self._match_start_time
                # Log match time periodically
                if self._step_count % 100 == 0:  # Every 100 steps
                    logger.info(f"Match duration: {match_duration:.1f}s, check at {self._outcome_check_delay}s")
                
                # Check if match has been running long enough to potentially be finished
                if match_duration > self._outcome_check_delay:  # Use configurable delay
                    if self._pending_outcome is None:
                        logger.info("=" * 60)
                        logger.info(f"MANUAL OUTCOME INPUT REQUIRED (after {match_duration:.1f}s)")
                        logger.info("=" * 60)
                        logger.info("Please enter the game outcome:")
                        logger.info("  1 - Win")
                        logger.info("  2 - Loss")
                        logger.info("  3 - Continue playing")
                        logger.info("  0 - Skip/Cancel")
                        logger.info("=" * 60)
                        
                        outcome = self._request_manual_outcome()
                        if outcome is not None:
                            if outcome == 'win':
                                reward += 1.0
                                terminated = True
                                logger.info("Game ended - Manual Victory Input")
                            elif outcome == 'loss':
                                reward -= 1.0
                                terminated = True
                                logger.info("Game ended - Manual Defeat Input")
                            elif outcome == 'continue':
                                # Continue playing and reset timer
                                self._match_start_time = time.time()  # Reset timer
                                logger.info("Continuing play - timer reset")
                            elif outcome == 'skip':
                                # Skip this time and reset timer
                                self._match_start_time = time.time()  # Reset timer
                                logger.info("Skipped input - timer reset")
                            self._pending_outcome = outcome
            else:
                # No match start time, set it now
                self._match_start_time = time.time()
                logger.info("Match timer started - manual input will be available after delay")
        
        # Small penalty for failed actions
        if not action_success:
            reward -= 0.05
        
        # Check for timeout (truncated)
        if self._match_start_time and (time.time() - self._match_start_time) > 120:  # 2 minutes
            truncated = True
            logger.info(f"Game ended - Timeout after {time.time() - self._match_start_time:.2f} seconds")
        else:
            # Debug: Log match time
            if self._match_start_time:
                match_time = time.time() - self._match_start_time
                logger.info(f"Match time: {match_time:.2f} seconds")
        
        # Debug: Log final reward and termination status
        logger.info(f"Final reward: {reward}, terminated: {terminated}, truncated: {truncated}")
        
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
    
    def _request_manual_outcome(self) -> Optional[str]:
        """
        Request manual outcome input from user for Phase 0 training.
        
        Returns:
            'win', 'loss', 'continue', or None
        """
        logger.info("=" * 50)
        logger.info("MANUAL OUTCOME INPUT REQUIRED")
        logger.info("=" * 50)
        logger.info("Please enter the game outcome:")
        logger.info("  1 - Win")
        logger.info("  2 - Loss")
        logger.info("  3 - Continue playing")
        logger.info("  0 - Skip/Cancel")
        logger.info("=" * 50)
        
        try:
            # Try to get input from user
            user_input = input("Enter outcome (1/2/3/0): ").strip()
            
            if user_input == '1':
                return 'win'
            elif user_input == '2':
                return 'loss'
            elif user_input == '3':
                return 'continue'
            elif user_input == '0':
                return None
            else:
                logger.warning(f"Invalid input: {user_input}. Skipping manual outcome.")
                return None
                
        except (EOFError, KeyboardInterrupt):
            logger.info("Input interrupted. Skipping manual outcome.")
            return None
        except Exception as e:
            logger.error(f"Error getting manual outcome: {e}")
            return None
    
    def set_manual_outcome(self, outcome: str) -> None:
        """
        Set manual outcome programmatically (for testing or automated input).
        
        Args:
            outcome: 'win', 'loss', or 'continue'
        """
        if outcome in ['win', 'loss', 'continue']:
            self._pending_outcome = outcome
            logger.info(f"Manual outcome set to: {outcome}")
        else:
            logger.warning(f"Invalid outcome: {outcome}")


# Phase 0 uses simplified BootstrapClashRoyaleEnv; full environment with advanced features added in Phase 1 (T023)