"""
Basic Clash Royale Environment for future expansion.

This module contains the basic ClashRoyaleEnv class that will be expanded
in future phases. For Phase 0, the main implementation is in
src/bootstrap/bootstrap_env.py.

Phase 0 environment uses simplified state and reward; upgraded in Phase 1
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional


class ClashRoyaleEnv(gym.Env):
    """
    Basic Clash Royale Environment for future expansion.
    
    This is a placeholder implementation that will be expanded in future phases.
    For Phase 0, use BootstrapClashRoyaleEnv from src/bootstrap/bootstrap_env.py.
    
    Phase 0 environment uses simplified state and reward; upgraded in Phase 1
    """
    
    def __init__(self):
        """Initialize the basic Clash Royale environment."""
        super().__init__()
        
        # Action space: MultiDiscrete([4, 32, 18])
        # card_slot: 0-3 (4 visible cards)
        # grid_x: 0-31 (horizontal grid)
        # grid_y: 0-17 (vertical grid)
        self.action_space = spaces.MultiDiscrete([4, 32, 18])
        
        # Observation space: Box(low=-1, high=1, shape=(53,), dtype=np.float32)
        # 53-dimensional state vector from Phase 0
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(53,), 
            dtype=np.float32
        )
        
        # Environment state
        self._state = None
        self._done = False
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset state
        self._state = np.zeros(53, dtype=np.float32)
        self._done = False
        
        info = {
            'phase': 'placeholder',
            'message': 'Use BootstrapClashRoyaleEnv for Phase 0'
        }
        
        return self._state, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self._done:
            raise RuntimeError("Environment is done. Call reset() before step().")
        
        # Placeholder implementation
        reward = 0.0
        terminated = False
        truncated = False
        
        info = {
            'phase': 'placeholder',
            'message': 'Use BootstrapClashRoyaleEnv for Phase 0',
            'action': action
        }
        
        return self._state, reward, terminated, truncated, info
    
    def render(self, mode: str = 'human'):
        """Render the environment (placeholder)."""
        pass
    
    def close(self):
        """Close the environment."""
        pass