"""
Policy interfaces for Clash Royale RL Agent.

This module defines the base interfaces and configurations for policy implementations
used in the Clash Royale reinforcement learning agent.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class PolicyConfig:
    """Configuration for policy implementations.

    Attributes:
        device: The device to run the policy on ('cpu' or 'cuda'). If 'auto',
            it will auto-detect CUDA.
        deterministic: Whether to use deterministic actions by default.
        action_space: The dimensions of the action space, in the format
            [card_slots, grid_x, grid_y].
        state_dim: The dimension of the state vector.
    """

    device: str = 'auto'
    deterministic: bool = False
    action_space: Tuple[int, int, int] = (4, 32, 18)
    state_dim: int = 53
    
    def __post_init__(self):
        """Validates the configuration after initialization."""
        # Auto-detect CUDA if requested
        if self.device == 'auto':
            try:
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                self.device = 'cpu'
        
        if self.device not in ['cpu', 'cuda']:
            raise ValueError(f"Invalid device: {self.device}. Must be 'cpu', 'cuda', or 'auto'")
        
        if len(self.action_space) != 3:
            raise ValueError("Action space must be a tuple of 3 integers")


class Policy(ABC):
    """An abstract base class for policy implementations.

    This interface defines the contract that all policy implementations must
    adhere to in order to be compatible with the Clash Royale RL agent.

    Attributes:
        config: The policy configuration.
        device: The device on which the policy will run.
    """

    def __init__(self, config: PolicyConfig):
        """Initializes the policy.

        Args:
            config: The policy configuration.
        """
        self.config = config
        self.device = torch.device(config.device)
    
    @abstractmethod
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the policy network.

        Args:
            state: A state tensor with shape (batch_size, state_dim).

        Returns:
            An action logits tensor with shape (batch_size, 4, 32, 18).
        """
        pass
    
    @abstractmethod
    def act(self, state: np.ndarray, deterministic: Optional[bool] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Selects an action given the current state.

        Args:
            state: The current state as a NumPy array with shape (state_dim,).
            deterministic: Whether to use deterministic action selection.

        Returns:
            A tuple containing the action and an info dictionary. The action is a
            NumPy array with the format [card_slot, grid_x, grid_y].
        """
        pass
    
    @abstractmethod
    def get_action_logits(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gets the action logits for each action dimension.

        Args:
            state: A state tensor with shape (batch_size, state_dim).

        Returns:
            A tuple of tensors representing the action logits for card, x, and
            y.
        """
        pass
    
    def set_deterministic(self, deterministic: bool):
        """Sets the default deterministic behavior.

        Args:
            deterministic: Whether to use deterministic actions by default.
        """
        self.config.deterministic = deterministic
    
    def to(self, device: Union[str, torch.device]):
        """Moves the policy to a different device.

        Args:
            device: The target device.

        Returns:
            The policy, for method chaining.
        """
        if isinstance(device, str):
            device = torch.device(device)
        
        self.device = device
        self.config.device = str(device)
        return self
    
    def eval(self):
        """Sets the policy to evaluation mode."""
        pass

    def train(self):
        """Sets the policy to training mode."""
        pass

    def save(self, path: str):
        """Saves the policy to disk.

        Args:
            path: The path where to save the policy.
        """
        raise NotImplementedError("Save method not implemented")
    
    def load(self, path: str):
        """Loads the policy from disk.

        Args:
            path: The path from where to load the policy.
        """
        raise NotImplementedError("Load method not implemented")