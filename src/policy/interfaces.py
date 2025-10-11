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
    """
    Configuration for policy implementations.
    
    Attributes:
        device: Device to run the policy on ('cpu' or 'cuda'). Auto-detects CUDA if 'auto'
        deterministic: Whether to use deterministic actions by default
        action_space: Action space dimensions [card_slots, grid_x, grid_y]
        state_dim: Dimension of the state vector
    """
    device: str = 'auto'
    deterministic: bool = False
    action_space: Tuple[int, int, int] = (4, 32, 18)
    state_dim: int = 53
    
    def __post_init__(self):
        """Validate configuration after initialization."""
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
    """
    Abstract base class for policy implementations.
    
    This interface defines the contract that all policy implementations must follow
    to be compatible with the Clash Royale RL agent.
    """
    
    def __init__(self, config: PolicyConfig):
        """
        Initialize the policy.
        
        Args:
            config: Policy configuration
        """
        self.config = config
        self.device = torch.device(config.device)
    
    @abstractmethod
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the policy network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Action logits tensor of shape (batch_size, 4, 32, 18)
        """
        pass
    
    @abstractmethod
    def act(self, state: np.ndarray, deterministic: Optional[bool] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Select an action given the current state.
        
        Args:
            state: Current state as numpy array of shape (state_dim,)
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (action, info) where action is [card_slot, grid_x, grid_y]
        """
        pass
    
    @abstractmethod
    def get_action_logits(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action logits for each action dimension.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Tuple of (card_logits, x_logits, y_logits) tensors
        """
        pass
    
    def set_deterministic(self, deterministic: bool):
        """
        Set the default deterministic behavior.
        
        Args:
            deterministic: Whether to use deterministic actions by default
        """
        self.config.deterministic = deterministic
    
    def to(self, device: Union[str, torch.device]):
        """
        Move the policy to a different device.
        
        Args:
            device: Target device
            
        Returns:
            Self for method chaining
        """
        if isinstance(device, str):
            device = torch.device(device)
        
        self.device = device
        self.config.device = str(device)
        return self
    
    def eval(self):
        """Set the policy to evaluation mode."""
        pass
    
    def train(self):
        """Set the policy to training mode."""
        pass
    
    def save(self, path: str):
        """
        Save the policy to disk.
        
        Args:
            path: Path to save the policy
        """
        raise NotImplementedError("Save method not implemented")
    
    def load(self, path: str):
        """
        Load the policy from disk.
        
        Args:
            path: Path to load the policy from
        """
        raise NotImplementedError("Load method not implemented")