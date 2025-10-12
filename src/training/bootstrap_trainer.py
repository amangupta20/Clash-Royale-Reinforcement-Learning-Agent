"""
Basic Bootstrap Trainer for Phase 0 (T012)

This module provides a basic trainer interface that can be extended
for future training implementations beyond PPO.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BootstrapTrainer(ABC):
    """
    Abstract base class for bootstrap trainers.
    
    This class defines the interface that all trainers must implement
    for training RL agents in the Clash Royale environment.
    """
    
    def __init__(self, env, policy, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the trainer.
        
        Args:
            env: The environment to train in
            policy: The policy to train
            config: Training configuration (optional)
        """
        self.env = env
        self.policy = policy
        self.config = config or {}
        self.training_stats = {}
    
    @abstractmethod
    def train(self, total_timesteps: int, **kwargs) -> Dict[str, Any]:
        """
        Train the policy.
        
        Args:
            total_timesteps: Number of timesteps to train for
            **kwargs: Additional training arguments
            
        Returns:
            Dictionary with training statistics
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str, **kwargs):
        """
        Save a training checkpoint.
        
        Args:
            path: Path to save the checkpoint
            **kwargs: Additional checkpoint data
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str, **kwargs):
        """
        Load a training checkpoint.
        
        Args:
            path: Path to load the checkpoint from
            **kwargs: Additional loading options
        """
        pass
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get current training statistics.
        
        Returns:
            Dictionary with training statistics
        """
        return self.training_stats.copy()
    
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """
        Log training metrics.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Training step
        """
        for key, value in metrics.items():
            logger.info(f"Step {step} - {key}: {value}")
        
        # Update training stats
        self.training_stats[step] = metrics