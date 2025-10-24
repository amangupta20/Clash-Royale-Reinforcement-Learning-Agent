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
    """An abstract base class for bootstrap trainers.

    This class defines the interface that all trainers must implement for
    training RL agents in the Clash Royale environment.

    Attributes:
        env: The environment to train in.
        policy: The policy to train.
        config: The training configuration.
        training_stats: A dictionary to store training statistics.
    """

    def __init__(self, env, policy, config: Optional[Dict[str, Any]] = None):
        """Initializes the trainer.

        Args:
            env: The environment to train in.
            policy: The policy to train.
            config: The training configuration.
        """
        self.env = env
        self.policy = policy
        self.config = config or {}
        self.training_stats = {}
    
    @abstractmethod
    def train(self, total_timesteps: int, **kwargs) -> Dict[str, Any]:
        """Trains the policy.

        Args:
            total_timesteps: The number of timesteps to train for.
            **kwargs: Additional training arguments.

        Returns:
            A dictionary with training statistics.
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str, **kwargs):
        """Saves a training checkpoint.

        Args:
            path: The path where to save the checkpoint.
            **kwargs: Additional checkpoint data.
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str, **kwargs):
        """Loads a training checkpoint.

        Args:
            path: The path from where to load the checkpoint.
            **kwargs: Additional loading options.
        """
        pass
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Gets the current training statistics.

        Returns:
            A dictionary with the training statistics.
        """
        return self.training_stats.copy()
    
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Logs training metrics.

        Args:
            metrics: A dictionary of metrics to log.
            step: The training step.
        """
        for key, value in metrics.items():
            logger.info(f"Step {step} - {key}: {value}")
        
        # Update training stats
        self.training_stats[step] = metrics