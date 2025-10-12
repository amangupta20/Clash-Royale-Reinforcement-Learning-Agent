"""
PPO Training Infrastructure for Bootstrap Agent (T012)

This module implements the PPO training infrastructure for the bootstrap agent,
including custom actor-critic policy and training utilities.

Phase 0 uses SB3 default hyperparameters; tuning in Phase 3
"""

import os
import time
import json
import logging
import subprocess
from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device
from torch.utils.tensorboard import SummaryWriter

# Import bootstrap components
from .bootstrap_env import BootstrapClashRoyaleEnv, EnvironmentConfig
from .mlp_policy import StructuredMLPPolicy, create_structured_mlp_policy
from policy.interfaces import PolicyConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """
    Configuration for PPO training.
    
    Phase 0 uses SB3 default hyperparameters; tuning in Phase 3
    """
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_sde: bool = False
    sde_sample_freq: int = -1
    target_kl: float = 0.01
    tensorboard_log: Optional[str] = None
    policy_kwargs: Optional[Dict[str, Any]] = None
    verbose: int = 1
    seed: Optional[int] = None
    device: str = "cpu"  # Force CPU for stability
    
    # Early stopping parameters
    early_stopping: bool = True
    early_stopping_patience: int = 100000  # 100K steps
    early_stopping_min_reward: float = 0.5
    
    # Checkpoint parameters
    checkpoint_freq: int = 10000  # Save every 10K steps
    checkpoint_dir: str = "./checkpoints"


class BootstrapActorCriticPolicy(ActorCriticPolicy):
    """
    Custom actor-critic policy that uses StructuredMLPPolicy as feature extractor.
    
    This policy combines the StructuredMLPPolicy from T011 with a value head
    to create an actor-critic architecture suitable for PPO training.
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: callable,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: type = nn.Tanh,
        *args,
        **kwargs,
    ):
        """
        Initialize the BootstrapActorCriticPolicy.
        
        Args:
            observation_space: Observation space
            action_space: Action space
            lr_schedule: Learning rate schedule
            net_arch: Network architecture (ignored, using StructuredMLPPolicy)
            activation_fn: Activation function (ignored, using StructuredMLPPolicy)
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        # Remove unused arguments
        kwargs.pop('features_extractor_class', None)
        kwargs.pop('features_extractor_kwargs', None)
        kwargs.pop('device', None)  # Remove device parameter as it's not accepted by SB3
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            activation_fn=activation_fn,
            *args,
            **kwargs,
        )
        
        # Initialize StructuredMLPPolicy as feature extractor
        # Update to support 5 card options (4 cards + 1 no action)
        policy_config = PolicyConfig(
            device=str(self.device),
            action_space=(5, 32, 18),  # Updated to support 5 options (4 cards + 1 no action)
            state_dim=53
        )
        self.structured_policy = create_structured_mlp_policy(policy_config)
        
        # Value head for critic
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        # Initialize value head
        for module in self.value_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy.
        
        Args:
            obs: Observation tensor
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (actions, values, log_prob)
        """
        # Extract features using StructuredMLPPolicy
        features = self._extract_features(obs)
        
        # Get action logits from structured policy (5 options: 4 cards + 1 no action)
        card_logits, x_logits, y_logits = self.structured_policy.get_action_logits(obs)
        
        # Apply action masking (now just returns the logits unchanged)
        final_card_logits = self.structured_policy._apply_action_masking(card_logits, obs)
        
        # Ensure we have 5 options (4 cards + 1 no action)
        if final_card_logits.shape[1] != 5:
            # If the structured policy doesn't support 5 options, add it manually
            if not hasattr(self, 'no_action_bias'):
                self.no_action_bias = nn.Parameter(torch.zeros(1, device=final_card_logits.device))
            
            final_card_logits = torch.cat([
                final_card_logits,
                self.no_action_bias.expand(final_card_logits.shape[0], 1)
            ], dim=1)
        
        # Sample actions
        if deterministic:
            card_action = torch.argmax(final_card_logits, dim=-1)
            x_action = torch.argmax(x_logits, dim=-1)
            y_action = torch.argmax(y_logits, dim=-1)
        else:
            # Handle case where all card logits are masked
            if torch.all(final_card_logits == float('-inf')):
                # Random choice between all 5 options
                card_action = torch.randint(0, 5, (obs.shape[0],), device=final_card_logits.device)
            else:
                # Replace -inf with a very large negative number for softmax
                safe_card_logits = torch.where(
                    final_card_logits == float('-inf'),
                    torch.tensor(-1e9, device=final_card_logits.device),
                    final_card_logits
                )
                # Ensure logits are valid for multinomial
                safe_card_logits = torch.clamp(safe_card_logits, min=-1e9, max=1e9)
                card_action = torch.multinomial(F.softmax(safe_card_logits, dim=-1), 1).squeeze(-1)
            
            # Ensure action is within valid range
            card_action = torch.clamp(card_action, 0, 4)
            
            x_action = torch.multinomial(F.softmax(x_logits, dim=-1), 1).squeeze(-1)
            y_action = torch.multinomial(F.softmax(y_logits, dim=-1), 1).squeeze(-1)
        
        # Debug: Log action selection
        if obs.shape[0] == 1:  # Only log for single batch
            current_elixir = obs[0, 0].item()
            logger.info(f"Current elixir: {current_elixir}, Selected card_slot: {card_action.item()}")
            detected_cards = obs[0, 13:17].float()
            logger.info(f"Detected cards: {detected_cards}")
            logger.info(f"Card logits: {final_card_logits[0]}")
            
            # Check if any cards are detected
            has_detected_cards = detected_cards.sum() > 0
            logger.info(f"Has detected cards: {has_detected_cards}")
            
            # Check card probabilities
            card_probs = F.softmax(final_card_logits, dim=-1)
            logger.info(f"Card probabilities: {card_probs[0]}")
        
        # Combine actions into MultiDiscrete format
        actions = torch.stack([card_action, x_action, y_action], dim=-1)
        
        # Calculate log probabilities
        card_log_prob = F.log_softmax(final_card_logits, dim=-1)
        x_log_prob = F.log_softmax(x_logits, dim=-1)
        y_log_prob = F.log_softmax(y_logits, dim=-1)
        
        # Gather log probabilities for taken actions
        action_log_prob = (
            card_log_prob[range(obs.shape[0]), card_action] +
            x_log_prob[range(obs.shape[0]), x_action] +
            y_log_prob[range(obs.shape[0]), y_action]
        )
        
        # Get value estimate
        values = self.value_head(features).squeeze(-1)
        
        # Log action choice to console
        if obs.shape[0] == 1:  # Only log for single batch
            action_type = "no action" if card_action.item() == 4 else f"card {card_action.item()}"
            logger.info(f"Action selected: {action_type} at grid ({x_action.item()}, {y_action.item()})")
        
        return actions, values, action_log_prob
    
    def _extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Extract features using the StructuredMLPPolicy.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Feature tensor
        """
        # Get the fused representation from StructuredMLPPolicy
        with torch.no_grad():
            # Forward pass to get internal representation
            global_state = obs[:, :13]
            hand_state = obs[:, 13:]
            cards = hand_state.view(-1, 4, 10)
            
            global_repr = self.structured_policy.global_processor(global_state)
            card_embeddings = self.structured_policy.card_encoder(cards)
            card_repr = card_embeddings.view(-1, 4 * 16)
            fused_repr = torch.cat([global_repr, card_repr], dim=1)
            features = self.structured_policy.fusion_layer(fused_repr)
        
        return features
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO training.
        
        Args:
            obs: Observation tensor
            actions: Action tensor
            
        Returns:
            Tuple of (values, log_prob, entropy)
        """
        # Extract features
        features = self._extract_features(obs)
        
        # Get action logits from structured policy (5 options: 4 cards + 1 no action)
        card_logits, x_logits, y_logits = self.structured_policy.get_action_logits(obs)
        
        # Apply action masking (now just returns the logits unchanged)
        final_card_logits = self.structured_policy._apply_action_masking(card_logits, obs)
        
        # Ensure we have 5 options (4 cards + 1 no action)
        if final_card_logits.shape[1] != 5:
            # If the structured policy doesn't support 5 options, add it manually
            if not hasattr(self, 'no_action_bias'):
                self.no_action_bias = nn.Parameter(torch.zeros(1, device=final_card_logits.device))
            
            final_card_logits = torch.cat([
                final_card_logits,
                self.no_action_bias.expand(final_card_logits.shape[0], 1)
            ], dim=1)
        
        # Replace -inf with a very large negative number for softmax calculations
        safe_card_logits = torch.where(
            final_card_logits == float('-inf'),
            torch.tensor(-1e9, device=final_card_logits.device),
            final_card_logits
        )
        
        # Ensure logits are valid
        safe_card_logits = torch.clamp(safe_card_logits, min=-1e9, max=1e9)
        
        # Calculate log probabilities
        card_log_prob = F.log_softmax(safe_card_logits, dim=-1)
        x_log_prob = F.log_softmax(x_logits, dim=-1)
        y_log_prob = F.log_softmax(y_logits, dim=-1)
        
        # Extract action components
        card_action = actions[:, 0].long()
        x_action = actions[:, 1].long()
        y_action = actions[:, 2].long()
        
        # Ensure actions are within valid range
        valid_card_actions = torch.clamp(card_action, 0, 4)  # Ensure within range (0-4)
        valid_x_actions = torch.clamp(x_action, 0, 31)  # Ensure within range (0-31)
        valid_y_actions = torch.clamp(y_action, 0, 17)  # Ensure within range (0-17)
        
        # Gather log probabilities for taken actions
        action_log_prob = (
            card_log_prob[range(obs.shape[0]), valid_card_actions] +
            x_log_prob[range(obs.shape[0]), valid_x_actions] +
            y_log_prob[range(obs.shape[0]), valid_y_actions]
        )
        
        # Calculate entropy
        card_entropy = -(F.softmax(safe_card_logits, dim=-1) * card_log_prob).sum(dim=-1)
        x_entropy = -(F.softmax(x_logits, dim=-1) * x_log_prob).sum(dim=-1)
        y_entropy = -(F.softmax(y_logits, dim=-1) * y_log_prob).sum(dim=-1)
        entropy = card_entropy + x_entropy + y_entropy
        
        # Get value estimate
        values = self.value_head(features).squeeze(-1)
        
        return values, action_log_prob, entropy
    
    def predict(self, obs: Union[np.ndarray, Dict[str, np.ndarray]], state: Optional[Tuple[np.ndarray, ...]] = None, 
                episode_start: Optional[np.ndarray] = None, deterministic: bool = False) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Predict actions for given observations.
        
        Args:
            obs: Observation
            state: Latent state (unused)
            episode_start: Episode start indicator (unused)
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (actions, states)
        """
        # Convert observation to tensor if needed
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
            # Add batch dimension if needed
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
        else:
            obs_tensor = obs
        
        # Get actions
        with torch.no_grad():
            actions, values, log_prob = self.forward(obs_tensor, deterministic=deterministic)
        
        # Convert actions to numpy arrays
        actions_np = actions.cpu().numpy()
        # Ensure actions are 1D arrays for MultiDiscrete
        if actions_np.ndim > 1:
            actions_np = actions_np.squeeze()
        
        return actions_np, None


class CheckpointCallback(BaseCallback):
    """
    Callback for saving checkpoints with metadata.
    """
    
    def __init__(self, save_freq: int, save_path: str, save_replay_buffer: bool = False, 
                 verbose: int = 1, **kwargs):
        """
        Initialize checkpoint callback.
        
        Args:
            save_freq: Save frequency in steps
            save_path: Directory to save checkpoints
            save_replay_buffer: Whether to save replay buffer
            verbose: Verbosity level
            **kwargs: Additional arguments
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_replay_buffer = save_replay_buffer
        
        # Create save directory
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Get git SHA for metadata
        try:
            self.git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                                cwd=os.getcwd()).decode('ascii').strip()
        except:
            self.git_sha = "unknown"
    
    def _on_step(self) -> bool:
        """Called at each step."""
        if self.n_calls % self.save_freq == 0:
            # Save checkpoint
            checkpoint_path = self.save_path / f"ppo_checkpoint_{self.n_calls}_steps"
            
            # Create metadata
            metadata = {
                'step': int(self.n_calls),
                'timestamp': float(time.time()),
                'git_sha': self.git_sha,
                'config': asdict(self.training_env.envs[0].config) if hasattr(self.training_env.envs[0], 'config') else {},
                'mean_reward': float(np.mean(self.locals.get('rewards', [0]))),
                'episode_length': float(np.mean(self.locals.get('episode_lengths', [0])))
            }
            
            # Save model
            self.model.save(str(checkpoint_path))
            
            # Save metadata
            with open(f"{checkpoint_path}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            if self.verbose > 0:
                logger.info(f"Saved checkpoint at step {self.n_calls} to {checkpoint_path}")
        
        return True


class EarlyStoppingCallback(BaseCallback):
    """
    Callback for early stopping based on reward plateau.
    """
    
    def __init__(self, patience: int, min_reward: float = 0.5, verbose: int = 1, **kwargs):
        """
        Initialize early stopping callback.
        
        Args:
            patience: Number of steps to wait for improvement
            min_reward: Minimum reward threshold
            verbose: Verbosity level
            **kwargs: Additional arguments
        """
        super().__init__(verbose)
        self.patience = patience
        self.min_reward = min_reward
        self.best_mean_reward = -np.inf
        self.last_improvement = 0
        self.no_improvement_count = 0
    
    def _on_step(self) -> bool:
        """Called at each step."""
        # Check if we have reward information
        if 'rewards' in self.locals:
            current_mean_reward = np.mean(self.locals['rewards'])
            
            # Check for improvement
            if current_mean_reward > self.best_mean_reward:
                self.best_mean_reward = current_mean_reward
                self.last_improvement = self.n_calls
                self.no_improvement_count = 0
                
                if self.verbose > 0:
                    logger.info(f"New best mean reward: {current_mean_reward:.3f} at step {self.n_calls}")
            
            # Check for early stopping
            elif current_mean_reward >= self.min_reward:
                self.no_improvement_count += 1
                
                if self.no_improvement_count >= self.patience:
                    if self.verbose > 0:
                        logger.info(f"Early stopping triggered at step {self.n_calls}")
                        logger.info(f"Best mean reward: {self.best_mean_reward:.3f}")
                    return False
        
        return True


class BootstrapPPOTrainer:
    """
    PPO trainer for the bootstrap agent.
    
    This class wraps SB3's PPO implementation and provides training infrastructure
    including logging, checkpointing, and early stopping.
    """
    
    def __init__(self, 
                 env_config: Optional[EnvironmentConfig] = None,
                 ppo_config: Optional[PPOConfig] = None,
                 tensorboard_log: Optional[str] = None):
        """
        Initialize the BootstrapPPOTrainer.
        
        Args:
            env_config: Environment configuration
            ppo_config: PPO configuration
            tensorboard_log: TensorBoard log directory
        """
        self.env_config = env_config or EnvironmentConfig()
        self.ppo_config = ppo_config or PPOConfig()
        
        # Override tensorboard log if provided
        if tensorboard_log:
            self.ppo_config.tensorboard_log = tensorboard_log
        
        # Create environment
        self.env = BootstrapClashRoyaleEnv(self.env_config)
        self.vec_env = DummyVecEnv([lambda: self.env])
        self.vec_env = VecMonitor(self.vec_env)
        
        # Create PPO model
        self.model = PPO(
            BootstrapActorCriticPolicy,
            self.vec_env,
            learning_rate=self.ppo_config.learning_rate,
            n_steps=self.ppo_config.n_steps,
            batch_size=self.ppo_config.batch_size,
            n_epochs=self.ppo_config.n_epochs,
            gamma=self.ppo_config.gamma,
            gae_lambda=self.ppo_config.gae_lambda,
            clip_range=self.ppo_config.clip_range,
            ent_coef=self.ppo_config.ent_coef,
            vf_coef=self.ppo_config.vf_coef,
            max_grad_norm=self.ppo_config.max_grad_norm,
            use_sde=self.ppo_config.use_sde,
            sde_sample_freq=self.ppo_config.sde_sample_freq,
            target_kl=self.ppo_config.target_kl,
            tensorboard_log=self.ppo_config.tensorboard_log,
            policy_kwargs=self.ppo_config.policy_kwargs,
            verbose=self.ppo_config.verbose,
            seed=self.ppo_config.seed,
            device=self.ppo_config.device
        )
        
        # Training statistics
        self.training_stats = {
            'start_time': None,
            'end_time': None,
            'total_timesteps': 0,
            'best_mean_reward': float(-np.inf),
            'checkpoints_saved': 0
        }
        
        logger.info("BootstrapPPOTrainer initialized")
        logger.info(f"Environment: {type(self.env).__name__}")
        logger.info(f"PPO Config: {asdict(self.ppo_config)}")
    
    def train(self, total_timesteps: int, reset_num_timesteps: bool = True) -> Dict[str, Any]:
        """
        Train the PPO model.
        
        Args:
            total_timesteps: Number of timesteps to train for
            reset_num_timesteps: Whether to reset timestep counter
            
        Returns:
            Dictionary with training statistics
        """
        logger.info(f"Starting training for {total_timesteps} timesteps")
        self.training_stats['start_time'] = time.time()
        self.training_stats['total_timesteps'] = total_timesteps
        
        # Create callbacks
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.ppo_config.checkpoint_freq,
            save_path=self.ppo_config.checkpoint_dir,
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping callback
        if self.ppo_config.early_stopping:
            early_stopping_callback = EarlyStoppingCallback(
                patience=self.ppo_config.early_stopping_patience,
                min_reward=self.ppo_config.early_stopping_min_reward,
                verbose=1
            )
            callbacks.append(early_stopping_callback)
        
        # Train model
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                reset_num_timesteps=reset_num_timesteps,
                log_interval=100
            )
            
            self.training_stats['end_time'] = time.time()
            training_time = self.training_stats['end_time'] - self.training_stats['start_time']
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Get final statistics from VecMonitor
            try:
                env_stats = self.vec_env.get_attr('episode_rewards')[0]
                if env_stats:
                    self.training_stats['final_mean_reward'] = float(np.mean(env_stats[-100:]) if len(env_stats) >= 100 else np.mean(env_stats))
                    self.training_stats['episodes_completed'] = int(len(env_stats))
            except:
                logger.warning("Could not retrieve episode statistics from VecMonitor")
            
            return self.training_stats
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def save_model(self, path: str, include_metadata: bool = True):
        """
        Save the trained model.
        
        Args:
            path: Path to save the model
            include_metadata: Whether to include metadata
        """
        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(path)
        
        # Save metadata if requested
        if include_metadata:
            metadata = {
                'training_stats': self.training_stats,
                'env_config': asdict(self.env_config),
                'ppo_config': asdict(self.ppo_config),
                'model_architecture': 'BootstrapActorCriticPolicy with StructuredMLPPolicy backbone',
                'timestamp': time.time(),
                'git_sha': self._get_git_sha()
            }
            
            with open(f"{path}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved to {path} with metadata")
        else:
            logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load a trained model.
        
        Args:
            path: Path to the saved model
        """
        self.model = PPO.load(path, env=self.vec_env)
        logger.info(f"Model loaded from {path}")
    
    def evaluate(self, n_eval_episodes: int = 10, deterministic: bool = True) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Args:
            n_eval_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic actions
            
        Returns:
            Dictionary with evaluation statistics
        """
        from stable_baselines3.common.evaluation import evaluate_policy
        
        episode_rewards, episode_lengths = evaluate_policy(
            self.model, 
            self.vec_env, 
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            return_episode_rewards=True
        )
        
        eval_stats = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'std_episode_length': np.std(episode_lengths),
            'episodes_evaluated': n_eval_episodes
        }
        
        logger.info(f"Evaluation results: {eval_stats}")
        return eval_stats
    
    def _get_git_sha(self) -> str:
        """Get current git SHA."""
        try:
            return subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                        cwd=os.getcwd()).decode('ascii').strip()
        except:
            return "unknown"
    
    def close(self):
        """Close the trainer and clean up resources."""
        if hasattr(self, 'vec_env'):
            self.vec_env.close()
        if hasattr(self, 'env'):
            self.env.close()
        logger.info("BootstrapPPOTrainer closed")


# Factory function for creating trainer instances
def create_bootstrap_ppo_trainer(
    env_config: Optional[EnvironmentConfig] = None,
    ppo_config: Optional[PPOConfig] = None,
    tensorboard_log: Optional[str] = None
) -> BootstrapPPOTrainer:
    """
    Create a BootstrapPPOTrainer instance.
    
    Args:
        env_config: Environment configuration
        ppo_config: PPO configuration
        tensorboard_log: TensorBoard log directory
        
    Returns:
        BootstrapPPOTrainer instance
    """
    return BootstrapPPOTrainer(env_config, ppo_config, tensorboard_log)