"""
Enhanced TensorBoard Callbacks for Phase 0 Bootstrap Training (T013)

This module implements comprehensive TensorBoard callbacks for visual representation
of all training aspects including action heatmaps, card selection patterns,
elixir efficiency, and performance metrics.

Features:
- Action heatmap logging (32×18 grid)
- Card selection frequency tracking
- Elixir efficiency monitoring
- Performance benchmarking
- Model weight and gradient tracking
- Custom Phase 0 specific metrics
"""

import os
import time
import json
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

# Configure logging
logger = logging.getLogger(__name__)


class Phase0TensorBoardCallback(BaseCallback):
    """
    Comprehensive TensorBoard callback for Phase 0 bootstrap training.
    
    This callback logs detailed metrics and visual representations including:
    - Episode rewards and win rates
    - Action heatmaps (32×18 grid)
    - Card selection patterns
    - Elixir efficiency metrics
    - Performance benchmarks
    - Model weights and gradients
    - Custom Phase 0 specific metrics
    """
    
    def __init__(self, 
                 verbose: int = 1,
                 log_dir: str = "./logs/phase0_bootstrap",
                 heatmap_log_freq: int = 5000,
                 distribution_log_freq: int = 2000,
                 model_log_freq: int = 10000):
        """
        Initialize the Phase0TensorBoardCallback.
        
        Args:
            verbose: Verbosity level
            log_dir: Directory for TensorBoard logs
            heatmap_log_freq: Frequency for logging heatmaps (in steps)
            distribution_log_freq: Frequency for logging distributions (in steps)
            model_log_freq: Frequency for logging model metrics (in steps)
        """
        super().__init__(verbose)
        
        self.log_dir = log_dir
        self.heatmap_log_freq = heatmap_log_freq
        self.distribution_log_freq = distribution_log_freq
        self.model_log_freq = model_log_freq
        
        # Initialize tracking variables
        self.action_heatmap = np.zeros((32, 18))
        self.card_usage = np.zeros(5)  # 4 cards + no action
        self.elixir_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=1000)
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_history = deque(maxlen=100)
        
        # Performance tracking
        self.step_times = deque(maxlen=1000)
        self.action_latencies = deque(maxlen=1000)
        self.perception_latencies = deque(maxlen=1000)
        
        # State tracking
        self.state_vectors = deque(maxlen=500)
        self.elixir_efficiency = []
        
        # Model tracking
        self.model_weights_history = {}
        self.gradient_norms = deque(maxlen=100)
        
        # Custom metrics
        self.phase0_metrics = defaultdict(list)
        
        # Create SummaryWriter
        self.writer = None
        
        logger.info("Phase0TensorBoardCallback initialized")
    
    def _on_training_start(self) -> None:
        """Called when training starts."""
        # Create TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Log configuration
        if hasattr(self.training_env, 'envs'):
            env_config = getattr(self.training_env.envs[0], 'config', {})
            try:
                # Convert config to dictionary for JSON serialization
                if hasattr(env_config, '__dict__'):
                    config_dict = env_config.__dict__
                else:
                    config_dict = str(env_config)
                self.writer.add_text('Config/Environment', json.dumps(config_dict, indent=2), 0)
            except Exception as e:
                logger.warning(f"Could not log environment config to TensorBoard: {e}")
                self.writer.add_text('Config/Environment', 'EnvironmentConfig (not serializable)', 0)
        
        # Log model architecture
        if hasattr(self.model, 'policy'):
            model_info = self._get_model_info()
            self.writer.add_text('Model/Architecture', json.dumps(model_info, indent=2), 0)
        
        # Log initial metrics
        self.writer.add_scalar('Training/Start', 1, 0)
        self.writer.add_text('Training/Status', 'Started', 0)
        
        logger.info(f"TensorBoard logging started at {self.log_dir}")
    
    def _on_step(self) -> bool:
        """Called at each step."""
        current_step = self.num_timesteps
        
        # Collect step data
        self._collect_step_data()
        
        # Log scalar metrics
        self._log_scalar_metrics(current_step)
        
        # Log distributions
        if current_step % self.distribution_log_freq == 0:
            self._log_distributions(current_step)
        
        # Log heatmaps
        if current_step % self.heatmap_log_freq == 0:
            self._log_heatmaps(current_step)
        
        # Log model metrics
        if current_step % self.model_log_freq == 0:
            self._log_model_metrics(current_step)
        
        # Log performance metrics
        if current_step % 1000 == 0:
            self._log_performance_metrics(current_step)
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        # Process episode data
        if hasattr(self.training_env, 'envs'):
            env_stats = self.training_env.envs[0].get_performance_metrics()
            
            # Log episode statistics
            if 'episode_rewards' in env_stats:
                self.episode_rewards.extend(env_stats['episode_rewards'])
                if self.episode_rewards:
                    mean_reward = np.mean(self.episode_rewards[-100:])
                    self.writer.add_scalar('Performance/MeanReward', mean_reward, self.num_timesteps)
            
            if 'episode_lengths' in env_stats:
                self.episode_lengths.extend(env_stats['episode_lengths'])
                if self.episode_lengths:
                    mean_length = np.mean(self.episode_lengths[-100:])
                    self.writer.add_scalar('Performance/MeanEpisodeLength', mean_length, self.num_timesteps)
    
    def _on_training_end(self) -> None:
        """Called when training ends."""
        # Log final statistics
        self._log_final_statistics()
        
        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
        
        logger.info("TensorBoard logging completed")
    
    def _collect_step_data(self) -> None:
        """Collect data from the current step."""
        # Get current observation and action
        if 'obs' in self.locals:
            obs = self.locals['obs']
            if isinstance(obs, np.ndarray) and obs.ndim == 2:
                obs = obs[0]  # Take first observation if batched
                
                # Store state vector
                self.state_vectors.append(obs)
                
                # Extract elixir (index 0)
                elixir = obs[0]
                self.elixir_history.append(elixir)
        
        # Get action info
        if 'actions' in self.locals:
            actions = self.locals['actions']
            if isinstance(actions, np.ndarray) and actions.ndim == 2:
                actions = actions[0]  # Take first action if batched
                
                # Update action heatmap (grid position)
                if len(actions) >= 3:
                    grid_x, grid_y = actions[1], actions[2]
                    if 0 <= grid_x < 32 and 0 <= grid_y < 18:
                        self.action_heatmap[grid_x, grid_y] += 1
                
                # Update card usage
                if len(actions) >= 1:
                    card_slot = actions[0]
                    if 0 <= card_slot < 5:
                        self.card_usage[card_slot] += 1
        
        # Get reward info
        if 'rewards' in self.locals:
            rewards = self.locals['rewards']
            if isinstance(rewards, (list, np.ndarray)):
                reward = rewards[0] if rewards else 0
                self.reward_history.append(reward)
                
                # Track wins (reward > 0.5 indicates win)
                if reward > 0.5:
                    self.win_history.append(1)
                elif reward < -0.5:
                    self.win_history.append(0)
        
        # Track step timing
        step_start_time = getattr(self, '_last_step_time', time.time())
        current_time = time.time()
        step_time = (current_time - step_start_time) * 1000  # Convert to ms
        self.step_times.append(step_time)
        self._last_step_time = current_time
    
    def _log_scalar_metrics(self, step: int) -> None:
        """Log scalar metrics to TensorBoard."""
        # Episode metrics
        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-100:]
            self.writer.add_scalar('Episodes/MeanReward', np.mean(recent_rewards), step)
            self.writer.add_scalar('Episodes/StdReward', np.std(recent_rewards), step)
            self.writer.add_scalar('Episodes/MinReward', np.min(recent_rewards), step)
            self.writer.add_scalar('Episodes/MaxReward', np.max(recent_rewards), step)
        
        # Win rate
        if len(self.win_history) >= 10:
            recent_wins = list(self.win_history)[-50:]
            win_rate = np.mean(recent_wins)
            self.writer.add_scalar('Performance/WinRate', win_rate, step)
            
            # Log Phase 0 specific target
            target_win_rate = 0.35
            self.writer.add_scalar('Performance/WinRateGap', target_win_rate - win_rate, step)
        
        # Elixir metrics
        if self.elixir_history:
            recent_elixir = list(self.elixir_history)[-100:]
            self.writer.add_scalar('Elixir/MeanLevel', np.mean(recent_elixir), step)
            self.writer.add_scalar('Elixir/MaxLevel', np.max(recent_elixir), step)
            self.writer.add_scalar('Elixir/MinLevel', np.min(recent_elixir), step)
            
            # Elixir efficiency (elixir usage vs rewards)
            if len(self.reward_history) >= 10:
                recent_rewards = list(self.reward_history)[-100:]
                if recent_elixir and recent_rewards:
                    efficiency = np.mean(recent_rewards) / (np.mean(recent_elixir) + 1e-6)
                    self.writer.add_scalar('Elixir/Efficiency', efficiency, step)
        
        # Card usage distribution
        if np.sum(self.card_usage) > 0:
            card_probs = self.card_usage / np.sum(self.card_usage)
            for i, prob in enumerate(card_probs):
                card_name = f"Card_{i}" if i < 4 else "No_Action"
                self.writer.add_scalar(f'CardUsage/{card_name}', prob, step)
        
        # Training metrics from SB3
        if hasattr(self.model, 'logger'):
            if hasattr(self.model.logger, 'name_to_value'):
                for key, value in self.model.logger.name_to_value.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f'Training/{key}', value, step)
    
    def _log_distributions(self, step: int) -> None:
        """Log distributions to TensorBoard."""
        # Action distribution
        if np.sum(self.card_usage) > 0:
            self.writer.add_histogram('Actions/CardDistribution', self.card_usage, step)
        
        # Elixir distribution
        if self.elixir_history:
            elixir_array = np.array(list(self.elixir_history))
            self.writer.add_histogram('Elixir/Distribution', elixir_array, step)
        
        # Reward distribution
        if self.reward_history:
            reward_array = np.array(list(self.reward_history))
            self.writer.add_histogram('Rewards/Distribution', reward_array, step)
        
        # State vector distribution
        if self.state_vectors:
            state_array = np.array(list(self.state_vectors))
            if state_array.ndim == 2 and state_array.shape[1] == 53:
                # Log key state components
                self.writer.add_histogram('State/Elixir', state_array[:, 0], step)
                self.writer.add_histogram('State/MatchTime', state_array[:, 1], step)
                self.writer.add_histogram('State/TowerHealth', state_array[:, 2:5].flatten(), step)
    
    def _log_heatmaps(self, step: int) -> None:
        """Log heatmaps to TensorBoard."""
        # Action heatmap (32×18 grid)
        if np.sum(self.action_heatmap) > 0:
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Normalize and create colormap
            normalized_heatmap = self.action_heatmap / np.max(self.action_heatmap)
            
            # Use custom colormap for better visibility
            cmap = plt.cm.YlOrRd
            im = ax.imshow(normalized_heatmap.T, cmap=cmap, origin='lower', aspect='auto')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Action Frequency')
            
            # Labels and title
            ax.set_xlabel('Grid X (0-31)')
            ax.set_ylabel('Grid Y (0-17)')
            ax.set_title(f'Action Heatmap - Step {step}')
            
            # Add grid
            ax.set_xticks(range(0, 32, 4))
            ax.set_yticks(range(0, 18, 3))
            ax.grid(True, alpha=0.3)
            
            # Save to TensorBoard
            self.writer.add_figure('Actions/Heatmap', fig, step)
            plt.close(fig)
        
        # Card selection frequency
        if np.sum(self.card_usage) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            card_labels = ['Card 0', 'Card 1', 'Card 2', 'Card 3', 'No Action']
            card_counts = self.card_usage
            
            bars = ax.bar(card_labels, card_counts, color=['blue', 'green', 'red', 'purple', 'gray'])
            ax.set_xlabel('Card Selection')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Card Selection Frequency - Step {step}')
            
            # Add value labels on bars
            for bar, count in zip(bars, card_counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(count)}', ha='center', va='bottom')
            
            self.writer.add_figure('Cards/SelectionFrequency', fig, step)
            plt.close(fig)
    
    def _log_model_metrics(self, step: int) -> None:
        """Log model-specific metrics."""
        if not hasattr(self.model, 'policy'):
            return
        
        policy = self.model.policy
        
        # Log model weights
        for name, param in policy.named_parameters():
            if param.requires_grad and param.data is not None:
                self.writer.add_histogram(f'Model/Weights/{name}', param.data, step)
                
                # Log weight statistics
                self.writer.add_scalar(f'Model/WeightMean/{name}', param.data.mean(), step)
                self.writer.add_scalar(f'Model/WeightStd/{name}', param.data.std(), step)
        
        # Log gradients if available
        if hasattr(policy, 'optimizer') and policy.optimizer.param_groups[0]['params']:
            for i, param_group in enumerate(policy.optimizer.param_groups):
                for j, param in enumerate(param_group['params']):
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm(2).item()
                        self.writer.add_scalar(f'Model/GradNorm/Group{i}_Param{j}', grad_norm, step)
                        
                        # Log gradient histogram
                        self.writer.add_histogram(f'Model/Gradients/Group{i}_Param{j}', param.grad.data, step)
        
        # Log learning rate
        if hasattr(policy, 'optimizer'):
            for i, param_group in enumerate(policy.optimizer.param_groups):
                lr = param_group.get('lr', 0)
                self.writer.add_scalar(f'Model/LearningRate/Group{i}', lr, step)
    
    def _log_performance_metrics(self, step: int) -> None:
        """Log performance metrics."""
        # Step timing
        if self.step_times:
            step_times_array = np.array(list(self.step_times))
            self.writer.add_scalar('Performance/StepTimeMean', np.mean(step_times_array), step)
            self.writer.add_scalar('Performance/StepTimeStd', np.std(step_times_array), step)
            self.writer.add_scalar('Performance/StepTimeP95', np.percentile(step_times_array, 95), step)
            
            self.writer.add_histogram('Performance/StepTimeDistribution', step_times_array, step)
        
        # Environment performance
        if hasattr(self.training_env, 'envs'):
            env = self.training_env.envs[0]
            if hasattr(env, 'get_performance_metrics'):
                metrics = env.get_performance_metrics()
                
                # Log component metrics
                if 'card_matcher' in metrics:
                    card_metrics = metrics['card_matcher']
                    for key, value in card_metrics.items():
                        if isinstance(value, (int, float)):
                            self.writer.add_scalar(f'Components/CardMatcher/{key}', value, step)
    
    def _log_final_statistics(self) -> None:
        """Log final training statistics."""
        if not self.writer:
            return
        
        # Summary statistics
        final_stats = {
            'total_steps': self.num_timesteps,
            'total_episodes': len(self.episode_rewards),
            'mean_final_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'final_win_rate': np.mean(list(self.win_history)) if self.win_history else 0,
            'total_actions': int(np.sum(self.card_usage)),
            'mean_step_time_ms': np.mean(list(self.step_times)) if self.step_times else 0,
        }
        
        self.writer.add_text('Summary/FinalStatistics', json.dumps(final_stats, indent=2), 0)
        
        # Log action coverage
        if np.sum(self.action_heatmap) > 0:
            total_positions = 32 * 18
            used_positions = np.sum(self.action_heatmap > 0)
            coverage = used_positions / total_positions
            self.writer.add_scalar('Summary/ActionCoverage', coverage, 0)
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        if not hasattr(self.model, 'policy'):
            return {}
        
        policy = self.model.policy
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        
        return {
            'model_type': type(policy).__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(next(policy.parameters()).device) if policy.parameters() else 'unknown'
        }


class PerformanceBenchmarkCallback(BaseCallback):
    """
    Callback for benchmarking performance during training.
    
    Tracks system performance metrics including:
    - Action loop latency
    - Perception processing time
    - Memory usage
    - System resource utilization
    """
    
    def __init__(self, verbose: int = 1, log_freq: int = 1000):
        """
        Initialize the performance benchmark callback.
        
        Args:
            verbose: Verbosity level
            log_freq: Frequency for logging metrics (in steps)
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        
        # Performance tracking
        self.action_times = deque(maxlen=1000)
        self.perception_times = deque(maxlen=1000)
        self.memory_usage = deque(maxlen=1000)
        
        # System metrics
        self.cpu_usage = deque(maxlen=1000)
        self.process_times = deque(maxlen=1000)
    
    def _on_step(self) -> bool:
        """Called at each step."""
        current_step = self.num_timesteps
        
        # Collect performance metrics
        self._collect_performance_metrics()
        
        # Log metrics
        if current_step % self.log_freq == 0:
            self._log_performance_metrics()
        
        return True
    
    def _collect_performance_metrics(self) -> None:
        """Collect performance metrics."""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            self.cpu_usage.append(cpu_percent)
            
            # Memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_usage.append(memory_mb)
            
            # Process time
            self.process_times.append(time.time())
            
        except ImportError:
            # psutil not available, skip system metrics
            pass
    
    def _log_performance_metrics(self) -> None:
        """Log performance metrics to logger."""
        if self.cpu_usage:
            avg_cpu = np.mean(list(self.cpu_usage)[-100:])
            logger.info(f"Performance: Avg CPU usage: {avg_cpu:.2f}%")
        
        if self.memory_usage:
            avg_memory = np.mean(list(self.memory_usage)[-100:])
            logger.info(f"Performance: Avg memory usage: {avg_memory:.2f} MB")
        
        if self.action_times:
            avg_action_time = np.mean(list(self.action_times)[-100:])
            logger.info(f"Performance: Avg action time: {avg_action_time:.2f} ms")