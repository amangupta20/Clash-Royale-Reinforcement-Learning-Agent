"""
Bootstrap Training Script for Phase 0 (T013)

This script implements the main training loop for the Phase 0 bootstrap experiment
with comprehensive TensorBoard logging and training continuation support.

Usage:
    python -m src.bootstrap.train_bootstrap.py --config research/configs/phase0_bootstrap.yaml

Features:
- Load configuration from YAML file
- Training continuation from checkpoints
- Comprehensive TensorBoard logging
- Performance benchmarking
- Experiment tracking in experiments.jsonl
- Automatic checkpointing and best model saving
"""

import os
import sys
import json
import time
import argparse
import hashlib
import logging
import subprocess
import signal
from pathlib import Path
from typing import Dict, Any, Optional

# Import BaseCallback for custom shutdown handling
try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    # Fallback for environments where SB3 might not be fully installed
    class BaseCallback:
        """Fallback BaseCallback class"""
        def __init__(self, verbose: int = 0):
            self.verbose = verbose
            self.training_env = None
            self.model = None
        
        def _on_step(self) -> bool:
            return True

import yaml
import torch
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import bootstrap components
from bootstrap.bootstrap_trainer import create_bootstrap_ppo_trainer, PPOConfig
from bootstrap.bootstrap_env import BootstrapClashRoyaleEnv, EnvironmentConfig
from bootstrap.tensorboard_callbacks import Phase0TensorBoardCallback, PerformanceBenchmarkCallback
from policy.interfaces import PolicyConfig

# Configure basic logging (will be updated later)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global flag for clean shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handles Ctrl+C and other signals for a clean shutdown.

    Args:
        signum: The signal number.
        frame: The current stack frame.
    """
    global shutdown_requested
    logger.info("Shutdown requested. Cleaning up...")
    shutdown_requested = True


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal


def _save_exit_checkpoint(trainer, exit_reason: str):
    """Saves a checkpoint when exiting training.

    This function is always called upon exiting training to ensure progress is
    saved.

    Args:
        trainer: The trainer instance.
        exit_reason: The reason for exiting, e.g., "shutdown", "interrupt",
            or "error".
    """
    try:
        # Get current training info
        current_timesteps = trainer.model.num_timesteps if hasattr(trainer.model, 'num_timesteps') else 0
        
        # Create exit checkpoint path
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_path = Path(trainer.ppo_config.checkpoint_dir) / f"exit_checkpoint_{exit_reason}_{timestamp}"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        logger.info(f"Saving exit checkpoint ({exit_reason}) at step {current_timesteps} to {checkpoint_path}")
        trainer.model.save(str(checkpoint_path))
        
        # Create metadata
        metadata = {
            'exit_reason': exit_reason,
            'step': int(current_timesteps),
            'timestamp': float(time.time()),
            'git_sha': trainer._get_git_sha() if hasattr(trainer, '_get_git_sha') else "unknown",
            'exit_time': time.strftime("%Y-%m-%d %H:%M:%S"),
            'mean_reward': 0.0,  # Would need to be tracked separately
            'episode_length': 0.0  # Would need to be tracked separately
        }
        
        # Save metadata
        with open(f"{checkpoint_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Also save as the latest checkpoint for easy access
        latest_path = Path(trainer.ppo_config.checkpoint_dir) / f"latest_checkpoint_{exit_reason}"
        trainer.model.save(str(latest_path))
        
        # Save metadata for latest checkpoint
        with open(f"{latest_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Exit checkpoint saved successfully")
        logger.info(f"Latest checkpoint: {latest_path}")
        
    except Exception as e:
        logger.error(f"Failed to save exit checkpoint: {e}")
        import traceback
        traceback.print_exc()


def setup_logging(log_dir: str = './logs/phase0_bootstrap') -> None:
    """Sets up logging with a file handler.

    This function is called after the necessary directories have been created.

    Args:
        log_dir: The directory for the log files.
    """
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Remove any existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configure handlers
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, 'train.log'))
    ]
    
    # Set up handlers with proper formatting
    for handler in handlers:
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
        force=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured. Log file: {os.path.join(log_dir, 'train.log')}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads the configuration from a YAML file.

    Args:
        config_path: The path to the configuration file.

    Returns:
        A dictionary with the configuration parameters.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded from {config_path}")
    return config


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Finds the latest checkpoint in the specified directory.

    Args:
        checkpoint_dir: The directory to search for checkpoints.

    Returns:
        The path to the latest checkpoint, or None if no checkpoint is found.
    """
    import glob
    import os
    
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Look for .zip files (SB3 model format)
    pattern = os.path.join(checkpoint_dir, "best_model_*.zip")
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        # Look for PPO checkpoints
        pattern = os.path.join(checkpoint_dir, "ppo_checkpoint_*_steps.zip")
        checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return None
    
    # Sort by modification time and get the latest
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    logger.info(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def create_directories(config: Dict[str, Any]) -> None:
    """Creates the necessary directories for training.

    Args:
        config: The configuration dictionary.
    """
    dirs_to_create = [
        config['training']['checkpoint_dir'],
        config['training']['tensorboard_log_dir'],
        config['outputs']['output_dir'],
        os.path.dirname('./logs/phase0_bootstrap/train.log')
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")


def get_git_sha() -> str:
    """Gets the current Git SHA for experiment tracking.

    Returns:
        The Git SHA as a string, or 'unknown' if Git is not available.
    """
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                    cwd=os.getcwd()).decode('ascii').strip()
    except:
        return "unknown"


def calculate_config_hash(config: Dict[str, Any]) -> str:
    """Calculates a hash of the configuration for reproducibility.

    Args:
        config: The configuration dictionary.

    Returns:
        The SHA hash of the configuration.
    """
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:10]


def log_experiment(config: Dict[str, Any],
                   git_sha: str,
                   config_hash: str,
                   results: Dict[str, Any]) -> None:
    """Logs experiment details to experiments.jsonl.

    Args:
        config: The configuration dictionary.
        git_sha: The Git SHA.
        config_hash: The configuration hash.
        results: The training results.
    """
    experiment_entry = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'name': config['experiment']['name'],
        'phase': config['experiment']['phase'],
        'target_win_rate': config['experiment']['target_win_rate'],
        'git_sha': git_sha,
        'config_hash': config_hash,
        'config': config,
        'results': results,
        'description': config['experiment']['description']
    }
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if hasattr(obj, 'dtype'):
            return float(obj)  # Convert numpy scalar to Python float
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_serializable(item) for item in obj)
        else:
            return obj
    
    # Convert results to serializable format
    serializable_entry = convert_to_serializable(experiment_entry)
    
    # Append to experiments.jsonl
    try:
        with open('research/experiments.jsonl', 'a') as f:
            f.write(json.dumps(serializable_entry) + '\n')
        
        logger.info(f"Experiment logged to research/experiments.jsonl")
    except Exception as e:
        logger.error(f"Failed to log experiment: {e}")
        # Try to log a simplified version
        try:
            simplified_entry = {
                'timestamp': experiment_entry['timestamp'],
                'name': experiment_entry.get('name', 'unknown'),
                'phase': experiment_entry.get('phase', 'unknown'),
                'git_sha': experiment_entry.get('git_sha', 'unknown'),
                'config_hash': experiment_entry.get('config_hash', 'unknown'),
                'error': 'Failed to serialize full results'
            }
            with open('research/experiments.jsonl', 'a') as f:
                f.write(json.dumps(simplified_entry) + '\n')
            logger.info(f"Simplified experiment logged to research/experiments.jsonl")
        except Exception as e2:
            logger.error(f"Failed to log even simplified experiment: {e2}")


def create_env_config(config: Dict[str, Any]) -> EnvironmentConfig:
    """Creates an environment configuration from a YAML config.

    Args:
        config: The YAML configuration dictionary.

    Returns:
        An `EnvironmentConfig` instance.
    """
    env_config_dict = config['environment']
    env_config = EnvironmentConfig(
        window_name=env_config_dict['window_name'],
        resolution=env_config_dict['resolution'],
        roi=env_config_dict['roi'],
        max_step_time_ms=env_config_dict['max_step_time_ms'],
        action_delay_ms=env_config_dict['action_delay_ms'],
        card_names=env_config_dict['card_names']
    )
    
    # Add manual outcome input configuration if specified
    if 'manual_outcome_input' in env_config_dict:
        env_config.manual_outcome_input = env_config_dict['manual_outcome_input']
    if 'outcome_check_delay_seconds' in env_config_dict:
        env_config.outcome_check_delay_seconds = env_config_dict['outcome_check_delay_seconds']
    
    return env_config


def create_ppo_config(config: Dict[str, Any]) -> PPOConfig:
    """Creates a PPO configuration from a YAML config.

    Args:
        config: The YAML configuration dictionary.

    Returns:
        A `PPOConfig` instance.
    """
    training_config = config['training']
    ppo_config_dict = config['ppo']
    
    return PPOConfig(
        learning_rate=float(ppo_config_dict['learning_rate']),
        n_steps=ppo_config_dict['n_steps'],
        batch_size=ppo_config_dict['batch_size'],
        n_epochs=ppo_config_dict['n_epochs'],
        gamma=ppo_config_dict['gamma'],
        gae_lambda=ppo_config_dict['gae_lambda'],
        clip_range=ppo_config_dict['clip_range'],
        ent_coef=ppo_config_dict['ent_coef'],
        vf_coef=ppo_config_dict['vf_coef'],
        max_grad_norm=ppo_config_dict['max_grad_norm'],
        use_sde=ppo_config_dict['use_sde'],
        sde_sample_freq=ppo_config_dict['sde_sample_freq'],
        target_kl=ppo_config_dict['target_kl'],
        tensorboard_log=training_config['tensorboard_log_dir'],
        verbose=training_config['verbose'],
        device=config['system']['device'],
        early_stopping=training_config['early_stopping'],
        early_stopping_patience=training_config['early_stopping_patience'],
        early_stopping_min_reward=training_config['early_stopping_min_reward'],
        checkpoint_freq=training_config['checkpoint_freq'],
        checkpoint_dir=training_config['checkpoint_dir']
    )


def setup_callbacks(config: Dict[str, Any]) -> list:
    """Sets up the training callbacks.

    Args:
        config: The configuration dictionary.

    Returns:
        A list of callback instances.
    """
    callbacks = []
    # Enhanced TensorBoard callback
    tensorboard_callback = Phase0TensorBoardCallback(
        verbose=1,
        log_dir=config['training']['tensorboard_log_dir'],
        heatmap_log_freq=config['logging']['heatmap_log_freq'],
        distribution_log_freq=config['logging']['distribution_log_freq'],
        model_log_freq=config['logging']['model_log_freq']
    )
    callbacks.append(tensorboard_callback)
    
    # Performance benchmark callback
    if config['training']['performance_benchmarking']:
        perf_callback = PerformanceBenchmarkCallback(
            verbose=1,
            log_freq=2000
        )
        callbacks.append(perf_callback)
    
    return callbacks


def train_model(config: Dict[str, Any],
                continue_training: bool = False,
                checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
    """Trains the model with the given configuration.

    Args:
        config: The configuration dictionary.
        continue_training: Whether to continue from a checkpoint.
        checkpoint_path: The path to the checkpoint to continue from.

    Returns:
        A dictionary with the training results.
    """
    logger.info("Starting model training...")
    
    # Create configurations
    env_config = create_env_config(config)
    ppo_config = create_ppo_config(config)
    
    # Create trainer
    trainer = create_bootstrap_ppo_trainer(
        env_config=env_config,
        ppo_config=ppo_config,
        tensorboard_log=config['training']['tensorboard_log_dir']
    )
    
    # Load checkpoint if continuing training
    if continue_training and checkpoint_path:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        trainer.load_model(checkpoint_path)
    
    # Set up callbacks
    callbacks = setup_callbacks(config)
    
    # Add a custom callback to handle shutdown
    class ShutdownCallback(BaseCallback):
        def __init__(self, verbose=1):
            super().__init__(verbose)
            self.training_stopped = False
        
        def _on_step(self) -> bool:
            # Check for shutdown flag
            if shutdown_requested:
                logger.info("Shutdown detected during training, stopping...")
                self.training_stopped = True
                return False  # Stop training
            return True
    
    shutdown_callback = ShutdownCallback()
    callbacks.append(shutdown_callback)
    
    # Training parameters
    total_timesteps = config['training']['total_timesteps']
    reset_num_timesteps = not continue_training
    
    # Start training with shutdown handling
    start_time = time.time()
    
    try:
        # Check for shutdown before starting
        if shutdown_requested:
            logger.info("Shutdown requested before training started.")
            return {
                'training_stats': {},
                'eval_stats': {},
                'training_time_seconds': 0,
                'total_timesteps': 0,
                'best_model_path': None,
                'win_rate': 0,
                'target_achieved': False,
                'shutdown': True
            }
        
        logger.info(f"Starting training for {total_timesteps} timesteps...")
        logger.info(f"Press Ctrl+C to stop training cleanly")
        
        # Train with callbacks (using PPO's learn method which accepts callbacks)
        training_stats = trainer.model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=reset_num_timesteps,
            callback=callbacks,
            log_interval=100
        )
        training_time = time.time() - start_time
        
        # Check if training was stopped by shutdown
        if shutdown_callback.training_stopped:
            logger.info("Training stopped by user request")
            training_stats = {
                'train_timesteps': trainer.model.num_timesteps,
                'training_time': training_time,
                'shutdown': True
            }
            # ALWAYS save checkpoint on shutdown
            _save_exit_checkpoint(trainer, "shutdown")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user (Ctrl+C)")
        training_time = time.time() - start_time
        training_stats = {
            'interrupted': True,
            'train_timesteps': trainer.model.num_timesteps if hasattr(trainer.model, 'num_timesteps') else 0,
            'training_time': training_time,
            'shutdown': True
        }
        # ALWAYS save checkpoint on interrupt
        _save_exit_checkpoint(trainer, "interrupt")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        training_time = time.time() - start_time
        training_stats = {
            'error': str(e),
            'train_timesteps': trainer.model.num_timesteps if hasattr(trainer.model, 'num_timesteps') else 0,
            'training_time': training_time
        }
        # ALWAYS save checkpoint on error
        _save_exit_checkpoint(trainer, "error")
        raise
    
    # Evaluate model
    logger.info("Evaluating trained model...")
    eval_stats = trainer.evaluate(
        n_eval_episodes=config['evaluation']['n_eval_episodes'],
        deterministic=config['evaluation']['deterministic']
    )
    
    # Save best model
    best_model_path = os.path.join(
        config['outputs']['checkpoint_dir'],
        f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    trainer.save_model(best_model_path, include_metadata=True)
    
    # Clean up
    trainer.close()
    
    # Compile results
    results = {
        'training_stats': training_stats,
        'eval_stats': eval_stats,
        'training_time_seconds': training_time,
        'total_timesteps': total_timesteps,
        'best_model_path': best_model_path,
        'win_rate': eval_stats['mean_reward'],  # Using mean reward as proxy for win rate
        'target_achieved': eval_stats['mean_reward'] >= config['experiment']['target_win_rate']
    }
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Final win rate: {results['win_rate']:.3f}")
    logger.info(f"Target achieved: {results['target_achieved']}")
    
    return results


def generate_learning_curves(config: Dict[str, Any], results: Dict[str, Any]) -> None:
    """Generates learning curves and visualizations.

    Args:
        config: The configuration dictionary.
        results: The training results.
    """
    try:
        import matplotlib.pyplot as plt
        
        output_dir = config['outputs']['output_dir']
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create a simple summary plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Win rate vs target
        win_rate = results['win_rate']
        target_rate = config['experiment']['target_win_rate']
        
        ax1.bar(['Achieved', 'Target'], [win_rate, target_rate], 
               color=['green' if win_rate >= target_rate else 'red', 'blue'])
        ax1.set_ylabel('Win Rate')
        ax1.set_title('Final Performance')
        ax1.set_ylim(0, 1)
        
        # Training time
        training_time = results['training_time_seconds'] / 60  # Convert to minutes
        ax2.bar(['Training Time'], [training_time], color='orange')
        ax2.set_ylabel('Time (minutes)')
        ax2.set_title('Training Duration')
        
        # Save plot
        plot_path = os.path.join(output_dir, 'training_summary.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Learning curves saved to {plot_path}")
        
    except ImportError:
        logger.warning("Matplotlib not available, skipping learning curve generation")
    except Exception as e:
        logger.error(f"Failed to generate learning curves: {e}")


def main():
    """The main training function."""
    parser = argparse.ArgumentParser(description='Phase 0 Bootstrap Training')
    parser.add_argument('--config', type=str,
                       default='research/configs/phase0_bootstrap.yaml',
                       help='Path to configuration file (default: research/configs/phase0_bootstrap.yaml)')
    parser.add_argument('--continue-training', action='store_true',
                       help='Continue training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to continue from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Auto-detect latest checkpoint if not specified
    if not args.continue_training and not args.checkpoint:
        latest_checkpoint = find_latest_checkpoint(config['training']['checkpoint_dir'])
        if latest_checkpoint:
            logger.info(f"Auto-detected latest checkpoint: {latest_checkpoint}")
            response = input("Continue from latest checkpoint? (y/n): ").strip().lower()
            if response in ['y', 'yes', '']:
                config['training']['continue_training'] = True
                config['training']['checkpoint_path'] = latest_checkpoint
                logger.info("Will continue from latest checkpoint")
            else:
                logger.info("Starting fresh training")
    
    # Override with command line arguments
    if args.continue_training:
        config['training']['continue_training'] = True
        if args.checkpoint:
            config['training']['checkpoint_path'] = args.checkpoint
    
    # Create directories
    create_directories(config)
    
    # Set up logging with file handler
    setup_logging(config['training']['tensorboard_log_dir'])
    
    # Get git SHA and config hash
    git_sha = get_git_sha()
    config_hash = calculate_config_hash(config)
    
    logger.info("=" * 60)
    logger.info("Phase 0 Bootstrap Training")
    logger.info("=" * 60)
    logger.info(f"Git SHA: {git_sha}")
    logger.info(f"Config hash: {config_hash}")
    logger.info(f"Target win rate: {config['experiment']['target_win_rate']}")
    logger.info(f"Total timesteps: {config['training']['total_timesteps']}")
    logger.info(f"Continue training: {config['training']['continue_training']}")
    if config['training']['checkpoint_path']:
        logger.info(f"Checkpoint path: {config['training']['checkpoint_path']}")
    
    try:
        # Train model
        results = train_model(
            config,
            continue_training=config['training']['continue_training'],
            checkpoint_path=config['training']['checkpoint_path']
        )
        
        # Generate learning curves
        if config['outputs']['save_learning_curves']:
            generate_learning_curves(config, results)
        
        # Log experiment
        log_experiment(config, git_sha, config_hash, results)
        
        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info(f"Final win rate: {results['win_rate']:.3f}")
        logger.info(f"Target achieved: {results['target_achieved']}")
        
        # Check if we need to continue training
        if not results['target_achieved'] and config['training']['total_timesteps'] < 500000:
            logger.info("Target not achieved. Consider extending training to 500K steps.")
            logger.info("To continue training, run:")
            logger.info(f"python -m src.bootstrap.train_bootstrap.py --config {args.config} --continue-training --checkpoint {results['best_model_path']}")
        
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    return results


if __name__ == "__main__":
    main()