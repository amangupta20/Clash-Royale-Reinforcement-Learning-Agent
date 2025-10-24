"""
PPO Training Example for Clash Royale Bootstrap Agent

This example demonstrates how to use the PPO training infrastructure
with the real BootstrapClashRoyaleEnv environment.
"""

import os
import sys
import time
import logging
from pathlib import Path

import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import bootstrap components
from bootstrap.bootstrap_trainer import create_bootstrap_ppo_trainer, PPOConfig
from bootstrap.bootstrap_env import BootstrapClashRoyaleEnv, EnvironmentConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """The main training function."""
    logger.info("Starting PPO training example with real environment...")

    # Create directories for outputs
    checkpoint_dir = Path("./checkpoints/ppo_example")
    log_dir = Path("./logs/ppo_example")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Environment configuration
    env_config = EnvironmentConfig(
        window_name="BlueStacks App Player 1",
        resolution="1920x1080",
        # Performance settings for real environment
        max_step_time_ms=500.0,
        action_delay_ms=1000.0
    )
    
    # PPO configuration for testing
    ppo_config = PPOConfig(
        learning_rate=3e-4,
        n_steps=8,  # Very small for testing
        batch_size=4,
        n_epochs=1,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        # Training settings
        checkpoint_freq=100,  # Save every 100 steps
        checkpoint_dir=str(checkpoint_dir),
        tensorboard_log=str(log_dir),
        # Early stopping
        early_stopping=False,  # Disable early stopping for testing
        early_stopping_patience=2000,  # 2K steps for testing
        early_stopping_min_reward=0.3,
        # Device settings
        device="auto",  # Auto-detect CPU/CUDA
        verbose=1
    )
    
    # Create trainer
    logger.info("Creating PPO trainer...")
    trainer = create_bootstrap_ppo_trainer(
        env_config=env_config,
        ppo_config=ppo_config
    )
    
    try:
        # Test environment reset
        logger.info("Testing environment reset...")
        obs, info = trainer.env.reset()
        logger.info(f"Environment reset successful. Observation shape: {obs.shape}")
        logger.info(f"Initial info: {info}")
        
        # Test single step
        logger.info("Testing single step...")
        action = trainer.env.action_space.sample()
        obs, reward, terminated, truncated, step_info = trainer.env.step(action)
        logger.info(f"Step completed. Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        
        # Short training run for testing
        logger.info("Starting short training run for validation...")
        training_stats = trainer.train(total_timesteps=1200, reset_num_timesteps=True)
        
        logger.info("Training completed successfully!")
        logger.info(f"Training statistics: {training_stats}")
        
        # Save the trained model
        model_path = "./models/ppo_example_model"
        model_dir = Path("./models")
        model_dir.mkdir(exist_ok=True)
        trainer.save_model(model_path, include_metadata=True)
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Model files: {model_path}.zip and {model_path}_metadata.json")
        logger.info(f"Model directory: {model_dir.absolute()}")
        
        # Quick evaluation
        logger.info("Running quick evaluation...")
        eval_stats = trainer.evaluate(n_eval_episodes=2, deterministic=True)
        logger.info(f"Evaluation results: {eval_stats}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.info("This is expected if the game environment is not running.")
        logger.info("Make sure Clash Royale is running in BlueStacks before running this example.")
    
    finally:
        # Clean up
        logger.info("Cleaning up resources...")
        trainer.close()
        logger.info("Example completed.")


def test_actor_critic_policy():
    """Tests the actor-critic policy with the real environment observation space."""
    logger.info("Testing actor-critic policy...")

    from bootstrap.bootstrap_trainer import BootstrapActorCriticPolicy
    import gymnasium as gym
    
    # Create environment to get spaces
    env = BootstrapClashRoyaleEnv()
    
    # Create policy
    def lr_schedule(progress):
        return 3e-4 * (1 - progress)
    
    policy = BootstrapActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lr_schedule
    )
    
    # Test with real observation
    obs, _ = env.reset()
    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
    
    # Forward pass
    actions, values, log_prob = policy.forward(obs_tensor, deterministic=True)
    logger.info(f"Policy forward pass successful. Actions: {actions}, Values: {values}")
    
    # Test evaluate_actions
    eval_values, eval_log_prob, entropy = policy.evaluate_actions(obs_tensor, actions)
    logger.info(f"Evaluate actions successful. Values: {eval_values}, Entropy: {entropy}")
    
    env.close()
    logger.info("Actor-critic policy test completed.")


if __name__ == "__main__":
    print("=" * 60)
    print("PPO Training Example for Clash Royale Bootstrap Agent")
    print("=" * 60)
    print()
    print("This example tests the PPO training infrastructure with the real environment.")
    print("Make sure Clash Royale is running in BlueStacks before proceeding.")
    print()
    
    # Test policy first
    try:
        test_actor_critic_policy()
        print()
    except Exception as e:
        logger.error(f"Policy test failed: {e}")
        logger.warning("Continuing with training despite policy test failure...")
        print()
    
    # Run main training example
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e:
        logger.error(f"Example failed: {e}")
        logger.info("This is expected if the game environment is not properly set up.")
    
    print()
    print("Example completed. Check the logs and checkpoints directories for outputs.")