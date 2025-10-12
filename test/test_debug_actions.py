"""
Test script to debug action generation.

This script adds extensive debugging to understand why actions aren't being generated.
"""

import os
import sys
import logging
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import torch and set to CPU
import torch
torch.set_default_dtype(torch.float32)
torch.cuda.is_available = lambda: False

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main test function."""
    logger.info("Testing PPO action generation with extensive debugging...")
    
    try:
        # Import bootstrap components
        from bootstrap.bootstrap_trainer import create_bootstrap_ppo_trainer, PPOConfig
        from bootstrap.bootstrap_env import BootstrapClashRoyaleEnv, EnvironmentConfig
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
        from stable_baselines3.common.callbacks import BaseCallback
        
        # Create environment configuration
        env_config = EnvironmentConfig(
            window_name="BlueStacks App Player 1",
            resolution="1920x1080",
            max_step_time_ms=500.0,
            action_delay_ms=1000.0
        )
        
        # Create environment
        env = BootstrapClashRoyaleEnv(env_config)
        
        # Wrap in VecEnv
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecMonitor(vec_env)
        
        # Create a custom callback to log actions
        class ActionLoggingCallback(BaseCallback):
            def __init__(self, verbose=0):
                super().__init__(verbose)
            
            def _on_step(self) -> bool:
                if 'actions' in self.locals:
                    actions = self.locals['actions']
                    logger.info(f"PPO Step {self.num_timesteps}: Actions generated: {actions}")
                return True
        
        # Create PPO model directly
        logger.info("Creating PPO model...")
        model = PPO(
            "MlpPolicy",  # Use simple MLP policy for testing
            vec_env,
            learning_rate=3e-4,
            n_steps=16,
            batch_size=8,
            n_epochs=2,
            verbose=1,
            device="cpu"
        )
        
        # Test environment reset
        logger.info("Testing environment reset...")
        obs = vec_env.reset()
        logger.info(f"Environment reset successful. Observation shape: {obs.shape}")
        
        # Test manual step
        logger.info("Testing manual step...")
        action = vec_env.action_space.sample()
        logger.info(f"Manual action: {action}")
        
        step_result = vec_env.step(action)
        logger.info(f"Manual step result: {step_result}")
        
        # Test PPO prediction
        logger.info("Testing PPO prediction...")
        action, _ = model.predict(obs, deterministic=True)
        logger.info(f"PPO prediction successful. Action: {action}")
        
        # Test PPO step
        logger.info("Testing PPO step...")
        step_result = vec_env.step(action)
        logger.info(f"PPO step result: {step_result}")
        
        # Test PPO learn with very few steps
        logger.info("Testing PPO learn with very few steps...")
        model.learn(
            total_timesteps=16,
            callback=ActionLoggingCallback(),
            log_interval=1
        )
        
        # Clean up
        vec_env.close()
        env.close()
        
        logger.info("Debug action test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Debug Action Test")
    print("=" * 60)
    print()
    
    success = main()
    
    if success:
        print("Test completed successfully!")
    else:
        print("Test failed. Check logs for details.")