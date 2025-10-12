"""
Test script to manually test action generation and execution.

This script manually tests the action generation and execution pipeline.
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main test function."""
    logger.info("Testing manual action generation and execution...")
    
    try:
        # Import bootstrap components
        from bootstrap.bootstrap_env import BootstrapClashRoyaleEnv, EnvironmentConfig
        from bootstrap.bootstrap_trainer import BootstrapActorCriticPolicy
        import gymnasium as gym
        
        # Create environment configuration
        env_config = EnvironmentConfig(
            window_name="BlueStacks App Player 1",
            resolution="1920x1080",
            max_step_time_ms=500.0,
            action_delay_ms=1000.0
        )
        
        # Create environment
        env = BootstrapClashRoyaleEnv(env_config)
        
        # Test environment reset
        logger.info("Testing environment reset...")
        obs, info = env.reset()
        logger.info(f"Environment reset successful. Observation shape: {obs.shape}")
        
        # Create policy
        def lr_schedule(progress):
            return 3e-4 * (1 - progress)
        
        policy = BootstrapActorCriticPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lr_schedule,
            device="cpu"
        )
        
        # Test policy forward pass
        logger.info("Testing policy forward pass...")
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        actions, values, log_prob = policy.forward(obs_tensor, deterministic=False)
        logger.info(f"Policy forward pass successful. Actions: {actions}, Values: {values}")
        
        # Test policy prediction
        logger.info("Testing policy prediction...")
        action, _ = policy.predict(obs, deterministic=False)
        logger.info(f"Policy prediction successful. Action: {action}")
        
        # Test environment step with random action
        logger.info("Testing environment step with random action...")
        random_action = env.action_space.sample()
        logger.info(f"Random action: {random_action}")
        
        # Log the action before stepping
        logger.info(f"About to step with action: {random_action}")
        
        step_result = env.step(random_action)
        logger.info(f"Step result: {step_result}")
        
        # Test environment step with policy action
        logger.info("Testing environment step with policy action...")
        policy_action = action
        logger.info(f"Policy action: {policy_action}")
        
        # Log the action before stepping
        logger.info(f"About to step with action: {policy_action}")
        
        step_result = env.step(policy_action)
        logger.info(f"Step result: {step_result}")
        
        # Reset environment if needed
        if step_result[2] or step_result[3]:  # terminated or truncated
            logger.info("Environment terminated after first step, resetting...")
            obs, _ = env.reset()
        
        # Test multiple steps
        logger.info("Testing multiple steps...")
        for i in range(3):
            # Reset environment if needed
            if step_result[2] or step_result[3]:  # terminated or truncated
                logger.info(f"Environment terminated at step {i+1}, resetting...")
                obs, _ = env.reset()
            
            # Get action from policy
            action, _ = policy.predict(obs, deterministic=False)
            logger.info(f"Step {i+1}: Policy action: {action}")
            
            # Step with the action
            step_result = env.step(action)
            logger.info(f"Step {i+1} result: {step_result}")
            
            # Get new observation
            obs = step_result[0]
        
        # Clean up
        env.close()
        
        logger.info("Manual action test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Manual Action Test")
    print("=" * 60)
    print()
    
    success = main()
    
    if success:
        print("Test completed successfully!")
    else:
        print("Test failed. Check logs for details.")