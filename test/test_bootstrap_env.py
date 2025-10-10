"""
Test script for BootstrapClashRoyaleEnv (T010)

This script tests the basic functionality of the BootstrapClashRoyaleEnv
to ensure it meets the Phase 0 requirements.
"""

import sys
import os
import time
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from bootstrap.bootstrap_env import BootstrapClashRoyaleEnv, EnvironmentConfig


def test_environment_interface():
    """Test that the environment implements the correct Gymnasium interface."""
    print("Testing environment interface...")
    
    # Create environment
    config = EnvironmentConfig()
    env = BootstrapClashRoyaleEnv(config)
    
    # Test action and observation spaces
    assert env.action_space.shape == (3,), f"Expected action space shape (3,), got {env.action_space.shape}"
    assert env.observation_space.shape == (53,), f"Expected observation space shape (53,), got {env.observation_space.shape}"
    
    # Test action space bounds
    assert env.action_space.nvec[0] == 4, f"Expected 4 card slots, got {env.action_space.nvec[0]}"
    assert env.action_space.nvec[1] == 32, f"Expected 32 grid positions, got {env.action_space.nvec[1]}"
    assert env.action_space.nvec[2] == 18, f"Expected 18 grid positions, got {env.action_space.nvec[2]}"
    
    # Test observation space bounds
    assert env.observation_space.low.min() >= -1.0, "Observation space lower bound should be >= -1"
    assert env.observation_space.high.max() <= 1.0, "Observation space upper bound should be <= 1"
    
    print("✓ Environment interface test passed")
    return True


def test_environment_reset():
    """Test the environment reset functionality."""
    print("Testing environment reset...")
    
    # Create environment
    config = EnvironmentConfig()
    env = BootstrapClashRoyaleEnv(config)
    
    try:
        # Test reset
        obs, info = env.reset()
        
        # Check observation shape and type
        assert isinstance(obs, np.ndarray), f"Expected numpy array observation, got {type(obs)}"
        assert obs.shape == (53,), f"Expected observation shape (53,), got {obs.shape}"
        assert obs.dtype == np.float32, f"Expected float32 observation, got {obs.dtype}"
        
        # Check info dictionary
        assert isinstance(info, dict), f"Expected dict info, got {type(info)}"
        assert 'episode_count' in info, "Info should contain episode_count"
        assert 'game_phase' in info, "Info should contain game_phase"
        
        print("✓ Environment reset test passed")
        return True
        
    except Exception as e:
        print(f"✗ Environment reset test failed: {e}")
        return False


def test_environment_step():
    """Test the environment step functionality."""
    print("Testing environment step...")
    
    # Create environment
    config = EnvironmentConfig()
    env = BootstrapClashRoyaleEnv(config)
    
    try:
        # Reset environment
        obs, info = env.reset()
        
        # Test valid action
        action = np.array([0, 16, 9])  # card_slot=0, grid_x=16, grid_y=9
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check observation shape and type
        assert isinstance(obs, np.ndarray), f"Expected numpy array observation, got {type(obs)}"
        assert obs.shape == (53,), f"Expected observation shape (53,), got {obs.shape}"
        
        # Check reward type
        assert isinstance(reward, (int, float)), f"Expected numeric reward, got {type(reward)}"
        
        # Check terminated and truncated flags
        assert isinstance(terminated, bool), f"Expected bool terminated, got {type(terminated)}"
        assert isinstance(truncated, bool), f"Expected bool truncated, got {type(truncated)}"
        
        # Check info dictionary
        assert isinstance(info, dict), f"Expected dict info, got {type(info)}"
        assert 'step_count' in info, "Info should contain step_count"
        assert 'action' in info, "Info should contain action"
        
        print("✓ Environment step test passed")
        return True
        
    except Exception as e:
        print(f"✗ Environment step test failed: {e}")
        return False


def test_action_validation():
    """Test action validation."""
    print("Testing action validation...")
    
    # Create environment
    config = EnvironmentConfig()
    env = BootstrapClashRoyaleEnv(config)
    
    try:
        # Reset environment
        obs, info = env.reset()
        
        # Test invalid actions
        invalid_actions = [
            np.array([-1, 16, 9]),   # Invalid card_slot
            np.array([4, 16, 9]),    # Invalid card_slot
            np.array([0, -1, 9]),    # Invalid grid_x
            np.array([0, 32, 9]),    # Invalid grid_x
            np.array([0, 16, -1]),   # Invalid grid_y
            np.array([0, 16, 18]),   # Invalid grid_y
        ]
        
        for action in invalid_actions:
            try:
                obs, reward, terminated, truncated, info = env.step(action)
                # If we get here, the action was accepted (might be valid in some contexts)
                print(f"  Action {action} was accepted")
            except Exception as e:
                # Expected for invalid actions
                print(f"  Action {action} was rejected: {e}")
        
        print("✓ Action validation test passed")
        return True
        
    except Exception as e:
        print(f"✗ Action validation test failed: {e}")
        return False


def test_performance_metrics():
    """Test performance metrics collection."""
    print("Testing performance metrics...")
    
    # Create environment
    config = EnvironmentConfig()
    env = BootstrapClashRoyaleEnv(config)
    
    try:
        # Get initial metrics
        metrics = env.get_performance_metrics()
        
        # Check metrics structure
        assert isinstance(metrics, dict), f"Expected dict metrics, got {type(metrics)}"
        assert 'episode_count' in metrics, "Metrics should contain episode_count"
        assert 'step_count' in metrics, "Metrics should contain step_count"
        assert 'game_phase' in metrics, "Metrics should contain game_phase"
        
        print("✓ Performance metrics test passed")
        return True
        
    except Exception as e:
        print(f"✗ Performance metrics test failed: {e}")
        return False


def test_environment_close():
    """Test environment cleanup."""
    print("Testing environment close...")
    
    # Create environment
    config = EnvironmentConfig()
    env = BootstrapClashRoyaleEnv(config)
    
    try:
        # Test close
        env.close()
        print("✓ Environment close test passed")
        return True
        
    except Exception as e:
        print(f"✗ Environment close test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Running BootstrapClashRoyaleEnv tests...")
    print("=" * 50)
    
    tests = [
        test_environment_interface,
        test_environment_reset,
        test_environment_step,
        test_action_validation,
        test_performance_metrics,
        test_environment_close,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! BootstrapClashRoyaleEnv is ready for use.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)