"""
Integration Example: Structured MLP Policy with BootstrapClashRoyaleEnv

This example demonstrates how to use the StructuredMLPPolicy with the BootstrapClashRoyaleEnv
for the Clash Royale RL agent. It shows the complete workflow from environment creation
to policy execution.

Structured MLP with shared card encoder for efficient learning
"""

import sys
import os
import numpy as np
import torch

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from bootstrap.mlp_policy import create_structured_mlp_policy, benchmark_policy
from policy.interfaces import PolicyConfig


def create_mock_state():
    """Creates a mock 53-dimensional state vector.

    This function creates a state vector similar to what the
    `BootstrapClashRoyaleEnv` would produce, demonstrating the structure that
    the policy expects. The vector includes global features like elixir, time,
    and tower health, as well as features for each of the four cards in the
    hand.

    Returns:
        A NumPy array of shape (53,) representing the mock state.
    """
    state = np.zeros(53, dtype=np.float32)
    
    # Global features (indices 0-12)
    state[0] = 7.0  # Player elixir
    state[1] = -1.0  # Opponent elixir placeholder
    state[2] = 95.0  # Match time in seconds
    
    # Tower health values (indices 3-8)
    # Order: friendly princess left, friendly princess right, friendly king,
    #         enemy princess left, enemy princess right, enemy king
    state[3:9] = [1200, 1150, 2500, 800, 750, 2000]
    
    # Phase indicators (indices 9-12)
    # Early game: 0-120s, Mid game: 120-180s, Late game: 180-240s, Overtime: 240s+
    state[9] = 1.0  # early
    state[10] = 0.0  # mid
    state[11] = 0.0  # late
    state[12] = 0.0  # overtime
    
    # Hand features (indices 13-52): 4 cards × 10 features each
    # Card 1: archers (id=1, cost=3, attributes=[1,1,0,0,0,0,0,0])
    state[13] = 1.0  # card_id
    state[14:22] = [1, 1, 0, 0, 0, 0, 0, 0]  # attributes
    state[22] = 3.0  # elixir_cost
    
    # Card 2: giant (id=2, cost=5, attributes=[0,0,0,1,0,0,0,0])
    state[23] = 2.0  # card_id
    state[24:32] = [0, 0, 0, 1, 0, 0, 0, 0]  # attributes
    state[32] = 5.0  # elixir_cost
    
    # Card 3: knight (id=3, cost=3, attributes=[0,0,0,0,0,0,0,0])
    state[33] = 3.0  # card_id
    state[34:42] = [0, 0, 0, 0, 0, 0, 0, 0]  # attributes
    state[42] = 3.0  # elixir_cost
    
    # Card 4: mini_pekka (id=4, cost=4, attributes=[0,0,0,0,0,0,0,0])
    state[43] = 4.0  # card_id
    state[44:52] = [0, 0, 0, 0, 0, 0, 0, 0]  # attributes
    state[52] = 4.0  # elixir_cost
    
    return state


def demonstrate_policy_usage():
    """Demonstrates how to use the structured MLP policy."""
    print("="*60)
    print("Structured MLP Policy Integration Example")
    print("="*60)
    
    # 1. Create policy configuration with auto device detection
    config = PolicyConfig(
        device='auto',  # Auto-detect CUDA if available
        deterministic=False,  # Use stochastic action selection
        action_space=(4, 32, 18),  # MultiDiscrete action space
        state_dim=53
    )
    
    # 2. Create policy instance
    policy = create_structured_mlp_policy(config)
    
    # 3. Display architecture information
    arch_info = policy.get_architecture_info()
    print(f"Policy Architecture: {arch_info['architecture_type']}")
    print(f"Total Parameters: {arch_info['total_parameters']:,}")
    print(f"Device: {arch_info['device']}")
    print()
    
    # 4. Create mock state
    state = create_mock_state()
    print("Mock State Vector (53 dimensions):")
    print(f"  Global features (0-12): {state[:13]}")
    print(f"  Hand features (13-52): {state[13:]}")
    print()
    
    # 5. Get action from policy
    print("Action Selection:")
    action, info = policy.act(state, deterministic=True)
    print(f"  Deterministic action: [card_slot={action[0]}, grid_x={action[1]}, grid_y={action[2]}]")
    
    action, info = policy.act(state, deterministic=False)
    print(f"  Stochastic action: [card_slot={action[0]}, grid_x={action[1]}, grid_y={action[2]}]")
    print()
    
    # 6. Show action probabilities
    print("Action Probabilities:")
    card_probs = info['action_probabilities']['card'][0]
    x_probs = info['action_probabilities']['x'][0]
    y_probs = info['action_probabilities']['y'][0]
    
    print(f"  Card selection probs: {card_probs}")
    print(f"  X position probs (first 5): {x_probs[:5]}...")
    print(f"  Y position probs (first 5): {y_probs[:5]}...")
    print()
    
    # 7. Demonstrate action masking
    print("Action Masking Demonstration:")
    low_elixir_state = state.copy()
    low_elixir_state[0] = 2.0  # Set elixir to 2 (can't afford most cards)
    
    action, info = policy.act(low_elixir_state, deterministic=True)
    masked_logits = info['masked_card_logits'][0]
    
    print(f"  With 2 elixir, masked logits: {masked_logits}")
    print(f"  Selected action: [card_slot={action[0]}, grid_x={action[1]}, grid_y={action[2]}]")
    print()
    
    # 8. Batch processing
    print("Batch Processing:")
    batch_states = np.array([state, low_elixir_state])
    state_tensor = torch.from_numpy(batch_states).float().to(policy.device)
    
    card_logits, x_logits, y_logits = policy.forward(state_tensor)
    print(f"  Batch shape: {state_tensor.shape}")
    print(f"  Card logits shape: {card_logits.shape}")
    print(f"  X logits shape: {x_logits.shape}")
    print(f"  Y logits shape: {y_logits.shape}")
    print()
    
    # 9. Performance benchmark
    print("Performance Benchmark:")
    metrics = benchmark_policy(policy, num_samples=100)
    print(f"  Average forward time: {metrics['avg_forward_time_ms']:.2f}ms")
    print(f"  Average action time: {metrics['avg_action_time_ms']:.2f}ms")
    print(f"  Samples per second: {metrics['samples_per_second']:.1f}")
    print()
    
    # 10. Save and load policy
    print("Save/Load Demonstration:")
    save_path = "example_policy_checkpoint.pth"
    policy.save(save_path)
    print(f"  Policy saved to: {save_path}")
    
    # Create new policy and load (ensure same device)
    new_policy = create_structured_mlp_policy(PolicyConfig(device=policy.config.device))
    new_policy.load(save_path)
    print(f"  Policy loaded successfully on {new_policy.config.device}")
    
    # Verify loaded policy gives same action
    action_loaded, _ = new_policy.act(state, deterministic=True)
    print(f"  Loaded policy action: [card_slot={action_loaded[0]}, grid_x={action_loaded[1]}, grid_y={action_loaded[2]}]")
    
    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"  Cleaned up checkpoint file")
    
    print()
    print("="*60)
    print("Integration example completed successfully!")
    print("="*60)


def demonstrate_environment_integration():
    """Demonstrates how the policy would integrate with `BootstrapClashRoyaleEnv`.

    This is a conceptual example, as it does not run the full environment.
    """
    print("\n" + "="*60)
    print("Environment Integration Concept")
    print("="*60)
    
    print("""
# Example of how to integrate with BootstrapClashRoyaleEnv:

import numpy as np
from bootstrap.bootstrap_env import BootstrapClashRoyaleEnv
from bootstrap.mlp_policy import create_structured_mlp_policy

# Create environment
env = BootstrapClashRoyaleEnv()

# Create policy
policy = create_structured_mlp_policy()

# Reset environment
obs, info = env.reset()

# RL Training/Inference Loop
for step in range(100):
    # Get action from policy
    action, policy_info = policy.act(obs, deterministic=True)
    
    # Execute action in environment
    obs, reward, terminated, truncated, env_info = env.step(action)
    
    # Process transition (for training)
    # store_transition(obs, action, reward, next_obs, terminated)
    
    if terminated or truncated:
        obs, info = env.reset()
        break

# Clean up
env.close()
""")
    
    print("Key Integration Points:")
    print("1. Environment produces 53-dimensional state vectors")
    print("2. Policy expects 53-dimensional state vectors")
    print("3. Policy outputs MultiDiscrete([4, 32, 18]) actions")
    print("4. Environment expects [card_slot, grid_x, grid_y] actions")
    print("5. Policy handles action masking based on elixir cost")
    print()


if __name__ == "__main__":
    try:
        demonstrate_policy_usage()
        demonstrate_environment_integration()
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback
        traceback.print_exc()