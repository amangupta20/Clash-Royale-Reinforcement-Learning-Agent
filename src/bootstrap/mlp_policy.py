"""
Structured MLP Policy with Shared Card Encoder for Clash Royale RL Agent (T011)

This module implements the StructuredMLPPolicy class that uses a shared card encoder
to efficiently process the 53-dimensional state vector from the BootstrapClashRoyaleEnv.

Architecture:
- Global Processor: Processes 13-dimensional global state (elixir, time, tower health, phase)
- Shared Card Encoder: Processes 10-dimensional card features for each of 4 cards
- Fusion Layer: Combines global and card representations
- Final Decision Layer: Produces action logits for MultiDiscrete([4, 32, 18]) action space

Structured MLP with shared card encoder for efficient learning
"""

import time
import pickle
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union

# Import policy interfaces
from policy.interfaces import Policy, PolicyConfig

# Configure logging
logger = logging.getLogger(__name__)


class CardEncoder(nn.Module):
    """A shared encoder for processing 10-dimensional card features.

    This module takes card features, including the card ID, 8 attributes, and
    elixir cost, and produces a 16-dimensional embedding for each card.

    Attributes:
        layers: A sequential container of layers for the encoder.
    """

    def __init__(self, input_dim: int = 10, embedding_dim: int = 16):
        """Initializes the card encoder.

        Args:
            input_dim: The dimension of the input card features (default: 10).
            embedding_dim: The dimension of the output embedding (default: 16).
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )
    
    def forward(self, cards: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the card encoder.

        Args:
            cards: A tensor of card features with shape (batch_size, 4, 10).

        Returns:
            A tensor of card embeddings with shape (batch_size, 4, 16).
        """
        return self.layers(cards)


class GlobalProcessor(nn.Module):
    """A processor for the 13-dimensional global state features.

    This module processes global state features, including elixir, time, tower
    health, and phase indicators.

    Attributes:
        layers: A sequential container of layers for the processor.
    """

    def __init__(self, input_dim: int = 13, output_dim: int = 64):
        """Initializes the global processor.

        Args:
            input_dim: The dimension of the input global state (default: 13).
            output_dim: The dimension of the output representation (default: 64).
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the global processor.

        Args:
            global_state: A tensor of the global state with shape
                (batch_size, 13).

        Returns:
            A tensor of the global representation with shape (batch_size, 64).
        """
        return self.layers(global_state)


class StructuredMLPPolicy(nn.Module, Policy):
    """A structured MLP policy with a shared card encoder for Clash Royale.

    This policy processes the 53-dimensional state vector by separating the
    global and hand state processing and using a shared card encoder for
    efficient learning. The architecture is as follows:
    1.  The 53-dimensional state is split into global (13) and hand (40)
        components.
    2.  The global state is processed with a GlobalProcessor, resulting in a
        64-dimensional representation.
    3.  The four cards in the hand are processed with a shared CardEncoder,
        resulting in a 4x16=64-dimensional representation.
    4.  The representations are fused and processed through final layers.
    5.  Action logits are produced for a MultiDiscrete action space with
        dimensions for card slot (0-3), grid x-coordinate (0-31), and grid
        y-coordinate (0-17).

    Attributes:
        global_processor: The processor for the global state.
        card_encoder: The shared encoder for the card features.
        fusion_layer: The layer that fuses the global and card representations.
        card_head: The output layer for the card selection logits.
        x_head: The output layer for the x-coordinate logits.
        y_head: The output layer for the y-coordinate logits.
    """

    def __init__(self, config: Optional[PolicyConfig] = None):
        """Initializes the structured MLP policy.

        Args:
            config: The policy configuration.
        """
        # Initialize Policy interface
        if config is None:
            config = PolicyConfig()
        
        # Force CPU device for stability
        config.device = "cpu"
        
        Policy.__init__(self, config)
        nn.Module.__init__(self)
        
        # Architecture dimensions
        self.global_input_dim = 13
        self.card_input_dim = 10
        self.num_cards = 4
        self.card_embedding_dim = 16
        self.global_output_dim = 64
        self.fusion_input_dim = self.global_output_dim + (self.num_cards * self.card_embedding_dim)
        self.fusion_hidden_dim = 256
        
        # Action space dimensions
        self.card_slots = config.action_space[0]
        self.grid_width = config.action_space[1]
        self.grid_height = config.action_space[2]
        
        # Initialize components
        self.global_processor = GlobalProcessor(
            input_dim=self.global_input_dim,
            output_dim=self.global_output_dim
        )
        
        self.card_encoder = CardEncoder(
            input_dim=self.card_input_dim,
            embedding_dim=self.card_embedding_dim
        )
        
        # Fusion layer combines global and card representations
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.fusion_input_dim, self.fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.fusion_hidden_dim, self.fusion_hidden_dim),
            nn.ReLU()
        )
        
        # Action heads for MultiDiscrete action space
        self.card_head = nn.Linear(self.fusion_hidden_dim, self.card_slots)
        self.x_head = nn.Linear(self.fusion_hidden_dim, self.grid_width)
        self.y_head = nn.Linear(self.fusion_hidden_dim, self.grid_height)
        
        # Initialize weights
        self._initialize_weights()
        
        # Move to device
        self.to(self.device)
    
    def _initialize_weights(self):
        """Initializes the network weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the policy network.

        Args:
            state: A state tensor with shape (batch_size, 53).

        Returns:
            A tuple of tensors representing the action logits for card, x, and
            y.
        """
        # Split state into global and hand components
        global_state = state[:, :self.global_input_dim]  # (batch_size, 13)
        hand_state = state[:, self.global_input_dim:]    # (batch_size, 40)
        
        # Reshape hand state for card processing
        cards = hand_state.view(-1, self.num_cards, self.card_input_dim)  # (batch_size, 4, 10)
        
        # Process global state
        global_repr = self.global_processor(global_state)  # (batch_size, 64)
        
        # Process cards with shared encoder
        card_embeddings = self.card_encoder(cards)  # (batch_size, 4, 16)
        card_repr = card_embeddings.view(-1, self.num_cards * self.card_embedding_dim)  # (batch_size, 64)
        
        # Fuse representations
        fused_repr = torch.cat([global_repr, card_repr], dim=1)  # (batch_size, 128)
        fused_repr = self.fusion_layer(fused_repr)  # (batch_size, 256)
        
        # Get action logits
        card_logits = self.card_head(fused_repr)  # (batch_size, 4)
        x_logits = self.x_head(fused_repr)        # (batch_size, 32)
        y_logits = self.y_head(fused_repr)        # (batch_size, 18)
        
        # Debug: Check tensor shapes
        logger.debug(f"card_logits shape: {card_logits.shape}")
        logger.debug(f"x_logits shape: {x_logits.shape}")
        logger.debug(f"y_logits shape: {y_logits.shape}")
        
        # Combine logits for MultiDiscrete action space
        # Return as separate tensors for easier handling
        return card_logits, x_logits, y_logits
    
    def get_action_logits(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gets the action logits for each action dimension.

        Args:
            state: A state tensor with shape (batch_size, 53).

        Returns:
            A tuple of tensors representing the action logits for card, x, and
            y.
        """
        return self.forward(state)
    
    def _apply_action_masking(self, card_logits: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Applies action masking to prevent selecting unavailable cards.

        In the current implementation, cards are only visible when they are
        playable (i.e., the player has enough elixir), so no masking is needed.

        Args:
            card_logits: The card selection logits, with shape (batch_size, 4).
            state: The state tensor, with shape (batch_size, 53).

        Returns:
            The unmasked card logits.
        """
        # No masking needed since cards are only visible when playable
        return card_logits
    
    def act(self, state: np.ndarray, deterministic: Optional[bool] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Selects an action given the current state.

        Args:
            state: The current state as a NumPy array with shape (53,).
            deterministic: Whether to use deterministic action selection.

        Returns:
            A tuple containing the action and an info dictionary. The action is a
            NumPy array with the format [card_slot, grid_x, grid_y].
        """
        if deterministic is None:
            deterministic = self.config.deterministic
        
        # Convert state to tensor
        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        else:
            state_tensor = state.float().unsqueeze(0).to(self.device)
        
        # Get action logits
        card_logits, x_logits, y_logits = self.forward(state_tensor)
        
        # Apply action masking to card selection
        masked_card_logits = self._apply_action_masking(card_logits, state_tensor)
        
        # Sample or select actions
        if deterministic:
            card_action = torch.argmax(masked_card_logits, dim=-1)
            x_action = torch.argmax(x_logits, dim=-1)
            y_action = torch.argmax(y_logits, dim=-1)
        else:
            # Enhanced exploration with temperature scaling and entropy
            
            # Apply temperature scaling to increase randomness
            temperature = 1.2  # Higher temperature = more randomness
            card_probs = F.softmax(masked_card_logits / temperature, dim=-1)
            x_probs = F.softmax(x_logits / temperature, dim=-1)
            y_probs = F.softmax(y_logits / temperature, dim=-1)
            
            # Handle case where all card logits are masked
            if torch.all(masked_card_logits == float('-inf')):
                # If all cards are masked, select randomly
                card_action = torch.randint(0, self.card_slots, (1,), device=self.device).squeeze(-1)
                card_probs = torch.ones_like(card_probs) / self.card_slots  # Uniform distribution
            else:
                # Add entropy bonus for more exploration
                card_entropy = -torch.sum(card_probs * torch.log(card_probs + 1e-8))
                entropy_bonus = 0.1 * card_entropy  # Scale the entropy bonus
                card_probs = F.softmax(masked_card_logits / temperature + entropy_bonus, dim=-1)
                card_action = torch.multinomial(card_probs, 1).squeeze(-1)
            
            # For grid positions, add spatial bias to encourage exploration
            # Add slight bias to avoid always selecting the same position
            x_bias = torch.randn_like(x_logits) * 0.1  # Small random bias
            y_bias = torch.randn_like(y_logits) * 0.1  # Small random bias
            
            x_probs = F.softmax((x_logits + x_bias) / temperature, dim=-1)
            y_probs = F.softmax((y_logits + y_bias) / temperature, dim=-1)
            
            # Add entropy bonus for grid positions as well
            x_entropy = -torch.sum(x_probs * torch.log(x_probs + 1e-8))
            y_entropy = -torch.sum(y_probs * torch.log(y_probs + 1e-8))
            
            x_entropy_bonus = 0.05 * x_entropy
            y_entropy_bonus = 0.05 * y_entropy
            
            x_probs = F.softmax((x_logits + x_bias) / temperature + x_entropy_bonus, dim=-1)
            y_probs = F.softmax((y_logits + y_bias) / temperature + y_entropy_bonus, dim=-1)
            
            # Sample actions
            x_action = torch.multinomial(x_probs, 1).squeeze(-1)
            y_action = torch.multinomial(y_probs, 1).squeeze(-1)
        
        # Convert to numpy
        action = np.array([card_action.item(), x_action.item(), y_action.item()])
        
        # Prepare info dictionary
        info = {
            'card_logits': card_logits.detach().cpu().numpy(),
            'x_logits': x_logits.detach().cpu().numpy(),
            'y_logits': y_logits.detach().cpu().numpy(),
            'masked_card_logits': masked_card_logits.detach().cpu().numpy(),
            'action_probabilities': {
                'card': F.softmax(masked_card_logits, dim=-1).detach().cpu().numpy(),
                'x': F.softmax(x_logits, dim=-1).detach().cpu().numpy(),
                'y': F.softmax(y_logits, dim=-1).detach().cpu().numpy()
            },
            'exploration_temperature': temperature
        }
        
        return action, info
    
    def save(self, path: str):
        """Saves the policy to disk.

        Args:
            path: The path where to save the policy.
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'architecture': {
                'global_input_dim': self.global_input_dim,
                'card_input_dim': self.card_input_dim,
                'num_cards': self.num_cards,
                'card_embedding_dim': self.card_embedding_dim,
                'global_output_dim': self.global_output_dim,
                'fusion_input_dim': self.fusion_input_dim,
                'fusion_hidden_dim': self.fusion_hidden_dim
            }
        }
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """Loads the policy from disk.

        Args:
            path: The path from where to load the policy.
        """
        try:
            # Try with weights_only=True first (PyTorch 2.6+ default)
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except (pickle.UnpicklingError, TypeError):
            # Fall back to weights_only=False for compatibility
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Load model state
        self.load_state_dict(checkpoint['model_state_dict'])
        
        # Update config if provided
        if 'config' in checkpoint:
            self.config = checkpoint['config']
            self.device = torch.device(self.config.device)
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Gets information about the policy architecture.

        Returns:
            A dictionary containing the architecture details.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture_type': 'Structured MLP with Shared Card Encoder',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'components': {
                'global_processor': {
                    'input_dim': self.global_input_dim,
                    'output_dim': self.global_output_dim
                },
                'card_encoder': {
                    'input_dim': self.card_input_dim,
                    'output_dim': self.card_embedding_dim,
                    'num_cards': self.num_cards
                },
                'fusion_layer': {
                    'input_dim': self.fusion_input_dim,
                    'hidden_dim': self.fusion_hidden_dim
                },
                'action_heads': {
                    'card_slots': self.card_slots,
                    'grid_width': self.grid_width,
                    'grid_height': self.grid_height
                }
            },
            'device': str(self.device),
            'deterministic': self.config.deterministic
        }


# Factory function for creating policy instances
def create_structured_mlp_policy(config: Optional[PolicyConfig] = None) -> StructuredMLPPolicy:
    """Creates a structured MLP policy instance.

    Args:
        config: The policy configuration.

    Returns:
        A `StructuredMLPPolicy` instance.
    """
    return StructuredMLPPolicy(config)


# Performance test function
def benchmark_policy(policy: StructuredMLPPolicy, num_samples: int = 1000) -> Dict[str, float]:
    """Benchmarks the policy inference performance.

    Args:
        policy: The policy instance to benchmark.
        num_samples: The number of samples to use for benchmarking.

    Returns:
        A dictionary with performance metrics.
    """
    policy.eval()
    
    # Generate random test data
    test_states = torch.randn(num_samples, 53).to(policy.device)
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            policy.forward(test_states[:1])
    
    # Benchmark forward pass
    start_time = time.perf_counter()
    with torch.no_grad():
        for i in range(num_samples):
            policy.forward(test_states[i:i+1])
    end_time = time.perf_counter()
    
    avg_time_ms = (end_time - start_time) * 1000 / num_samples
    
    # Benchmark action selection
    start_time = time.perf_counter()
    with torch.no_grad():
        for i in range(num_samples):
            state_np = test_states[i].cpu().numpy()
            policy.act(state_np, deterministic=True)
    end_time = time.perf_counter()
    
    avg_action_time_ms = (end_time - start_time) * 1000 / num_samples
    
    return {
        'avg_forward_time_ms': avg_time_ms,
        'avg_action_time_ms': avg_action_time_ms,
        'samples_per_second': 1000.0 / avg_action_time_ms
    }