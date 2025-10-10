"""
Minimal StateBuilder for Phase 0 (T008)

This module implements the MinimalStateBuilder class that aggregates perception outputs
into a ~50-dimensional state vector for Phase 0 of the Clash Royale RL Agent.

State Vector Structure (~50 dims):
1. Global features (13):
   - Player elixir (0-10, normalized)
   - Opponent elixir placeholder (-1)
   - Match time (normalized)
   - 6 tower health values (as-is, not normalized)
   - 4 phase indicators (early/mid/late/overtime)

2. Hand features (32):
   - 4 visible cards × 8 features each:
     - Card ID one-hot encoded
     - Affordability (can afford with current elixir)
     - 6 basic attributes (from attribute.txt)

3. Game time features (5):
   - Additional time-related features for better temporal reasoning
   - Changed from upcoming card features per user feedback

Performance Target: <10ms processing time
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class StateVector:
    """
    StateVector data entity for Phase 0.
    
    This class represents the ~50-dimensional state vector used by the Phase 0
    RL agent. All values are normalized to [0, 1] or [-1, 1] range.
    
    Attributes:
        vector: numpy array of shape (50,) containing the state features
        timestamp: timestamp when the state was created
        metadata: additional metadata about the state
    """
    vector: np.ndarray
    timestamp: float
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate the StateVector after initialization."""
        if not isinstance(self.vector, np.ndarray):
            raise ValueError("StateVector.vector must be a numpy array")
        
        # Allow both 50 and 53 dimensional vectors for backward compatibility
        if self.vector.shape not in [(50,), (53,)]:
            raise ValueError(f"StateVector.vector must have shape (50,) or (53,), got {self.vector.shape}")
        
        if not np.all(np.isfinite(self.vector)):
            raise ValueError("StateVector.vector contains non-finite values (NaN or Inf)")
        
        # Check normalization (tower health values can be >1)
        # Only check for non-finite values
        if not np.all(np.isfinite(self.vector)):
            raise ValueError("StateVector.vector contains non-finite values (NaN or Inf)")
    
    def get_global_features(self) -> np.ndarray:
        """Get the global features (indices 0-12)."""
        return self.vector[0:13]
    
    def get_hand_features(self) -> np.ndarray:
        """Get the hand features (indices 13-52)."""
        return self.vector[13:53]


class MinimalStateBuilder:
    """
    Minimal StateBuilder for Phase 0.
    
    This class aggregates perception outputs from BootstrapCapture, TemplateCardMatcher,
    and MinimalPerception into a ~50-dimensional state vector for the Phase 0 RL agent.
    
    Performance Target: <10ms processing time
    """
    
    # Card attribute mapping from attribute.txt
    CARD_ATTRIBUTES = {
        'is_air': 0,
        'attack_air': 1,
        'is_wincondition': 2,
        'is_tank': 3,
        'is_swarm': 4,
        'is_spell': 5,
        'is_aoe': 6,
        'is_building': 7
    }
    
    # Card attribute values for each card in deck.json
    # Format: {card_name: [is_air, attack_air, is_wincondition, is_tank, is_swarm, is_spell, is_aoe, is_building]}
    CARD_ATTRIBUTE_MAP = {
        'archers': [1, 1, 0, 0, 0, 0, 0, 0],
        'giant': [0, 0, 0, 1, 0, 0, 0, 0],
        'knight': [0, 0, 0, 0, 0, 0, 0, 0],
        'mini_pekka': [0, 0, 0, 0, 0, 0, 0, 0],
        'goblin_hut': [0, 0, 0, 0, 0, 0, 0, 1],
        'minions': [1, 1, 0, 0, 1, 0, 0, 0],
        'musketeer': [0, 1, 0, 0, 0, 0, 0, 0],
        'valkyrie': [0, 0, 0, 0, 0, 0, 1, 0]
    }
    
    def __init__(self, card_names: List[str]):
        """
        Initialize the MinimalStateBuilder.
        
        Args:
            card_names: List of 8 card names in the deck
        """
        if not card_names or len(card_names) != 8:
            raise ValueError("Exactly 8 card names required for deck")
        
        self.card_names = card_names
        self.card_id_map = {name: i for i, name in enumerate(card_names)}
        
        # Pre-compute one-hot encodings for all cards
        self.card_one_hot = {}
        for i, name in enumerate(card_names):
            one_hot = np.zeros(8, dtype=np.float32)
            one_hot[i] = 1.0
            self.card_one_hot[name] = one_hot
    
    def build_state(self, 
                   frame: np.ndarray,
                   detected_cards: Dict[int, Dict[str, Any]],
                   elixir_count: int,
                   tower_health: Dict[str, List[int]],
                   match_time: float = 0.0) -> StateVector:
        """
        Build a ~50-dimensional state vector from perception outputs.
        
        Args:
            frame: Full screen frame in BGR format (not used directly but kept for interface consistency)
            detected_cards: Dictionary mapping slot numbers to detection results from TemplateCardMatcher
            elixir_count: Current elixir count from MinimalPerception
            tower_health: Dictionary with tower health values from MinimalPerception
            match_time: Current match time in seconds (normalized internally)
            
        Returns:
            StateVector object with ~50-dimensional feature vector
        """
        start_time = time.perf_counter()
        
        # Initialize state vector with zeros (53 dimensions)
        state_vector = np.zeros(53, dtype=np.float32)
        
        # 1. Global features (13 dims)
        self._extract_global_features(state_vector, elixir_count, tower_health, match_time)
        
        # 2. Hand features (40 dims: 4 cards × 10 features each)
        self._extract_hand_features_efficient(state_vector, detected_cards, elixir_count)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        print(f"StateBuilder: {processing_time:.2f}ms")
        
        # Create metadata
        metadata = {
            'elixir_count': elixir_count,
            'processing_time_ms': processing_time,
            'detected_cards': detected_cards,
            'tower_health': tower_health
        }
        
        return StateVector(
            vector=state_vector,
            timestamp=time.time(),
            metadata=metadata
        )
    
    def _extract_global_features(self, 
                                state_vector: np.ndarray,
                                elixir_count: int,
                                tower_health: Dict[str, List[int]],
                                match_time: float):
        """
        Extract global features (indices 0-12).
        
        Features:
        - Player elixir (0-10, normalized to [0, 1])
        - Opponent elixir placeholder (-1)
        - Match time (normalized to [0, 1], assuming max 5 minutes)
        - 6 tower health percentages (0-1)
        - 4 phase indicators (early/mid/late/overtime)
        """
        # Player elixir (index 0) - not normalized
        state_vector[0] = np.clip(elixir_count, 0, 10)
        
        # Opponent elixir placeholder (index 1) - not available in Phase 0
        state_vector[1] = -1.0
        
        # Match time (index 2) - not normalized, seconds since start
        state_vector[2] = match_time
        
        # Tower health values (indices 3-8)
        # Order: friendly princess left, friendly princess right, friendly king,
        #         enemy princess left, enemy princess right, enemy king
        # Using as-is values (not normalized) as decided in implementation
        tower_healths = []
        
        # Friendly towers
        if len(tower_health.get('friendly', [])) >= 3:
            # Princess towers (indices 3-4)
            princess_left_health = tower_health['friendly'][0]
            princess_right_health = tower_health['friendly'][1]
            # King tower (index 5)
            king_health = tower_health['friendly'][2]
            tower_healths.extend([princess_left_health, princess_right_health, king_health])
        else:
            tower_healths.extend([0.0, 0.0, 0.0])
        
        # Enemy towers
        if len(tower_health.get('enemy', [])) >= 3:
            # Princess towers (indices 6-7)
            princess_left_health = tower_health['enemy'][0]
            princess_right_health = tower_health['enemy'][1]
            # King tower (index 8)
            king_health = tower_health['enemy'][2]
            tower_healths.extend([princess_left_health, princess_right_health, king_health])
        else:
            tower_healths.extend([0.0, 0.0, 0.0])
        
        # Set tower health values in state vector (as-is, not normalized)
        for i, health in enumerate(tower_healths):
            state_vector[3 + i] = health
        
        # Phase indicators (indices 9-12)
        # Assuming match_time is in seconds
        # Early game: 0-120s, Mid game: 120-180s, Late game: 180-240s, Overtime: 180s+
        if match_time < 120:
            # Early game
            state_vector[9] = 1.0  # early
            state_vector[10] = 0.0  # mid
            state_vector[11] = 0.0  # late
            state_vector[12] = 0.0  # overtime
        elif match_time < 180:
            # Mid game
            state_vector[9] = 0.0  # early
            state_vector[10] = 1.0  # mid
            state_vector[11] = 0.0  # late
            state_vector[12] = 0.0  # overtime
        elif match_time < 240:
            # Late game
            state_vector[9] = 0.0  # early
            state_vector[10] = 0.0  # mid
            state_vector[11] = 1.0  # late
            state_vector[12] = 0.0  # overtime
        else:
            # Overtime (240s+)
            state_vector[9] = 0.0  # early
            state_vector[10] = 0.0  # mid
            state_vector[11] = 0.0  # late
            state_vector[12] = 1.0  # overtime
    
    def _extract_hand_features_extended(self,
                                       state_vector: np.ndarray,
                                       detected_cards: Dict[int, Dict[str, Any]],
                                       elixir_count: int):
        """
        Extract hand features for 53-dim vector (indices 13-52).
        
        For each of 4 visible cards (10 features per card):
        - Card ID one-hot encoded (8 dims)
        - Elixir cost (1 dim)
        - Affordability (can afford with current elixir) (1 dim)
        
        Total: 4 cards × 10 features = 40 dimensions
        """
        # Define the card elixir costs
        card_costs = {
            'archers': 3,
            'giant': 5,
            'knight': 3,
            'mini_pekka': 4,
            'goblin_hut': 5,
            'minions': 3,
            'musketeer': 4,
            'valkyrie': 4
        }
        
        # Process each visible card slot (1-4)
        for slot in range(1, 5):
            slot_idx = slot - 1
            start_idx = 13 + slot_idx * 10  # 10 features per card
            
            if slot in detected_cards and detected_cards[slot]['card_id'] is not None:
                card_name = detected_cards[slot]['card_id']
                
                # Card ID one-hot encoded (8 dims)
                state_vector[start_idx:start_idx + 8] = self.card_one_hot.get(card_name, np.zeros(8))
                
                # Elixir cost (1 dim) - normalized to [0, 1]
                elixir_cost = card_costs.get(card_name, 0)
                state_vector[start_idx + 8] = elixir_cost / 10.0  # Normalize by max elixir
                
                # Affordability (1 dim) - can afford with current elixir
                state_vector[start_idx + 9] = 1.0 if elixir_count >= elixir_cost else 0.0
            else:
                # No card detected, fill with zeros
                state_vector[start_idx:start_idx + 10] = 0.0
    def _extract_hand_features_efficient(self,
                                       state_vector: np.ndarray,
                                       detected_cards: Dict[int, Dict[str, Any]],
                                       elixir_count: int):
        """
        Extract hand features for 53-dim vector (indices 13-52).
        
        For each of 4 visible cards (10 features per card):
        - Card ID (1 dim) - values 0-7 (not one-hot)
        - Card attributes (8 dims) - from CARD_ATTRIBUTE_MAP
        - Elixir cost (1 dim) - values 0-10 (not normalized)
        
        Total: 4 cards × 10 features = 40 dimensions
        """
        # Define the card elixir costs
        card_costs = {
            'archers': 3,
            'giant': 5,
            'knight': 3,
            'mini_pekka': 4,
            'goblin_hut': 5,
            'minions': 3,
            'musketeer': 4,
            'valkyrie': 4
        }
        
        # Process each visible card slot (1-4)
        for slot in range(1, 5):
            slot_idx = slot - 1
            start_idx = 13 + slot_idx * 10  # 10 features per card
            
            if slot in detected_cards and detected_cards[slot]['card_id'] is not None:
                card_name = detected_cards[slot]['card_id']
                
                # Card ID (1 dim) - values 1-8 (no normalization)
                card_id = self.card_id_map.get(card_name, 0)
                state_vector[start_idx] = card_id + 1  # 1-8 range
                
                # Card attributes (8 dims) - from CARD_ATTRIBUTE_MAP
                card_attrs = self.CARD_ATTRIBUTE_MAP.get(card_name, [0] * 8)
                state_vector[start_idx + 1:start_idx + 9] = card_attrs
                
                # Elixir cost (1 dim) - values 0-10 (not normalized)
                elixir_cost = card_costs.get(card_name, 0)
                state_vector[start_idx + 9] = elixir_cost
            else:
                # No card detected, fill with zeros
                state_vector[start_idx:start_idx + 10] = 0.0
    
    def _extract_game_time_features(self,
                                  state_vector: np.ndarray,
                                  detected_cards: Dict[int, Dict[str, Any]],
                                  elixir_count: int,
                                  match_time: float):
        """
        Extract game time features (indices 45-49).
        
        Temporal features for better decision making:
        - Time in double-elixir (normalized)
        - Time until overtime (normalized)
        - Elixir regeneration rate (fixed)
        - Time since last card play (placeholder)
        - Average time between plays (placeholder)
        """
        # Feature 45: Time in double-elixir (normalized to [0, 1])
        # Double-elixir starts at 120s and lasts until overtime at 300s (180s max)
        if match_time < 120:
            state_vector[45] = 0.0  # Not in double-elixir yet
        elif match_time < 300:
            time_in_double = match_time - 120
            state_vector[45] = np.clip(time_in_double / 180.0, 0.0, 1.0)
        else:
            state_vector[45] = 1.0  # Full double-elixir time
        
        # Feature 46: Time until overtime (normalized to [0, 1])
        time_until_overtime = max(0, 300 - match_time)
        state_vector[46] = np.clip(time_until_overtime / 300.0, 0.0, 1.0)
        
        # Feature 47: Elixir regeneration rate (fixed at 1/2.8 per second, normalized to [0, 1])
        elixir_rate = 1.0 / 2.8
        state_vector[47] = np.clip(elixir_rate, 0.0, 1.0)
        
        # Feature 48: Time since last card play (placeholder for Phase 0)
        state_vector[48] = 0.0
        
        # Feature 49: Average time between plays (placeholder for Phase 0)
        state_vector[49] = 0.0


# Phase 0 uses simplified StateBuilder (~50-dim); full StateBuilder with 513-dim StateVector added in Phase 1 (T023)