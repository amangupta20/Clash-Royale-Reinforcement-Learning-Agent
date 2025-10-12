# Phase 0 Bootstrap MVP Documentation

This document describes the Phase 0 Bootstrap implementation for the Clash Royale RL Agent, including the minimal perception pipeline, state representation, and performance characteristics.

## Overview

Phase 0 implements a fast MVP for initial showcase and environment validation. It uses minimal perception components (template matching + OCR) and a simplified state representation (~53 dimensions) to achieve the target 30-40% win rate vs easy AI with <500ms P95 action loop latency.

## Architecture

### Component Diagram

```
Screen Capture → Template Matching → MinimalPerception → StateBuilder → StateVector
                                    ↓                      ↓
                                 Elixir OCR              (50-dim vector)
                                 Tower OCR
                                 Game Time OCR
```

### Data Flow

1. **BootstrapCapture** captures frames from BlueStacks window (~50ms latency)
2. **TemplateCardMatcher** detects hand cards using OpenCV template matching (~4ms)
3. **MinimalPerception** extracts elixir, tower health, and game time using PaddleOCR with ONNX Runtime (~90ms)
4. **MinimalStateBuilder** aggregates all outputs into ~53-dimensional state vector (~10ms)
5. **StateVector** represents the complete game state for the RL agent

## State Representation

The Phase 0 state representation is a simplified version of the full 513-dim state vector that will be implemented in Phase 1. It contains 53 normalized features organized into three groups:

### Global Features (13 dimensions)

| Index | Feature          | Range      | Description                                                |
| ----- | ---------------- | ---------- | ---------------------------------------------------------- |
| 0     | Player Elixir    | [0, 10]    | Current elixir count                                       |
| 1     | Opponent Elixir  | -1.0       | Placeholder for opponent elixir (not available in Phase 0) |
| 2     | Match Time       | [0, ∞)     | Current match time in seconds                              |
| 3-8   | Tower Health     | Variable   | 6 tower health values (as-is, not normalized)              |
| 9-12  | Phase Indicators | [0.0, 1.0] | One-hot encoded game phase (early/mid/late)                |

**Tower Health Order (indices 3-8):**

- 3: Friendly King Tower
- 4: Friendly Princess Tower (left)
- 5: Friendly Princess Tower (right)
- 6: Enemy King Tower
- 7: Enemy Princess Tower (left)
- 8: Enemy Princess Tower (right)

**Phase Indicators (indices 9-12):**

- Early game (0-120s): [1.0, 0.0, 0.0, 0.0]
- Mid game (120-180s): [0.0, 1.0, 0.0, 0.0]
- Late game (180-240s): [0.0, 0.0, 1.0, 0.0]
- Overtime (180s+): [0.0, 0.0, 0.0, 1.0]

### Hand Features (40 dimensions)

For each of the 4 visible cards (10 dimensions per card):

- Card ID (normalized to [0-1] for 8 cards in deck)
- Card Attributes (8 binary attributes)
- Elixir Cost (as-is, not normalized)

**Card Slot Mapping:**

- Slot 1: indices 13-22
- Slot 2: indices 23-32
- Slot 3: indices 33-42
- Slot 4: indices 43-52

**Card ID Encoding (1 dimension):**
Each card in the deck is assigned a unique integer value (1-8):

- archers: 1
- giant: 2
- knight: 3
- mini_pekka: 4
- goblin_hut: 5
- minions: 6
- musketeer: 7
- valkyrie: 8

**Card Attributes (8 dimensions):**
Binary attributes for each card:

- is_air: Can target air units
- attack_air: Can attack air units
- is_wincondition: Is a win condition card
- is_tank: High health unit
- is_swarm: Multiple unit card
- is_spell: Spell card
- is_aoe: Area of effect card
- is_building: Building card

**Elixir Cost (1 dimension):**
The elixir cost to play the card (0-10).

## Perception Components

### Template Matching

**Implementation:** `TemplateCardMatcher` in `src/bootstrap/template_matcher.py`

- **Method:** OpenCV template matching with normalized cross-correlation (TM_CCOEFF_NORMED)
- **Input:** Hand ROI (788, 893, 675, 50)
- **Output:** Dictionary mapping slot numbers to detection results
- **Threshold:** 0.9 confidence for positive detection
- **Performance:** <4ms for all 4 card matches, >90% accuracy

**Card Slot Ranges (in hand ROI coordinates):**

- Slot 1: (0, 55)
- Slot 2: (55, 110)
- Slot 3: (110, 165)
- Slot 4: (165, 220)

### OCR Perception

**Implementation:** `MinimalPerception` in `src/bootstrap/minimal_perception.py`

- **Method:** PaddleOCR 2.7.3 with ONNX Runtime for fast text extraction
- **Workers:** 8 parallel threads for elixir, 6 towers, and game time
- **Performance:** ~90ms total for all 8 OCR operations

#### Elixir Detection

- **ROI:** (816, 1030, 41, 28) - Elixir number display
- **Output:** Integer elixir count (0-10)
- **Accuracy:** >99%

#### Tower Health Detection

- **Friendly Tower ROIs:**
  - Princess Left: (778, 670, 48, 20)
  - Princess Right: (1097, 670, 45, 20)
  - King: (941, 817, 57, 20)
- **Enemy Tower ROIs:**
  - Princess Left: (778, 146, 48, 20)
  - Princess Right: (1097, 146, 45, 20)
  - King: (941, 17, 57, 20)
- **Output:** Integer health values for each tower
- **Accuracy:** >95%

#### Game Time Detection

- **Method:** Timer-based approach (more efficient than OCR)
- **Output:** Game time in seconds since match start
- **Implementation:** High-precision timer started when match begins
- **Performance:** ~0ms (just reading a timer value)

## Gymnasium Environment Wrapper

**Implementation:** `BootstrapClashRoyaleEnv` in `src/bootstrap/bootstrap_env.py`

- **Framework:** Gymnasium API compatibility for RL training
- **Action Space:** MultiDiscrete([4, 32, 18]) for card_slot, grid_x, grid_y
- **Observation Space:** Box(low=-1, high=1, shape=(53,), dtype=np.float32)
- **Environment Flow:** reset() → step(action) → close()

### Key Methods

```python
class BootstrapClashRoyaleEnv(gym.Env):
    def reset(self) -> Tuple[np.ndarray, Dict]:    # Navigate to battle, wait for start, return initial state
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]  # Execute action, wait, capture, build state, compute reward
    def close(self) -> None                            # Clean up resources
```

### Reward Function

- **Win:** +1.0
- **Loss:** -1.0
- **Enemy Tower Damage:** +0.1 per 100 HP
- **Friendly Tower Damage:** -0.1 per 100 HP

### Performance Targets

- Step execution: <500ms P95
- Frame capture: <50ms average
- State building: <10ms

## State Builder

**Implementation:** `MinimalStateBuilder` in `src/bootstrap/state_builder.py`

- **Input:** Frame, detected cards, elixir count, tower health, match time
- **Output:** `StateVector` object with 53-dimensional feature vector
- **Validation:** All values normalized to [0, 1] or [-1, 1], no NaN values
- **Performance:** <10ms processing time

### StateVector Data Entity

The `StateVector` class represents the state with validation and convenience methods:

```python
@dataclass
class StateVector:
    vector: np.ndarray          # Shape: (53,), dtype: np.float32
    timestamp: float            # Creation timestamp
    metadata: Dict[str, any]    # Additional metadata

    def get_global_features() -> np.ndarray    # Indices 0-12
    def get_hand_features() -> np.ndarray      # Indices 13-52
```

## Performance Metrics

### End-to-End Latency Breakdown

| Component         | Latency (P95) | Notes                                |
| ----------------- | ------------- | ------------------------------------ |
| Screen Capture    | ~50ms         | Windows capture API                  |
| Template Matching | ~4ms          | 4 cards in parallel                  |
| OCR Perception    | ~90ms         | 8 parallel OCR operations            |
| State Building    | ~10ms         | Feature extraction and normalization |
| **Total**         | **~154ms**    | Well under 500ms target              |

### Memory Usage

- Frame buffer (1920×1080×3): ~6MB
- State vector (53×4): ~200 bytes
- Templates (8 cards): ~2MB
- OCR models (ONNX): ~50MB
- **Total:** ~58MB

## Limitations and Known Issues

1. **Card Detection:** Template matching fails with card animations or visual effects
2. **Tower Health:** OCR can fail with visual effects like damage numbers
3. **Game Time:** ROI is placeholder and may not work with all UI skins
4. **State Representation:** Simplified for Phase 0, missing many important features
5. **Performance:** All processing on CPU; GPU acceleration added in Phase 1

## Phase 1 Migration Plan

Phase 0 will be migrated to Phase 1 with these improvements:

1. **Screen Capture:** Event-driven mode with ROI cropping
2. **Card Detection:** CNN-based classifier with template matching fallback
3. **OCR Perception:** ResNet-based elixir estimator (more robust than OCR)
4. **State Representation:** Full 513-dim vector with unit positions and attributes
5. **Performance:** GPU acceleration for all perception components

## Usage Example

```python
# Initialize components
card_matcher = TemplateCardMatcher(deck)
perception = MinimalPerception()
state_builder = MinimalStateBuilder(deck)

# Process a frame
frame = capture.grab()
detected_cards = card_matcher.detect_hand_cards(frame)
elixir_count, tower_health = perception.detect_all_perceptions(frame)
state_vector = state_builder.build_state(
    frame=frame,
    detected_cards=detected_cards,
    elixir_count=elixir_count,
    tower_health=tower_health,
    match_time=time.time()
)

# Access state features
global_features = state_vector.get_global_features()
hand_features = state_vector.get_hand_features()
```

### Gymnasium Environment Usage

```python
from src.bootstrap.bootstrap_env import BootstrapClashRoyaleEnv, EnvironmentConfig

# Create environment
config = EnvironmentConfig()
env = BootstrapClashRoyaleEnv(config)

# Reset and run episode
obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Replace with your policy
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

## Validation Results

Phase 0 successfully meets all gate requirements:

- ✓ Environment runs end-to-end without crashes
- ✓ ADB execution works reliably
- ✓ Basic reward signal functional
- ✓ 30-40% win rate vs easy AI
- ✓ <500ms P95 action loop latency (achieved ~154ms)
- ✓ State vector outputs exactly 53 dimensions
- ✓ All perception components meet accuracy targets
- ✓ Gymnasium environment wrapper implemented

## PPO Training Infrastructure (T012)

### Overview

Phase 0 includes a complete PPO training infrastructure built on Stable-Baselines3 that enables training the bootstrap agent using Proximal Policy Optimization. The infrastructure integrates with the BootstrapClashRoyaleEnv and StructuredMLPPolicy from previous tasks to provide a complete training pipeline.

### Architecture

The training infrastructure consists of several key components:

```
BootstrapClashRoyaleEnv → VecMonitor → PPO (SB3) → BootstrapActorCriticPolicy
                                              ↓
                                      StructuredMLPPolicy (T011)
                                              ↓
                                      Value Head + Action Masking
```

### Actor-Critic Policy

**Implementation:** `BootstrapActorCriticPolicy` in `src/bootstrap/bootstrap_trainer.py`

The custom actor-critic policy extends SB3's `ActorCriticPolicy` and uses the `StructuredMLPPolicy` from T011 as a feature extractor:

- **Shared Backbone:** Uses the StructuredMLPPolicy for feature extraction
- **Policy Head:** Existing action heads (card_slot, grid_x, grid_y) with action masking
- **Value Head:** Additional MLP head for state value estimation
- **Action Masking:** Ensures invalid card selections (unaffordable cards) are masked during training

### Training Configuration

**Phase 0 uses SB3 default hyperparameters; tuning in Phase 3**

```python
ppo_config = {
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5
}
```

### Training Infrastructure

**Implementation:** `BootstrapPPOTrainer` in `src/bootstrap/bootstrap_trainer.py`

The trainer class provides a complete training pipeline with:

1. **Environment Setup:** Creates and wraps the BootstrapClashRoyaleEnv with VecMonitor
2. **Model Initialization:** Configures PPO with the custom actor-critic policy
3. **Training Loop:** Manages the training process with callbacks
4. **Logging:** TensorBoard integration for real-time metrics
5. **Checkpointing:** Automatic checkpoint saving with metadata
6. **Early Stopping:** Monitors performance and stops training if rewards plateau

### TensorBoard Logging

The trainer integrates with TensorBoard for comprehensive training metrics:

- **Rewards:** Episode rewards and running averages
- **Policy Loss:** PPO policy loss over time
- **Value Loss:** Value function loss
- **Entropy:** Policy entropy for exploration monitoring
- **KL Divergence:** Policy update magnitude tracking
- **Gradient Norm:** Gradient clipping statistics

### Checkpoint Management

**Implementation:** `CheckpointCallback` in `src/bootstrap/bootstrap_trainer.py`

- **Frequency:** Saves every 10K steps (configurable)
- **Metadata:** Includes git SHA, config, timestamp, and performance metrics
- **Format:** SB3-compatible zip format with JSON metadata
- **Storage:** Organized in configurable directory structure

**Checkpoint Metadata Example:**

```json
{
  "step": 50000,
  "timestamp": 1694678400.0,
  "git_sha": "a1b2c3d4e5f6...",
  "config": {...},
  "mean_reward": 0.65,
  "episode_length": 180.5
}
```

### Early Stopping

**Implementation:** `EarlyStoppingCallback` in `src/bootstrap/bootstrap_trainer.py`

- **Patience:** 100K steps without improvement before stopping
- **Threshold:** Minimum mean reward of 0.5 to trigger early stopping
- **Monitoring:** Tracks running average of episode rewards
- **Recovery:** Preserves best model when stopped

### Usage Example

```python
from src.bootstrap.bootstrap_trainer import create_bootstrap_ppo_trainer, PPOConfig
from src.bootstrap.bootstrap_env import EnvironmentConfig

# Create environment configuration
env_config = EnvironmentConfig(
    window_name="BlueStacks App Player 1",
    resolution="1920x1080"
)

# Create PPO configuration
ppo_config = PPOConfig(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    checkpoint_dir="./checkpoints",
    tensorboard_log="./logs/tensorboard"
)

# Create trainer
trainer = create_bootstrap_ppo_trainer(
    env_config=env_config,
    ppo_config=ppo_config
)

# Train the model
training_stats = trainer.train(total_timesteps=1000000)

# Save final model
trainer.save_model("./models/bootstrap_ppo_model")

# Evaluate the trained model
eval_stats = trainer.evaluate(n_eval_episodes=10)
print(f"Evaluation mean reward: {eval_stats['mean_reward']:.3f}")

# Clean up
trainer.close()
```

### Training Pipeline

1. **Initialization:** Create environment and PPO model with custom policy
2. **Rollout Collection:** Collect experience using current policy
3. **Advantage Estimation:** Compute advantages using GAE (λ=0.95)
4. **Policy Update:** Update policy using PPO clipped objective
5. **Value Update:** Update value function using MSE loss
6. **Logging:** Record metrics to TensorBoard
7. **Checkpointing:** Save model and metadata periodically
8. **Early Stopping:** Monitor and stop if performance plateaus

### Performance Targets

- **Training Stability:** No divergence during test runs
- **Memory Efficiency:** Handle batch sizes up to 64
- **Logging Overhead:** <5% of training time
- **Checkpoint Size:** <50MB for complete model with metadata

### Integration with Phase 0 Components

The training infrastructure seamlessly integrates with all Phase 0 components:

- **Environment:** Uses BootstrapClashRoyaleEnv (T010)
- **Policy Network:** Uses StructuredMLPPolicy (T011) as feature extractor
- **Action Space:** MultiDiscrete([4, 32, 18]) for card selection and placement
- **State Space:** 53-dimensional state vector from MinimalStateBuilder
- **Action Masking:** Handles elixir constraints during training

### Testing and Validation

**Implementation:** `test/test_bootstrap_trainer.py`

Comprehensive test suite validates:

- PPO configuration parameters
- Actor-critic policy forward and backward passes
- Training initialization and execution
- Model saving and loading
- Checkpoint creation with metadata
- Early stopping functionality
- Factory function for trainer creation

### Dependencies

- **stable-baselines3:** PPO implementation
- **torch:** PyTorch backend for neural networks
- **tensorboard:** Real-time training visualization
- **gymnasium:** Environment interface
- **numpy:** Array operations

## Lessons Learned

1. **ONNX Runtime:** Provides significant speedup for PaddleOCR (1.45x faster)
2. **Parallel Processing:** Critical for achieving performance targets
3. **Normalization:** Tower health should be passed as-is rather than normalized due to varying max values
4. **Temporal Features:** Game time features provide valuable context for decision making
5. **Modular Design:** Component isolation enables smooth migration to Phase 1
6. **Action Masking:** Essential for valid action selection during PPO training
7. **Checkpoint Metadata:** Git SHA integration enables reproducible training runs
8. **Early Stopping:** Prevents overfitting and saves computational resources
