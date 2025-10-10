# BootstrapActionExecutor - ADB-based Action Executor

## Overview

The `BootstrapActionExecutor` is a Phase 0 implementation of the action execution system for the Clash Royale RL Agent. It provides ADB-based card deployment capabilities with basic humanization features for the BlueStacks emulator.

## Features

- **ADB Connection**: Connects to BlueStacks emulator via adb-shell library
- **Coordinate Mapping**: Maps action space (card_slot, grid_x, grid_y) to screen coordinates
- **Humanization**: Basic jitter and timing variations for anti-detection
- **Error Handling**: Graceful failure handling with logging
- **Multi-Resolution Support**: Supports both 1920×1080 and 1280×720 resolutions

## Action Space

- **card_slot**: Integer (0-3) for card selection from 4 visible cards in hand
- **grid_x**: Integer (0-31) for x position on the 32×18 grid
- **grid_y**: Integer (0-17) for y position on the 32×18 grid

## Installation

Ensure the required dependencies are installed:

```bash
pip install adb-shell python-dotenv
```

## Configuration

Create a `.env` file in the project root with the following settings:

```env
# ADB device configuration
ADB_DEVICE_IP=127.0.0.1
ADB_DEVICE_PORT=5555
```

## Usage

### Basic Usage

```python
from bootstrap.executor import BootstrapActionExecutor

# Create executor
executor = BootstrapActionExecutor(
    resolution="1920x1080",
    jitter_range=(5, 10),
    delay_range=(50, 200)
)

# Connect to BlueStacks
if executor.connect():
    # Execute an action
        action = {
            'card_slot': 0,  # First visible card in hand
            'grid_x': 16,    # Center of arena (x)
            'grid_y': 8,     # Center of arena (y)
        }

    success = executor.execute(action)
    if success:
        print("Action executed successfully!")

    # Disconnect
    executor.disconnect()
```

### Advanced Usage with Result Details

```python
# Execute action and get detailed result
result = executor.execute_with_result(action)

if result.success:
    print(f"Action executed in {result.execution_time_ms:.2f}ms")
    print(f"Metadata: {result.metadata}")
else:
    print(f"Action failed: {result.error_message}")
```

## Coordinate System

### Grid Coordinates

The game arena is mapped to a 32×18 grid:

- Grid (0, 0) = Top-left corner of arena
- Grid (31, 17) = Bottom-right corner of arena
- Grid (16, 8) = Center of arena

### Screen Resolutions

#### 1920×1080 Resolution

- Card slots: Y coordinate at 985px
- Arena bounds: Left=60, Right=1860, Top=360, Bottom=1020

#### 1280×720 Resolution

- Card slots: Y coordinate at 656px
- Arena bounds: Left=40, Right=1240, Top=240, Bottom=680

## Humanization Features

### Coordinate Jitter

- Random jitter of ±5-10 pixels for anti-detection
- Applied to both card selection and deployment coordinates

### Variable Timing

- Random delay of 50-200ms between card selection and deployment
- Simulates human reaction time variations

## Testing

### Dry Run Tests

```bash
cd src/bootstrap
python test_executor.py
```

### Live Tests

```bash
cd src/bootstrap
python live_test_executor.py
```

## Integration with Perception Pipeline

The executor can be integrated with the perception pipeline:

```python
from bootstrap.executor import BootstrapActionExecutor
from bootstrap.capture import BootstrapCapture
from bootstrap.template_matcher import TemplateCardMatcher
from bootstrap.minimal_perception import MinimalPerception

# Create pipeline components
executor = BootstrapActionExecutor()
capture = BootstrapCapture()
card_matcher = TemplateCardMatcher()
perception = MinimalPerception()

# Get current game state
frame = capture.get_frame()
detected_cards = card_matcher.detect_cards(frame)
elixir_count = perception.detect_elixir(frame)

# Execute action based on game state
if detected_cards and elixir_count >= 3:
    action = {
        'card_slot': 0,
        'grid_x': 16,
        'grid_y': 8
    }
    executor.execute(action)
```

## Error Handling

The executor provides comprehensive error handling:

- **Connection Errors**: Handles ADB connection failures gracefully
- **Validation Errors**: Validates action parameters before execution
- **Execution Errors**: Logs and continues on action failures
- **Timeout Handling**: Configurable timeouts for ADB operations

## Performance

- **Target**: <100ms per action execution
- **Connection**: ~9s timeout for ADB connection
- **Humanization**: 50-200ms delays between actions

## Troubleshooting

### Connection Issues

1. Ensure BlueStacks is running
2. Enable ADB in BlueStacks settings
3. Check IP and port configuration
4. Verify firewall settings

### Action Issues

1. Ensure Clash Royale is open and in a battle
2. Check elixir availability
3. Verify card is in hand
4. Validate target coordinates are within arena bounds

## Future Enhancements

- Phase 4: More sophisticated humanization features
- Phase 4: Advanced timing patterns
- Phase 4: Action success verification
- Phase 4: Adaptive jitter based on game state

## Architecture

The executor follows a clean architecture:

- `BaseActionExecutor`: Abstract base class defining interface
- `BootstrapActionExecutor`: Concrete Phase 0 implementation
- `ActionResult`: Data class for execution results

This design allows for easy extension in future phases while maintaining backward compatibility.
