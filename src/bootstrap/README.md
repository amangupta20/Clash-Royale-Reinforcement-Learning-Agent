# Phase 0 Bootstrap Components

This directory contains the minimal perception and execution components for Phase 0 of the Clash Royale RL Agent project. Phase 0 focuses on fast MVP development using simplified perception techniques before upgrading to full computer vision pipelines in Phase 1.

## Components

### 1. Screen Capture (`capture.py`)

The `BootstrapCapture` class provides efficient screen capture from BlueStacks emulator with ROI support and error handling.

**Performance Target:** <50ms average capture latency

**Key Features:**

- Thread-safe double buffering
- ROI cropping support
- Error handling for window not found
- FPS monitoring

**Usage Example:**

```python
from capture import BootstrapCapture

# Initialize capture
capture = BootstrapCapture(window_name="BlueStacks App Player 1")

# Start capture
capture.start_capture()

# Grab frames
frame = capture.grab()

# Stop capture
capture.stop_capture()
```

### 2. Template Matching (`template_matcher.py`)

The `TemplateCardMatcher` class uses OpenCV template matching to detect 8 deck cards in hand slots with high performance.

**Performance Target:** <4ms for all 4 card matches, >80% accuracy

**Key Features:**

- Parallel template matching using ThreadPoolExecutor
- Normalized cross-correlation (TM_CCOEFF_NORMED)
- Confidence threshold >0.7 for detection
- Detailed timing metrics

**Usage Example:**

```python
from template_matcher import TemplateCardMatcher

# Initialize with deck of 8 cards
deck = ["archers", "giant", "knight", "mini_pekka", "minions", "musketeer", "Valkyrie", "goblin_hut"]
matcher = TemplateCardMatcher(deck)

# Detect cards in frame
results = matcher.detect_hand_cards(frame)
print(results)
# Output: {1: {'card_id': 'archers', 'confidence': 0.85, 'position': (x, y)}, ...}
```

### 3. Minimal Perception (`minimal_perception.py`)

The `MinimalPerception` class provides fast perception using simple techniques:

- Pixel counting for elixir detection
- OCR for tower health extraction

**Performance Targets:**

- 100% elixir accuracy
- > 95% tower health OCR accuracy

**Key Features:**

- HSV color detection for elixir segments
- EasyOCR for tower health values
- Health percentage normalization
- Edge case handling

**Usage Example:**

```python
from minimal_perception import MinimalPerception

# Initialize perception
perception = MinimalPerception()

# Detect elixir count (0-10)
elixir = perception.detect_elixir(frame)

# Detect tower health
health_dict = perception.detect_tower_health(frame)
health_pct = perception.get_tower_health_percentages(health_dict)
```

### 4. Example Usage (`example_usage.py`)

A complete example demonstrating how to use all Phase 0 components together.

**Running the Example:**

```bash
cd src/bootstrap
python example_usage.py
```

## Performance Metrics

All components include detailed timing metrics to monitor performance:

- **Capture:** <50ms average latency
- **Template Matching:** <4ms for all 4 cards
- **Elixir Detection:** <1ms
- **Tower Health OCR:** <20ms

## Dependencies

- OpenCV (`opencv-python`)
- Windows Capture (`windows-capture`)
- EasyOCR (`easyocr`)
- NumPy (`numpy`)
- PyTorch (`torch`)

## Configuration

The components use hardcoded ROI coordinates for 1920x1080 resolution. These may need adjustment for different screen resolutions.

## Troubleshooting

1. **Capture Issues:**

   - Ensure BlueStacks window name matches "BlueStacks App Player 1"
   - Check that BlueStacks is running and visible

2. **Template Matching Issues:**

   - Verify card templates exist in `assets/cards/`
   - Check that deck.json contains exactly 8 valid card names

3. **OCR Issues:**
   - Ensure GPU is available for faster OCR processing
   - Check that tower health bars are visible in the defined ROIs

## Next Steps

After Phase 0 validation, these components will be upgraded in Phase 1:

- Template matching → CNN card classifier
- Pixel counting → ResNet elixir estimator
- OCR → Enhanced detection system
