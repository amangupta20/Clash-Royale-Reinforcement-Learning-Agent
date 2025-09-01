# Helper Scripts

This directory contains various helper scripts for different tasks.

## Configuration

Most scripts in this directory are configured using a `.env` file. A `.env.example` file is provided as a template.

1.  **Create a `.env` file:**
    ```bash
    cp .env.example .env
    ```
2.  **Edit the `.env` file** with your desired settings.

### `requirements.txt`

This file lists the Python dependencies required for the scripts. Install them using pip:

```bash
pip install -r requirements.txt
```

## Scripts

### `Card_Template_matching_Example.py`

This script performs template matching on a game state image to find a specific card.

- **Input:** A game state image and a card template image.
- **Output:** Displays the game state image with a rectangle drawn around the matched card.
- **Configuration:** `CARD_IMAGE_PATH` and `GAME_STATE_IMAGE_PATH` in the `.env` file.

### `frame-extract.py`

This script extracts frames from a video file at a specified interval.

- **Input:** A video file.
- **Output:** A folder of JPEG images extracted from the video.
- **Configuration:**
  - `VIDEO_PATH`: Path to the input video.
  - `OUTPUT_DIR`: Directory to save the extracted frames.
  - `CAPTURE_INTERVAL`: Interval in seconds to capture frames.
  - `TARGET_RESOLUTION_WIDTH`: Width of the output images.
  - `TARGET_RESOLUTION_HEIGHT`: Height of the output images.

### `windows-capture-with required resolution.py`

This script captures a specific window, crops it, and resizes it to a target resolution.

- **Input:** A running window.
- **Output:** A single captured, cropped, and resized image named `image.png`.
- **Configuration:**
  - `WINDOW_NAME`: The name of the window to capture.
  - `CROP_LEFT`, `CROP_RIGHT`: Pixels to crop from the sides.
  - `TARGET_WIDTH`, `TARGET_HEIGHT`: The final resolution of the output image.

### `windows-capture-testing.py`

A simple script to test window capturing. It captures a single frame from the specified window and saves it as `image.png`.

- **Input:** A running window with the name "BlueStacks App Player 1".
- **Output:** A single captured frame named `image.png`.
- **Configuration:** The window name is hardcoded in the script.

### `downscale.py`

A command-line utility to downscale an image to a specific resolution.

- **Usage:**
  ```bash
  python downscale.py <input_image_path> [output_image_path]
  ```
- **Configuration:** The target resolution is hardcoded to 480x854.

### `py_adb.py`

This script establishes communication with Android devices or emulators (e.g., BlueStacks) using the adb-shell pure Python library for cross-platform compatibility.

- **Input:** User-typed ADB shell commands (e.g., `ls /sdcard/`, `pm list packages`, `input keyevent 26`).
- **Output:** Command results printed in the terminal.
- **Configuration:**

  - Fully self-contained with no external ADB dependencies; works on any host OS with Python.
  - Set device IP and port via environment variables `ADB_DEVICE_IP` (default: 127.0.0.1) and `ADB_DEVICE_PORT` (default: 5555).
  - Ensure device/emulator has ADB enabled and reachable.

- **Usage:**

```bash
python py_adb.py
```

Once running, type ADB shell commands at the `adb>` prompt. For example:

```
adb> ls /sdcard/
adb> pm list packages
```

Type `exit` or `quit` to leave the interactive session.
