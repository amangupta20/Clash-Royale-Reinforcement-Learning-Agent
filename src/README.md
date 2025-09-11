# Clash Royale Bot - Source Code Documentation

This document provides an overview of the source code for the Clash Royale Bot. The bot is designed to play the game automatically by capturing the screen, detecting cards, and deploying them based on predefined strategies.

## Project Structure

The `src` directory contains the following Python files:

-   `main.py`: The main entry point of the application.
-   `capture.py`: Handles screen capture from the game window.
-   `template_matcher.py`: Detects cards in the player's hand using template matching.
-   `deploy_cards.py`: Provides high-level functions for different card deployment strategies.
-   `random_deployment.py`: Contains the low-level logic for deploying cards, including screen coordinates and ADB commands.

## File Descriptions

### `main.py`

This is the main script that orchestrates the entire bot's functionality. It initializes the screen capture, loads the card deck, and enters a loop to continuously detect and deploy cards.

**Key Functions:**

-   `extract_hand_roi(frame)`: Extracts the region of interest (ROI) containing the player's hand from a game frame.
-   `__main__()`: The main function that runs the application. It handles the initialization of all modules, the main processing loop, and graceful shutdown.

### `capture.py`

This module is responsible for capturing the screen of the "BlueStacks App Player 1" window, where the game is running. It uses the `windows-capture` library for efficient screen capture and a double buffer to prevent race conditions.

**Key Components:**

-   `DoubleBuffer`: A class that implements a double buffer for thread-safe frame handling between the capture and processing threads.
-   `capture_thread(buffer, stop_event)`: The main function for the capture thread. It initializes the screen capture, handles frame arrival, and manages the capture lifecycle.
-   `optimize_frame_for_processing(frame)`: A utility function to optimize captured frames for better performance in computer vision tasks.

### `template_matcher.py`

This module detects which cards are currently available in the player's hand. It uses template matching with OpenCV, optimized for speed with parallel processing.

**Key Components:**

-   `DeckMatcher`: A class that manages the card detection process. It loads card templates, performs template matching in parallel using a thread pool, and determines the card in each of the four hand slots.
-   `detect_slots(hand_roi)`: The main method that takes the hand ROI and returns a dictionary mapping slot numbers to the detected card names.
-   `_match_single_template(...)`: A helper function that performs template matching for a single card.

### `deploy_cards.py`

This module provides high-level functions for implementing different card deployment strategies. It uses the low-level functions from `random_deployment.py` to execute the deployments.

**Key Functions:**

-   `run_continuous_deployment(...)`: Deploys cards continuously at random intervals.
-   `deploy_specific_strategy(strategy)`: Deploys cards according to a predefined sequence of moves.
-   `deploy_multiple_cards(...)`: Deploys a specified number of cards to either specific or random positions.

### `random_deployment.py`

This module contains the low-level logic for deploying cards. It defines the screen coordinates for card slots and deployment positions and uses the Android Debug Bridge (ADB) to simulate taps on the screen.

**Key Components:**

-   `CARD_POSITIONS`, `DEPLOY_POSITIONS`, `DEPLOY_WEIGHTS`: Dictionaries that define the coordinates for card slots, deployment positions, and the weights for random deployment.
-   `adb_tap(x, y)`: A function that executes an ADB tap command at the given coordinates.
-   `deploy_card(...)`: The main function that deploys a card from a given slot to a given position, with options for randomness.
-   `add_randomness(coord, offset)`: A utility function to add a small amount of randomness to coordinates to simulate human-like input.
-   `update_weights(weights)`: A function to dynamically adjust the deployment weights to make the bot's behavior less predictable.
