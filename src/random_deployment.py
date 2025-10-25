import random
import time
import subprocess

# This file contains functions for automating card deployment in the game.
# It uses ADB (Android Debug Bridge) to simulate taps on the screen.
# The coordinates are hardcoded for a specific screen resolution and device.

# The (x, y) coordinates for the center of each of the four card slots in the hand.
CARD_POSITIONS = {
    1: (215, 1785),  # Card 1 (far left)
    2: (409, 1785),  # Card 2 (left-middle)
    3: (600, 1785),  # Card 3 (right-middle)
    4: (794, 1785),  # Card 4 (far right)
}

# The (x, y) coordinates for various deployment positions on the screen.
DEPLOY_POSITIONS = {
    1: (300, 1200),  # Left side
    2: (700, 1200),  # Right side
    3: (500, 1350),  # Just above middle of the card deck
    4: (500, 1100),  # Higher above the middle
    5: (200, 800),   # Left high top
    6: (800, 800),   # Right high top
}

# The initial weights for each deployment position.
# These weights are used to determine the probability of deploying a card to a specific position.
# A higher weight means a higher probability.
DEPLOY_WEIGHTS = {
    1: 3,  # Left side
    2: 3,  # Right side
    3: 3,  # Just above middle
    4: 2,  # Higher above middle
    5: 1,  # Left high top
    6: 1,  # Right high top
}

def adb_tap(x: int, y: int) -> None:
    """Executes an ADB tap command at the given (x, y) coordinates.

    This function requires ADB to be installed and configured on the system.

    Args:
        x: The x-coordinate of the tap location.
        y: The y-coordinate of the tap location.
    """
    subprocess.run(['adb', 'shell', 'input', 'tap', str(x), str(y)])
    time.sleep(0.1)

def add_randomness(coord: tuple[int, int], offset: int = 5) -> tuple[int, int]:
    """Adds a small amount of randomness to a coordinate.

    This function helps simulate human-like input by introducing slight
    variations to the tap coordinates.

    Args:
        coord: The (x, y) coordinate.
        offset: The maximum offset to add to the coordinate.

    Returns:
        The new coordinate with added randomness.
    """
    x, y = coord
    return (x + random.randint(-offset, offset), y + random.randint(-offset, offset))

def update_weights(weights: dict[int, int]) -> dict[int, int]:
    """Randomly adjusts the deployment weights.

    This function helps create a more dynamic deployment strategy by preventing
    the deployment from becoming too predictable.

    Args:
        weights: The current deployment weights.

    Returns:
        The updated deployment weights.
    """
    new_weights = weights.copy()
    
    # Randomly select a position and increase or decrease its weight
    pos = random.randint(1, len(weights))
    new_weights[pos] = max(1, new_weights[pos] + random.choice([-1, 1]))

    # If the total weight exceeds a certain threshold, reduce all weights by 1
    if sum(new_weights.values()) > 20:
        new_weights = {k: max(1, v - 1) for k, v in new_weights.items()}
    
    return new_weights

def deploy_card(
    card_slot: int | None = None,
    deploy_position: int | None = None,
    weights: dict[int, int] | None = None,
) -> tuple[int, int, dict[int, int]]:
    """Deploys a card from a given slot to a given position.

    If the card slot is not specified, a random slot is chosen. If the
    deployment position is not specified, a random position is chosen based on
    the weights.

    Args:
        card_slot: The card slot to deploy from (1-4).
        deploy_position: The position to deploy to (1-6).
        weights: The weights to use for random deployment.

    Returns:
        A tuple containing the card slot used, the deployment position used, and
        the updated weights.
    """
    if weights is None:
        weights = DEPLOY_WEIGHTS.copy()
    
    # If no card slot is specified, choose one at random
    if card_slot is None:
        card_slot = random.randint(1, 4)
    
    # If no deployment position is specified, choose one based on the weights
    if deploy_position is None:
        positions = list(DEPLOY_POSITIONS.keys())
        weight_values = [weights[p] for p in positions]
        deploy_position = random.choices(positions, weights=weight_values, k=1)[0]
    
    # Get the coordinates for the card and deployment positions
    card_coords = CARD_POSITIONS[card_slot]
    deploy_coords = DEPLOY_POSITIONS[deploy_position]
    
    # Add a small amount of randomness to the coordinates to simulate human-like input
    card_coords_exact = add_randomness(card_coords, offset=2)
    random_deploy_coords = add_randomness(deploy_coords, offset=10)
    
    # Tap the card to select it, then tap the deployment position to deploy it
    adb_tap(*card_coords_exact)
    time.sleep(random.uniform(0.3, 0.5))
    adb_tap(*random_deploy_coords)
    
    # Update the weights for the next deployment
    updated_weights = update_weights(weights)
    return card_slot, deploy_position, updated_weights

def get_card_positions() -> dict[int, tuple[int, int]]:
    """Returns a copy of the card position dictionary.

    Returns:
        A dictionary mapping card slots to their (x, y) coordinates.
    """
    return CARD_POSITIONS.copy()

def get_deploy_positions() -> dict[int, tuple[int, int]]:
    """Returns a copy of the deployment position dictionary.

    Returns:
        A dictionary mapping deployment positions to their (x, y) coordinates.
    """
    return DEPLOY_POSITIONS.copy()

def get_deploy_weights() -> dict[int, int]:
    """Returns a copy of the default deployment weights.

    Returns:
        A dictionary mapping deployment positions to their weights.
    """
    return DEPLOY_WEIGHTS.copy()