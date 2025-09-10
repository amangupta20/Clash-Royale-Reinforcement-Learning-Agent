import random
import time
import subprocess

# Base card slot coordinates (center points)
CARD_POSITIONS = {
    1: (215, 1785),  # Card 1 (far left)
    2: (409, 1785),  # Card 2 (left-middle)
    3: (600, 1785),  # Card 3 (right-middle)
    4: (794, 1785),  # Card 4 (far right)
}

# Deployment areas
DEPLOY_POSITIONS = {
    1: (300, 1200),  # Left side
    2: (700, 1200),  # Right side
    3: (500, 1350),  # Just above middle of the card deck
    4: (500, 1100),  # Higher above the middle
    5: (200, 800),   # Left high top
    6: (800, 800),   # Right high top
}

# Starting weights for each deploy position
DEPLOY_WEIGHTS = {
    1: 3,  # Left side
    2: 3,  # Right side
    3: 3,  # Just above middle
    4: 2,  # Higher above middle
    5: 1,  # Left high top
    6: 1,  # Right high top
}

def adb_tap(x, y):
    """Execute an ADB tap at the specified coordinates using ADB SHELL.
    Works consistently across macOS, Windows, and Linux.
    """
    tap_cmd = f"adb shell input tap {x} {y}"
    subprocess.run(tap_cmd, shell=True)
    time.sleep(0.1)

def add_randomness(coord, offset=5):
    """Add minimal randomness to the coordinates.
    
    For card selection, use a smaller offset (1-2 pixels) to ensure accurate selection.
    For deployment, a slightly larger offset (5-10 pixels) is acceptable.
    """
    x, y = coord
    return (x + random.randint(-offset, offset), y + random.randint(-offset, offset))

def update_weights(weights):
    """Randomly adjust weights to shift bias dynamically."""
    new_weights = weights.copy()
    
    pos = random.randint(1, len(weights))
    new_weights[pos] = max(1, new_weights[pos] + random.choice([-1, 1]))

    if sum(new_weights.values()) > 20:
        new_weights = {k: max(1, v - 1) for k, v in new_weights.items()}
    
    return new_weights

def deploy_card(card_slot=None, deploy_position=None, weights=None):
    """
    Deploy a card from a specific slot to a specific position with optional randomness.
    Always ensures card is selected before deployment.
    
    Args:
        card_slot (int, optional): Card slot number (1-4). If None, randomly chosen.
        deploy_position (int, optional): Deploy position (1-6). If None, chosen based on weights.
        weights (dict, optional): Custom weights for deployment positions. If None, uses default weights.
    
    Returns:
        tuple: (used_card_slot, used_deploy_position, updated_weights)
    """
    if weights is None:
        weights = DEPLOY_WEIGHTS.copy()
    
    if card_slot is None:
        card_slot = random.randint(1, 4)
    
    if deploy_position is None:
        positions = list(DEPLOY_POSITIONS.keys())
        weight_values = [weights[p] for p in positions]
        deploy_position = random.choices(positions, weights=weight_values, k=1)[0]
    
    card_coords = CARD_POSITIONS[card_slot]
    deploy_coords = DEPLOY_POSITIONS[deploy_position]
    
    card_coords_exact = add_randomness(card_coords, offset=2)
    random_deploy_coords = add_randomness(deploy_coords, offset=10)
    
    adb_tap(*card_coords_exact)
    time.sleep(random.uniform(0.3, 0.5))
    adb_tap(*random_deploy_coords)
    
    updated_weights = update_weights(weights)
    return card_slot, deploy_position, updated_weights

def get_card_positions():
    """Return the card position dictionary."""
    return CARD_POSITIONS.copy()

def get_deploy_positions():
    """Return the deployment position dictionary."""
    return DEPLOY_POSITIONS.copy()

def get_deploy_weights():
    """Return the default deployment weights."""
    return DEPLOY_WEIGHTS.copy()