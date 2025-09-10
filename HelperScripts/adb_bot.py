import random
import time
import subprocess

# Card slot coordinates
cards = [
    (333, 1715),  # Card 1 (far left)
    (540, 1715),  # Card 2 (left-middle)
    (740, 1715),  # Card 3 (right-middle)
    (945, 1715),  # Card 4 (far right)
]

# Base deployment areas
deploy_areas = [
    (300, 1200),  # Left side
    (700, 1200),  # Right side
    (500, 1350),  # Just above middle of the card deck
    (500, 1100),  # Higher above the middle
    (200, 800),  # Left high top
    (800, 800),  # Right high top
]


def adb_tap(x, y):
    subprocess.run(["adb", "shell", "input", "tap", str(x), str(y)])
    # Add a small delay after each tap to ensure command is processed
    time.sleep(0.1)


def add_randomness(coord, offset=40):
    """Add slight +/- randomness to coordinates"""
    x, y = coord
    return (x + random.randint(-offset, offset), y + random.randint(-offset, offset))


print("Starting deployment bot. Each deployment will select a card first.")

while True:
    # Always select a card first for each deployment
    card_index = random.randint(0, 3)  # Select card 0-3 (index for the cards list)
    card_x, card_y = cards[card_index]
    
    # Tap on the card to select it
    print(f"Selecting card at position {card_index+1}: ({card_x}, {card_y})")
    adb_tap(card_x, card_y)
    
    # Wait for selection animation to complete
    time.sleep(0.3)  # small delay between selecting and deploying

    # Weighted deployment choice (more bias to sides & middle-above)
    deploy_choice = random.choices(
        population=deploy_areas, weights=[3, 3, 3, 2, 1, 1], k=1
    )[0]

    # Add randomness to deployment coordinates
    deploy_x, deploy_y = add_randomness(deploy_choice, offset=40)
    print(f"Deploying to coordinates: ({deploy_x}, {deploy_y})")
    adb_tap(deploy_x, deploy_y)

    # Wait before next deployment
    wait_time = random.uniform(3, 6)
    print(f"Waiting {wait_time:.2f} seconds before next deployment")
    time.sleep(wait_time)
