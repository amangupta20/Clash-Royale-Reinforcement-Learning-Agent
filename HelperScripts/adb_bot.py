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


def add_randomness(coord, offset=40):
    """Add slight +/- randomness to coordinates"""
    x, y = coord
    return (x + random.randint(-offset, offset), y + random.randint(-offset, offset))


while True:
    # Pick a random card
    card_x, card_y = random.choice(cards)
    adb_tap(card_x, card_y)
    time.sleep(0.3)  # small delay between selecting and deploying

    # Weighted deployment choice (more bias to sides & middle-above)
    deploy_choice = random.choices(
        population=deploy_areas, weights=[3, 3, 3, 2, 1, 1], k=1
    )[0]

    # Add randomness to deployment coordinates
    deploy_x, deploy_y = add_randomness(deploy_choice, offset=40)
    adb_tap(deploy_x, deploy_y)

    # Wait 4–6 seconds before next deployment
    time.sleep(random.uniform(3, 6))
