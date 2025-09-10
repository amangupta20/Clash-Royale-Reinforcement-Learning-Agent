import random
import time
from random_deployment import (
    deploy_card,
    get_card_positions,
    get_deploy_positions,
    get_deploy_weights,
    update_weights
)

def run_continuous_deployment(interval_range=(4, 6), num_deployments=None):
    """
    Run continuous deployment with random intervals.
    
    Args:
        interval_range (tuple): Range for random interval (min, max) in seconds.
        num_deployments (int, optional): Number of deployments to make. If None, runs indefinitely.
    """
    weights = get_deploy_weights()
    count = 0
    
    try:
        print("Starting continuous deployment. Each deployment will select a card first.")
        while num_deployments is None or count < num_deployments:
            # For each iteration, select a new card then deploy
            card_slot = random.randint(1, 4)
            used_card, used_position, weights = deploy_card(card_slot=card_slot, weights=weights)
            
            print(f"Selected card {used_card}, deployed at position {used_position} | Weights: {weights}")
            
            # Increment the counter
            count += 1
            
            # If we've reached the requested number of deployments, stop
            if num_deployments is not None and count >= num_deployments:
                print(f"Completed {num_deployments} deployments")
                break
                
            # Wait before next deployment
            wait_time = random.uniform(*interval_range)
            print(f"Waiting {wait_time:.2f} seconds before next deployment")
            time.sleep(wait_time)
            
    except KeyboardInterrupt:
        print("Deployment stopped by user.")
        

def deploy_specific_strategy(strategy):
    """
    Deploy cards according to a predefined strategy.
    
    Args:
        strategy (list): List of tuples (card_slot, deploy_position) to deploy in order.
    
    Returns:
        bool: True if all deployments were successful, False otherwise.
    """
    weights = get_deploy_weights()
    
    try:
        print(f"Deploying {len(strategy)} cards according to strategy")
        
        for i, (card, position) in enumerate(strategy):
            print(f"Strategy step {i+1}/{len(strategy)}: Deploying card {card} to position {position}")
            
            # Deploy with exact specifications
            used_card, used_position, weights = deploy_card(
                card_slot=card, 
                deploy_position=position, 
                weights=weights
            )
            
            # Wait a random amount before next deployment
            if i < len(strategy) - 1:
                wait_time = random.uniform(1, 2)
                print(f"Waiting {wait_time:.2f} seconds before next strategic deployment")
                time.sleep(wait_time)
                
        return True
        
    except Exception as e:
        print(f"Strategy deployment failed: {e}")
        return False


def deploy_multiple_cards(num_cards=1, positions=None):
    """
    Deploy multiple cards to specified or random positions.
    
    Args:
        num_cards (int): Number of cards to deploy.
        positions (list, optional): List of position numbers to deploy to.
                                   If None, positions are chosen randomly.
    
    Returns:
        list: List of (card, position) tuples that were deployed.
    """
    weights = get_deploy_weights()
    deployments = []
    
    for i in range(num_cards):
        # If positions are specified, use them in order (cycling if needed)
        deploy_position = None
        if positions and len(positions) > 0:
            deploy_position = positions[i % len(positions)]
        
        # Deploy a random card to the specified position
        card_slot, used_position, weights = deploy_card(
            deploy_position=deploy_position,
            weights=weights
        )
        
        deployments.append((card_slot, used_position))
        print(f"Deployed card {card_slot} to position {used_position}")
        
        # Small delay between deployments
        if i < num_cards - 1:
            time.sleep(random.uniform(0.5, 1.0))
    
    return deployments


if __name__ == "__main__":
    # Example usage
    print("Choose a deployment mode:")
    print("1. Continuous deployment")
    print("2. Strategic deployment")
    print("3. Deploy multiple cards")
    
    try:
        choice = int(input("Enter choice (1-3): "))
        
        if choice == 1:
            # Run continuous deployment
            num = input("Number of deployments (leave empty for infinite): ")
            num_deployments = int(num) if num.strip() else None
            interval_min = float(input("Minimum interval between deployments (seconds): ") or "4")
            interval_max = float(input("Maximum interval between deployments (seconds): ") or "6")
            
            run_continuous_deployment(
                interval_range=(interval_min, interval_max),
                num_deployments=num_deployments
            )
            
        elif choice == 2:
            # Strategic deployment example
            # Format: [(card_slot, deploy_position), ...]
            strategy = [
                (1, 1),  # Deploy card 1 to position 1 (Left side)
                (2, 6),  # Deploy card 2 to position 6 (Right high top)
                (3, 3),  # Deploy card 3 to position 3 (Just above middle)
                (4, 4),  # Deploy card 4 to position 4 (Higher above middle)
            ]
            deploy_specific_strategy(strategy)
            
        elif choice == 3:
            # Deploy multiple cards
            num = int(input("Number of cards to deploy: ") or "3")
            pos_input = input("Enter positions separated by commas (or leave empty for random): ")
            
            positions = None
            if pos_input.strip():
                try:
                    positions = [int(p) for p in pos_input.split(",")]
                except ValueError:
                    print("Invalid position input, using random positions")
            
            deploy_multiple_cards(num_cards=num, positions=positions)
            
        else:
            print("Invalid choice")
            
    except ValueError:
        print("Invalid input")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")