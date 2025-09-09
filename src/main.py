import json

from template_matcher import DeckMatcher

import cv2 as cv



def __main__():

    # loading card deck from the json file
    deck_path = "deck.json"
    deck=[]
    try:
        with open(deck_path, "r") as f:
            deck = json.load(f)
            if not len(deck) == 8:
                raise ValueError("Deck must contain exactly 8 cards.")
                return 
            print(f"the cards in the current deck are {deck}")



    except FileNotFoundError:
        print(f"File not found: {deck_path}")
        print("Exiting ...")
        return
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {deck_path} ")
        print("Exiting ...")
        return
    except ValueError as e:
        print(f"Error: {e}")
        print("Exiting ...")
        return
    
    # Initialize the DeckMatcher with the loaded deck
    deck_matcher = DeckMatcher(deck)
    game_state_image_path="assets/deck/deck_area.png"
    game_image = cv.imread(game_state_image_path, cv.IMREAD_REDUCED_GRAYSCALE_2)
    detected_slots = deck_matcher.detect_slots(game_image)
    print(f"Detected slots: {detected_slots}")

__main__()
