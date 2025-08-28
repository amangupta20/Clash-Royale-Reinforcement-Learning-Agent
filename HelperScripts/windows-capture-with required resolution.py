from windows_capture import WindowsCapture, Frame, InternalCaptureControl
import cv2 as cv
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
# Crop settings - adjust these values manually
CROP_LEFT = int(os.getenv("CROP_LEFT", 657))
CROP_RIGHT = int(os.getenv("CROP_RIGHT", 657))
TARGET_WIDTH = int(os.getenv("TARGET_WIDTH", 480))
TARGET_HEIGHT = int(os.getenv("TARGET_HEIGHT", 854))
WINDOW_NAME = os.getenv("WINDOW_NAME")

# --- Validation ---
if not WINDOW_NAME:
    print("Error: WINDOW_NAME is not set in the .env file.")
    exit()

# --- Capture Initialization ---
try:
    capture = WindowsCapture(
        cursor_capture=None,
        draw_border=None,
        monitor_index=None,
        window_name=WINDOW_NAME,
    )
except Exception as e:
    print(f"Error initializing window capture: {e}")
    print(f"Please ensure a window with the name '{WINDOW_NAME}' is open.")
    exit()


@capture.event
def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
    print("New Frame Arrived")

    # Save the original frame first for debugging
    frame.save_as_image("original_frame.png")
    
    # Load the saved image for processing
    image = cv.imread("original_frame.png", cv.IMREAD_COLOR)
    
    if image is not None:
        height, width = image.shape[:2]
        print(f"Original size: {width}x{height}")
        
        # Apply cropping
        if CROP_LEFT + CROP_RIGHT < width:
            cropped_image = image[:, CROP_LEFT:width-CROP_RIGHT]
            print(f"Cropped size: {cropped_image.shape[1]}x{cropped_image.shape[0]}")
            
            # Resize to target dimensions
            resized_image = cv.resize(cropped_image, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv.INTER_AREA)
            print(f"Final size: {TARGET_WIDTH}x{TARGET_HEIGHT}")
            
            # Save the processed image
            cv.imwrite("image.png", resized_image)
            print("Saved processed image as image.png")
        else:
            print("Error: Crop values are too large for the image width")
            frame.save_as_image("image.png")  # Save original if cropping fails
    else:
        print("Error: Could not load the saved frame")
        frame.save_as_image("image.png")  # Save original if loading fails

    # Gracefully Stop The Capture Thread
    capture_control.stop()


# Called When The Capture Item Closes Usually When The Window Closes, Capture
# Session Will End After This Function Ends
@capture.event
def on_closed():
    print("Capture Session Closed")


capture.start()