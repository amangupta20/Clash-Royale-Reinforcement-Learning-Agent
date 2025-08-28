from windows_capture import WindowsCapture, Frame, InternalCaptureControl
import cv2 as cv
import numpy as np

# Crop settings - adjust these values manually
CROP_LEFT = 657  # pixels to crop from left
CROP_RIGHT = 657  # pixels to crop from right
TARGET_WIDTH = 480
TARGET_HEIGHT = 854

# Every Error From on_closed and on_frame_arrived Will End Up Here

capture = WindowsCapture(
    cursor_capture=None,
    draw_border=None,
    monitor_index=None,
    window_name="BlueStacks App Player 1",
)


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