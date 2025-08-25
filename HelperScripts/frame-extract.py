import cv2
import os
import math

def extract_frames(video_path, output_folder, interval_seconds=5):
    """
    Extracts frames from a video at a specified interval.

    Args:
        video_path (str): The full path to the input video file.
        output_folder (str): The folder where the extracted frames will be saved.
        interval_seconds (int): The interval in seconds at which to extract frames.
    """
    # --- 1. Setup and Validation ---
    
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        print(f"Creating output directory: '{output_folder}'")
        os.makedirs(output_folder)

    # --- 2. Video Processing ---

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return

    # Get the frames per second (fps) of the video
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Could not determine video FPS. Assuming 60.")
        fps = 60 # Fallback for videos with missing FPS metadata

    # Calculate the number of frames to skip between captures
    frames_to_skip = int(round(fps * interval_seconds))
    print(f"Video FPS: {fps:.2f}. Capturing one frame every {frames_to_skip} frames.")

    frame_count = 0
    saved_frame_count = 0

    # --- 3. Frame Extraction Loop ---
    
    while True:
        # Read the next frame from the video
        success, frame = video_capture.read()

        # If 'success' is False, we have reached the end of the video
        if not success:
            break

        # Check if the current frame is the one we want to save
        if frame_count % frames_to_skip == 0:
            # Construct the output filename
            # e.g., "frame_0.png", "frame_5.png", etc.
            seconds = math.floor(frame_count / fps)
            filename = f"frame_at_{seconds}s.png"
            output_path = os.path.join(output_folder, filename)

            # Save the current frame as a PNG image
            cv2.imwrite(output_path, frame)
            saved_frame_count += 1
            print(f"Saved {output_path}")

        frame_count += 1

    # --- 4. Cleanup ---

    # Release the video capture object to free up resources
    video_capture.release()
    print("\n-------------------------------------------------")
    print("Extraction complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Total frames saved: {saved_frame_count}")
    print(f"Images are saved in: '{output_folder}'")
    print("-------------------------------------------------")


# --- HOW TO USE ---
if __name__ == '__main__':
    # 1. Set the path to your Clash Royale video file
    #    Example for Windows: "C:\\Users\\YourUser\\Videos\\clash_royale.mp4"
    #    Example for Mac/Linux: "/home/user/videos/clash_royale.mp4"
    input_video = "path/to/your/video.mp4" 

    # 2. Set the folder where you want to save the images
    output_dir = "clash_royale_dataset"

    # 3. Set the interval in seconds (e.g., 5 for one frame every 5 seconds)
    capture_interval = 5

    # Run the function
    extract_frames(input_video, output_dir, capture_interval)
