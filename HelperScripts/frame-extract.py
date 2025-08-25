import cv2
import os
import math
from concurrent.futures import ThreadPoolExecutor
import threading

def save_frame_worker(frame_data):
    """Worker function to save a frame with resizing"""
    frame, output_path, target_width, target_height = frame_data
    # Resize frame to target resolution
    resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    # Save as JPEG with high quality for faster saving
    cv2.imwrite(output_path, resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return output_path

def extract_frames(video_path, output_folder, interval_seconds=5, target_resolution=(640, 480), max_workers=10):
    """
    Extracts frames from a video at a specified interval with optimized performance.

    Args:
        video_path (str): The full path to the input video file.
        output_folder (str): The folder where the extracted frames will be saved.
        interval_seconds (int): The interval in seconds at which to extract frames.
        target_resolution (tuple): Target width and height for extracted frames (width, height).
        max_workers (int): Number of threads for parallel frame saving.
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

    # Optimize video capture settings for speed
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for faster reading

    # Get the frames per second (fps) of the video
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Could not determine video FPS. Assuming 60.")
        fps = 60 # Fallback for videos with missing FPS metadata

    # Calculate the number of frames to skip between captures
    frames_to_skip = int(round(fps * interval_seconds))
    print(f"Video FPS: {fps:.2f}. Capturing one frame every {frames_to_skip} frames.")
    print(f"Target resolution: {target_resolution[0]}x{target_resolution[1]}")

    frame_count = 0
    saved_frame_count = 0
    target_width, target_height = target_resolution

    # Thread pool for parallel frame saving
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

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
                # e.g., "frame_0.jpg", "frame_5.jpg", etc.
                seconds = math.floor(frame_count / fps)
                filename = f"{os.path.basename(video_path).split('.')[0]}_{frame_count}.jpg"
                output_path = os.path.join(output_folder, filename)

                # Submit frame for parallel processing
                frame_data = (frame.copy(), output_path, target_width, target_height)
                future = executor.submit(save_frame_worker, frame_data)
                futures.append(future)
                
                saved_frame_count += 1
                if saved_frame_count % 10 == 0:  # Print progress every 10 frames
                    print(f"Queued {saved_frame_count} frames for processing...")

            frame_count += 1

        # Wait for all frames to be saved
        print("Waiting for all frames to be saved...")
        for future in futures:
            try:
                output_path = future.result()
                # Uncomment the line below if you want to see each saved frame
                # print(f"Saved {output_path}")
            except Exception as e:
                print(f"Error saving frame: {e}")

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
    input_video = "vids/2025-08-25-pc.mkv" 

    # 2. Set the folder where you want to save the images
    output_dir = "dataset"

    # 3. Set the interval in seconds (e.g., 5 for one frame every 5 seconds)
    capture_interval = 5

    # Run the function
    extract_frames(input_video, output_dir, capture_interval)
