import cv2
import os
import math
from concurrent.futures import ThreadPoolExecutor
import threading
from dotenv import load_dotenv

def save_frame_worker(frame_data):
    """Worker function to save a frame with resizing"""
    frame, output_path, target_width, target_height = frame_data
    # Resize frame to target resolution
    resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    # Save as JPEG with optimized settings for maximum speed
    cv2.imwrite(output_path, resized_frame, [
        cv2.IMWRITE_JPEG_QUALITY, 85,
        cv2.IMWRITE_JPEG_OPTIMIZE, 1
    ])
    return output_path

def extract_frames(video_path, output_folder, interval_seconds=5, target_resolution=(720,1280), max_workers=10):
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
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Start from beginning
    
    # Skip frames in video reading instead of reading every frame
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

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

    # Calculate target frame numbers for seeking
    target_frames = list(range(0, total_frames, frames_to_skip))
    print(f"Will extract {len(target_frames)} frames from {total_frames} total frames")

    # Thread pool for parallel frame saving
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        counter=0

        # --- 3. Frame Extraction Loop with Seeking ---
        
        for target_frame in target_frames:
            # Seek directly to the target frame
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            
            # Read the frame at this position
            success, frame = video_capture.read()

            # If 'success' is False, we've reached the end or there's an error
            if not success:
                print(f"Could not read frame {target_frame}, stopping extraction")
                break

            # Construct the output filename
            seconds = math.floor(target_frame / fps)
            filename = f"{os.path.basename(video_path).split('.')[0]}_{counter}.jpg"
            counter+=1
            output_path = os.path.join(output_folder, filename)

            # Submit frame for parallel processing
            frame_data = (frame.copy(), output_path, target_width, target_height)
            future = executor.submit(save_frame_worker, frame_data)
            futures.append(future)
            
            saved_frame_count += 1
            if saved_frame_count % 10 == 0:  # Print progress every 10 frames
                print(f"Queued {saved_frame_count} frames for processing...")

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
    load_dotenv()

    # 1. Get configuration from environment variables
    input_video = os.getenv("VIDEO_PATH")
    output_dir = os.getenv("OUTPUT_DIR", "dataset")  # Default to "dataset"
    capture_interval = int(os.getenv("CAPTURE_INTERVAL", 5))
    target_width = int(os.getenv("TARGET_RESOLUTION_WIDTH", 480))
    target_height = int(os.getenv("TARGET_RESOLUTION_HEIGHT", 854))

    # 2. Validate required environment variables
    if not input_video:
        print("Error: VIDEO_PATH environment variable not set. Please create a .env file and set it.")
        exit()

    # 3. Validate capture interval
    if not 0 < capture_interval <= 3600:
        print(f"Invalid capture interval: {capture_interval}. Must be between 1 and 3600 seconds.")
        print("Using default of 5 seconds.")
        capture_interval = 5

    # 4. Run the extraction function
    target_resolution = (target_width, target_height)
    try:
        extract_frames(input_video, output_dir, capture_interval, target_resolution=target_resolution, max_workers=32)
    except KeyboardInterrupt:
        print("\nExtraction interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("Script finished.")
