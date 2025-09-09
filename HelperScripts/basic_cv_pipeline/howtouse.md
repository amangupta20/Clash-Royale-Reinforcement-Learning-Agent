
# How to Use This Application

This application is designed to capture a specific window, perform template matching on the captured frames, and display the results in real-time. It uses a multi-threaded architecture to ensure that the screen capture and frame processing do not block each other.

## File Structure

The application is divided into three main files:

- `main.py`: The entry point of the application. It is responsible for starting and managing the other threads.
- `capture.py`: This script handles the screen capture of the specified window.
- `template_matching.py`: This script performs template matching on the captured frames.

## How it Works

The application uses a multi-threaded approach to separate the concerns of capturing the screen and processing the frames.

- **`main.py` - The Conductor:** This script acts as the conductor of the application. It creates a `FrameShare` object and a `threading.Event` to manage the other threads. The `FrameShare` object is used to share the captured frame between the `capture` and `template_matching` threads, while the `threading.Event` is used to signal all threads to stop gracefully.

- **`capture.py` - The Eye:** This script is responsible for capturing the specified window. It runs in a separate thread and uses the `windows-capture` library to capture the screen. The captured frames are stored in the `FrameShare` object.

- **`template_matching.py` - The Brain:** This script runs in its own thread and is responsible for processing the frames. It retrieves the latest frame from the `FrameShare` object, performs template matching, and displays the results.

### The `FrameShare` Class

The `FrameShare` class is the key to sharing data between the threads. It contains the following attributes:

- `latest_frame`: This variable holds the most recent frame captured from the window.
- `lock`: This is a `threading.Lock` that is used to ensure that only one thread can access the `latest_frame` at a time. This is crucial to prevent race conditions.
- `frame_count`: This variable is used to keep track of the number of frames captured for FPS calculation.

## How to Run the Application

To run the application, simply execute the `main.py` script:

```bash
python main.py
```

## Accessing the Frame from Other Functions or Files

To access the captured frame from other functions or files (e.g., for a different template matching algorithm or a YOLO object detection model), you need to import the `frame_share` object from the `main` module. However, since `main.py` is the entry point, you can't directly import from it. 

A better approach is to modify `main.py` to pass the `frame_share` object to your new function or thread.

Here's an example of how you can create a new file called `yolo_processing.py` and access the frame:

**`yolo_processing.py`**

```python
import time
import threading
from capture import FrameShare

def yolo_processing_thread(frame_share: FrameShare, stop_event: threading.Event):
    while not stop_event.is_set():
        frame_copy = None
        with frame_share.lock:
            if frame_share.latest_frame is not None:
                frame_copy = frame_share.latest_frame.copy()
        
        if frame_copy is not None:
            # Perform YOLO object detection on frame_copy
            print("Performing YOLO object detection...")

        time.sleep(1) # Adjust the sleep time as needed
```

**`main.py` (modified)**

```python
import threading
import time
from capture import FrameShare, capture_thread
from template_matching import template_matching_thread
from yolo_processing import yolo_processing_thread # Import the new function

def main():
    stop_event = threading.Event()
    frame_share = FrameShare()

    # Start the capture thread
    capture = threading.Thread(target=capture_thread, args=(frame_share, stop_event))
    capture.start()

    # Wait for the first frame to be captured
    print("Waiting for the first frame...")
    while not stop_event.is_set():
        with frame_share.lock:
            if frame_share.latest_frame is not None:
                break
        time.sleep(0.5)

    if not stop_event.is_set():
        # Start the template matching thread
        template_matcher = threading.Thread(target=template_matching_thread, args=(frame_share, stop_event))
        template_matcher.start()

        # Start the YOLO processing thread
        yolo_processor = threading.Thread(target=yolo_processing_thread, args=(frame_share, stop_event))
        yolo_processor.start()
    else:
        template_matcher = None
        yolo_processor = None

    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping threads...")
        stop_event.set()
    
    capture.join()
    if template_matcher:
        template_matcher.join()
    if yolo_processor:
        yolo_processor.join()
    print("Threads stopped.")

if __name__ == "__main__":
    main()

```

**Key Points to Remember:**

- **Always use the lock:** When accessing the `latest_frame` from any thread, always acquire the `lock` first to prevent data corruption.
- **Pass the `frame_share` object:** To share the frame with a new function or thread, pass the `frame_share` object as an argument.
- **Signal for shutdown:** Use the `stop_event` to gracefully stop all threads.
