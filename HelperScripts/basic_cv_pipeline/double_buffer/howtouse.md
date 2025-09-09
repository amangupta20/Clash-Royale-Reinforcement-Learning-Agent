
# High-Performance Screen Capture and Processing Pipeline

## 1. Introduction

This application provides a robust, high-performance pipeline for capturing a specific window and processing its frames in real-time. It is designed with low-latency and thread-safety as primary goals, making it an ideal foundation for demanding tasks like reinforcement learning bots, real-time object detection, or any application that needs to "see" and react to screen content instantly.

It leverages Python's `threading` for concurrency, `windows-capture` for efficient, hardware-accelerated screen grabbing, and `OpenCV` for image processing.

## 2. Core Concept: The Double Buffer Pattern

### The Challenge of Sharing Data Between Threads

When multiple threads access the same piece of data, problems can arise. If one thread (a "writer") is updating the data while another thread (a "reader") is trying to read it, the reader might see a corrupted, inconsistent version of the data. This is known as a **race condition**, and a specific type of error, a **torn read**, is a major risk in our case. A torn read would mean our processing thread gets a frame that is a mix of an old and a new frame, leading to incorrect analysis.

### The Solution: Lock-Free Reading with a Double Buffer

To solve this, we use a classic high-performance computing pattern called a **double buffer**. Instead of having one shared frame that all threads fight over (requiring locks that can introduce delays), we use two:

-   A **`back_buffer`**: The capture thread (the writer) exclusively writes new frames here. It never touches the other buffer.
-   A **`front_buffer`**: All processing threads (the readers) exclusively read from here. They never touch the back buffer.

When the writer has a new frame ready in the `back_buffer`, it performs a single, instantaneous **swap**. The `back_buffer` becomes the new `front_buffer` (making the new frame available to readers), and the old `front_buffer` becomes the new `back_buffer` (ready to be overwritten with the next frame).

This has a huge advantage: **readers never have to wait for a lock**. They can read from the `front_buffer` at any time, guaranteed to get a complete, valid frame. This is the key to achieving minimal latency for your processing logic.

## 3. Code Architecture

The system is composed of three main files:

### `main.py`: The Application Orchestrator

This is the entry point. It sets up the shared objects and manages the lifecycle of all threads.

-   **Responsibilities:** Initializes the `DoubleBuffer` and a `threading.Event` for shutdown signals, launches the capture and processing threads, and handles a graceful exit on `Ctrl+C` or when a window is closed.

### `capture.py`: The Frame Producer

This module's sole purpose is to capture the screen as fast as possible and make the frames available.

-   **`DoubleBuffer` Class:** This class encapsulates the double-buffering logic. It holds the `front_buffer`, `back_buffer`, and a `lock` that is used *only* by the writer thread to perform the atomic buffer swap, ensuring the swap itself is thread-safe.
-   **`capture_thread`:** This is the main function running in the capture thread. It sets up the `windows-capture` callback (`on_frame_arrived`). This callback is the heart of the writer; it receives a new frame and immediately calls `buffer.write()` to place it in the `back_buffer` and then trigger the swap.
-   **`fps_counter`:** A utility thread that prints the capture FPS every 5 seconds, allowing you to monitor the capture performance independently of any processing lag.

### `template_matching.py`: An Example Frame Consumer

This file serves as a template and example for any kind of processing you want to do on the captured frames.

-   **`template_matching_thread`:** This is a "reader" or "consumer" thread. It runs in a loop, performing these steps:
    1.  Calls `buffer.read()` to get a reference to the current `front_buffer`.
    2.  **Crucially, it makes a local copy (`frame.copy()`) of the frame.**
    3.  Performs all its processing (in this case, template matching) on the **local copy**.

#### The Importance of `frame.copy()`

Even with a double buffer, you must copy the frame before processing. Why? The `buffer.read()` gives you a pointer to the `front_buffer`. If your processing takes a while, the `capture_thread` could perform another buffer swap in the background. The buffer your variable points to would suddenly become the `back_buffer` and could be overwritten by the capture thread *while you are still processing it*. Copying the frame gives your thread a private snapshot that will not be affected by any other thread.

## 4. How to Run the Application

1.  **Install Dependencies:**
    ```bash
    pip install windows-capture opencv-python numpy
    ```

2.  **Run the Main Script:**
    ```bash
    python main.py
    ```

3.  **Stopping the Application:**
    -   Press the 'q' key while the OpenCV window from the template matching script is active.
    -   Press `Ctrl+C` in the terminal where `main.py` is running.

## 5. Extending the Application: Adding Your Own Processor

This is how you can add your own logic (e.g., a YOLO model, another CV task) to the pipeline.

1.  **Create a New File:** Let's call it `my_yolo_processor.py`.

2.  **Write Your Processor Function:** Use the following template. It must accept the `DoubleBuffer` and `stop_event` objects as arguments.

    ```python
    # my_yolo_processor.py
    import time
    import threading
    from capture import DoubleBuffer # Import the shared buffer class

    def yolo_thread(buffer: DoubleBuffer, stop_event: threading.Event):
        # --- Your one-time setup code here --- #
        # model = YOLO("yolov8n.pt") 
        print("YOLO processor thread started.")

        while not stop_event.is_set():
            # 1. Read the latest frame from the buffer
            frame = buffer.read()

            if frame is not None:
                # 2. Make a local copy to work on
                frame_copy = frame.copy()

                # 3. Perform your processing on the copy
                # results = model(frame_copy)
                # annotated_frame = results[0].plot()
                # print("YOLO processing complete.")

                # You could display it in a new window if you want
                # cv2.imshow("YOLO Output", annotated_frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #    stop_event.set()

            # Control how often your model runs
            time.sleep(0.1) # e.g., run inference 10 times per second

        print("YOLO processor thread stopped.")

    ```

3.  **Integrate it into `main.py`:**

    Modify `main.py` to import and run your new thread.

    ```python
    # main.py (modified)
    import threading
    import time
    from capture import DoubleBuffer, capture_thread
    from template_matching import template_matching_thread
    from my_yolo_processor import yolo_thread # <-- 1. IMPORT your new function

    def main():
        stop_event = threading.Event()
        buffer = DoubleBuffer()

        # Start the capture thread
        capture = threading.Thread(target=capture_thread, args=(buffer, stop_event))
        capture.start()

        print("Waiting for the first frame...")
        while not stop_event.is_set():
            if buffer.read() is not None:
                print("First frame received!")
                break
            time.sleep(0.5)

        if not stop_event.is_set():
            # Start the template matching thread
            template_matcher = threading.Thread(target=template_matching_thread, args=(buffer, stop_event))
            template_matcher.start()

            # <-- 2. START your new thread -->
            yolo_processor = threading.Thread(target=yolo_thread, args=(buffer, stop_event))
            yolo_processor.start()
        else:
            template_matcher = None
            yolo_processor = None # <-- 3. Handle the variable

        try:
            while not stop_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping threads...")
            stop_event.set()
        
        capture.join()
        if template_matcher:
            template_matcher.join()
        if yolo_processor: # <-- 4. JOIN your new thread
            yolo_processor.join()
        print("Threads stopped.")

    if __name__ == "__main__":
        main()
    ```

## 6. Troubleshooting

-   **`Error: Could not load template image 'archers.png'`**: Make sure the `archers.png` file is in the same directory as the script that is trying to load it (`template_matching.py`).
-   **`Capturing window: BlueStacks App Player 1` followed by nothing**: Make sure the window with the exact name "BlueStacks App Player 1" is open and visible.
-   **Low FPS**: If the capture FPS is low, it might be due to high CPU/GPU load from other processes. The processing in your consumer threads (like template matching) will not affect the capture FPS, as they run in separate threads.
