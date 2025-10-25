import threading
import time
from typing import Optional

import numpy as np
import cv2 as cv

from windows_capture import Frame, InternalCaptureControl, WindowsCapture


# The name of the window to capture
WINDOW_NAME = "BlueStacks App Player 1"

def optimize_frame_for_processing(frame: np.ndarray) -> np.ndarray:
    """Optimizes a frame for faster processing.

    This function ensures that the frame is in a C-contiguous memory layout
    and has a data type of uint8, which is optimal for OpenCV operations.

    Args:
        frame: The input frame as a NumPy array.

    Returns:
        The optimized frame, ready for computer vision processing.
    """
    # Ensure the frame is in the correct format and contiguous
    if not frame.flags['C_CONTIGUOUS']:
        frame = np.ascontiguousarray(frame)
    
    # Ensure uint8 data type for OpenCV compatibility
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    
    return frame

def create_frame_roi(frame: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """Creates a Region of Interest (ROI) using NumPy slicing.

    This function extracts a rectangular region from the input frame without
    copying the underlying data, making it a memory-efficient operation.

    Args:
        frame: The input frame as a NumPy array.
        x: The x-coordinate of the top-left corner of the ROI.
        y: The y-coordinate of the top-left corner of the ROI.
        width: The width of the ROI.
        height: The height of the ROI.

    Returns:
        A view of the original frame representing the ROI.
    """
    return frame[y:y+height, x:x+width]


class DoubleBuffer:
    """A thread-safe double buffer for handling frames.

    This class prevents race conditions between the frame capture thread and the
    main processing thread. The capture thread writes to a back buffer, while
    the main thread reads from a front buffer. Buffers are swapped atomically
    to ensure the main thread always gets a complete frame.

    Attributes:
        front_buffer: The buffer read by the consumer thread.
        back_buffer: The buffer written to by the producer thread.
        lock: A threading.Lock to ensure atomic buffer swaps.
        frame_count: A counter for the number of frames written.
    """
    def __init__(self):
        self.front_buffer: Optional[np.ndarray] = None
        self.back_buffer: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.frame_count = 0

    def write(self, frame: np.ndarray):
        """Writes a new frame to the back buffer and swaps it with the front buffer.

        This method is designed to be called from the capture thread.

        Args:
            frame: The new frame to write to the buffer.
        """
        # Optimize frame for better performance before writing
        self.back_buffer = optimize_frame_for_processing(frame)
        with self.lock:
            # Atomic swap of front and back buffers
            self.front_buffer, self.back_buffer = self.back_buffer, self.front_buffer
            self.frame_count += 1

    def read(self) -> Optional[np.ndarray]:
        """Reads the front buffer.

        This method returns a direct reference to the frame. The caller should create
        a copy if modifications are needed to avoid race conditions.

        Returns:
            The frame as a NumPy array, or None if the buffer is empty.
        """
        return self.front_buffer
    
    def read_copy(self) -> Optional[np.ndarray]:
        """Reads the front buffer and returns a copy of the frame.

        This is the recommended method for reading frames to ensure thread safety.

        Returns:
            A copy of the frame as a NumPy array, or None if the buffer is empty.
        """
        if self.front_buffer is not None:
            return self.front_buffer.copy()
        return None
    
    def get_frame_info(self) -> dict:
        """Gets information about the current frame.

        Returns:
            A dictionary containing frame metadata, or a status message if the
            buffer is empty.
        """
        if self.front_buffer is not None:
            return {
                'shape': self.front_buffer.shape,
                'dtype': self.front_buffer.dtype,
                'memory_usage_mb': self.front_buffer.nbytes / (1024 * 1024),
                'is_contiguous': self.front_buffer.flags['C_CONTIGUOUS']
            }
        return {'status': 'no_frame'}


def fps_counter(stop_event: threading.Event, buffer: DoubleBuffer):
    """Calculates and prints the frames per second (FPS) of the capture.

    This function runs in a separate thread and periodically calculates the FPS
    based on the number of frames captured over a time interval.

    Args:
        stop_event: A threading.Event to signal when the thread should stop.
        buffer: The DoubleBuffer instance used for frame capture.
    """
    start_time = time.time()
    while not stop_event.is_set():
        time.sleep(5)
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time > 0:
            with buffer.lock:
                fps = buffer.frame_count / elapsed_time
                print(f"Capture FPS: {fps:.2f}")
                buffer.frame_count = 0
        start_time = current_time

def capture_thread(buffer: DoubleBuffer, stop_event: threading.Event):
    """The main thread for capturing frames from the specified window.

    This function initializes the window capture, sets up event handlers,
    and starts the capture process.

    Args:
        buffer: The DoubleBuffer to write captured frames to.
        stop_event: A threading.Event to signal when the capture should stop.
    """
    # Initialize the window capture
    capture = WindowsCapture(window_name=WINDOW_NAME)

    @capture.event
    def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
        """
        This function is called whenever a new frame is available.
        It converts the frame to a BGR format and writes it to the double buffer.
        """
        # Get frame as NumPy array (already optimized by windows-capture)
        frame_array = frame.frame_buffer
        
        # The windows-capture library returns frames in BGRA format.
        # We convert it to BGR for compatibility with OpenCV.
        # Efficient NumPy slicing for BGRA to BGR conversion (zero-copy view)
        if frame_array.ndim >= 3 and frame_array.shape[2] == 4:  # BGRA format
            # Use numpy view instead of copy for better performance
            frame_bgr = frame_array[:, :, :3]  # Remove alpha channel (creates a view, not copy)
            # Ensure contiguous memory layout for OpenCV (only copies if needed)
            frame_bgr = np.ascontiguousarray(frame_bgr, dtype=np.uint8)
        else:
            # If frame_array is not 3D or does not have 4 channels, use as-is
            frame_bgr = frame_array

        # Write the processed frame to the buffer
        buffer.write(frame_bgr)

        # Stop the capture if the stop event is set
        if stop_event.is_set():
            capture_control.stop()

    @capture.event
    def on_closed():
        """
        This function is called when the capture is closed.
        It sets the stop event to signal other threads to terminate.
        """
        print("Capture session closed")
        stop_event.set()

    print(f"Capturing window: {WINDOW_NAME}")

    # Start the FPS counter thread
    fps_thread = threading.Thread(target=fps_counter, args=(stop_event, buffer))
    fps_thread.start()

    # Start the capture and wait for it to finish
    capture.start()
    fps_thread.join()