import threading
import time
from typing import Optional

import numpy as np
import cv2 as cv

from windows_capture import Frame, InternalCaptureControl, WindowsCapture


# The name of the window to capture
WINDOW_NAME = "BlueStacks App Player 1"

def optimize_frame_for_processing(frame: np.ndarray) -> np.ndarray:
    """
    Optimize frame for faster processing using NumPy operations.
    
    Args:
        frame: Input frame as NumPy array
        
    Returns:
        Optimized frame ready for computer vision processing
    """
    # Ensure the frame is in the correct format and contiguous
    if not frame.flags['C_CONTIGUOUS']:
        frame = np.ascontiguousarray(frame)
    
    # Ensure uint8 data type for OpenCV compatibility
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    
    return frame

def create_frame_roi(frame: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """
    Create Region of Interest (ROI) using NumPy slicing (zero-copy operation).
    
    Args:
        frame: Input frame
        x, y: Top-left coordinates
        width, height: ROI dimensions
        
    Returns:
        ROI as a view of the original array (no memory copy)
    """
    return frame[y:y+height, x:x+width]


class DoubleBuffer:
    """
    A simple double buffer implementation for thread-safe frame handling.
    This class is used to prevent race conditions between the frame capture thread
    and the main processing thread. The capture thread writes to a back buffer,
    while the main thread reads from a front buffer. The buffers are swapped
    atomically to ensure that the main thread always gets a complete frame.
    """
    def __init__(self):
        self.front_buffer: Optional[np.ndarray] = None
        self.back_buffer: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.frame_count = 0

    def write(self, frame: np.ndarray):
        """
        Write a new frame to the back buffer and then swaps it with the front buffer.
        This method is designed to be called from the capture thread.
        """
        # Optimize frame for better performance before writing
        self.back_buffer = optimize_frame_for_processing(frame)
        with self.lock:
            # Atomic swap of front and back buffers
            self.front_buffer, self.back_buffer = self.back_buffer, self.front_buffer
            self.frame_count += 1

    def read(self) -> Optional[np.ndarray]:
        """
        Read the front buffer. This method returns a reference to the frame.
        The caller should create a copy if it needs to modify the frame.
        """
        return self.front_buffer
    
    def read_copy(self) -> Optional[np.ndarray]:
        """
        Read the front buffer and return a copy of the frame.
        This is the recommended method for reading frames to ensure thread safety.
        """
        if self.front_buffer is not None:
            return self.front_buffer.copy()
        return None
    
    def get_frame_info(self) -> dict:
        """Get information about the current frame."""
        if self.front_buffer is not None:
            return {
                'shape': self.front_buffer.shape,
                'dtype': self.front_buffer.dtype,
                'memory_usage_mb': self.front_buffer.nbytes / (1024 * 1024),
                'is_contiguous': self.front_buffer.flags['C_CONTIGUOUS']
            }
        return {'status': 'no_frame'}


def fps_counter(stop_event: threading.Event, buffer: DoubleBuffer):
    """
    A simple thread that calculates and prints the frames per second (FPS) of the capture.
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
    """
    The main thread for capturing frames from the specified window.
    This function initializes the window capture, sets up event handlers,
    and starts the capture process.
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