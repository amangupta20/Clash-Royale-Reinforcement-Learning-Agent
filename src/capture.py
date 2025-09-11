import threading
import time
from typing import Optional

import numpy as np
import cv2 as cv

from windows_capture import Frame, InternalCaptureControl, WindowsCapture


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
    def __init__(self):
        self.front_buffer: Optional[np.ndarray] = None
        self.back_buffer: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.frame_count = 0

    def write(self, frame: np.ndarray):
        """Write frame to back buffer with NumPy optimization."""
        # Optimize frame for better performance
        self.back_buffer = optimize_frame_for_processing(frame)
        with self.lock:
            self.front_buffer, self.back_buffer = self.back_buffer, self.front_buffer
            self.frame_count += 1

    def read(self) -> Optional[np.ndarray]:
        """Read from front buffer - returns a reference, caller should copy if needed."""
        return self.front_buffer
    
    def read_copy(self) -> Optional[np.ndarray]:
        """Read from front buffer and return a copy for safe processing."""
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
    capture = WindowsCapture(window_name=WINDOW_NAME)

    @capture.event
    def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
        # Get frame as NumPy array (already optimized by windows-capture)
        frame_array = frame.frame_buffer
        
        # Efficient NumPy slicing for BGRA to BGR conversion (zero-copy view)
        if frame_array.shape[2] == 4:  # BGRA format
            # Use numpy view instead of copy for better performance
            frame_bgr = frame_array[:, :, :3]  # Remove alpha channel (creates a view, not copy)
            # Ensure contiguous memory layout for OpenCV (only copies if needed)
            frame_bgr = np.ascontiguousarray(frame_bgr, dtype=np.uint8)
        else:
            frame_bgr = frame_array
            
        buffer.write(frame_bgr)
        if stop_event.is_set():
            capture_control.stop()

    @capture.event
    def on_closed():
        print("Capture session closed")
        stop_event.set()

    print(f"Capturing window: {WINDOW_NAME}")
    fps_thread = threading.Thread(target=fps_counter, args=(stop_event, buffer))
    fps_thread.start()

    capture.start()
    fps_thread.join()