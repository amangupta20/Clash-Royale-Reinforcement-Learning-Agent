"""
Bootstrap Screen Capture for Phase 0

This module implements the BootstrapCapture class for Phase 0 of the Clash Royale RL Agent.
It provides efficient screen capture from BlueStacks emulator with ROI support and
error handling.

Performance Target: <50ms average capture latency
"""

import threading
import time
from typing import Optional, Tuple

import numpy as np
import cv2 as cv

from windows_capture import Frame, InternalCaptureControl, WindowsCapture


class BootstrapCapture:
    """A screen capture wrapper for the BlueStacks emulator.

    This class provides an efficient screen capture mechanism with support for
    cropping to a region of interest (ROI). It uses the `windows-capture`
    library for optimized frame handling and aims for an average capture
    latency of less than 50ms.

    Attributes:
        window_name: The name of the window to capture.
        roi: The region of interest to capture, as a tuple of (x, y, width,
            height).
        show_fps: Whether to display FPS information.
        buffer: A DoubleBuffer for thread-safe frame handling.
        stop_event: A threading.Event to signal the capture thread to stop.
        capture_thread: The thread used for capturing frames.
        fps_thread: The thread used for calculating and displaying the FPS.
        is_capturing: A boolean indicating whether the capture is active.
    """

    def __init__(self, window_name: str = "BlueStacks App Player 1", roi: Optional[Tuple[int, int, int, int]] = None, show_fps: bool = True):
        """Initializes the BootstrapCapture.

        Args:
            window_name: The name of the window to capture.
            roi: The region of interest, as a tuple of (x, y, width, height).
                If None, the full window is captured.
            show_fps: Whether to display FPS information.
        """
        self.window_name = window_name
        self.roi = roi
        self.show_fps = show_fps
        self.buffer = DoubleBuffer()
        self.stop_event = threading.Event()
        self.capture_thread = None
        self.fps_thread = None
        self.is_capturing = False
        
    def grab(self) -> Optional[np.ndarray]:
        """Grabs a single frame from the buffer.

        Returns:
            An BGR frame as a NumPy array with shape (H, W, 3), or None if no
            frame is available.
        """
        frame = self.buffer.read_copy()
        if frame is not None and self.roi is not None:
            x, y, width, height = self.roi
            frame = frame[y:y+height, x:x+width]
        return frame
    
    def start_capture(self) -> bool:
        """Starts the capture process.

        Returns:
            True if the capture started successfully, False otherwise.
        """
        if self.is_capturing:
            return True
            
        self.stop_event.clear()
        self.capture_thread = threading.Thread(
            target=self._capture_thread_func,
            args=(self.buffer, self.stop_event)
        )
        self.fps_thread = threading.Thread(
            target=self._fps_counter,
            args=(self.buffer, self.stop_event)
        )
        
        self.is_capturing = True
        self.capture_thread.start()
        
        if self.show_fps:
            self.fps_thread.start()
        
        return True
    
    def stop_capture(self):
        """Stops the capture process."""
        if not self.is_capturing:
            return
            
        self.stop_event.set()
        if self.capture_thread:
            self.capture_thread.join()
        if self.show_fps and self.fps_thread:
            self.fps_thread.join()
        self.is_capturing = False
    
    def _capture_thread_func(self, buffer: 'DoubleBuffer', stop_event: threading.Event):
        """The internal capture thread function.

        Args:
            buffer: The DoubleBuffer to write frames to.
            stop_event: The threading.Event to signal when to stop.
        """
        try:
            capture = WindowsCapture(window_name=self.window_name)

            @capture.event
            def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
                """Handles new frame arrival.

                Args:
                    frame: The captured frame.
                    capture_control: The capture control object.
                """
                if stop_event.is_set():
                    capture_control.stop()
                    return
                    
                # Get frame as NumPy array
                frame_array = frame.frame_buffer
                
                # Convert BGRA to BGR for OpenCV compatibility
                if frame_array.ndim >= 3 and frame_array.shape[2] == 4:
                    frame_bgr = frame_array[:, :, :3]
                    frame_bgr = np.ascontiguousarray(frame_bgr, dtype=np.uint8)
                else:
                    frame_bgr = frame_array

                # Write to buffer
                buffer.write(frame_bgr)

            @capture.event
            def on_closed():
                """Handles the capture closed event."""
                print("Capture session closed")
                stop_event.set()

            print(f"Capturing window: {self.window_name}")
            capture.start()
            
        except Exception as e:
            print(f"Error in capture thread: {e}")
            stop_event.set()
    
    def _fps_counter(self, buffer: 'DoubleBuffer', stop_event: threading.Event):
        """Calculates and prints the FPS.

        Args:
            buffer: The DoubleBuffer to read the frame count from.
            stop_event: The threading.Event to signal when to stop.
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


class DoubleBuffer:
    """A simple double buffer for thread-safe frame handling.

    This class provides a way to share frames between threads without race
    conditions. One thread can write to the back buffer while another thread
    reads from the front buffer. The buffers are swapped atomically.

    Attributes:
        front_buffer: The buffer that is read from.
        back_buffer: The buffer that is written to.
        lock: A threading.Lock to ensure atomic operations.
        frame_count: The number of frames that have been written.
    """

    def __init__(self):
        self.front_buffer: Optional[np.ndarray] = None
        self.back_buffer: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.frame_count = 0

    def write(self, frame: np.ndarray):
        """Writes a new frame to the back buffer and swaps it with the front buffer.

        Args:
            frame: The new frame to write.
        """
        # Optimize frame for better performance
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
            
        self.back_buffer = frame
        with self.lock:
            # Atomic swap
            self.front_buffer, self.back_buffer = self.back_buffer, self.front_buffer
            self.frame_count += 1

    def read(self) -> Optional[np.ndarray]:
        """Reads the front buffer.

        Returns:
            The frame as a NumPy array, or None if the buffer is empty.
        """
        return self.front_buffer

    def read_copy(self) -> Optional[np.ndarray]:
        """Reads the front buffer and returns a copy.

        Returns:
            A copy of the frame as a NumPy array, or None if the buffer is empty.
        """
        if self.front_buffer is not None:
            return self.front_buffer.copy()
        return None


# Legacy functions for backward compatibility
def optimize_frame_for_processing(frame: np.ndarray) -> np.ndarray:
    """Optimizes a frame for faster processing using NumPy operations.

    Args:
        frame: The input frame as a NumPy array.

    Returns:
        The optimized frame, ready for computer vision processing.
    """
    if not frame.flags['C_CONTIGUOUS']:
        frame = np.ascontiguousarray(frame)
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    return frame


def create_frame_roi(frame: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """Creates a Region of Interest (ROI) using NumPy slicing.

    Args:
        frame: The input frame.
        x: The x-coordinate of the top-left corner.
        y: The y-coordinate of the top-left corner.
        width: The width of the ROI.
        height: The height of the ROI.

    Returns:
        A view of the original array representing the ROI.
    """
    return frame[y:y+height, x:x+width]


# Standalone functions for backward compatibility with original capture.py
def capture_thread(buffer: DoubleBuffer, stop_event: threading.Event):
    """The main thread for capturing frames from the specified window.

    This function initializes the window capture, sets up event handlers, and
    starts the capture process.

    Args:
        buffer: The DoubleBuffer to write frames to.
        stop_event: The threading.Event to signal when to stop.
    """
    # Initialize the window capture
    capture = WindowsCapture(window_name="BlueStacks App Player 1")

    @capture.event
    def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
        """This function is called whenever a new frame is available.

        It converts the frame to a BGR format and writes it to the double
        buffer.

        Args:
            frame: The captured frame.
            capture_control: The capture control object.
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
        """This function is called when the capture is closed.

        It sets the stop event to signal other threads to terminate.
        """
        print("Capture session closed")
        stop_event.set()

    print("Capturing window: BlueStacks App Player 1")

    # Start the capture and wait for it to finish
    capture.start()


def fps_counter(stop_event: threading.Event, buffer: DoubleBuffer):
    """A simple thread that calculates and prints the FPS of the capture.

    Args:
        stop_event: The threading.Event to signal when to stop.
        buffer: The DoubleBuffer to read the frame count from.
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