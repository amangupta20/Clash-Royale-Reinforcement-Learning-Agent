import threading
import time
from typing import Optional

import numpy as np

from windows_capture import Frame, InternalCaptureControl, WindowsCapture


WINDOW_NAME = "BlueStacks App Player 1"

class DoubleBuffer:
    def __init__(self):
        self.front_buffer: Optional[np.ndarray] = None
        self.back_buffer: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.frame_count = 0

    def write(self, frame: np.ndarray):
        self.back_buffer = frame
        with self.lock:
            self.front_buffer, self.back_buffer = self.back_buffer, self.front_buffer
            self.frame_count += 1

    def read(self) -> Optional[np.ndarray]:
        return self.front_buffer

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
        buffer.write(frame.frame_buffer)
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