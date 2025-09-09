
import threading
import time
from typing import Optional

import numpy as np

from windows_capture import Frame, InternalCaptureControl, WindowsCapture


WINDOW_NAME = "BlueStacks App Player 1"

class FrameShare:
    def __init__(self):
        self.latest_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.frame_count = 0

def fps_counter(stop_event: threading.Event, frame_share: FrameShare):
    start_time = time.time()
    while not stop_event.is_set():
        time.sleep(5)
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time > 0:
            with frame_share.lock:
                fps = frame_share.frame_count / elapsed_time
                print(f"Capture FPS: {fps:.2f}")
                frame_share.frame_count = 0
        start_time = current_time

def capture_thread(frame_share: FrameShare, stop_event: threading.Event):
    capture = WindowsCapture(window_name=WINDOW_NAME)

    @capture.event
    def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
        with frame_share.lock:
            frame_share.latest_frame = frame.frame_buffer
            frame_share.frame_count += 1
        if stop_event.is_set():
            capture_control.stop()

    @capture.event
    def on_closed():
        print("Capture session closed")
        stop_event.set()

    print(f"Capturing window: {WINDOW_NAME}")
    fps_thread = threading.Thread(target=fps_counter, args=(stop_event, frame_share))
    fps_thread.start()

    capture.start()
    fps_thread.join()
