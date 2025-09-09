
import threading
import time
from capture import FrameShare, capture_thread
from template_matching import template_matching_thread

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
                print("First frame received!")
                break
        time.sleep(0.5)

    if not stop_event.is_set():
        # Start the template matching thread
        template_matcher = threading.Thread(target=template_matching_thread, args=(frame_share, stop_event))
        template_matcher.start()
    else:
        template_matcher = None

    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping threads...")
        stop_event.set()
    
    capture.join()
    if template_matcher:
        template_matcher.join()
    print("Threads stopped.")

if __name__ == "__main__":
    main()
