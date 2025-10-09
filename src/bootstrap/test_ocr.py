"""
OCR Test Script

This script tests the OCR functionality for elixir and tower health detection.
It runs the OCR every 15 seconds and displays the results.
"""

import time
import threading
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor

from capture import DoubleBuffer
from capture import capture_thread
from minimal_perception import MinimalPerception
from template_matcher import TemplateCardMatcher


class OCRTester:
    """
    Test class for OCR functionality.
    """
    
    def __init__(self, save_debug_images: bool = True):
        """Initialize the OCR tester."""
        self.buffer = DoubleBuffer()
        self.stop_event = threading.Event()
        self.perception = MinimalPerception(save_debug_images=save_debug_images)
        self.running = False
        
    def start(self):
        """Start the OCR test."""
        # Start the screen capture in a separate thread
        capture = threading.Thread(target=capture_thread, args=(self.buffer, self.stop_event))
        capture.start()
        
        # Wait for the first frame to be captured
        print("Waiting for the first frame...")
        while not self.stop_event.is_set():
            if self.buffer.read() is not None:
                print("First frame received!")
                break
            time.sleep(0.5)
        
        self.running = True
        self.stop_event.clear()
        
        # Start the OCR thread
        ocr_thread = threading.Thread(target=self._ocr_loop)
        ocr_thread.daemon = True
        ocr_thread.start()
        
        print("OCR test started. Press Ctrl+C to stop.")
        return True
    
    def stop(self):
        """Stop the OCR test."""
        self.running = False
        self.stop_event.set()
        print("OCR test stopped.")
    
    def _ocr_loop(self):
        """Main OCR loop that runs every 15 seconds."""
        while self.running and not self.stop_event.is_set():
            try:
                # Get a frame directly from buffer
                frame = self.buffer.read_copy()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Run OCR on the frame
                self._run_ocr(frame)
                
                # Wait for 15 seconds
                self.stop_event.wait(15)
                
            except Exception as e:
                print(f"Error in OCR loop: {e}")
                time.sleep(1)
    
    def _run_ocr(self, frame):
        """
        Run OCR on a frame and display results.
        
        Args:
            frame: Frame to process
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'='*50}")
        print(f"OCR Results at {timestamp}")
        print(f"{'='*50}")
        
        # Run elixir and tower health detections in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit elixir detection task (now always uses OCR)
            elixir_future = executor.submit(self.perception.detect_elixir, frame)
            
            # Submit tower health detection task
            tower_health_future = executor.submit(self.perception.detect_tower_health, frame)
            
            # Collect results
            elixir = elixir_future.result()
            tower_health = tower_health_future.result()
        
        print(f"Elixir: {elixir}/10")
        
        # Display tower health as raw values (4-digit numbers)
        print("\nFriendly Tower Health:")
        for i, health in enumerate(tower_health['friendly']):
            tower_type = "Princess" if i < 2 else "King"
            print(f"  {tower_type} Tower: {health}")
        
        print("\nEnemy Tower Health:")
        for i, health in enumerate(tower_health['enemy']):
            if i < 2:
                tower_type = "Princess"
            else:
                tower_type = "King"
            print(f"  {tower_type} Tower: {health}")
        
        print(f"{'='*50}")
    
    def manual_test(self):
        """Run a manual test on demand."""
        frame = self.buffer.read_copy()
        if frame is None:
            print("No frame available")
            return
        
        self._run_ocr(frame)


def main():
    """Main function."""
    tester = OCRTester()
    
    if not tester.start():
        print("Failed to start OCR tester")
        return
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(15)
    except KeyboardInterrupt:
        print("\nStopping OCR tester...")
        tester.stop()


if __name__ == "__main__":
    main()