from windows_capture import WindowsCapture, Frame, InternalCaptureControl
window_names_to_try = [
    "BlueStacks App Player 1",
    "BlueStacks",
    "HD-Player", 
    "BlueStacks App Player",
    None  # This will capture the entire screen as fallback
]
# Every Error From on_closed and on_frame_arrived Will End Up Here
for name in window_names_to_try:
    capture = WindowsCapture(
        cursor_capture=None,
        draw_border=None,
        monitor_index=None,
        window_name=name,
    )


    @capture.event
    def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
        print("New Frame Arrived")

        # Save The Frame As An Image To The Specified Path
        frame.save_as_image("image.png")

        # Gracefully Stop The Capture Thread
        capture_control.stop()


    # Called When The Capture Item Closes Usually When The Window Closes, Capture
    # Session Will End After This Function Ends
    @capture.event
    def on_closed():
        print("Capture Session Closed")


    capture.start()