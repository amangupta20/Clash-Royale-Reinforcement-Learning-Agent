import sys
from ppadb.client import Client

# The default address for BlueStacks ADB instance.
DEVICE_ADDRESS = "127.0.0.1:5555"

def connect_to_device():
    """
    Connects to the specified device address using ppadb.

    Returns:
        Device object if successful, None otherwise.
    """
    try:
        host, port = DEVICE_ADDRESS.rsplit(":", 1)
        port = int(port)
        client = Client(host="127.0.0.1", port=5037)  # ADB server host, usually default
        client.remote_connect(host, port)  # Connect to device over network
        device = client.device(DEVICE_ADDRESS)
        if device:
            print(f"Successfully connected to {DEVICE_ADDRESS}\n")
            return device
        else:
            print(f"Device {DEVICE_ADDRESS} not found after connection attempt.")
            return None
    except Exception as e:
        print(f"Connection failed: {e}")
        print(
            "Please ensure BlueStacks is running and ADB is enabled in its settings."
        )
        return None

def command_loop(device):
    """
    Main loop to accept and execute user commands using ppadb.
    """
    print("Ready to send commands. Type 'exit' or 'quit' to close the script.")

    while True:
        try:
            user_input = input(f"adb@{DEVICE_ADDRESS} > ")
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting.")
            break

        if not user_input:
            continue

        try:
            # Execute shell command on device
            output = device.shell(user_input)
            if output:
                print(output)

            # If command has no output (e.g., some commands), just print prompt again
        except Exception as e:
            print(f"An error occurred while executing the command: {e}")

if __name__ == "__main__":
    device = connect_to_device()
    if device:
        command_loop(device)

    print("\nScript finished.")