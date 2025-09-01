import os
import sys
from dotenv import load_dotenv
from adb_shell.adb_device import AdbDeviceTcp

# Load environment variables from .env file
load_dotenv()

# Device IP and port from environment variables, with defaults
DEVICE_IP = os.environ.get("ADB_DEVICE_IP", "127.0.0.1")
DEVICE_PORT = os.environ.get("ADB_DEVICE_PORT", "5555")
DEVICE_ADDRESS = f"{DEVICE_IP}:{DEVICE_PORT}"

print(f"Debug: Using device address: {DEVICE_ADDRESS}")
print(f"Debug: DEVICE_IP={DEVICE_IP}, DEVICE_PORT={DEVICE_PORT}")

def connect_to_device():
    """
    Connects to the specified device address using adb_shell.

    Returns:
        AdbDeviceTcp object if successful, None otherwise.
    """
    try:
        print(f"Attempting to connect to device at {DEVICE_IP}:{int(DEVICE_PORT)} using adb_shell...")
        # Add a default transport timeout to the constructor, as per documentation
        device = AdbDeviceTcp(DEVICE_IP, int(DEVICE_PORT), default_transport_timeout_s=9.0)

        # Connect to the device, with an auth timeout
        device.connect(rsa_keys=None, auth_timeout_s=5)
        print(f"Successfully connected to device at {DEVICE_ADDRESS}")

        # Test the connection with a simple command
        print(f"Testing connection to {DEVICE_ADDRESS}...")
        try:
            result = device.shell("echo 'connection test'")
            print(f"Successfully connected and tested device at {DEVICE_ADDRESS}")
            print(f"Test result: {result.strip()}\n")
        except Exception as test_error:
            print(f"Device found but not responding: {test_error}")
            return None

        return device

    except Exception as e:
        print(f"Connection failed: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print("\nPlease ensure:")
        print("1. BlueStacks or emulator is running")
        print("2. ADB is enabled in BlueStacks settings")
        print("3. Device is reachable at specified IP and port")
        return None

def command_loop(device):
    """
    Main loop to accept and execute user commands using adb_shell.
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
        # Automatically launch Clash Royale
        launch_command = "am start -n com.supercell.clashroyale/com.supercell.titan.GameApp"
        print(f"Connection successful. Attempting to launch Clash Royale...")
        try:
            output = device.shell(launch_command)
            if "Error" in output:
                print(f"Could not launch Clash Royale. Error: {output}")
            else:
                print("Clash Royale launched successfully.")
        except Exception as e:
            print(f"An error occurred while trying to launch Clash Royale: {e}")

        # Proceed to the interactive command loop
        command_loop(device)

    print("\nScript finished.")