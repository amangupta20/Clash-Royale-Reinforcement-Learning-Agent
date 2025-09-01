import subprocess
import sys
import shlex

# The default address for BlueStacks ADB instance.
DEVICE_ADDRESS = "127.0.0.1:5555"


def connect_to_device():
    """
    Connects ADB to the specified device address.

    Returns:
        bool: True if connection is successful, False otherwise.
    """
    print(f"Attempting to connect to device at {DEVICE_ADDRESS}...")
    try:
        command = ["adb", "connect", DEVICE_ADDRESS]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)

        output = result.stdout.strip()
        if "connected to" in output or "already connected" in output:
            print(f"Successfully connected to {DEVICE_ADDRESS}\n")
            return True
        else:
            print("--- Connection Failed ---")
            print(f"ADB Output: {output}")
            print(f"ADB Error: {result.stderr.strip()}")
            print("-------------------------")
            print(
                "Please ensure BlueStacks is running and ADB is enabled in its settings."
            )
            return False

    except FileNotFoundError:
        print("Error: 'adb' command not found.")
        print(
            "Please make sure the Android SDK Platform-Tools are installed and 'adb' is in your system's PATH."
        )
        return False
    except subprocess.TimeoutExpired:
        print("Error: The connection attempt timed out.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during connection: {e}")
        return False


def command_loop():
    """
    Main loop to accept and execute user commands.
    """
    print("Ready to send commands. Type 'exit' or 'quit' to close the script.")

    while True:
        # Create a prompt to show which device is targeted
        try:
            user_input = input(f"adb@{DEVICE_ADDRESS} > ")
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nExiting.")
            break

        if user_input.lower() in ["exit", "quit"]:
            print("Disconnecting and exiting.")
            break

        if not user_input:
            continue

        # Use shlex to safely split the command string, handling quotes
        try:
            command_parts = shlex.split(user_input)
        except ValueError as e:
            print(f"Error parsing command: {e}")
            continue

        # Construct the full ADB command with the -s flag to target the specific device
        full_command = ["adb", "-s", DEVICE_ADDRESS] + command_parts

        try:
            # Execute the command and stream the output
            process = subprocess.Popen(
                full_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",  # Handles potential encoding errors in output
            )

            # Print stdout and stderr as they come in
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    print(output.strip())

            # Print any remaining error output
            error_output = process.stderr.read()
            if error_output:
                print(f"Error: {error_output.strip()}", file=sys.stderr)

        except FileNotFoundError:
            print(
                "Error: 'adb' command not found. It might have been removed after the script started."
            )
            break
        except Exception as e:
            print(f"An error occurred while executing the command: {e}")


if __name__ == "__main__":
    if connect_to_device():
        command_loop()

    # Attempt to disconnect when the script is finished
    try:
        subprocess.run(["adb", "disconnect", DEVICE_ADDRESS], capture_output=True)
    except Exception:
        # Ignore errors on final disconnect
        pass

    print("\nScript finished.")
